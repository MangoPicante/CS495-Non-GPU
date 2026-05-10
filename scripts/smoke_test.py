"""
Smoke test for both inference models and all Phase 4 scripts.

For each model the test:
  1. Loads the model ONCE (one Popen session)
  2. Pipes all prompts through stdin, reads responses from stdout
  3. Prints the prompt, response, PASS/FAIL, and cost estimate
  4. Exits the session (closes stdin), unloading the model

Then verifies compare_runs.py output files, and that metrics_tracker.py
and eval_accuracy.py parse arguments cleanly.

Exit 0 = all checks passed.  Non-zero = at least one failure.

Usage:
    python scripts/smoke_test.py
    make smoke-test
"""

import csv
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT      = Path(__file__).parent.parent
SCRIPTS   = ROOT / "scripts"
RESULTS   = ROOT / "results"

BITNET_DIR   = ROOT.parent / "Models" / "BitNet"
BITNET_MODEL = BITNET_DIR / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"
BITNET_CLI   = BITNET_DIR / "build" / "bin" / "Release" / "llama-cli.exe"

QWEN_DIR   = ROOT.parent / "Models" / "Qwen"
QWEN_MODEL = QWEN_DIR / "qwen2.5-1.5b-instruct-q8_0.gguf"
QWEN_CLI   = QWEN_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"

HARDWARE_RATE     = 0.170   # AWS c5.xlarge on-demand $/hr
N_TOKENS          = 16      # tokens to generate per prompt
THREADS           = 4
CTX               = 512
STARTUP_TIMEOUT   = 60      # seconds to wait for model to load and show first prompt
INFERENCE_TIMEOUT = 90      # seconds per prompt in the interactive session

# (prompt, [acceptable keywords], test label)
INFERENCE_CASES = [
    ("What is 2+2?",                   ["4", "four"],                                        "basic arithmetic"),
    ("What is the capital of France?", ["Paris"],                                             "factual recall"),
    ("The sky is",                     ["blue", "clear", "bright", "often", "vast", "usually"], "common sense"),
]

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail and not condition else ""
    print(f"  {mark}  {label}{suffix}")
    if not condition:
        _failures.append(label)


def cost_per_1k(tps: float) -> str:
    return f"${(1000.0 / tps / 3600.0) * HARDWARE_RATE:.5f}"


# ── Session helpers ───────────────────────────────────────────────────────────

# llama-cli prints this as its interactive input prompt (to stdout)
_PROMPT_SENTINEL = "\n> "


def _char_reader(pipe, q: queue.Queue) -> None:
    """Thread: read binary pipe one byte at a time, push decoded chars to queue."""
    try:
        while True:
            b = pipe.read(1)
            if not b:
                break
            q.put(b.decode("utf-8", errors="replace"))
    finally:
        q.put(None)  # EOF sentinel


def _collect_until_sentinel(q: queue.Queue, timeout: float) -> str:
    """
    Accumulate chars from q until _PROMPT_SENTINEL appears or timeout/EOF.
    Returns the full accumulated string (including the sentinel if found).
    """
    buf = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            ch = q.get(timeout=min(remaining, 0.1))
        except queue.Empty:
            continue
        if ch is None:
            break
        buf += ch
        if buf.endswith(_PROMPT_SENTINEL):
            return buf
    return buf


def _drain_queue(q: queue.Queue, timeout: float = 0.5) -> str:
    """Non-blocking drain: collect any chars already in q within timeout seconds."""
    buf = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            ch = q.get(timeout=min(remaining, 0.05))
        except queue.Empty:
            break
        if ch is None:
            break
        buf += ch
    return buf


def _parse_tps(stdout_chunk: str, stderr_chunk: str) -> float | None:
    """
    Parse tokens/sec from either:
      - New upstream llama.cpp: '[ Prompt: X t/s | Generation: X.X t/s ]' in stdout
      - Old llama.cpp fork:     'eval time = ... X ms per token' in stderr
    """
    for line in stdout_chunk.splitlines():
        m = re.search(r"Generation:\s*(\d+\.?\d*)\s*t/s", line)
        if m:
            return float(m.group(1))
    for line in stderr_chunk.splitlines():
        m = re.search(r"eval time\s*=.*?(\d+\.?\d*)\s*ms per token", line)
        if m:
            return 1000.0 / float(m.group(1))
    return None


# ── Session-based model runner ────────────────────────────────────────────────

def run_model_session(
    name: str,
    cli: Path,
    model: Path,
    extra_flags: list[str] | None = None,
) -> list[float]:
    """
    Load the model ONCE, pipe all INFERENCE_CASES through stdin, collect responses,
    then close stdin to unload.  Returns a list of measured tok/s values.

    extra_flags: additional llama-cli flags (e.g. ["-i"] for older interactive mode).
    """
    if extra_flags is None:
        extra_flags = []

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Model: {model.name}")
    print(f"  CLI:   {cli.name}  ({cli.parent.parent.parent.parent.name})")
    print(f"{'='*60}")

    if not model.exists():
        print(f"  SKIP  model file not found: {model}")
        _failures.append(f"{name}: model not found")
        return []
    if not cli.exists():
        print(f"  SKIP  {cli.name} not found: {cli}")
        _failures.append(f"{name}: llama-cli not found")
        return []

    cmd = [
        str(cli),
        "-m", str(model),
        "-n", str(N_TOKENS),
        "-t", str(THREADS),
        "-c", str(CTX),
        *extra_flags,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except Exception as e:
        print(f"  FAIL  Could not start {cli.name}: {e}")
        _failures.append(f"{name}: failed to start")
        return []

    out_q: queue.Queue = queue.Queue()
    err_q: queue.Queue = queue.Queue()
    threading.Thread(target=_char_reader, args=(proc.stdout, out_q), daemon=True).start()
    threading.Thread(target=_char_reader, args=(proc.stderr, err_q), daemon=True).start()

    tps_list = []

    try:
        print("  Loading model...", end="", flush=True)
        startup = _collect_until_sentinel(out_q, timeout=STARTUP_TIMEOUT)
        if not startup.rstrip().endswith(">"):
            rc = proc.poll()
            print(f"\n  FAIL  no interactive prompt within {STARTUP_TIMEOUT}s (rc={rc})")
            _failures.append(f"{name}: startup timeout")
            return []
        print(" ready.", flush=True)

        for prompt, keywords, label in INFERENCE_CASES:
            print(f"\n  [{label}]")
            print(f"  Prompt:    {prompt!r}")

            proc.stdin.write((prompt + "\n").encode("utf-8"))
            proc.stdin.flush()

            raw_out = _collect_until_sentinel(out_q, timeout=INFERENCE_TIMEOUT)
            raw_err = _drain_queue(err_q, timeout=0.5)

            if not raw_out:
                check(f"{label}: response received", False, "TIMEOUT or empty")
                continue

            # Strip the echoed prompt from the front
            response = raw_out
            if prompt in response:
                response = response[response.index(prompt) + len(prompt):]
            # Remove inline perf line and trailing sentinel
            response = re.sub(r"\[.*?Generation:.*?\]", "", response)
            response = response.rstrip().rstrip(">").strip()

            print(f"  Response:  {response!r}")

            hit = any(kw.lower() in response.lower() for kw in keywords)
            expected_str = " or ".join(repr(k) for k in keywords)
            check(f"{label}: response contains {expected_str}", hit)

            tps = _parse_tps(raw_out, raw_err)
            if tps is not None:
                tps_list.append(tps)
                print(f"  Speed:     {tps:.1f} tok/s   ->{cost_per_1k(tps)} / 1k tokens"
                      f"  (AWS c5.xlarge @ ${HARDWARE_RATE}/hr)")

    except Exception as e:
        print(f"  ERROR  {e}")
        _failures.append(f"{name}: exception during inference")

    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    if tps_list:
        avg = sum(tps_list) / len(tps_list)
        print(f"\n  Average:   {avg:.1f} tok/s   ->{cost_per_1k(avg)} / 1k tokens")

    return tps_list


# ── Inference smoke tests ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Model Inference Smoke Tests")
print("="*60)

# BitNet fork requires -i to enter interactive mode; Qwen upstream is interactive by default
bitnet_tps = run_model_session("BitNet b1.58 2B4T",           BITNET_CLI, BITNET_MODEL, ["-i"])
qwen_tps   = run_model_session("Qwen2.5-1.5B-Instruct Q8_0", QWEN_CLI,   QWEN_MODEL)

if bitnet_tps and qwen_tps:
    b_avg = sum(bitnet_tps) / len(bitnet_tps)
    q_avg = sum(qwen_tps)   / len(qwen_tps)
    print(f"\n{'='*60}")
    print(f"  Cost comparison  (AWS c5.xlarge @ ${HARDWARE_RATE}/hr)")
    print(f"  {'Model':<35} {'tok/s':>8}  {'$/1k tokens':>13}")
    print(f"  {'-'*58}")
    print(f"  {'BitNet b1.58 2B4T':<35} {b_avg:>8.1f}  {cost_per_1k(b_avg):>13}")
    print(f"  {'Qwen2.5-1.5B-Instruct Q8_0':<35} {q_avg:>8.1f}  {cost_per_1k(q_avg):>13}")
    print("="*60)

# ── compare_runs.py ───────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  compare_runs.py")
print(f"{'='*60}")


def run_script(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *cmd],
        capture_output=True, text=True, cwd=ROOT,
    )


result = run_script(str(SCRIPTS / "compare_runs.py"))
check("exits 0", result.returncode == 0, result.stderr.strip())

for plot_name in [
    "throughput_comparison.png",
    "memory_comparison.png",
    "bitnet_throughput_configs.png",
    "accuracy_comparison.png",
    "cost_accuracy.png",
]:
    p = RESULTS / "plots" / plot_name
    check(f"plots/{plot_name} non-empty", p.exists() and p.stat().st_size > 0)

csv_path = RESULTS / "comparison_table.csv"
check("comparison_table.csv exists", csv_path.exists())
if csv_path.exists():
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    check("7 data rows (5 FP16 + paper + ours)", len(rows) == 7, f"got {len(rows)}")
    required_cols = {
        "model", "source", "throughput_tokens_s", "peak_rss_mb",
        "cost_per_1k_tokens", "arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu",
    }
    missing = required_cols - set(rows[0].keys()) if rows else required_cols
    check("all expected columns present", not missing, f"missing {missing}")
    ours = next((r for r in rows if r.get("source") == "ours"), None)
    check("'ours' row present", ours is not None)
    if ours:
        check("'ours' row has throughput",         ours["throughput_tokens_s"] != "")
        check("'ours' row has cost_per_1k_tokens", ours["cost_per_1k_tokens"] != "")
        check("'ours' row has arc_easy",           ours["arc_easy"] != "")

# ── metrics_tracker.py ────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  metrics_tracker.py")
print(f"{'='*60}")

result = run_script(str(SCRIPTS / "metrics_tracker.py"), "--help")
check("--help exits 0", result.returncode == 0, result.stderr.strip())
check("--no-energy flag present", "--no-energy" in result.stdout)

# ── eval_accuracy.py ──────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  eval_accuracy.py")
print(f"{'='*60}")

result = run_script(str(SCRIPTS / "eval_accuracy.py"), "--help")
check("--help exits 0", result.returncode == 0, result.stderr.strip())
check("--task flag present", "--task" in result.stdout)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
if _failures:
    print(f"  FAILED  ({len(_failures)} check(s) failed):")
    for f in _failures:
        print(f"    - {f}")
    print("="*60)
    sys.exit(1)
else:
    print("  All checks passed.")
    print("="*60)
