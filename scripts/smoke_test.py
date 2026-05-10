"""
Smoke test for both inference models and all Phase 4 scripts.

For each model and test case, prints:
  - the prompt sent to the model
  - the response received
  - whether the expected keyword is present (PASS/FAIL)
  - measured throughput and cost estimate (AWS c5.xlarge @ $0.170/hr)

Then verifies compare_runs.py output files, and that metrics_tracker.py
and eval_accuracy.py parse arguments cleanly.

Exit 0 = all checks passed.  Non-zero = at least one failure.

Usage:
    python scripts/smoke_test.py              # test both models
    python scripts/smoke_test.py bitnet       # test BitNet only
    python scripts/smoke_test.py qwen         # test Qwen only
    make smoke-test
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

_parser = argparse.ArgumentParser(description="Smoke test for inference models and Phase 4 scripts.")
_parser.add_argument("model", nargs="?", choices=["bitnet", "qwen"], default=None,
                     help="Which model to test (default: both)")
_args = _parser.parse_args()
RUN_BITNET = _args.model in (None, "bitnet")
RUN_QWEN   = _args.model in (None, "qwen")

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
N_TOKENS          = 16      # tokens to generate per test prompt (keep smoke test fast)
THREADS           = 4
CTX               = 512     # small context keeps KV cache tiny during smoke test
INFERENCE_TIMEOUT = 120     # seconds per llama-cli invocation before reporting TIMEOUT

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


def parse_tps(stdout: str, stderr: str) -> float | None:
    """Parse tokens/sec from either output format.

    New upstream llama.cpp: '[ Prompt: X t/s | Generation: X.X t/s ]' in stdout.
    Old llama.cpp fork (BitNet): 'eval time = ... X ms per token' in stderr.
    """
    for line in stdout.splitlines():
        m = re.search(r"Generation:\s*(\d+\.?\d*)\s*t/s", line)
        if m:
            return float(m.group(1))
    for line in stderr.splitlines():
        m = re.search(r"eval time\s*=.*?(\d+\.?\d*)\s*ms per token", line)
        if m:
            return 1000.0 / float(m.group(1))
    return None


def run_model(cli: Path, model: Path, prompt: str,
              extra_flags: list[str] | None = None) -> tuple[str, float | None, int]:
    """Invoke llama-cli once; return (response_text, tokens_per_sec, exit_code).
    Returns ('TIMEOUT', None, -1) on timeout."""
    cmd = [
        str(cli),
        "-m", str(model),
        "-p", prompt,
        "-n", str(N_TOKENS),
        "-t", str(THREADS),
        "-c", str(CTX),
        *(extra_flags or []),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=INFERENCE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None, -1
    tps = parse_tps(proc.stdout, proc.stderr)
    out = proc.stdout
    response = out[out.index(prompt) + len(prompt):].strip() if prompt in out else out.strip()
    # Strip inline perf line that the new llama.cpp appends to stdout
    response = re.sub(r"\[.*?Generation:.*?\]", "", response).strip()
    return response, tps, proc.returncode


def smoke_model(name: str, cli: Path, model: Path,
                extra_flags: list[str] | None = None) -> list[float]:
    """Run all inference test cases for one model; return list of tps values."""
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

    tps_list = []
    for prompt, keywords, label in INFERENCE_CASES:
        print(f"\n  [{label}]")
        print(f"  Prompt:    {prompt!r}")
        response, tps, rc = run_model(cli, model, prompt, extra_flags)
        print(f"  Response:  {response!r}")

        if response == "TIMEOUT":
            check(f"{label}: completed within {INFERENCE_TIMEOUT}s", False, "TIMEOUT")
            continue

        hit = any(kw.lower() in response.lower() for kw in keywords)
        expected_str = " or ".join(repr(k) for k in keywords)
        check(f"{label}: response contains {expected_str}", hit)

        if tps is not None:
            tps_list.append(tps)
            print(f"  Speed:     {tps:.1f} tok/s   ->{cost_per_1k(tps)} / 1k tokens"
                  f"  (AWS c5.xlarge @ ${HARDWARE_RATE}/hr)")
        else:
            check(f"{label}: llama-cli exited cleanly", rc == 0, f"exit code {rc}")

    if tps_list:
        avg = sum(tps_list) / len(tps_list)
        print(f"\n  Average:   {avg:.1f} tok/s   ->{cost_per_1k(avg)} / 1k tokens")

    return tps_list


# ── Inference smoke tests ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Model Inference Smoke Tests")
print("="*60)

# BitNet's llama.cpp fork is non-interactive by default: exits after -n tokens.
# Qwen's upstream llama.cpp defaults to interactive conversation mode and ignores
# stdin EOF; --single-turn makes it process one prompt and exit while still
# applying the chat template (unlike --no-cnv which strips it and returns nothing).
bitnet_tps = smoke_model("BitNet b1.58 2B4T",           BITNET_CLI, BITNET_MODEL) \
             if RUN_BITNET else []
qwen_tps   = smoke_model("Qwen2.5-1.5B-Instruct Q8_0", QWEN_CLI,   QWEN_MODEL, ["--single-turn"]) \
             if RUN_QWEN else []

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
