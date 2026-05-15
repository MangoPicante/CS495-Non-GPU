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
    python scripts/smoke_test.py              # test all three models
    python scripts/smoke_test.py bitnet       # test BitNet only
    python scripts/smoke_test.py qwen         # test Qwen Q8_0 only
    python scripts/smoke_test.py qwen-q4      # test Qwen Q4_K_M only
    make smoke-test
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── Pretty-printing ───────────────────────────────────────────────────────────


def _enable_ansi_on_windows() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING on stdout (handle -11)
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


_enable_ansi_on_windows()
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
WIDTH = 64


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def bold(s):   return _c("1",  s)
def dim(s):    return _c("2",  s)
def red(s):    return _c("31", s)
def green(s):  return _c("32", s)
def yellow(s): return _c("33", s)
def cyan(s):   return _c("36", s)


def heading(title: str) -> None:
    bar = "━" * WIDTH
    print(f"\n{cyan(bar)}")
    print(f"  {bold(title)}")
    print(cyan(bar))


def subheading(title: str) -> None:
    print(f"\n  {bold(title)}")
    print(f"  {dim('─' * (WIDTH - 2))}")


def kv(key: str, value: str) -> None:
    print(f"      {dim(f'{key:<9}')} {value}")


def skip(name: str, reason: str) -> None:
    print(f"      {yellow('[ SKIP ]')}  {name}  {dim('(' + reason + ')')}")


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = green("[ PASS ]") if condition else red("[ FAIL ]")
    suffix = f"  {dim(f'({detail})')}" if detail and not condition else ""
    print(f"      {mark}  {name}{suffix}")
    if not condition:
        _failures.append(name)


# ── Config ────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="Smoke test for inference models and Phase 4 scripts.")
_parser.add_argument("model", nargs="?", choices=["bitnet", "qwen", "qwen-q4"], default=None,
                     help="Which model to test (default: all three)")
_args = _parser.parse_args()
RUN_BITNET  = _args.model in (None, "bitnet")
RUN_QWEN    = _args.model in (None, "qwen")
RUN_QWEN_Q4 = _args.model in (None, "qwen-q4")

ROOT      = Path(__file__).parent.parent
SCRIPTS   = ROOT / "scripts"
RESULTS   = ROOT / "results"

BITNET_DIR   = ROOT.parent / "Models" / "BitNet"
BITNET_MODEL = BITNET_DIR / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"
BITNET_CLI   = BITNET_DIR / "build" / "bin" / "Release" / "llama-cli.exe"

QWEN_DIR      = ROOT.parent / "Models" / "Qwen"
QWEN_MODEL    = QWEN_DIR / "qwen2.5-1.5b-instruct-q8_0.gguf"
QWEN_Q4_MODEL = QWEN_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
QWEN_CLI      = QWEN_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"

BITNET_SERVER    = BITNET_DIR / "build" / "bin" / "Release" / "llama-server.exe"
QWEN_SERVER      = QWEN_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
EVAL_PORT        = 8081
EVAL_SERVER_URL  = f"http://127.0.0.1:{EVAL_PORT}"
EVAL_SMOKE_OUT   = RESULTS / "smoke_accuracy.json"  # separate from main results; deleted each run

HARDWARE_RATE     = 0.170   # AWS c5.xlarge on-demand $/hr
N_TOKENS          = 16      # tokens to generate per test prompt (keep smoke test fast)
THREADS           = 4
CTX               = 512     # small context keeps KV cache tiny during smoke test
INFERENCE_TIMEOUT = 120     # seconds per llama-cli invocation before reporting TIMEOUT
EVAL_TIMEOUT      = 300     # seconds per eval task subprocess (dataset fetch + 1 sample)
EVAL_LIMIT        = 5       # samples per task in the smoke run

# (prompt, [acceptable keywords], test label)
# Direct questions only — open-ended continuations ("The sky is ___") produce
# too much output variance to be a reliable smoke check.
INFERENCE_CASES = [
    ("What is 2+2?",                          ["4", "four"], "basic arithmetic"),
    ("What is the capital of France?",        ["Paris"],     "factual recall"),
    ("What color is the sky on a clear day?", ["blue"],      "common sense"),
]

_failures: list[str] = []
_eval_server: subprocess.Popen | None = None


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
    subheading(name)
    kv("Model", model.name)
    kv("CLI",   f"{cli.name}  {dim('(' + cli.parent.parent.parent.parent.name + ')')}")
    kv("Pricing", f"AWS c5.xlarge  {dim('·')}  ${HARDWARE_RATE}/hr")

    if not model.exists():
        skip(name, f"model file not found: {model}")
        _failures.append(f"{name}: model not found")
        return []
    if not cli.exists():
        skip(name, f"{cli.name} not found: {cli}")
        _failures.append(f"{name}: llama-cli not found")
        return []

    tps_list = []
    for prompt, keywords, label in INFERENCE_CASES:
        print(f"\n    {bold(label)}")
        response, tps, rc = run_model(cli, model, prompt, extra_flags)
        kv("Prompt",   repr(prompt))
        kv("Response", repr(response))

        if response == "TIMEOUT":
            check(f"{label}: completed within {INFERENCE_TIMEOUT}s", False, "TIMEOUT")
            continue

        hit = any(kw.lower() in response.lower() for kw in keywords)
        expected_str = " or ".join(repr(k) for k in keywords)
        check(f"response contains {expected_str}", hit)

        if tps is not None:
            tps_list.append(tps)
            kv("Speed", f"{tps:5.1f} tok/s   {dim('·')}   {cost_per_1k(tps)} / 1k tokens")
        else:
            check(f"{label}: llama-cli exited cleanly", rc == 0, f"exit code {rc}")

    if tps_list:
        avg = sum(tps_list) / len(tps_list)
        print()
        kv("Average", f"{bold(f'{avg:5.1f} tok/s')}   {dim('·')}   {bold(cost_per_1k(avg))} / 1k tokens")

    return tps_list


def _start_eval_server(server_bin: Path, model: Path) -> bool:
    global _eval_server
    cmd = [
        str(server_bin), "-m", str(model),
        "-c", "512", "-t", str(THREADS), "-ub", "128", "-ngl", "0",
        "--host", "127.0.0.1", "--port", str(EVAL_PORT), "-cb",
    ]
    _eval_server = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(40):
        time.sleep(2)
        try:
            import requests as _req
            if _req.get(f"{EVAL_SERVER_URL}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
    return False


def _stop_eval_server():
    global _eval_server
    if _eval_server is not None:
        _eval_server.terminate()
        try:
            _eval_server.wait(timeout=5)
        except Exception:
            _eval_server.kill()
        _eval_server = None


# ── Inference smoke tests ─────────────────────────────────────────────────────

heading("Model Inference Smoke Tests")

# BitNet's llama.cpp fork is non-interactive by default: exits after -n tokens.
# Qwen's upstream llama.cpp defaults to interactive conversation mode and ignores
# stdin EOF; --single-turn makes it process one prompt and exit while still
# applying the chat template (unlike --no-cnv which strips it and returns nothing).
bitnet_tps  = smoke_model("BitNet b1.58 2B4T",            BITNET_CLI, BITNET_MODEL) \
              if RUN_BITNET else []
qwen_tps    = smoke_model("Qwen2.5-1.5B-Instruct Q8_0",   QWEN_CLI,   QWEN_MODEL,    ["--single-turn"]) \
              if RUN_QWEN else []
qwen_q4_tps = smoke_model("Qwen2.5-1.5B-Instruct Q4_K_M", QWEN_CLI,   QWEN_Q4_MODEL, ["--single-turn"]) \
              if RUN_QWEN_Q4 else []

if (bitnet_tps and qwen_tps) or (bitnet_tps and qwen_q4_tps) or (qwen_tps and qwen_q4_tps):
    heading(f"Cost Comparison  {dim(f'(AWS c5.xlarge @ ${HARDWARE_RATE}/hr)')}")
    hdr = f"{'Model':<35} {'tok/s':>8}   {'$/1k tokens':>13}"
    print(f"  {bold(hdr)}")
    print(f"  {dim('─' * (WIDTH - 2))}")
    for name, tps_list in [
        ("BitNet b1.58 2B4T",            bitnet_tps),
        ("Qwen2.5-1.5B-Instruct Q8_0",   qwen_tps),
        ("Qwen2.5-1.5B-Instruct Q4_K_M", qwen_q4_tps),
    ]:
        if tps_list:
            avg = sum(tps_list) / len(tps_list)
            print(f"  {name:<35} {avg:>8.1f}   {cost_per_1k(avg):>13}")

# ── compare_runs.py ───────────────────────────────────────────────────────────

heading("compare_runs.py")


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
    check("≥7 data rows (5 FP16 + paper + ours [+ Qwen ours])", len(rows) >= 7, f"got {len(rows)}")
    required_cols = {
        "model", "source", "throughput_tokens_s", "peak_rss_mb",
        "cost_per_1k_tokens", "energy_cost_per_1k_tokens",
        "arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu",
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

heading("metrics_tracker.py")

result = run_script(str(SCRIPTS / "metrics_tracker.py"), "--help")
check("--help exits 0", result.returncode == 0, result.stderr.strip())
check("--no-energy flag present", "--no-energy" in result.stdout)

# ── eval_accuracy.py ──────────────────────────────────────────────────────────

heading("eval_accuracy.py")

result = run_script(str(SCRIPTS / "eval_accuracy.py"), "--help")
check("--help exits 0", result.returncode == 0, result.stderr.strip())
check("--task flag present", "--task" in result.stdout)

# ── eval_accuracy.py: accuracy smoke (5 samples per task) ────────────────────

heading(f"eval_accuracy.py: Accuracy Smoke  {dim(f'(mmlu / arc_easy / arc_challenge / hellaswag, {EVAL_LIMIT} samples)')}")

# eval_accuracy.py defaults to --paper-targets=both, so each smoke run shows
# gaps against both BitNet i2_s and Qwen FP16 paper baselines per task.
_eval_configs = []
for _run_flag, _name, _server_bin, _model_path in [
    (RUN_BITNET,  "BitNet b1.58 2B4T",            BITNET_SERVER, BITNET_MODEL),
    (RUN_QWEN,    "Qwen2.5-1.5B-Instruct Q8_0",   QWEN_SERVER,   QWEN_MODEL),
    (RUN_QWEN_Q4, "Qwen2.5-1.5B-Instruct Q4_K_M", QWEN_SERVER,   QWEN_Q4_MODEL),
]:
    if not _run_flag:
        continue
    if not _server_bin.exists():
        skip(_name, f"llama-server not found: {_server_bin}")
        _failures.append(f"eval accuracy smoke {_name}: llama-server not found")
    elif not _model_path.exists():
        skip(_name, f"model not found: {_model_path}")
        _failures.append(f"eval accuracy smoke {_name}: model not found")
    else:
        _eval_configs.append((_name, _server_bin, _model_path))

for _name, _server_bin, _model_path in _eval_configs:
    subheading(_name)
    print(f"      {dim('Starting llama-server...')}", end=" ", flush=True)
    if not _start_eval_server(_server_bin, _model_path):
        print()
        check(f"{_name}: llama-server started", False, "timed out after ~80s")
        continue
    print(green("ready."))
    EVAL_SMOKE_OUT.unlink(missing_ok=True)  # always start fresh; no stale checkpoint
    for task in ("mmlu", "arc_easy", "arc_challenge", "hellaswag"):
        print(f"\n    {bold(task)}", flush=True)
        extra = ["--max-subjects", "1"] if task == "mmlu" else []
        try:
            r = subprocess.run(
                [sys.executable, str(SCRIPTS / "eval_accuracy.py"),
                 "--task", task, "--limit", str(EVAL_LIMIT),
                 "--out", str(EVAL_SMOKE_OUT),
                 "--server", EVAL_SERVER_URL, "--verbose",
                 "--model", str(_model_path),
                 # --paper-targets defaults to "both"; show BitNet i2_s and
                 # Qwen FP16 gaps for every task, regardless of which model
                 # is currently being smoke-tested.
                 "--no-energy", *extra],
                stderr=subprocess.PIPE, text=True, cwd=ROOT,
                timeout=EVAL_TIMEOUT,
            )
            check(f"{task}: exits 0", r.returncode == 0, r.stderr.strip()[:200])
            # An overall 0% is essentially impossible for these models on this
            # sample size by chance — it almost always indicates a scoring
            # methodology regression (e.g. prompt format mismatch, tokenizer
            # mismatch, broken continuation-scoring path). Flag it loudly.
            if r.returncode == 0 and EVAL_SMOKE_OUT.exists():
                try:
                    acc = json.loads(EVAL_SMOKE_OUT.read_text()).get(task, {}).get("accuracy")
                except (json.JSONDecodeError, OSError) as e:
                    check(f"{task}: accuracy readable", False, str(e)[:100])
                else:
                    ok = isinstance(acc, (int, float)) and acc > 0
                    check(f"{task}: accuracy > 0%", ok, f"got {acc!r}")
        except subprocess.TimeoutExpired:
            check(f"{task}: completed within {EVAL_TIMEOUT}s", False, "TIMEOUT")
    _stop_eval_server()

# ── Summary ───────────────────────────────────────────────────────────────────

print()
bar = "━" * WIDTH
if _failures:
    print(red(bar))
    print(f"  {bold(red(f'FAILED   {len(_failures)} check(s) failed'))}")
    print(red(bar))
    for f in _failures:
        print(f"    {red('•')} {f}")
    print(red(bar))
    sys.exit(1)
else:
    print(green(bar))
    print(f"  {bold(green('PASSED   all checks succeeded'))}")
    print(green(bar))
