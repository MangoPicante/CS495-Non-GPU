"""
Benchmark BitNet b1.58 2B4T inference via bitnet.cpp.

Runs the compiled llama-cli (or run_inference.py) binary, captures timing
data from llama_print_timings output, monitors peak RSS memory with psutil,
and optionally tracks energy with CodeCarbon.  Results are appended to
results/step_metrics.csv.

Usage:
    python scripts/metrics_tracker.py \
        --binary path/to/llama-cli \
        --model  path/to/ggml-model-i2_s.gguf \
        --threads 4 --runs 5 --n-predict 256
"""

import argparse
import csv
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:
    print("psutil not found — install it: pip install psutil", file=sys.stderr)
    sys.exit(1)

RESULTS_DIR = Path(__file__).parent.parent / "results"
METRICS_CSV = RESULTS_DIR / "step_metrics.csv"

CSV_FIELDS = [
    "timestamp",
    "model",
    "threads",
    "n_predict",
    "prompt_tokens",
    "generated_tokens",
    "load_time_ms",
    "prompt_eval_ms",
    "prompt_ms_per_token",
    "eval_ms",
    "eval_ms_per_token",
    "tokens_per_sec",
    "peak_memory_mb",
    "energy_kwh",
    "total_time_ms",
    "notes",
]

# Compiled against llama_print_timings format
_RE_LOAD = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
_RE_PROMPT = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*token")
_RE_EVAL = re.compile(r"\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*run")
_RE_TPS = re.compile(r"eval time.*?([\d.]+)\s*tokens per second")
_RE_TOTAL = re.compile(r"total time\s*=\s*([\d.]+)\s*ms")


def _parse_timings(output: str) -> dict:
    result: dict = {}

    m = _RE_LOAD.search(output)
    if m:
        result["load_time_ms"] = float(m.group(1))

    m = _RE_PROMPT.search(output)
    if m:
        ms, toks = float(m.group(1)), int(m.group(2))
        result["prompt_eval_ms"] = ms
        result["prompt_tokens"] = toks
        result["prompt_ms_per_token"] = round(ms / toks, 3) if toks else None

    m = _RE_EVAL.search(output)
    if m:
        ms, runs = float(m.group(1)), int(m.group(2))
        result["eval_ms"] = ms
        result["generated_tokens"] = runs
        result["eval_ms_per_token"] = round(ms / runs, 3) if runs else None

    m = _RE_TPS.search(output)
    if m:
        result["tokens_per_sec"] = float(m.group(1))

    m = _RE_TOTAL.search(output)
    if m:
        result["total_time_ms"] = float(m.group(1))

    return result


def _watch_memory(pid: int, stop: threading.Event, out: list) -> None:
    peak = 0.0
    try:
        proc = psutil.Process(pid)
        while not stop.is_set():
            try:
                peak = max(peak, proc.memory_info().rss / 1024**2)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)
    finally:
        out.append(peak)


def run_once(
    binary: str,
    model: str,
    prompt: str,
    n_predict: int,
    threads: int,
    extra_args: list[str],
    notes: str,
) -> dict:
    cmd = [
        binary,
        "-m", model,
        "-p", prompt,
        "-n", str(n_predict),
        "-t", str(threads),
        *extra_args,
    ]

    # Start optional CodeCarbon tracker
    tracker = None
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(log_level="error", save_to_file=False)
        tracker.start()
    except ImportError:
        pass

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stop_event = threading.Event()
    mem_buf: list[float] = []
    mem_thread = threading.Thread(
        target=_watch_memory, args=(proc.pid, stop_event, mem_buf), daemon=True
    )
    mem_thread.start()

    stdout, stderr = proc.communicate()
    stop_event.set()
    mem_thread.join(timeout=2)

    energy_kwh = tracker.stop() if tracker else None

    if proc.returncode != 0:
        print(f"\n[WARN] inference exited {proc.returncode}:\n{stderr[-500:]}", file=sys.stderr)

    timings = _parse_timings(stdout + "\n" + stderr)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": Path(model).stem,
        "threads": threads,
        "n_predict": n_predict,
        "prompt_tokens": timings.get("prompt_tokens", ""),
        "generated_tokens": timings.get("generated_tokens", ""),
        "load_time_ms": timings.get("load_time_ms", ""),
        "prompt_eval_ms": timings.get("prompt_eval_ms", ""),
        "prompt_ms_per_token": timings.get("prompt_ms_per_token", ""),
        "eval_ms": timings.get("eval_ms", ""),
        "eval_ms_per_token": timings.get("eval_ms_per_token", ""),
        "tokens_per_sec": timings.get("tokens_per_sec", ""),
        "peak_memory_mb": round(mem_buf[0], 1) if mem_buf else "",
        "energy_kwh": round(energy_kwh, 8) if energy_kwh is not None else "",
        "total_time_ms": timings.get("total_time_ms", ""),
        "notes": notes,
    }


def append_row(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BitNet b1.58 2B4T inference.")
    parser.add_argument("--binary", required=True, help="Path to llama-cli or run_inference binary")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument(
        "--prompt",
        default=(
            "Explain the significance of the French Revolution in three paragraphs."
        ),
        help="Input prompt for generation",
    )
    parser.add_argument("--n-predict", type=int, default=128, help="Tokens to generate per run")
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark repetitions")
    parser.add_argument("--notes", default="", help="Free-text notes stored with results")
    parser.add_argument("--output", default=str(METRICS_CSV), help="CSV file to append results to")
    args, extra_args = parser.parse_known_args()

    output_path = Path(args.output)
    print(f"Model  : {args.model}")
    print(f"Threads: {args.threads}   n_predict: {args.n_predict}   runs: {args.runs}")
    print(f"Output : {output_path}\n")

    for i in range(args.runs):
        print(f"Run {i + 1}/{args.runs} ... ", end="", flush=True)
        row = run_once(
            binary=args.binary,
            model=args.model,
            prompt=args.prompt,
            n_predict=args.n_predict,
            threads=args.threads,
            extra_args=extra_args,
            notes=args.notes,
        )
        append_row(row, output_path)
        tps = row.get("tokens_per_sec")
        mem = row.get("peak_memory_mb")
        tps_str = f"{float(tps):.1f} tok/s" if tps != "" else "n/a tok/s"
        mem_str = f"{float(mem):.0f} MB" if mem != "" else "n/a MB"
        print(f"{tps_str}  {mem_str}")

    print(f"\nDone — results in {output_path}")


if __name__ == "__main__":
    main()
