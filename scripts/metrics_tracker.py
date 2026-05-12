"""
Benchmark runner for llama.cpp-based models (BitNet, Qwen, etc.).

Invokes llama-bench with a fixed set of prompt/generation sizes, captures
per-run JSON output, measures peak RSS via psutil, and tracks energy and
CO2 via CodeCarbon (enabled by default; skip with --no-energy).  Results
are appended to the output CSV (default: results/step_metrics.csv).

Usage:
    # BitNet
    python scripts/metrics_tracker.py \
        --bitnet-dir ../Models/BitNet \
        --model ../Models/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
        --threads 4

    # Qwen (or any other llama.cpp build)
    python scripts/metrics_tracker.py \
        --bitnet-dir ../Models/Qwen/llama.cpp \
        --model ../Models/Qwen/qwen2.5-1.5b-instruct-q8_0.gguf \
        --out results/qwen_metrics.csv \
        --threads 4
"""

import argparse
import contextlib
import csv
import io
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import psutil

DEFAULT_OUT = Path(__file__).parent.parent / "results" / "step_metrics.csv"

CSV_FIELDS = [
    "timestamp",
    "threads",
    "n_prompt",
    "n_gen",
    "avg_latency_ms_token",
    "throughput_tokens_s",
    "peak_rss_mb",
    "energy_kwh",
    "co2_kg",
]

# (n_prompt, n_gen) pairs matching arXiv:2504.12285 Table 1 conditions
BENCH_CONFIGS = [
    (512, 128),
    (512, 512),
    (1, 512),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bitnet-dir", required=True, help="Path to a built llama.cpp repo (BitNet or Qwen)")
    p.add_argument("--model", required=True, help="Path to .gguf model file")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--ubatch", type=int, default=128,
        help="Micro-batch size passed to llama-bench -ub (default 128; "
             "BitNet TL2 kernels require <=128 to avoid stack overflow)",
    )
    p.add_argument(
        "--no-energy",
        action="store_true",
        help="Skip CodeCarbon energy tracking (faster, no internet needed)",
    )
    return p.parse_args()


def find_llama_bench(bitnet_dir: Path) -> Path:
    candidate = bitnet_dir / "build" / "bin" / "Release" / "llama-bench.exe"
    if candidate.exists():
        return candidate
    # fallback for non-Windows builds
    candidate2 = bitnet_dir / "build" / "bin" / "llama-bench"
    if candidate2.exists():
        return candidate2
    sys.exit(f"llama-bench not found under {bitnet_dir}/build. Run 'make bitnet-build' or 'make qwen-build' first.")


def run_bench(llama_bench: Path, model: Path, n_prompt: int, n_gen: int, threads: int, ubatch: int = 128):
    cmd = [
        str(llama_bench),
        "-m", str(model),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-t", str(threads),
        "-ub", str(ubatch),
        "-o", "json",
        "-r", "3",   # 3 repetitions → median is more stable
    ]
    proc = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    peak_rss = 0
    stdout_chunks = []

    while proc.poll() is None:
        try:
            mem = proc.memory_info().rss
            if mem > peak_rss:
                peak_rss = mem
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print(stderr.decode(errors="replace"), file=sys.stderr)
        sys.exit(f"llama-bench exited with code {proc.returncode}")

    # also sample after it finishes in case RSS peaked at end
    try:
        mem = proc.memory_info().rss
        if mem > peak_rss:
            peak_rss = mem
    except psutil.NoSuchProcess:
        pass

    return json.loads(stdout.decode()), peak_rss


def extract_metrics(bench_json: list[dict], n_prompt: int, n_gen: int) -> dict:
    """Pull latency and throughput from llama-bench JSON output."""
    pp_entry = next(
        (r for r in bench_json if r.get("n_prompt", 0) == n_prompt and r.get("n_gen", 0) == 0),
        None,
    )
    tg_entry = next(
        (r for r in bench_json if r.get("n_prompt", 0) == 0 and r.get("n_gen", 0) == n_gen),
        None,
    )
    # llama-bench reports t/s; convert to ms/token
    tg_ts = tg_entry["avg_ts"] if tg_entry else None
    latency = (1000.0 / tg_ts) if tg_ts else None
    throughput = tg_ts
    return {"avg_latency_ms_token": latency, "throughput_tokens_s": throughput}


def ensure_csv(out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        with out.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_row(row: dict, out: Path):
    with out.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def main():
    args = parse_args()
    bitnet_dir = Path(args.bitnet_dir)
    model = Path(args.model)
    out = args.out
    if not model.exists():
        sys.exit(f"Model not found: {model}")

    llama_bench = find_llama_bench(bitnet_dir)
    ensure_csv(out)

    tracker = None
    if not args.no_energy:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                from codecarbon import EmissionsTracker
            logging.getLogger("codecarbon").setLevel(logging.CRITICAL)
            tracker = EmissionsTracker(
                project_name="bitnet-benchmark",
                output_dir=str(RESULTS_CSV.parent),
                log_level="error",
                save_to_file=False,
            )
        except ImportError:
            print("codecarbon not installed — skipping energy tracking", file=sys.stderr)

    for n_prompt, n_gen in BENCH_CONFIGS:
        print(f"\n--- n_prompt={n_prompt}  n_gen={n_gen}  threads={args.threads} ---")

        energy_kwh = None
        co2_kg = None

        if tracker:
            tracker.start()

        bench_json, peak_rss = run_bench(llama_bench, model, n_prompt, n_gen, args.threads, args.ubatch)

        if tracker:
            # codecarbon 2.7 has a Windows lock-file bug that spams stderr;
            # redirect_stderr silences it without affecting our own output.
            with contextlib.redirect_stderr(io.StringIO()):
                emissions = tracker.stop()
            co2_kg = float(emissions) if emissions is not None else None
            try:
                energy_kwh = tracker.final_emissions_data.energy_consumed
            except Exception:
                energy_kwh = None

        metrics = extract_metrics(bench_json, n_prompt, n_gen)
        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "threads": args.threads,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "avg_latency_ms_token": round(metrics["avg_latency_ms_token"], 3) if metrics["avg_latency_ms_token"] else "",
            "throughput_tokens_s": round(metrics["throughput_tokens_s"], 3) if metrics["throughput_tokens_s"] else "",
            "peak_rss_mb": round(peak_rss / 1024 / 1024, 1),
            "energy_kwh": round(energy_kwh, 6) if energy_kwh else "",
            "co2_kg": round(co2_kg, 6) if co2_kg else "",
        }
        append_row(row, out)
        print(
            f"  latency={row['avg_latency_ms_token']} ms/tok  "
            f"throughput={row['throughput_tokens_s']} tok/s  "
            f"RSS={row['peak_rss_mb']} MB"
        )

    print(f"\nResults appended to {out}")


if __name__ == "__main__":
    main()
