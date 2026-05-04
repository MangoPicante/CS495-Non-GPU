"""
metrics_tracker.py — Run a single BitNet b1.58 inference benchmark and append results to step_metrics.csv.

Usage:
    python scripts/metrics_tracker.py [options]

Options:
    --model PATH        Path to GGUF model (default: ../BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf)
    --bitnet-dir PATH   Path to BitNet repo root (default: ../BitNet)
    --prompt TEXT       Prompt string (default: a fixed benchmark prompt)
    --n-tokens INT      Tokens to generate (default: 128)
    --threads INT       CPU threads (default: 4)
    --ctx-size INT      Context size (default: 2048)
    --tag TEXT          Optional label stored in the 'tag' column (e.g. "baseline", "t8")
    --out PATH          Output CSV (default: results/step_metrics.csv)
    --no-energy         Skip CodeCarbon energy measurement
"""

import argparse
import csv
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BITNET_DIR = REPO_ROOT.parent / "BitNet"
DEFAULT_MODEL = DEFAULT_BITNET_DIR / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"
DEFAULT_OUT = REPO_ROOT / "results" / "step_metrics.csv"

BENCHMARK_PROMPT = (
    "The scientific method is a systematic approach to understanding the natural world. "
    "It involves making observations, forming hypotheses, designing experiments, collecting data, "
    "and drawing conclusions. Explain each step in detail."
)

CSV_FIELDS = [
    "timestamp", "tag", "model", "threads", "n_tokens",
    "prompt_tokens", "gen_tokens",
    "prompt_ms_per_token", "gen_ms_per_token", "gen_tokens_per_sec",
    "load_time_ms", "total_time_ms",
    "peak_rss_mb",
    "energy_kg_co2", "energy_kwh",
]


def find_llama_cli(bitnet_dir: Path) -> Path:
    if platform.system() == "Windows":
        candidates = [
            bitnet_dir / "build" / "bin" / "Release" / "llama-cli.exe",
            bitnet_dir / "build" / "bin" / "llama-cli.exe",
        ]
    else:
        candidates = [
            bitnet_dir / "build" / "bin" / "llama-cli",
        ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"llama-cli not found in {bitnet_dir}/build. Did you build bitnet.cpp?"
    )


def run_inference(llama_cli: Path, model: Path, prompt: str, n_tokens: int,
                  threads: int, ctx_size: int) -> tuple[str, float, int]:
    """Run llama-cli and return (combined stdout+stderr, wall_time_s, peak_rss_mb)."""
    cmd = [
        str(llama_cli),
        "-m", str(model),
        "-n", str(n_tokens),
        "-t", str(threads),
        "-p", prompt,
        "-ngl", "0",
        "-c", str(ctx_size),
        "--temp", "0",       # greedy decoding for reproducibility
        "-b", "1",
        "--no-warmup",       # skip warmup for cleaner timing
    ]

    peak_rss_mb = None
    start = time.perf_counter()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    rss_samples = []

    def _monitor_memory():
        if not HAS_PSUTIL:
            return
        try:
            ps = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    rss_samples.append(ps.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.1)
        except Exception:
            pass

    monitor_thread = threading.Thread(target=_monitor_memory, daemon=True)
    monitor_thread.start()

    # communicate() reads stdout to prevent pipe deadlock on Windows
    stdout, _ = proc.communicate()
    wall_time = time.perf_counter() - start
    monitor_thread.join(timeout=2)

    if rss_samples:
        peak_rss_mb = max(rss_samples) / (1024 ** 2)

    if proc.returncode != 0:
        print("llama-cli stderr/stdout:", stdout[-2000:], file=sys.stderr)
        raise RuntimeError(f"llama-cli exited with code {proc.returncode}")

    return stdout, wall_time, peak_rss_mb


def parse_perf(output: str) -> dict:
    """Extract timing stats from llama-cli perf footer."""
    metrics = {}

    m = re.search(r"load time\s*=\s*([\d.]+)\s*ms", output)
    if m:
        metrics["load_time_ms"] = float(m.group(1))

    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token", output)
    if m:
        metrics["prompt_ms_per_token"] = float(m.group(3))
        metrics["prompt_tokens"] = int(m.group(2))

    m = re.search(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second", output)
    if m:
        metrics["gen_ms_per_token"] = float(m.group(3))
        metrics["gen_tokens_per_sec"] = float(m.group(4))
        metrics["gen_tokens"] = int(m.group(2))

    m = re.search(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    if m:
        metrics["total_time_ms"] = float(m.group(1))

    return metrics


def append_csv(out_path: Path, row: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="BitNet inference benchmark tracker")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--bitnet-dir", type=Path, default=DEFAULT_BITNET_DIR)
    parser.add_argument("--prompt", type=str, default=BENCHMARK_PROMPT)
    parser.add_argument("--n-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-energy", action="store_true")
    args = parser.parse_args()

    llama_cli = find_llama_cli(args.bitnet_dir)
    print(f"Binary : {llama_cli}")
    print(f"Model  : {args.model}")
    print(f"Threads: {args.threads}   Tokens: {args.n_tokens}")
    print()

    if not HAS_PSUTIL:
        print("Warning: psutil not installed — peak RSS will not be recorded. pip install psutil")
    if not HAS_CODECARBON and not args.no_energy:
        print("Warning: codecarbon not installed — energy will not be recorded. pip install codecarbon")

    tracker = None
    if HAS_CODECARBON and not args.no_energy:
        tracker = EmissionsTracker(
            project_name="bitnet-benchmark",
            output_dir=str(args.out.parent),
            log_level="error",
            save_to_file=False,
        )
        tracker.start()

    output, wall_time, peak_rss_mb = run_inference(
        llama_cli, args.model, args.prompt, args.n_tokens, args.threads, args.ctx_size
    )

    energy_kg_co2 = None
    energy_kwh = None
    if tracker is not None:
        emissions = tracker.stop()
        energy_kg_co2 = emissions  # kg CO2eq
        energy_kwh = tracker._total_energy.kWh if hasattr(tracker, "_total_energy") else None

    perf = parse_perf(output)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "model": args.model.name,
        "threads": args.threads,
        "n_tokens": args.n_tokens,
        "prompt_tokens": perf.get("prompt_tokens", ""),
        "gen_tokens": perf.get("gen_tokens", ""),
        "prompt_ms_per_token": perf.get("prompt_ms_per_token", ""),
        "gen_ms_per_token": perf.get("gen_ms_per_token", ""),
        "gen_tokens_per_sec": perf.get("gen_tokens_per_sec", ""),
        "load_time_ms": perf.get("load_time_ms", ""),
        "total_time_ms": perf.get("total_time_ms", ""),
        "peak_rss_mb": f"{peak_rss_mb:.1f}" if peak_rss_mb else "",
        "energy_kg_co2": f"{energy_kg_co2:.6f}" if energy_kg_co2 else "",
        "energy_kwh": f"{energy_kwh:.6f}" if energy_kwh else "",
    }

    append_csv(args.out, row)

    print("--- Results ---")
    print(f"  Load time        : {row['load_time_ms']} ms")
    print(f"  Prompt eval      : {row['prompt_ms_per_token']} ms/token ({row['prompt_tokens']} tokens)")
    print(f"  Generation       : {row['gen_ms_per_token']} ms/token  ({row['gen_tokens_per_sec']} tok/s)")
    print(f"  Peak RSS         : {row['peak_rss_mb']} MB")
    print(f"  Energy           : {row['energy_kg_co2']} kg CO2eq")
    print(f"  Wall time        : {wall_time:.2f} s")
    print(f"\nAppended to {args.out}")


if __name__ == "__main__":
    main()
