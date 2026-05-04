"""
run_lm_eval.py — Start the BitNet llama-server, run lm-evaluation-harness, then stop the server.

Usage:
    python scripts/run_lm_eval.py [options]

Options:
    --tasks TASK[,TASK...]  lm-eval task names (default: arc_easy,arc_challenge,hellaswag,winogrande,mmlu)
    --num-fewshot INT       Few-shot count (default: 0; use 5 for MMLU)
    --threads INT           CPU threads for the server (default: 4)
    --ctx-size INT          Context size (default: 2048)
    --port INT              Server port (default: 8080)
    --limit INT             Limit samples per task for quick testing (default: no limit)
    --output-dir PATH       Where to save lm-eval JSON results (default: results/lm_eval/)
    --bitnet-dir PATH       Path to BitNet repo root (default: ../BitNet)
    --model PATH            Path to GGUF model (default: ../BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf)

Example (quick sanity check — 50 samples per task):
    python scripts/run_lm_eval.py --tasks arc_easy --limit 50

Example (full benchmark run):
    python scripts/run_lm_eval.py --tasks arc_easy,arc_challenge,hellaswag,winogrande --num-fewshot 0
    python scripts/run_lm_eval.py --tasks mmlu --num-fewshot 5
"""

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BITNET_DIR = REPO_ROOT.parent / "BitNet"
DEFAULT_MODEL = DEFAULT_BITNET_DIR / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "lm_eval"

DEFAULT_TASKS = "arc_easy,arc_challenge,hellaswag,winogrande,mmlu"


def find_server_binary(bitnet_dir: Path) -> Path:
    if platform.system() == "Windows":
        candidates = [
            bitnet_dir / "build" / "bin" / "Release" / "llama-server.exe",
            bitnet_dir / "build" / "bin" / "llama-server.exe",
        ]
    else:
        candidates = [bitnet_dir / "build" / "bin" / "llama-server"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"llama-server not found in {bitnet_dir}/build. Did you build bitnet.cpp?"
    )


def wait_for_server(port: int, timeout: int = 60) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False


def start_server(server_bin: Path, model: Path, threads: int, ctx_size: int, port: int) -> subprocess.Popen:
    cmd = [
        str(server_bin),
        "-m", str(model),
        "-c", str(ctx_size),
        "-t", str(threads),
        "-ngl", "0",
        "--host", "127.0.0.1",
        "--port", str(port),
        "-cb",
    ]
    print(f"Starting server on port {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def run_eval(tasks: str, num_fewshot: int, port: int, output_dir: Path, limit):
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "gguf",
        "--model_args", f"base_url={base_url}",
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--output_path", str(output_dir),
        "--log_samples",
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    print(f"Running lm-eval: tasks={tasks}, few-shot={num_fewshot}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness against BitNet via llama-server")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bitnet-dir", type=Path, default=DEFAULT_BITNET_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    server_bin = find_server_binary(args.bitnet_dir)

    server_proc = start_server(server_bin, args.model, args.threads, args.ctx_size, args.port)

    try:
        print("Waiting for server to be ready...")
        if not wait_for_server(args.port, timeout=120):
            print("ERROR: Server did not become ready within 120s.", file=sys.stderr)
            server_proc.terminate()
            sys.exit(1)
        print("Server ready.")

        rc = run_eval(args.tasks, args.num_fewshot, args.port, args.output_dir, args.limit)
    finally:
        print("Shutting down server...")
        server_proc.terminate()
        server_proc.wait(timeout=10)

    sys.exit(rc)


if __name__ == "__main__":
    main()
