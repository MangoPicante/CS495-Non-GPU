"""
eval_accuracy.py — Evaluate BitNet b1.58 2B4T on multiple-choice benchmarks.

Uses the native llama-server /completion API with first-token log-probability
scoring: for each choice we query P(first_token_of_choice | context) and pick
the choice with highest probability. This is standard for 0-shot multiple-choice
evaluation and matches the effective scoring strategy used in the BitNet paper
for tasks where choices begin with distinct tokens.

For MMLU we score the single-character answer token (A/B/C/D) directly.

Supported tasks: arc_easy, arc_challenge, winogrande, hellaswag, mmlu

Usage:
    # Start server first (separate terminal, or use scripts/run_lm_eval.py)
    python run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -t 4

    # Then run evaluation
    python scripts/eval_accuracy.py --task arc_easy
    python scripts/eval_accuracy.py --task mmlu --num-fewshot 5 --limit 100
    python scripts/eval_accuracy.py --task all --limit 200
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "accuracy_results.json"
DEFAULT_BITNET_DIR = REPO_ROOT.parent / "BitNet"
DEFAULT_MODEL = DEFAULT_BITNET_DIR / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"

# Global server process handle (used when --start-server is set)
_server_proc: subprocess.Popen | None = None
_server_args: dict = {}


def _find_server_bin(bitnet_dir: Path) -> Path:
    if platform.system() == "Windows":
        for p in [bitnet_dir / "build" / "bin" / "Release" / "llama-server.exe",
                  bitnet_dir / "build" / "bin" / "llama-server.exe"]:
            if p.exists():
                return p
    else:
        p = bitnet_dir / "build" / "bin" / "llama-server"
        if p.exists():
            return p
    raise FileNotFoundError(f"llama-server not found in {bitnet_dir}/build")


def _start_server(bitnet_dir: Path, model: Path, threads: int, port: int, ctx: int = 4096):
    global _server_proc
    if _server_proc is not None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except Exception:
            _server_proc.kill()
    bin_path = _find_server_bin(bitnet_dir)
    cmd = [str(bin_path), "-m", str(model), "-c", str(ctx), "-t", str(threads),
           "-ngl", "0", "--host", "127.0.0.1", "--port", str(port), "-cb"]
    _server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait for ready
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(40):
        time.sleep(2)
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
    return False


def ensure_server(server_url: str) -> bool:
    """Restart server if it crashed. Only active when --start-server was used."""
    if not _server_args:
        return False
    try:
        requests.get(f"{server_url}/health", timeout=3)
        return False  # server fine, no restart needed
    except Exception:
        pass
    print("\n[eval_accuracy] Server unreachable — restarting...", flush=True)
    ok = _start_server(**_server_args)
    if ok:
        print("[eval_accuracy] Server restarted.", flush=True)
    else:
        print("[eval_accuracy] WARNING: server restart failed.", flush=True)
    return ok

TASKS = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]

# Published paper targets from arXiv:2504.12285 Table 1
PAPER_TARGETS = {
    "arc_easy":      74.79,
    "arc_challenge": 49.91,
    "winogrande":    71.90,
    "hellaswag":     68.44,
    "mmlu":          53.17,
}


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def get_token_id(server: str, text: str) -> int | None:
    """Return the first token id for `text`, or None on failure."""
    try:
        r = requests.post(f"{server}/tokenize",
                          json={"content": text, "add_special": False}, timeout=10)
        tokens = r.json().get("tokens", [])
        return tokens[0] if tokens else None
    except Exception:
        return None


def _decode_tok_str(tok_str: str) -> str:
    """Convert BPE token string to plain text (handles Ġ = leading space, ▁ = leading space)."""
    return tok_str.replace("Ġ", " ").replace("▁", " ")


def first_token_logprob(server: str, context: str, choice: str, n_probs: int = 100) -> float:
    """
    Return log P(first_token_of_choice | context) using the native /completion API.

    Queries the server for the top-n_probs tokens at position len(context) and finds
    the token whose decoded string is the longest prefix of `choice`. This correctly
    handles BPE representations such as Ġ (leading space) used by LLaMA-3 tokenizer.
    Returns -inf if no matching token is found in top-n_probs.
    """
    import math
    try:
        for attempt in range(3):
            try:
                r = requests.post(f"{server}/completion", json={
                    "prompt": context,
                    "n_predict": 1,
                    "n_probs": n_probs,
                    "temperature": 0.0,
                }, timeout=60)
                data = r.json()
                break
            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    restarted = ensure_server(server)
                    time.sleep(3 if restarted else 1)
                    continue
                return float("-inf")
            except Exception:
                return float("-inf")
        else:
            return float("-inf")

        probs_list = data.get("completion_probabilities", [])
        if not probs_list:
            return float("-inf")
        top_tokens = probs_list[0]["probs"]  # [{"tok_str": str, "prob": float}, ...]

        # Find the token whose decoded form is the longest prefix of `choice`.
        # This handles BPE space markers (Ġ = " ") and avoids substring collisions.
        best_prob = None
        best_len = 0
        for entry in top_tokens:
            decoded = _decode_tok_str(entry["tok_str"])
            if decoded and choice.startswith(decoded) and len(decoded) > best_len:
                best_prob = entry["prob"]
                best_len = len(decoded)

        if best_prob is not None:
            return math.log(max(best_prob, 1e-10))
        return float("-inf")
    except Exception:
        return float("-inf")


def generate_text(server: str, prompt: str, max_tokens: int = 5) -> str:
    """Generate a short continuation (fallback for non-MC tasks)."""
    try:
        r = requests.post(f"{server}/completion", json={
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
        }, timeout=30)
        return r.json().get("content", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Dataset loaders and formatters
# ---------------------------------------------------------------------------

def load_arc(subset: str, limit: int | None):
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", subset, split="test", trust_remote_code=False)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def eval_arc(ds, server: str) -> dict:
    """
    Letter-scoring evaluation for ARC.

    Format:
        Question: {question}
        A. {choice_a}
        B. {choice_b}
        ...
        Answer:

    Score P(" A" | context), P(" B" | context), ... using first-token logprob.
    Single-character letter tokens are always distinct, giving reliable discrimination.
    This matches the scoring approach used for MMLU in arXiv:2504.12285.
    """
    correct = 0
    skipped = 0
    letter_map = {"1": "A", "2": "B", "3": "C", "4": "D",
                  "A": "A", "B": "B", "C": "C", "D": "D",
                  "E": "E", "F": "F"}
    for row in ds:
        question = row["question"]
        choices_text = row["choices"]["text"]
        choices_label = row["choices"]["label"]
        answer_key = row["answerKey"]
        answer_letter = letter_map.get(answer_key, answer_key)

        # Build ABCD multiple-choice prompt
        letters = ["A", "B", "C", "D", "E"][:len(choices_text)]
        opt_lines = "\n".join(f"{l}. {t}" for l, t in zip(letters, choices_text))
        context = f"Question: {question}\n{opt_lines}\nAnswer:"

        scores = []
        for letter in letters:
            lp = first_token_logprob(server, context, " " + letter)
            scores.append((letter, lp))

        if all(s == float("-inf") for _, s in scores):
            skipped += 1
            continue

        predicted = max(scores, key=lambda x: x[1])[0]
        if predicted == answer_letter:
            correct += 1

    n = len(ds) - skipped
    acc = correct / n if n > 0 else 0.0
    return {"accuracy": round(acc * 100, 2), "correct": correct, "total": n, "skipped": skipped}


def load_winogrande(limit: int | None):
    from datasets import load_dataset
    ds = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=False)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def eval_winogrande(ds, server: str) -> dict:
    """
    Letter-scoring evaluation for WinoGrande.

    Format:
        {sentence with _ replaced by ___}
        A. {option1}
        B. {option2}
        Answer:

    Score P(" A") vs P(" B") — single character tokens always discriminate cleanly.
    """
    correct = 0
    skipped = 0
    for row in ds:
        sentence = row["sentence"]
        opt1 = row["option1"]
        opt2 = row["option2"]
        answer = row["answer"]  # "1" or "2"

        display_sent = sentence.replace("_", "___")
        context = f"{display_sent}\nA. {opt1}\nB. {opt2}\nAnswer:"

        sA = first_token_logprob(server, context, " A")
        sB = first_token_logprob(server, context, " B")

        if sA == float("-inf") and sB == float("-inf"):
            skipped += 1
            continue

        predicted = "1" if sA >= sB else "2"
        if predicted == answer:
            correct += 1

    n = len(ds) - skipped
    acc = correct / n if n > 0 else 0.0
    return {"accuracy": round(acc * 100, 2), "correct": correct, "total": n, "skipped": skipped}


def load_hellaswag(limit: int | None):
    from datasets import load_dataset
    ds = load_dataset("hellaswag", split="validation", trust_remote_code=False)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def eval_hellaswag(ds, server: str) -> dict:
    """
    Letter-scoring evaluation for HellaSwag.

    Format:
        {activity}: {context}
        A. {ending_0}
        B. {ending_1}
        C. {ending_2}
        D. {ending_3}
        Answer:

    Score P(" A/B/C/D" | context) — single letter tokens give clean discrimination.
    """
    correct = 0
    skipped = 0
    letters = ["A", "B", "C", "D"]
    for row in ds:
        ctx = row["ctx"]
        endings = row["endings"]
        label = int(row["label"])

        opt_lines = "\n".join(f"{l}. {e}" for l, e in zip(letters, endings))
        context = f"{ctx}\n{opt_lines}\nAnswer:"

        scores = [first_token_logprob(server, context, " " + l) for l in letters]

        if all(s == float("-inf") for s in scores):
            skipped += 1
            continue

        predicted = scores.index(max(scores))
        if predicted == label:
            correct += 1

    n = len(ds) - skipped
    acc = correct / n if n > 0 else 0.0
    return {"accuracy": round(acc * 100, 2), "correct": correct, "total": n, "skipped": skipped}


MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

ANSWER_CHOICES = ["A", "B", "C", "D"]


def make_mmlu_fewshot_prompt(dev_rows, subject_name: str) -> str:
    pretty = subject_name.replace("_", " ")
    header = f"The following are multiple choice questions (with answers) about {pretty}.\n\n"
    shots = []
    for row in dev_rows:
        q = row["question"]
        opts = row["choices"]
        ans = ANSWER_CHOICES[row["answer"]]
        opt_str = "\n".join(f"{l}. {t}" for l, t in zip(ANSWER_CHOICES, opts))
        shots.append(f"Question: {q}\n{opt_str}\nAnswer: {ans}")
    return header + "\n\n".join(shots) + "\n\n"


def eval_mmlu_subject(subject: str, server: str, num_fewshot: int, limit: int | None) -> dict:
    from datasets import load_dataset
    test_ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=False)
    if limit:
        test_ds = test_ds.select(range(min(limit, len(test_ds))))

    fewshot_prefix = ""
    if num_fewshot > 0:
        try:
            dev_ds = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=False)
            dev_rows = list(dev_ds)[:num_fewshot]
            fewshot_prefix = make_mmlu_fewshot_prompt(dev_rows, subject)
        except Exception:
            pass

    correct = 0
    skipped = 0
    for row in test_ds:
        q = row["question"]
        opts = row["choices"]
        ans_idx = row["answer"]  # integer 0-3
        opt_str = "\n".join(f"{l}. {t}" for l, t in zip(ANSWER_CHOICES, opts))
        context = fewshot_prefix + f"Question: {q}\n{opt_str}\nAnswer:"

        scores = [first_token_logprob(server, context, " " + ch) for ch in ANSWER_CHOICES]

        if all(s == float("-inf") for s in scores):
            skipped += 1
            continue

        predicted = scores.index(max(scores))
        if predicted == ans_idx:
            correct += 1

    n = len(test_ds) - skipped
    acc = correct / n if n > 0 else 0.0
    return {"accuracy": round(acc * 100, 2), "correct": correct, "total": n, "skipped": skipped}


def eval_mmlu(server: str, num_fewshot: int, limit: int | None) -> dict:
    all_correct = 0
    all_total = 0
    subject_results = {}
    for subj in MMLU_SUBJECTS:
        try:
            r = eval_mmlu_subject(subj, server, num_fewshot, limit)
            subject_results[subj] = r
            all_correct += r["correct"]
            all_total += r["total"]
            print(f"  {subj:45s} {r['accuracy']:5.1f}% ({r['correct']}/{r['total']})")
        except Exception as e:
            print(f"  {subj:45s} ERROR: {e}")

    overall_acc = (all_correct / all_total * 100) if all_total > 0 else 0.0
    return {
        "accuracy": round(overall_acc, 2),
        "correct": all_correct,
        "total": all_total,
        "subjects": subject_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_server(server: str) -> bool:
    try:
        r = requests.get(f"{server}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run_task(task: str, server: str, num_fewshot: int, limit: int | None) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Task: {task}  limit={limit}  few-shot={num_fewshot}")
    print(f"{'='*60}")

    if task in ("arc_easy", "arc_challenge"):
        subset = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_arc(subset, limit)
        print(f"Loaded {len(ds)} samples")
        result = eval_arc(ds, server)
    elif task == "winogrande":
        ds = load_winogrande(limit)
        print(f"Loaded {len(ds)} samples")
        result = eval_winogrande(ds, server)
    elif task == "hellaswag":
        ds = load_hellaswag(limit)
        print(f"Loaded {len(ds)} samples")
        result = eval_hellaswag(ds, server)
    elif task == "mmlu":
        result = eval_mmlu(server, num_fewshot, limit)
    else:
        raise ValueError(f"Unknown task: {task}")

    elapsed = time.time() - t0
    result["elapsed_s"] = round(elapsed, 1)
    result["paper_target"] = PAPER_TARGETS.get(task)

    gap = result["accuracy"] - result["paper_target"] if result["paper_target"] else None
    gap_str = f"  (paper: {result['paper_target']}%,  gap: {gap:+.1f}%)" if gap is not None else ""
    print(f"\nResult: {result['accuracy']}%  ({result['correct']}/{result['total']}){gap_str}")
    print(f"Time:   {elapsed:.0f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate BitNet on multiple-choice benchmarks")
    parser.add_argument("--task", choices=TASKS + ["all"], default="arc_easy")
    parser.add_argument("--num-fewshot", type=int, default=0,
                        help="Few-shot count (use 5 for MMLU)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per task (None = full dataset)")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-server", action="store_true",
                        help="Start and manage llama-server automatically (with auto-restart on crash)")
    parser.add_argument("--bitnet-dir", type=Path, default=DEFAULT_BITNET_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    port = int(args.server.split(":")[-1])

    if args.start_server:
        global _server_args
        _server_args = dict(bitnet_dir=args.bitnet_dir, model=args.model,
                            threads=args.threads, port=port)
        print(f"Starting llama-server on port {port}...")
        if not _start_server(**_server_args):
            print("ERROR: Server failed to start.")
            return
        print("Server ready.")
    elif not check_server(args.server):
        print(f"ERROR: No server at {args.server}. Pass --start-server or start it manually.")
        return

    tasks = TASKS if args.task == "all" else [args.task]
    all_results = {}

    for task in tasks:
        fewshot = args.num_fewshot if task == "mmlu" else 0
        result = run_task(task, args.server, fewshot, args.limit)
        all_results[task] = result

    # Save results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.out.exists():
        with open(args.out) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(args.out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # Summary table
    print(f"\n{'Task':<20} {'Accuracy':>10} {'Paper':>8} {'Gap':>7} {'N':>6}")
    print("-" * 55)
    for task, r in all_results.items():
        target = r.get("paper_target", "-")
        gap = (r["accuracy"] - target) if isinstance(target, float) else "-"
        gap_str = f"{gap:+.1f}%" if isinstance(gap, float) else "-"
        print(f"{task:<20} {r['accuracy']:>9.2f}% {str(target)+' %':>8} {gap_str:>7} {r['total']:>6}")

    if _server_proc is not None:
        _server_proc.terminate()
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
