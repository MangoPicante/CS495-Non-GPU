"""
eval_accuracy.py — Evaluate llama.cpp-based models on multiple-choice benchmarks.

Two scoring strategies are used depending on the task:

  First-token scoring (ARC, MMLU):
    Query P(first_token_of_choice | context) and pick the highest.  Works well
    when choices begin with distinct single tokens (letter labels A/B/C/D).

  Continuation scoring (WinoGrande, HellaSwag):
    Compute sum of log P(token_i | context + tokens_0..i-1) over all tokens in
    each candidate continuation, then pick the highest.  HellaSwag scores are
    additionally normalized by token count to remove length bias.  This matches
    the methodology used in the paper (arXiv:2504.12285) for these tasks and produces
    meaningful scores rather than near-random letter-guessing.

Supported tasks: arc_easy, arc_challenge, winogrande, hellaswag, mmlu

Usage:
    # Start server first (separate terminal), or use --start-server
    python scripts/eval_accuracy.py --task arc_easy --start-server
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
DEFAULT_LLAMA_DIR = REPO_ROOT.parent / "Models" / "BitNet"
DEFAULT_MODEL = DEFAULT_LLAMA_DIR / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"

# Global server process handle (used when --start-server is set)
_server_proc: subprocess.Popen | None = None
_server_args: dict = {}


def _find_server_bin(llama_dir: Path) -> Path:
    if platform.system() == "Windows":
        for p in [llama_dir / "build" / "bin" / "Release" / "llama-server.exe",
                  llama_dir / "build" / "bin" / "llama-server.exe"]:
            if p.exists():
                return p
    else:
        p = llama_dir / "build" / "bin" / "llama-server"
        if p.exists():
            return p
    raise FileNotFoundError(f"llama-server not found in {llama_dir}/build")


def _start_server(llama_dir: Path, model: Path, threads: int, port: int, ctx: int = 4096):
    global _server_proc
    if _server_proc is not None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except Exception:
            _server_proc.kill()
    bin_path = _find_server_bin(llama_dir)
    cmd = [str(bin_path), "-m", str(model), "-c", str(ctx), "-t", str(threads),
           "-ub", "128",  # conservative default; BitNet TL2 kernels require <=128
           "-ngl", "0", "--host", "127.0.0.1", "--port", str(port), "-cb"]
    _server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        return False
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

def _decode_tok_str(tok_str: str) -> str:
    return tok_str.replace("Ġ", " ").replace("▁", " ")


def _query_completion(server: str, context: str, n_probs: int) -> list | None:
    """POST one /completion request; return top-token list or None on failure."""
    for attempt in range(3):
        try:
            r = requests.post(f"{server}/completion", json={
                "prompt": context,
                "n_predict": 1,
                "n_probs": n_probs,
                "temperature": 0.0,
            }, timeout=60)
        except requests.exceptions.ConnectionError:
            if attempt < 2:
                restarted = ensure_server(server)
                time.sleep(3 if restarted else 1)
                continue
            return None
        except Exception:
            return None
        try:
            data = r.json()
        except Exception:
            return None
        probs_list = data.get("completion_probabilities", [])
        if not probs_list:
            return []
        item = probs_list[0]
        # New llama.cpp: {"token": "...", "logprob": ..., "top_logprobs": [...]}
        # Old llama.cpp: {"probs": [{"tok_str": "...", "prob": ...}]}
        return item.get("top_logprobs") or item.get("probs", [])
    return None


def _best_prefix_match(top_tokens: list, target: str) -> tuple[float, str]:
    """
    Find the token in top_tokens whose decoded form is the longest prefix of
    target.  Returns (log_prob, matched_text) or (float('-inf'), '') if none match.

    Handles both llama.cpp response formats:
      Old: {"tok_str": "▁A", "prob": 0.8}   (prob is a probability, must log it)
      New: {"token": " A",  "logprob": -0.2} (already a log-probability)
    """
    import math
    best_logprob = None
    best_len = 0
    best_decoded = ""
    for entry in top_tokens:
        raw = entry.get("tok_str") or entry.get("token", "")
        decoded = _decode_tok_str(raw)
        if decoded and target.startswith(decoded) and len(decoded) > best_len:
            best_len = len(decoded)
            best_decoded = decoded
            if "logprob" in entry:
                best_logprob = entry["logprob"]
            else:
                best_logprob = math.log(max(entry.get("prob", 0.0), 1e-10))
    if best_logprob is not None:
        return best_logprob, best_decoded
    return float("-inf"), ""


def first_token_logprob(server: str, context: str, choice: str, n_probs: int = 100) -> float:
    """Return log P(first token of choice | context).  Used by ARC and MMLU."""
    top_tokens = _query_completion(server, context, n_probs)
    if not top_tokens:
        return float("-inf")
    logp, _ = _best_prefix_match(top_tokens, choice)
    return logp


def continuation_logprob(
    server: str, context: str, continuation: str, normalize: bool = False
) -> float:
    """
    Return sum of log P(token_i | context + tokens_0..i-1) over every token in
    `continuation`, making one API call per token.

    If normalize=True the sum is divided by token count (use for HellaSwag to
    remove length bias between candidate endings of unequal length).

    The leading space in `continuation` should encode the word-boundary: BPE
    tokenizers represent word-initial tokens with a Ġ prefix (decoded as " "),
    so pass " " + word rather than just word when the continuation begins a new
    word after a non-space context.
    """
    total = 0.0
    n_tokens = 0
    current_ctx = context
    remaining = continuation

    while remaining:
        if not remaining.strip():
            break
        top_tokens = _query_completion(server, current_ctx, n_probs=500)
        if top_tokens is None:
            return float("-inf")
        logp, consumed = _best_prefix_match(top_tokens, remaining)
        if logp == float("-inf") or not consumed:
            return float("-inf")
        total += logp
        current_ctx += consumed
        remaining = remaining[len(consumed):]
        n_tokens += 1

    if n_tokens == 0:
        return float("-inf")
    return total / n_tokens if normalize else total


# ---------------------------------------------------------------------------
# Dataset loaders and evaluators
# ---------------------------------------------------------------------------

def load_arc(subset: str, limit: int | None):
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", subset, split="test", trust_remote_code=False)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def eval_arc(ds, server: str, verbose: bool = False) -> dict:
    letter_map = {"1": "A", "2": "B", "3": "C", "4": "D",
                  "A": "A", "B": "B", "C": "C", "D": "D", "E": "E"}
    correct = 0
    skipped = 0
    for row in ds:
        choices_text = row["choices"]["text"]
        choices_label = row["choices"]["label"]
        answer_letter = letter_map.get(row["answerKey"], row["answerKey"])
        letters = ["A", "B", "C", "D", "E"][:len(choices_text)]
        opt_lines = "\n".join(f"{l}. {t}" for l, t in zip(letters, choices_text))
        context = f"Question: {row['question']}\n{opt_lines}\nAnswer:"
        scores = [(l, first_token_logprob(server, context, " " + l)) for l in letters]
        if all(s == float("-inf") for _, s in scores):
            skipped += 1
            continue
        predicted = max(scores, key=lambda x: x[1])[0]
        if verbose:
            print(f"  Question: {row['question']}")
            for l, t in zip(letters, choices_text):
                print(f"    {l}. {t}")
            print(f"  Expected: {answer_letter}  Predicted: {predicted}  "
                  f"{'CORRECT' if predicted == answer_letter else 'WRONG'}")
        if predicted == answer_letter:
            correct += 1
    n = len(ds) - skipped
    return {"accuracy": round(correct / n * 100, 2) if n else 0.0,
            "correct": correct, "total": n, "skipped": skipped}


def load_winogrande(limit: int | None):
    from datasets import load_dataset
    ds = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=False)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


CHECKPOINT_INTERVAL = 100  # save after every N samples for long-running tasks


def _checkpoint_task(out: Path, task: str, data: dict):
    """Persist partial task results so a crash only loses at most CHECKPOINT_INTERVAL samples."""
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists():
        try:
            with open(out) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing[task] = data
    with open(out, "w") as f:
        json.dump(existing, f, indent=2)


def _load_task_checkpoint(out: Path | None, task: str) -> dict:
    """Return partial checkpoint for task if one exists (has n_processed field), else {}."""
    if not out or not out.exists():
        return {}
    try:
        with open(out) as f:
            ckpt = json.load(f).get(task, {})
        if "n_processed" in ckpt:
            return ckpt
    except Exception:
        pass
    return {}


def eval_winogrande(ds, server: str, out: Path | None = None) -> dict:
    """
    Continuation scoring: score each option as a direct completion of the sentence
    prefix (text before the blank).  Strip the trailing space from the prefix and
    prepend it to the option so the BPE Ġ word-boundary token aligns correctly.
    """
    correct = 0
    skipped = 0
    n_processed = 0

    ckpt = _load_task_checkpoint(out, "winogrande")
    if ckpt:
        correct, skipped, n_processed = ckpt["correct"], ckpt["skipped"], ckpt["n_processed"]
        print(f"  Resuming winogrande: {n_processed}/{len(ds)} samples done.", flush=True)

    for row in ds.select(range(n_processed, len(ds))):
        prefix = row["sentence"].split("_")[0].rstrip()
        s1 = continuation_logprob(server, prefix, " " + row["option1"])
        s2 = continuation_logprob(server, prefix, " " + row["option2"])
        if s1 == float("-inf") and s2 == float("-inf"):
            skipped += 1
        else:
            pred = "1" if s1 >= s2 else "2"
            if pred == row["answer"]:
                correct += 1
        n_processed += 1
        if out and n_processed % CHECKPOINT_INTERVAL == 0:
            n = n_processed - skipped
            _checkpoint_task(out, "winogrande", {
                "accuracy": round(correct / n * 100, 2) if n else 0.0,
                "correct": correct, "total": n, "skipped": skipped,
                "n_processed": n_processed,
            })

    n = len(ds) - skipped
    return {"accuracy": round(correct / n * 100, 2) if n else 0.0,
            "correct": correct, "total": n, "skipped": skipped}


def load_hellaswag(limit: int | None):
    from datasets import load_dataset
    ds = load_dataset("hellaswag", split="validation", trust_remote_code=False)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def prefetch_all_datasets(tasks: list[str], limit: int | None, max_subjects: int | None = None):
    """Download all required datasets to the HuggingFace cache before the server starts."""
    from datasets import load_dataset
    print("Pre-fetching datasets...", flush=True)
    for task in tasks:
        if task in ("arc_easy", "arc_challenge"):
            subset = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
            print(f"  ai2_arc/{subset}...", end=" ", flush=True)
            load_arc(subset, limit)
            print("ok", flush=True)
        elif task == "winogrande":
            print(f"  winogrande_xl (validation)...", end=" ", flush=True)
            load_winogrande(limit)
            print("ok", flush=True)
        elif task == "hellaswag":
            print(f"  hellaswag (validation)...", end=" ", flush=True)
            load_hellaswag(limit)
            print("ok", flush=True)
        elif task == "mmlu":
            subjects = MMLU_SUBJECTS[:max_subjects] if max_subjects else MMLU_SUBJECTS
            print(f"  mmlu ({len(subjects)} subjects)...", flush=True)
            for subj in subjects:
                print(f"    {subj}...", end=" ", flush=True)
                try:
                    load_dataset("cais/mmlu", subj, split="test", trust_remote_code=False)
                    load_dataset("cais/mmlu", subj, split="dev", trust_remote_code=False)
                    print("ok", flush=True)
                except Exception as e:
                    print(f"ERROR: {e}", flush=True)
    print("Datasets ready.", flush=True)


def eval_hellaswag(ds, server: str, out: Path | None = None, verbose: bool = False) -> dict:
    """
    Continuation scoring with length normalization: score each ending as a
    completion of ctx, normalized by token count to remove length bias between
    endings of unequal length.
    """
    correct = 0
    skipped = 0
    n_processed = 0

    ckpt = _load_task_checkpoint(out, "hellaswag")
    if ckpt:
        correct, skipped, n_processed = ckpt["correct"], ckpt["skipped"], ckpt["n_processed"]
        print(f"  Resuming hellaswag: {n_processed}/{len(ds)} samples done.", flush=True)

    for row in ds.select(range(n_processed, len(ds))):
        context = row["ctx"]
        scores = [continuation_logprob(server, context, " " + e, normalize=True)
                  for e in row["endings"]]
        if all(s == float("-inf") for s in scores):
            skipped += 1
        else:
            label = int(row["label"])
            pred_idx = scores.index(max(scores))
            if verbose:
                ctx_short = context[:120] + "..." if len(context) > 120 else context
                print(f"  Context: {ctx_short}")
                for i, e in enumerate(row["endings"]):
                    marker = "  ← expected" if i == label else ""
                    print(f"    {i}: {e}{marker}")
                print(f"  Predicted: {pred_idx}  "
                      f"{'CORRECT' if pred_idx == label else 'WRONG'}")
            if pred_idx == label:
                correct += 1
        n_processed += 1
        if out and n_processed % CHECKPOINT_INTERVAL == 0:
            n = n_processed - skipped
            _checkpoint_task(out, "hellaswag", {
                "accuracy": round(correct / n * 100, 2) if n else 0.0,
                "correct": correct, "total": n, "skipped": skipped,
                "n_processed": n_processed,
            })

    n = len(ds) - skipped
    return {"accuracy": round(correct / n * 100, 2) if n else 0.0,
            "correct": correct, "total": n, "skipped": skipped}


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
        opts = row["choices"]
        ans = ANSWER_CHOICES[row["answer"]]
        opt_str = "\n".join(f"{l}. {t}" for l, t in zip(ANSWER_CHOICES, opts))
        shots.append(f"Question: {row['question']}\n{opt_str}\nAnswer: {ans}")
    return header + "\n\n".join(shots) + "\n\n"


def eval_mmlu_subject(subject: str, server: str, num_fewshot: int, limit: int | None,
                      verbose: bool = False) -> dict:
    from datasets import load_dataset
    test_ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=False)
    if limit:
        test_ds = test_ds.select(range(min(limit, len(test_ds))))
    fewshot_prefix = ""
    if num_fewshot > 0:
        try:
            dev_ds = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=False)
            fewshot_prefix = make_mmlu_fewshot_prompt(list(dev_ds)[:num_fewshot], subject)
        except Exception:
            pass
    correct = 0
    skipped = 0
    for row in test_ds:
        opts = row["choices"]
        opt_str = "\n".join(f"{l}. {t}" for l, t in zip(ANSWER_CHOICES, opts))
        context = fewshot_prefix + f"Question: {row['question']}\n{opt_str}\nAnswer:"
        scores = [first_token_logprob(server, context, " " + ch) for ch in ANSWER_CHOICES]
        if all(s == float("-inf") for s in scores):
            skipped += 1
            continue
        pred_idx = scores.index(max(scores))
        if verbose:
            print(f"  Question: {row['question']}")
            for l, t in zip(ANSWER_CHOICES, opts):
                print(f"    {l}. {t}")
            expected = ANSWER_CHOICES[row["answer"]]
            predicted = ANSWER_CHOICES[pred_idx]
            print(f"  Expected: {expected}  Predicted: {predicted}  "
                  f"{'CORRECT' if pred_idx == row['answer'] else 'WRONG'}")
        if pred_idx == row["answer"]:
            correct += 1
    n = len(test_ds) - skipped
    return {"accuracy": round(correct / n * 100, 2) if n else 0.0,
            "correct": correct, "total": n, "skipped": skipped}


def _checkpoint_mmlu(out: Path, subject_results: dict, all_correct: int, all_total: int):
    """Write partial MMLU results to the output file after each subject completes."""
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists():
        try:
            with open(out) as f:
                existing = json.load(f)
        except Exception:
            pass
    overall = (all_correct / all_total * 100) if all_total else 0.0
    existing["mmlu"] = {
        "accuracy": round(overall, 2),
        "correct": all_correct,
        "total": all_total,
        "subjects": subject_results,
    }
    with open(out, "w") as f:
        json.dump(existing, f, indent=2)


def eval_mmlu(server: str, num_fewshot: int, limit: int | None,
              max_subjects: int | None = None, out: Path | None = None,
              verbose: bool = False) -> dict:
    all_correct = 0
    all_total = 0
    subject_results = {}

    # Resume from checkpoint if output file already has partial MMLU results
    if out and out.exists():
        try:
            with open(out) as f:
                prev = json.load(f).get("mmlu", {}).get("subjects", {})
            for subj, r in prev.items():
                subject_results[subj] = r
                all_correct += r["correct"]
                all_total += r["total"]
            if subject_results:
                print(f"  Resuming: {len(subject_results)} subjects already in checkpoint.", flush=True)
        except Exception:
            pass

    subjects = MMLU_SUBJECTS[:max_subjects] if max_subjects else MMLU_SUBJECTS
    for subj in subjects:
        if subj in subject_results:
            print(f"  {subj:45s} {subject_results[subj]['accuracy']:5.1f}% (cached)", flush=True)
            continue
        try:
            r = eval_mmlu_subject(subj, server, num_fewshot, limit, verbose=verbose)
            subject_results[subj] = r
            all_correct += r["correct"]
            all_total += r["total"]
            print(f"  {subj:45s} {r['accuracy']:5.1f}% ({r['correct']}/{r['total']})", flush=True)
            if out:
                _checkpoint_mmlu(out, subject_results, all_correct, all_total)
        except Exception as e:
            print(f"  {subj:45s} ERROR: {e}", flush=True)

    overall = (all_correct / all_total * 100) if all_total else 0.0
    return {"accuracy": round(overall, 2), "correct": all_correct, "total": all_total,
            "subjects": subject_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_server(server: str) -> bool:
    try:
        return requests.get(f"{server}/health", timeout=5).status_code == 200
    except Exception:
        return False


def run_task(task: str, server: str, num_fewshot: int, limit: int | None,
             max_subjects: int | None = None, out: Path | None = None,
             verbose: bool = False) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}\nTask: {task}  limit={limit}  few-shot={num_fewshot}\n{'='*60}")
    if task in ("arc_easy", "arc_challenge"):
        subset = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_arc(subset, limit)
        print(f"Loaded {len(ds)} samples")
        result = eval_arc(ds, server, verbose=verbose)
    elif task == "winogrande":
        ds = load_winogrande(limit)
        print(f"Loaded {len(ds)} samples")
        result = eval_winogrande(ds, server, out=out)
    elif task == "hellaswag":
        ds = load_hellaswag(limit)
        print(f"Loaded {len(ds)} samples")
        result = eval_hellaswag(ds, server, out=out, verbose=verbose)
    elif task == "mmlu":
        result = eval_mmlu(server, num_fewshot, limit, max_subjects=max_subjects, out=out,
                           verbose=verbose)
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
    parser = argparse.ArgumentParser(description="Evaluate a llama.cpp model on multiple-choice benchmarks")
    parser.add_argument("--task", choices=TASKS + ["all"], default="arc_easy")
    parser.add_argument("--num-fewshot", type=int, default=0,
                        help="Few-shot count (use 5 for MMLU)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per task (None = full dataset)")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Max MMLU subjects to evaluate (default: all 57)")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verbose", action="store_true",
                        help="Print prompt and expected/predicted answer for each sample")
    parser.add_argument("--start-server", action="store_true",
                        help="Start and manage llama-server automatically")
    parser.add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    port = int(args.server.split(":")[-1])
    tasks = TASKS if args.task == "all" else [args.task]

    prefetch_all_datasets(tasks, args.limit, max_subjects=args.max_subjects)

    if args.start_server:
        global _server_args
        _server_args = dict(llama_dir=args.llama_dir, model=args.model,
                            threads=args.threads, port=port)
        print(f"Starting llama-server on port {port}...")
        if not _start_server(**_server_args):
            print("ERROR: Server failed to start.")
            return
        print("Server ready.")
    elif not check_server(args.server):
        print(f"ERROR: No server at {args.server}. Pass --start-server or start it manually.")
        return

    all_results = {}
    for task in tasks:
        fewshot = args.num_fewshot if task == "mmlu" else 0
        all_results[task] = run_task(task, args.server, fewshot, args.limit,
                                     max_subjects=args.max_subjects if task == "mmlu" else None,
                                     out=args.out, verbose=args.verbose)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.out.exists():
        with open(args.out) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(args.out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {args.out}")

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
