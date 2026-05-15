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
import contextlib
import io
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "accuracy_results_bitnet.json"
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

# Published paper targets from arXiv:2504.12285 Table 1.
# BitNet entries are the b1.58 2B4T numbers; Qwen entries are the Qwen2.5 1.5B
# FP16 numbers (compared against our Q8_0 measurement, which typically matches
# FP16 within ~0.5pt on these benchmarks).
PAPER_TARGETS_BY_MODEL = {
    "bitnet": {
        "arc_easy":      74.79,
        "arc_challenge": 49.91,
        "winogrande":    71.90,
        "hellaswag":     68.44,
        "mmlu":          53.17,
    },
    "qwen": {
        "arc_easy":      79.92,
        "arc_challenge": 52.82,
        "winogrande":    66.61,
        "hellaswag":     70.95,
        "mmlu":          61.11,
    },
}

PAPER_LABEL_BY_MODEL = {
    "bitnet": "BitNet b1.58 2B4T (paper i2_s, arXiv:2504.12285 Table 1)",
    "qwen":   "Qwen2.5 1.5B (paper FP16, arXiv:2504.12285 Table 1)",
}


def resolve_paper_targets(choice: str, model_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """
    Pick which paper baselines to compare against.

    Returns ({model_key: targets_dict}, {model_key: human_label}).

    choice="both" (default) — compare against both BitNet i2_s and Qwen FP16.
    choice="auto"           — same as "both"; the model_path argument is kept
                              for backward compat but no longer disambiguates.
    choice="bitnet"|"qwen"  — single baseline only.
    choice="none"           — empty dicts; no paper comparison rendered.
    """
    _ = model_path  # retained for backward compat; selection is no longer path-based
    if choice == "none":
        return {}, {}
    keys = ["bitnet", "qwen"] if choice in ("auto", "both") else [choice]
    return (
        {k: PAPER_TARGETS_BY_MODEL[k] for k in keys},
        {k: PAPER_LABEL_BY_MODEL[k] for k in keys},
    )


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

import math

# Per-server capability cache populated lazily by _server_caps().
_SERVER_CAPS: dict[str, dict] = {}

# Bias size used to force a target token. Large enough to dominate softmax
# across vocab sizes up to ~256k.
_BIAS_VALUE = 100.0

# n_probs for fallback path (older forks return post-bias probs under logit_bias,
# so we have to search top-K instead).  5000 covers most continuation tokens; the
# BitNet fork crashes at much larger values.
_FALLBACK_N_PROBS = 5000


def _decode_tok_str(tok_str: str) -> str:
    return tok_str.replace("Ġ", " ").replace("▁", " ")


def _tokenize(server: str, text: str, add_special: bool = True) -> list[int]:
    """Return token IDs for `text` using the server's tokenizer."""
    try:
        r = requests.post(f"{server}/tokenize",
                          json={"content": text, "add_special": add_special},
                          timeout=30)
        return r.json().get("tokens", [])
    except Exception:
        return []


def _detokenize(server: str, tokens: list[int]) -> str:
    """Return the text decoded from `tokens`."""
    try:
        r = requests.post(f"{server}/detokenize",
                          json={"tokens": tokens},
                          timeout=30)
        return r.json().get("content", "")
    except Exception:
        return ""


def _server_caps(server: str) -> dict:
    """
    Probe the server once to learn how it handles `logit_bias`.

    Modern llama.cpp returns the *natural* (pre-bias) logprob of the forced
    token when `post_sampling_probs:false` is set — this lets us read
    log P(target | context) exactly, for any target.

    Older forks (e.g. the BitNet llama.cpp fork) apply the bias before
    reporting probs, so the forced token always comes back at prob≈1.0; we
    must fall back to top-K search instead.

    Detection: force " London" after "The capital of France is".  A natural
    response gives logprob ≈ -7; a post-bias response gives ≈ 0.

    Returns {"bias_natural": bool, "fmt": "new"|"old"}.
    """
    if server in _SERVER_CAPS:
        return _SERVER_CAPS[server]
    caps = {"bias_natural": False, "fmt": "old"}
    try:
        toks = _tokenize(server, " London", add_special=False)
        if not toks:
            _SERVER_CAPS[server] = caps
            return caps
        target_id = toks[0]
        r = requests.post(f"{server}/completion", json={
            "prompt": "The capital of France is",
            "n_predict": 1,
            "n_probs": 1,
            "temperature": 0.0,
            "post_sampling_probs": False,
            "logit_bias": [[target_id, _BIAS_VALUE]],
        }, timeout=30)
        data = r.json()
        cp = data.get("completion_probabilities", [])
        if cp:
            item = cp[0]
            if "logprob" in item or "top_logprobs" in item:
                caps["fmt"] = "new"
                lp = item.get("logprob", 0.0)
                if lp < -1.0:
                    caps["bias_natural"] = True
            elif "probs" in item:
                caps["fmt"] = "old"
                probs = item["probs"]
                if probs and probs[0].get("prob", 1.0) < 0.5:
                    caps["bias_natural"] = True
    except Exception:
        pass
    _SERVER_CAPS[server] = caps
    return caps


def _post_completion(
    server: str,
    prompt: list[int] | str,
    n_probs: int,
    target_id: int | None = None,
    cache_prompt: bool = True,
) -> dict | None:
    """POST one /completion request; return completion_probabilities[0] or None."""
    body = {
        "prompt": prompt,
        "n_predict": 1,
        "n_probs": n_probs,
        "temperature": 0.0,
        "cache_prompt": cache_prompt,
        "post_sampling_probs": False,
    }
    if target_id is not None:
        body["logit_bias"] = [[target_id, _BIAS_VALUE]]
    for attempt in range(3):
        try:
            r = requests.post(f"{server}/completion", json=body, timeout=120)
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
        cp = data.get("completion_probabilities", [])
        return cp[0] if cp else None
    return None


def _find_token_logprob(item: dict, target_id: int, target_str: str) -> float | None:
    """Locate `target_id`/`target_str` in the response's top-K list; None if absent."""
    if "top_logprobs" in item:
        for p in item["top_logprobs"]:
            if p.get("id") == target_id:
                return p.get("logprob")
        return None
    if "probs" in item:
        for p in item["probs"]:
            if p.get("tok_str") == target_str:
                prob = p.get("prob", 0.0)
                return math.log(prob) if prob > 0 else None
        return None
    return None


def _min_logprob(item: dict) -> float:
    """Smallest logprob in the response's top-K, used as a conservative bound."""
    if "top_logprobs" in item:
        lps = [p.get("logprob", 0.0) for p in item["top_logprobs"]]
        return min(lps) if lps else 0.0
    if "probs" in item:
        lps = [math.log(p["prob"]) for p in item["probs"]
               if 0 < p.get("prob", 0) < 1]
        return min(lps) if lps else 0.0
    return 0.0


def _best_prefix_match(top_tokens: list, target: str) -> tuple[float, str]:
    """
    Find the token in top_tokens whose decoded form is the longest prefix of
    target.  Returns (log_prob, matched_text) or (float('-inf'), '') if none match.
    Used only by first_token_logprob (ARC/MMLU), where the target is a single
    label character and reliably appears in top-K.
    """
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
    item = _post_completion(server, context, n_probs=n_probs)
    if not item:
        return float("-inf")
    top = item.get("top_logprobs") or item.get("probs", [])
    if not top:
        return float("-inf")
    logp, _ = _best_prefix_match(top, choice)
    return logp


def continuation_logprob(
    server: str, context: str, continuation: str, normalize: bool = False
) -> float:
    """
    Score Σ log P(t_i | ctx, t_0..t_i-1) over the tokens of `continuation` in
    token space.  Used by WinoGrande and HellaSwag.

    Strategy depends on the server (detected once via _server_caps):

      * Modern llama.cpp: force each target token with logit_bias and
        post_sampling_probs:false, then read its *natural* logprob from the
        response.  Exact for every token regardless of rarity.

      * Older forks (BitNet's): logit_bias is applied before the reported
        probs, so forcing is unusable.  Fall back to top-K search with
        n_probs=5000; for tokens rarer than top-K, use the smallest seen
        logprob minus a small penalty (a conservative lower bound that keeps
        comparisons across candidate continuations meaningful).

    If normalize=True, returns mean per-token logprob (length-normalized;
    use for HellaSwag).
    """
    ctx_tokens = _tokenize(server, context, add_special=True)
    full_tokens = _tokenize(server, context + continuation, add_special=True)
    if not ctx_tokens or not full_tokens:
        return float("-inf")

    # Align boundary: full must start with ctx.  If BPE merged across the
    # boundary, walk ctx back to the longest common prefix.
    if full_tokens[:len(ctx_tokens)] != ctx_tokens:
        n = 0
        for a, b in zip(ctx_tokens, full_tokens):
            if a != b:
                break
            n += 1
        ctx_tokens = full_tokens[:n]

    cont_tokens = full_tokens[len(ctx_tokens):]
    if not cont_tokens:
        return float("-inf")

    caps = _server_caps(server)
    use_bias = caps["bias_natural"]

    total = 0.0
    prompt_tokens = list(ctx_tokens)

    for target_id in cont_tokens:
        if use_bias:
            item = _post_completion(server, prompt_tokens, n_probs=1, target_id=target_id)
            if not item:
                return float("-inf")
            lp = item.get("logprob")
            if lp is None:
                probs = item.get("probs", [])
                if probs:
                    prob = probs[0].get("prob", 0.0)
                    lp = math.log(prob) if prob > 0 else None
            if lp is None:
                return float("-inf")
        else:
            item = _post_completion(server, prompt_tokens, n_probs=_FALLBACK_N_PROBS)
            if not item:
                return float("-inf")
            target_str = _detokenize(server, [target_id])
            lp = _find_token_logprob(item, target_id, target_str)
            if lp is None:
                # Rarer than top-K: use min seen logprob as a conservative bound.
                lp = _min_logprob(item) - 1.0
        total += lp
        prompt_tokens.append(target_id)

    return total / len(cont_tokens) if normalize else total


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
    """
    Length-normalized loglikelihood scoring (lm-eval-harness acc_norm).

    Prompt:    "Question: {question}\nAnswer:"
    Candidates: each full choice text, prefixed with a space
    Pick:      candidate with highest mean per-token logprob.

    The earlier first-token letter scoring listed all choices in the
    prompt and asked which letter came next; that lets the model use
    in-prompt cross-option comparison and rewards format familiarity
    rather than content understanding, inflating ARC-Challenge ~19pt
    above the paper.  This methodology matches what the paper reports.
    """
    correct = 0
    skipped = 0
    for row in ds:
        choices_text = row["choices"]["text"]
        choices_label = row["choices"]["label"]
        try:
            answer_idx = choices_label.index(row["answerKey"])
        except ValueError:
            skipped += 1
            continue
        context = f"Question: {row['question']}\nAnswer:"
        scores = [continuation_logprob(server, context, " " + ct, normalize=True)
                  for ct in choices_text]
        if all(s == float("-inf") for s in scores):
            skipped += 1
            continue
        pred_idx = scores.index(max(scores))
        if verbose:
            print(f"  Question: {row['question']}")
            for i, t in enumerate(choices_text):
                marker = "  <- expected" if i == answer_idx else ""
                print(f"    [{i}] {t}{marker}")
            print(f"  Predicted: [{pred_idx}]  "
                  f"{'CORRECT' if pred_idx == answer_idx else 'WRONG'}")
        if pred_idx == answer_idx:
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
    Partial-context scoring (lm-eval-harness methodology):

      Replace "_" with each option, then score log P(suffix | prefix + option),
      where suffix is the part of the sentence AFTER the blank.  The model
      uses the surrounding context to judge which fill-in produces a coherent
      sentence — this is what WinoGrande actually tests.

      The earlier P(option | prefix) approach ignores everything after the
      blank, which is where the disambiguating signal lives, and scores near
      random.
    """
    correct = 0
    skipped = 0
    n_processed = 0

    ckpt = _load_task_checkpoint(out, "winogrande")
    if ckpt:
        correct, skipped, n_processed = ckpt["correct"], ckpt["skipped"], ckpt["n_processed"]
        print(f"  Resuming winogrande: {n_processed}/{len(ds)} samples done.", flush=True)

    for row in ds.select(range(n_processed, len(ds))):
        sentence = row["sentence"]
        blank = sentence.index("_")
        prefix = sentence[:blank]
        suffix = " " + sentence[blank + 1:].strip()
        s1 = continuation_logprob(server, prefix + row["option1"], suffix)
        s2 = continuation_logprob(server, prefix + row["option2"], suffix)
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


def _hellaswag_preprocess(text: str) -> str:
    """
    Standard HellaSwag text cleanup (lm-eval-harness): turn wikiHow "[title]"
    markers into sentence breaks and strip remaining bracketed annotations.
    """
    import re as _re
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = _re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def eval_hellaswag(ds, server: str, out: Path | None = None, verbose: bool = False) -> dict:
    """
    Length-normalized continuation scoring with lm-eval-harness preprocessing.

    Context = activity_label + ": " + ctx_a + " " + Ctx_b (with [title]/[xxx]
    cleanup).  Without the activity_label prefix the model loses the framing
    that tells it what scenario the endings are continuing.

    Score each cleaned ending as log P(ending | context) divided by ending
    token count, so endings of unequal length compete on per-token coherence.
    """
    correct = 0
    skipped = 0
    n_processed = 0

    ckpt = _load_task_checkpoint(out, "hellaswag")
    if ckpt:
        correct, skipped, n_processed = ckpt["correct"], ckpt["skipped"], ckpt["n_processed"]
        print(f"  Resuming hellaswag: {n_processed}/{len(ds)} samples done.", flush=True)

    for row in ds.select(range(n_processed, len(ds))):
        context = _hellaswag_preprocess(
            row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize()
        )
        scores = [continuation_logprob(server, context, " " + _hellaswag_preprocess(e),
                                       normalize=True)
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
    """
    Build the 5-shot prompt prefix in lm-eval-harness MMLU format:

        The following are multiple choice questions (with answers) about {subject}.

        {q1}
        A. ...
        B. ...
        C. ...
        D. ...
        Answer: {a1}

        ...

    Note: question is rendered as raw text (no "Question:" prefix), matching
    lm-eval-harness's mmlu.doc_to_text.
    """
    pretty = subject_name.replace("_", " ")
    header = f"The following are multiple choice questions (with answers) about {pretty}.\n\n"
    shots = []
    for row in dev_rows:
        opts = row["choices"]
        ans = ANSWER_CHOICES[row["answer"]]
        opt_str = "\n".join(f"{l}. {t}" for l, t in zip(ANSWER_CHOICES, opts))
        shots.append(f"{row['question'].strip()}\n{opt_str}\nAnswer: {ans}")
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
        context = fewshot_prefix + f"{row['question'].strip()}\n{opt_str}\nAnswer:"
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


def _clear_stale_codecarbon_lock() -> None:
    """
    Remove any leftover %TEMP%/.codecarbon.lock from an abnormally terminated
    prior run.  Without this, codecarbon refuses to start (treating the stale
    lock as a concurrent instance) and silently returns None from stop(),
    zeroing the energy_kwh/co2_kg fields without surfacing an error.
    """
    lock = Path(tempfile.gettempdir()) / ".codecarbon.lock"
    if lock.exists():
        try:
            lock.unlink()
        except OSError:
            pass


def _make_emissions_tracker(out_dir: Path):
    """
    Build a CodeCarbon EmissionsTracker, or return None if codecarbon isn't
    installed.  Mirrors metrics_tracker.py — silences codecarbon's chatty
    logging and disables on-disk emissions.csv (we store kWh/CO2 inline in
    the per-task accuracy JSON instead).
    """
    _clear_stale_codecarbon_lock()
    try:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=FutureWarning)
            from codecarbon import EmissionsTracker
        import logging as _logging
        _logging.getLogger("codecarbon").setLevel(_logging.CRITICAL)
        return EmissionsTracker(
            project_name="llama-eval-accuracy",
            output_dir=str(out_dir),
            log_level="error",
            save_to_file=False,
        )
    except ImportError:
        return None


def run_task(task: str, server: str, num_fewshot: int, limit: int | None,
             max_subjects: int | None = None, out: Path | None = None,
             verbose: bool = False, track_energy: bool = True,
             paper_targets: dict | None = None) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}\nTask: {task}  limit={limit}  few-shot={num_fewshot}\n{'='*60}")

    tracker = None
    if track_energy:
        tracker = _make_emissions_tracker(out.parent if out else REPO_ROOT / "results")
        if tracker is None:
            print("  (codecarbon not installed — skipping energy tracking)", flush=True)
    if tracker:
        tracker.start()

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

    energy_kwh = co2_kg = None
    if tracker:
        # codecarbon 2.7 has a Windows lock-file bug that spams stderr on stop().
        with contextlib.redirect_stderr(io.StringIO()):
            emissions = tracker.stop()
        co2_kg = float(emissions) if emissions is not None else None
        try:
            energy_kwh = tracker.final_emissions_data.energy_consumed
        except Exception:
            energy_kwh = None

    elapsed = time.time() - t0
    result["elapsed_s"] = round(elapsed, 1)
    # energy_kwh / co2_kg cover this invocation only — checkpoint-resumed runs
    # do not retain energy from the prior run, mirroring how elapsed_s behaves.
    if energy_kwh is not None:
        result["energy_kwh"] = round(energy_kwh, 6)
    if co2_kg is not None:
        result["co2_kg"] = round(co2_kg, 6)
    # paper_targets is now {model_key: {task: float, ...}, ...}.  Record every
    # available target for this task and emit one gap line per baseline so the
    # user sees how the run compares against both BitNet i2_s and Qwen FP16.
    per_baseline = {}
    for model_key, ts in (paper_targets or {}).items():
        t = ts.get(task)
        if t is not None:
            per_baseline[model_key] = t
    if per_baseline:
        result["paper_targets"] = per_baseline
    print(f"\nResult: {result['accuracy']}%  ({result['correct']}/{result['total']})")
    for model_key, t in per_baseline.items():
        gap = result["accuracy"] - t
        label = PAPER_LABEL_BY_MODEL.get(model_key, model_key)
        print(f"  vs {label}: {t}%  gap: {gap:+.1f}%")
    print(f"Time:   {elapsed:.0f}s", end="")
    if energy_kwh is not None:
        print(f"   Energy: {energy_kwh*1000:.2f} Wh   CO2: {(co2_kg or 0)*1000:.2f} g", end="")
    print()
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
    parser.add_argument("--no-energy", action="store_true",
                        help="Skip CodeCarbon energy tracking (faster, no internet needed)")
    parser.add_argument("--paper-targets",
                        choices=["both", "auto", "bitnet", "qwen", "none"],
                        default="both",
                        help="Which paper baselines to compare against "
                             "(default: both — show BitNet i2_s and Qwen FP16 "
                             "gaps for every task)")
    args = parser.parse_args()

    paper_targets, paper_labels = resolve_paper_targets(args.paper_targets, args.model)

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
                                     out=args.out, verbose=args.verbose,
                                     track_energy=not args.no_energy,
                                     paper_targets=paper_targets)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.out.exists():
        with open(args.out) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(args.out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {args.out}")

    if paper_labels:
        # Print which baselines are being compared against, then a single table
        # with one (target, gap) pair per baseline.  Hardcoded keys keep the
        # column layout stable; new baselines would need a column added here.
        keys = list(paper_labels.keys())
        print("\nPaper baselines (arXiv:2504.12285 Table 1):")
        for k in keys:
            print(f"  {k:<6} = {paper_labels[k]}")
        col_titles = {"bitnet": "BitNet", "qwen": "Qwen"}
        header = f"\n{'Task':<20} {'Accuracy':>9}"
        for k in keys:
            header += f"   {col_titles.get(k, k.title()):>8} {'Δ':>7}"
        header += f"   {'N':>5}"
        print(header)
        print("-" * len(header.lstrip("\n")))
        for task, r in all_results.items():
            row = f"{task:<20} {r['accuracy']:>8.2f}%"
            targets = r.get("paper_targets", {})
            for k in keys:
                t = targets.get(k)
                if isinstance(t, (int, float)):
                    gap = r["accuracy"] - t
                    row += f"   {t:>7.2f}% {gap:>+6.1f}%"
                else:
                    row += f"   {'-':>8} {'-':>7}"
            row += f"   {r['total']:>5}"
            print(row)
    else:
        # --paper-targets none: no comparison shown.
        print(f"\n{'Task':<20} {'Accuracy':>10} {'N':>6}")
        print("-" * 38)
        for task, r in all_results.items():
            print(f"{task:<20} {r['accuracy']:>9.2f}% {r['total']:>6}")

    if _server_proc is not None:
        _server_proc.terminate()
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
