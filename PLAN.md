# PLAN.md

## Project Overview

**Title:** Non-GPU LLM Inference: Benchmarking BitNet b1.58 2B4T and Qwen2.5-1.5B on CPU
**Author:** Sean Michael
**Date:** May 2026

An independent reproduction and extension of Microsoft's published inference
benchmarks for BitNet b1.58 2B4T ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285)).
Rather than training from scratch, this project runs the pre-trained BitNet b1.58
2B4T GGUF model locally via `bitnet.cpp` and benchmarks inference latency,
throughput, memory footprint, and energy consumption on commodity CPU hardware.
Qwen2.5-1.5B-Instruct Q8_0 is benchmarked alongside BitNet as a practical
FP16-comparable baseline using upstream `llama.cpp`. Accuracy is compared against
Microsoft's published FP16 baselines — LLaMA 3.2 1B, Gemma-3 1B, Qwen2.5 1.5B,
SmolLM2 1.7B, and MiniCPM 2B — to validate whether 1-bit quantization delivers
meaningful real-world efficiency gains without significant accuracy loss.

### Objectives

- Run BitNet b1.58 2B4T locally via `bitnet.cpp` on CPU
- Run Qwen2.5-1.5B-Instruct Q8_0 via upstream `llama.cpp` as a CPU baseline
- Benchmark inference latency, throughput, memory, and energy for both models
- Evaluate accuracy on ARC-Easy, ARC-Challenge, WinoGrande, HellaSwag, MMLU (5-shot)
- Compare measured numbers against the paper's published values (arXiv:2504.12285)
- Compare accuracy against the five FP16 baselines reported in the paper
- Produce a cost–accuracy trade-off analysis with a carbon-footprint proxy

---

## Dependencies

| Dependency | Version / Source | Location |
|---|---|---|
| Python | 3.11 (numpy 1.26 has no wheels for 3.13+) | local |
| Poetry | any recent (>=1.6) | local |
| CMake | >= 3.22 | local |
| Visual Studio | 2022+, "Desktop development with C++" | local |
| ClangCL (BitNet only) | VS components: `Microsoft.VisualStudio.Component.VC.Llvm.Clang` and `...ClangToolset` | local |
| Hugging Face CLI | `hf` (in `huggingface_hub[cli]`) | local |
| `microsoft/BitNet` | commit `01eb415772c342d9f20dc42772f1583ae1e5b102` | `../Models/BitNet` (sibling) |
| `ggml-org/llama.cpp` | commit `1e5ad35d560b90a8ac447d149c8f8447ae1fcaa0` | `../Models/Qwen/llama.cpp` (sibling) |
| BitNet GGUF | `microsoft/BitNet-b1.58-2B-4T-gguf` (`ggml-model-i2_s.gguf`, 1.71 GiB) | `../Models/BitNet/models/BitNet-b1.58-2B-4T/` |
| Qwen Q8_0 GGUF | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (`qwen2.5-1.5b-instruct-q8_0.gguf`, 1.65 GiB) | `../Models/Qwen/` |
| Qwen Q4_K_M GGUF | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (`qwen2.5-1.5b-instruct-q4_k_m.gguf`, ~1.0 GiB) | `../Models/Qwen/` |

Python packages (managed by Poetry, see `pyproject.toml`): `pandas`,
`matplotlib`, `numpy`, `psutil`, `codecarbon`, `requests`, `datasets`, `rich`.

Reference hardware (REPORT.md): Intel Core i5-9400F @ 2.90 GHz (6 cores, 4
threads used), 16 GB RAM, Windows 11.

### Build Patches (BitNet, applied automatically by `make bitnet-patch`)

Three patches in `patches/` are required at the pinned BitNet commit when
building with ClangCL 18+ on Windows:

- `bitnet-clangcl-const.patch` — adds `const` to a non-const pointer init in
  `src/ggml-bitnet-mad.cpp:811`; ClangCL 20+ treats this as a hard error.
- `llama-chrono.patch` — adds missing `#include <chrono>` in
  `3rdparty/llama.cpp/common/{common,log}.cpp`.
- `llama-chrono-examples.patch` — same `<chrono>` fix in
  `examples/{imatrix,perplexity}/*.cpp`.

---

## Repository Layout

```
CS495-Non-GPU/
├── Makefile                       # All reproducibility entry points
├── pyproject.toml                 # Poetry dependency lock
├── README.md                      # User-facing quickstart
├── PLAN.md                        # This document
├── BITNET_SUMMARY.md              # Model card + quantization notes (BitNet)
├── QWEN_SUMMARY.md                # Model card + inference-stack notes (Qwen)
├── REPORT.md                      # Phase 3 benchmarking report (BitNet headline numbers)
├── FINAL_REPORT.md                # Phase 4 dashboard: BitNet vs Qwen + paper FP16 baselines
│
├── patches/                       # Build patches for the pinned BitNet commit
│   ├── bitnet-clangcl-const.patch
│   ├── llama-chrono.patch
│   └── llama-chrono-examples.patch
│
├── scripts/
│   ├── metrics_tracker.py         # llama-bench wrapper → step_metrics CSV (latency/throughput/RSS/energy)
│   ├── eval_accuracy.py           # llama-server-driven multiple-choice eval → accuracy_results JSON
│   ├── compare_runs.py            # Generates comparison_table.csv + all plots in results/plots/
│   └── smoke_test.py              # End-to-end smoke test for all of the above
│
└── results/
    ├── bitnet_step_metrics.csv     # BitNet llama-bench results
    ├── qwen_step_metrics.csv       # Qwen Q8_0 llama-bench results
    ├── qwen_q4_step_metrics.csv    # Qwen Q4_K_M llama-bench results
    ├── accuracy_results_bitnet.json
    ├── accuracy_results_qwen.json
    ├── accuracy_results_qwen_q4.json
    ├── comparison_table.csv        # Aggregated paper+ours summary
    └── plots/                      # PNGs generated by compare_runs.py
```

External (not in this repo): `../Models/BitNet/` and `../Models/Qwen/`.

---

## Reproducibility — End-to-End

All commands run from the repo root.  The Makefile is the single source of
truth; nothing here is "do it this way in a shell instead."

### 1. Environment

```bash
make install          # Poetry: install runtime deps (--only main)
# or
make install-dev      # Poetry: install runtime + dev (ruff)

make check-deps       # Verify cmake, Python 3.11, git, and ClangCL
```

### 2. Build the two inference stacks

```bash
make bitnet-setup     # Clone microsoft/BitNet at pinned commit, patch, build (ClangCL)
make bitnet-model     # Download ggml-model-i2_s.gguf (~1.7 GiB)
make bitnet-verify    # Sanity check: 32-token completion of "What is 2+2?"

make qwen-setup       # Clone ggml-org/llama.cpp at pinned commit, build (MSVC), download Q8_0 GGUF
make qwen-verify      # Sanity check: same prompt, Qwen Q8_0

make qwen-q4-model    # Download Qwen2.5-1.5B Q4_K_M GGUF (~1 GB, reuses qwen-setup build)
make qwen-q4-verify   # Sanity check: same prompt, Qwen Q4_K_M
```

Override sibling-dir locations if needed: `make bitnet-setup BITNET_DIR=...`,
`make qwen-setup QWEN_DIR=...`.

### 3. Smoke test

```bash
make smoke-test               # Both models
make smoke-test-bitnet        # BitNet only
make smoke-test-qwen          # Qwen only
```

`scripts/smoke_test.py` exercises the full pipeline end-to-end: runs three
inference prompts per model, verifies `compare_runs.py` produces the expected
plot files and CSV, runs `--help` on the other scripts, and then runs a small
(5-sample) accuracy sweep per model via `llama-server`.  Exit 0 = all checks
pass.

### 4. Benchmarks

```bash
make benchmark-bitnet         # → results/bitnet_step_metrics.csv
make benchmark-qwen           # → results/qwen_step_metrics.csv      (Qwen Q8_0)
make benchmark-qwen-q4        # → results/qwen_q4_step_metrics.csv   (Qwen Q4_K_M)
make benchmark                # All three
```

Each benchmark runs `llama-bench` over the three `(n_prompt, n_gen)` configs
defined in `scripts/metrics_tracker.py` (`(512, 128)`, `(512, 512)`, `(1, 512)`)
matching the paper's Table 1 conditions, captures latency / throughput / peak
RSS, and tracks energy + CO₂ via CodeCarbon.

### 5. Accuracy evaluation

Per-task:

```bash
make eval-arc-easy-bitnet      eval-arc-easy-qwen      eval-arc-easy
make eval-arc-challenge-bitnet eval-arc-challenge-qwen eval-arc-challenge
make eval-mmlu-bitnet          eval-mmlu-qwen          eval-mmlu      # 5-shot
make eval-winogrande-bitnet    eval-winogrande-qwen    eval-winogrande
make eval-hellaswag-bitnet     eval-hellaswag-qwen     eval-hellaswag
```

All tasks:

```bash
make eval-accuracy-bitnet     # All 5 tasks, BitNet only
make eval-accuracy-qwen       # All 5 tasks, Qwen Q8_0 only
make eval-accuracy-qwen-q4    # All 5 tasks, Qwen Q4_K_M only
make eval-accuracy            # All three models, all 5 tasks
```

Each target uses `--start-server` so `eval_accuracy.py` brings up
`llama-server`, runs the eval, and shuts the server down.  Override
`LIMIT=N` to sample `N` items per task (default 500); use `LIMIT=0` for the
full split.

### 6. Plots and comparison table

```bash
make plots          # → results/comparison_table.csv + all PNGs in results/plots/
```

### 7. Cleanup

```bash
make clean          # Remove results/plots/ and cached .pyc
make nuke           # clean + remove Poetry virtualenv
make bitnet-clean   # Remove ../Models/BitNet
make qwen-clean     # Remove ../Models/Qwen
```

---

## Implementation

### `scripts/metrics_tracker.py`

llama-bench wrapper.  For each `(n_prompt, n_gen)` config in `BENCH_CONFIGS`:

1. Clears any stale `.codecarbon.lock` (CodeCarbon 2.7 leaves the lock behind
   on abnormal termination, which silently zeroes subsequent energy readings).
2. Starts a CodeCarbon `EmissionsTracker` (unless `--no-energy`).
3. Shells out to `llama-bench` from `--llama-dir/build/bin/Release/`, parses
   its JSON output, and records `avg_latency_ms_token` / `throughput_tokens_s`.
4. Samples `psutil.Process(...).memory_info().rss` during the run; the peak
   value is recorded as `peak_rss_mb`.
5. Stops the CodeCarbon tracker, records `energy_kwh` / `co2_kg`, and appends
   one row per config to the CSV (default `results/bitnet_step_metrics.csv`;
   override with `--out`).

CSV schema:
`timestamp, threads, n_prompt, n_gen, avg_latency_ms_token,
throughput_tokens_s, peak_rss_mb, energy_kwh, co2_kg`.

### `scripts/eval_accuracy.py`

llama-server-driven multiple-choice eval that matches the methodology in
`lm-evaluation-harness` for each task:

- **ARC-Easy / ARC-Challenge** — length-normalized loglikelihood (`acc_norm`)
  of the full answer text, scored as a continuation of the question.
- **WinoGrande** — partial-context scoring `P(suffix | prefix + option)`.
- **HellaSwag** — length-normalized loglikelihood with WikiHow `[title]`
  cleanup, prefixed by `activity_label`.
- **MMLU** — 5-shot, first-token letter scoring with the standard
  "The following are multiple choice questions..." header.

At startup, `_server_caps()` probes the server once and picks one of two
continuation-scoring paths depending on the llama.cpp build:

- **Upstream (Qwen):** force the target token with `logit_bias:[[id, +100]]`
  and read back the natural pre-bias logprob via `post_sampling_probs:false`.
  Exact for any vocab token.
- **Fork (BitNet):** the bias trick is unusable (reported probs are
  post-bias), so we fall back to a top-K=5000 search and use
  `min(top_K_logprob) − 1.0` as a conservative lower bound when the target
  token is rarer than top-5000.

This branching is the only methodology asymmetry between the two models and is
documented inline in `eval_accuracy.py` and in §6.1 of `QWEN_SUMMARY.md`.

Output JSON schema (per task): `accuracy`, `correct`, `total`, plus a
per-subject breakdown for MMLU.

### `scripts/compare_runs.py`

Reads the local CSVs and accuracy JSONs, joins them against the five FP16
baselines (`OTHER_BASELINES`, `BITNET_PAPER`, `QWEN_PAPER` — published numbers
from arXiv:2504.12285 Table 1), and writes:

1. **`results/comparison_table.csv`** — one row per model/source, columns:
   `model, source, throughput_tokens_s, peak_rss_mb, cost_per_1k_tokens,
   energy_cost_per_1k_tokens, arc_easy, arc_challenge, winogrande,
   hellaswag, mmlu`.
   - `cost_per_1k_tokens` — AWS proxy: throughput × `--hardware-rate`
     (default `$0.170/hr` c5.xlarge on-demand).  Available for every row.
   - `energy_cost_per_1k_tokens` — local electricity cost from CodeCarbon's
     measured `energy_kwh` × `--electricity-rate` (default `$0.16/kWh` US
     residential average).  Populated only for "ours" rows; paper rows
     leave it blank because we have no measured energy for them.

2. **Plots** in `results/plots/`:
   - `throughput_comparison.png` — two-panel: (a) cross-model comparison
     (paper FP16 baselines + BitNet paper/ours + Qwen paper/Q8 ours/Q4 ours)
     at the reference (512, 128) config; (b) per-config sensitivity for our
     three locally measured models across all three (n_prompt, n_gen) configs
   - `memory_comparison.png` — peak RSS, same model layout as throughput panel (a)
   - `accuracy_comparison.png` — grouped bars per task across all models
   - `cost_accuracy.png` — cost vs **mean of 5 benchmarks** (hollow ○ = paper, filled ♦ = ours)
   - `{task}_cost_accuracy.png` for each of `arc_easy, arc_challenge, winogrande, hellaswag, mmlu`
   - `memory_accuracy_pareto.png` — memory vs **mean of 5 benchmarks**
   - `{task}_memory_accuracy.png` per task
   - `energy_carbon_comparison.png` — three panels: Wh, gCO₂, and USD (local
     electricity cost at `--electricity-rate`) per 1k tokens
   - `cloud_cost_comparison.png` — log-scale bar chart of $/1k output tokens
     comparing self-hosted (BitNet/Qwen, both AWS-proxy and local-electricity
     framings) against cloud API services (OpenAI GPT-4o / GPT-4o mini,
     Anthropic Claude Haiku 4.5 / Sonnet 4.5 / Opus 4.7).  API prices are
     hardcoded in `CLOUD_API_PRICING` and should be re-verified before
     publication.

   The cost– and memory–accuracy scatters share a single `_accuracy_scatter`
   helper so the hollow-vs-filled marker convention and dotted paper→ours
   connector are consistent across the mean and per-task variants.

### `scripts/smoke_test.py`

Self-contained end-to-end check.  Runs three inference prompts per model
(basic arithmetic, factual recall, common sense) and asserts keyword hits;
then verifies that `compare_runs.py` produces all expected plots and that
`comparison_table.csv` has all expected columns; then runs `--help` on
`metrics_tracker.py` and `eval_accuracy.py`; finally runs a 5-sample accuracy
sweep per model.  ANSI-colored output with PASS/FAIL/SKIP badges; non-zero
exit on any failure.

---

## Results Summary (current `results/comparison_table.csv`)

| Model | Source | tok/s | Peak RSS | $/1k tok | ARC-E | ARC-C | Wino | HellaSwag | MMLU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaMA 3.2 1B | paper (FP16) | 4.5 | 2,600 | $0.01049 | 69.87 | 41.04 | 60.77 | 61.05 | 42.12 |
| Gemma-3 1B | paper (FP16) | 4.1 | 2,700 | $0.01152 | 79.42 | 46.25 | 66.38 | 72.15 | 50.33 |
| SmolLM2 1.7B | paper (FP16) | 3.5 | 3,300 | $0.01349 | 81.82 | 52.99 | 68.67 | 72.29 | 51.77 |
| MiniCPM 2B | paper (FP16) | 2.9 | 4,100 | $0.01628 | 82.20 | 51.96 | 68.27 | 75.08 | 53.07 |
| BitNet b1.58 2B4T | paper | 20.0 | 1,400 | $0.00236 | 74.79 | 49.91 | 71.90 | 68.44 | 53.17 |
| **BitNet b1.58 2B4T** | **ours** | **21.2** | **1,247** | **$0.00223** | **74.2** | **46.0** | **75.2** | **58.6** | **54.69** |
| Qwen2.5 1.5B | paper (FP16) | 3.8 | 3,100 | $0.01243 | 79.92 | 52.82 | 66.61 | 70.95 | 61.11 |
| **Qwen2.5-1.5B-Instruct Q8_0** | **ours** | **15.1** | **1,667** | **$0.00313** | **74.2** | **44.2** | **65.8** | **59.0** | **62.28** |

BitNet (ours) is ~40% faster, ~25% less memory, and ~29% cheaper than Qwen
(ours).  Mean-of-5 accuracy ties (61.7% vs 61.1%); BitNet wins WinoGrande by
+9.4pt, Qwen wins MMLU by +7.6pt.

---

## Tasks

### Phase 1 — Repository Study

- [x] `BITNET_SUMMARY.md` — paper summary, absmean quantization, STE, FP16 baseline table
- [x] `QWEN_SUMMARY.md` — Qwen2.5 architecture, Q8_0 quantization, upstream vs fork llama.cpp differences

### Phase 2 — Environment Setup & Model Acquisition

- [x] Clone `microsoft/BitNet` at pinned commit (`make bitnet-setup`)
- [x] Apply ClangCL patches (`make bitnet-patch` — three patches in `patches/`)
- [x] Build `bitnet.cpp` (`make bitnet-build`)
- [x] Download BitNet b1.58 2B4T GGUF (`make bitnet-model`)
- [x] Verify BitNet inference (`make bitnet-verify`)
- [x] Clone upstream `llama.cpp` at pinned commit + build (`make qwen-setup`)
- [x] Download Qwen2.5-1.5B-Instruct Q8_0 GGUF (`make qwen-model`)
- [x] Verify Qwen inference (`make qwen-verify`)
- [x] Set up Python environment via Poetry (`make install`)
- [x] Extend `eval_accuracy.py` and `metrics_tracker.py` to work with any llama.cpp build
- [x] Smoke-test the full pipeline (`make smoke-test`)

### Phase 3 — Inference Benchmarking

- [x] BitNet inference benchmark (`make benchmark-bitnet`) → `results/bitnet_step_metrics.csv`
- [x] BitNet accuracy eval, all 5 tasks (`make eval-accuracy-bitnet`) → `results/accuracy_results_bitnet.json`
- [x] CodeCarbon energy + CO₂ recorded per benchmark row (`energy_kwh`, `co2_kg`)
- [x] `REPORT.md` written with sanity-check against arXiv:2504.12285 Tables
- [x] Qwen inference benchmark (`make benchmark-qwen`) → `results/qwen_step_metrics.csv`
- [x] Qwen accuracy eval, all 5 tasks (`make eval-accuracy-qwen`) → `results/accuracy_results_qwen.json`

### Phase 4 — Cost Comparison

- [x] Compile published FP16 baselines + BitNet + Qwen into `comparison_table.csv`
- [x] `compare_runs.py` generates throughput, memory, accuracy, cost, and energy/CO₂ plots
- [x] Cost-accuracy trade-off proxy: time × hardware rate (AWS c5.xlarge on-demand, `$0.170/hr`)
- [x] Mean-of-5-tasks variant for cost–accuracy and memory–accuracy plots
- [x] Per-task variants: `{task}_cost_accuracy.png` and `{task}_memory_accuracy.png`
- [x] Energy / CO₂ comparison plot (`energy_carbon_comparison.png`)
- [x] Qwen "ours" present in every plot and table row
- [x] Wire Qwen Q4_K_M as a third "ours" model — Makefile targets
      (`benchmark-qwen-q4`, `eval-accuracy-qwen-q4`, `qwen-q4-model`,
      `qwen-q4-verify`, `smoke-test-qwen-q4`), `compare_runs.py` plot/table
      paths, and `smoke_test.py` model loop. Q4 row appears in every plot
      and the comparison CSV as data accumulates.
- [ ] Run Qwen Q4_K_M benchmarks and accuracy eval (`make qwen-q4-model && make benchmark-qwen-q4 && make eval-accuracy-qwen-q4`)
      so the Q4 row populates in `comparison_table.csv` and the plots.
- [ ] Refresh `FINAL_REPORT.md` with the Q4 row in the headline tables and
      a short discussion of the Q8 vs Q4 quantization-sensitivity result.
- [x] Compare measured energy against FP16 estimates from the literature
      (`FINAL_REPORT.md` §4 — paper J/tok values cross-referenced; documents
      the ~100–200× gap as a methodology mismatch between marginal-inference
      energy and CodeCarbon system-level wall power)
- [x] Produce final benchmark dashboard (plots + `comparison_table.csv`) in
      `FINAL_REPORT.md` — executive summary, all 18 plots referenced,
      discussion, threats-to-validity, conclusion

### Phase 5 — Optimization & Writeup

Scope note: this project benchmarks two fixed pre-trained models, not a training
run. Phase 5 is therefore scoped to *inference-side* tuning and to writing up
the comparison — there is no model-size scaling study to do.

- [x] Document the `-ub 128` constraint required by the TL2 kernel (REPORT.md §2, Makefile)
- [ ] Inference-side optimization sweep — pick at least one beyond the build-time `-ub` cap:
      thread-count sensitivity (e.g. 1 / 2 / 4 / 6 on the i5-9400F), `--mlock` vs mmap,
      or batched continuation scoring in `eval_accuracy.py`. Record results as additional
      rows in the existing `*_step_metrics.csv` files with a `tag` column noting the variant.
- [ ] Characterize where BitNet's efficiency advantage concentrates across the three benchmarked
      workload shapes — prompt-heavy `(512, 128)` vs generation-heavy `(1, 512)` vs long-context
      `(512, 512)` — using the per-config rows already in `bitnet_step_metrics.csv` /
      `qwen_step_metrics.csv`. Note any regime where Qwen narrows the throughput or memory gap.
- [ ] Cost-model sensitivity: re-run `compare_runs.py --hardware-rate` for at least one
      alternative (spot c5.xlarge, ARM Graviton on-demand, local desktop $0/hr) and confirm
      whether the BitNet < Qwen < FP16-baselines cost ordering is robust to the rate choice.
- [ ] Write capstone research report to `FINAL_REPORT.md` — methodology, headline numbers from
      `comparison_table.csv`, plots from `results/plots/`, and an explicit threats-to-validity
      section (single-CPU run, the logit-bias asymmetry between upstream and BitNet's fork,
      0-shot vs paper's 5-shot MMLU framing, hardware-rate sensitivity).
- [ ] Prepare final presentation slides in `presentation.pptx`.
- [ ] Clean up repository for reproducibility — `README.md` still references the obsolete
      `scripts/run_lm_eval.py` and should be aligned with the current `eval_accuracy.py` /
      `metrics_tracker.py` Makefile entry points.
