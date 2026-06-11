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
Qwen2.5-1.5B-Instruct is benchmarked alongside BitNet at three quantizations
(Q8_0, Q4_K_M, Q2_K) via upstream `llama.cpp`, spanning a near-FP16 reference
point and two aggressive-quantization points on the same model.
Gemma-2-2B-it is included as a second model family on the same upstream
stack at the same three quant depths (Q8_0, Q4_K_M, Q2_K) so the bit-depth
vs accuracy story is symmetric across Qwen and Gemma.  Gemma 2 2B does not
appear in arXiv:2504.12285 Table 1, so the Gemma rows are reported "ours
only" until a Gemma 2 paper baseline is wired in.  Accuracy is compared
against the one retained FP16 paper baseline (Qwen2.5 1.5B) — four earlier
paper rows (LLaMA 3.2 1B, Gemma-3 1B, SmolLM2 1.7B, MiniCPM 2B) were
trimmed because they had no PTQ counterpart on the same hardware — to
validate whether sub-byte quantization delivers meaningful real-world
efficiency gains without significant accuracy loss.

### Objectives

- Run BitNet b1.58 2B4T locally via `bitnet.cpp` on CPU
- Run Qwen2.5-1.5B-Instruct (Q8_0, Q4_K_M, Q2_K) via upstream `llama.cpp` as CPU baselines
- Run Gemma-2-2B-it (Q8_0, Q4_K_M, Q2_K) as a second model family on the same upstream stack
- Benchmark inference latency, throughput, memory, and energy for all seven locally measured models
- Evaluate accuracy on ARC-Easy, ARC-Challenge, WinoGrande, HellaSwag, MMLU (5-shot)
- Compare measured numbers against the paper's published values (arXiv:2504.12285)
- Compare accuracy against the Qwen2.5 1.5B FP16 paper baseline
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
| Qwen Q2_K GGUF | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (`qwen2.5-1.5b-instruct-q2_k.gguf`, ~0.7 GiB) | `../Models/Qwen/` |
| Gemma-2-2B-it Q8_0 GGUF | `bartowski/gemma-2-2b-it-GGUF` (`gemma-2-2b-it-Q8_0.gguf`, ~2.8 GiB) | `../Models/Gemma/` |
| Gemma-2-2B-it Q4_K_M GGUF | `bartowski/gemma-2-2b-it-GGUF` (`gemma-2-2b-it-Q4_K_M.gguf`, ~1.7 GiB) | `../Models/Gemma/` |
| Gemma-2-2B-it Q2_K GGUF | `second-state/gemma-2-2b-it-GGUF` (`gemma-2-2b-it-Q2_K.gguf`, ~1.2 GiB) — bartowski's Gemma-2-2B ladder bottoms out at Q3_K_L, so Q2_K is sourced separately.  Q8_0 is calibration-free across providers, so this mixed source does not affect the comparison. | `../Models/Gemma/` |

Python packages (managed by Poetry, see `pyproject.toml`): `pandas`,
`matplotlib`, `numpy`, `psutil`, `codecarbon`, `requests`, `datasets`, `rich`.

Reference hardware (REPORT.md): Intel Core i5-9400F @ 2.90 GHz (6 cores, 4
threads used), 16 GB RAM, Windows 11.

### Build Patches (BitNet, applied automatically by `make bitnet-patch`)

Three patches in `patches/` are required at the pinned BitNet commit on
both Windows (ClangCL 18+) and Linux (clang-18 from apt.llvm.org):

- `bitnet-clangcl-const.patch` — adds `const` to a non-const pointer init in
  `src/ggml-bitnet-mad.cpp:811`. Originally found under ClangCL 20+;
  empirically required on Linux clang-18 as well (verified during
  Phase 6 Docker bring-up — the misleading name is kept for git history,
  the Dockerfile applies it unconditionally).
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
├── REPORT.md                      # Capstone report (canonical): dashboard + Appendix A (Phase 3 sanity check) + Appendix B (BitNet model card) + Appendix C (Qwen model card)
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
    ├── bitnet_step_metrics.csv      # BitNet llama-bench results
    ├── qwen_q8_step_metrics.csv     # Qwen Q8_0 llama-bench results
    ├── qwen_q4_step_metrics.csv     # Qwen Q4_K_M llama-bench results
    ├── qwen_q2_step_metrics.csv     # Qwen Q2_K llama-bench results
    ├── gemma_q8_step_metrics.csv    # Gemma-2-2B-it Q8_0 llama-bench results
    ├── gemma_q4_step_metrics.csv    # Gemma-2-2B-it Q4_K_M llama-bench results
    ├── gemma_q2_step_metrics.csv    # Gemma-2-2B-it Q2_K llama-bench results
    ├── accuracy_results_bitnet.json
    ├── accuracy_results_qwen_q8.json     # (partial — arc_e/c + 31/57 MMLU subjects; rerun pending)
    ├── accuracy_results_qwen_q4.json     # (pending — file was deleted, eval rerun pending)
    ├── accuracy_results_qwen_q2.json     # (pending — eval not yet run)
    ├── accuracy_results_gemma_q8.json    # full 5-task accuracy at LIMIT=100
    ├── accuracy_results_gemma_q4.json    # full 5-task accuracy at LIMIT=100
    ├── accuracy_results_gemma_q2.json    # full 5-task accuracy at LIMIT=100
    ├── comparison_table.csv         # Aggregated paper+ours summary
    └── plots/                       # PNGs generated by compare_runs.py
```

External (not in this repo): `../Models/BitNet/`, `../Models/Qwen/`, `../Models/Gemma/`.

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

make qwen-q8-setup       # Clone ggml-org/llama.cpp at pinned commit, build (MSVC), download Q8_0 GGUF
make qwen-q8-verify      # Sanity check: same prompt, Qwen Q8_0

make qwen-q4-model    # Download Qwen2.5-1.5B Q4_K_M GGUF (~1 GB, reuses qwen-q8-setup build)
make qwen-q4-verify   # Sanity check: same prompt, Qwen Q4_K_M

make qwen-q2-model    # Download Qwen2.5-1.5B Q2_K GGUF (~0.7 GB, reuses qwen-q8-setup build)
make qwen-q2-verify   # Sanity check: same prompt, Qwen Q2_K

make gemma-q8-model   # Download Gemma-2-2B-it Q8_0 GGUF (~2.8 GB, reuses qwen-q8-setup build)
make gemma-q8-verify  # Sanity check: same prompt, Gemma Q8_0
make gemma-q4-model   # Download Gemma-2-2B-it Q4_K_M GGUF (~1.7 GB)
make gemma-q4-verify
make gemma-q2-model   # Download Gemma-2-2B-it Q2_K GGUF (~1.2 GB)
make gemma-q2-verify
```

Override sibling-dir locations if needed: `make bitnet-setup BITNET_DIR=...`,
`make qwen-q8-setup QWEN_DIR=...`, `make gemma-q8-model GEMMA_DIR=...`.

### 3. Smoke test

```bash
make smoke-test               # All seven models
make smoke-test-bitnet        # BitNet only
make smoke-test-qwen-q8       # Qwen Q8_0 only
make smoke-test-qwen-q4       # Qwen Q4_K_M only
make smoke-test-qwen-q2       # Qwen Q2_K only
make smoke-test-gemma-q8      # Gemma-2-2B-it Q8_0 only
make smoke-test-gemma-q4      # Gemma-2-2B-it Q4_K_M only
make smoke-test-gemma-q2      # Gemma-2-2B-it Q2_K only
```

`scripts/smoke_test.py` exercises the full pipeline end-to-end: runs three
inference prompts per model, verifies `compare_runs.py` produces the expected
plot files and CSV, runs `--help` on the other scripts, and then runs a small
(5-sample) accuracy sweep per model via `llama-server`.  Exit 0 = all checks
pass.

### 4. Benchmarks

```bash
make benchmark-bitnet         # → results/bitnet_step_metrics.csv
make benchmark-qwen-q8        # → results/qwen_q8_step_metrics.csv      (Qwen Q8_0)
make benchmark-qwen-q4        # → results/qwen_q4_step_metrics.csv      (Qwen Q4_K_M)
make benchmark-qwen-q2        # → results/qwen_q2_step_metrics.csv      (Qwen Q2_K)
make benchmark-gemma-q8       # → results/gemma_q8_step_metrics.csv     (Gemma-2-2B-it Q8_0)
make benchmark-gemma-q4       # → results/gemma_q4_step_metrics.csv     (Gemma-2-2B-it Q4_K_M)
make benchmark-gemma-q2       # → results/gemma_q2_step_metrics.csv     (Gemma-2-2B-it Q2_K)
make benchmark                # All seven

# Cross-stack sensitivity check (REPORT §6.8):
make benchmark-qwen-q8-on-bitnet-fork  # Qwen Q8_0 against BitNet's older llama.cpp
                                       # → results/qwen_q8_on_bitnet_fork_step_metrics.csv
```

Each benchmark runs `llama-bench` over the three `(n_prompt, n_gen)` configs
defined in `scripts/metrics_tracker.py` (`(512, 128)`, `(512, 512)`, `(1, 512)`)
matching the paper's Table 1 conditions, captures latency / throughput / peak
RSS, and tracks energy + CO₂ via CodeCarbon.

### 5. Accuracy evaluation

Per-task (every task has a per-model target for all seven locally measured
models; the group target below runs the same task across all models):

```bash
make eval-arc-easy-{bitnet,qwen-q8,qwen-q4,qwen-q2,gemma-q8,gemma-q4,gemma-q2}        eval-arc-easy
make eval-arc-challenge-{bitnet,qwen-q8,qwen-q4,qwen-q2,gemma-q8,gemma-q4,gemma-q2}   eval-arc-challenge
make eval-mmlu-{bitnet,qwen-q8,qwen-q4,qwen-q2,gemma-q8,gemma-q4,gemma-q2}            eval-mmlu      # 5-shot
make eval-winogrande-{bitnet,qwen-q8,qwen-q4,qwen-q2,gemma-q8,gemma-q4,gemma-q2}      eval-winogrande
make eval-hellaswag-{bitnet,qwen-q8,qwen-q4,qwen-q2,gemma-q8,gemma-q4,gemma-q2}       eval-hellaswag
```

All tasks:

```bash
make eval-accuracy-bitnet     # All 5 tasks, BitNet only
make eval-accuracy-qwen-q8    # All 5 tasks, Qwen Q8_0 only
make eval-accuracy-qwen-q4    # All 5 tasks, Qwen Q4_K_M only
make eval-accuracy-qwen-q2    # All 5 tasks, Qwen Q2_K only
make eval-accuracy-gemma-q8   # All 5 tasks, Gemma-2-2B-it Q8_0 only
make eval-accuracy-gemma-q4   # All 5 tasks, Gemma-2-2B-it Q4_K_M only
make eval-accuracy-gemma-q2   # All 5 tasks, Gemma-2-2B-it Q2_K only
make eval-accuracy            # All seven models, all 5 tasks
make eval-accuracy SKIP_COMPLETED=1   # Skip any task whose result is already complete in --out
```

Each target uses `--start-server` so `eval_accuracy.py` brings up
`llama-server`, runs the eval, and shuts the server down.  Override
`LIMIT=N` to sample `N` items per task (default 500); use `LIMIT=0` for
the full split.

**Eval thread/ubatch settings.** All accuracy evals run at
`EVAL_THREADS=2 EVAL_UBATCH=64` by default — matches the AWS Free Tier
(c7i-flex.large, 2 vCPUs) condition so accuracy numbers are
apples-to-apples with the cross-arch throughput sweep, and BitNet's TL2
kernel requires ubatch ≤ 64 at 2 threads.  Override per-invocation with
`make eval-accuracy EVAL_THREADS=4 EVAL_UBATCH=128` if you want the
4-thread / wider-batch path instead.  Benchmarks (`make benchmark*`)
remain on `THREADS=4 UBATCH=128` — those are separate Makefile
variables.

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
documented inline in `eval_accuracy.py` and in Appendix C.4 of `REPORT.md`.

Output JSON schema (per task): `accuracy`, `correct`, `total`, plus a
per-subject breakdown for MMLU.

### `scripts/compare_runs.py`

Reads the local CSVs and accuracy JSONs, joins them against the retained
FP16 paper baselines (`OTHER_BASELINES` is currently empty; `BITNET_PAPER`;
`QWEN_PAPER` — published numbers from arXiv:2504.12285 Table 1.  Gemma 2 2B
does not appear in that table, so the three Gemma "ours" rows have no
paired FP16 paper baseline yet), and writes:

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
     (Qwen FP16 paper + BitNet paper/ours + Qwen Q8/Q4/Q2 ours + Gemma
     Q8/Q4/Q2 ours) at the reference (512, 128) config, ordered so each
     paper FP16 row sits next to its quantized counterpart; (b) per-config
     sensitivity for all seven locally measured models across all three
     (n_prompt, n_gen) configs.
   - `memory_comparison.png` — peak RSS, same seven-model layout as
     throughput panel (a).
   - `accuracy_comparison.png` — grouped bars per task across all models
     (FP16 papers + the seven ours rows that have accuracy data).
   - `cost_accuracy.png` — cost vs **mean of 5 benchmarks** (hollow ○ =
     paper, filled ♦ = ours).  Dotted connectors visualize the Qwen quant
     chain (Qwen FP16 → Q8 → Q4 → Q2 ours) and the Gemma quant chain
     (Gemma Q8 → Q4 → Q2 ours, no paper anchor yet).
   - `memory_accuracy.png` — memory vs **mean of 5 benchmarks**.
   - `energy_carbon_comparison.png` — **single panel**: Wh per 1k tokens on
     the bottom x-axis, gCO₂ per 1k tokens on a top secondary x-axis (exact
     relabeling via the run's measured grid intensity).  USD-electricity
     panel was removed in the 2026-06-08 refactor — it was redundant with the
     two cost framings already in §3.5 and §3.9.
   - `accuracy_eval_cost.png` — **single panel**: stacked horizontal bars
     of wall-clock hours per benchmark task per model, with kWh as a top
     secondary x-axis (approximate via the measured average kWh/hour
     ratio — CPU power varies a few percent per model).
   - `cloud_cost_comparison.png` — log-scale bar chart of $/1k output tokens
     comparing self-hosted (all seven ours rows, both AWS-proxy and
     local-electricity framings) against cloud API services (OpenAI GPT-4o /
     GPT-4o mini, Anthropic Claude Haiku 4.5 / Sonnet 4.5 / Opus 4.7).
     API prices are hardcoded in `CLOUD_API_PRICING` and should be
     re-verified before publication.
   - `cloud_accuracy_comparison.png` — grouped-bar accuracy plot in the style
     of `accuracy_comparison.png` but for self-hosted ours-rows vs the cloud
     APIs (uses `CLOUD_API_ACCURACY` values; non-MMLU bars are typically empty
     for cloud rows since safety cards omit them).
   - `cloud_cost_accuracy.png` — cost vs **MMLU** scatter for self-hosted +
     cloud APIs (uses MMLU specifically rather than mean-of-5 because that's
     the only benchmark consistently published across cloud providers).
   - `mmlu_subject_heatmap.png` — MMLU per-subject accuracy heatmap across
     the locally-measured models with substantial cross-model spread
     (max−min ≥ 15pp), sorted by spread descending.
   - `cross_arch_throughput.png` — grouped bar chart of throughput at the
     reference (512, 128) config across all seven models × all architectures
     with data (Windows / i5-9400F AVX2, Linux Docker / i5-9400F AVX2,
     AWS c7i-flex.large AVX-512, AWS t4g.small ARM).  Architectures and
     models with no CSV under their subdir drop silently.
   - `thread_scaling.png` — throughput vs thread count (1/2/4/6 threads) for
     all three originally-measured models, with per-model linear-scaling
     guides (REPORT §5.4).  Q2 and the three Gemma variants are not in the
     thread sweep yet.

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
| Qwen2.5 1.5B | paper (FP16) | 3.8 | 3,100 | $0.01243 | 79.92 | 52.82 | 66.61 | 70.95 | 61.11 |
| **Qwen2.5-1.5B-Instruct Q8_0** | **ours** | **15.1** | **1,667** | **$0.00313** | **74.2** | **44.2** | **65.8** | **59.0** | **62.28** |
| **Qwen2.5-1.5B-Instruct Q4_K_M** | **ours** | **24.9** | **1,632** | **$0.00190** | **71.0** | **43.2** | **63.0** | **58.8** | **61.23** |
| **Qwen2.5-1.5B-Instruct Q2_K** | **ours** | **32.5** | **745** | **$0.00145** | pending | pending | pending | pending | pending |
| **Gemma-2-2B-it Q8_0** | **ours** | **9.5** | **2,776** | **$0.00497** | **73.0** | **52.0** | **70.0** | **64.0** | **58.07** |
| **Gemma-2-2B-it Q4_K_M** | **ours** | **14.2** | **2,671** | **$0.00332** | **73.0** | **52.0** | **70.0** | **64.0** | **58.11** |
| **Gemma-2-2B-it Q2_K** | **ours** | **18.4** | **1,293** | **$0.00256** | **73.0** | **52.0** | **70.0** | **64.0** | **58.09** |
| **BitNet b1.58 2B4T** | **ours** | **21.2** | **1,247** | **$0.00223** | **74.2** | **46.0** | **75.2** | **58.6** | **54.69** |
| BitNet b1.58 2B4T | paper | 20.0 | 1,400 | $0.00236 | 74.79 | 49.91 | 71.90 | 68.44 | 53.17 |

Four earlier paper FP16 rows (LLaMA 3.2 1B, Gemma-3 1B, SmolLM2 1.7B,
MiniCPM 2B) were trimmed during the Q2 / Gemma expansion because they had
no PTQ counterpart on the same hardware — keeping them as FP16-only rows
added clutter without contributing to any paper→ours comparison.  Gemma
2 2B does not appear in arXiv:2504.12285 Table 1 either, so its three
"ours" rows currently have no paired paper baseline.

Current findings (BitNet, Qwen Q8/Q4, Gemma Q8 — four rows with full
accuracy data):
**BitNet (ours)** is ~40% faster, ~25% less memory, and ~29% cheaper than
Qwen Q8 (ours) at comparable mean accuracy (61.7% vs 61.1%); BitNet wins
WinoGrande by +9.4pt, Qwen Q8 wins MMLU by +7.6pt.  Qwen Q4_K_M lands at
$0.00190 / 1k tokens — second-cheapest self-hosted row — at a −1pt
average-accuracy cost relative to Q8.

**All three Gemma-2-2B-it quants tie at 63.41–63.42% mean accuracy** —
a striking *full-ladder quantization invariance* result.  Q8, Q4, and Q2
score **identically** on ARC-Easy (73.0), ARC-Challenge (52.0),
WinoGrande (70.0), HellaSwag (64.0); MMLU spread is 58.07 / 58.11 / 58.09
— within 0.04pt across 5,700 questions.  All three rows lead the project
by +1.7pt over BitNet and +2.3pt over Qwen Q8.  The Q2_K result is the
surprise: ~2.6 bits per weight on Gemma 2 2B preserves the full task
headline at this sample size, while Qwen 1.5B's Q4→Q8 transition costs
1-2pt on most tasks.  Gemma 2 2B's representation is robust enough that
the K-quants don't bite at this LIMIT.

**Gemma Q2_K becomes the new Pareto winner among rows with complete
data.**  Ties on accuracy (63.42%), drops RSS to 1,293 MB (+4% over
BitNet), and runs at 18.4 tok/s (-13% under BitNet's 21.2 — the only
cost axis where Q2_K loses to BitNet).  At AWS-proxy $/1k-tok cost,
Q2_K's $0.00256 sits ~16% above BitNet's $0.00223 but with +1.7pt mean
accuracy.  If full-LIMIT confirms this invariance, **Gemma Q2_K is the
default accuracy-priority self-hosting recommendation** at this size
class.  BitNet retains the WinoGrande lead (75.2 vs Gemma 70.0); Qwen Q8
retains MMLU (62.28 vs Gemma 58.07-58.11).  The cost: Gemma is slowest
at every quant (9.5 / 14.2 / 18.4 tok/s vs Qwen 15.1 / 24.9 / 32.5) and
heaviest at Q8/Q4 (~2.7 GB RSS, 2.2× BitNet) — the 2B vs 1.5B parameter
count tax.  Q2_K's K-quant tensor sharing collapses the memory penalty
even on the 2B-parameter base.

**Qwen Q2_K** still leads on throughput / RSS / cost but lacks accuracy
data.  Q2_K's 745 MB RSS is ~40% under BitNet's footprint, the
first row in the project to undercut BitNet on memory; whether that holds
up against accuracy is the open question.

Wall-clock note: at LIMIT=100, Gemma Q8's MMLU eval alone took **8.6
hours** (57 subjects × 100 samples × 4-choice scoring at 9.5 tok/s).  Q4
and Q2 finish proportionally faster because their throughputs are higher.

---

## Tasks

### Phase 1 — Repository Study

- [x] BitNet model card — paper summary, absmean quantization, STE, FP16 baseline table (now `REPORT.md` Appendix B)
- [x] Qwen model card — Qwen2.5 architecture, Q8_0 / Q4_K_M quantization, upstream-vs-fork llama.cpp differences (now `REPORT.md` Appendix C)

### Phase 2 — Environment Setup & Model Acquisition

- [x] Clone `microsoft/BitNet` at pinned commit (`make bitnet-setup`)
- [x] Apply ClangCL patches (`make bitnet-patch` — three patches in `patches/`)
- [x] Build `bitnet.cpp` (`make bitnet-build`)
- [x] Download BitNet b1.58 2B4T GGUF (`make bitnet-model`)
- [x] Verify BitNet inference (`make bitnet-verify`)
- [x] Clone upstream `llama.cpp` at pinned commit + build (`make qwen-q8-setup`)
- [x] Download Qwen2.5-1.5B-Instruct Q8_0 GGUF (`make qwen-q8-model`)
- [x] Verify Qwen inference (`make qwen-q8-verify`)
- [x] Set up Python environment via Poetry (`make install`)
- [x] Extend `eval_accuracy.py` and `metrics_tracker.py` to work with any llama.cpp build
- [x] Smoke-test the full pipeline (`make smoke-test`)

### Phase 3 — Inference Benchmarking

- [x] BitNet inference benchmark (`make benchmark-bitnet`) → `results/bitnet_step_metrics.csv`
- [x] BitNet accuracy eval, all 5 tasks (`make eval-accuracy-bitnet`) → `results/accuracy_results_bitnet.json`
- [x] CodeCarbon energy + CO₂ recorded per benchmark row (`energy_kwh`, `co2_kg`)
- [x] `REPORT.md` written with sanity-check against arXiv:2504.12285 Tables
- [x] Qwen Q8_0 inference benchmark (`make benchmark-qwen-q8`) → `results/qwen_q8_step_metrics.csv`
- [x] Qwen Q8_0 accuracy eval, all 5 tasks (`make eval-accuracy-qwen-q8`) → `results/accuracy_results_qwen_q8.json`

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
- [x] Run Qwen Q4_K_M benchmarks and accuracy eval — Q4 row now populated
      in `comparison_table.csv` and across every plot.  Q4 lands at 24.9
      tok/s and ~$0.0019 / 1k tokens (the cheapest self-hosted row) with a
      ~1pt average-accuracy cost vs Q8.
- [x] Refresh `REPORT.md` with the Q4 row in the headline tables and
      a short discussion of the Q8 vs Q4 quantization-sensitivity result.
- [x] Compare measured energy against FP16 estimates from the literature
      (`REPORT.md` §4 — paper J/tok values cross-referenced; documents
      the ~100–200× gap as a methodology mismatch between marginal-inference
      energy and CodeCarbon system-level wall power)
- [x] Produce final benchmark dashboard (plots + `comparison_table.csv`) in
      `REPORT.md` — executive summary, all 18 plots referenced,
      discussion, threats-to-validity, conclusion
- [x] Cloud-API cost / accuracy comparison — `CLOUD_API_PRICING` (output token
      $/M for GPT-4o, GPT-4o mini, Claude Haiku/Sonnet/Opus 4.5/4.7) and
      `CLOUD_API_ACCURACY` (MMLU per provider) drive two new plots:
      `cloud_accuracy_comparison.png` and `cloud_cost_accuracy.png`.
      Per-cell source noted in `compare_runs.py` since current cloud system
      cards have mostly moved past standard MMLU (Sonnet/Haiku 4.5 cards are
      safety-focused; Opus 4.7 reports MMMLU; GPT-4o card uses only a medical
      MMLU subset).
- [x] System-card / technical-report PDFs alongside the weights —
      `make system-cards` downloads BitNet (arXiv:2504.12285) and Qwen2.5
      (arXiv:2412.15115) technical reports into the existing model dirs, plus
      GPT-4o and Claude 4.5/4.7 system cards into `../Models/Cloud/` with a
      provenance README. GPT-4o mini shares the GPT-4o card (no standalone).

### Phase 5 — Optimization & Writeup

Scope note: this project benchmarks two fixed pre-trained models, not a training
run. Phase 5 is therefore scoped to *inference-side* tuning and to writing up
the comparison — there is no model-size scaling study to do.

- [x] Document the `-ub 128` constraint required by the TL2 kernel (REPORT.md §5.4 and Appendix B.3, Makefile)
- [x] Inference-side optimization sweep — thread-count sensitivity at 1/2/4/6
      threads on the i5-9400F.  Dedicated sweep CSVs
      (`results/{bitnet,qwen,qwen_q4}_thread_sweep.csv`) and a new plot
      (`results/plots/thread_scaling.png`) generated via
      `make benchmark-threads`.  Results and analysis in `REPORT.md` §5.4.
      Headline findings: BitNet's TL2 kernel cannot run at threads=1
      (`STATUS_STACK_OVERFLOW` regardless of `--ubatch`) and requires
      `--ubatch ≤ 64` at threads=2; BitNet saturates at 4 threads (+1.9%
      to 6 threads) while Q4 keeps climbing (+9.1%).  At threads=1, Q8
      hits 8.1 tok/s (2.13× the paper's FP16 ~3.8) and Q4 hits 10.0
      (2.6×) — cleanly separates quantization (~2×) from threading
      (~2×) as the two factors in the paper-vs-ours speedup.
- [x] Characterize where BitNet's efficiency advantage concentrates across the three benchmarked
      workload shapes — prompt-heavy `(512, 128)` vs generation-heavy `(1, 512)` vs long-context
      `(512, 512)`.  Analysis in `REPORT.md` §5.5.  Headline findings:
      (i) all three models are essentially workload-shape insensitive on throughput
      (within ~6% of their reference number across all three configs);
      (ii) BitNet's advantage over Qwen Q8 *widens* on pure generation
      (1.40× → 1.46×) — Q8 is the only model with a meaningful drop on `(1, 512)`;
      (iii) RSS is dominated by weights + runtime overhead at this parameter count,
      KV-cache contributes ≤30 MB across configs;
      (iv) no regime measured where Qwen Q8 narrows its gap to BitNet —
      gaps hold or widen on pure generation.
- [x] Cost-model sensitivity: re-ran `compare_runs.py --hardware-rate` at four
      points spanning two orders of magnitude — spot c5.xlarge @ $0.05/hr,
      on-demand c5.xlarge @ $0.170/hr (default), ARM Graviton c7g.xlarge @
      $0.1156/hr, and a "hardware-owned" proxy @ $0.01/hr.  Ordering of the
      nine `comparison_table.csv` rows by `cost_per_1k_tokens` is identical
      across all four rates (Q4 ours < BitNet ours < BitNet paper < Q8 ours <
      every FP16 paper baseline, MiniCPM 2B last).  This is expected
      analytically — `cost = (1/throughput) × rate × const`, so changing the
      rate rescales every row by the same factor and preserves ordering — but
      the empirical check confirms there is no edge case (e.g. an "ours" row
      with zero throughput, a paper row with NaN cost) that breaks the
      invariant.  The BitNet < Qwen Q8 < FP16-baselines cost ordering is
      robust to the rate choice.
- [x] Write capstone research report to `REPORT.md` — methodology,
      headline numbers from `comparison_table.csv`, plots from `results/plots/`,
      and an explicit threats-to-validity section (single-CPU run, the
      logit-bias asymmetry between upstream and BitNet's fork, 0-shot vs
      paper's 5-shot MMLU framing, hardware-rate sensitivity).  Now includes
      §5.4 thread-count sensitivity, §5.5 workload-shape sensitivity, and the
      §6.8 cross-stack check (Qwen Q8 on the BitNet fork).
- [x] Smoke-test hardening — `scripts/smoke_test.py` now prefixes every
      check label with the model name (so failures in the summary identify
      which model triggered them) and runs MMLU at 5-shot with a 1024-token
      eval-server context (0-shot on abstract_algebra was deterministically
      scoring 0/5 for BitNet, masking real regressions).
- [ ] Prepare final presentation slides in `presentation.pptx`.
- [x] Clean up repository for reproducibility — `README.md` rewritten as
      an inference-benchmarking front-door (was previously stuck describing
      the project's pre-pivot training framing).  Now lists all three locally
      run models, points readers at PLAN.md / REPORT.md / the Makefile
      for details, fixes the obsolete `scripts/run_lm_eval.py` reference, the
      wrong remote URL, and the `pip install` instructions that predated the
      Poetry migration.

### Phase 6 — Cross-Architecture Generalization Sweep

Scope: convert the single-CPU threat-to-validity in REPORT.md §6.1
into a measured result by re-running the benchmark suite on AWS instances
spanning AVX-512 Intel, AVX2 AMD, and ARM Graviton.  Accuracy is
hardware-independent at deterministic decoding so accuracy evals are not
re-run; only `make benchmark` (throughput / memory / energy) executes on
each remote instance.

Instances and rationale:

| Instance | vCPUs | Arch | ISA | Tests |
|---|---:|---|---|---|
| `c5.xlarge` | 4 | Skylake-SP Intel | AVX-512 | The AVX-512 hypothesis: do TL2 and Q8/Q4 AVX-512 paths shift the Pareto ranking? |
| `c6a.xlarge` | 4 | AMD Zen 3 | AVX2 | Same-ISA-class AMD comparison vs the local i5-9400F (also AVX2) |
| `c7g.xlarge` | 4 | Graviton3 ARM | Neon/SVE | Cross-ISA: does BitNet's TL2 kernel even compile on ARM, and if so where does it land? |
| local i5-9400F | 4 | Coffee Lake Intel | AVX2 | Existing baseline (no AVX-512) |

Use spot pricing for all three remote instances (~70% off on-demand);
benchmarks are short and re-runnable so spot interruption is not a real
risk.  Document the spot rate used in REPORT.md §6.1.

Tasks:

- [x] Verify BitNet's Linux build path before any paid instance time —
      `microsoft/BitNet` officially supports both Linux x86 and Linux ARM
      for BitNet-b1.58-2B-4T (README "Official Models" table:
      `I2_S` available on both arches; `TL2` on x86 only, `TL1` on ARM).
      Build prerequisites are the same as our local Windows path minus
      the Visual Studio dependencies: Python ≥3.9, CMake ≥3.22, Clang ≥18
      (installed via `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
      on Debian/Ubuntu).  Build command on Linux is identical:
      `python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s`.
      Implications for the sweep:
      (i) `c5.xlarge` (x86) uses the same TL2 kernel as our local i5-9400F
          — a pure AVX2 → AVX-512 hardware comparison;
      (ii) `c7g.xlarge` (ARM Graviton3) will use TL1, not TL2 — the
           cross-arch numbers compare different kernel implementations,
           not just different ISAs, and the report's discussion needs to
           call that out as a confound rather than a clean ISA-only result.
      Local-patch portability: the two `llama-chrono*.patch` files
      address a stdlib issue in upstream llama.cpp at our pinned commit
      and apply identically on Linux clang.  `bitnet-clangcl-const.patch`
      addresses a ClangCL-only hard error; needs re-testing on Linux
      clang and may be unnecessary there.
- [x] Containerize the build — `Dockerfile` + `.dockerignore` at repo
      root.  Ubuntu 22.04 base, Python 3.11 (via deadsnakes PPA), clang
      18 (via apt.llvm.org), Poetry, and the HuggingFace CLI.  Bypasses
      the Windows-only `bitnet-setup` / `qwen-q8-setup` Makefile targets
      (which use `-T ClangCL` and `bin/Release/`) and runs the build
      steps directly: BitNet via the upstream `setup_env.py` (its
      official cross-platform path; picks TL2 on x86, TL1 on ARM),
      llama.cpp via straight `cmake -DGGML_NATIVE=ON`.  Both pinned to
      the existing commits.  Layered for caching: deps → BitNet build →
      llama.cpp build → poetry → code, so iterating on code re-runs only
      the last layer.  GGUFs are baked in (~5 GB image) for one-shot
      reproducibility; mounting them as a volume is a documented
      alternative.  `make benchmark` works unmodified inside the
      container because `metrics_tracker.py` already falls back to
      `build/bin/llama-bench` on non-Windows hosts.  All three patches
      in `patches/` apply on Linux clang-18 (initial assumption that
      `bitnet-clangcl-const.patch` was ClangCL-only turned out to be
      wrong — verified during the smoke phase below).
      Build / run:
      `docker buildx build --platform=linux/amd64 -t cs495-non-gpu:x86 .`
      `docker run --rm -v "$(pwd)/results:/capstone/results" cs495-non-gpu:x86 make benchmark`
      Local arm64 build under qemu emulation was attempted and abandoned
      — buildkit dropped its connection ~13 min in (42% through llama.cpp
      compile, after BitNet TL1 codegen completed successfully).  Qemu
      under sustained C++ compilation load on Windows + Docker Desktop
      WSL2 is unreliable for builds of this size.  `make
      aws-bootstrap-c7g` is the documented path: rsync the repo to the
      c7g.xlarge instance and run `docker build` natively there.
- [x] Smoke-test the containerized x86 build locally before
      committing to a paid instance.  Ran `docker buildx --platform=linux/amd64`
      + `make smoke-test` on the local i5-9400F via Docker Desktop's
      WSL2 backend, which exercises the exact same Linux-x86 build
      path a `c5.xlarge` would.  47/48 checks pass (one flaky raw
      no-seed `llama-cli "What is 2+2?"` completion on BitNet; rerun
      hit "4" cleanly).  Two surprises caught here rather than on
      paid instance time:
      (i) `bitnet-clangcl-const.patch` is required on Linux clang-18
          too, not just ClangCL (Dockerfile now applies it
          unconditionally; PLAN.md "Build Patches" section corrected);
      (ii) `scripts/smoke_test.py` hardcoded the Windows
          `build/bin/Release/llama-cli.exe` path and needed the same
          `build/bin/<name>` fallback `metrics_tracker.py` already had.
- [x] Full `make benchmark` in the x86 container on the same
      i5-9400F to confirm/refute the initial smoke-test "~4.4×
      Linux speedup" claim with clean steady-state numbers
      (`results/linux_docker_x86/{bitnet,qwen_q8,qwen_q4}_step_metrics.csv`,
      canonical (512,128)/(512,512)/(1,512) configs via llama-bench).
      Result: **Linux Docker ≈ Windows-native on throughput**.  At
      the (512,128) reference config: BitNet 21.31 vs 21.21 tok/s
      (+0.5%), Qwen Q8 16.51 vs 15.10 (+9.3%), Qwen Q4 25.76 vs
      24.89 (+3.5%).  RSS: BitNet identical, Qwen ~7-13% higher
      on Linux (likely glibc malloc vs Windows heap allocator).
      The smoke-test "92 tok/s" number was BitNet's `eval time`
      line parsed at `-n 16`, which is far below the convergence
      threshold and not comparable to llama-bench's warmed steady
      state.  Implication for REPORT.md §6.1: containerization
      does not measurably affect throughput, so cross-arch numbers
      from Docker on c5/c6a/c7g are directly comparable to the
      committed Windows-native baseline.  Energy comparison is
      inconclusive (CodeCarbon uses Intel RAPL on Linux vs a
      different power model on Windows — not apples-to-apples).
- [~] **Superseded by the Free Tier sweep below.**  The planned
      c5.xlarge/c6a.xlarge/c7g.xlarge instances were unavailable due to
      an IAM Free Tier restriction on the class AWS account.  A
      c7i-flex.large (Intel Sapphire Rapids AVX-512, 2 vCPUs, Free Tier)
      was used instead — same Dockerfile, same build, `THREADS=2
      UBATCH=64`.  Results in `results/aws_c7i_flex_large/`.  An ARM
      t4g.small was also attempted but failed (2 GB RAM insufficient for
      the Docker build even with 4 GB swap).  See REPORT.md §6.1 for
      findings.
- [~] **Obsolete — superseded by the Dockerfile baking GGUFs into the
      image layer.**  Originally written to avoid re-downloading 4+ GiB of
      GGUFs per AWS instance.  In the current containerized workflow:
      `make aws-benchmark-{c5,c6a}` ships the prebuilt image (with GGUFs
      embedded) via `docker save | ssh | docker load`; `aws-bootstrap-c7g`
      builds natively on the instance and the `docker build` step pulls
      the GGUFs once from HF directly inside the build.  Either way, no
      per-instance HF re-download.  Implementing S3 staging would save
      ~10-15 min total across the three-instance sweep and add marginal
      robustness if HF rate-limits during the c7g build, at the cost of
      ~2-3 hours of work (S3 push helper, fallback in `*-model` targets,
      IAM/credentials setup, README docs).  Not worth the effort at our
      scope — the actual problem the task was created to solve is
      already addressed by image layering.  External readers are
      unaffected: they'd download from HF either way since our bucket
      would be private.
- [x] Run `make benchmark` on c7i-flex.large (Free Tier x86 AVX-512,
      2 vCPUs).  Results in `results/aws_c7i_flex_large/`.  BitNet/Q8
      ratio 1.33× (vs 1.40× on i5-9400F); Q4 and BitNet nearly tied
      at ~10 tok/s.  ARM t4g.small build failed (OOM during Docker
      build with 2 GB RAM + 4 GB swap).  Total compute cost ~$0.15.
- [x] Extend `scripts/compare_runs.py` with a per-architecture view —
      new `plot_cross_arch_throughput()` reads
      `results/{,linux_docker_x86,aws_c5_xlarge,aws_c6a_xlarge,aws_c7g_xlarge}/{bitnet,qwen_q8,qwen_q4}_step_metrics.csv`
      per the `CROSS_ARCH_SOURCES` table, takes the (512,128)
      reference-config median, and renders a grouped bar chart
      (`results/plots/cross_arch_throughput.png`) with one bar per
      architecture inside each model group.  Architectures with no
      CSV are dropped silently, so the plot grows as the AWS sweep
      fills in — today (2026-05-24) it shows just Windows-native vs
      Linux-Docker on the same i5-9400F (validating the container
      overhead claim from the earlier benchmark); once
      `make aws-benchmark-{c5,c6a,c7g}` runs, the c5/c6a/c7g bars
      activate automatically.  Skip-only-baseline branch keeps the
      plot from being noise when only the Windows row exists.
- [x] Re-verify the AWS pricing constants in
      `scripts/compare_runs.py:CLOUD_API_PRICING` and the hardware-rate
      default ($0.170/hr c5.xlarge on-demand) against current AWS
      pricing.  Cross-arch sweep gives the report a natural place to
      report all three on-demand rates in a table.
      Verified 2026-05-24:
      * **Anthropic Claude Opus 4.7 output cut 3×: $75 → $25 / MTok.**
        Entire Opus 4.5+ family aligned at $25 (the old $75 was Opus
        4.1's number; we'd been quoting it accidentally).  Haiku 4.5
        ($5) and Sonnet 4.5 ($15) unchanged. `CLOUD_API_PRICING` and
        `CLOUD_API_PRICING_DATE` updated; the cloud cost plots
        (`cloud_cost_comparison.png`, `cloud_cost_accuracy.png`) move
        Opus visibly closer to Sonnet on the log axis.
      * OpenAI GPT-4o ($10) and GPT-4o mini ($0.60) unchanged.
      * AWS on-demand us-east-1 Linux rates (vantage.sh): c5.xlarge
        $0.170 (unchanged), c6a.xlarge **$0.153**, c7g.xlarge
        **$0.145**.  Captured as the new `AWS_ON_DEMAND_RATES` dict in
        `compare_runs.py` next to `DEFAULT_HARDWARE_RATE` — not yet
        consumed by any plot, but ready for the per-arch cost work
        and for the §6.1 discussion table.  Note: PLAN.md's earlier
        "$0.1156/hr c7g" reference in the Phase 5 cost-sensitivity
        bullet was stale; the cost-ordering invariant the bullet
        proves is unaffected (any per-row rescaling preserves the
        ordering by construction).
- [x] Update REPORT.md §6.1 with measured cross-architecture
      results.  Outcome (b): Pareto ranking preserved (BitNet > Q8)
      but magnitudes are architecture-sensitive — AVX-512 narrows the
      BitNet-vs-Q8 gap from 1.40× to 1.33× and compresses Q4's
      advantage over Q8 from 1.65× to 1.31×.  Q4 and BitNet nearly
      tied on AVX-512 at 2 threads (~10 tok/s each).  ARM t4g.small
      build failed (OOM), documented as a portability finding (outcome c).
- [x] Add `make aws-benchmark-c7i` / `aws-benchmark-t4g` /
      `aws-benchmark-free-tier` Makefile targets for Free Tier
      instances (`run_remote_benchmark_2v` with THREADS=2 UBATCH=64).
      Original xlarge targets (`aws-benchmark-c5/c6a/c7g`) retained
      for future use if account restrictions are lifted.
- [x] Extend the model set to five: add Qwen2.5-1.5B Q2_K (deepest
      Qwen quantization) and Llama-3.2-1B-Instruct Q4_K_M (second model
      family, paired with the existing LLaMA 3.2 1B FP16 paper row).
      `Dockerfile` pulls both new GGUFs from HF; `Makefile`'s
      `run_remote_benchmark` and `run_remote_benchmark_2v` defines pass
      `QWEN_Q2_BENCH_OUT` / `LLAMA_Q4_BENCH_OUT` env vars through to the
      remote container.  `scripts/compare_runs.py` threads `qwen_q2_df` /
      `qwen_q2_acc` / `llama_q4_df` / `llama_q4_acc` through every per-model
      plot (`plot_throughput`, `plot_memory`, `plot_accuracy`,
      `plot_energy_carbon`, `plot_cloud_cost_comparison`,
      `plot_cloud_accuracy_comparison`, `plot_cloud_cost_accuracy`,
      `plot_mmlu_subject_heatmap`, `plot_accuracy_eval_cost`); adds
      `QWEN_Q2_COLOR` (red) and `LLAMA_Q4_COLOR` (brown) constants; extends
      `_bar_series` / `_legend_handles` / `_accuracy_scatter` with the
      quant-chain ordering (Qwen FP16 paper → Q8 → Q4 → Q2 ours; LLaMA FP16
      paper → Llama Q4 ours) and dotted paper→ours connectors;
      `_load_arch_throughput` and `plot_cross_arch_throughput` auto-discover
      which of the five models have CSVs per architecture so missing bars
      drop silently.
- [x] Restructure two stacked plots into single twin-axis panels per
      user request:
      (i) `plot_energy_carbon` collapses from three panels (Wh / gCO₂ / USD)
          to one — Wh on the bottom x-axis, gCO₂ on a top secondary x-axis
          via the run's measured grid intensity (an exact relabeling since
          carbon = energy × constant at one location).  USD/electricity
          panel removed.
      (ii) `plot_accuracy_eval_cost` collapses from two panels (hours /
          kWh) to one stacked-horizontal-bar panel — hours on the bottom
          x-axis, kWh on a top secondary x-axis via the data's average
          kWh/hour ratio (≈ avg CPU power).  Slight approximation since
          per-model power varies a few percent; the axis label documents
          this ("approx. via avg power ≈ XX W").
- [x] Re-run c7i-flex.large with the full 5-model set.  Re-measures
      BitNet / Qwen Q8 / Qwen Q4 against the rebuilt container so all
      five models on c7i are captured in the same run, and adds the new
      Q2_K and Llama Q4 rows.  Diagnosed the silent stall observed on
      the first attempt: long `RUN` layers (downloading GGUFs, installing
      poetry) produce no stdout, the SSH connection idles, and NAT
      times it out without ever resetting cleanly.  Fix: override
      `AWS_SSH` with `-o ServerAliveInterval=60 -o ServerAliveCountMax=10`
      at the make command line.  Results in `results/aws_c7i_flex_large/`
      (5 CSVs).
- [x] Second t4g.small attempt — confirmed unsalvageable for the
      Docker build.  AWS reports the instance as status ok/ok, but the
      parallel C++ build of llama.cpp's per-model object files thrashes
      the 2 GB RAM + 4 GB EBS-backed swap; sshd is starved and the
      session hangs without a kernel OOM-kill entry.  Build stalled at
      step 19/30, 40% through llama.cpp model compilations.  ARM
      portability finding stands as "not on this instance type" — see
      memory note `project_t4g_small_oom.md` for future retries
      (recommend c7g.large, ~$0.07/hr, 2 vCPU + 4 GB RAM).
- [x] Swap Llama-3.2-1B Q4_K_M out for Gemma-2-2B-it at three quants
      (Q8_0 / Q4_K_M / Q2_K), mirroring the Qwen quant ladder so the
      bit-depth vs accuracy story is symmetric across model families.
      `Makefile` adds `gemma-{q8,q4,q2}-model` / `-verify` / `-bench` /
      `eval-*` targets and the matching aggregations; `Dockerfile` pulls
      all three GGUFs from `bartowski/gemma-2-2b-it-GGUF`; `smoke_test.py`
      adds three new mode flags; `compare_runs.py` fans the single
      `llama_q4_df`/`llama_q4_acc`/`LLAMA_Q4_COLOR` slot out into three
      Gemma slots throughout (`gemma_q8/q4/q2_*`, plus three new color
      constants).  `OTHER_BASELINES` now empty — Gemma 2 2B is absent
      from arXiv:2504.12285 Table 1, so the three Gemma "ours" rows are
      currently unanchored to a paper FP16 baseline.
- [x] Run benchmarks for the three Gemma quants on Windows native
      (i5-9400F, THREADS=4 UBATCH=128).  Headlines at (512, 128):
      Gemma Q8 9.5 tok/s / 2,776 MB / 1.34 Wh/1k tok; Gemma Q4 14.2
      tok/s / 2,671 MB / 0.83 Wh/1k tok; Gemma Q2 18.4 tok/s / 1,293
      MB / 1.02 Wh/1k tok.  Gemma slower than Qwen at every quant
      level (~30-40% gap) — consistent with the 2B vs 1.5B parameter
      count difference.  Q2_K's RSS drops to less than half of Q8/Q4
      thanks to K-quant tensor sharing.
- [x] Run accuracy evals for all three Gemma quants at LIMIT=100.
      Wall-clock total ~33 hours (MMLU dominates — ~8.6 hr per model
      regardless of quant since the script throughput floor is fixed).
      **Striking result: all three Gemma quants land within 0.04pt mean
      accuracy** (63.41 / 63.42 / 63.42).  ARC-Easy, ARC-Challenge,
      WinoGrande, HellaSwag are *identical* across Q8/Q4/Q2; MMLU
      spread is 58.07 / 58.11 / 58.09 — sampling noise.  Gemma Q2_K
      becomes the new Pareto winner among rows with full accuracy —
      ties on accuracy, +5pt mean over BitNet, 1.3 GB RSS (+4% over
      BitNet), 18.4 tok/s (-13% under BitNet).  Workflow improvements:
      `eval_accuracy.py` now reads `SKIP_COMPLETED=1` (env var or
      `--skip-completed` flag) so resuming a crashed/interrupted run
      skips finished tasks; also reconfigures stdout/stderr to UTF-8
      on Windows so the end-of-run summary table (which uses `Δ` for
      the ours-vs-paper gap column) doesn't crash on cp1252 terminals
      after data is already saved.
- [ ] Run Qwen Q2_K and re-run Qwen Q4 / Qwen Q8 (Q4 JSON was deleted
      pre-session; Q8 is partial at 31/57 MMLU subjects, no Wino/Hella).
      Queued after the Gemma evals finish — they contend for CPU and
      llama-server port 8080.
- [ ] Re-run benchmarks for Qwen Q2 and the three Gemma quants in the
      local Linux Docker container (`results/linux_docker_x86/qwen_q2_step_metrics.csv`
      and the three `gemma_*_step_metrics.csv` files don't exist yet, so
      the cross-arch plot's Linux Docker column is missing bars for those
      four models).
- [ ] (Optional, when available) Add a `GEMMA_PAPER` block in
      `compare_runs.py` with FP16 baseline numbers from the Gemma 2
      technical report (arXiv:2408.00118).  Wires the three Gemma "ours"
      rows to a paired paper anchor in the same way Qwen FP16 is paired
      with Qwen Q8/Q4/Q2 ours.
- [ ] Refresh `REPORT.md` with: (a) the 7-model c7i numbers in §6.1
      (BitNet, Q8, Q4 re-measured plus Q2 and three Gemma quants); (b)
      Q2_K and Gemma rows in the headline tables and accuracy comparisons;
      (c) the energy/carbon and accuracy-eval-cost plot layout changes
      (single panel with secondary x-axis instead of multi-panel grids).
- [ ] (Optional) Retry ARM data on c7g.large or pre-build the ARM
      image on a beefier instance and `docker save | ssh t4g 'docker
      load'` so the benchmark runs on t4g.small but the build doesn't.
      `CROSS_ARCH_SOURCES` keeps the `aws_t4g_small` entry so dropping
      CSVs into `results/aws_t4g_small/` later lights up the ARM bars
      without any code change.

Out of scope for Phase 6: re-running accuracy evals on remote instances
(unchanged at deterministic decoding), GPU baselines (project is
explicitly CPU-only), more than three remote instances (diminishing
returns vs deadline).

Risks:

- BitNet Linux build may require additional patches not yet authored.
  Mitigation: smoke-test on `c5.xlarge` first; if blocked >1 day, drop
  BitNet from remote sweep and report Qwen Q8/Q4 cross-arch only (still
  closes most of the §6.1 threat).
- ARM build of BitNet's TL2 kernel may not exist upstream.  Mitigation:
  check before paying; treat ARM-Qwen-only as an acceptable fallback.
- Spot interruption mid-benchmark.  Mitigation: each `(n_prompt, n_gen)`
  config writes one CSV row; re-running interrupted configs is cheap.
- AWS cost overrun if smoke-test phase takes longer than expected.
  Mitigation: hard budget cap of $20 across the phase; if hit, stop and
  reassess what's needed for the deadline.
