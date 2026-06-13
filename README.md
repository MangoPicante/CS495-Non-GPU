# CS495 Capstone — Non-GPU LLM Inference

Benchmarking BitNet b1.58 2B4T and Qwen2.5-1.5B on commodity CPU hardware.

This project independently reproduces and extends the inference-efficiency
claims in [arXiv:2504.12285](https://arxiv.org/abs/2504.12285) (BitNet b1.58
2B4T) on consumer x86 CPU. It compares **seven locally measured models** —
**BitNet b1.58 2B4T (i2_s)**, **Qwen2.5-1.5B-Instruct (Q8_0, Q4_K_M, Q2_K)**,
and **Gemma-2-2B-it (Q8_0, Q4_K_M, Q2_K)** — against each other and against
the Qwen2.5 1.5B FP16 paper baseline. Gemma 2 2B does not appear in
arXiv:2504.12285 Table 1, so the three Gemma rows are reported "ours only"
until a Gemma 2 paper baseline is wired in. Earlier FP16-only rows
(LLaMA 3.2 1B, Gemma-3 1B, SmolLM2 1.7B, MiniCPM 2B) were removed during
the Q2 / Gemma expansion because they had no PTQ counterpart on the same
hardware.

## Headline finding

Reference workload: `n_prompt=512`, `n_gen=128`, **2 threads** on an
Intel i5-9400F (matches the AWS Free Tier `c7i-flex.large` cross-arch
condition; see [`REPORT.md`](REPORT.md) for methodology, threats to
validity, and plots).

| Metric | BitNet | Qwen Q8_0 | Qwen Q4_K_M | Qwen Q2_K | Gemma Q8_0 | Gemma Q4_K_M | Gemma Q2_K |
|---|---:|---:|---:|---:|---:|---:|---:|
| Throughput (tok/s) | 17.8 | 13.4 | 17.6 | **20.3** | 8.0 | 9.3 | 11.7 |
| Peak RSS (MB) | 1,240 | 1,659 | 1,624 | **737** | 2,766 | 2,662 | 1,283 |
| Mean accuracy across 5 tasks (%) | 61.74 | **63.58** | 61.86 | 52.21 | 63.41 | 63.28 | 57.99 |
| Cost: AWS c5.xlarge proxy ($/1k tok) | 0.00265 | 0.00353 | 0.00269 | **0.00232** | 0.00591 | 0.00507 | 0.00403 |
| Cost: local electricity ($/1k tok) | **0.000202** | 0.000339 | 0.000217 | 0.000304 | 0.000536 | 0.000386 | 0.000494 |

The original three models (BitNet, Q8, Q4) trace a speed/accuracy Pareto
where **BitNet sits at Qwen-Q4 speed** (17.8 vs 17.6 tok/s, essentially
tied) with accuracy between Q4 and Q8 (61.74% vs Q4's 61.86% and Q8's
63.58%) while winning every reasoning task — WinoGrande +3 to +16pt
over the three Qwen quants. The Q2 / Gemma expansion adds three
findings:

- **Qwen Q2_K** is the throughput / memory / AWS-rental-cost leader,
  but no longer by huge margins after the threads=2 standardization
  (~16% over Q4 on speed at this condition; ~40% over BitNet on RSS).
  The accuracy eval since shows Q2 *collapses* 11.4pt vs Q8 — the
  speed/cost win is real but comes with a steep capability cost.
- **Gemma 2 2B is dramatically more quantization-robust than Qwen 1.5B.**
  Gemma Q4_K_M matches Gemma Q8_0 within sampling noise (63.28% vs
  63.41% mean) — Q4 is essentially free on Gemma. Qwen Q4_K_M loses
  1.72pt vs Qwen Q8_0 (61.86% vs 63.58%) — a real but small tax.
  At Q2_K the gap widens further: Gemma Q2 loses 5.4pt from Q8
  (57.99%); Qwen Q2 *collapses* by 11.4pt (52.21%) — falling below
  BitNet and approaching random-guess territory on the hardest tasks.
  Larger parameter count + family-specific K-quant calibration both
  contribute.
- **Accuracy-per-dollar leader: BitNet.** 61.74% mean at $0.00265 AWS
  proxy / $0.000202 local electricity — both the highest accuracy/cost
  ratio in the table and the cheapest electricity row outright.  Qwen
  Q8 wins absolute accuracy (63.58%), but at 33% higher AWS cost and
  68% higher electricity cost than BitNet for +1.84pt — a steep
  efficiency tradeoff.  Gemma Q4_K_M (previously the Pareto winner at
  threads=4) is now dominated by Qwen Q8 at this condition: Qwen Q8 is
  cheaper *and* more accurate.
- **Cost picture is uneven across families.** Gemma is the slowest
  model at every quant (8.0 / 9.3 / 11.7 tok/s vs Qwen 13.4 / 17.6 /
  20.3) and the heaviest at Q8/Q4 (~2.7 GB RSS, 2.2× BitNet) — Gemma's
  2B parameter count vs Qwen's 1.5B explains the consistent throughput
  / memory gap.  Q2_K's K-quant tensor sharing collapses Gemma's
  memory penalty (1.3 GB, within 4% of BitNet) — but at the 5.4pt
  accuracy cost noted above.

The four self-hosted rows with full accuracy data (BitNet, Qwen Q8, Qwen
Q4, Gemma Q8) all beat every commercial cloud API tier on $/1k tokens at
this size class — with the strong caveat that the comparison only holds
for workloads where a 1–2B-parameter model's capability is sufficient.

---

## Repository overview

| File / dir | What's in it |
|---|---|
| [`Makefile`](Makefile) | Single source of truth for all reproducible commands. Run `make help` for the full list. |
| [`PLAN.md`](PLAN.md) | Canonical project reference: dependencies, layout, reproducibility, per-script implementation, phased task list. |
| [`REPORT.md`](REPORT.md) | Canonical capstone report: dashboard, energy/cost analysis, threats to validity, and three appendices (A: Phase 3 sanity check; B: BitNet model card with absmean/STE/TL2; C: Qwen model card with Q8_0/Q4_K_M formats and upstream-vs-fork inference-stack notes). |
| `scripts/` | `metrics_tracker.py` (`llama-bench` wrapper), `eval_accuracy.py` (lm-eval-harness–style accuracy via `llama-server`), `compare_runs.py` (table + plot generation), `smoke_test.py`. |
| `results/` | Bench CSVs, accuracy JSONs, `comparison_table.csv`, and ~20 plots in `plots/`. |
| `patches/` | Three patches BitNet needs to build on Windows + ClangCL. |

---

## Prerequisites

- Python 3.11 (numpy 1.26 has no wheels for 3.13+)
- Poetry (≥ 1.6)
- CMake (≥ 3.22)
- Visual Studio 2022+ with the "Desktop development with C++" workload
- ClangCL (BitNet only — VS components `Microsoft.VisualStudio.Component.VC.Llvm.Clang` + `...ClangToolset`)
- `hf` CLI (`pip install huggingface_hub[cli]`)
- ~5 GB free disk space for model weights

External dependencies (cloned and built into sibling directories by the
`make ...-setup` targets):

- `../Models/BitNet/` — `microsoft/BitNet` at commit `01eb4157` (the inference fork with the TL2 ternary-lookup kernel)
- `../Models/Qwen/llama.cpp/` — `ggml-org/llama.cpp` at commit `1e5ad35d` (upstream, used for all three Qwen variants and all three Gemma variants)
- `../Models/Gemma/` — Gemma-2-2B-it Q8_0 / Q4_K_M / Q2_K GGUFs (community quants: Q8 + Q4 from bartowski, Q2_K from second-state — bartowski's Gemma-2-2B ladder bottoms out at Q3_K_L, so Q2_K is sourced separately; Google does not ship official GGUFs)
- `../Models/Cloud/` — populated by `make system-cards-cloud` with the
  GPT-4o / Claude 4.5 / Claude Opus 4.7 system-card PDFs used by the
  cost-vs-cloud comparison (§3.9 of `REPORT.md`).

Reference hardware: Intel Core i5-9400F (6 cores, no SMT), 16 GB RAM,
Windows 11.

---

## Quick start

```bash
git clone https://github.com/MangoPicante/CS495-Non-GPU.git
cd CS495-Non-GPU

# Python environment
make install               # Poetry: install runtime deps
make check-deps            # Verify cmake, Python 3.11, git, ClangCL

# Build BitNet's llama.cpp fork (ClangCL) and download the GGUF
make bitnet-setup
make bitnet-model
make bitnet-verify

# Clone + build upstream llama.cpp for Qwen, download all three Qwen quants
make qwen-q8-setup         # builds + downloads Q8_0
make qwen-q8-verify
make qwen-q4-model         # add the Q4_K_M GGUF
make qwen-q4-verify
make qwen-q2-model         # add the Q2_K GGUF (deepest Qwen quantization)
make qwen-q2-verify

# Gemma-2-2B-it (second model family — three quants, all use the same
# upstream llama.cpp build from qwen-q8-setup, no separate build)
make gemma-q8-model
make gemma-q8-verify
make gemma-q4-model
make gemma-q4-verify
make gemma-q2-model
make gemma-q2-verify

# Optional: download the paper / system-card PDFs into the Models dirs
make system-cards          # arXiv reports for BitNet & Qwen, plus the
                           # GPT-4o / Claude 4.5 / Opus 4.7 system cards
                           # into ../Models/Cloud/

# Verify the full pipeline end-to-end
make smoke-test            # All seven models: three prompts each, then a
                           # 5-sample 5-shot accuracy sweep per model
                           # (~55-70 min total at 8-10 min/model)
```

For the full end-to-end command list (benchmarks, accuracy, plots,
sensitivity sweeps), see [`PLAN.md`](PLAN.md#reproducibility--end-to-end).

---

## Running benchmarks and evaluations

All commands run from the repo root. The most common targets:

```bash
# Inference throughput / memory / energy (writes results/*_step_metrics.csv)
make benchmark             # All seven models. Canonical reported numbers
                           # were measured at THREADS=2 UBATCH=64 (matches
                           # accuracy evals + AWS Free Tier c7i-flex.large);
                           # override via `make benchmark THREADS=2 UBATCH=64`.

# Accuracy evaluation (writes results/accuracy_results_*.json)
make eval-accuracy         # All seven models, all five tasks, LIMIT=500
make eval-accuracy LIMIT=100   # Quick sanity check at smaller sample size
make eval-accuracy SKIP_COMPLETED=1   # Skip tasks already complete in the per-model JSON

# Per-task evals (ARC-Easy/Challenge, WinoGrande, HellaSwag, MMLU 5-shot)
make eval-mmlu-bitnet
make eval-arc-easy-qwen-q8
make eval-arc-easy-qwen-q4
make eval-mmlu-qwen-q2
make eval-mmlu-gemma-q8
make eval-mmlu-gemma-q4
make eval-mmlu-gemma-q2
# ...see make help for the full list

# Regenerate the comparison table and all ~13 plots (incl. cloud comparison)
make plots                 # → results/comparison_table.csv + results/plots/*.png
                           # Plots include cloud_cost_comparison.png,
                           # cloud_accuracy_comparison.png, and
                           # cloud_cost_accuracy.png — the cost vs cloud-API
                           # tier headline lives there.

# Override the hardware rate to test cost-model sensitivity (Phase 5):
poetry run python scripts/compare_runs.py --hardware-rate 0.05         # spot c5.xlarge
poetry run python scripts/compare_runs.py --hardware-rate 0.085        # c7i-flex.large (actually measured arch)
# Ordering of comparison_table.csv rows by $/1k tok is invariant across rates.

# Phase 5 sensitivity sweeps
make benchmark-threads                # Thread-count sweep (1/2/4/6 cores)
make benchmark-qwen-q8-on-bitnet-fork # Cross-stack sensitivity check
```

`make help` lists every target with one-line descriptions.

---

## Project structure

```
CS495-Non-GPU/
├── Makefile                     # All reproducibility entry points
├── pyproject.toml               # Poetry environment
├── README.md                    # This file
├── PLAN.md                      # Canonical project reference
├── REPORT.md                    # Canonical capstone report (with appendices A: Phase 3 sanity check, B: BitNet model card, C: Qwen model card)
├── patches/                     # ClangCL build patches for BitNet
├── scripts/
│   ├── metrics_tracker.py       # llama-bench wrapper
│   ├── eval_accuracy.py         # llama-server lm-eval-style accuracy
│   ├── compare_runs.py          # comparison_table.csv + all plots
│   └── smoke_test.py            # End-to-end smoke test
└── results/
    ├── *_step_metrics.csv       # bench results per model
    ├── *_thread_sweep.csv       # Phase 5 thread-count sweep data
    ├── accuracy_results_*.json
    ├── comparison_table.csv     # Aggregated paper + ours
    └── plots/                   # PNGs generated by compare_runs.py
```

External (not in this repo): `../Models/BitNet/`, `../Models/Qwen/`, and
`../Models/Gemma/` — created and populated by the `make bitnet-setup`,
`qwen-q8-setup`, and `gemma-q*-model` targets.

---

## Authors

Sean Michael · Prof. Dr. Pedro Albuquerque

> Any academic, research, or commercial use of this work must cite the
> original repository and authors.  See [`CITATION.cff`](CITATION.cff)
> for a machine-readable citation.

## Key papers

- **BitNet b1.58 2B4T technical report** — Wang et al. (2025), [arXiv:2504.12285](https://arxiv.org/abs/2504.12285)
- **BitNet b1.58 1.58-bit foundations** — Ma et al. (2024), [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- **Qwen2.5 technical report** — Qwen Team (2024), [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

## License

See [`LICENSE`](LICENSE). Data sources:
[microsoft/BitNet](https://github.com/microsoft/BitNet),
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp),
[Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF),
and the standard NLP benchmarks (ARC, HellaSwag, WinoGrande, MMLU).
