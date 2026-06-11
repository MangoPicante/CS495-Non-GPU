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

Reference workload: `n_prompt=512`, `n_gen=128`, 4 threads on an Intel
i5-9400F. Full details (methodology, threats to validity, plots) in
[`REPORT.md`](REPORT.md).

| Metric | BitNet | Qwen Q8_0 | Qwen Q4_K_M | Qwen Q2_K | Gemma Q8_0 | Gemma Q4_K_M | Gemma Q2_K |
|---|---:|---:|---:|---:|---:|---:|---:|
| Throughput (tok/s) | 21.2 | 15.1 | 24.9 | **32.5** | 9.5 | 14.2 | 18.4 |
| Peak RSS (MB) | 1,247 | 1,667 | 1,632 | **745** | 2,776 | 2,671 | 1,293 |
| Mean accuracy across 5 tasks (%) | 61.74 | 61.10 | 59.45 | pending | 63.41 | 63.42 | **63.42** |
| Cost: AWS c5.xlarge proxy ($/1k tok) | 0.00223 | 0.00313 | 0.00190 | **0.00145** | 0.00497 | 0.00332 | 0.00256 |
| Cost: local electricity ($/1k tok) | **0.000131** | 0.000224 | 0.000132 | 0.000162 | 0.000334 | 0.000208 | 0.000255 |

The original three models (BitNet, Q8, Q4) traced a clean speed/accuracy
Pareto where **BitNet matched Q8's accuracy at near-Q4 speed** while
winning memory and reasoning tasks (WinoGrande +9–12pt over Qwen). The
recent Q2 / Gemma expansion adds three findings:

- **Qwen Q2_K** is the throughput / memory / AWS-rental-cost leader by
  large margins (~30% over Q4 on speed; ~40% over BitNet on RSS) — pending
  the accuracy eval that will say whether the deeper quantization holds up.
- **All three Gemma-2-2B-it quants tie at 63.41–63.42% mean accuracy**
  at LIMIT=100 — a striking *full-ladder quantization invariance*
  result.  Q8, Q4, and Q2 score **identically** on ARC-Easy (73.0),
  ARC-Challenge (52.0), WinoGrande (70.0), and HellaSwag (64.0); MMLU
  spread is 58.07 / 58.11 / 58.09 — within 0.04pt across 5,700
  questions.  +1.7pt over BitNet, +2.3pt over Qwen Q8.  The Q2_K
  result is the surprise: ~2.6 bits per weight on Gemma 2 2B preserves
  the full task headline at this sample size.
- **Gemma Q2_K is the new Pareto winner among rows with complete
  accuracy.**  Ties the Gemma family on accuracy (63.42%), drops RSS to
  1,293 MB (under half of Q8/Q4, +4% over BitNet), and runs at 18.4
  tok/s (~2× Gemma Q8, 87% of BitNet).  At AWS-proxy $/1k-tok cost,
  Q2_K's $0.00256 sits between Qwen Q4 and Qwen Q8 — about 16% above
  BitNet's $0.00223 but with +1.7pt mean-accuracy advantage.  If the
  full-LIMIT eval confirms this invariance, Gemma Q2_K becomes the
  default recommendation for accuracy-priority self-hosting at this
  size class.
- **The cost gradient is real but small.**  Gemma is the slowest model
  in the project at every quant (9.5 / 14.2 / 18.4 tok/s vs Qwen 15.1
  / 24.9 / 32.5) and the heaviest at Q8/Q4 (2.7 GB RSS).  Gemma's 2B
  parameter count vs Qwen's 1.5B explains the consistent throughput
  gap.  Q2_K's K-quant tensor sharing collapses the memory penalty
  even on the 2B-parameter base.

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
make benchmark             # All seven models at THREADS=4

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
poetry run python scripts/compare_runs.py --hardware-rate 0.05   # spot c5.xlarge
poetry run python scripts/compare_runs.py --hardware-rate 0.1156 # ARM Graviton c7g.xlarge
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

External (not in this repo): `../Models/BitNet/` and `../Models/Qwen/`,
created and populated by the `make bitnet-setup` / `qwen-q8-setup` targets.

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
