# CS495 Capstone — Non-GPU LLM Inference

Benchmarking BitNet b1.58 2B4T and Qwen2.5-1.5B on commodity CPU hardware.

This project independently reproduces and extends the inference-efficiency
claims in [arXiv:2504.12285](https://arxiv.org/abs/2504.12285) (BitNet b1.58
2B4T) on consumer x86 CPU. It compares three locally measured models —
**BitNet b1.58 2B4T (i2_s)**, **Qwen2.5-1.5B-Instruct (Q8_0)**, and
**Qwen2.5-1.5B-Instruct (Q4_K_M)** — against each other and against five
published FP16 baselines from the paper's Table 1 (LLaMA 3.2 1B, Gemma-3 1B,
Qwen2.5 1.5B, SmolLM2 1.7B, MiniCPM 2B).

## Headline finding

Reference workload: `n_prompt=512`, `n_gen=128`, 4 threads on an Intel
i5-9400F. Full details (methodology, threats to validity, plots) in
[`FINAL_REPORT.md`](FINAL_REPORT.md).

| Metric | BitNet | Qwen Q8_0 | Qwen Q4_K_M |
|---|---:|---:|---:|
| Throughput (tok/s) | 21.2 | 15.1 | **24.9** |
| Peak RSS (MB) | **1,247** | 1,667 | 1,632 |
| Mean accuracy across 5 tasks (%) | **61.74** | 61.10 | 59.45 |
| Cost: AWS c5.xlarge proxy ($/1k tok) | 0.00223 | 0.00313 | **0.00190** |
| Cost: local electricity ($/1k tok) | **0.000131** | 0.000224 | 0.000132 |

The three models trace a clean speed/accuracy Pareto: **Q4** wins raw speed
and AWS-rental cost, **BitNet** wins accuracy, memory, and reasoning tasks
(WinoGrande +9–12pt), **Q8** has no obvious operational niche unless its
MMLU edge over Q4 (+1pt) is binding.

All three self-hosted models beat every commercial cloud API tier on $/1k
tokens at this size class — 4.6× cheaper than the cheapest API tier
(GPT-4o mini) on local electricity, 573× cheaper than Claude Opus 4.7 —
with the strong caveat that the comparison only holds for workloads where
a 2B-parameter model's capability is sufficient.

---

## Repository overview

| File / dir | What's in it |
|---|---|
| [`Makefile`](Makefile) | Single source of truth for all reproducible commands. Run `make help` for the full list. |
| [`PLAN.md`](PLAN.md) | Canonical project reference: dependencies, layout, reproducibility, per-script implementation, phased task list. |
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | Phase 4/5 dashboard with all measurements, plots, discussion, and threats to validity. |
| [`BITNET_SUMMARY.md`](BITNET_SUMMARY.md) | BitNet model card and quantization notes (absmean, STE, TL2 kernel). |
| [`QWEN_SUMMARY.md`](QWEN_SUMMARY.md) | Qwen model card and notes on the upstream-vs-fork `llama.cpp` differences. |
| [`REPORT.md`](REPORT.md) | Phase 3 BitNet-only sanity check (predates the Qwen comparison). |
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
- `../Models/Qwen/llama.cpp/` — `ggml-org/llama.cpp` at commit `1e5ad35d` (upstream, used for both Qwen variants)

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

# Clone + build upstream llama.cpp for Qwen, download both quants
make qwen-setup            # builds + downloads Q8_0
make qwen-verify
make qwen-q4-model         # add the Q4_K_M GGUF
make qwen-q4-verify

# Verify the full pipeline end-to-end
make smoke-test            # ~5 minutes; three models, three prompts each,
                           # then a 5-sample accuracy sweep per model
```

For the full end-to-end command list (benchmarks, accuracy, plots,
sensitivity sweeps), see [`PLAN.md`](PLAN.md#reproducibility--end-to-end).

---

## Running benchmarks and evaluations

All commands run from the repo root. The most common targets:

```bash
# Inference throughput / memory / energy (writes results/*_step_metrics.csv)
make benchmark             # All three models at THREADS=4

# Accuracy evaluation (writes results/accuracy_results_*.json)
make eval-accuracy         # All three models, all five tasks, LIMIT=500
make eval-accuracy LIMIT=100   # Quick sanity check at smaller sample size

# Per-task evals (ARC-Easy/Challenge, WinoGrande, HellaSwag, MMLU 5-shot)
make eval-mmlu-bitnet
make eval-arc-easy-qwen
# ...see make help for the full list

# Regenerate the comparison table and all plots
make plots                 # → results/comparison_table.csv + results/plots/*.png

# Phase 5 sensitivity sweeps
make benchmark-threads             # Thread-count sweep (1/2/4/6 cores)
make benchmark-qwen-on-bitnet-fork # Cross-stack sensitivity check
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
├── FINAL_REPORT.md              # Phase 4/5 dashboard
├── REPORT.md                    # Phase 3 BitNet-only sanity check
├── BITNET_SUMMARY.md            # BitNet model card
├── QWEN_SUMMARY.md              # Qwen model card
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
created and populated by the `make bitnet-setup` / `qwen-setup` targets.

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
