# Non-GPU LLM Training: Efficient Transformer Training with BitNet 📌

![Data Science Capstone](https://img.shields.io/badge/Data%20Science-Capstone-blue)
![BitNet](https://img.shields.io/badge/BitNet-Quantization-teal)
![CPU Training](https://img.shields.io/badge/CPU-Training-orange)

---

## Project Theme 📖

Modern LLM training demands expensive GPU clusters, limiting access and driving up energy costs. This project investigates whether **1-bit weight quantization (BitNet)** makes transformer pretraining viable on commodity CPUs — trading a modest accuracy penalty for dramatic reductions in memory, cost, and carbon footprint.

---

## Objectives 🎯

### Primary Research Questions

- Can 1-bit transformers scale to meaningful language tasks?
- How much accuracy is lost under extreme quantization vs. FP16/FP32 baselines?
- Are CPUs a viable substrate for LLM pretraining with BitNet?
- How does low-precision affect convergence stability?
- What workloads benefit most from low-precision training?

### Methodology

| Phase | Description |
|-------|-------------|
| 1 — Repository Study | Analyze BitNet architecture, quantization strategy, and weight binarization |
| 2 — Baseline Training | Train small transformer with FP16/FP32; record time, memory, accuracy |
| 3 — BitNet Implementation | Train equivalent model with 1-bit weights on CPU or low-resource hardware |
| 4 — Cost Comparison | Benchmark hardware cost, training time, energy, memory, and accuracy |
| 5 — Optimization | Explore hybrid precision, quantization schedules, and scaling laws |

### Deliverables

- **Research report** — quantization theory, BitNet architecture, benchmark results, environmental impact
- **Training pipelines** — baseline + BitNet scripts, configs, and reproducibility instructions
- **Benchmark dashboard** — loss curves, training time, memory, and accuracy comparisons
- **Systems cost analysis** — GPU vs. CPU dollar cost, energy usage, and carbon footprint proxy
- **Final presentation**

### Evaluation Metrics

| Metric | Tool |
|--------|------|
| Perplexity | WikiText-2 validation set |
| Zero-shot accuracy | lm-evaluation-harness (ARC, HellaSwag, WinoGrande…) |
| Training time | `MetricsTracker` (wall-clock) |
| Memory (CPU/GPU) | `psutil` + `pynvml` |
| Energy / CO₂ | `codecarbon` |
| Throughput | tokens/sec logged per step |

---

## 🧰 Tools & Technologies

- **Languages**: Python 3.11+
- **Inference Runtime**: [microsoft/BitNet (bitnet.cpp)](https://github.com/microsoft/BitNet)
- **Evaluation**: [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **Memory Profiling**: psutil
- **Carbon Tracking**: CodeCarbon
- **Model**: BitNet b1.58 2B4T GGUF (`microsoft/bitnet-b1.58-2B-4T-gguf`)

---

## 🚀 Setup

### Prerequisites

- Python 3.11+
- CMake 3.22+
- Clang 18+ / Visual Studio 2022 with C++ workload (Windows)
- `huggingface-cli` (`pip install huggingface_hub[cli]`)
- ~3 GB free disk space for the model

### 1 — Clone this repo

```bash
git clone https://github.com/SeanMWX/CS495-Non-GPU.git
cd CS495-Non-GPU
```

### 2 — Install Python dependencies

```bash
pip install lm_eval psutil codecarbon
```

### 3 — Set up BitNet (external dependency)

Clone and build `bitnet.cpp` **at the exact commit used in this project** into a sibling directory (`../BitNet`):

```bash
# From the parent of CS495-Non-GPU
git clone https://github.com/microsoft/BitNet.git
cd BitNet
git checkout 01eb415772c342d9f20dc42772f1583ae1e5b102

# Install Python build deps
pip install -r requirements.txt

# Build (Windows — requires Visual Studio 2022 + Clang)
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

> The `setup_env.py` script builds the project **and** downloads the model in one step.
> If you prefer to build and download separately:
> ```bash
> # Build only
> cmake -B build -DCMAKE_BUILD_TYPE=Release
> cmake --build build --config Release
>
> # Download model only
> huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf \
>     --local-dir models/BitNet-b1.58-2B-4T
> ```

### 4 — Verify the model runs

```bash
# From inside the BitNet directory
python run_inference.py \
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "What is 2+2?" \
    -n 32 -t 4
```

You should see a short generated answer followed by timing stats (`~47 ms/token` on a modern laptop CPU with 4 threads).

---

## 📊 Running Benchmarks

All scripts are run from the **repo root** (`CS495-Non-GPU/`).

### Inference latency & memory benchmark

Records latency, throughput, peak RSS, and energy to `results/step_metrics.csv`:

```bash
python scripts/metrics_tracker.py --threads 4 --n-tokens 128 --tag "baseline"
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--threads` | 4 | CPU threads |
| `--n-tokens` | 128 | Tokens to generate |
| `--tag` | `""` | Label stored in CSV |
| `--no-energy` | off | Skip CodeCarbon measurement |
| `--out` | `results/step_metrics.csv` | Output file |

### Accuracy evaluation (lm-evaluation-harness)

Starts the BitNet server, runs lm-eval, then shuts the server down:

```bash
# Quick sanity check — 50 samples per task
python scripts/run_lm_eval.py --tasks arc_easy --limit 50

# Full benchmark (0-shot)
python scripts/run_lm_eval.py --tasks arc_easy,arc_challenge,hellaswag,winogrande

# MMLU (5-shot, as reported in arXiv:2504.12285)
python scripts/run_lm_eval.py --tasks mmlu --num-fewshot 5
```

Results are saved as JSON to `results/lm_eval/`.

---

## Project Structure

```
CS495-Non-GPU/
├── scripts/
│   ├── metrics_tracker.py   # Inference latency / memory / energy → step_metrics.csv
│   ├── run_lm_eval.py       # lm-eval harness wrapper (manages llama-server lifecycle)
│   └── compare_runs.py      # (Phase 4) Generate comparison plots
│
├── results/
│   ├── step_metrics.csv     # Per-run inference benchmark results
│   ├── lm_eval/             # lm-evaluation-harness JSON outputs
│   └── comparison_table.csv # (Phase 4) Aggregated comparison table
│
├── PLAN.md                  # Project plan and task tracking
├── REPORT.md                # Paper annotations, quantization notes, baseline results
└── pyproject.toml
```

> **External dependency:** `../BitNet` — the microsoft/BitNet repo (not tracked in this repo).
> See setup step 3 above for the pinned commit and build instructions.

---

## 📅 Timeline

| Week | Phase |
|------|-------|
| 1–2  | Phase 1 — Repository study & architecture review |
| 3–4  | Phase 2 — Baseline FP16 training & profiling |
| 5–6  | Phase 3 — BitNet b1.58 implementation & CPU training |
| 7–8  | Phase 4 — Cost comparison & benchmark dashboard |
| 9–10 | Phase 5 — Optimization & final report/presentation |

---

## Key Papers

- **BitNet b1.58**: Ma et al. (2024) — *The Era of 1-bit LLMs* — [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- **BitNet**: Wang et al. (2023) — *Scaling 1-bit Transformers* — [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)
- **LLaMA**: Touvron et al. (2023) — architecture reference

---

## 👥 Authors

Sean Michael · Prof. Dr. Pedro Albuquerque

---

> Any academic, research, or commercial usage must cite the original repository and authors.
>
> Data source: [microsoft/BitNet](https://github.com/microsoft/BitNet) · Standard NLP benchmarks for perplexity and accuracy evaluation
