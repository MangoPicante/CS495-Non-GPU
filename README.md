# Non-GPU LLM Training: Efficient Transformer Training with BitNet 📌

## Project theme 📖
Modern LLM training demands expensive GPU clusters, limiting access and driving up energy costs. This project investigates whether 1-bit weight quantization (BitNet) makes transformer pretraining viable on commodity CPUs — trading a modest accuracy penalty for dramatic reductions in memory, cost, and carbon footprint.

## Objectives 🎯
### Primary research questions

- Can 1-bit transformers scale to meaningful language tasks?
- How much accuracy is lost under extreme quantization vs. FP16/FP32 baselines?
- Are CPUs a viable substrate for LLM pretraining with BitNet?
- How does low-precision affect convergence stability?
- What workloads benefit most from low-precision training?

### Methodology

| Phase | Description |
|-------|-------------|
| 1 — Repository study | Analyze BitNet architecture, quantization strategy, and weight binarization |
| 2 — Baseline training | Train small transformer with FP16/FP32; record time, memory, accuracy |
| 3 — BitNet implementation | Train equivalent model with 1-bit weights on CPU or low-resource hardware |
| 4 — Cost comparison | Benchmark hardware cost, training time, energy, memory, and accuracy |
| 5 — Optimization | Explore hybrid precision, quantization schedules, and scaling laws |

### Deliverables

- **Research report** — quantization theory, BitNet architecture, benchmark results, environmental impact
- **Training pipelines** — baseline + BitNet scripts, configs, and reproducibility instructions
- **Benchmark dashboard** — loss curves, training time, memory, and accuracy comparisons
- **Systems cost analysis** — GPU vs. CPU dollar cost, energy usage, and carbon footprint proxy
- **Final presentation**

### Evaluation metrics

- Training time, memory footprint, energy consumption (GPU vs. CPU)
- Model accuracy and perplexity under 1-bit vs. FP16/FP32 precision
- Cost-accuracy trade-off and carbon footprint proxy

🧰 Tools / Technologies (e.g., Python, PyTorch, SQL, etc.)

🚀 How to Run (basic instructions)

👥 Sean Michael, Prof. Dr. Pedro Albuquerque

📅 Timeline (optional but recommended)

Any academic, research, or commercial usage must cite the original repository and authors.
# Non-GPU LLM Training with BitNet

> Capstone project exploring 1-bit transformer training on CPU-class hardware.

![Data Science Capstone](https://img.shields.io/badge/Data%20Science-Capstone-blue)
![BitNet](https://img.shields.io/badge/BitNet-Quantization-teal)
![CPU Training](https://img.shields.io/badge/CPU-Training-orange)

---

## Project theme

Modern LLM training demands expensive GPU clusters, limiting access and driving up energy costs. This project investigates whether **1-bit weight quantization (BitNet)** makes transformer pretraining viable on commodity CPUs — trading a modest accuracy penalty for dramatic reductions in memory, cost, and carbon footprint.

---

## Goals





---



---



---

Data source: [microsoft/BitNet](https://github.com/microsoft/BitNet) · Standard NLP benchmarks for perplexity and accuracy evaluation
