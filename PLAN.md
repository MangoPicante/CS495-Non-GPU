# PLAN.md

## Project Overview

**Title:** Non-GPU LLM Training: Efficient Transformer Training with BitNet
**Author:** Sean Michael
**Date:** April 27, 2026

## Dependencies

| Dependency | Location |
| --- | --- |
| `bitnet.cpp` (built) | `../Models/BitNet` (external — outside this repo) |

### Description

An independent reproduction and extension of Microsoft's published inference benchmarks for BitNet b1.58 2B4T ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285)). Rather than training models from scratch, this project runs the BitNet b1.58 2B4T GGUF model locally via `bitnet.cpp` and benchmarks inference latency, throughput, memory footprint, and energy consumption on commodity CPU hardware. Accuracy results are compared against Microsoft's published FP16 baselines — LLaMA 3.2 1B, Gemma-3 1B, Qwen2.5 1.5B, SmolLM2 1.7B, and MiniCPM 2B — to validate whether 1-bit quantization delivers meaningful real-world efficiency gains without significant accuracy loss.

### Objectives

- Run the pre-trained BitNet b1.58 2B4T model locally via `bitnet.cpp` on CPU hardware
- Independently benchmark inference latency, throughput, memory footprint, and energy consumption
- Evaluate output quality via zero-shot accuracy on standard NLP benchmarks (MMLU, HellaSwag, ARC, etc.)
- Compare locally measured efficiency numbers against Microsoft's published results (arXiv:2504.12285)
- Compare accuracy results against published FP16 baselines: LLaMA 3.2 1B, Gemma-3 1B, Qwen2.5 1.5B, SmolLM2 1.7B, and MiniCPM 2B
- Produce a cost-accuracy trade-off analysis including a carbon footprint proxy

---

## Tasks

### Phase 1 — Repository Study

- [ ] Create @BITNET_SUMMARY.md to describe the model
  - [ ] Summarize the [BitNet b1.58 paper (Ma et al., 2024)](https://podcast.aiedus.org/uploads/pdf/pdf-1759573772863-893673714.pdf)
  - [ ] Summarize the [BitNet b1.58 2B4T technical report (arXiv:2504.12285)](https://arxiv.org/abs/2504.12285)
  - [ ] Document the absmean quantization function and Straight-Through Estimator in @BITNET_SUMMARY.md
  - [ ] Document published FP16 baseline results (LLaMA 3.2 1B, Gemma-3 1B, Qwen2.5 1.5B, SmolLM2 1.7B, MiniCPM 2B) to use as comparison targets

### Phase 2 — Environment Setup & Model Acquisition

- [x] Clone the [microsoft/BitNet](https://github.com/microsoft/BitNet) repository
- [x] Configure `bitnet.cpp` inference environment on local CPU hardware
  - Built from commit `01eb415` with ClangCL 20 on Windows 11
  - Three patches applied (const-pointer fix, missing `#include <chrono>` in common and examples)
  - Full pipeline captured in `Makefile` (`make bitnet-setup bitnet-model`)
- [x] Download the BitNet b1.58 2B4T GGUF model checkpoint (`ggml-model-i2_s.gguf`)
- [x] Verify the model loads and produces output correctly via `bitnet.cpp` (`make bitnet-verify`)
- [x] Set up Python environment via Poetry (`pyproject.toml`, `make install`)
- [x] Set up `scripts/eval_accuracy.py` for accuracy evaluation
  - Uses `llama-server` `/completion` API with first-token log-probability scoring
  - Supports ARC (easy/challenge), WinoGrande, HellaSwag, MMLU (0-shot and few-shot)
  - Includes `--start-server` flag for automatic server lifecycle management
- [x] Set up `scripts/metrics_tracker.py` to record latency, memory, energy, and throughput per run
  - Invokes `llama-bench` with `--output json`; monitors peak RSS via `psutil`
  - Optional energy tracking via CodeCarbon; appends rows to `results/step_metrics.csv`
- [ ] Smoke-test scripts/metrics_tracker.py and scripts/eval_accuracy.py to confirm both run without errors and produce well-formed output

### Phase 3 — Inference Benchmarking

- [x] Run inference latency and throughput benchmarks on BitNet b1.58 2B4T via `make benchmark`
- [x] Record latency (ms per token), throughput (tokens/s), and peak memory to @step_metrics.csv
- [x] Run `scripts/eval_accuracy.py` on BitNet b1.58 2B4T (ARC, HellaSwag, WinoGrande, MMLU)
- [x] Log energy consumption per run using CodeCarbon — estimated via TDP proxy in @REPORT.md
- [x] Create @REPORT.md and record all results; sanity-check against arXiv:2504.12285 Tables

### Phase 4 — Cost Comparison

- [ ] Compile local benchmark results alongside published FP16 baselines (LLaMA 3.2 1B, Gemma-3 1B, Qwen2.5 1.5B, SmolLM2 1.7B, MiniCPM 2B) into @comparison_table.csv
- [ ] Run `scripts/compare_runs.py` (`make plots`) to generate latency, memory, and accuracy comparison plots
- [ ] Compute cost-accuracy trade-off (dollar cost proxy: time × hardware rate)
- [ ] Estimate carbon footprint using CodeCarbon output and compare against FP16 estimates from the literature
- [ ] Produce final benchmark dashboard (plots + @comparison_table.csv) in @FINAL_REPORT.md

### Phase 5 — Optimization & Writeup

- [ ] Explore at least one optimization (e.g. batch size tuning, quantization-aware runtime flags)
- [ ] Document scaling observations — at what size do efficiency gains become most significant?
- [ ] Write capstone research report to @FINAL_REPORT.md
- [ ] Prepare final presentation slides in @presentation.pptx
- [ ] Clean up repository for reproducibility (configs, instructions in @README.md, results)
