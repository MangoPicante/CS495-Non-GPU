# REPORT.md

## Project: Non-GPU LLM Training — Efficient Transformer Training with BitNet
**Author:** Sean Michael  
**Last Updated:** May 3, 2026

---

## Paper Annotations

### Paper 1: BitNet b1.58 (Ma et al., 2024)

**Citation:** Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, F., & Wei, F. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764.

**Core Contribution:** Introduces BitNet b1.58 — a 1-bit LLM variant where every weight parameter is ternary: {-1, 0, +1}. The "1.58" label comes from log₂(3) ≈ 1.58 bits per weight. This achieves the same or better performance than full-precision (FP16) LLMs at equal model size and training tokens, while being dramatically cheaper to run.

**Key Architectural Change — BitLinear Layer:** The standard `nn.Linear` layer is replaced with a `BitLinear` layer that:
1. Quantizes weights to {-1, 0, +1} using AbsMean quantization (see below)
2. Quantizes activations to 8-bit integers using AbsMax quantization (see below)
3. Performs matrix multiplication using only additions (no multiplications needed for ternary weights)

**Design Choices:**
- Normalization: SubLN (Sub-Layer Normalization) before each BitLinear call to stabilize training
- No bias terms in any BitLinear or normalization layer
- Activations: not quantized before the non-linear function; only quantized at the input to each BitLinear layer
- Training: from scratch with quantization applied in every forward pass (not post-training quantization)

---

### Paper 2: BitNet b1.58 2B4T Technical Report (Microsoft Research, 2025)

**Citation:** Microsoft Research. (2025). *BitNet b1.58 2B4T Technical Report.* arXiv:2504.12285.

**Core Contribution:** Releases the first open-weight, general-purpose 1-bit LLM suitable for real-world deployment: BitNet b1.58 2B4T (~2B parameters, trained on 4 trillion tokens). Demonstrates that a 1.58-bit model can match or exceed competitive FP16 baselines in the 1–3B parameter class while delivering an order-of-magnitude improvement in inference efficiency on commodity CPU hardware.

**Model Specifications:**

| Property | Value |
|---|---|
| Parameter count | ~2 billion |
| Training tokens | 4 trillion |
| Context length | 4,096 tokens |
| Weight precision | 1.58-bit ternary {-1, 0, +1} |
| Activation precision | 8-bit integer (per-token absmax) |
| Tokenizer | LLaMA 3 BPE, vocabulary 128,256 |
| Position embeddings | Rotary Position Embeddings (RoPE) |
| FFN activation | Squared ReLU (ReLU²) |
| Normalization | SubLN (Sub-Layer Normalization) |
| Bias terms | None |

**Training Pipeline:**
- Phase 1 — Pre-training (4T tokens): two-stage LR schedule; Stage 1 uses high peak LR with cosine decay on general web data (DCLM, FineWeb-EDU); Stage 2 (cooldown) uses lower LR on curated high-quality data including synthetic math
- Phase 2 — Supervised Fine-Tuning (SFT): WildChat, LMSYS-Chat-1M, WizardLM Evol-Instruct, SlimOrca, GLAN, MathScale; uses sum loss (not mean) per batch, which empirically improves 1-bit model alignment
- Phase 3 — DPO: 2 epochs, LR = 2×10⁻⁷, beta = 0.1; UltraFeedback + MagPie datasets; Liger Kernel for efficiency

---

## AbsMean Quantization Function

BitNet b1.58 uses two distinct quantization schemes: AbsMean for weights, and AbsMax for activations.

### Weight Quantization — AbsMean → {-1, 0, +1}

**Step 1:** Compute the absolute mean (gamma) across all elements of the weight matrix W (shape n × m):
```
γ = (1 / n·m) · Σᵢⱼ |Wᵢⱼ|
```

**Step 2:** Normalize and round with hard clipping to {-1, 0, +1}:
```
W̃ = RoundClip(W / (γ + ε), -1, 1)
```

**RoundClip definition:**
```
RoundClip(x, a, b) = max(a, min(b, round(x)))
```

Where ε is a small constant (e.g., 1e-6) for numerical stability. The result maps each scalar weight to the nearest integer in {-1, 0, +1}. The gamma factor serves as a per-tensor scale that is stored in FP16 for dequantization during inference.

**Why "AbsMean":** The scale is derived from the mean of absolute values, not the maximum — this makes the scale robust to outlier weights that would otherwise dominate a max-based scheme.

**Storage:** Four ternary values are packed into a single `int8` byte at inference time, reducing memory to ~2 bits per weight in practice.

### Activation Quantization — AbsMax → INT8

Activations are quantized per-token (not per-tensor) to 8-bit integers:
```
x̃ = Clip(round(x · (Qb / max|xᵢ|)), -Qb, Qb)
```

Where Qb = 127 for INT8. The scale is computed independently for each token in the sequence. This per-token granularity is important because activation magnitudes vary significantly across tokens.

---

## Straight-Through Estimator (STE)

The `round()` function inside RoundClip is non-differentiable: its gradient is zero almost everywhere and undefined at integers. Standard backpropagation cannot update weights through it. The Straight-Through Estimator (STE) is the solution.

**Core idea:** During the forward pass, use the quantized (discrete) value. During the backward pass, pretend the quantization was an identity function and pass the gradient straight through.

**Forward pass:**
```
W̃ = RoundClip(W / (γ + ε), -1, 1)
```

**Backward pass (clipped STE):** The gradient is passed unchanged for weights whose normalized value falls within the clip range [-1, 1], and zeroed out (killed) for weights outside:
```
∂L/∂W = (∂L/∂W̃) · (1/γ) · 1[|W/(γ+ε)| ≤ 1]
```

The `1/γ` factor accounts for the division by gamma in the forward pass (chain rule). The indicator function `1[|·| ≤ 1]` zeroed out saturated weights, preventing gradients from accumulating on already-clipped values.

**Why it works:** The error introduced by replacing `∂round(x)/∂x = 0` with `∂id(x)/∂x = 1` is acceptable in practice because weights are continuously adjusted by SGD toward values that quantize well. Over training, the model learns weight distributions that are naturally concentrated near {-1, 0, +1}.

**Practical note:** The 2B4T report trains the entire model from scratch with quantization on every forward pass. This is quantization-aware training (QAT), not post-training quantization (PTQ). The STE allows gradients to flow through the entire 2B-parameter network despite all linear-layer weights being ternary.

---

## Published FP16 Baseline Results — Comparison Targets

The following results are reproduced from arXiv:2504.12285, Table 1 (instruction-tuned models, evaluated under a uniform pipeline using `lm-evaluation-harness`).

### Accuracy Benchmarks

| Benchmark | BitNet b1.58 2B4T | LLaMA 3.2 1B | Gemma-3 1B | Qwen2.5 1.5B | SmolLM2 1.7B | MiniCPM 2B |
|---|---|---|---|---|---|---|
| **MMLU** | 53.17 | 45.58 | 39.91 | **60.25** | 49.24 | 51.82 |
| **HellaSwag** | 68.44 | 60.80 | 57.69 | 68.28 | **71.71** | 70.81 |
| **ARC-Easy** | 74.79 | 63.17 | 63.13 | **76.01** | 62.92 | 72.14 |
| **ARC-Challenge** | **49.91** | 37.80 | 38.40 | 46.67 | 43.52 | 44.80 |
| **WinoGrande** | **71.90** | 59.51 | 58.48 | 62.83 | 68.98 | 61.80 |
| **PIQA** | **77.09** | 74.21 | 71.93 | 76.12 | 76.12 | 76.66 |
| **OpenbookQA** | 41.60 | 34.80 | 38.80 | 40.80 | **46.00** | 40.20 |
| **CommonsenseQA** | 71.58 | 58.48 | 42.10 | **76.41** | 63.55 | 71.74 |
| **BoolQ** | 80.18 | 64.65 | 74.22 | 78.04 | 75.78 | **80.67** |
| **TriviaQA** | 33.57 | 37.60 | 23.49 | 38.37 | **45.97** | 34.13 |
| **TruthfulQA** | 45.31 | 43.80 | 38.66 | **46.67** | 39.90 | 41.41 |
| **HumanEval+** | 38.40 | 31.10 | 37.20 | **50.60** | 28.00 | 43.90 |
| **GSM8K** | **58.38** | 38.21 | 31.16 | 56.79 | 45.11 | 4.40 |
| **MATH-500** | 43.40 | 23.00 | 42.00 | **53.00** | 17.60 | 14.80 |
| **IFEval** | 53.48 | 62.71 | **66.67** | 50.12 | 57.91 | 36.81 |
| **MT-Bench** | 5.85 | 5.43 | 6.40 | 6.12 | 5.50 | **6.57** |
| **Average** | 54.19 | 44.90 | 43.74 | **55.23** | 48.70 | 42.05 |

**Bold = best in row.** Higher is better for all benchmarks.

### Observations from Accuracy Results
- BitNet b1.58 2B4T scores an average of 54.19 — competitive with Qwen2.5 1.5B (55.23) despite Qwen being trained on 18T tokens (4.5x more data)
- BitNet leads on reasoning tasks: ARC-Challenge, WinoGrande, PIQA, CommonsenseQA, GSM8K
- Qwen2.5 1.5B leads on knowledge and coding (MMLU, HumanEval+, MATH-500)
- MiniCPM 2B has anomalous GSM8K score (4.40) — likely a different chat-format issue in evaluation
- Gemma-3 1B and LLaMA 3.2 1B trail significantly, consistent with their smaller effective capacity

### Published Inference Efficiency Results (CPU Hardware)

**Test system:** Surface Laptop Studio 2, Intel Core i7-13800H (13th Gen), 8 CPU threads, 128 tokens generated

| Model | Precision | Non-Emb Memory | Latency (ms/tok) | Throughput (tok/s) | Energy per Token |
|---|---|---|---|---|---|
| **BitNet b1.58 2B4T** | 1.58-bit | **0.4 GB** | **29 ms** | **~34.5** | **0.028 J** |
| LLaMA 3.2 1B | FP16 | 2.0 GB | 48 ms | ~20.8 | 0.258 J |
| Gemma-3 1B | FP16 | 1.4 GB | 41 ms | ~24.4 | 0.186 J |
| Qwen2.5 1.5B | FP16 | 2.6 GB | 65 ms | ~15.4 | 0.347 J |
| SmolLM2 1.7B | FP16 | 3.2 GB | 67 ms | ~14.9 | 0.425 J |
| MiniCPM 2B | FP16 | 4.8 GB | 124 ms | ~8.1 | 0.649 J |

**Notes:**
- Non-embedding memory excludes the embedding table (shared across models)
- Energy estimated via Horowitz (2014) arithmetic operations model at 7nm process node
- Ternary weights eliminate multiplications entirely; only additions are needed → primary source of energy savings
- FP16 models run via `llama.cpp`; BitNet via `bitnet.cpp` — comparison reflects real-world tool choice

**Key efficiency ratios vs. MiniCPM 2B (closest parameter match):**
- 12× lower memory (0.4 GB vs 4.8 GB)
- 4.3× lower latency (29 ms vs 124 ms)
- 23× lower energy per token (0.028 J vs 0.649 J)

---

## Benchmark Target Confirmation

**Target model:** `microsoft/BitNet-b1.58-2B-4T` (GGUF, i2_s quantization)  
**Inference runtime:** `bitnet.cpp` (built at `../BitNet/build/`)  
**Model path:** `../BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`  
**Status:** Model downloaded (1.11 GB GGUF), runtime built, ready for Phase 3 benchmarking

Local hardware specifications (to be recorded at benchmark time):
- CPU: [record at benchmark time]
- RAM: [record at benchmark time]
- OS: Windows 11
- Thread count used: [to be determined per run]

---

*This report will be updated throughout Phases 3–5 with local benchmark results, comparison plots, and the final cost-accuracy analysis.*
