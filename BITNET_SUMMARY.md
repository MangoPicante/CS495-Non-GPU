# BitNet — Model Summary

**Author:** Sean Michael  
**Date:** May 2026

---

## 1. BitNet b1.58 (Ma et al., 2024 — arXiv:2402.17764)

### 1.1 Overview

BitNet b1.58 is a 1-bit LLM variant in which every weight is constrained to ternary values {−1, 0, +1}. The name comes from the information-theoretic bit-width of ternary: log₂(3) ≈ 1.585 bits per parameter. The core claim is that a model trained natively with ternary weights can match a full-precision (FP16/BF16) Transformer of the same size on both perplexity and downstream benchmarks, while substantially reducing latency, memory, throughput, and energy costs at inference time.

### 1.2 Absmean Quantization

Weights are quantized to ternary before each forward pass using the **absmean quantization** function:

```
W̃ = RoundClip(W / (γ + ε), −1, 1)
```

where:
- `γ = (1/nm) Σ|Wᵢⱼ|` — the per-tensor mean absolute value (the scale factor)
- `ε` — a small constant for numerical stability
- `RoundClip(x, a, b) = max(a, min(b, round(x)))` — rounds to nearest integer and clamps to [−1, 1]

The scale factor `γ` is stored in FP16 alongside the ternary weights and is used to dequantize during computation. This design replaces floating-point multiply-accumulate operations with integer additions and table lookups, enabling the TL2 (Ternary LUT 2-bit) kernel in `bitnet.cpp`.

### 1.3 Activation Quantization

Activations are quantized to 8-bit integers per token before each matrix multiply using absmax quantization:

```
x̃ = RoundClip(x / (η + ε) × Qb, −Qb, Qb)
```

where `η = max(|xᵢ|)` is the per-token absolute maximum and `Qb = 2^(b−1) − 1 = 127` for 8-bit. Scaling is per-token (not per-tensor) to preserve token-level dynamic range.

### 1.4 Straight-Through Estimator (STE)

The `RoundClip` function has zero gradient almost everywhere, which would block backpropagation. BitNet b1.58 uses the **Straight-Through Estimator (STE)** to allow gradients to flow through the quantization step. The STE approximates the gradient of the rounding as the identity function:

```
∂L/∂W ≈ ∂L/∂W̃    (where |W| ≤ 1)
```

In practice: gradients pass through unchanged when the pre-quantization weight is within [−1, 1], and are zeroed when clamping is active. Full-precision latent weights are maintained during training and re-quantized at every forward pass.

### 1.5 Architecture

BitNet b1.58 uses a LLaMA-compatible architecture to ease integration with existing tooling:
- **BitLinear layers** replace all standard `nn.Linear` layers (weights are quantized; biases are removed)
- **RMSNorm** replaces LayerNorm
- **SwiGLU** activation function
- **Rotary Position Embeddings (RoPE)**
- No bias terms anywhere in the network

### 1.6 Training and Results

Pre-trained on RedPajama (100B tokens). At 3B parameters, BitNet b1.58 achieves:
- **Perplexity:** 9.91 vs LLaMA 3B at 10.04 (matches)
- **Memory:** 2.22 GB vs 7.89 GB (3.55× reduction)
- **Latency:** 1.87 ms/token vs 5.07 ms/token (2.71× faster)

Downstream accuracy (ARC, HellaSwag, Winogrande) matches LLaMA at 3B and above.

---

## 2. BitNet b1.58 2B4T (Wang et al., 2025 — arXiv:2504.12285)

### 2.1 Overview

BitNet b1.58 2B4T is the first open-source, native 1-bit LLM at the 2-billion parameter scale, trained on 4 trillion tokens. It extends the 2402.17764 work with a full production training pipeline (pre-training → SFT → DPO), a larger and higher-quality dataset, and optimized CPU/GPU inference kernels. The goal is to demonstrate that a 1-bit model can be competitive with full-precision open-weight models of similar size on a broad range of tasks.

### 2.2 Architecture Differences from b1.58 (2402.17764)

| Component | b1.58 (2402.17764) | 2B4T (2504.12285) |
|---|---|---|
| Activation fn | SwiGLU | Squared ReLU |
| Normalization | RMSNorm | subLN (sub-LayerNorm) |
| Positional emb | RoPE | RoPE |
| Tokenizer | — | LLaMA 3 BPE, 128,256 vocab |
| Parameters | Up to 3B tested | 2.74B |

Squared ReLU (x → max(0, x)²) was chosen over SwiGLU for improved activation sparsity, which benefits the TL2 CPU kernel.

### 2.3 Training Pipeline

**Stage 1 — Pre-training (4T tokens):**
- Two-stage learning rate schedule: high initial LR with warmup, then cosine cooldown
- Two-stage weight decay: cosine schedule peaking at 0.1, then disabled
- Data mix: DCLM web crawl, FineWeb-EDU, synthetic math data

**Stage 2 — Supervised Fine-tuning (SFT):**
- Datasets: WildChat, LMSYS-Chat-1M, WizardLM Evol-Instruct, SlimOrca, plus synthetic data
- Loss aggregated by summation (not averaging)
- Larger learning rates and more epochs than typical FP16 SFT

**Stage 3 — Direct Preference Optimization (DPO):**
- Datasets: UltraFeedback, MagPie
- 2 epochs, LR = 2×10⁻⁷, β = 0.1

### 2.4 Inference Kernels

**GPU (CUDA):** W1.58A8 kernel packs four ternary weight values into one int8 for HBM storage; values are unpacked to SRAM for computation. Avoids full dequantization to FP16.

**CPU (bitnet.cpp — TL2 kernel):** Ternary LUT 2-bit kernel. Each pair of ternary weights is encoded as a 2-bit value (4 values per byte), and matrix-vector products are computed via lookup tables rather than multiply-accumulate. Activations remain 8-bit. The scale factor `γ` stored in FP16 is applied once per tile after the integer accumulation.

### 2.5 Published Benchmark Results (Table 1 — single x86 CPU core, 4 threads)

| Model | Size (GB) | Throughput (tok/s) | Latency (ms/tok) | Energy (J/tok) | ARC-E | ARC-C | WinoGrande | HellaSwag | MMLU |
|---|---|---|---|---|---|---|---|---|---|
| LLaMA 3.2 1B | 2.0 (FP16) | ~4.5 | ~222 | ~0.258 | 69.87 | 41.04 | 60.77 | 61.05 | 42.12 |
| Gemma-3 1B | 2.0 (FP16) | ~4.1 | ~244 | — | 79.42 | 46.25 | 66.38 | 72.15 | 50.33 |
| Qwen2.5 1.5B | 3.0 (FP16) | ~3.8 | ~263 | ~0.347 | 79.92 | 52.82 | 66.61 | 70.95 | 61.11 |
| SmolLM2 1.7B | 3.4 (FP16) | ~3.5 | ~286 | ~0.425 | 81.82 | 52.99 | 68.67 | 72.29 | 51.77 |
| MiniCPM 2B | 4.0 (FP16) | ~2.9 | ~345 | — | 82.20 | 51.96 | 68.27 | 75.08 | 53.07 |
| **BitNet b1.58 2B4T** | **1.71 (i2_s)** | **~20** | **~29** | **~0.028** | **74.79** | **49.91** | **71.90** | **68.44** | **53.17** |

**Key efficiency claims:**
- Non-embedding memory: 0.4 GB — **5–12× reduction** vs FP16 baselines
- Energy per token: 0.028 J — **9–23× reduction** vs FP16 baselines
- Decoding latency: 29 ms/tok — **1.6–2.3× faster** than comparable FP16 models on CPU
- Accuracy: competitive with or above all listed FP16 baselines on average (54.19% avg vs 44.90–55.23%)

---

## 3. Key Concepts Summary

| Concept | Description |
|---|---|
| **Absmean quantization** | Maps weights to {−1, 0, +1} using per-tensor mean absolute value as scale: `W̃ = RoundClip(W/γ, −1, 1)` |
| **STE** | Passes gradients through rounding unchanged (`∂L/∂W ≈ ∂L/∂W̃`), enabling backprop through quantization |
| **Activation quantization** | 8-bit per-token absmax scaling before each matrix multiply |
| **TL2 kernel** | CPU inference kernel using 2-bit ternary lookup tables instead of multiply-accumulate |
| **i2_s format** | GGUF quantization format used by bitnet.cpp; packs ternary weights at ~1.71 GiB for 2.74B params |
| **subLN** | Sub-LayerNorm: normalization applied inside the attention/FFN sublayers for training stability at scale |
