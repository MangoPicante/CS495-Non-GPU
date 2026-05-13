# Qwen — Model Summary

**Author:** Sean Michael
**Date:** May 2026

---

## 1. Qwen2.5 Model Family (Qwen Team, 2024 — arXiv:2412.15115)

### 1.1 Overview

Qwen2.5 is Alibaba's third-generation open-weight LLM series, released in September 2024. The family spans seven base sizes — 0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B — each offered as a **Base** model (next-token-prediction pretraining only) and an **Instruct** variant (SFT + DPO-aligned). Pretraining corpus is up to 18 trillion tokens of multilingual text, code, and math. Most sizes (including the 1.5B used here) are released under Apache 2.0; the 3B and 72B variants are Qwen Research licensed.

The core design choice across the family is to be a "boring but strong" LLaMA-architecture model: standard decoder-only Transformer, Grouped-Query Attention, RoPE, SwiGLU, and RMSNorm — no architectural exotics — with the headline gains coming from better data (a much larger and more carefully filtered pretraining mix) and a longer alignment pipeline. This makes the family a useful reference point for "what a competitive open-weight FP16 model at a given parameter count looks like today."

### 1.2 Architecture

| Component | Choice |
|---|---|
| Architecture | Decoder-only Transformer |
| Attention | Grouped-Query Attention (GQA) |
| Positional embedding | RoPE (with YaRN extension for long-context variants) |
| FFN activation | SwiGLU |
| Normalization | RMSNorm |
| Tokenizer | BPE, 151,646 base + 152,064 with special tokens |
| Native context | 32,768 tokens (32K) |
| Bias terms | None on linear layers; only in attention QKV projections of some sizes |

### 1.3 Training Pipeline

**Stage 1 — Pre-training:** Up to 18T tokens of curated multilingual text. The mix is heavier on code and math than typical web-text-dominated corpora, which materially boosts MMLU and reasoning scores in this size class.

**Stage 2 — Supervised Fine-tuning (SFT):** Instruction data plus high-quality dialogue. Qwen Team's release notes describe staged curriculum data (easier first, harder later) and explicit long-context training.

**Stage 3 — Direct Preference Optimization (DPO):** Offline preference learning over SFT model. For the smaller variants (including 1.5B-Instruct), the alignment pipeline ends at DPO; the larger sizes additionally do online RL.

---

## 2. Qwen2.5-1.5B-Instruct (the model used here)

### 2.1 Specifications

| Property | Value |
|---|---|
| Total parameters | 1.54B |
| Non-embedding parameters | 1.31B |
| Layers | 28 |
| Hidden dim | 1,536 |
| FFN intermediate dim | 8,960 |
| Query heads | 12 |
| Key/Value heads | 2 (GQA ratio 6:1) |
| Tied embeddings | Yes (input and output) |
| Vocab | 151,646 (BPE) |
| Context window | 32,768 tokens |

Tied embeddings and a GQA ratio of 6:1 are the main parameter-saving choices at this scale: the model spends a much smaller fraction of its parameters on the KV cache and output projection than a comparable MQA-only or full-MHA design would.

### 2.2 Variant Choice (Instruct vs Base)

We use the **Instruct** variant. Base would be more directly comparable to BitNet b1.58 2B4T's *base* model in a strict apples-to-apples sense (BitNet 2B4T also has Instruct/DPO variants but the headline efficiency numbers in arXiv:2504.12285 Table 1 come from the post-SFT/DPO model). Instruct is the right comparison for end-user-relevant accuracy and for inference timing: chat-template overhead and instruction-following are reflected in real-world cost. The benchmarks in this project (ARC, WinoGrande, HellaSwag, MMLU) are zero-shot multiple-choice and do not depend on chat-template wrapping.

---

## 3. Q8_0 Quantization

### 3.1 Format

Q8_0 is GGUF's "8-bit symmetric, fixed-block" quantization scheme — the highest-fidelity quant tier in common use on CPU. The weight tensor is partitioned into blocks of 32 elements; each block is stored as:

```
struct block_q8_0 {
    float16 scale;        // 2 bytes
    int8    qs[32];       // 32 bytes
};                        // 34 bytes / 32 elements = 8.5 bits/parameter
```

### 3.2 Algorithm

For each 32-element block `W`:

```
scale = max(|Wᵢ|) / 127                      (FP16)
qᵢ    = clamp(round(Wᵢ / scale), -127, 127)  (int8)
```

Reconstruction at inference time: `W̃ᵢ = qᵢ × scale`. The dequantization is fused into the GEMM kernel — weights are upcast to FP32 (or BF16, on some backends) tile by tile inside the matmul, never materialized in full precision in memory.

### 3.3 Accuracy vs FP16

Q8_0 is empirically near-lossless on standard benchmarks: published llama.cpp comparisons consistently report <0.1% perplexity increase and indistinguishable accuracy on MMLU/HellaSwag/ARC at the LLaMA 7B-13B scale. The trade-off vs FP16:

- Model size: ~1.65 GiB (Q8_0) vs ~3.0 GiB (FP16) — **~45% reduction**
- KV cache: same as FP16 (KV cache is not quantized by Q8_0 alone)
- Throughput: small but real CPU-side speedup (smaller weights → fewer cache misses → faster GEMM)
- Accuracy: within sampling noise of FP16 on the benchmarks in this project

Q8_0 is the natural choice when memory is not the binding constraint and you want to minimize quantization-attributable accuracy loss. For more aggressive memory savings, GGUF offers Q4_K_M, Q5_K_M, etc., at the cost of measurable accuracy drops.

---

## 4. Published Accuracy Numbers

### 4.1 As reported in the BitNet paper (arXiv:2504.12285 Table 1)

These are the numbers we directly compare against in `compare_runs.py`:

| Benchmark | Qwen2.5 1.5B (FP16) |
|---|---|
| ARC-Easy | 79.92 |
| ARC-Challenge | 52.82 |
| WinoGrande | 66.61 |
| HellaSwag | 70.95 |
| MMLU | 61.11 |

### 4.2 Throughput / Memory / Energy (same source, n_prompt=512, n_gen=128, 1×CPU)

| Metric | Value |
|---|---|
| Model size | 3.0 GB (FP16) |
| Throughput | ~3.8 tok/s |
| Latency | ~263 ms/tok |
| Energy | ~0.347 J/tok |

These are the headline FP16 numbers that the BitNet paper uses to argue 9–23× energy efficiency. Reproducing them locally is a primary goal of this project.

### 4.3 Methodology Note

The BitNet paper's accuracy numbers were generated via `lm-evaluation-harness` with standard task configurations:

- **ARC-Easy / ARC-Challenge**: length-normalized loglikelihood scoring (`acc_norm`) of full answer text
- **WinoGrande**: partial-context scoring — `P(suffix | prefix + option)`
- **HellaSwag**: length-normalized loglikelihood with WikiHow `[title]` cleanup, prefixed by `activity_label`
- **MMLU**: 5-shot, first-token letter scoring with the standard "The following are multiple choice questions..." header

Our `scripts/eval_accuracy.py` matches each of these exactly. An earlier version used first-token letter scoring for ARC and a `P(option | prefix)` reading for WinoGrande; those were corrected after producing ~+19pt above-paper ARC-C and near-random WinoGrande, both of which were methodology artifacts rather than model differences.

---

## 5. Why Qwen as the CPU Baseline

Three constraints narrow the practical choice of "what FP16 model do we run alongside BitNet on the same CPU to make the comparison honest":

1. **Size class.** BitNet b1.58 2B4T is 2.74B parameters. A fair efficiency baseline needs to be the same order of magnitude — comparing a 1-bit 2B model against a 7B FP16 model would conflate "quantization helps" with "smaller model is cheaper." Qwen2.5-1.5B (1.54B params) is the closest comparable Apache-licensed FP16 model whose published accuracy lands in the same band as BitNet 2B4T.

2. **Licensing and reproducibility.** The model is Apache 2.0, GGUFs are first-party on HuggingFace, and Q8_0 conversion is deterministic. Anyone re-running this benchmark gets bit-identical weights.

3. **Inference stack alignment.** Qwen2.5 is a pure LLaMA-architecture model that runs on **upstream `llama.cpp`** without any custom kernels or model-side patches. This matters because the BitNet fork lags upstream by ~1 year of llama.cpp development; isolating the FP16 baseline on upstream lets us see what a current llama.cpp build can do on the same hardware.

Alternatives considered and rejected:
- **LLaMA 3.2 1B** would be a more direct match to the BitNet paper's exact comparison table, but is gated on Meta's license agreement.
- **Gemma-3 1B** is Apache 2.0 but uses non-standard normalization and gating that complicates fair comparison.
- **Phi-3-mini (3.8B)** is too large for the chosen size class.
- **SmolLM2 1.7B** and **MiniCPM 2B** are reasonable but have less mature GGUF ecosystems.

---

## 6. Inference Stack: Upstream `llama.cpp` vs BitNet's Fork

Both BitNet b1.58 2B4T and Qwen2.5-1.5B-Instruct are served via `llama-server` HTTP endpoints in this project. Crucially, **they use different `llama.cpp` builds**:

| Aspect | Upstream `llama.cpp` (Qwen) | `microsoft/BitNet` fork (BitNet) |
|---|---|---|
| Base commit | Current (2026) | Forked from ~Q2 2024 |
| Custom kernels | None beyond standard quants | TL2 (Ternary LUT 2-bit) |
| Custom quant format | F16/BF16, Q8_0, Q5/Q4/Q3/Q2_K | i2_s (BitNet ternary), I2_S/TL2-specific tiles |
| Default ubatch | 512 | Must be ≤128 (TL2 kernel stack overflow above this) |
| `/completion` response | `{token, logprob, top_logprobs[]}` (new) | `{content, probs[{tok_str, prob}]}` (old) |
| `post_sampling_probs:false` | Supported — returns *natural* logprob under `logit_bias` | Not supported — bias is applied before reported probs |
| `n_probs` upper bound | Effectively vocab-size | Crashes at ≳50,000 (segfault); 5,000 is safe |
| OpenAI-compatible endpoint | `/v1/completions`, `/v1/chat/completions` | Partial; older spec |
| `/tokenize`, `/detokenize` | Yes | Yes |

### 6.1 Consequences for the Eval Pipeline

`scripts/eval_accuracy.py` probes the server once at startup (`_server_caps()`) and picks one of two continuation-scoring paths:

- **Upstream (Qwen) path:** for each target continuation token, send `logit_bias:[[token_id, +100]]` with `post_sampling_probs:false`. The forced token comes back with its **natural** (pre-bias) logprob, exact for any token in the vocabulary.
- **Fork (BitNet) path:** the bias trick is unusable because the reported probs are post-bias. Fall back to top-K search at `n_probs=5000`; when the target token is rarer than top-5000 (uncommon in practice), use `min(top_K_logprob) − 1.0` as a conservative lower bound.

This is the single largest source of methodology asymmetry between the two models in this project, and the only one we cannot eliminate without rebuilding BitNet against current upstream. It is documented inline in `eval_accuracy.py` and reflects the (acceptable) reality that the BitNet inference stack is the actual artifact under test, so we should not paper over its API behavior.

### 6.2 Consequences for Benchmarking

`scripts/metrics_tracker.py` shells out to `llama-bench` from whichever build is configured. The JSON schema is stable across the fork divergence point — both produce `{n_prompt, n_gen, avg_ts, ...}` records — so latency, throughput, and peak-RSS comparisons are apples-to-apples. Energy (via CodeCarbon) and peak-RSS (via `psutil`) are measured at the OS level and are stack-agnostic.

---

## 7. Key Concepts Summary

| Concept | Description |
|---|---|
| **Q8_0** | 8-bit symmetric quantization with 32-element blocks; ~45% size reduction vs FP16, near-lossless accuracy |
| **GQA** | Grouped-Query Attention; 12 query heads share 2 KV heads in Qwen2.5-1.5B (6:1 KV-cache compression) |
| **SwiGLU** | Swish-gated linear unit FFN activation; contrast with BitNet 2B4T's squared-ReLU choice for activation sparsity |
| **RMSNorm** | Root-mean-square normalization; same as BitNet b1.58, distinct from BitNet 2B4T's subLN |
| **DPO** | Direct Preference Optimization — offline preference learning that replaces RLHF in the Qwen2.5 1.5B-Instruct pipeline |
| **GGUF** | Successor to GGML; the binary format `llama.cpp` uses for quantized weights, metadata, and tokenizer |
| **YaRN** | Position-interpolation method for extending RoPE context beyond the training length; not used in our 32K runs |
