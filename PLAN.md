# PLAN.md

## Project Overview

**Title:** Non-GPU LLM Training: Efficient Transformer Training with BitNet
**Author:** Sean Michael
**Date:** April 27, 2026

### Description

A comparison between the performance and accuracy of ternary trained models (BitNet b1.58, weights in {-1, 0, 1}) vs FP16 trained models, with a focus on whether 1-bit quantization makes transformer pretraining viable on commodity CPUs — and at what cost to accuracy, memory, and energy.

### Objectives

- Build a decoder-only transformer trained in FP16 as a reproducible baseline
- Build an equivalent transformer trained with ternary (1.58-bit) weights using BitNet b1.58
- Compare both models on perplexity, zero-shot accuracy, training time, memory footprint, and energy consumption
- Produce a cost-accuracy trade-off analysis including a carbon footprint proxy
- Determine whether CPUs are a viable substrate for low-precision LLM pretraining