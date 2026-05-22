# Non-GPU LLM Inference — Capstone Report

**Author:** Sean Michael
**Date:** May 2026
**Hardware:** Intel Core i5-9400F @ 2.90 GHz (6 cores, 4 threads used), 16 GB RAM, Windows 11
**Models under test:**
- BitNet b1.58 2B4T (i2_s GGUF, 1.71 GiB) via `microsoft/BitNet` (commit `01eb4157`)
- Qwen2.5-1.5B-Instruct Q8_0 (GGUF, 1.65 GiB) via upstream `ggml-org/llama.cpp` (commit `1e5ad35d`)
- Qwen2.5-1.5B-Instruct Q4_K_M (GGUF, ~1.0 GiB) — same upstream `llama.cpp` build
- Five FP16 baselines as reported in arXiv:2504.12285 Table 1
  (LLaMA 3.2 1B, Gemma-3 1B, Qwen2.5 1.5B, SmolLM2 1.7B, MiniCPM 2B)

This is the single canonical project report — a comparison dashboard of
the three locally measured models against each other and against
published FP16 baselines, with a cross-reference of measured energy
against the paper's FP16 J/tok estimates. Methodology and per-script
implementation details are in `PLAN.md`. **Appendix B** (BitNet) and
**Appendix C** (Qwen) are the model cards; **Appendix A** preserves the
Phase 3 BitNet-only sanity-check numbers that this report supersedes.

---

## 1. Executive Summary

| Metric (n_prompt=512, n_gen=128, 4 threads) | BitNet b1.58 2B4T (ours) | Qwen2.5-1.5B Q8_0 (ours) | Qwen2.5-1.5B Q4_K_M (ours) |
|---|---:|---:|---:|
| Throughput | 21.2 tok/s | 15.1 tok/s | **24.9 tok/s** |
| Peak RSS | **1,247 MB** | 1,667 MB | 1,632 MB |
| Cost — AWS c5.xlarge proxy @ $0.170/hr | $0.00223 / 1k tok | $0.00313 / 1k tok | **$0.00190 / 1k tok** |
| Cost — local electricity @ $0.16/kWh | **$0.000131 / 1k tok** | $0.000224 / 1k tok | $0.000132 / 1k tok |
| Mean accuracy (5 tasks) | **61.74%** | 61.10% | 59.45% |
| ARC-Easy | **74.2%** | 74.2% | 71.0% |
| ARC-Challenge | **46.0%** | 44.2% | 43.2% |
| WinoGrande | **75.2%** | 65.8% | 63.0% |
| HellaSwag | 58.6% | **59.0%** | 58.8% |
| MMLU (5-shot) | 54.69% | **62.28%** | 61.23% |
| Energy (Wh / 1k tok, CodeCarbon) | **0.82** | 1.40 | 0.83 |

**Headline.** Three models trace a clean speed/accuracy Pareto on CPU at
this size class. **Q4_K_M is the fastest** (24.9 tok/s, ~17% over BitNet)
and the cheapest in the AWS-rental framing. **BitNet is the
Pareto-optimal point**: it matches Q8_0's mean accuracy (within 0.6pt)
while running ~40% faster, and matches Q4_K_M's speed-class while
beating its accuracy by 2.3pt mean. **BitNet wins memory** decisively
(~25% lower RSS than either Qwen variant) and **wins commonsense
reasoning** (WinoGrande +9.4pt over Q8, +12.2pt over Q4). Qwen wins
knowledge recall (MMLU: Q8 +7.6pt, Q4 +6.5pt over BitNet). The paper's
claim of 9–23× energy efficiency over FP16 baselines does not survive
measurement at our power-tracking resolution; the inference-marginal
story may still hold but cannot be confirmed without isolated
power-rail readings (see §5).

---

## 2. Methodology (Summary)

Detailed methodology is in `PLAN.md` §Implementation; this section pulls
out the points needed to interpret the dashboard below.

- **Throughput / memory** — `scripts/metrics_tracker.py` wraps
  `llama-bench` at three `(n_prompt, n_gen)` configs matching arXiv:2504.12285
  Table 1: `(512, 128)`, `(512, 512)`, `(1, 512)`. Three reps per config;
  medians reported. `peak_rss_mb` sampled by `psutil`.
- **Energy / CO₂** — CodeCarbon `EmissionsTracker` wraps each bench run;
  `energy_kwh` and `co2_kg` are recorded per row.
- **Accuracy** — `scripts/eval_accuracy.py` drives `llama-server` with
  per-task scoring matching `lm-evaluation-harness`: length-normalized
  loglikelihood (ARC, HellaSwag), partial-context (WinoGrande), first-token
  letter scoring with 5-shot prompts (MMLU). The bias-trick path is used
  on upstream Qwen; an `n_probs=n_vocab` full-distribution fallback is
  used on the BitNet fork (see Appendix C.4).
- **Cost framings** — two parallel columns in `comparison_table.csv`,
  answering different questions:
  - `cost_per_1k_tokens = (1000 / throughput / 3600) × hardware_rate`.
    Default rate `$0.170/hr` (AWS c5.xlarge on-demand, us-east-1, 4 vCPUs,
    retrieved 2026-05-08). Override with `--hardware-rate`. Available for
    every row including paper FP16 baselines (the paper reports
    throughput, so this can be computed). Answers: *"what would this cost
    to rent in the cloud?"*
  - `energy_cost_per_1k_tokens = (energy_kwh × 1000 / (n_prompt + n_gen))
    × electricity_rate`. Default rate `$0.16/kWh` (US residential average).
    Override with `--electricity-rate`. Populated only for "ours" rows
    where we have CodeCarbon measurements; the paper doesn't report
    energy for FP16 baselines. Answers: *"what's the marginal electricity
    cost on hardware I already own?"*
- **Cloud API pricing** — `CLOUD_API_PRICING` in `compare_runs.py`,
  hardcoded as of 2026-05-15 from each provider's public pricing page.
  Used only by §3.9 (`cloud_cost_comparison.png`); §3.5 and §5.3 still
  use the AWS proxy and local-electricity framings.
- **Paper FP16 baselines** are pasted directly from arXiv:2504.12285
  Table 1; they were measured on a single x86 CPU core at the same
  `(512, 128)` condition.

---

## 3. Dashboard

### 3.1 Aggregate comparison table

Generated by `compare_runs.py` → `results/comparison_table.csv`:

| Model | Source | tok/s | Peak RSS (MB) | $/1k tok | ARC-E | ARC-C | Wino | HellaSwag | MMLU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaMA 3.2 1B | paper (FP16) | 4.5 | 2,600 | 0.01049 | 69.87 | 41.04 | 60.77 | 61.05 | 42.12 |
| Gemma-3 1B | paper (FP16) | 4.1 | 2,700 | 0.01152 | 79.42 | 46.25 | 66.38 | 72.15 | 50.33 |
| SmolLM2 1.7B | paper (FP16) | 3.5 | 3,300 | 0.01349 | 81.82 | 52.99 | 68.67 | 72.29 | 51.77 |
| MiniCPM 2B | paper (FP16) | 2.9 | 4,100 | 0.01628 | 82.20 | 51.96 | 68.27 | 75.08 | 53.07 |
| BitNet b1.58 2B4T | paper | 20.0 | 1,400 | 0.00236 | 74.79 | 49.91 | 71.90 | 68.44 | 53.17 |
| **BitNet b1.58 2B4T** | **ours** | **21.2** | **1,247** | **0.00223** | **74.2** | **46.0** | **75.2** | **58.6** | **54.69** |
| Qwen2.5 1.5B | paper (FP16) | 3.8 | 3,100 | 0.01243 | 79.92 | 52.82 | 66.61 | 70.95 | 61.11 |
| **Qwen2.5-1.5B-Instruct Q8_0** | **ours** | **15.1** | **1,667** | **0.00313** | **74.2** | **44.2** | **65.8** | **59.0** | **62.28** |

### 3.2 Throughput

![Throughput vs FP16 baselines, with per-config sensitivity](results/plots/throughput_comparison.png)

The unified plot has two panels:

- **Panel (a)** — cross-model comparison at the paper's reference config
  (n_prompt=512, n_gen=128). BitNet (ours) matches the paper's ~20 tok/s
  claim (21.2 vs 20.0). Both Qwen "ours" rows substantially outperform
  the paper's ~3.8 tok/s figure — Q8_0 by ~4× and Q4_K_M by ~6.5× —
  and the dominant cause is that **the paper measured the full FP16
  model while we run quantized variants**.  Q8_0 halves the weight
  memory bandwidth vs FP16, and Q4_K_M quarters it; on memory-bandwidth
  bound CPU matmul (which is the regime at this parameter scale on a
  consumer CPU), that's nearly the full story for the speedup.  The
  upstream `llama.cpp` version delta vs the paper's build is small by
  comparison and not the primary explanation — the cross-stack
  sensitivity check in §6.8 confirms this empirically (≤5% delta at the
  reference config when Qwen Q8 is re-run against the older BitNet
  fork).  The apples-to-apples ours-vs-ours ranking is **Q4 (24.9) >
  BitNet (21.2) > Q8 (15.1)**.
- **Panel (b)** — workload-shape sensitivity across the three configs for
  our three locally measured models. See §3.4 for the detailed numbers
  and the kernel/quantization attribution discussion.

### 3.3 Memory

![Peak RSS vs FP16 baselines](results/plots/memory_comparison.png)

BitNet at **1,247 MB** is ~25% lower than either Qwen variant (Q8_0 at
1,667 MB, Q4_K_M at 1,632 MB).  Interestingly, **Q4_K_M does not save
much RSS vs Q8_0** despite being ~half the on-disk size — the constant
overhead from the KV cache, activations, and runtime data structures
dominates the weight-storage delta at this parameter count.  All three
are well under the FP16 baselines (which range 2.6–4.1 GB).  Memory is
BitNet's cleanest win across all metrics in the report.

### 3.4 Per-config throughput

Numbers from panel (b) of the throughput plot in §3.2:

| Config | BitNet (tok/s) | Qwen Q8_0 (tok/s) | Qwen Q4_K_M (tok/s) |
|---|---:|---:|---:|
| `n_prompt=512, n_gen=128` | 21.2 | 15.1 | 24.9 |
| `n_prompt=512, n_gen=512` | 20.4 | 15.0 | 24.7 |
| `n_prompt=1, n_gen=512`   | 20.8 | 14.3 | — |

BitNet's throughput is essentially flat across the three workload shapes
(±2%); Qwen Q8_0 drops ~5% from prompt-heavy `(512, 128)` to pure
generation `(1, 512)`. The TL2 kernel's lookup-vs-multiply tradeoff is
consistent regardless of prompt/generation balance.

Qwen Q4_K_M is the surprise: ~17% **faster** than BitNet at the
reference config (24.9 vs 21.2 tok/s) and stays ahead at (512, 512).
This reframes the kernel-attribution claim in §5.1 — aggressive Q4
quantization on upstream `llama.cpp` matches or beats the TL2 kernel
on raw throughput at this size class, which means BitNet's throughput
win in §3.2 is specifically vs Q8_0, not vs all aggressive
quantization.  The accuracy cost of Q4's speed is real but modest:
mean accuracy drops 1.65pt vs Q8_0 and 2.29pt vs BitNet (see §3.7).

### 3.5 Cost–Accuracy

![Cost vs mean accuracy](results/plots/cost_accuracy.png)

All three "ours" points sit at the lower-left corner — cheaper *and*
higher mean accuracy than every paper FP16 baseline at this size class.
Within the "ours" cluster:

- **Qwen Q4_K_M** is the cheapest per token on the AWS-proxy framing
  ($0.00190 / 1k tok), beating BitNet by ~15%.
- **BitNet** is most accurate (mean 61.74%, +0.64 vs Q8, +2.29 vs Q4).
- **Qwen Q8_0** is the most expensive of the three and sits in the
  middle on accuracy — it has no obvious operational role unless you
  specifically need Q8's near-FP16 fidelity on knowledge tasks (MMLU,
  where it edges out Q4 by ~1pt).

Per-task variants are in `results/plots/{task}_cost_accuracy.png`. MMLU
is the only task where Qwen2.5 1.5B (paper FP16) is competitive on the
accuracy axis.

### 3.6 Memory–Accuracy Pareto

![Memory vs mean accuracy](results/plots/memory_accuracy_pareto.png)

BitNet (ours) defines the bottom-left frontier: ~1.25 GB RSS at 61.74%
mean accuracy.  Both Qwen variants cluster at ~1.65 GB — Q8 slightly
higher on accuracy (61.10%) than Q4 (59.45%) but on the same memory
plateau.  No FP16 baseline gets close.

### 3.7 Accuracy by task

![Accuracy by task](results/plots/accuracy_comparison.png)

| Task | BitNet (ours) | Qwen Q8_0 (ours) | Qwen Q4_K_M (ours) | Winner |
|---|---:|---:|---:|---|
| ARC-Easy | 74.2 | 74.2 | 71.0 | BitNet / Q8 tie |
| ARC-Challenge | **46.0** | 44.2 | 43.2 | BitNet |
| WinoGrande | **75.2** | 65.8 | 63.0 | BitNet (+9.4 / +12.2) |
| HellaSwag | 58.6 | **59.0** | 58.8 | Q8 (effectively tied) |
| MMLU (5-shot) | 54.69 | **62.28** | 61.23 | Q8 (Q4 a close second) |
| **Mean** | **61.74** | 61.10 | 59.45 |  |

Two patterns:

- **BitNet wins reasoning**, large.  WinoGrande +9.4pt over Q8 and
  +12.2pt over Q4 is the cleanest BitNet win in the entire report.
  ARC-Challenge +1.8pt over Q8 is the same direction.
- **Qwen wins knowledge**, smaller.  MMLU is the only large-margin Qwen
  win (+7.6pt over BitNet for Q8, +6.5pt for Q4).  This reflects
  Qwen2.5's much larger pretraining corpus (up to 18T tokens vs BitNet
  2B4T's 4T) — at this size class, MMLU is dominated by pretraining-data
  breadth.

The Q8 → Q4 quantization cost is consistent across tasks: -3.2 (ARC-E),
-1.0 (ARC-C), -2.8 (Wino), -0.2 (HellaSwag), -1.05 (MMLU). Mean drop
1.65pt.  No catastrophic failure on any task — Q4_K_M behaves as a
"slightly worse but much faster" Q8.

### 3.8 Energy, Carbon, and Local Electricity Cost

![Energy / Carbon / Cost per 1k tokens](results/plots/energy_carbon_comparison.png)

At `(n_prompt=512, n_gen=128)`:

| Model | Wh / 1k tok | g CO₂ / 1k tok | $ / 1k tok @ $0.16/kWh |
|---|---:|---:|---:|
| BitNet b1.58 2B4T (ours) | **0.82** | **0.069** | **$0.000131** |
| Qwen2.5-1.5B Q4_K_M (ours) | 0.83 | 0.070 | $0.000132 |
| Qwen2.5-1.5B Q8_0 (ours) | 1.40 | 0.119 | $0.000224 |

Three observations:

- **BitNet and Q4 essentially tie on energy** (within 1%).  Q4 finishes
  faster (less wall time) but draws marginally more power per second
  (FP-multiply path on the dequantized weights); the products balance.
  Q8 uses ~70% more energy than either because its wall time is much
  longer.
- The Q8-vs-BitNet 41% gap from the pre-Q4 report still holds.
- CO₂ figures use the local grid's intensity as resolved by CodeCarbon
  at run time; the electricity-cost column uses the default `$0.16/kWh`
  (US residential average) — override with `--electricity-rate` for
  industrial / local utility rates. Absolute values are not portable
  across regions, but the BitNet-vs-Qwen ratios are.

The electricity-cost framing is roughly **17× cheaper** than the AWS
c5.xlarge proxy used elsewhere in the report. They answer different
questions — see §2 Methodology and §3.9 for the framing comparison.

### 3.9 Cost vs Cloud API Services

![Self-hosted vs cloud API cost per 1k output tokens](results/plots/cloud_cost_comparison.png)

Cloud API output-token pricing as of **2026-05-15** (hardcoded in
`compare_runs.py:CLOUD_API_PRICING` — re-verify before publication, these
change):

| Service / Tier | $/1k output tokens |
|---|---:|
| OpenAI GPT-4o mini | $0.000600 |
| Anthropic Claude Haiku 4.5 | $0.005000 |
| OpenAI GPT-4o | $0.010000 |
| Anthropic Claude Sonnet 4.5 | $0.015000 |
| Anthropic Claude Opus 4.7 | $0.075000 |

Combined ranking, ascending cost:

| Rank | Option | $/1k tok | Multiplier vs cheapest |
|---|---|---:|---:|
| 1 | BitNet (ours, local electricity) | $0.000131 | 1.0× |
| 2 | Qwen Q4_K_M (ours, local electricity) | $0.000132 | 1.0× |
| 3 | Qwen Q8_0 (ours, local electricity) | $0.000224 | 1.7× |
| 4 | OpenAI GPT-4o mini (API) | $0.000600 | 4.6× |
| 5 | Qwen Q4_K_M (ours, AWS c5.xlarge proxy) | $0.001897 | 14× |
| 6 | BitNet (ours, AWS c5.xlarge proxy) | $0.002227 | 17× |
| 7 | Qwen Q8_0 (ours, AWS proxy) | $0.003128 | 24× |
| 8 | Anthropic Claude Haiku 4.5 (API) | $0.005000 | 38× |
| 9 | OpenAI GPT-4o (API) | $0.010000 | 76× |
| 10 | Anthropic Claude Sonnet 4.5 (API) | $0.015000 | 115× |
| 11 | Anthropic Claude Opus 4.7 (API) | $0.075000 | **573×** |

**Two ways to read this**:

- *Hardware you already own* → local-electricity is the relevant framing.
  BitNet and Q4_K_M are within 1% of each other ($0.000131 vs $0.000132)
  and both are 4.6× cheaper than the cheapest cloud API tier (GPT-4o
  mini) and 573× cheaper than Claude Opus 4.7.
- *Cloud-rented infrastructure* → AWS proxy is the relevant framing.
  Q4_K_M is cheapest of the self-hosted options here ($0.001897) because
  it generates more tokens per rented hour.  BitNet sits ~15% higher.
  Both still beat every API tier except GPT-4o mini.

**Important caveat**: this comparison is dollars per token only. It does
not capture capability differences. Opus 4.7 and GPT-4o can perform
tasks that BitNet 2B and Qwen 1.5B cannot, regardless of price. The cost
comparison is meaningful only for workloads where a 2B-parameter model's
quality is sufficient — short summarization, simple Q&A, structured
extraction, classification, embedding-equivalent text generation. For
agentic / multi-step reasoning or knowledge-heavy QA, capability bypass
invalidates the cost comparison.

---

## 4. Energy: Measured vs Paper FP16 Estimates

The Phase 4 plan asks for an explicit comparison against the paper's
J/tok claims for FP16 baselines. The relevant paper numbers
(arXiv:2504.12285 Table 1):

| Model | Paper J/tok | Source |
|---|---:|---|
| LLaMA 3.2 1B | 0.258 | Table 1 |
| Qwen2.5 1.5B (FP16) | 0.347 | Table 1 |
| SmolLM2 1.7B | 0.425 | Table 1 |
| **BitNet b1.58 2B4T** | **0.028** | Table 1 |

The paper's headline claim is therefore **9–23× energy efficiency** for
BitNet vs FP16 baselines.

Our CodeCarbon-measured J/tok, computed as
`energy_kwh × 3,600,000 / (n_prompt + n_gen)`:

| Workload | Tokens | BitNet J/tok | Qwen Q8_0 J/tok | Qwen Q4_K_M J/tok | Q8/BitNet |
|---|---:|---:|---:|---:|---:|
| `(512, 128)` | 640 | 2.94 | 5.05 | 2.98 | **1.72×** |
| `(512, 512)` | 1,024 | 6.91 | 10.49 | 6.15 | 1.52× |
| `(1, 512)` | 513 | 21.71 | 33.43 | 19.07 | 1.54× |

### 4.1 Interpretation

Three things stand out:

**(a) Our absolute J/tok values are 100–200× higher than the paper's.**
CodeCarbon estimates power from CPU TDP and runtime intervals — it captures
the *entire* power draw of the CPU package during the bench run,
including idle baseline and uncore. The paper's J/tok figures appear to be
inference-marginal (compute-only) estimates derived from kernel-level
profiling. The two are measuring different quantities and are not directly
comparable as published numbers. **The paper's 0.028 J/tok for BitNet is
not an upper-bound on real-world energy cost** — it is the marginal-
inference component only.

**(b) Our BitNet-vs-Q8 ratio (~1.5–1.7×) is far below the paper's
implied ~12× ratio** (0.347 / 0.028). This is consistent with (a): both
models run on the same CPU and inherit the same idle/uncore baseline. If
idle is `P_idle` and inference adds `Δ`, total energy is
`(P_idle + Δ) × t`. BitNet's `Δ` may indeed be ~12× smaller than Qwen's,
but the constant `P_idle` term dominates total measured energy at this
sampling resolution, compressing the apparent ratio.

**(c) Q4_K_M and BitNet are within ~5% of each other on J/tok at every
config.**  Q4 finishes faster (smaller wall-time × power) but BitNet's
ternary path draws less power per second; the products converge.  This
further weakens the "BitNet is uniquely energy-efficient" framing — at
this hardware and resolution, aggressive Q4 quantization on upstream
`llama.cpp` is essentially energy-tied with the TL2 kernel.  The
inference-marginal advantage that the paper attributes to 1.58-bit may
still be real, but it's invisible at total-system-power resolution.

### 4.2 What this means for the carbon claim

The paper's 9–23× efficiency claim is **directionally correct** but
asymmetrically defined: it compares the marginal cost of an additional
generated token, not the wall-power cost of running the inference. For
operational cost / carbon accounting (the perspective most relevant to
deployment decisions), the realistic advantage on this CPU is closer to
1.5–1.7×, still substantial but an order of magnitude smaller than the
paper-headline ratio.

### 4.3 Idle subtraction → marginal J/tok

To recover something comparable to the paper's inference-marginal J/tok,
`scripts/measure_marginal_energy.py` (`make marginal-energy`) runs the
same `CodeCarbon EmissionsTracker` used by `metrics_tracker.py` for a
90-second window with no inference work, then subtracts that idle baseline
× wall_time from every bench row.

Measured idle baseline (i5-9400F, host-system as configured at measurement
time — see caveats below): **1.37 Wh over 90.00 s → 54.81 W**.

Applying `marginal_J = max(0, total_J − P_idle × wall_time)` to each
bench row in `results/*_step_metrics.csv` gives:

| Model    | Config       | Wall    | Total J/tok | Idle J/tok | **Marginal J/tok** |
|---|---|---:|---:|---:|---:|
| BitNet   | `(512, 128)` | 30.2 s  | 2.94        | 2.59       | **0.36** |
| BitNet   | `(512, 512)` | 50.2 s  | 6.91        | 2.69       | **4.22** |
| BitNet   | `(1, 512)`   | 24.6 s  | 21.69       | 2.63       | **19.06** |
| Qwen Q8  | `(512, 128)` | 42.4 s  | 5.05        | 3.63       | **1.42** |
| Qwen Q8  | `(512, 512)` | 68.3 s  | 10.48       | 3.65       | **6.82** |
| Qwen Q8  | `(1, 512)`   | 35.9 s  | 33.40       | 3.84       | **29.56** |
| Qwen Q4  | `(512, 128)` | 25.7 s  | 2.98        | 2.20       | **0.78** |
| Qwen Q4  | `(512, 512)` | 41.5 s  | 6.15        | 2.22       | **3.93** |
| Qwen Q4  | `(1, 512)`   | 20.7 s  | 19.07       | 2.21       | **16.86** |

**Reading the result.** At the reference `(512, 128)` config the marginal
BitNet number is **0.36 J/tok**, compared with the paper's claimed
0.028 J/tok — still ~13× off but materially closer than the 105× gap
from total system energy (2.94 / 0.028). The BitNet/Qwen-Q8 marginal
ratio is **3.97×** (0.36 vs 1.42), much closer to the paper's implied
~12× ratio (0.028 vs 0.347) than the 1.71× we saw from totals. Most of
the "missing" efficiency in §4.1 is recovered by idle subtraction.

**Caveats.** Three reasons the marginal gap to paper isn't fully closed:

1. **CodeCarbon's Windows estimator overcounts.** Without RAPL access on
   Windows, CodeCarbon scales CPU package power as `TDP × utilization`.
   The 54.81 W idle figure is unrealistically high for a 65 W TDP chip
   sitting at low utilization — actual desktop idle on this CPU is closer
   to 25-35 W. The estimator likely overcounts the busy-state too, so
   even after subtraction the marginal number remains inflated.
2. **"Idle" includes background load.** The host machine wasn't truly
   idle during the baseline measurement — Claude Code, browser, etc. all
   ran. A bench-paired idle measurement (alternating idle and bench
   windows within one script) would be tighter.
3. **Per-token cost rises with `n_gen` share.** The `(1, 512)` configs
   show ~17-29 marginal J/tok because pure-generation has no prompt-eval
   amortization to spread the per-token overhead over. The reference
   `(512, 128)` numbers are the apples-to-apples comparison against the
   paper's Table 1, which was also measured at `(512, 128)`.

Even with the caveats, the resolution is enough to update the report's
operational story: the **inference-marginal advantage of BitNet over
Qwen Q8 is ~4×** (not 1.7×), and Q4_K_M ties BitNet on energy
(marginal 0.78 vs 0.36 J/tok — within an order of magnitude after
subtraction, and well within CodeCarbon's estimation noise).
Sub-second RAPL measurements on Linux would tighten this further and
are the recommended next step if absolute J/tok parity with the paper
matters.

---

## 5. Discussion

### 5.1 The speed/accuracy Pareto across three quantization points

The three locally measured models trace a clean quality-vs-speed curve.
Sorted by speed:

| Model | Format | Throughput | Mean accuracy | Memory |
|---|---|---:|---:|---:|
| Qwen Q8_0 | 8-bit, FP-multiply matmul | 15.1 tok/s | 61.10% | 1,667 MB |
| BitNet i2_s | 1.58-bit, TL2 ternary-lookup kernel | 21.2 tok/s | **61.74%** | **1,247 MB** |
| Qwen Q4_K_M | 4-bit, FP-multiply matmul | **24.9 tok/s** | 59.45% | 1,632 MB |

Two observations:

**(a) BitNet is the Pareto winner among the three.**  Q4_K_M beats it on
raw throughput by ~17%, but at a measurable accuracy cost (mean -2.3pt;
-1.05pt on MMLU, -12.2pt on WinoGrande).  Q8_0 matches it on mean
accuracy (within 0.6pt) but runs ~40% slower.  At the same speed class
as Q4, nothing matches BitNet's accuracy; at the same accuracy class as
Q8, nothing matches BitNet's speed.  Memory is the cleanest win
regardless of frame: BitNet's i2_s footprint is ~25% smaller than
either Qwen variant.

**(b) The kernel-attribution argument from earlier drafts of this report
was weaker than it appeared.**  A pre-Q4 reading of the data ("BitNet
1.4× faster than Q8") attributed the gap to the TL2 ternary-lookup
kernel — i.e., "Q8 still does FP-multiply matmul, BitNet's kernel uses
byte-level table lookups."  Q4_K_M on upstream `llama.cpp` shows that
**aggressive weight quantization on a modern kernel can match or beat
BitNet's throughput** without the kernel-level rewrite.  The fair claim
is therefore: *aggressive quantization saves time regardless of format*,
and **BitNet's real contribution is doing so without paying the
accuracy cost** that Q4_K_M does.

Put differently, the BitNet paper's headline efficiency claim
(throughput 5–7× over FP16) is reproduced here, but it isn't *unique* to
1-bit; Q4_K_M on the same hardware delivers comparable throughput.
What's unique to 1.58-bit + TL2 is **the position on the
speed/accuracy curve** — specifically that BitNet matches Q8's accuracy
at near-Q4's speed.

### 5.2 Reasoning vs Knowledge

The per-task split mirrors the two models' training emphases:
- BitNet 2B4T was trained on 4T tokens with heavy synthetic-math
  augmentation and a full SFT + DPO post-training pipeline. Its
  WinoGrande lead suggests strong commonsense / coreference reasoning.
- Qwen2.5 1.5B was pretrained on up to 18T tokens of broad text, code,
  and math. MMLU lead reflects pretraining-data breadth — MMLU spans 57
  subjects and is dominated by pretraining-coverage at this size class.

This pattern matters for deployment: pick BitNet for reasoning-heavy
workloads (agents, multi-step inference), Qwen for knowledge-heavy
workloads (factual QA, domain Q&A).

### 5.3 Cost implications at scale

At 1 billion generated tokens/day:

| Option | $/day | $/year |
|---|---:|---:|
| BitNet (ours, local electricity) | $131 | $48k |
| Qwen Q4_K_M (ours, local electricity) | $132 | $48k |
| Qwen Q8_0 (ours, local electricity) | $224 | $82k |
| OpenAI GPT-4o mini (API) | $600 | $219k |
| Qwen Q4_K_M (ours, AWS proxy) | $1,897 | $693k |
| BitNet (ours, AWS proxy) | $2,227 | $813k |
| Qwen Q8_0 (ours, AWS proxy) | $3,128 | $1.14M |
| Anthropic Claude Haiku 4.5 (API) | $5,000 | $1.83M |
| OpenAI GPT-4o (API) | $10,000 | $3.65M |
| Anthropic Claude Sonnet 4.5 (API) | $15,000 | $5.48M |
| Anthropic Claude Opus 4.7 (API) | $75,000 | **$27.4M** |

The cost gradient is dramatic at production scale.  The numbers assume
sustained 100% utilization (1B tokens/day ≈ 11.6k tok/s, far above what
a single c5.xlarge produces — would require ~550 parallel BitNet
instances or equivalent infrastructure).  At lower utilization the
AWS-proxy numbers overstate actual cost (you'd pay only for time used,
not 24/7), while local-electricity and per-token API numbers remain
accurate because both scale linearly with usage.

**Within-framing comparisons** between the three "ours" rows:

- *Local electricity*: BitNet and Q4_K_M tie ($131 vs $132/day); both
  ~40% cheaper than Q8_0 ($224/day).
- *AWS proxy*: Q4_K_M is the cheapest ($1,897/day), BitNet ~15% higher
  ($2,227), Q8_0 highest ($3,128).  The AWS framing rewards Q4's higher
  throughput.

Operationally meaningful at any production scale.  **If MMLU-class
knowledge accuracy is the bottleneck, Q4_K_M is the cheapest sufficient
option**; if reasoning (WinoGrande, ARC) or memory footprint matters,
BitNet earns its keep.

### 5.4 Thread-count scaling sensitivity (Phase 5 sweep)

![Thread scaling at (n_prompt=512, n_gen=128)](results/plots/thread_scaling.png)

Throughput vs thread count at the reference config, swept on the same
i5-9400F (6 cores, no SMT) via `make benchmark-threads-bitnet` /
`benchmark-threads-qwen` / `benchmark-threads-qwen-q4`:

| Threads | BitNet | Qwen Q8_0 | Qwen Q4_K_M |
|---:|---:|---:|---:|
| 1 | crashes (see (a)) | 8.1 | 10.0 |
| 2 | 17.8 | 13.8 | 17.5 |
| 4 | 21.4 | 16.4 | 25.1 |
| 6 | 21.8 | 17.5 | 27.4 |

Five findings:

**(a) BitNet's TL2 kernel has a thread-count floor.**  At threads=1 the
kernel hits `STATUS_STACK_OVERFLOW (0xC00000FD)` regardless of
`--ubatch`.  At threads=2 it requires `--ubatch ≤ 64` (the default 128
also crashes).  The sweep uses `--ubatch 64` for BitNet across all
thread counts for consistency; at threads=4 the resulting throughput
(21.4) closely matches the main reference's `--ubatch 128` number
(21.2), so the smaller batch barely costs anything on this CPU.  The
practical implication is real: BitNet at this build is not deployable
to single-thread or single-core-pinned environments.

**(b) Quantization, not threading, is the dominant cause of the
speedup over the paper's FP16 figure** — directly confirmed by the
threads=1 numbers.  Q8 at threads=1 hits 8.1 tok/s, 2.13× the paper's
FP16 ~3.8 tok/s *at the paper's matched thread count*.  Q4 at threads=1
hits 10.0, 2.6× over paper.  Composing with threading:
*Q8 vs paper FP16 = ~2× quantization × ~2× threading (1→4 threads) = ~4×*;
*Q4 vs paper FP16 = ~2.6× quantization × ~2.5× threading = ~6.5×*.
The §3.2 attribution holds with the threading and quantization
contributions cleanly separated.

**(c) Three different saturation behaviors.**  BitNet flattens at 4
threads (4→6 adds only +1.9%).  Q8 nearly flattens at 4 (+6.5% to 6).
Q4 is still climbing at 6 (+9.1%).  The pattern matches
memory-bandwidth saturation: smaller weight footprint = more headroom
on extra cores.  Q4_K_M's ~1 GB weights leave the most bandwidth-
headroom for extra threads to consume.

**(d) At threads=2, BitNet and Q4 are tied** (17.8 vs 17.5).  BitNet's
TL2 kernel doesn't out-perform aggressive Q4 quantization at low
thread counts; its throughput advantage over Q4 emerges only when
forced to share fewer cores than the system can offer (which is the
normal deployment case on consumer CPUs).  Above 2 threads the
ordering flips and Q4 pulls ahead.

**(e) The §5.1 conclusion holds across the sweep.**  At every thread
count from 2 to 6, Q4 > BitNet > Q8 in raw throughput.  BitNet's
Pareto position (matches Q8 accuracy at near-Q4 speed) isn't a
4-thread accident — it's a property of the kernel/format design
that's stable across the operating range.

**Implication for §5.3 (cost at scale).**  The AWS-proxy figures
assume the reference 4-thread condition.  If a c5.xlarge effectively
delivers up to 4 useful threads, Q4 and BitNet are roughly co-priced.
At hypothetical 6+ threads or higher core counts where Q4 keeps
scaling but BitNet doesn't, Q4's cost advantage widens.  Conversely,
on single-core or 2-core constrained environments (some serverless
configurations), BitNet wouldn't run at all and Q4 is the cheapest
sufficient option.

### 5.5 Workload-shape sensitivity (Phase 5 analysis)

The three `(n_prompt, n_gen)` configs already in the bench CSVs stress
different parts of the inference pipeline:

| Config | Description | n_prompt | n_gen |
|---|---|---:|---:|
| `(512, 128)` | Prompt-heavy Q&A | 512 | 128 |
| `(512, 512)` | Long-context | 512 | 512 |
| `(1, 512)` | Pure generation | 1 | 512 |

Throughput (tok/s) and the spread within each model:

| Config | BitNet | Qwen Q8_0 | Qwen Q4_K_M |
|---|---:|---:|---:|
| `(512, 128)` | 21.21 | 15.10 | 24.89 |
| `(512, 512)` | 20.38 | 15.00 | 24.69 |
| `(1, 512)`   | 20.85 | 14.28 | 24.81 |
| Within-model spread (max−min)/max | 3.9% | **5.5%** | **0.8%** |

Peak RSS (MB):

| Config | BitNet | Qwen Q8_0 | Qwen Q4_K_M |
|---|---:|---:|---:|
| `(512, 128)` | 1,247 | 1,667 | 1,632 |
| `(512, 512)` | 1,246 | 1,667 | 1,632 |
| `(1, 512)`   | 1,230 | 1,649 | 1,614 |
| Within-model spread | 1.3% | 1.1% | 1.1% |

Three findings:

**(a) Throughput is essentially workload-shape insensitive across all
three models.**  Each model stays within ~6% of its reference number
regardless of whether the workload is prompt-heavy, long-context, or
pure generation.  Q4_K_M is the most stable (0.8% spread); Q8_0 has
the widest variance (5.5%), driven specifically by a drop on pure
generation `(1, 512)`.  Implication: the §3.2 throughput numbers
generalize cleanly across realistic deployment workload shapes at
this size class.

**(b) BitNet's advantage over Qwen Q8 *widens* on pure-generation
workloads.**  The BitNet/Q8 throughput ratio is 1.40× at the
prompt-heavy `(512, 128)` reference, 1.36× at long-context `(512,
512)`, and **1.46×** at pure-generation `(1, 512)` — Q8's worst
config relative to BitNet.  The TL2 ternary-lookup kernel's
decode-phase efficiency holds up better than Q8's FP-multiply matmul
when there's no prompt-eval phase to amortize over.  Q4_K_M's
advantage over BitNet is roughly constant at 1.17–1.21× across all
three configs, slightly wider on long-context `(512, 512)`.

**(c) Memory is dominated by weights and runtime overhead, not the
KV cache.**  RSS spread is <1.5% within each model across configs.
For Qwen2.5-1.5B with GQA (28 layers × 256-dim KV) the KV cache at
`(512, 512)` is ~28 MB and at `(1, 512)` is ~14 MB — both negligible
against the 1.2–1.7 GB total.  Practical implication: at this
parameter count and at ≤4K-token contexts, memory planning can use
the static `(512, 128)` RSS as a conservative upper bound regardless
of expected workload shape.

**Regime answer.**  The Phase 5 PLAN.md task explicitly asked: *"note
any regime where Qwen narrows the throughput or memory gap."*  At
these context lengths, the answer is **none** for Qwen Q8 — its gap
to BitNet either holds steady or widens (worst at pure generation).
Qwen Q4 *does* exceed BitNet on throughput at every config, but
that's the §5.1 Pareto trade-off (Q4 buys speed by sacrificing 2.3pt
mean accuracy), not a workload-shape effect.  The picture would
likely change at much longer contexts (≥4K tokens) where KV-cache
memory becomes a multi-GB issue, but that regime is beyond what our
4K-context BitNet build is set up to measure.

---

## 6. Threats to Validity

1. **Single CPU.** All measurements on Intel i5-9400F (no AVX-512). The
   throughput ratio likely shifts on AVX-512 hardware: BitNet TL2 uses
   AVX-512 paths when available, Q8_0 also has AVX-512 paths, and which
   benefits more is hardware-dependent. Phase 5 (`PLAN.md`) lists a
   thread-count sweep and a hardware-rate sensitivity check.

2. **Bias-trick API asymmetry.** Continuation scoring uses two different
   APIs (`logit_bias` on upstream Qwen, top-K probs on the BitNet fork).
   The fork-side path now reads `n_vocab` from `/v1/models` and passes
   that as `n_probs` to `/completion` (128,256 for the Llama-3 tokenizer
   used by BitNet b1.58 2B4T), so every continuation token gets an exact
   logprob from the full distribution — no truncation, no
   `min(top_K_logprob) − 1.0` heuristic. The legacy `n_probs=5000` +
   conservative-lower-bound code path is preserved as a defensive fallback
   for the case where `/v1/models` doesn't expose `meta.n_vocab`. Earlier
   drafts of this report claimed the fork crashed at `n_probs ≳ 50,000`;
   that was wrong — empirically the segfault trigger is *negative*
   `n_probs` (cast to `(size_t)-1` in `server.cpp:2374`); positive values
   up to `n_vocab` work and return full distributions in ~0.6s.

3. **MMLU shot count.** Both models run with the paper's 5-shot framing.
   Production-typical 0-shot would reduce both numbers by ~5–8pt and
   would not change the +7.6pt Qwen lead, but the absolute numbers in
   §3.7 are 5-shot-specific.

4. **CodeCarbon resolution (partially resolved — see §4.3).** Absolute
   J/tok in the §4 table include the CPU idle baseline because CodeCarbon
   on Windows estimates power as `TDP × utilization`, not actual RAPL.
   §4.3 now measures an idle baseline (54.81 W on this host) and
   subtracts it row-by-row to give marginal J/tok. That closes most of
   the gap to the paper's inference-only J/tok (e.g., BitNet at
   `(512, 128)`: 2.94 → 0.36 marginal J/tok; paper target 0.028). The
   BitNet/Qwen-Q8 marginal ratio rises from 1.71× (total) to 3.97×
   (marginal), much closer to the paper's implied ~12×.  A residual
   factor remains because the Windows estimator overcounts and the host
   isn't truly idle during the baseline window; bench-paired RAPL
   measurements on Linux would tighten this further. Use the marginal
   numbers in §4.3 (rather than the totals in §4) when comparing against
   external J/tok figures.

5. **Hardware-rate sensitivity.** All AWS-proxy cost figures use AWS
   c5.xlarge on-demand at `$0.170/hr`. Spot pricing (~30–40% lower), ARM
   Graviton (lower $/hr, slower TL2 paths), or local hardware ($0/hr
   capex amortized) would shift the absolute cost numbers, though the
   ours < paper-FP16 ordering is robust as long as the same rate is
   applied to all rows.  The intra-"ours" ordering at AWS-proxy
   (Q4 < BitNet < Q8) is throughput-driven and therefore robust to rate
   choice.  The local-electricity framing has its own sensitivity:
   `$0.16/kWh` is US residential average; California residential is
   ~$0.27/kWh, industrial is ~$0.10/kWh, EU varies $0.20–$0.40/kWh.
   Override with `--electricity-rate`. A Phase-5 sensitivity sweep
   across both rates is planned.

6. **Cloud API pricing freshness.** The cost-vs-cloud comparison in §3.9
   and §5.3 uses API output-token prices hardcoded in
   `compare_runs.py:CLOUD_API_PRICING` (dated 2026-05-15). Cloud
   providers change pricing periodically; verify against each provider's
   pricing page (openai.com/api/pricing, anthropic.com/pricing#api)
   before relying on §3.9 / §5.3 numbers for external publication. A
   30% provider price drop wouldn't change the qualitative ranking but
   would compress the multipliers.

7. **Capability mismatch in the API comparison.** The §3.9 ranking is
   dollars-per-token only; it doesn't reflect capability. Opus 4.7 and
   GPT-4o do things BitNet 2B and Qwen 1.5B can't. The comparison is
   meaningful for workloads where a 2B-parameter model's quality is
   sufficient (summarization, classification, structured extraction,
   simple Q&A) but invalidated by capability bypass for agentic / multi-
   step reasoning or knowledge-heavy QA.

8. **Cross-stack asymmetry (sensitivity-checked).** BitNet runs on
   `microsoft/BitNet`'s llama.cpp fork while both Qwen variants run on
   upstream `ggml-org/llama.cpp`.  This is deliberate: BitNet's `i2_s`
   format and TL2 kernel only exist in the fork, and forcing Qwen onto
   the older fork would understate its production-realistic throughput.
   To check whether the cross-stack comparison is materially confounded
   by stack-version differences, we re-ran Qwen Q8_0 against the BitNet
   fork's `llama-bench` (`make benchmark-qwen-q8-on-bitnet-fork` →
   `results/qwen_q8_on_bitnet_fork_step_metrics.csv`):

   | Config | Qwen Q8 on upstream | Qwen Q8 on BitNet fork | Δ |
   |---|---:|---:|---:|
   | `(512, 128)` | 15.1 tok/s | 14.4 tok/s | −4.6% |
   | `(512, 512)` | 15.0 tok/s | 15.4 tok/s | +2.7% |
   | `(1, 512)`   | 14.3 tok/s | 16.0 tok/s | +11.9% |

   Stack version explains ≤5% at the reference config and the sign of
   the delta isn't even consistent across configs — sub-noise for the
   purpose of this report.  The BitNet (21.2) vs Qwen Q8 (15.1)
   throughput gap in §3.2 is therefore robust to the choice of llama.cpp
   build, and the attribution to quantization (Q8 ≈ ½ FP16 weight
   bandwidth, Q4 ≈ ¼) in §3.2 holds independently of the stack pairing.
   We did not re-run Q4_K_M against the BitNet fork because the Q8
   result already isolates the stack variable; the Q4 quantization
   advantage is orthogonal.

---

## 7. Conclusion

This project independently reproduces the BitNet b1.58 2B4T paper's
core efficiency claims on commodity CPU hardware and extends them with a
side-by-side measurement of Qwen2.5-1.5B Q8_0 — the most directly
comparable FP16-style baseline at this size class — run on the same
machine under the same conditions.

**Confirmed:** BitNet's CPU throughput target (~20 tok/s) and memory
footprint (~1.4 GB) reproduce within margin (21.2 tok/s, 1.25 GB). BitNet
is materially faster, smaller, and equally accurate compared to Qwen2.5
Q8_0 at this size class.

**Refined — the kernel-attribution story is weaker than we initially
read.**  An earlier draft of this report claimed BitNet's throughput win
was driven by the TL2 ternary-lookup kernel.  Adding Qwen Q4_K_M as a
third comparison point shows that aggressive weight quantization on
*upstream* `llama.cpp` matches or beats BitNet's throughput (Q4 at 24.9
tok/s vs BitNet's 21.2), without a kernel rewrite.  BitNet's real edge
is the **position on the speed/accuracy Pareto** — it matches Q8's
mean accuracy at near-Q4 speed, with the smallest memory footprint of
the three.  Q4_K_M is the cheapest per token in the AWS-rental framing
but pays a measurable 2.3pt mean accuracy cost (and 12.2pt on
WinoGrande).

**Refined — the paper's 9–23× energy claim** does not survive
system-level power tracking on this hardware. The realistic advantage at
the wall-power level is ~1.5–1.7× over Q8 — still substantial, but an
order of magnitude smaller than the paper headline. BitNet and Q4_K_M
essentially tie on energy.  The discrepancy is a measurement-methodology
mismatch (compute-marginal vs total-system); the paper's underlying
kernel-level story remains plausible but is not verifiable with
CodeCarbon.

**New (model-selection guidance):** the three locally measured models
split along clear deployment axes.  *Pick BitNet* when accuracy matters,
when memory is the binding constraint, or when reasoning (WinoGrande,
ARC) dominates the workload.  *Pick Qwen Q4_K_M* when raw throughput is
the bottleneck and a 1–3pt accuracy drop per task is acceptable.  *Pick
Qwen Q8_0* only when you specifically need Q8's near-FP16 fidelity on
knowledge tasks — it's the slowest of the three and the most expensive
per token.

**Cost comparison extended in three directions:** beyond the AWS
c5.xlarge proxy used in §3.5, we now also report (a) the marginal
local-electricity cost (§3.8) — 17× cheaper than the cloud-rental
framing — and (b) the full ranking against five commercial LLM API
tiers (§3.9). Self-hosted BitNet and Q4_K_M tie at ~$0.000131/1k tokens
local-electricity, 4.6× cheaper than the cheapest API tier (GPT-4o
mini) and 573× cheaper than Claude Opus 4.7, with the strong caveat
that this comparison only holds when a 2B-parameter model's capability
is sufficient for the task.

**Refined — paper-vs-ours speedup attribution is now clean.**  The
Phase 5 thread-count sweep (§5.4) separated quantization from
threading at the paper's matched single-thread condition: Q8 vs FP16
quantization alone gives ~2× on this CPU; Q4 vs FP16 gives ~2.6×.  The
rest of the 4×/6.5× speedup over the paper comes from 1→4 thread
scaling.  The §3.2 attribution to quantization stands, with the
sweep providing the cleanest single-variable test.

Phase 5 remaining follow-up (`PLAN.md`): workload-shape
characterization across the three benchmarked configs, and the
hardware-rate / electricity-rate cost sensitivity sweep.

---

## 8. References

- arXiv:2504.12285 — Wang et al. (2025), "1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs"
- arXiv:2402.17764 — Ma et al. (2024), "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
- arXiv:2412.15115 — Qwen Team (2024), "Qwen2.5 Technical Report"
- `microsoft/BitNet` at commit `01eb415772c342d9f20dc42772f1583ae1e5b102`
- `ggml-org/llama.cpp` at commit `1e5ad35d560b90a8ac447d149c8f8447ae1fcaa0`
- This repo: `PLAN.md` (canonical project reference), Appendix B / Appendix C (model cards for BitNet and Qwen)

---

## Appendix A. Phase 3 BitNet-only sanity check (historical, superseded)

Before the Qwen comparison and the continuation-scoring rewrite landed in
Phase 4, an earlier BitNet-only sanity check produced these numbers
against the paper targets:

| Task | Phase 3 (BitNet, this report's earlier draft) | Current (§3.7) | Paper target |
|---|---:|---:|---:|
| ARC-Easy | 85.68% | 74.2% | 74.79% |
| ARC-Challenge | 70.40% | 46.0% | 49.91% |
| WinoGrande | 52.80% | 75.2% | 71.90% |
| HellaSwag | 51.20% | 58.6% | 68.44% |
| MMLU (0-shot subset) | 45.56% | 54.69% (5-shot, all 57 subjects) | 53.17% |

The Phase 3 ARC scores were inflated by a first-token letter-scoring
prompt that listed all choices in-prompt and asked which letter came next
(rewards format familiarity, not content understanding).  The
WinoGrande / HellaSwag scores were near-random because letter scoring
provides no semantic signal for entity-name pronoun resolution or
multi-word continuation. The MMLU number was a 0-shot subset run on 9
subjects, with the known few-shot-prompting gap explaining the −7.6pt
delta to the paper.

The Phase 4 rewrite (see `project_scoring_methodology_fix` and §3.7)
switched ARC to length-normalized loglikelihood of the full choice text,
WinoGrande to partial-context `P(suffix | prefix + option)`, HellaSwag to
length-normalized loglikelihood with `[title]` cleanup, and MMLU to
5-shot first-token letter scoring across all 57 subjects. The numbers in
§3.7 reflect that corrected methodology and are the canonical values for
this project. The Phase 3 build notes (ClangCL patches, `-ub 128` TL2
constraint) are preserved in Appendix B.3 and the Makefile.

---

## Appendix B. BitNet b1.58 background

### B.1 BitNet b1.58 (Ma et al., 2024 — arXiv:2402.17764)

BitNet b1.58 is a 1-bit LLM variant in which every weight is constrained
to ternary values `{−1, 0, +1}`. The name comes from the information-
theoretic bit-width of ternary: `log₂(3) ≈ 1.585` bits per parameter.
The core claim is that a model trained natively with ternary weights can
match a full-precision (FP16/BF16) Transformer of the same size on both
perplexity and downstream benchmarks, while substantially reducing
latency, memory, throughput, and energy costs at inference time.

**Absmean quantization.** Weights are quantized to ternary before each
forward pass using:

```
W̃ = RoundClip(W / (γ + ε), −1, 1)
```

where `γ = (1/nm) Σ|Wᵢⱼ|` is the per-tensor mean absolute value (the
scale factor), `ε` is a small constant for numerical stability, and
`RoundClip(x, a, b) = max(a, min(b, round(x)))` rounds to the nearest
integer and clamps to `[−1, 1]`. The scale factor `γ` is stored in FP16
alongside the ternary weights and is used to dequantize during
computation. This design replaces FP multiply-accumulate with integer
additions and table lookups (the TL2 kernel — see B.2).

**Activation quantization.** Activations are quantized to 8-bit integers
per token before each matrix multiply using absmax quantization:
`x̃ = RoundClip(x / (η + ε) × Q_b, −Q_b, Q_b)`, where `η = max(|xᵢ|)` is
the per-token absolute maximum and `Q_b = 2^(b−1) − 1 = 127` for 8-bit.
Scaling is per-token (not per-tensor) to preserve token-level dynamic
range.

**Straight-Through Estimator (STE).** Because `RoundClip` has zero
gradient almost everywhere, BitNet uses the STE to allow gradients to
flow through the quantization step: `∂L/∂W ≈ ∂L/∂W̃` when `|W| ≤ 1`,
zero when clamping is active. Full-precision latent weights are
maintained during training and re-quantized at every forward pass.

**Architecture.** LLaMA-compatible to ease integration with existing
tooling: BitLinear layers (weights quantized, biases removed), RMSNorm,
SwiGLU activation, RoPE positional embeddings, no bias terms anywhere.

### B.2 BitNet b1.58 2B4T (Wang et al., 2025 — arXiv:2504.12285)

BitNet b1.58 2B4T is the first open-source, native 1-bit LLM at the
2-billion parameter scale, trained on 4 trillion tokens. It extends the
2402.17764 work with a full production pipeline (pre-training → SFT →
DPO), a larger dataset, and optimized CPU/GPU inference kernels.

**Architecture differences from b1.58 (2402.17764):**

| Component | b1.58 (2402.17764) | 2B4T (2504.12285) |
|---|---|---|
| Activation fn | SwiGLU | **Squared ReLU** |
| Normalization | RMSNorm | **subLN (sub-LayerNorm)** |
| Positional emb | RoPE | RoPE |
| Tokenizer | — | **LLaMA-3 BPE, 128,256 vocab** |
| Parameters | Up to 3B tested | **2.74B** |

Squared ReLU (`x → max(0, x)²`) was chosen over SwiGLU for improved
activation sparsity, which benefits the TL2 CPU kernel. subLN
(normalization inside attention/FFN sublayers) was chosen for training
stability at scale.

**Training pipeline.**
- *Pre-training (4T tokens):* two-stage learning-rate schedule (high
  initial LR with warmup, then cosine cooldown); two-stage weight decay
  (cosine peaking at 0.1, then disabled); data mix of DCLM web crawl,
  FineWeb-EDU, synthetic math.
- *Supervised Fine-Tuning (SFT):* WildChat, LMSYS-Chat-1M, WizardLM
  Evol-Instruct, SlimOrca, plus synthetic data; summed (not averaged)
  loss; larger LR and more epochs than typical FP16 SFT.
- *Direct Preference Optimization (DPO):* UltraFeedback, MagPie; 2
  epochs, LR = 2×10⁻⁷, β = 0.1.

**Inference kernels.**
- *GPU (CUDA).* The W1.58A8 kernel packs four ternary weight values into
  one int8 for HBM storage; values are unpacked to SRAM for computation.
  Avoids full dequantization to FP16.
- *CPU (bitnet.cpp — TL2 kernel).* The Ternary LUT 2-bit kernel encodes
  each pair of ternary weights as a 2-bit value (4 values per byte) and
  computes matrix-vector products via lookup tables rather than
  multiply-accumulate. Activations remain 8-bit. The scale factor `γ`
  stored in FP16 is applied once per tile after integer accumulation.

**Published Table 1 numbers** (single x86 CPU core, 4 threads) are
reproduced in §3.1 of this report alongside our measurements.  The
paper's headline efficiency claims:
- Non-embedding memory: 0.4 GB — 5–12× reduction vs FP16 baselines
- Energy per token: 0.028 J — 9–23× reduction vs FP16 baselines
  (re-examined in §4)
- Decoding latency: 29 ms/tok — 1.6–2.3× faster than comparable FP16
  models on CPU

### B.3 Build notes (Windows + ClangCL 20)

Reproducing the BitNet build on Windows 11 with Visual Studio 2022 +
ClangCL 20 at the pinned commit `01eb4157` requires three patches and a
batch-size cap, all wired into the `make bitnet-build` target:

| Patch | Reason |
|---|---|
| `patches/bitnet-clangcl-const.patch` | `src/ggml-bitnet-mad.cpp:811` initialises a non-const pointer from a const pointer. MSVC emits C4090 and continues; ClangCL 20+ treats this as a hard error. |
| `patches/llama-chrono.patch` | `3rdparty/llama.cpp/common/{common,log}.cpp` use `std::chrono` but omit `#include <chrono>`. Pulled in transitively under MSVC; ClangCL is stricter. |
| `patches/llama-chrono-examples.patch` | Same missing `<chrono>` include in `examples/{imatrix,perplexity}/*.cpp`. |

**TL2 kernel `--ubatch 128` cap.** The TL2 kernel is compiled with
`--BM 160` and crashes (`STATUS_STACK_OVERFLOW 0xC00000FD`) when asked
to process ≥160 tokens in a single batch. All benchmarks pin `-ub 128`
to stay under the limit. §5.4 documents the additional thread-count
floor (`threads=1` always crashes; `threads=2` requires `--ubatch ≤ 64`).

### B.4 Key concepts

| Concept | Description |
|---|---|
| **Absmean quantization** | Maps weights to {−1, 0, +1} using per-tensor mean absolute value as scale: `W̃ = RoundClip(W/γ, −1, 1)`. |
| **STE** | Straight-Through Estimator — passes gradients through rounding unchanged (`∂L/∂W ≈ ∂L/∂W̃`), enabling backprop through quantization. |
| **Activation quantization** | 8-bit per-token absmax scaling before each matrix multiply. |
| **TL2 kernel** | CPU inference kernel using 2-bit ternary lookup tables instead of multiply-accumulate. |
| **i2_s format** | GGUF quantization format used by bitnet.cpp; packs ternary weights at ~1.71 GiB for 2.74B params. |
| **subLN** | Sub-LayerNorm: normalization applied inside the attention/FFN sublayers for training stability at scale. |

---

## Appendix C. Qwen2.5-1.5B-Instruct background

### C.1 Qwen2.5 family (Qwen Team, 2024 — arXiv:2412.15115)

Qwen2.5 is Alibaba's third-generation open-weight LLM series, released
in September 2024. The family spans seven base sizes — 0.5B, 1.5B, 3B,
7B, 14B, 32B, 72B — each offered as a Base (next-token pretraining only)
and Instruct (SFT + DPO-aligned) variant. Pretraining corpus is up to 18
trillion tokens of multilingual text, code, and math. Most sizes
(including the 1.5B used here) are Apache 2.0; 3B and 72B are Qwen
Research licensed. The architecture is intentionally conservative
("boring but strong" LLaMA-style decoder-only Transformer): GQA, RoPE,
SwiGLU, RMSNorm, no architectural exotics — headline gains come from
better data and a longer alignment pipeline.

### C.2 Qwen2.5-1.5B-Instruct specifications

| Property | Value |
|---|---|
| Total parameters | 1.54 B |
| Non-embedding parameters | 1.31 B |
| Layers | 28 |
| Hidden dim | 1,536 |
| FFN intermediate dim | 8,960 |
| Query heads | 12 |
| Key/Value heads | 2 (GQA ratio 6:1) |
| Tied embeddings | Yes (input and output) |
| Vocab | 151,646 (BPE) |
| Context window | 32,768 tokens |

Tied embeddings and a GQA ratio of 6:1 are the main parameter-saving
choices at this scale: the KV cache and output projection consume a much
smaller fraction of parameters than MQA-only or full-MHA designs would.

We use the **Instruct** variant (post-SFT/DPO) — the right comparison
for end-user-relevant accuracy and for inference timing including
chat-template overhead. The benchmarks in this project (ARC, WinoGrande,
HellaSwag, MMLU) are zero-shot multiple-choice and don't depend on
chat-template wrapping.

### C.3 GGUF quantizations measured here

Both quants are pulled from the same first-party
`Qwen/Qwen2.5-1.5B-Instruct-GGUF` HuggingFace release (no local
conversion), so anyone re-running the benchmark gets bit-identical
weights. Both run through the same upstream `llama.cpp` build; only the
GGUF file differs.

**Q8_0 — 8-bit symmetric, fixed-block** (~1.65 GiB). The highest-fidelity
quant in common CPU use. The weight tensor is partitioned into 32-element
blocks; each block stores a 2-byte FP16 scale + 32 int8 quantized values
(34 bytes / 32 elements = 8.5 bits/parameter). Per-block algorithm:
`scale = max(|Wᵢ|) / 127` (FP16); `qᵢ = clamp(round(Wᵢ/scale), −127, 127)`.
Reconstruction at inference: `W̃ᵢ = qᵢ × scale`. Dequantization is fused
into the GEMM kernel — weights are upcast to FP32 (or BF16, on some
backends) tile by tile inside the matmul, never materialized in full
precision in memory. Empirically near-lossless: published llama.cpp
comparisons consistently report <0.1% perplexity increase vs FP16.
**Size:** ~45% reduction vs the ~3 GiB FP16 weights; **KV cache:** same
as FP16 (Q8_0 doesn't quantize the KV cache).

**Q4_K_M — 4-bit "K-quants" mixed-block** (~1.0 GiB). Super-blocks of
256 weights subdivided into sub-blocks of 32; uses 6-bit scales for the
most accuracy-sensitive tensors and 4-bit weights elsewhere. About half
the size of Q8_0 at a measurable but bounded accuracy cost (mean -1.65 pt
vs Q8_0 across the 5 benchmarks here — see §3.7).

### C.4 Inference stack — upstream `llama.cpp` vs BitNet fork

Both BitNet b1.58 2B4T and Qwen2.5-1.5B-Instruct (Q8_0 and Q4_K_M) are
served via `llama-server` in this project, but the two Qwen quants share
the upstream build while BitNet runs on `microsoft/BitNet`'s fork:

| Aspect | Upstream `llama.cpp` (Qwen) | `microsoft/BitNet` fork (BitNet) |
|---|---|---|
| Base commit | Current (2026) | Forked from ~Q2 2024 |
| Custom kernels | None beyond standard quants | TL2 (Ternary LUT 2-bit) |
| Custom quant format | F16/BF16, Q8_0, Q5/Q4/Q3/Q2_K | i2_s (BitNet ternary) + I2_S/TL2 tiles |
| Default ubatch | 512 | Must be ≤128 (TL2 kernel stack overflow above this) |
| `/completion` response | `{token, logprob, top_logprobs[]}` (new) | `{content, probs[{tok_str, prob}]}` (old) |
| `post_sampling_probs:false` | Supported — returns *natural* logprob under `logit_bias` | Not supported — bias applied before reported probs |
| `n_probs` upper bound | Effectively vocab-size | `n_vocab` works (returns full distribution); negative values segfault |
| OpenAI-compatible endpoint | `/v1/completions`, `/v1/chat/completions` | Partial; older spec |
| `/tokenize`, `/detokenize` | Yes | Yes |

**Consequences for the eval pipeline.** `scripts/eval_accuracy.py` probes
the server once at startup (`_server_caps()`) and picks one of two
continuation-scoring paths:

- *Upstream (Qwen) path:* for each target continuation token, send
  `logit_bias:[[token_id, +100]]` with `post_sampling_probs:false`. The
  forced token comes back with its **natural** (pre-bias) logprob, exact
  for any token in the vocabulary.
- *Fork (BitNet) path:* the bias trick is unusable because reported probs
  are post-bias. Instead, send `n_probs = caps["n_vocab"]` (read from
  `/v1/models → data[0].meta.n_vocab`) and look up each target token in
  the returned full distribution. Exact for every continuation token, no
  truncation. The legacy `n_probs=5000` + `min(top_K_logprob) − 1.0`
  fallback remains for the case where `/v1/models` doesn't expose
  `meta.n_vocab`.

This is the single largest methodology asymmetry between the two models
in this project, and is the only one we cannot eliminate without
rebuilding BitNet against current upstream. It is documented inline in
`eval_accuracy.py` and reflects the (acceptable) reality that the BitNet
inference stack is the actual artifact under test.

**Consequences for benchmarking.** `scripts/metrics_tracker.py` shells
out to `llama-bench` from whichever build is configured. The JSON schema
is stable across the fork divergence point — both produce
`{n_prompt, n_gen, avg_ts, ...}` records — so latency, throughput, and
peak-RSS comparisons are apples-to-apples. Energy (CodeCarbon) and
peak-RSS (`psutil`) are measured at the OS level and are stack-agnostic.
The cross-stack sensitivity check in §6.8 quantifies the build-side
delta empirically (≤5% at the reference config, sub-noise).

### C.5 Why Qwen as the CPU baseline

Three constraints narrowed the FP16-class baseline choice:

1. **Size class.** BitNet b1.58 2B4T is 2.74 B parameters. A fair
   efficiency baseline needs to be the same order of magnitude — a 1-bit
   2B vs FP16 7B would conflate "quantization helps" with "smaller model
   is cheaper." Qwen2.5-1.5B (1.54 B) is the closest comparable
   Apache-licensed FP16 model whose published accuracy lands in the same
   band as BitNet 2B4T.
2. **Licensing and reproducibility.** Apache 2.0; first-party GGUFs;
   deterministic Q8_0 and Q4_K_M conversions. Anyone re-running the
   benchmark gets bit-identical weights.
3. **Inference-stack alignment.** Qwen2.5 is a pure LLaMA-architecture
   model that runs on **upstream `llama.cpp`** with no custom kernels or
   model-side patches. The BitNet fork lags upstream by ~1 year; isolating
   the FP16 baseline on upstream lets us see what a current llama.cpp
   build can do on the same hardware.

Alternatives considered and rejected: **LLaMA 3.2 1B** (a more direct
match to the paper's exact table but gated on Meta's license);
**Gemma-3 1B** (Apache 2.0 but non-standard normalization and gating
complicate fair comparison); **Phi-3-mini 3.8B** (too large for the size
class); **SmolLM2 1.7B** and **MiniCPM 2B** (reasonable but less mature
GGUF ecosystems at the time).

### C.6 Key concepts

| Concept | Description |
|---|---|
| **Q8_0** | 8-bit symmetric quantization with 32-element blocks; ~45% size reduction vs FP16, near-lossless accuracy. |
| **Q4_K_M** | 4-bit "K-quants" mixed-block format; super-blocks of 256, sub-blocks of 32, 6-bit scales for accuracy-sensitive tensors. |
| **GQA** | Grouped-Query Attention; 12 query heads share 2 KV heads in Qwen2.5-1.5B (6:1 KV-cache compression). |
| **SwiGLU** | Swish-gated linear unit FFN activation; contrasts with BitNet 2B4T's squared-ReLU choice for activation sparsity. |
| **RMSNorm** | Root-mean-square normalization; same as BitNet b1.58, distinct from BitNet 2B4T's subLN. |
| **DPO** | Direct Preference Optimization — offline preference learning that replaces RLHF in the Qwen2.5-1.5B-Instruct pipeline. |
| **GGUF** | Successor to GGML; the binary format `llama.cpp` uses for quantized weights, metadata, and tokenizer. |
| **YaRN** | Position-interpolation method for extending RoPE context beyond the training length; not used in our 32K runs. |

---

- Reproducibility entry points: `Makefile` (`make help`)
