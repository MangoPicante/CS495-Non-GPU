"""
Generates comparison plots and a comparison CSV for Phase 4 of the capstone.

Reads local benchmark CSVs and accuracy JSONs, overlays published FP16 baseline
numbers from arXiv:2504.12285 Table 1, and saves plots to results/plots/ plus a
summary CSV to --csv (default: results/comparison_table.csv).

  BitNet:  results/bitnet_step_metrics.csv  +  results/accuracy_results_bitnet.json
  Qwen:    results/qwen_q8_step_metrics.csv    +  results/accuracy_results_qwen_q8.json  (optional)

Hardware rate default: AWS c5.xlarge on-demand, us-east-1 ($0.170/hr, 4 vCPUs).
On-demand pricing is used rather than spot for reproducibility — spot prices
change hourly and would make cost numbers non-comparable across runs.
Override with --hardware-rate if running on different hardware.

Electricity rate default: $0.16/kWh (US residential average).  Used to convert
CodeCarbon's measured energy_kwh into a dollar cost per 1k tokens, which lives
alongside the AWS proxy cost in comparison_table.csv (column:
energy_cost_per_1k_tokens).  Override with --electricity-rate.

Usage:
    python scripts/compare_runs.py
        [--results results/bitnet_step_metrics.csv]
        [--accuracy results/accuracy_results_bitnet.json]
        [--qwen-q8-results results/qwen_q8_step_metrics.csv]
        [--qwen-q8-accuracy results/accuracy_results_qwen_q8.json]
        [--csv results/comparison_table.csv]
        [--hardware-rate 0.170]
        [--electricity-rate 0.16]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR = Path(__file__).parent.parent / "results" / "plots"
DEFAULT_CSV = Path(__file__).parent.parent / "results" / "bitnet_step_metrics.csv"
DEFAULT_ACCURACY_JSON = Path(__file__).parent.parent / "results" / "accuracy_results_bitnet.json"
DEFAULT_COMPARISON_CSV = Path(__file__).parent.parent / "results" / "comparison_table.csv"
DEFAULT_QWEN_Q8_CSV = Path(__file__).parent.parent / "results" / "qwen_q8_step_metrics.csv"
DEFAULT_QWEN_Q8_ACCURACY_JSON = Path(__file__).parent.parent / "results" / "accuracy_results_qwen_q8.json"
DEFAULT_QWEN_Q4_CSV = Path(__file__).parent.parent / "results" / "qwen_q4_step_metrics.csv"
DEFAULT_QWEN_Q4_ACCURACY_JSON = Path(__file__).parent.parent / "results" / "accuracy_results_qwen_q4.json"
DEFAULT_QWEN_Q2_CSV = Path(__file__).parent.parent / "results" / "qwen_q2_step_metrics.csv"
DEFAULT_QWEN_Q2_ACCURACY_JSON = Path(__file__).parent.parent / "results" / "accuracy_results_qwen_q2.json"
DEFAULT_GEMMA_Q8_CSV = Path(__file__).parent.parent / "results" / "gemma_q8_step_metrics.csv"
DEFAULT_GEMMA_Q8_ACCURACY_JSON = Path(__file__).parent.parent / "results" / "accuracy_results_gemma_q8.json"
DEFAULT_GEMMA_Q4_CSV = Path(__file__).parent.parent / "results" / "gemma_q4_step_metrics.csv"
DEFAULT_GEMMA_Q4_ACCURACY_JSON = Path(__file__).parent.parent / "results" / "accuracy_results_gemma_q4.json"
DEFAULT_GEMMA_Q2_CSV = Path(__file__).parent.parent / "results" / "gemma_q2_step_metrics.csv"
DEFAULT_GEMMA_Q2_ACCURACY_JSON = Path(__file__).parent.parent / "results" / "accuracy_results_gemma_q2.json"

# AWS c5.xlarge on-demand, us-east-1 (4 vCPUs — matches 4-thread benchmark condition)
# Source: https://instances.vantage.sh/aws/ec2/c5.xlarge (re-verified 2026-05-24:
# unchanged from the 2026-05-08 snapshot — c5 is mature pricing).
DEFAULT_HARDWARE_RATE = 0.170

# Per-instance on-demand rates for the cross-architecture sweep
# (Phase 6).  Not currently consumed by any plot — DEFAULT_HARDWARE_RATE
# above still drives `cost_per_1k_tokens` in comparison_table.csv — but
# kept here as the verified source-of-truth for the §6.1 discussion
# table and for whenever per-arch cost-accuracy plots get added.
# All us-west-2, Linux, on-demand.  Re-verify before publication.
# Source: https://instances.vantage.sh/aws/ec2/<instance>   (2026-05-25)
AWS_ON_DEMAND_RATES = {
    "c5.xlarge":       0.170,   # Intel Xeon Skylake-SP, AVX-512 (original plan)
    "c6a.xlarge":      0.153,   # AMD EPYC 7R13 Zen 3, AVX2 (original plan)
    "c7g.xlarge":      0.145,   # AWS Graviton3 ARM, Neon/SVE (original plan)
    "c7i-flex.large":  0.0848,  # Intel Sapphire Rapids, AVX-512 (Free Tier, 2 vCPUs)
    "t4g.small":       0.0168,  # AWS Graviton2 ARM, Neon (Free Tier, 2 vCPUs)
}

# US residential average electricity rate (EIA, 2026 estimate, $/kWh).
# Used to convert CodeCarbon's energy_kwh into a dollar cost.  Override with
# --electricity-rate for industrial (~$0.10) or your local utility's rate.
DEFAULT_ELECTRICITY_RATE = 0.16

# Published FP16 baseline numbers.
# Throughput condition for Qwen: n_prompt=512, n_gen=128, single-thread x86
# CPU (arXiv:2504.12285 Table 1).  Only families for which we also
# evaluate a locally-measured PTQ variant are retained — keeps every
# FP16 paper row paired to one of our "ours" rows for a clean
# before/after comparison:
#   Qwen2.5 1.5B  paper FP16  ↔  Qwen2.5-1.5B-Instruct Q8_0/Q4_K_M/Q2_K (ours)
#   Gemma 2 2B    paper PT    ↔  Gemma-2-2B-it Q8_0/Q4_K_M/Q2_K (ours)
# Qwen2.5 1.5B is broken out separately (QWEN_PAPER below).
# Gemma 2 2B paper numbers come from the official Hugging Face model card
# (https://huggingface.co/google/gemma-2-2b) which mirrors the Gemma 2
# technical report (arXiv:2408.00118) Table 6.  Numbers are for the
# PRE-TRAINED BASE model (PT 2B), not the instruct -it variant — same
# convention QWEN_PAPER uses.  Methodology asymmetry vs our setup
# (worth noting when reading the gaps):
#   * MMLU 5-shot — matches our setup exactly
#   * ARC-Easy 0-shot — matches our setup exactly
#   * WinoGrande partial-score — matches our partial-context scoring
#   * ARC-Challenge 25-shot — WE use 0-shot, so paper inflates by ~6-10pt
#   * HellaSwag 10-shot — WE use 0-shot, so paper inflates by ~3-5pt
# CPU throughput / RSS for Gemma 2 2B are NOT reported in the Gemma
# paper (it focuses on accuracy, not CPU inference cost), so left as
# None — handled below as empty cells in the throughput/RSS/cost
# columns of comparison_table.csv and skipped from cost/memory plots.
# Gemma-3 1B, LLaMA 3.2 1B, SmolLM2 1.7B, MiniCPM 2B removed — no PTQ
# counterpart in this study, so their rows were FP16-only and unactionable.
OTHER_BASELINES = {
    "Gemma 2 2B (PT)": {
        "throughput_tokens_s": None,
        "peak_rss_mb": None,
        "arc_easy": 80.1, "arc_challenge": 55.4,
        "winogrande": 70.9, "hellaswag": 73.0, "mmlu": 51.3,
    },
}

# BitNet b1.58 2B4T numbers as reported in arXiv:2504.12285
BITNET_PAPER = {
    "throughput_tokens_s": 20.0, "peak_rss_mb": 1400,
    "arc_easy": 74.79, "arc_challenge": 49.91,
    "winogrande": 71.90, "hellaswag": 68.44, "mmlu": 53.17,
}

# Qwen2.5 1.5B FP16 numbers as reported in arXiv:2504.12285 Table 1.
# Compared against our Q8_0 measurement; Q8_0 typically matches FP16 within
# ~0.5pt on these benchmarks, so the paper row is treated as the target.
QWEN_PAPER = {
    "throughput_tokens_s": 3.8, "peak_rss_mb": 3100,
    "arc_easy": 79.92, "arc_challenge": 52.82,
    "winogrande": 66.61, "hellaswag": 70.95, "mmlu": 61.11,
}

# Color convention used across all comparison plots — FAMILY-CODED.
# Each model family gets its own hue; within a family, the shade darkens
# as model size / faithfulness to the original increases (paper > Q8 > Q4
# > Q2).  This lets a reader spot family at a glance and ranking-within-
# family from the shade gradient.
#
#   BitNet     → orange family
#       paper      → burnt orange (sienna)
#       ours (i2_s)→ light orange (sandy)
#   Qwen 1.5B  → blue family
#       paper FP16 → deep navy
#       Q8_0 ours  → medium blue (steel)
#       Q4_K_M ours→ light blue
#       Q2_K ours  → sky blue
#   Gemma 2 2B → green family
#       paper PT   → deep forest green
#       Q8_0 ours  → dark green
#       Q4_K_M ours→ medium green
#       Q2_K ours  → light green
#   Cloud APIs / generic fallback → neutral gray
BITNET_PAPER_COLOR = "#A0522D"   # sienna / burnt orange
BITNET_COLOR       = "#F4A460"   # sandy / light orange
QWEN_PAPER_COLOR   = "#0D47A1"   # deep navy
QWEN_Q8_COLOR      = "#1976D2"   # medium blue
QWEN_Q4_COLOR      = "#42A5F5"   # light blue
QWEN_Q2_COLOR      = "#90CAF9"   # sky blue
GEMMA_PAPER_COLOR  = "#1B5E20"   # deep forest green
GEMMA_Q8_COLOR     = "#388E3C"   # dark green
GEMMA_Q4_COLOR     = "#66BB6A"   # medium green
GEMMA_Q2_COLOR     = "#A5D6A7"   # light green
CLOUD_API_COLOR    = "#7F7F7F"   # neutral gray
# Back-compat alias: a few legacy code paths still reference OTHER_COLOR
# as a generic FP16-baseline fallback when the specific family isn't
# known.  Aliased to the deepest Qwen-paper navy so existing call sites
# render in a family-consistent way; new code should prefer
# QWEN_PAPER_COLOR / GEMMA_PAPER_COLOR / BITNET_PAPER_COLOR directly.
OTHER_COLOR        = QWEN_PAPER_COLOR

# ─────────────────────────────────────────────────────────────────────────────
# Cross-architecture sources for `plot_cross_arch_throughput` (REPORT §6.1).
# ─────────────────────────────────────────────────────────────────────────────
# Each entry points at one (CPU family / OS) result set.  The reference
# `subdir=None` row reads the canonical results/ CSVs (Windows native on
# the local i5-9400F); every other entry expects an analogously-named
# triple (bitnet_step_metrics.csv / qwen_q8_step_metrics.csv /
# qwen_q4_step_metrics.csv) under results/<subdir>/.
#
# Architectures with no data on disk are skipped silently, so this list
# can outpace the AWS sweep — the plot grows bars as CSVs land.
CROSS_ARCH_SOURCES = [
    # (label,                                     subdir,               color)
    # Cross-arch bars use a purple palette so they never collide with the
    # family-coded model colors (BitNet orange / Qwen blue / Gemma green
    # / cloud-API gray).  Shade walks darker → lighter from the most-
    # authoritative reference (Windows native, the canonical bench host)
    # to the more derivative architectures.  A gold accent reserved for
    # ARM Graviton if/when that data ever lands — also distinct from
    # every model-family hue.
    ("Windows / i5-9400F (AVX2)",                 None,                 "#311B92"),
    ("Linux Docker / i5-9400F (AVX2)",            "linux_docker_x86",   "#7E57C2"),
    ("AWS c7i-flex.large (Intel AVX-512, 2v)",    "aws_c7i_flex_large", "#B39DDB"),
    ("AWS t4g.small (ARM Graviton2, 2v)",         "aws_t4g_small",      "#FFB300"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Cloud API output-token pricing
# ─────────────────────────────────────────────────────────────────────────────
# These prices are HARDCODED as of CLOUD_API_PRICING_DATE.  Cloud providers
# change pricing periodically; re-verify against each provider's pricing page
# before quoting these numbers in REPORT.md or any external artifact.
# Bump CLOUD_API_PRICING_DATE whenever you refresh the dict so the plot title
# and stdout note stay accurate.
#
# Quoted in $ per 1 million OUTPUT tokens.  Output pricing is what's directly
# comparable to our self-hosted throughput metric (which measures generated
# tokens); input pricing is typically much lower.
#
#   OpenAI:     https://openai.com/api/pricing
#   Anthropic:  https://www.anthropic.com/pricing#api
# ─────────────────────────────────────────────────────────────────────────────
CLOUD_API_PRICING_DATE = "2026-05-24"
CLOUD_API_PRICING = {
    "OpenAI GPT-4o mini":          0.60,
    "Anthropic Claude Haiku 4.5":  5.00,
    "OpenAI GPT-4o":              10.00,
    "Anthropic Claude Sonnet 4.5": 15.00,
    # Opus 4.7 was cut from $75 to $25 between the original 2026-05-15
    # snapshot and the 2026-05-24 re-verification (Anthropic aligned
    # the entire Opus 4.5+ family at $25 output; the old $75 was Opus
    # 4.1's number).  Re-verify against the docs.claude.com/pricing
    # table before publication — Anthropic refreshes periodically.
    "Anthropic Claude Opus 4.7":  25.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# Cloud API accuracy numbers
# ─────────────────────────────────────────────────────────────────────────────
# Where each cell comes from (verified against the PDFs in Models/Cloud/ on
# CLOUD_API_ACCURACY_DATE):
#
#   - GPT-4o-System-Card.pdf includes only a *medical subset* of MMLU and
#     explicitly warns (footnote 6) that those scores "should not be compared
#     with publicly reported benchmarks".  The MMLU value below is from
#     OpenAI's May-2024 launch announcement instead.
#   - GPT-4o mini has no standalone system card; MMLU value is from OpenAI's
#     July-2024 GPT-4o mini announcement.
#   - Claude Haiku 4.5 and Sonnet 4.5 cards are safety-focused (AUP / RSP
#     evaluations) and contain no general benchmark numbers.  MMLU values
#     below come from Anthropic's launch announcements and third-party
#     leaderboards.
#   - Claude Opus 4.7's card §8 reports MMMLU (multilingual MMLU) = 91.5%,
#     the closest in-card analog.  The MMLU value below (92.4%) is from the
#     standalone Opus 4.7 launch announcement so it stays comparable to the
#     other rows on the same scale.
#
# Re-verify before publication; vendors release point updates that shift
# scores under the same product name.  Bump CLOUD_API_ACCURACY_DATE on
# refresh.
# ─────────────────────────────────────────────────────────────────────────────
CLOUD_API_ACCURACY_DATE = "2026-05-27"
CLOUD_API_ACCURACY: dict[str, dict[str, float | None]] = {
    "OpenAI GPT-4o mini": {
        # OpenAI GPT-4o mini announcement, July 2024.  No standalone card.
        "arc_easy": None, "arc_challenge": None,
        "winogrande": None, "hellaswag": None,
        "mmlu": 82.0,
    },
    "Anthropic Claude Haiku 4.5": {
        # Anthropic announcement / third-party leaderboards — system card
        # is safety-focused and reports no general benchmarks.
        "arc_easy": None, "arc_challenge": None,
        "winogrande": None, "hellaswag": None,
        "mmlu": 78.0,
    },
    "OpenAI GPT-4o": {
        # OpenAI launch announcement, May 2024.  System card's MMLU is a
        # medical-only subset (footnote 6) and not directly comparable.
        # ARC/WinoGrande/HellaSwag were previously listed here but traced
        # back to the GPT-4 tech report (March 2023), not GPT-4o — the
        # GPT-4o system card does not publish these benchmarks.
        "arc_easy": None, "arc_challenge": None,
        "winogrande": None, "hellaswag": None,
        "mmlu": 88.7,
    },
    "Anthropic Claude Sonnet 4.5": {
        # Anthropic launch announcement + third-party leaderboards (Artificial
        # Analysis, Skywork AI) — system card is safety-focused and reports no
        # general benchmarks.  89.2% is the standard-MMLU figure; not to be
        # confused with MMLU-Pro (86%).
        "arc_easy": None, "arc_challenge": None,
        "winogrande": None, "hellaswag": None,
        "mmlu": 89.2,
    },
    "Anthropic Claude Opus 4.7": {
        # Anthropic Opus 4.7 launch announcement, April 2026.  The system
        # card's in-table value is MMMLU (multilingual MMLU) = 91.5%, very
        # close to this number — keeping the standard-MMLU value here so the
        # row stays apples-to-apples with the other cloud entries.
        "arc_easy": None, "arc_challenge": None,
        "winogrande": None, "hellaswag": None,
        "mmlu": 92.4,
    },
}

# Display names for the five accuracy tasks (used in per-task plot titles/labels)
TASK_LABELS = {
    "arc_easy":      "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "winogrande":    "WinoGrande",
    "hellaswag":     "HellaSwag",
    "mmlu":          "MMLU",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(DEFAULT_CSV))
    p.add_argument("--accuracy", default=str(DEFAULT_ACCURACY_JSON))
    p.add_argument("--qwen-q8-results", default=str(DEFAULT_QWEN_Q8_CSV))
    p.add_argument("--qwen-q8-accuracy", default=str(DEFAULT_QWEN_Q8_ACCURACY_JSON))
    p.add_argument("--qwen-q4-results", default=str(DEFAULT_QWEN_Q4_CSV))
    p.add_argument("--qwen-q4-accuracy", default=str(DEFAULT_QWEN_Q4_ACCURACY_JSON))
    p.add_argument("--qwen-q2-results", default=str(DEFAULT_QWEN_Q2_CSV))
    p.add_argument("--qwen-q2-accuracy", default=str(DEFAULT_QWEN_Q2_ACCURACY_JSON))
    p.add_argument("--gemma-q8-results", default=str(DEFAULT_GEMMA_Q8_CSV))
    p.add_argument("--gemma-q8-accuracy", default=str(DEFAULT_GEMMA_Q8_ACCURACY_JSON))
    p.add_argument("--gemma-q4-results", default=str(DEFAULT_GEMMA_Q4_CSV))
    p.add_argument("--gemma-q4-accuracy", default=str(DEFAULT_GEMMA_Q4_ACCURACY_JSON))
    p.add_argument("--gemma-q2-results", default=str(DEFAULT_GEMMA_Q2_CSV))
    p.add_argument("--gemma-q2-accuracy", default=str(DEFAULT_GEMMA_Q2_ACCURACY_JSON))
    p.add_argument("--csv", default=str(DEFAULT_COMPARISON_CSV))
    p.add_argument(
        "--hardware-rate",
        type=float,
        default=DEFAULT_HARDWARE_RATE,
        metavar="$/HR",
        help="On-demand $/hr for the target instance (default: %(default)s — AWS c5.xlarge us-east-1)",
    )
    p.add_argument(
        "--electricity-rate",
        type=float,
        default=DEFAULT_ELECTRICITY_RATE,
        metavar="$/KWH",
        help="Local electricity rate $/kWh used to convert CodeCarbon energy_kwh "
             "to dollars (default: %(default)s — US residential average)",
    )
    return p.parse_args()


def load_local(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"No benchmark results at {csv_path} — throughput/memory/energy plots "
              f"will be skipped. Run 'make benchmark' to generate them.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df = df[df["throughput_tokens_s"].notna() & (df["throughput_tokens_s"] != "")]
    df["throughput_tokens_s"] = pd.to_numeric(df["throughput_tokens_s"])
    df["peak_rss_mb"] = pd.to_numeric(df["peak_rss_mb"])
    return df


def _has_bench_data(df: pd.DataFrame | None) -> bool:
    """True if df has at least one row matching the (512, 128) reference config."""
    if df is None or df.empty:
        return False
    if "n_prompt" not in df.columns or "n_gen" not in df.columns:
        return False
    return not df[(df["n_prompt"] == 512) & (df["n_gen"] == 128)].empty


def load_accuracy(json_path: Path) -> dict:
    if not json_path.exists():
        print(f"No accuracy results at {json_path} — accuracy columns will be empty.")
        return {}
    with json_path.open() as f:
        data = json.load(f)
    return {
        "arc_easy":      data.get("arc_easy",      {}).get("accuracy"),
        "arc_challenge": data.get("arc_challenge", {}).get("accuracy"),
        "winogrande":    data.get("winogrande",    {}).get("accuracy"),
        "hellaswag":     data.get("hellaswag",     {}).get("accuracy"),
        "mmlu":          data.get("mmlu",          {}).get("accuracy"),
    }


def load_accuracy_full(json_path: Path) -> dict | None:
    """Return the entire accuracy JSON (subjects, elapsed_s, energy, etc.)."""
    if not json_path.exists():
        return None
    with json_path.open() as f:
        return json.load(f)


def load_qwen(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[df["throughput_tokens_s"].notna() & (df["throughput_tokens_s"] != "")]
    df["throughput_tokens_s"] = pd.to_numeric(df["throughput_tokens_s"])
    df["peak_rss_mb"] = pd.to_numeric(df["peak_rss_mb"])
    return df


def _bench_row(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return (median throughput, median peak_rss) for the n_prompt=512, n_gen=128 condition."""
    if df is None or df.empty or "n_prompt" not in df.columns:
        return None, None
    row = df[(df["n_prompt"] == 512) & (df["n_gen"] == 128)]
    tps = row["throughput_tokens_s"].median() if not row.empty else None
    rss = row["peak_rss_mb"].median() if not row.empty else None
    return tps, rss


def cost_per_1k(throughput_tokens_s: float, rate_per_hour: float) -> float:
    """Dollar cost to generate 1,000 tokens at the given throughput and hourly hardware rate."""
    return (1000.0 / throughput_tokens_s / 3600.0) * rate_per_hour


def energy_cost_per_1k(df: pd.DataFrame | None, electricity_rate: float) -> float | None:
    """
    Dollar cost of electricity to generate 1,000 tokens at the (512, 128)
    reference config, using CodeCarbon's measured energy_kwh.

    Returns None when no energy data is present (paper rows; --no-energy runs).
    """
    if df is None or df.empty or "energy_kwh" not in df.columns:
        return None
    row = df[(df["n_prompt"] == 512) & (df["n_gen"] == 128)].copy()
    row["energy_kwh"] = pd.to_numeric(row["energy_kwh"], errors="coerce")
    e = row["energy_kwh"].dropna().median()
    if pd.isna(e):
        return None
    return (e / (512 + 128)) * 1000.0 * electricity_rate


def build_comparison_df(
    local_df: pd.DataFrame,
    local_acc: dict,
    hardware_rate: float,
    qwen_q8_df: pd.DataFrame | None = None,
    qwen_q8_acc: dict | None = None,
    qwen_q4_df: pd.DataFrame | None = None,
    qwen_q4_acc: dict | None = None,
    qwen_q2_df: pd.DataFrame | None = None,
    qwen_q2_acc: dict | None = None,
    gemma_q8_df: pd.DataFrame | None = None,
    gemma_q8_acc: dict | None = None,
    gemma_q4_df: pd.DataFrame | None = None,
    gemma_q4_acc: dict | None = None,
    gemma_q2_df: pd.DataFrame | None = None,
    gemma_q2_acc: dict | None = None,
    electricity_rate: float = DEFAULT_ELECTRICITY_RATE,
) -> pd.DataFrame:
    ACC_FIELDS = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]
    rows = []

    # Row order: FP16 papers (other baselines → Qwen FP16 paper) → Qwen ours
    # (Q8, Q4) → BitNet ours → BitNet paper.  Keeps all FP16 papers together at
    # the top and pushes BitNet to the bottom so the comparison reads
    # "literature numbers first, our measurements next, BitNet last".
    for name, b in OTHER_BASELINES.items():
        tps = b.get("throughput_tokens_s")
        rss = b.get("peak_rss_mb")
        rows.append({
            "model": name,
            "source": "paper",
            "throughput_tokens_s": tps if tps is not None else "",
            "peak_rss_mb": rss if rss is not None else "",
            "cost_per_1k_tokens": (round(cost_per_1k(tps, hardware_rate), 6)
                                   if tps is not None else ""),
            # No CodeCarbon measurement for paper rows — would require running
            # those baselines locally, which is outside this project's scope.
            "energy_cost_per_1k_tokens": "",
            **{f: b[f] for f in ACC_FIELDS},
        })

    rows.append({
        "model": "Qwen2.5 1.5B",
        "source": "paper (FP16)",
        "throughput_tokens_s": QWEN_PAPER["throughput_tokens_s"],
        "peak_rss_mb": QWEN_PAPER["peak_rss_mb"],
        "cost_per_1k_tokens": round(cost_per_1k(QWEN_PAPER["throughput_tokens_s"], hardware_rate), 6),
        "energy_cost_per_1k_tokens": "",
        **{f: QWEN_PAPER[f] for f in ACC_FIELDS},
    })

    q8_acc = qwen_q8_acc or {}
    if qwen_q8_df is not None or any(q8_acc.get(f) is not None for f in ACC_FIELDS):
        q8_tps, q8_rss = _bench_row(qwen_q8_df) if qwen_q8_df is not None else (None, None)
        q8_cost = round(cost_per_1k(q8_tps, hardware_rate), 6) if q8_tps else ""
        q8_e_cost = energy_cost_per_1k(qwen_q8_df, electricity_rate)
        rows.append({
            "model": "Qwen2.5-1.5B-Instruct Q8_0",
            "source": "ours",
            "throughput_tokens_s": round(q8_tps, 2) if q8_tps is not None else "",
            "peak_rss_mb": round(q8_rss, 0) if q8_rss is not None else "",
            "cost_per_1k_tokens": q8_cost,
            "energy_cost_per_1k_tokens": round(q8_e_cost, 6) if q8_e_cost is not None else "",
            **{f: (round(q8_acc[f], 2) if q8_acc.get(f) is not None else "") for f in ACC_FIELDS},
        })

    q4_acc = qwen_q4_acc or {}
    if qwen_q4_df is not None or any(q4_acc.get(f) is not None for f in ACC_FIELDS):
        q4_tps, q4_rss = _bench_row(qwen_q4_df) if qwen_q4_df is not None else (None, None)
        q4_cost = round(cost_per_1k(q4_tps, hardware_rate), 6) if q4_tps else ""
        q4_e_cost = energy_cost_per_1k(qwen_q4_df, electricity_rate)
        rows.append({
            "model": "Qwen2.5-1.5B-Instruct Q4_K_M",
            "source": "ours",
            "throughput_tokens_s": round(q4_tps, 2) if q4_tps is not None else "",
            "peak_rss_mb": round(q4_rss, 0) if q4_rss is not None else "",
            "cost_per_1k_tokens": q4_cost,
            "energy_cost_per_1k_tokens": round(q4_e_cost, 6) if q4_e_cost is not None else "",
            **{f: (round(q4_acc[f], 2) if q4_acc.get(f) is not None else "") for f in ACC_FIELDS},
        })

    q2_acc = qwen_q2_acc or {}
    if qwen_q2_df is not None or any(q2_acc.get(f) is not None for f in ACC_FIELDS):
        q2_tps, q2_rss = _bench_row(qwen_q2_df) if qwen_q2_df is not None else (None, None)
        q2_cost = round(cost_per_1k(q2_tps, hardware_rate), 6) if q2_tps else ""
        q2_e_cost = energy_cost_per_1k(qwen_q2_df, electricity_rate)
        rows.append({
            "model": "Qwen2.5-1.5B-Instruct Q2_K",
            "source": "ours",
            "throughput_tokens_s": round(q2_tps, 2) if q2_tps is not None else "",
            "peak_rss_mb": round(q2_rss, 0) if q2_rss is not None else "",
            "cost_per_1k_tokens": q2_cost,
            "energy_cost_per_1k_tokens": round(q2_e_cost, 6) if q2_e_cost is not None else "",
            **{f: (round(q2_acc[f], 2) if q2_acc.get(f) is not None else "") for f in ACC_FIELDS},
        })

    gemma_q8_acc_d = gemma_q8_acc or {}
    if gemma_q8_df is not None or any(gemma_q8_acc_d.get(f) is not None for f in ACC_FIELDS):
        g8_tps, g8_rss = _bench_row(gemma_q8_df) if gemma_q8_df is not None else (None, None)
        g8_cost = round(cost_per_1k(g8_tps, hardware_rate), 6) if g8_tps else ""
        g8_e_cost = energy_cost_per_1k(gemma_q8_df, electricity_rate)
        rows.append({
            "model": "Gemma-2-2B-it Q8_0",
            "source": "ours",
            "throughput_tokens_s": round(g8_tps, 2) if g8_tps is not None else "",
            "peak_rss_mb": round(g8_rss, 0) if g8_rss is not None else "",
            "cost_per_1k_tokens": g8_cost,
            "energy_cost_per_1k_tokens": round(g8_e_cost, 6) if g8_e_cost is not None else "",
            **{f: (round(gemma_q8_acc_d[f], 2) if gemma_q8_acc_d.get(f) is not None else "") for f in ACC_FIELDS},
        })

    gemma_q4_acc_d = gemma_q4_acc or {}
    if gemma_q4_df is not None or any(gemma_q4_acc_d.get(f) is not None for f in ACC_FIELDS):
        g4_tps, g4_rss = _bench_row(gemma_q4_df) if gemma_q4_df is not None else (None, None)
        g4_cost = round(cost_per_1k(g4_tps, hardware_rate), 6) if g4_tps else ""
        g4_e_cost = energy_cost_per_1k(gemma_q4_df, electricity_rate)
        rows.append({
            "model": "Gemma-2-2B-it Q4_K_M",
            "source": "ours",
            "throughput_tokens_s": round(g4_tps, 2) if g4_tps is not None else "",
            "peak_rss_mb": round(g4_rss, 0) if g4_rss is not None else "",
            "cost_per_1k_tokens": g4_cost,
            "energy_cost_per_1k_tokens": round(g4_e_cost, 6) if g4_e_cost is not None else "",
            **{f: (round(gemma_q4_acc_d[f], 2) if gemma_q4_acc_d.get(f) is not None else "") for f in ACC_FIELDS},
        })

    gemma_q2_acc_d = gemma_q2_acc or {}
    if gemma_q2_df is not None or any(gemma_q2_acc_d.get(f) is not None for f in ACC_FIELDS):
        g2_tps, g2_rss = _bench_row(gemma_q2_df) if gemma_q2_df is not None else (None, None)
        g2_cost = round(cost_per_1k(g2_tps, hardware_rate), 6) if g2_tps else ""
        g2_e_cost = energy_cost_per_1k(gemma_q2_df, electricity_rate)
        rows.append({
            "model": "Gemma-2-2B-it Q2_K",
            "source": "ours",
            "throughput_tokens_s": round(g2_tps, 2) if g2_tps is not None else "",
            "peak_rss_mb": round(g2_rss, 0) if g2_rss is not None else "",
            "cost_per_1k_tokens": g2_cost,
            "energy_cost_per_1k_tokens": round(g2_e_cost, 6) if g2_e_cost is not None else "",
            **{f: (round(gemma_q2_acc_d[f], 2) if gemma_q2_acc_d.get(f) is not None else "") for f in ACC_FIELDS},
        })

    bitnet_tps, bitnet_rss = _bench_row(local_df)
    our_cost = round(cost_per_1k(bitnet_tps, hardware_rate), 6) if bitnet_tps else ""
    bitnet_e_cost = energy_cost_per_1k(local_df, electricity_rate)
    rows.append({
        "model": "BitNet b1.58 2B4T",
        "source": "ours",
        "throughput_tokens_s": round(bitnet_tps, 2) if bitnet_tps is not None else "",
        "peak_rss_mb": round(bitnet_rss, 0) if bitnet_rss is not None else "",
        "cost_per_1k_tokens": our_cost,
        "energy_cost_per_1k_tokens": round(bitnet_e_cost, 6) if bitnet_e_cost is not None else "",
        **{f: (round(local_acc[f], 2) if local_acc.get(f) is not None else "") for f in ACC_FIELDS},
    })

    rows.append({
        "model": "BitNet b1.58 2B4T",
        "source": "paper",
        "throughput_tokens_s": BITNET_PAPER["throughput_tokens_s"],
        "peak_rss_mb": BITNET_PAPER["peak_rss_mb"],
        "cost_per_1k_tokens": round(cost_per_1k(BITNET_PAPER["throughput_tokens_s"], hardware_rate), 6),
        "energy_cost_per_1k_tokens": "",
        **{f: BITNET_PAPER[f] for f in ACC_FIELDS},
    })

    return pd.DataFrame(rows)


def write_comparison_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


# Canonical bar-chart order across every plot in the report:
#   BitNet (paper → ours) → Qwen (paper → Q8 → Q4 → Q2 ours) →
#   Gemma (paper → Q8 → Q4 → Q2 ours).
# Within each family: paper baseline first, then quants from largest
# (closest to FP16) to smallest (most aggressive PTQ).
#
# Each plot function expresses the same family layout: BitNet, Qwen,
# Gemma in that order.  Plot helpers compute family-boundary indices
# from the returned labels by detecting prefix transitions, so they
# can draw a thin separator line between groups.

_FAMILY_PREFIXES = ("BitNet", "Qwen", "Gemma")


def _family_boundaries(labels: list[str]) -> list[int]:
    """
    Return the indices where a new model family starts (BitNet / Qwen / Gemma).
    The first family transition is index 1 onward; index 0 (the first BitNet
    row) is omitted since you don't want a separator at the very start.
    Used by plotting code to add `axhline` / `axvline` separators between
    families without each plot re-implementing the prefix detection.
    """
    out = []
    last_family = None
    for i, label in enumerate(labels):
        fam = next((p for p in _FAMILY_PREFIXES if label.startswith(p)), None)
        if fam != last_family and last_family is not None:
            out.append(i)
        last_family = fam
    return out


def _bar_series(local_df: pd.DataFrame, qwen_q8_df: pd.DataFrame | None,
                qwen_q4_df: pd.DataFrame | None,
                metric: str,
                qwen_q2_df: pd.DataFrame | None = None,
                gemma_q8_df: pd.DataFrame | None = None,
                gemma_q4_df: pd.DataFrame | None = None,
                gemma_q2_df: pd.DataFrame | None = None,
                ) -> tuple[list[str], list[float], list[str], list[str]]:
    """
    Build the (labels, values, colors, hatches) tuple for a horizontal bar chart.

    Order: BitNet paper → BitNet ours → Qwen paper → Qwen Q8/Q4/Q2 ours →
    [Gemma paper if present in OTHER_BASELINES] → Gemma Q8/Q4/Q2 ours.
    Within each family: paper baseline first, then quants largest → smallest.

    Each "ours" row is conditional on local data existing.  `metric` selects
    which field to read from each paper dict / which bench column.
    """
    metric_col = "throughput_tokens_s" if metric == "throughput_tokens_s" else "peak_rss_mb"
    bitnet_tps, bitnet_rss = _bench_row(local_df)
    bitnet_local = bitnet_tps if metric_col == "throughput_tokens_s" else bitnet_rss

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    hatches: list[str] = []

    # ── BitNet ────────────────────────────────────────────────────────
    labels.append("BitNet b1.58 2B4T (paper)")
    values.append(BITNET_PAPER[metric_col])
    colors.append(BITNET_PAPER_COLOR)
    hatches.append("///")

    if bitnet_local is not None:
        labels.append("BitNet b1.58 2B4T (ours)")
        values.append(bitnet_local)
        colors.append(BITNET_COLOR)
        hatches.append("")

    # ── Qwen 1.5B ─────────────────────────────────────────────────────
    # Paper FP16 row anchors the Qwen-family blue: paper is the deepest
    # navy, Q8 / Q4 / Q2 walk down the family palette as quantization
    # depth grows.
    labels.append("Qwen2.5 1.5B (paper FP16)")
    values.append(QWEN_PAPER[metric_col])
    colors.append(QWEN_PAPER_COLOR)
    hatches.append("")

    q8_tps, q8_rss = _bench_row(qwen_q8_df) if qwen_q8_df is not None else (None, None)
    q8_val = q8_tps if metric_col == "throughput_tokens_s" else q8_rss
    if q8_val is not None:
        labels.append("Qwen2.5-1.5B-Instruct Q8_0 (ours)")
        values.append(q8_val)
        colors.append(QWEN_Q8_COLOR)
        hatches.append("")

    q4_tps, q4_rss = _bench_row(qwen_q4_df) if qwen_q4_df is not None else (None, None)
    q4_val = q4_tps if metric_col == "throughput_tokens_s" else q4_rss
    if q4_val is not None:
        labels.append("Qwen2.5-1.5B-Instruct Q4_K_M (ours)")
        values.append(q4_val)
        colors.append(QWEN_Q4_COLOR)
        hatches.append("")

    q2_tps, q2_rss = _bench_row(qwen_q2_df) if qwen_q2_df is not None else (None, None)
    q2_val = q2_tps if metric_col == "throughput_tokens_s" else q2_rss
    if q2_val is not None:
        labels.append("Qwen2.5-1.5B-Instruct Q2_K (ours)")
        values.append(q2_val)
        colors.append(QWEN_Q2_COLOR)
        hatches.append("")

    # ── Gemma 2 2B ────────────────────────────────────────────────────
    # OTHER_BASELINES holds the Gemma 2 PT paper row; appears here only
    # if the paper baseline has the requested metric (throughput/RSS).
    # Anchors the Gemma-family green palette in the same way Qwen FP16
    # anchors the blue family above.
    for m, paper in OTHER_BASELINES.items():
        v = paper.get(metric_col)
        if v is None:
            continue
        labels.append(m)
        values.append(v)
        colors.append(GEMMA_PAPER_COLOR)
        hatches.append("")

    g8_tps, g8_rss = _bench_row(gemma_q8_df) if gemma_q8_df is not None else (None, None)
    g8_val = g8_tps if metric_col == "throughput_tokens_s" else g8_rss
    if g8_val is not None:
        labels.append("Gemma-2-2B-it Q8_0 (ours)")
        values.append(g8_val)
        colors.append(GEMMA_Q8_COLOR)
        hatches.append("")

    g4_tps, g4_rss = _bench_row(gemma_q4_df) if gemma_q4_df is not None else (None, None)
    g4_val = g4_tps if metric_col == "throughput_tokens_s" else g4_rss
    if g4_val is not None:
        labels.append("Gemma-2-2B-it Q4_K_M (ours)")
        values.append(g4_val)
        colors.append(GEMMA_Q4_COLOR)
        hatches.append("")

    g2_tps, g2_rss = _bench_row(gemma_q2_df) if gemma_q2_df is not None else (None, None)
    g2_val = g2_tps if metric_col == "throughput_tokens_s" else g2_rss
    if g2_val is not None:
        labels.append("Gemma-2-2B-it Q2_K (ours)")
        values.append(g2_val)
        colors.append(GEMMA_Q2_COLOR)
        hatches.append("")

    return labels, values, colors, hatches


def _legend_handles(qwen_q8_df: pd.DataFrame | None, qwen_q4_df: pd.DataFrame | None = None,
                    qwen_q2_df: pd.DataFrame | None = None,
                    gemma_q8_df: pd.DataFrame | None = None,
                    gemma_q4_df: pd.DataFrame | None = None,
                    gemma_q2_df: pd.DataFrame | None = None):
    from matplotlib.patches import Patch
    # Family-coded legend: BitNet (orange), Qwen (blue), Gemma (green)
    # — each paper baseline is the darkest shade and each quant walks
    # the family palette toward lighter as quantization deepens.
    handles = [
        Patch(facecolor=BITNET_PAPER_COLOR, hatch="///", edgecolor="#444444",
              label="BitNet b1.58 2B4T (paper)"),
        Patch(facecolor=BITNET_COLOR, edgecolor="#cccccc",
              label="BitNet b1.58 2B4T (ours)"),
        Patch(facecolor=QWEN_PAPER_COLOR, edgecolor="#cccccc",
              label="Qwen2.5 1.5B FP16 (paper)"),
    ]
    if qwen_q8_df is not None:
        handles.append(Patch(facecolor=QWEN_Q8_COLOR, edgecolor="#cccccc", label="Qwen2.5-1.5B Q8_0 (ours)"))
    if qwen_q4_df is not None:
        handles.append(Patch(facecolor=QWEN_Q4_COLOR, edgecolor="#cccccc", label="Qwen2.5-1.5B Q4_K_M (ours)"))
    if qwen_q2_df is not None:
        handles.append(Patch(facecolor=QWEN_Q2_COLOR, edgecolor="#cccccc", label="Qwen2.5-1.5B Q2_K (ours)"))
    if OTHER_BASELINES:
        handles.append(Patch(facecolor=GEMMA_PAPER_COLOR, edgecolor="#cccccc",
                             label="Gemma 2 2B PT (paper)"))
    if gemma_q8_df is not None:
        handles.append(Patch(facecolor=GEMMA_Q8_COLOR, edgecolor="#cccccc", label="Gemma-2-2B-it Q8_0 (ours)"))
    if gemma_q4_df is not None:
        handles.append(Patch(facecolor=GEMMA_Q4_COLOR, edgecolor="#cccccc", label="Gemma-2-2B-it Q4_K_M (ours)"))
    if gemma_q2_df is not None:
        handles.append(Patch(facecolor=GEMMA_Q2_COLOR, edgecolor="#cccccc", label="Gemma-2-2B-it Q2_K (ours)"))
    return handles


def plot_throughput(local_df: pd.DataFrame, out_dir: Path,
                    qwen_q8_df: pd.DataFrame | None = None,
                    qwen_q4_df: pd.DataFrame | None = None,
                    qwen_q2_df: pd.DataFrame | None = None,
                    gemma_q8_df: pd.DataFrame | None = None,
                    gemma_q4_df: pd.DataFrame | None = None,
                    gemma_q2_df: pd.DataFrame | None = None):
    """
    Single unified throughput plot with two panels:

      (a) Cross-model bar chart at (n_prompt=512, n_gen=128) — paper FP16
          baselines + BitNet (paper, ours) + Qwen (paper FP16, Q8/Q4/Q2 ours)
          + Gemma 2 2B (Q8/Q4/Q2 ours; no paper baseline).
      (b) Per-config sensitivity — grouped bars across the three benchmarked
          (n_prompt, n_gen) configs for our locally measured models.  The
          paper FP16 baselines only publish numbers at (512, 128), so they
          don't appear in this panel.
    """
    has_b  = _has_bench_data(local_df)
    has_q8 = _has_bench_data(qwen_q8_df)
    has_q4 = _has_bench_data(qwen_q4_df)
    has_q2 = _has_bench_data(qwen_q2_df)
    has_gemma_q8 = _has_bench_data(gemma_q8_df)
    has_gemma_q4 = _has_bench_data(gemma_q4_df)
    has_gemma_q2 = _has_bench_data(gemma_q2_df)
    if not (has_b or has_q8 or has_q4 or has_q2 or has_gemma_q8 or has_gemma_q4 or has_gemma_q2):
        print("Skipping throughput plot: no benchmark CSVs (run 'make benchmark').")
        return

    labels, values, colors, hatches = _bar_series(
        local_df, qwen_q8_df, qwen_q4_df, "throughput_tokens_s",
        qwen_q2_df=qwen_q2_df,
        gemma_q8_df=gemma_q8_df, gemma_q4_df=gemma_q4_df, gemma_q2_df=gemma_q2_df,
    )
    # Per-config panel needs >=2 measured configs to be useful.  After the
    # 2026-06-13 single-config bench reduction, most CSVs only contain
    # (512, 128) (+ optional (512, 0) pp512), so a per-config panel would
    # render with 1-2 empty columns.  Detect this and skip panel (b) when
    # no model has data for both (512, 512) and (1, 512).
    def _has_secondary_configs(df):
        if df is None or df.empty or "n_prompt" not in df.columns:
            return False
        configs = {(int(r.n_prompt), int(r.n_gen))
                   for r in df.itertuples()
                   if r.n_prompt and r.n_gen}
        return (512, 512) in configs and (1, 512) in configs

    has_secondary = any(_has_secondary_configs(d) for d in
                        (local_df, qwen_q8_df, qwen_q4_df, qwen_q2_df,
                         gemma_q8_df, gemma_q4_df, gemma_q2_df))

    if has_secondary:
        fig = plt.figure(figsize=(13, max(9, len(labels) * 0.55 + 4)))
        gs = fig.add_gridspec(2, 1, height_ratios=[len(labels) * 0.55, 4.5], hspace=0.35)
        ax_main = fig.add_subplot(gs[0])
        ax_cfg  = fig.add_subplot(gs[1])
    else:
        fig, ax_main = plt.subplots(figsize=(13, max(5.5, len(labels) * 0.55 + 1.5)))
        ax_cfg = None

    # ── (a) Cross-model comparison at (512, 128) ─────────────────────────
    max_val = max(v for v in values if v) or 1
    for i, (val, color, hatch) in enumerate(zip(values, colors, hatches)):
        ax_main.barh(i, val, color=color, hatch=hatch,
                     edgecolor="#444444" if hatch else "#cccccc", linewidth=0.5)
        ax_main.text(val + max_val * 0.01, i, f"{val:.1f}", va="center", fontsize=9)
    ax_main.set_yticks(range(len(labels)))
    ax_main.set_yticklabels(labels)
    ax_main.set_xlabel("Throughput (tokens/s)")
    ax_main.set_title("(a) Cross-model comparison at n_prompt=512, n_gen=128", loc="left")
    ax_main.set_xlim(0, max_val * 1.15)
    ax_main.invert_yaxis()
    # Family separators between BitNet / Qwen / Gemma groups.
    for b in _family_boundaries(labels):
        ax_main.axhline(b - 0.5, color="#888888", linewidth=0.8,
                        linestyle="--", alpha=0.6)
    # Legend goes OUTSIDE axes so it never occludes bar value labels.
    ax_main.legend(
        handles=_legend_handles(qwen_q8_df, qwen_q4_df, qwen_q2_df,
                                gemma_q8_df, gemma_q4_df, gemma_q2_df),
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=8, frameon=True, framealpha=0.95,
        handlelength=1.4, handletextpad=0.5, labelspacing=0.4,
        borderpad=0.4, borderaxespad=0.0,
    )

    # ── (b) Per-config sensitivity (only when secondary configs exist) ───
    if ax_cfg is not None:
        configs = [(512, 128), (512, 512), (1, 512)]
        config_labels = [f"p={p} / g={g}" for p, g in configs]
        series = []
        if has_q8:
            series.append(("Qwen2.5-1.5B Q8_0",    qwen_q8_df,  QWEN_Q8_COLOR))
        if has_q4:
            series.append(("Qwen2.5-1.5B Q4_K_M",  qwen_q4_df,  QWEN_Q4_COLOR))
        if has_q2:
            series.append(("Qwen2.5-1.5B Q2_K",    qwen_q2_df,  QWEN_Q2_COLOR))
        if has_gemma_q8:
            series.append(("Gemma-2-2B-it Q8_0",   gemma_q8_df, GEMMA_Q8_COLOR))
        if has_gemma_q4:
            series.append(("Gemma-2-2B-it Q4_K_M", gemma_q4_df, GEMMA_Q4_COLOR))
        if has_gemma_q2:
            series.append(("Gemma-2-2B-it Q2_K",   gemma_q2_df, GEMMA_Q2_COLOR))
        if has_b:
            series.append(("BitNet b1.58 2B4T",    local_df,    BITNET_COLOR))

        x = np.arange(len(configs))
        width = 0.8 / max(len(series), 1)
        offsets = np.linspace(-(len(series) - 1) * width / 2,
                              (len(series) - 1) * width / 2, len(series))
        all_vals: list[float] = []
        for idx, (name, df, color) in enumerate(series):
            vals = []
            for p, g in configs:
                row = df[(df["n_prompt"] == p) & (df["n_gen"] == g)]
                v = float(row["throughput_tokens_s"].median()) if not row.empty else 0.0
                vals.append(v)
            all_vals.extend(vals)
            ax_cfg.bar(x + offsets[idx], vals, width, label=name, color=color)
        y_top = max(all_vals) * 1.18 if all_vals else 1
        for idx, (name, df, color) in enumerate(series):
            vals = []
            for p, g in configs:
                row = df[(df["n_prompt"] == p) & (df["n_gen"] == g)]
                v = float(row["throughput_tokens_s"].median()) if not row.empty else 0.0
                vals.append(v)
            for xi, v in zip(x + offsets[idx], vals):
                if v > 0:
                    ax_cfg.text(xi, v + y_top * 0.015, f"{v:.1f}",
                                ha="center", fontsize=8)
        ax_cfg.set_xticks(x)
        ax_cfg.set_xticklabels(config_labels)
        ax_cfg.set_ylabel("Throughput (tokens/s)")
        ax_cfg.set_ylim(0, y_top)
        ax_cfg.set_title("(b) Per-config sensitivity (locally measured models)", loc="left")
        ax_cfg.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      fontsize=8, frameon=True)
        ax_cfg.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Inference Throughput: BitNet b1.58 2B4T, Qwen2.5 1.5B & Gemma-2 2B vs FP16 Baselines (CPU)",
        fontsize=13,
    )
    fig.tight_layout()
    # bbox_inches="tight" prevents the long y-axis labels in panel (a) from
    # being clipped by the gridspec's default left margin.
    path = out_dir / "throughput_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_thread_scaling(out_dir: Path):
    """
    Throughput vs thread count for the seven locally measured models at the
    (n_prompt=512, n_gen=128) reference config.

    Reads dedicated *_thread_sweep.csv files written by
    `make benchmark-threads-<model>` (per-model targets defined in the
    Makefile).  Skips any model whose CSV doesn't exist.
    """
    # Canonical legend order: BitNet → Qwen (largest → smallest quant) →
    # Gemma (largest → smallest quant).  No paper rows on this plot since
    # we have no thread-sweep data for any of the paper baselines.
    sweeps = []
    for name, path, color in [
        ("BitNet b1.58 2B4T",    Path("results/bitnet_thread_sweep.csv"),   BITNET_COLOR),
        ("Qwen2.5-1.5B Q8_0",    Path("results/qwen_q8_thread_sweep.csv"),  QWEN_Q8_COLOR),
        ("Qwen2.5-1.5B Q4_K_M",  Path("results/qwen_q4_thread_sweep.csv"),  QWEN_Q4_COLOR),
        ("Qwen2.5-1.5B Q2_K",    Path("results/qwen_q2_thread_sweep.csv"),  QWEN_Q2_COLOR),
        ("Gemma-2-2B-it Q8_0",   Path("results/gemma_q8_thread_sweep.csv"), GEMMA_Q8_COLOR),
        ("Gemma-2-2B-it Q4_K_M", Path("results/gemma_q4_thread_sweep.csv"), GEMMA_Q4_COLOR),
        ("Gemma-2-2B-it Q2_K",   Path("results/gemma_q2_thread_sweep.csv"), GEMMA_Q2_COLOR),
    ]:
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            if "n_prompt" in df.columns and "threads" in df.columns:
                sweeps.append((name, df, color))

    if not sweeps:
        print("Skipping thread scaling plot: no thread sweep CSVs "
              "(run 'make benchmark-threads').")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    max_y = 0
    for name, df, color in sweeps:
        ref = df[(df["n_prompt"] == 512) & (df["n_gen"] == 128)].copy()
        if ref.empty:
            continue
        grouped = ref.groupby("threads")["throughput_tokens_s"].median().reset_index()
        grouped = grouped.sort_values("threads")
        ax.plot(grouped["threads"], grouped["throughput_tokens_s"],
                marker="o", color=color, label=name, linewidth=2, markersize=8)
        max_y = max(max_y, float(grouped["throughput_tokens_s"].max()))
        for x, y in zip(grouped["threads"], grouped["throughput_tokens_s"]):
            ax.annotate(f"{y:.1f}", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)

    # Ideal-linear-scaling guide: extend from each model's 1-thread number.
    # Helps visualize where each model falls off perfect scaling.
    for name, df, color in sweeps:
        ref = df[(df["n_prompt"] == 512) & (df["n_gen"] == 128)]
        one_t = ref[ref["threads"] == 1]["throughput_tokens_s"]
        if not one_t.empty:
            base = float(one_t.median())
            xs = [1, 2, 4, 6]
            ys = [base * t for t in xs]
            ax.plot(xs, ys, linestyle=":", color=color, alpha=0.35, linewidth=1)

    ax.set_xlabel("Threads")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(
        "Thread-count sensitivity at n_prompt=512, n_gen=128\n"
        "(Intel i5-9400F, 6 cores, no SMT; dotted lines = perfect linear scaling from 1-thread)"
    )
    ax.set_xticks([1, 2, 4, 6])
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0, max_y * 1.25 if max_y else 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "thread_scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _arch_csv_path(subdir: str | None, filename: str) -> Path:
    """Resolve a per-architecture CSV path.  subdir=None means the canonical
    results/<filename> (the Windows-native baseline)."""
    root = Path(__file__).parent.parent / "results"
    return root / subdir / filename if subdir else root / filename


def _load_arch_throughput(subdir: str | None) -> dict[str, float] | None:
    """Read the (512, 128) reference-config throughput for each of the
    locally-measured models from a given results subdirectory.  Returns
    None if no model CSV is readable under that subdir (so the caller
    can skip the arch entirely)."""
    out: dict[str, float] = {}
    for model, filename in [
        ("BitNet b1.58 2B4T",    "bitnet_step_metrics.csv"),
        ("Qwen2.5-1.5B Q8_0",    "qwen_q8_step_metrics.csv"),
        ("Qwen2.5-1.5B Q4_K_M",  "qwen_q4_step_metrics.csv"),
        ("Qwen2.5-1.5B Q2_K",    "qwen_q2_step_metrics.csv"),
        ("Gemma-2-2B-it Q8_0",   "gemma_q8_step_metrics.csv"),
        ("Gemma-2-2B-it Q4_K_M", "gemma_q4_step_metrics.csv"),
        ("Gemma-2-2B-it Q2_K",   "gemma_q2_step_metrics.csv"),
    ]:
        path = _arch_csv_path(subdir, filename)
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if "n_prompt" not in df.columns or "throughput_tokens_s" not in df.columns:
            continue
        ref = df[(df["n_prompt"] == 512) & (df["n_gen"] == 128)]
        if ref.empty:
            continue
        out[model] = float(pd.to_numeric(ref["throughput_tokens_s"]).median())
    return out or None


def plot_cross_arch_throughput(out_dir: Path):
    """
    Throughput across (architecture, model) at the (512, 128) reference config.

    Bars grouped by model on the x-axis; one bar per architecture within
    each group.  Reads from results/<subdir>/{bitnet,qwen_q8,qwen_q4,
    qwen_q2,gemma_q8,gemma_q4,gemma_q2}_step_metrics.csv per
    CROSS_ARCH_SOURCES.  Models and architectures with no readable CSVs
    are dropped silently so the plot degrades gracefully while the AWS
    sweep is being filled in.

    Backs REPORT §6.1 (cross-architecture generalization): if BitNet's
    advantage over Qwen Q8 holds on AMD AVX2 and ARM Neon, the Pareto
    ranking claim generalizes; if not, this plot localizes where it
    breaks.
    """
    archs = []
    for label, subdir, color in CROSS_ARCH_SOURCES:
        data = _load_arch_throughput(subdir)
        if data:
            archs.append((label, color, data))

    if not archs:
        print("Skipping cross-arch throughput plot: no per-arch CSVs.")
        return
    if len(archs) == 1:
        # Only the baseline is present.  Skip — there's nothing cross-arch
        # to compare against, and plot_throughput already shows this row.
        print("Skipping cross-arch throughput plot: only the baseline arch "
              "has data (run `make aws-benchmark` to populate others).")
        return

    # Restrict to models that have a measurement on at least one architecture
    # so the plot doesn't reserve x-slots for models that nothing populates.
    candidate_models = [
        "BitNet b1.58 2B4T", "Qwen2.5-1.5B Q8_0", "Qwen2.5-1.5B Q4_K_M",
        "Qwen2.5-1.5B Q2_K",
        "Gemma-2-2B-it Q8_0", "Gemma-2-2B-it Q4_K_M", "Gemma-2-2B-it Q2_K",
    ]
    models = [m for m in candidate_models
              if any(m in data for _, _, data in archs)]
    n_models = len(models)
    n_archs = len(archs)
    bar_width = 0.8 / n_archs
    x = list(range(n_models))

    fig, ax = plt.subplots(figsize=(11, 6))
    max_y = 0.0
    for i, (label, color, data) in enumerate(archs):
        offsets = [xi + (i - (n_archs - 1) / 2) * bar_width for xi in x]
        ys = [data.get(m, 0.0) for m in models]
        bars = ax.bar(offsets, ys, width=bar_width, color=color,
                      label=label, edgecolor="black", linewidth=0.5)
        for bx, by in zip(offsets, ys):
            if by > 0:
                ax.annotate(f"{by:.1f}", (bx, by),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=7)
        if ys:
            max_y = max(max_y, max(ys))

    ax.set_xticks(x)
    # Long model labels overlap horizontally — rotate 25° and right-anchor
    # so they slope off below the axis without colliding.
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(
        "Cross-architecture throughput at n_prompt=512, n_gen=128\n"
        "(higher is better; bars within each model = different CPU/OS combinations)"
    )
    ax.set_ylim(0, max_y * 1.18 if max_y else 1)
    # Family separators between BitNet / Qwen / Gemma groups.
    for b in _family_boundaries(models):
        ax.axvline(b - 0.5, color="#888888", linewidth=0.8,
                   linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "cross_arch_throughput.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_memory(local_df: pd.DataFrame, out_dir: Path,
                qwen_q8_df: pd.DataFrame | None = None,
                qwen_q4_df: pd.DataFrame | None = None,
                qwen_q2_df: pd.DataFrame | None = None,
                gemma_q8_df: pd.DataFrame | None = None,
                gemma_q4_df: pd.DataFrame | None = None,
                gemma_q2_df: pd.DataFrame | None = None):
    if not any(_has_bench_data(d) for d in
               (local_df, qwen_q8_df, qwen_q4_df, qwen_q2_df,
                gemma_q8_df, gemma_q4_df, gemma_q2_df)):
        print("Skipping memory plot: no benchmark CSVs (run 'make benchmark').")
        return
    labels, values, colors, hatches = _bar_series(
        local_df, qwen_q8_df, qwen_q4_df, "peak_rss_mb",
        qwen_q2_df=qwen_q2_df,
        gemma_q8_df=gemma_q8_df, gemma_q4_df=gemma_q4_df, gemma_q2_df=gemma_q2_df,
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.55)))
    max_val = max(v for v in values if v) or 1
    for i, (val, color, hatch) in enumerate(zip(values, colors, hatches)):
        ax.barh(i, val, color=color, hatch=hatch,
                edgecolor="#444444" if hatch else "#cccccc", linewidth=0.5)
        ax.text(val + max_val * 0.01, i, f"{val:.0f} MB", va="center", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Peak RSS (MB)")
    ax.set_title("Peak Memory: BitNet b1.58 2B4T, Qwen2.5 1.5B & Gemma-2 2B vs FP16 Baselines\n"
                 "(n_prompt=512, n_gen=128, CPU)")
    ax.set_xlim(0, max_val * 1.22)  # extra room so value labels don't kiss the right edge
    ax.invert_yaxis()
    # Family separators between BitNet / Qwen / Gemma groups.
    for b in _family_boundaries(labels):
        ax.axhline(b - 0.5, color="#888888", linewidth=0.8,
                   linestyle="--", alpha=0.6)
    # Legend goes OUTSIDE the axes (right of plot) so it can never occlude
    # bar value labels — previously the in-axes legend at lower-right ate
    # the "2,662 MB" Gemma Q4_K_M label.
    ax.legend(
        handles=_legend_handles(qwen_q8_df, qwen_q4_df, qwen_q2_df,
                                gemma_q8_df, gemma_q4_df, gemma_q2_df),
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=8, frameon=True, framealpha=0.95,
        handlelength=1.4, handletextpad=0.5, labelspacing=0.4,
        borderpad=0.4, borderaxespad=0.0,
    )
    fig.tight_layout()
    path = out_dir / "memory_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_accuracy(local_acc: dict, out_dir: Path,
                  qwen_q8_acc: dict | None = None,
                  qwen_q4_acc: dict | None = None,
                  qwen_q2_acc: dict | None = None,
                  gemma_q8_acc: dict | None = None,
                  gemma_q4_acc: dict | None = None,
                  gemma_q2_acc: dict | None = None):
    tasks = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]
    task_labels = ["ARC-Easy", "ARC-Challenge", "WinoGrande", "HellaSwag", "MMLU"]
    task_colors = ["#4C72B0", "#55A868", "#8172B2", "#64B5CD", "#C44E52"]

    # Canonical column order across the report:
    #   BitNet paper → BitNet ours →
    #   Qwen paper FP16 → Qwen Q8 → Qwen Q4 → Qwen Q2 (ours) →
    #   Gemma 2 2B paper (PT, from OTHER_BASELINES) → Gemma Q8/Q4/Q2 (ours).
    # Within each family: paper baseline first, then quants largest → smallest.
    # Family boundaries get a thin vertical separator so the BitNet / Qwen /
    # Gemma groups read as visually distinct sections.
    other_models = list(OTHER_BASELINES.keys())  # currently just Gemma 2 2B (PT)
    columns: list[tuple[str, str, callable]] = []
    columns.append(("BitNet b1.58 2B4T\n(paper)",
                    "BitNet", lambda t: BITNET_PAPER[t]))
    columns.append(("BitNet b1.58 2B4T\n(ours)",
                    "BitNet", lambda t: local_acc.get(t) or 0))
    columns.append(("Qwen2.5 1.5B\n(paper FP16)",
                    "Qwen", lambda t: QWEN_PAPER[t]))
    if qwen_q8_acc is not None:
        columns.append(("Qwen2.5-1.5B\nQ8_0 (ours)",
                        "Qwen", lambda t: qwen_q8_acc.get(t) or 0))
    if qwen_q4_acc is not None:
        columns.append(("Qwen2.5-1.5B\nQ4_K_M (ours)",
                        "Qwen", lambda t: qwen_q4_acc.get(t) or 0))
    if qwen_q2_acc is not None:
        columns.append(("Qwen2.5-1.5B\nQ2_K (ours)",
                        "Qwen", lambda t: qwen_q2_acc.get(t) or 0))
    for m in other_models:
        # Snapshot m so the late-bound lambda doesn't pick up the loop var.
        columns.append((m.replace(" ", "\n", 1),
                        "Gemma", (lambda mm: lambda t: OTHER_BASELINES[mm][t])(m)))
    if gemma_q8_acc is not None:
        columns.append(("Gemma-2-2B-it\nQ8_0 (ours)",
                        "Gemma", lambda t: gemma_q8_acc.get(t) or 0))
    if gemma_q4_acc is not None:
        columns.append(("Gemma-2-2B-it\nQ4_K_M (ours)",
                        "Gemma", lambda t: gemma_q4_acc.get(t) or 0))
    if gemma_q2_acc is not None:
        columns.append(("Gemma-2-2B-it\nQ2_K (ours)",
                        "Gemma", lambda t: gemma_q2_acc.get(t) or 0))

    all_models = [c[0] for c in columns]
    column_families = [c[1] for c in columns]
    column_value_fns = [c[2] for c in columns]

    x = np.arange(len(all_models))
    n_tasks = len(tasks)
    width = 0.15
    offsets = np.linspace(-(n_tasks - 1) * width / 2, (n_tasks - 1) * width / 2, n_tasks)

    fig, ax = plt.subplots(figsize=(max(14, len(all_models) * 1.6), 6))
    for i, (task, label, color) in enumerate(zip(tasks, task_labels, task_colors)):
        vals = [fn(task) for fn in column_value_fns]
        ax.bar(x + offsets[i], vals, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, ha="center")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    # Family separators between BitNet / Qwen / Gemma groups.
    for i in range(1, len(column_families)):
        if column_families[i] != column_families[i - 1]:
            ax.axvline(i - 0.5, color="#888888", linewidth=0.8,
                       linestyle="--", alpha=0.6)
    ax.set_title(
        "Accuracy Comparison: BitNet b1.58 2B4T, Qwen2.5 1.5B & Gemma-2 2B vs FP16 Baselines\n"
        "(0-shot except MMLU 5-shot; WinoGrande & HellaSwag use continuation scoring)"
    )
    ax.legend()
    fig.tight_layout()
    path = out_dir / "accuracy_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _accuracy_scatter(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    x_col: str,
    y_metric: str,        # one of TASK_LABELS keys, or the special value "mean_acc"
    x_label: str,
    y_label: str,
    title: str,
    filename: str,
    legend_loc: str = "best",
) -> None:
    """
    Shared scatter for both cost–accuracy and memory–accuracy comparisons.

    Hollow circles = paper-reported numbers; filled diamonds = our local
    measurements.  A dotted connector between each model's (paper, ours) pair
    makes the paper-to-measurement delta visible at a glance.

    `y_metric` selects the accuracy axis: either a task column name (e.g.
    "mmlu") or "mean_acc" — the mean across the five tasks in TASK_LABELS.
    """
    plot_df = df[df[x_col].notna() & (df[x_col] != "")].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col])

    if y_metric == "mean_acc":
        def _mean(row):
            vals = []
            for f in TASK_LABELS:
                v = row[f]
                if v == "" or pd.isna(v):
                    return None
                vals.append(float(v))
            return sum(vals) / len(vals)
        plot_df["_y"] = plot_df.apply(_mean, axis=1)
    else:
        plot_df = plot_df[plot_df[y_metric].notna() & (plot_df[y_metric] != "")].copy()
        plot_df["_y"] = pd.to_numeric(plot_df[y_metric], errors="coerce")

    plot_df = plot_df[plot_df["_y"].notna()]
    if plot_df.empty:
        print(f"Skipping {filename}: no rows with {x_col} + {y_metric} data.")
        return

    # Wider canvas: 10×6 was clipping the two-line title at left edge and
    # forcing point labels into the legend's column on the right.  12×6.5
    # gives the title room and leaves slack for label staggers around the
    # BitNet / Qwen-Q4 / Gemma-Q4 cluster.
    fig, ax = plt.subplots(figsize=(12, 6.5))

    QWEN_OURS_NAME    = "Qwen2.5-1.5B-Instruct Q8_0"
    QWEN_Q4_OURS_NAME = "Qwen2.5-1.5B-Instruct Q4_K_M"
    QWEN_Q2_OURS_NAME = "Qwen2.5-1.5B-Instruct Q2_K"
    GEMMA_Q8_OURS_NAME = "Gemma-2-2B-it Q8_0"
    GEMMA_Q4_OURS_NAME = "Gemma-2-2B-it Q4_K_M"
    GEMMA_Q2_OURS_NAME = "Gemma-2-2B-it Q2_K"
    QWEN_PAPER_NAME   = "Qwen2.5 1.5B"
    BITNET_NAME       = "BitNet b1.58 2B4T"

    by_key = {
        (row["model"], row["source"]): (row[x_col], row["_y"])
        for _, row in plot_df.iterrows()
    }
    # Midpoint of x-range used to steer paper labels AWAY from the nearer
    # plot edge.  cost_accuracy and memory_accuracy put Qwen FP16 paper
    # at the rightmost marker (label must go LEFT to avoid right-edge
    # clipping); speed_accuracy puts it at the leftmost marker because
    # FP16 is slow (label must go RIGHT instead).  Hard-coding the
    # direction per offset map clips in whichever plot we didn't tune for.
    _x_vals = pd.to_numeric(plot_df[x_col], errors="coerce").dropna()
    _x_mid = (_x_vals.min() + _x_vals.max()) / 2 if not _x_vals.empty else 0
    # Dotted connectors visualize the paper→ours or quant-chain delta.
    # Qwen chain: FP16 paper → Q8 → Q4 → Q2 (ours).
    # Gemma chain: Q8 → Q4 → Q2 (ours) — no paper FP16 baseline available.
    for ours_key, from_key, color in [
        ((BITNET_NAME, "ours"),          (BITNET_NAME, "paper"),             BITNET_COLOR),
        ((QWEN_OURS_NAME, "ours"),       (QWEN_PAPER_NAME, "paper (FP16)"),  QWEN_Q8_COLOR),
        ((QWEN_Q4_OURS_NAME, "ours"),    (QWEN_OURS_NAME, "ours"),           QWEN_Q4_COLOR),
        ((QWEN_Q2_OURS_NAME, "ours"),    (QWEN_Q4_OURS_NAME, "ours"),        QWEN_Q2_COLOR),
        ((GEMMA_Q4_OURS_NAME, "ours"),   (GEMMA_Q8_OURS_NAME, "ours"),       GEMMA_Q4_COLOR),
        ((GEMMA_Q2_OURS_NAME, "ours"),   (GEMMA_Q4_OURS_NAME, "ours"),       GEMMA_Q2_COLOR),
    ]:
        if ours_key in by_key and from_key in by_key:
            x0, y0 = by_key[from_key]
            x1, y1 = by_key[ours_key]
            ax.plot([x0, x1], [y0, y1], linestyle=":", color=color, alpha=0.55, zorder=2)

    for _, row in plot_df.iterrows():
        source, model = row["source"], row["model"]
        is_qwen_q8_ours    = source == "ours" and model == QWEN_OURS_NAME
        is_qwen_q4_ours    = source == "ours" and model == QWEN_Q4_OURS_NAME
        is_qwen_q2_ours    = source == "ours" and model == QWEN_Q2_OURS_NAME
        is_gemma_q8_ours   = source == "ours" and model == GEMMA_Q8_OURS_NAME
        is_gemma_q4_ours   = source == "ours" and model == GEMMA_Q4_OURS_NAME
        is_gemma_q2_ours   = source == "ours" and model == GEMMA_Q2_OURS_NAME
        is_ours = source == "ours"
        # Family-coded scatter colors: Qwen FP16 paper uses the deepest
        # Qwen-family navy, BitNet paper uses the deepest BitNet-family
        # burnt orange — same gradient story as in plot_throughput.
        if source == "paper (FP16)":
            color, label = QWEN_PAPER_COLOR, "Qwen2.5 1.5B FP16 (paper)"
        elif source == "paper":
            color, label = BITNET_PAPER_COLOR, "BitNet b1.58 2B4T (paper)"
        elif is_qwen_q2_ours:
            color, label = QWEN_Q2_COLOR, "Qwen2.5-1.5B Q2_K (ours)"
        elif is_qwen_q4_ours:
            color, label = QWEN_Q4_COLOR, "Qwen2.5-1.5B Q4_K_M (ours)"
        elif is_qwen_q8_ours:
            color, label = QWEN_Q8_COLOR, "Qwen2.5-1.5B Q8_0 (ours)"
        elif is_gemma_q2_ours:
            color, label = GEMMA_Q2_COLOR, "Gemma-2-2B-it Q2_K (ours)"
        elif is_gemma_q4_ours:
            color, label = GEMMA_Q4_COLOR, "Gemma-2-2B-it Q4_K_M (ours)"
        elif is_gemma_q8_ours:
            color, label = GEMMA_Q8_COLOR, "Gemma-2-2B-it Q8_0 (ours)"
        else:
            color, label = BITNET_COLOR, "BitNet b1.58 2B4T (ours)"

        if is_ours:
            ax.scatter(row[x_col], row["_y"],
                       facecolors=color, edgecolors=color, marker="D",
                       s=160, linewidths=1.5, label=label, zorder=3)
        else:
            ax.scatter(row[x_col], row["_y"],
                       facecolors="white", edgecolors=color, marker="o",
                       s=110, linewidths=2, label=label, zorder=3)

        # Paper annotations use the full model name to match the naming
        # convention in the CSV and other plots.  Our-measurement annotations
        # are kept SHORT (drop "_0", "_K_M", "_K" suffixes — the legend
        # already disambiguates) so labels don't overlap in clusters.
        if is_ours:
            ann = (model
                   .replace(" b1.58 2B4T", "")
                   .replace("Qwen2.5-1.5B-Instruct ", "Qwen ")
                   .replace("Gemma-2-2B-it ", "Gemma ")
                   .replace(" Q8_0", " Q8")
                   .replace(" Q4_K_M", " Q4")
                   .replace(" Q2_K", " Q2")) + " (ours)"
        else:
            # Shorten paper labels to match the ours-side abbreviation —
            # the legend already carries the full model name, so the
            # annotation only needs to distinguish row identity.
            ann = (model.replace(" (FP16)", "")
                        .replace("BitNet b1.58 2B4T", "BitNet")
                        .replace("Qwen2.5 1.5B", "Qwen 1.5B")) + " (paper)"
        # Stagger ours offsets so co-located points don't overlap.
        # Each variant gets a distinct quadrant + vertical slot relative
        # to its marker so labels never share screen real estate when
        # markers cluster (very common in cost_accuracy / memory_accuracy
        # near the Q4–Q8 high-accuracy band).  Horizontal offsets are
        # tuned to the shortened label set ("Qwen Q4 (ours)" etc.) — keep
        # in sync with the ann construction above.
        #   BitNet (paper) → UP, slight right (BitNet paper sits at the
        #                    plot's left edge — left offsets clip)
        #   Qwen FP16 paper → UP-LEFT (Qwen FP16 sits at right edge —
        #                     right offsets clip)
        #   BitNet (ours)  → UP-RIGHT
        #   Qwen Q8 (ours) → DOWN-RIGHT
        #   Qwen Q4 (ours) → DOWN-RIGHT, further below Q8 + BitNet-ours
        #   Qwen Q2 (ours) → DOWN-RIGHT (standalone)
        #   Gemma Q8 (ours) → UP-LEFT
        #   Gemma Q4 (ours) → DOWN-LEFT
        #   Gemma Q2 (ours) → DOWN-LEFT (standalone)
        if is_qwen_q2_ours:
            xy_offset = (10, -14)
        elif is_qwen_q4_ours:
            xy_offset = (10, -32)
        elif is_qwen_q8_ours:
            xy_offset = (10, -16)
        elif is_gemma_q2_ours:
            xy_offset = (-95, -14)
        elif is_gemma_q4_ours:
            xy_offset = (-95, -16)
        elif is_gemma_q8_ours:
            xy_offset = (-95, 12)
        elif is_ours:
            # BitNet — UP-LEFT of its marker, occupying the empty space
            # between BitNet-paper (up-left at higher y) and Qwen-Q2
            # (down-left at much lower y).  Right side of BitNet marker
            # is owned by Qwen Q4 / Q8 (ours) labels, which is why a
            # left offset is necessary even though the legend is on
            # the right of the figure.
            xy_offset = (-65, 8)
        elif "BitNet" in model:
            # BitNet paper baseline.  Edge-aware: if it's on the LEFT
            # half of x-range (cost / memory plots), go UP-RIGHT so the
            # label doesn't clip left.  On the RIGHT half (speed plot —
            # BitNet is the fastest model so its paper marker is at
            # high x), go UP-LEFT instead.
            if row[x_col] >= _x_mid:
                xy_offset = (-95, 12)
            else:
                xy_offset = (5, 12)
        else:
            # FP16 paper baseline for the bigger Qwen-family.  Edge-
            # aware: at the rightmost x in cost / memory plots go LEFT;
            # at the leftmost x in speed plot (FP16 is slow) go RIGHT.
            if row[x_col] >= _x_mid:
                xy_offset = (-105, 8)
            else:
                xy_offset = (8, 8)
        ax.annotate(ann, (row[x_col], row["_y"]),
                    textcoords="offset points", xytext=xy_offset, fontsize=8)

    handles, labels_list = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels_list):
        if l and l not in seen:
            seen[l] = h
    # Move legend outside the axes to the right so it never overlaps
    # the clustered (BitNet / Qwen Q4 / Gemma Q4) markers.
    ax.legend(seen.values(), seen.keys(), fontsize=9,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0)

    # Pad axes so annotation labels offset below/left of points don't
    # clip out of frame.  "Qwen Q2_K (ours)" sits below its marker
    # (offset -34 pt) and was being cut off at the x-axis.
    ax.margins(x=0.18, y=0.12)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cost_accuracy(df: pd.DataFrame, out_dir: Path, hardware_rate: float):
    """Cost vs mean accuracy (one plot, no per-task variants)."""
    x_label = f"Cost per 1,000 tokens (USD, c5.xlarge @ ${hardware_rate:.3f}/hr)"
    _accuracy_scatter(
        df, out_dir,
        x_col="cost_per_1k_tokens", y_metric="mean_acc",
        x_label=x_label,
        y_label="Mean accuracy across 5 benchmarks (%)",
        title="Cost–Accuracy Trade-off: BitNet b1.58 2B4T, Qwen2.5 1.5B & Gemma-2 2B vs FP16 Baselines\n"
              "(mean of ARC-Easy, ARC-Challenge, WinoGrande, HellaSwag, MMLU;  "
              "hollow ○ = paper,  filled ♦ = ours)",
        filename="cost_accuracy.png",
    )


def plot_speed_accuracy(df: pd.DataFrame, out_dir: Path):
    """tg128 throughput vs mean accuracy at (512, 128).

    The other axis-trade scatters (cost, memory) measure deployment
    economics; this one isolates *raw inference speed against capability*,
    making the speed-vs-accuracy Pareto visible without the cost / RSS
    confound.  Useful for the §3.4 + §3.7 discussion: which model is on
    the speed/accuracy frontier when everything else is set aside.
    """
    _accuracy_scatter(
        df, out_dir,
        x_col="throughput_tokens_s", y_metric="mean_acc",
        x_label="Throughput tg128 (tokens/s) at n_prompt=512, n_gen=128",
        y_label="Mean accuracy across 5 benchmarks (%)",
        title="Speed–Accuracy Trade-off: BitNet b1.58 2B4T, Qwen2.5 1.5B & Gemma-2 2B vs FP16 Baselines\n"
              "(mean of ARC-E, ARC-C, WinoGrande, HellaSwag, MMLU;  hollow ○ = paper,  filled ♦ = ours)",
        filename="speed_accuracy.png",
        legend_loc="lower right",
    )


def plot_energy_carbon(local_df: pd.DataFrame, qwen_q8_df: pd.DataFrame | None,
                       out_dir: Path, qwen_q4_df: pd.DataFrame | None = None,
                       qwen_q2_df: pd.DataFrame | None = None,
                       gemma_q8_df: pd.DataFrame | None = None,
                       gemma_q4_df: pd.DataFrame | None = None,
                       gemma_q2_df: pd.DataFrame | None = None):
    """
    Energy + carbon footprint per 1,000 tokens at n_prompt=512, n_gen=128.

    Single panel of horizontal bars; carbon shares the energy axis via a
    top-side twin axis transformed by the run's grid intensity
    (g CO₂ / kWh), which CodeCarbon resolves once per location.  Carbon
    is exactly proportional to energy at a given location, so the twin
    axis is an exact relabeling, not a separate measurement.

    FP16 baselines aren't shown — the paper doesn't report energy/CO₂.  Skips
    silently if no metrics CSV has populated energy_kwh.
    """
    def per_1k(df: pd.DataFrame | None, col: str, scale: float) -> float | None:
        if df is None or col not in df.columns:
            return None
        row = df[(df["n_prompt"] == 512) & (df["n_gen"] == 128)].copy()
        row[col] = pd.to_numeric(row[col], errors="coerce")
        vals = row[col].dropna()
        if vals.empty:
            return None
        return float(vals.median() * scale / (512 + 128))

    # Wh = kWh × 1000;  g = kg × 1000.  Both × 1000 again to express "per 1k tokens".
    bitnet_wh     = per_1k(local_df,     "energy_kwh", 1_000_000)
    qwen_q8_wh    = per_1k(qwen_q8_df,   "energy_kwh", 1_000_000)
    qwen_q4_wh    = per_1k(qwen_q4_df,   "energy_kwh", 1_000_000)
    qwen_q2_wh    = per_1k(qwen_q2_df,   "energy_kwh", 1_000_000)
    gemma_q8_wh   = per_1k(gemma_q8_df,  "energy_kwh", 1_000_000)
    gemma_q4_wh   = per_1k(gemma_q4_df,  "energy_kwh", 1_000_000)
    gemma_q2_wh   = per_1k(gemma_q2_df,  "energy_kwh", 1_000_000)
    bitnet_gco2   = per_1k(local_df,     "co2_kg",     1_000_000)
    qwen_q8_gco2  = per_1k(qwen_q8_df,   "co2_kg",     1_000_000)
    qwen_q4_gco2  = per_1k(qwen_q4_df,   "co2_kg",     1_000_000)
    qwen_q2_gco2  = per_1k(qwen_q2_df,   "co2_kg",     1_000_000)
    gemma_q8_gco2 = per_1k(gemma_q8_df,  "co2_kg",     1_000_000)
    gemma_q4_gco2 = per_1k(gemma_q4_df,  "co2_kg",     1_000_000)
    gemma_q2_gco2 = per_1k(gemma_q2_df,  "co2_kg",     1_000_000)

    if all(v is None for v in (bitnet_wh, qwen_q8_wh, qwen_q4_wh, qwen_q2_wh,
                               gemma_q8_wh, gemma_q4_wh, gemma_q2_wh)):
        print("Skipping energy/carbon plot: no energy_kwh data in benchmark CSVs.")
        return

    # Canonical row order: BitNet → Qwen (Q8/Q4/Q2) → Gemma (Q8/Q4/Q2).
    rows: list[tuple[str, str, float | None, float | None]] = []
    if bitnet_wh is not None:
        rows.append(("BitNet b1.58 2B4T (ours)", BITNET_COLOR, bitnet_wh, bitnet_gco2))
    if qwen_q8_wh is not None:
        rows.append(("Qwen2.5-1.5B Q8_0 (ours)", QWEN_Q8_COLOR, qwen_q8_wh, qwen_q8_gco2))
    if qwen_q4_wh is not None:
        rows.append(("Qwen2.5-1.5B Q4_K_M (ours)", QWEN_Q4_COLOR, qwen_q4_wh, qwen_q4_gco2))
    if qwen_q2_wh is not None:
        rows.append(("Qwen2.5-1.5B Q2_K (ours)", QWEN_Q2_COLOR, qwen_q2_wh, qwen_q2_gco2))
    if gemma_q8_wh is not None:
        rows.append(("Gemma-2-2B-it Q8_0 (ours)", GEMMA_Q8_COLOR, gemma_q8_wh, gemma_q8_gco2))
    if gemma_q4_wh is not None:
        rows.append(("Gemma-2-2B-it Q4_K_M (ours)", GEMMA_Q4_COLOR, gemma_q4_wh, gemma_q4_gco2))
    if gemma_q2_wh is not None:
        rows.append(("Gemma-2-2B-it Q2_K (ours)", GEMMA_Q2_COLOR, gemma_q2_wh, gemma_q2_gco2))

    # Grid intensity (g CO₂ per Wh) is constant per location — derive from any
    # row with both metrics so the twin axis is an exact relabeling.  Average
    # across rows to absorb floating-point noise in per_1k's median step.
    pairs = [(r[2], r[3]) for r in rows if r[2] and r[3]]
    g_per_wh = (sum(c / e for e, c in pairs) / len(pairs)) if pairs else None

    fig, ax = plt.subplots(figsize=(11, max(3.5, len(rows) * 0.9 + 1)))
    finite_wh = [r[2] for r in rows if r[2] is not None]
    max_wh = max(finite_wh) if finite_wh else 1.0
    for i, (label, color, wh, gco2) in enumerate(rows):
        if wh is None:
            ax.text(0, i, "  (no data)", va="center", fontsize=9, color="#888")
            continue
        ax.barh(i, wh, color=color, edgecolor="#444444", linewidth=0.5)
        annotation = f"{wh:.2f} Wh"
        if gco2 is not None:
            annotation += f"   ·   {gco2:.3f} g CO₂"
        ax.text(wh + max_wh * 0.02, i, annotation, va="center", fontsize=10)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max_wh * 1.45)
    ax.set_xlabel("Energy per 1,000 tokens (Wh)")
    ax.grid(axis="x", alpha=0.3)
    # Family separators between BitNet / Qwen / Gemma groups.
    for b in _family_boundaries([r[0] for r in rows]):
        ax.axhline(b - 0.5, color="#888888", linewidth=0.8,
                   linestyle="--", alpha=0.6)

    if g_per_wh is not None:
        ax_c = ax.secondary_xaxis(
            "top", functions=(lambda x: x * g_per_wh, lambda x: x / g_per_wh)
        )
        ax_c.set_xlabel(
            f"Carbon emissions per 1,000 tokens (g CO₂  ·  "
            f"grid intensity ≈ {g_per_wh * 1000:.0f} g/kWh)"
        )

    ax.set_title(
        "Inference Energy and Carbon per 1,000 tokens\n"
        "(CPU at n_prompt=512, n_gen=128; CO₂ axis = energy × local grid intensity from CodeCarbon)"
    )
    fig.tight_layout()
    path = out_dir / "energy_carbon_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cloud_cost_comparison(local_df: pd.DataFrame,
                               qwen_q8_df: pd.DataFrame | None,
                               qwen_q4_df: pd.DataFrame | None,
                               out_dir: Path,
                               hardware_rate: float,
                               electricity_rate: float,
                               qwen_q2_df: pd.DataFrame | None = None,
                               gemma_q8_df: pd.DataFrame | None = None,
                               gemma_q4_df: pd.DataFrame | None = None,
                               gemma_q2_df: pd.DataFrame | None = None):
    """
    Cost-per-1k-output-tokens across self-hosted (BitNet / Qwen) and cloud
    API services.  Each self-hosted model contributes two bars:
      - AWS c5.xlarge proxy (hatched) — what cloud rental would cost
      - local electricity (solid)     — marginal cost on hardware you own

    Cloud-API rows are taken from `CLOUD_API_PRICING` (output token rates).
    Log-x scale because prices span ~3 orders of magnitude.
    """
    # Source order (BitNet → Qwen → Gemma) is irrelevant downstream — entries
    # get sorted by cost before plotting — but kept in the canonical order so
    # the code reads consistently with the rest of the report's plots.
    entries: list[tuple[str, float, str, str]] = []  # (label, cost, color, hatch)
    for name, df, color in [
        ("BitNet b1.58 2B4T",    local_df,    BITNET_COLOR),
        ("Qwen2.5-1.5B Q8_0",    qwen_q8_df,  QWEN_Q8_COLOR),
        ("Qwen2.5-1.5B Q4_K_M",  qwen_q4_df,  QWEN_Q4_COLOR),
        ("Qwen2.5-1.5B Q2_K",    qwen_q2_df,  QWEN_Q2_COLOR),
        ("Gemma-2-2B-it Q8_0",   gemma_q8_df, GEMMA_Q8_COLOR),
        ("Gemma-2-2B-it Q4_K_M", gemma_q4_df, GEMMA_Q4_COLOR),
        ("Gemma-2-2B-it Q2_K",   gemma_q2_df, GEMMA_Q2_COLOR),
    ]:
        if df is None or df.empty:
            continue
        tps, _ = _bench_row(df)
        if tps is not None:
            entries.append((f"{name} (ours, AWS c5.xlarge proxy)",
                            cost_per_1k(tps, hardware_rate), color, "///"))
        e = energy_cost_per_1k(df, electricity_rate)
        if e is not None:
            entries.append((f"{name} (ours, local electricity)", e, color, ""))

    for name, price_per_million in CLOUD_API_PRICING.items():
        entries.append((f"{name} (API output)",
                        price_per_million / 1000.0, CLOUD_API_COLOR, ""))

    if not entries:
        print("Skipping cloud cost plot: no self-hosted cost data.")
        return

    entries.sort(key=lambda e: e[1])  # cheapest at top after invert_yaxis

    # Widen significantly: long row labels (e.g. "Qwen2.5-1.5B Q4_K_M
    # (ours, local electricity)") + dollar value labels on the right +
    # legend outside need ~16 inches to avoid clipping.
    fig, ax = plt.subplots(figsize=(16, max(5, len(entries) * 0.45)))
    for i, (label, cost, color, hatch) in enumerate(entries):
        ax.barh(i, cost, color=color, hatch=hatch,
                edgecolor="#444444" if hatch else "#cccccc", linewidth=0.5)
        if cost >= 1.0:
            text = f"${cost:.2f}"
        elif cost >= 0.01:
            text = f"${cost:.4f}"
        else:
            text = f"${cost:.6f}"
        ax.text(cost * 1.15, i, text, va="center", fontsize=9)

    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e[0] for e in entries])
    ax.invert_yaxis()
    ax.set_xscale("log")
    # Extend xlim so the text labels at the right edge of the longest bars
    # don't collide with the figure border on log scale.
    finite_costs = [c for _, c, *_ in entries]
    ax.set_xlim(min(finite_costs) * 0.6, max(finite_costs) * 6)
    ax.set_xlabel("Cost per 1,000 output tokens (USD, log scale)")
    # Escape $ as \$ so matplotlib's mathtext doesn't treat them as delimiters
    # (it would otherwise italicize the text between two literal $ signs).
    ax.set_title(
        "Cost: Self-hosted (BitNet / Qwen) vs Cloud API Services\n"
        f"(AWS c5.xlarge @ \\${hardware_rate:.3f}/hr · electricity @ \\${electricity_rate:.2f}/kWh"
        f" · API pricing per provider, retrieved {CLOUD_API_PRICING_DATE})"
    )

    from matplotlib.patches import Patch
    handles = []
    have_bitnet   = any(label.startswith("BitNet") for label, *_ in entries)
    have_qwen_q8  = any(label.startswith("Qwen2.5-1.5B Q8_0") for label, *_ in entries)
    have_q4       = any(label.startswith("Qwen2.5-1.5B Q4_K_M") for label, *_ in entries)
    have_q2       = any(label.startswith("Qwen2.5-1.5B Q2_K") for label, *_ in entries)
    have_gemma_q8 = any(label.startswith("Gemma-2-2B-it Q8_0") for label, *_ in entries)
    have_gemma_q4 = any(label.startswith("Gemma-2-2B-it Q4_K_M") for label, *_ in entries)
    have_gemma_q2 = any(label.startswith("Gemma-2-2B-it Q2_K") for label, *_ in entries)
    if have_qwen_q8:
        handles += [
            Patch(facecolor=QWEN_Q8_COLOR, hatch="///", edgecolor="#444444",
                  label="Qwen Q8_0 (ours, AWS proxy)"),
            Patch(facecolor=QWEN_Q8_COLOR, edgecolor="#cccccc",
                  label="Qwen Q8_0 (ours, local electricity)"),
        ]
    if have_q4:
        handles += [
            Patch(facecolor=QWEN_Q4_COLOR, hatch="///", edgecolor="#444444",
                  label="Qwen Q4_K_M (ours, AWS proxy)"),
            Patch(facecolor=QWEN_Q4_COLOR, edgecolor="#cccccc",
                  label="Qwen Q4_K_M (ours, local electricity)"),
        ]
    if have_q2:
        handles += [
            Patch(facecolor=QWEN_Q2_COLOR, hatch="///", edgecolor="#444444",
                  label="Qwen Q2_K (ours, AWS proxy)"),
            Patch(facecolor=QWEN_Q2_COLOR, edgecolor="#cccccc",
                  label="Qwen Q2_K (ours, local electricity)"),
        ]
    if have_gemma_q8:
        handles += [
            Patch(facecolor=GEMMA_Q8_COLOR, hatch="///", edgecolor="#444444",
                  label="Gemma Q8_0 (ours, AWS proxy)"),
            Patch(facecolor=GEMMA_Q8_COLOR, edgecolor="#cccccc",
                  label="Gemma Q8_0 (ours, local electricity)"),
        ]
    if have_gemma_q4:
        handles += [
            Patch(facecolor=GEMMA_Q4_COLOR, hatch="///", edgecolor="#444444",
                  label="Gemma Q4_K_M (ours, AWS proxy)"),
            Patch(facecolor=GEMMA_Q4_COLOR, edgecolor="#cccccc",
                  label="Gemma Q4_K_M (ours, local electricity)"),
        ]
    if have_gemma_q2:
        handles += [
            Patch(facecolor=GEMMA_Q2_COLOR, hatch="///", edgecolor="#444444",
                  label="Gemma Q2_K (ours, AWS proxy)"),
            Patch(facecolor=GEMMA_Q2_COLOR, edgecolor="#cccccc",
                  label="Gemma Q2_K (ours, local electricity)"),
        ]
    if have_bitnet:
        handles += [
            Patch(facecolor=BITNET_COLOR, hatch="///", edgecolor="#444444",
                  label="BitNet (ours, AWS proxy)"),
            Patch(facecolor=BITNET_COLOR, edgecolor="#cccccc",
                  label="BitNet (ours, local electricity)"),
        ]
    handles.append(Patch(facecolor=CLOUD_API_COLOR, edgecolor="#cccccc",
                          label="Cloud API (output token price)"))
    # "center right" sits between the short (top) and long (bottom) bars at
    # the log-scale's right edge, where there's no bar competition either way.
    # Legend outside the plot — was previously overlapping the right-side
    # value labels at "center right".
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)

    fig.tight_layout()
    path = out_dir / "cloud_cost_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    print(f"  Note: Cloud API prices hardcoded as of {CLOUD_API_PRICING_DATE}; "
          f"verify before publication (compare_runs.py:CLOUD_API_PRICING).")


def plot_memory_accuracy(df: pd.DataFrame, out_dir: Path):
    """
    Memory vs mean accuracy (one plot, no per-task variants).

    Lower-left is more efficient (less memory per accuracy point); BitNet b1.58
    typically sits in a corner the FP16 baselines can't reach.  Same hollow-vs-
    filled marker convention as plot_cost_accuracy.
    """
    _accuracy_scatter(
        df, out_dir,
        x_col="peak_rss_mb", y_metric="mean_acc",
        x_label="Peak RSS (MB) — lower is better",
        y_label="Mean accuracy across 5 benchmarks (%)",
        title="Memory–Accuracy: BitNet b1.58 2B4T, Qwen2.5 1.5B & Gemma-2 2B vs FP16 Baselines\n"
              "(mean of ARC-Easy, ARC-Challenge, WinoGrande, HellaSwag, MMLU;  "
              "hollow ○ = paper,  filled ♦ = ours)",
        filename="memory_accuracy.png",
        legend_loc="lower right",
    )


def _cloud_display_name(full_name: str) -> str:
    """Wrap provider prefix onto its own line so x-axis labels stay compact."""
    if full_name.startswith("OpenAI "):
        return "OpenAI\n" + full_name[len("OpenAI "):]
    if full_name.startswith("Anthropic "):
        return "Anthropic\n" + full_name[len("Anthropic "):]
    return full_name


def plot_cloud_accuracy_comparison(local_acc: dict, out_dir: Path,
                                   qwen_q8_acc: dict | None = None,
                                   qwen_q4_acc: dict | None = None,
                                   qwen_q2_acc: dict | None = None,
                                   gemma_q8_acc: dict | None = None,
                                   gemma_q4_acc: dict | None = None,
                                   gemma_q2_acc: dict | None = None):
    """
    Grouped-bar accuracy plot mirroring plot_accuracy(), but comparing our
    locally-measured models (BitNet, Qwen Q8/Q4/Q2, Gemma 2 2B Q8/Q4/Q2)
    against cloud subscription APIs from CLOUD_API_PRICING.

    Cloud values come from CLOUD_API_ACCURACY.  Most providers only publish
    MMLU, so non-MMLU bars for cloud rows are typically empty (rendered as 0
    height — visually absent).  Cloud models appear in CLOUD_API_PRICING order
    (ascending output price).
    """
    tasks = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]
    task_labels = ["ARC-Easy", "ARC-Challenge", "WinoGrande", "HellaSwag", "MMLU"]
    task_colors = ["#4C72B0", "#55A868", "#8172B2", "#64B5CD", "#C44E52"]

    # Canonical self-hosted order: BitNet → Qwen (Q8 → Q4 → Q2) →
    # Gemma (Q8 → Q4 → Q2).  Cloud APIs appended below in pricing order.
    all_models: list[str] = []
    model_accs: dict[str, dict] = {}
    label = "BitNet b1.58 2B4T\n(ours)"
    all_models.append(label); model_accs[label] = local_acc
    if qwen_q8_acc is not None:
        label = "Qwen2.5-1.5B\nQ8_0 (ours)"
        all_models.append(label); model_accs[label] = qwen_q8_acc
    if qwen_q4_acc is not None:
        label = "Qwen2.5-1.5B\nQ4_K_M (ours)"
        all_models.append(label); model_accs[label] = qwen_q4_acc
    if qwen_q2_acc is not None:
        label = "Qwen2.5-1.5B\nQ2_K (ours)"
        all_models.append(label); model_accs[label] = qwen_q2_acc
    if gemma_q8_acc is not None:
        label = "Gemma-2-2B-it\nQ8_0 (ours)"
        all_models.append(label); model_accs[label] = gemma_q8_acc
    if gemma_q4_acc is not None:
        label = "Gemma-2-2B-it\nQ4_K_M (ours)"
        all_models.append(label); model_accs[label] = gemma_q4_acc
    if gemma_q2_acc is not None:
        label = "Gemma-2-2B-it\nQ2_K (ours)"
        all_models.append(label); model_accs[label] = gemma_q2_acc
    # Mark where the self-hosted block ends so the separator goes there
    # rather than between Gemma rows.
    self_hosted_end = len(all_models)

    for cloud_name in CLOUD_API_PRICING:
        if cloud_name not in CLOUD_API_ACCURACY:
            continue
        display = _cloud_display_name(cloud_name)
        all_models.append(display)
        model_accs[display] = CLOUD_API_ACCURACY[cloud_name]

    x = np.arange(len(all_models))
    n_tasks = len(tasks)
    width = 0.15
    offsets = np.linspace(-(n_tasks - 1) * width / 2, (n_tasks - 1) * width / 2, n_tasks)

    fig, ax = plt.subplots(figsize=(max(14, len(all_models) * 1.6), 6))
    for i, (task, label, color) in enumerate(zip(tasks, task_labels, task_colors)):
        vals = []
        for m in all_models:
            v = model_accs[m].get(task)
            vals.append(float(v) if v is not None else 0.0)
        ax.bar(x + offsets[i], vals, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, ha="center")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    # Family separators between BitNet / Qwen / Gemma groups, plus a
    # stronger separator at the self-hosted ↔ cloud boundary.
    self_hosted_labels = all_models[:self_hosted_end]
    for b in _family_boundaries(self_hosted_labels):
        ax.axvline(b - 0.5, color="#888888", linewidth=0.8,
                   linestyle="--", alpha=0.6)
    if self_hosted_end < len(all_models):
        ax.axvline(self_hosted_end - 0.5, color="#444444", linewidth=1.2,
                   linestyle="-", alpha=0.8)
    ax.set_title(
        "Cloud-API vs Self-Hosted Accuracy\n"
        f"(self-hosted = our local Qwen / BitNet runs; cloud rows = each provider's "
        f"published evals as of {CLOUD_API_ACCURACY_DATE} — missing bars indicate "
        f"the provider hasn't published that benchmark)"
    )
    ax.legend()
    fig.tight_layout()
    path = out_dir / "cloud_accuracy_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cloud_cost_accuracy(comparison_df: pd.DataFrame, out_dir: Path):
    """
    Cost vs accuracy scatter mirroring plot_cost_accuracy(), but extended with
    cloud APIs from CLOUD_API_PRICING / CLOUD_API_ACCURACY.

    Uses MMLU on the y-axis instead of mean-of-5 because MMLU is the only
    benchmark consistently published across all cloud providers — a mean-of-5
    would silently penalize cloud rows whose other four scores aren't public.
    """
    QWEN_Q8_NAME = "Qwen2.5-1.5B-Instruct Q8_0"
    QWEN_Q4_NAME = "Qwen2.5-1.5B-Instruct Q4_K_M"
    QWEN_Q2_NAME = "Qwen2.5-1.5B-Instruct Q2_K"
    GEMMA_Q8_NAME = "Gemma-2-2B-it Q8_0"
    GEMMA_Q4_NAME = "Gemma-2-2B-it Q4_K_M"
    GEMMA_Q2_NAME = "Gemma-2-2B-it Q2_K"
    BITNET_NAME = "BitNet b1.58 2B4T"

    fig, ax = plt.subplots(figsize=(11, 7))

    ours_df = comparison_df[comparison_df["source"] == "ours"]
    for _, row in ours_df.iterrows():
        cost = row["cost_per_1k_tokens"]
        mmlu = row["mmlu"]
        if cost == "" or mmlu == "":
            continue
        cost, mmlu = float(cost), float(mmlu)
        model = row["model"]
        if model == BITNET_NAME:
            color, label = BITNET_COLOR, "BitNet b1.58 2B4T (ours)"
            ann, xy_offset = "BitNet (ours)", (8, -10)
        elif model == QWEN_Q8_NAME:
            color, label = QWEN_Q8_COLOR, "Qwen2.5-1.5B Q8_0 (ours)"
            ann, xy_offset = "Qwen Q8_0 (ours)", (8, -10)
        elif model == QWEN_Q4_NAME:
            color, label = QWEN_Q4_COLOR, "Qwen2.5-1.5B Q4_K_M (ours)"
            ann, xy_offset = "Qwen Q4_K_M (ours)", (8, -22)
        elif model == QWEN_Q2_NAME:
            color, label = QWEN_Q2_COLOR, "Qwen2.5-1.5B Q2_K (ours)"
            ann, xy_offset = "Qwen Q2_K (ours)", (8, -34)
        elif model == GEMMA_Q8_NAME:
            color, label = GEMMA_Q8_COLOR, "Gemma-2-2B-it Q8_0 (ours)"
            ann, xy_offset = "Gemma Q8_0 (ours)", (-100, -10)
        elif model == GEMMA_Q4_NAME:
            color, label = GEMMA_Q4_COLOR, "Gemma-2-2B-it Q4_K_M (ours)"
            ann, xy_offset = "Gemma Q4_K_M (ours)", (-100, -22)
        elif model == GEMMA_Q2_NAME:
            color, label = GEMMA_Q2_COLOR, "Gemma-2-2B-it Q2_K (ours)"
            ann, xy_offset = "Gemma Q2_K (ours)", (-100, -34)
        else:
            continue
        ax.scatter(cost, mmlu, facecolors=color, edgecolors=color, marker="D",
                   s=160, linewidths=1.5, label=label, zorder=3)
        ax.annotate(ann, (cost, mmlu), textcoords="offset points",
                    xytext=xy_offset, fontsize=8)

    for cloud_name, price_per_million in CLOUD_API_PRICING.items():
        acc = CLOUD_API_ACCURACY.get(cloud_name, {}).get("mmlu")
        if acc is None:
            continue
        cost = price_per_million / 1000.0
        ax.scatter(cost, acc, facecolors=CLOUD_API_COLOR, edgecolors="#222222",
                   marker="s", s=120, linewidths=1.0, label="Cloud API (output token price)",
                   zorder=3)
        ax.annotate(cloud_name, (cost, acc), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)

    handles, labels_list = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels_list):
        if l and l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9, loc="lower right")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 output tokens (USD, log scale)")
    ax.set_ylabel("MMLU Accuracy (%)")
    # Escape $ for matplotlib mathtext (otherwise $...$ italicizes the text).
    ax.set_title(
        "Cost vs MMLU Accuracy: Self-Hosted (Qwen / BitNet) vs Cloud APIs\n"
        f"(self-hosted = AWS c5.xlarge proxy · cloud = published output rate · "
        f"accuracy/pricing as of {CLOUD_API_ACCURACY_DATE})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "cloud_cost_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_mmlu_subject_heatmap(
    bitnet_full: dict | None,
    qwen_q8_full: dict | None,
    qwen_q4_full: dict | None,
    out_dir: Path,
    qwen_q2_full: dict | None = None,
    gemma_q8_full: dict | None = None,
    gemma_q4_full: dict | None = None,
    gemma_q2_full: dict | None = None,
):
    """
    Heatmap of MMLU per-subject accuracy for the locally-measured models.

    Filtered to subjects with substantial cross-model spread (max - min across
    models >= SPREAD_THRESHOLD percentage points), sorted by spread descending
    so the most differentiating subjects appear at top.  Columns are models;
    cells are annotated with accuracy and colored on a sequential blue scale.
    """
    from matplotlib.colors import Normalize

    SPREAD_THRESHOLD = 15.0

    # Canonical column order: BitNet → Qwen (Q8/Q4/Q2) → Gemma (Q8/Q4/Q2).
    models: list[tuple[str, dict]] = []
    for label, full in [
        ("BitNet 2B4T",  bitnet_full),
        ("Qwen Q8_0",    qwen_q8_full),
        ("Qwen Q4_K_M",  qwen_q4_full),
        ("Qwen Q2_K",    qwen_q2_full),
        ("Gemma Q8_0",   gemma_q8_full),
        ("Gemma Q4_K_M", gemma_q4_full),
        ("Gemma Q2_K",   gemma_q2_full),
    ]:
        if full is not None and "mmlu" in full and "subjects" in full["mmlu"]:
            models.append((label, full["mmlu"]["subjects"]))

    if len(models) < 2:
        print("Skipping MMLU subject heatmap: need at least 2 models with per-subject data.")
        return

    all_subjects: set[str] = set()
    for _, subj_dict in models:
        all_subjects.update(subj_dict.keys())

    def subj_accs(subj: str) -> list[float]:
        return [m[1][subj]["accuracy"] for m in models
                if subj in m[1] and m[1][subj].get("accuracy") is not None]

    spreads: dict[str, float] = {}
    for subj in all_subjects:
        accs = subj_accs(subj)
        if len(accs) == len(models):
            spreads[subj] = max(accs) - min(accs)

    differentiating = [s for s, sp in spreads.items() if sp >= SPREAD_THRESHOLD]
    subjects_sorted = sorted(differentiating, key=lambda s: spreads[s], reverse=True)

    if not subjects_sorted:
        print(f"Skipping MMLU subject heatmap: no subjects with spread >= {SPREAD_THRESHOLD}pp.")
        return

    n_subj = len(subjects_sorted)
    n_models = len(models)
    matrix = np.full((n_subj, n_models), np.nan)
    for j, (_, subj_dict) in enumerate(models):
        for i, subj in enumerate(subjects_sorted):
            acc = subj_dict.get(subj, {}).get("accuracy")
            if acc is not None:
                matrix[i, j] = acc

    display_names = [
        f"{s.replace('_', ' ').title()}  (Δ{spreads[s]:.0f})"
        for s in subjects_sorted
    ]

    fig, ax = plt.subplots(figsize=(max(7, n_models * 2.2), max(6, n_subj * 0.38)))
    norm = Normalize(vmin=25, vmax=80)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", norm=norm)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels([m[0] for m in models], fontsize=10)
    ax.set_yticks(range(n_subj))
    ax.set_yticklabels(display_names, fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    # Family separators between BitNet / Qwen / Gemma columns.
    for b in _family_boundaries([m[0] for m in models]):
        ax.axvline(b - 0.5, color="white", linewidth=2.0,
                   linestyle="-", alpha=0.95)

    for i in range(n_subj):
        for j in range(n_models):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > 62 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Accuracy (%)")

    total_subjects = len(all_subjects)
    ax.set_title(
        "MMLU Per-Subject Accuracy: BitNet b1.58 2B4T vs Qwen2.5 1.5B vs Gemma-2 2B\n"
        f"({n_subj}/{total_subjects} subjects with cross-model spread "
        f"≥ {SPREAD_THRESHOLD:.0f}pp, sorted by spread — 5-shot)",
        pad=20,
    )
    fig.tight_layout()
    path = out_dir / "mmlu_subject_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_accuracy_eval_cost(
    bitnet_full: dict | None,
    qwen_q8_full: dict | None,
    qwen_q4_full: dict | None,
    out_dir: Path,
    qwen_q2_full: dict | None = None,
    gemma_q8_full: dict | None = None,
    gemma_q4_full: dict | None = None,
    gemma_q2_full: dict | None = None,
):
    """
    Total wall-clock time and energy consumed by the accuracy evaluation
    suite, broken down by benchmark task and model.

    Single panel of stacked horizontal bars (hours per task per model) with
    a twin x-axis on top showing equivalent energy in kWh.  Inference is
    CPU-bound at ~100% utilization, so average power is roughly constant
    across models and energy ≈ hours × avg_power_kW — the twin axis is an
    approximation, set from the run's measured kWh/hour ratio (per model
    variance is typically a few percent).

    Useful for the reproducibility discussion: shows how expensive it is
    to replicate the full evaluation.  Skips silently if no accuracy JSON
    contains elapsed_s.
    """
    tasks = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]
    task_display = ["ARC-Easy", "ARC-Challenge", "WinoGrande", "HellaSwag", "MMLU"]
    task_colors = ["#4C72B0", "#55A868", "#8172B2", "#64B5CD", "#C44E52"]

    # Canonical row order: BitNet → Qwen (largest → smallest quant) →
    # Gemma (largest → smallest quant).  No paper rows on this plot since
    # eval cost is by definition a measurement of OUR runs.
    series: list[tuple[str, str, dict]] = []
    for label, color, full in [
        ("BitNet 2B4T",  BITNET_COLOR,   bitnet_full),
        ("Qwen Q8_0",    QWEN_Q8_COLOR,  qwen_q8_full),
        ("Qwen Q4_K_M",  QWEN_Q4_COLOR,  qwen_q4_full),
        ("Qwen Q2_K",    QWEN_Q2_COLOR,  qwen_q2_full),
        ("Gemma Q8_0",   GEMMA_Q8_COLOR, gemma_q8_full),
        ("Gemma Q4_K_M", GEMMA_Q4_COLOR, gemma_q4_full),
        ("Gemma Q2_K",   GEMMA_Q2_COLOR, gemma_q2_full),
    ]:
        if full is None:
            continue
        has_time = any(full.get(t, {}).get("elapsed_s") is not None for t in tasks)
        if has_time:
            series.append((label, color, full))

    if not series:
        print("Skipping accuracy eval cost plot: no elapsed_s data in accuracy JSONs.")
        return

    # Different eval runs used different sample sizes (e.g. BitNet's
    # original ARC/Wino/Hella ran at LIMIT=500, the rest at LIMIT=100;
    # MMLU mixed LIMIT=100 and post-bugfix MMLU_LIMIT=10 reruns).  Scale
    # every (model, task) bar to the canonical "100 samples per scoring
    # invocation" so the totals are an apples-to-apples comparison of
    # "what one full eval pass takes."  Per-subject for MMLU; per-task
    # for the rest.  Detection comes straight from the recorded `total`
    # field (or first subject's `total` for MMLU).
    REF_SAMPLES = 100  # canonical samples per task (per-subject for MMLU)
    any_scaled = False

    fig, ax = plt.subplots(figsize=(13, max(4, len(series) * 0.9 + 1.5)))
    y = np.arange(len(series))

    left_t = np.zeros(len(series))
    total_hours = np.zeros(len(series))
    total_kwh = np.zeros(len(series))
    for ti, (task, td, tc) in enumerate(zip(tasks, task_display, task_colors)):
        hours = []
        kwh = []
        for _, _, full in series:
            info = full.get(task, {})
            h = info.get("elapsed_s")
            e = info.get("energy_kwh")
            scale = 1.0
            if task == "mmlu":
                subs = info.get("subjects") or {}
                if subs:
                    sample_size = next(iter(subs.values())).get("total", REF_SAMPLES)
                    if sample_size and sample_size != REF_SAMPLES:
                        scale = REF_SAMPLES / sample_size
                        any_scaled = True
            else:
                # Non-MMLU: scale by the task's total sample count.  Note
                # `total` excludes skipped samples in the result dict, but
                # for time/energy normalization we want sample count
                # processed (total + skipped), which is what the eval ran.
                processed = info.get("total", 0) + info.get("skipped", 0)
                if processed and processed != REF_SAMPLES:
                    scale = REF_SAMPLES / processed
                    any_scaled = True
            hours.append((h / 3600.0) * scale if h is not None else 0)
            kwh.append((e * scale) if e is not None else 0)
        hours_arr = np.array(hours)
        kwh_arr = np.array(kwh)
        ax.barh(y, hours_arr, left=left_t, label=td, color=tc,
                edgecolor="white", linewidth=0.5)
        left_t += hours_arr
        total_hours += hours_arr
        total_kwh += kwh_arr

    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in series])
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock time (hours)")
    max_h = float(max(total_hours)) if len(total_hours) else 1.0
    ax.set_xlim(0, max_h * 1.18)
    for yi, h, k in zip(y, total_hours, total_kwh):
        ax.text(h + max_h * 0.015, yi, f"{h:.1f} h   ·   {k:.3f} kWh",
                va="center", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    # Family separators between BitNet / Qwen / Gemma groups.
    for b in _family_boundaries([s[0] for s in series]):
        ax.axhline(b - 0.5, color="#888888", linewidth=0.8,
                   linestyle="--", alpha=0.6)
    # Legend outside the axes — was previously eating the "12.8 h" value
    # label on Gemma Q2_K's bar at lower-right.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, title="Benchmark", frameon=True)

    # Twin x-axis: avg kWh/hour ratio gives an approximate energy scale.
    # CPU-bound inference ⇒ power ≈ constant, so this is close to exact in
    # practice (per-model variance is typically a few percent).
    valid = [(h, k) for h, k in zip(total_hours, total_kwh) if h > 0 and k > 0]
    if valid:
        kwh_per_hour = sum(k / h for h, k in valid) / len(valid)
        ax_e = ax.secondary_xaxis(
            "top",
            functions=(lambda x: x * kwh_per_hour, lambda x: x / kwh_per_hour),
        )
        ax_e.set_xlabel(
            f"Energy (kWh)  —  approx. via avg power ≈ {kwh_per_hour * 1000:.0f} W"
        )

    scaled_note = (
        f"\nNormalized to {REF_SAMPLES} samples per task (per-subject for MMLU) — "
        "off-canonical runs scaled linearly for apples-to-apples comparison"
        if any_scaled else ""
    )
    ax.set_title(
        "Total Accuracy Evaluation Cost\n"
        "(stacked by benchmark — ARC-Easy / ARC-Challenge / WinoGrande / HellaSwag / MMLU)"
        + scaled_note,
        loc="left",
    )
    fig.tight_layout()
    path = out_dir / "accuracy_eval_cost.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = parse_args()
    local_df = load_local(Path(args.results))
    local_acc = load_accuracy(Path(args.accuracy))
    local_acc_full = load_accuracy_full(Path(args.accuracy))
    qwen_q8_df = load_qwen(Path(args.qwen_q8_results))
    qwen_q8_acc = load_accuracy(Path(args.qwen_q8_accuracy))
    qwen_q8_acc_full = load_accuracy_full(Path(args.qwen_q8_accuracy))
    qwen_q4_df = load_qwen(Path(args.qwen_q4_results))
    qwen_q4_acc = load_accuracy(Path(args.qwen_q4_accuracy))
    qwen_q4_acc_full = load_accuracy_full(Path(args.qwen_q4_accuracy))
    qwen_q2_df = load_qwen(Path(args.qwen_q2_results))
    qwen_q2_acc = load_accuracy(Path(args.qwen_q2_accuracy))
    qwen_q2_acc_full = load_accuracy_full(Path(args.qwen_q2_accuracy))
    gemma_q8_df = load_qwen(Path(args.gemma_q8_results))
    gemma_q8_acc = load_accuracy(Path(args.gemma_q8_accuracy))
    gemma_q8_acc_full = load_accuracy_full(Path(args.gemma_q8_accuracy))
    gemma_q4_df = load_qwen(Path(args.gemma_q4_results))
    gemma_q4_acc = load_accuracy(Path(args.gemma_q4_accuracy))
    gemma_q4_acc_full = load_accuracy_full(Path(args.gemma_q4_accuracy))
    gemma_q2_df = load_qwen(Path(args.gemma_q2_results))
    gemma_q2_acc = load_accuracy(Path(args.gemma_q2_accuracy))
    gemma_q2_acc_full = load_accuracy_full(Path(args.gemma_q2_accuracy))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df = build_comparison_df(
        local_df, local_acc, args.hardware_rate,
        qwen_q8_df, qwen_q8_acc,
        qwen_q4_df, qwen_q4_acc,
        qwen_q2_df=qwen_q2_df, qwen_q2_acc=qwen_q2_acc,
        gemma_q8_df=gemma_q8_df, gemma_q8_acc=gemma_q8_acc,
        gemma_q4_df=gemma_q4_df, gemma_q4_acc=gemma_q4_acc,
        gemma_q2_df=gemma_q2_df, gemma_q2_acc=gemma_q2_acc,
        electricity_rate=args.electricity_rate,
    )
    write_comparison_csv(comparison_df, Path(args.csv))

    plot_throughput(local_df, PLOTS_DIR, qwen_q8_df, qwen_q4_df,
                    qwen_q2_df=qwen_q2_df,
                    gemma_q8_df=gemma_q8_df, gemma_q4_df=gemma_q4_df,
                    gemma_q2_df=gemma_q2_df)
    plot_thread_scaling(PLOTS_DIR)
    plot_cross_arch_throughput(PLOTS_DIR)
    plot_memory(local_df, PLOTS_DIR, qwen_q8_df, qwen_q4_df,
                qwen_q2_df=qwen_q2_df,
                gemma_q8_df=gemma_q8_df, gemma_q4_df=gemma_q4_df,
                gemma_q2_df=gemma_q2_df)
    plot_accuracy(local_acc, PLOTS_DIR, qwen_q8_acc, qwen_q4_acc,
                  qwen_q2_acc=qwen_q2_acc,
                  gemma_q8_acc=gemma_q8_acc, gemma_q4_acc=gemma_q4_acc,
                  gemma_q2_acc=gemma_q2_acc)
    plot_cost_accuracy(comparison_df, PLOTS_DIR, args.hardware_rate)
    plot_speed_accuracy(comparison_df, PLOTS_DIR)
    plot_energy_carbon(local_df, qwen_q8_df, PLOTS_DIR, qwen_q4_df,
                       qwen_q2_df=qwen_q2_df,
                       gemma_q8_df=gemma_q8_df, gemma_q4_df=gemma_q4_df,
                       gemma_q2_df=gemma_q2_df)
    plot_cloud_cost_comparison(local_df, qwen_q8_df, qwen_q4_df, PLOTS_DIR,
                               args.hardware_rate, args.electricity_rate,
                               qwen_q2_df=qwen_q2_df,
                               gemma_q8_df=gemma_q8_df, gemma_q4_df=gemma_q4_df,
                               gemma_q2_df=gemma_q2_df)
    plot_memory_accuracy(comparison_df, PLOTS_DIR)
    plot_cloud_accuracy_comparison(local_acc, PLOTS_DIR, qwen_q8_acc, qwen_q4_acc,
                                   qwen_q2_acc=qwen_q2_acc,
                                   gemma_q8_acc=gemma_q8_acc, gemma_q4_acc=gemma_q4_acc,
                                   gemma_q2_acc=gemma_q2_acc)
    plot_cloud_cost_accuracy(comparison_df, PLOTS_DIR)
    plot_mmlu_subject_heatmap(local_acc_full, qwen_q8_acc_full, qwen_q4_acc_full, PLOTS_DIR,
                              qwen_q2_full=qwen_q2_acc_full,
                              gemma_q8_full=gemma_q8_acc_full,
                              gemma_q4_full=gemma_q4_acc_full,
                              gemma_q2_full=gemma_q2_acc_full)
    plot_accuracy_eval_cost(local_acc_full, qwen_q8_acc_full, qwen_q4_acc_full, PLOTS_DIR,
                            qwen_q2_full=qwen_q2_acc_full,
                            gemma_q8_full=gemma_q8_acc_full,
                            gemma_q4_full=gemma_q4_acc_full,
                            gemma_q2_full=gemma_q2_acc_full)

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
