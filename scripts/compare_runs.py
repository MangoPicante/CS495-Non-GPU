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

# AWS c5.xlarge on-demand, us-east-1 (4 vCPUs — matches 4-thread benchmark condition)
# Source: https://instances.vantage.sh/aws/ec2/c5.xlarge (retrieved 2026-05-08)
DEFAULT_HARDWARE_RATE = 0.170

# US residential average electricity rate (EIA, 2026 estimate, $/kWh).
# Used to convert CodeCarbon's energy_kwh into a dollar cost.  Override with
# --electricity-rate for industrial (~$0.10) or your local utility's rate.
DEFAULT_ELECTRICITY_RATE = 0.16

# Published FP16 baseline numbers from arXiv:2504.12285 Table 1
# Throughput condition: n_prompt=512, n_gen=128, single-thread x86 CPU
# Qwen2.5 1.5B is broken out separately (QWEN_PAPER) because we run it locally
# and want to plot the paper target alongside our measurement.
OTHER_BASELINES = {
    "LLaMA 3.2 1B": {
        "throughput_tokens_s": 4.5,  "peak_rss_mb": 2600,
        "arc_easy": 69.87, "arc_challenge": 41.04,
        "winogrande": 60.77, "hellaswag": 61.05, "mmlu": 42.12,
    },
    "Gemma-3 1B": {
        "throughput_tokens_s": 4.1,  "peak_rss_mb": 2700,
        "arc_easy": 79.42, "arc_challenge": 46.25,
        "winogrande": 66.38, "hellaswag": 72.15, "mmlu": 50.33,
    },
    "SmolLM2 1.7B": {
        "throughput_tokens_s": 3.5,  "peak_rss_mb": 3300,
        "arc_easy": 81.82, "arc_challenge": 52.99,
        "winogrande": 68.67, "hellaswag": 72.29, "mmlu": 51.77,
    },
    "MiniCPM 2B": {
        "throughput_tokens_s": 2.9,  "peak_rss_mb": 4100,
        "arc_easy": 82.20, "arc_challenge": 51.96,
        "winogrande": 68.27, "hellaswag": 75.08, "mmlu": 53.07,
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

# Color convention used across all comparison plots:
#   FP16 paper baselines   → blue   (Qwen FP16 paper grouped with the other FP16 papers)
#   BitNet                 → orange (paper hatched, ours solid)
#   Qwen Q8_0 (ours)       → green
#   Qwen Q4_K_M (ours)     → purple
OTHER_COLOR     = "#4C72B0"
BITNET_COLOR    = "#DD8452"
QWEN_Q8_COLOR      = "#55A868"
QWEN_Q4_COLOR   = "#8172B2"
CLOUD_API_COLOR = "#7F7F7F"

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
# can outpace the AWS sweep — the plot grows bars as CSVs land.  As of
# 2026-05-24 only the two i5-9400F rows are populated; c5/c6a/c7g
# placeholders activate once `make aws-benchmark-{c5,c6a,c7g}` writes
# their CSVs.
CROSS_ARCH_SOURCES = [
    # (label,                            subdir,             color)
    ("Windows / i5-9400F (AVX2)",        None,               "#4C72B0"),
    ("Linux Docker / i5-9400F (AVX2)",   "linux_docker_x86", "#7AAEDC"),
    ("AWS c5.xlarge (Intel AVX-512)",    "aws_c5_xlarge",    "#1F77B4"),
    ("AWS c6a.xlarge (AMD Zen3 AVX2)",   "aws_c6a_xlarge",   "#D62728"),
    ("AWS c7g.xlarge (ARM Graviton3)",   "aws_c7g_xlarge",   "#2CA02C"),
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
CLOUD_API_PRICING_DATE = "2026-05-15"
CLOUD_API_PRICING = {
    "OpenAI GPT-4o mini":          0.60,
    "Anthropic Claude Haiku 4.5":  5.00,
    "OpenAI GPT-4o":              10.00,
    "Anthropic Claude Sonnet 4.5": 15.00,
    "Anthropic Claude Opus 4.7":  75.00,
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
CLOUD_API_ACCURACY_DATE = "2026-05-20"
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
        # OpenAI tech report / launch announcement, May 2024.  System card's
        # MMLU is a medical-only subset (see footnote 6 of GPT-4o card) and
        # is not directly comparable.
        "arc_easy": None, "arc_challenge": 96.4,
        "winogrande": 87.0, "hellaswag": 95.3,
        "mmlu": 88.7,
    },
    "Anthropic Claude Sonnet 4.5": {
        # Anthropic announcement / third-party leaderboards — system card
        # is safety-focused and reports no general benchmarks.
        "arc_easy": None, "arc_challenge": None,
        "winogrande": None, "hellaswag": None,
        "mmlu": 88.0,
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
    electricity_rate: float = DEFAULT_ELECTRICITY_RATE,
) -> pd.DataFrame:
    ACC_FIELDS = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]
    rows = []

    # Row order: FP16 papers (other baselines → Qwen FP16 paper) → Qwen ours
    # (Q8, Q4) → BitNet ours → BitNet paper.  Keeps all FP16 papers together at
    # the top and pushes BitNet to the bottom so the comparison reads
    # "literature numbers first, our measurements next, BitNet last".
    for name, b in OTHER_BASELINES.items():
        rows.append({
            "model": name,
            "source": "paper (FP16)",
            "throughput_tokens_s": b["throughput_tokens_s"],
            "peak_rss_mb": b["peak_rss_mb"],
            "cost_per_1k_tokens": round(cost_per_1k(b["throughput_tokens_s"], hardware_rate), 6),
            # No CodeCarbon measurement for paper rows — would require running
            # those FP16 baselines locally, which is outside this project's scope.
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


def _bar_series(local_df: pd.DataFrame, qwen_q8_df: pd.DataFrame | None,
                qwen_q4_df: pd.DataFrame | None,
                metric: str) -> tuple[list[str], list[float], list[str], list[str]]:
    """
    Build the (labels, values, colors, hatches) tuple for a horizontal bar chart.

    Order: other FP16 paper baselines → Qwen paper FP16 → Qwen Q8 ours →
    Qwen Q4 ours → BitNet ours → BitNet paper.  Each "ours" row is conditional
    on local data existing.  `metric` selects which field to read from each
    paper dict / which bench column.
    """
    metric_col = "throughput_tokens_s" if metric == "throughput_tokens_s" else "peak_rss_mb"
    bitnet_tps, bitnet_rss = _bench_row(local_df)
    bitnet_local = bitnet_tps if metric_col == "throughput_tokens_s" else bitnet_rss

    # Qwen FP16 paper is rendered identically to the other FP16 baselines
    # (same color, no hatch) so they collapse to a single "FP16 baseline (paper)"
    # legend entry.
    labels = list(OTHER_BASELINES.keys()) + ["Qwen2.5 1.5B (paper FP16)"]
    values = [OTHER_BASELINES[m][metric_col] for m in OTHER_BASELINES] + [QWEN_PAPER[metric_col]]
    colors  = [OTHER_COLOR] * (len(OTHER_BASELINES) + 1)
    hatches = [""]          * (len(OTHER_BASELINES) + 1)

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

    if bitnet_local is not None:
        labels.append("BitNet b1.58 2B4T (ours)")
        values.append(bitnet_local)
        colors.append(BITNET_COLOR)
        hatches.append("")

    labels.append("BitNet b1.58 2B4T (paper)")
    values.append(BITNET_PAPER[metric_col])
    colors.append(BITNET_COLOR)
    hatches.append("///")

    return labels, values, colors, hatches


def _legend_handles(qwen_q8_df: pd.DataFrame | None, qwen_q4_df: pd.DataFrame | None = None):
    from matplotlib.patches import Patch
    # All FP16 papers (including Qwen2.5 1.5B) share a single legend entry so
    # the legend doesn't double-count Qwen as both "FP16 baseline" and a
    # standalone hatched series.
    handles = [
        Patch(facecolor=OTHER_COLOR, edgecolor="#cccccc", label="FP16 baseline (paper)"),
    ]
    if qwen_q8_df is not None:
        handles.append(Patch(facecolor=QWEN_Q8_COLOR, edgecolor="#cccccc", label="Qwen2.5-1.5B Q8_0 (ours)"))
    if qwen_q4_df is not None:
        handles.append(Patch(facecolor=QWEN_Q4_COLOR, edgecolor="#cccccc", label="Qwen2.5-1.5B Q4_K_M (ours)"))
    handles += [
        Patch(facecolor=BITNET_COLOR, edgecolor="#cccccc", label="BitNet b1.58 2B4T (ours)"),
        Patch(facecolor=BITNET_COLOR, hatch="///", edgecolor="#444444", label="BitNet b1.58 2B4T (paper)"),
    ]
    return handles


def plot_throughput(local_df: pd.DataFrame, out_dir: Path,
                    qwen_q8_df: pd.DataFrame | None = None,
                    qwen_q4_df: pd.DataFrame | None = None):
    """
    Single unified throughput plot with two panels:

      (a) Cross-model bar chart at (n_prompt=512, n_gen=128) — paper FP16
          baselines + BitNet (paper, ours) + Qwen (paper FP16, Q8 ours, Q4 ours).
      (b) Per-config sensitivity — grouped bars across the three benchmarked
          (n_prompt, n_gen) configs for our three locally measured models.
          The paper FP16 baselines only publish numbers at (512, 128), so
          they don't appear in this panel.
    """
    has_b  = _has_bench_data(local_df)
    has_q8 = _has_bench_data(qwen_q8_df)
    has_q4 = _has_bench_data(qwen_q4_df)
    if not (has_b or has_q8 or has_q4):
        print("Skipping throughput plot: no benchmark CSVs (run 'make benchmark').")
        return

    labels, values, colors, hatches = _bar_series(local_df, qwen_q8_df, qwen_q4_df, "throughput_tokens_s")
    fig = plt.figure(figsize=(13, max(9, len(labels) * 0.55 + 4)))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(labels) * 0.55, 4.5], hspace=0.35)
    ax_main = fig.add_subplot(gs[0])
    ax_cfg  = fig.add_subplot(gs[1])

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
    ax_main.legend(handles=_legend_handles(qwen_q8_df, qwen_q4_df), loc="upper right", fontsize=8)

    # ── (b) Per-config sensitivity for the locally measured models ───────
    configs = [(512, 128), (512, 512), (1, 512)]
    config_labels = [f"p={p} / g={g}" for p, g in configs]
    series = []
    if has_q8:
        series.append(("Qwen2.5-1.5B Q8_0",   qwen_q8_df,    QWEN_Q8_COLOR))
    if has_q4:
        series.append(("Qwen2.5-1.5B Q4_K_M", qwen_q4_df, QWEN_Q4_COLOR))
    if has_b:
        series.append(("BitNet b1.58 2B4T",   local_df,   BITNET_COLOR))

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
    ax_cfg.legend(loc="upper right", fontsize=8)
    ax_cfg.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Inference Throughput: BitNet b1.58 2B4T & Qwen2.5 1.5B vs FP16 Baselines (CPU)",
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
    Throughput vs thread count for the three locally measured models at the
    (n_prompt=512, n_gen=128) reference config.

    Reads dedicated *_thread_sweep.csv files written by:
      make benchmark-threads-bitnet
      make benchmark-threads-qwen-q8
      make benchmark-threads-qwen-q4
    Skips silently when those files don't exist yet.
    """
    sweeps = []
    for name, path, color in [
        ("Qwen2.5-1.5B Q8_0",   Path("results/qwen_q8_thread_sweep.csv"),     QWEN_Q8_COLOR),
        ("Qwen2.5-1.5B Q4_K_M", Path("results/qwen_q4_thread_sweep.csv"),  QWEN_Q4_COLOR),
        ("BitNet b1.58 2B4T",   Path("results/bitnet_thread_sweep.csv"),   BITNET_COLOR),
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
    """Read the (512, 128) reference-config throughput for each of the three
    models from a given results subdirectory.  Returns None if no model CSV
    is readable under that subdir (so the caller can skip the arch entirely)."""
    out: dict[str, float] = {}
    for model, filename in [
        ("BitNet b1.58 2B4T",    "bitnet_step_metrics.csv"),
        ("Qwen2.5-1.5B Q8_0",    "qwen_q8_step_metrics.csv"),
        ("Qwen2.5-1.5B Q4_K_M",  "qwen_q4_step_metrics.csv"),
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
    each group.  Reads from results/{,linux_docker_x86,aws_c5_xlarge,
    aws_c6a_xlarge,aws_c7g_xlarge}/{bitnet,qwen_q8,qwen_q4}_step_metrics.csv
    per CROSS_ARCH_SOURCES.  Architectures with no readable CSVs are
    dropped silently so the plot degrades gracefully while the AWS
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

    models = ["BitNet b1.58 2B4T", "Qwen2.5-1.5B Q8_0", "Qwen2.5-1.5B Q4_K_M"]
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
    ax.set_xticklabels(models)
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(
        "Cross-architecture throughput at n_prompt=512, n_gen=128\n"
        "(higher is better; bars within each model = different CPU/OS combinations)"
    )
    ax.set_ylim(0, max_y * 1.18 if max_y else 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "cross_arch_throughput.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_memory(local_df: pd.DataFrame, out_dir: Path,
                qwen_q8_df: pd.DataFrame | None = None,
                qwen_q4_df: pd.DataFrame | None = None):
    if not _has_bench_data(local_df) and not _has_bench_data(qwen_q8_df) and not _has_bench_data(qwen_q4_df):
        print("Skipping memory plot: no benchmark CSVs (run 'make benchmark').")
        return
    labels, values, colors, hatches = _bar_series(local_df, qwen_q8_df, qwen_q4_df, "peak_rss_mb")

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.55)))
    max_val = max(v for v in values if v) or 1
    for i, (val, color, hatch) in enumerate(zip(values, colors, hatches)):
        ax.barh(i, val, color=color, hatch=hatch,
                edgecolor="#444444" if hatch else "#cccccc", linewidth=0.5)
        ax.text(val + max_val * 0.01, i, f"{val:.0f} MB", va="center", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Peak RSS (MB)")
    ax.set_title("Peak Memory: BitNet b1.58 2B4T & Qwen2.5 1.5B vs FP16 Baselines\n"
                 "(n_prompt=512, n_gen=128, CPU)")
    ax.set_xlim(0, max_val * 1.18)
    ax.invert_yaxis()
    ax.legend(handles=_legend_handles(qwen_q8_df, qwen_q4_df), loc="lower right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "memory_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_accuracy(local_acc: dict, out_dir: Path,
                  qwen_q8_acc: dict | None = None,
                  qwen_q4_acc: dict | None = None):
    tasks = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]
    task_labels = ["ARC-Easy", "ARC-Challenge", "WinoGrande", "HellaSwag", "MMLU"]
    task_colors = ["#4C72B0", "#55A868", "#8172B2", "#64B5CD", "#C44E52"]

    other_models = list(OTHER_BASELINES.keys())
    # Column order: other FP16 paper baselines → Qwen paper FP16 → Qwen Q8/Q4
    # ours → BitNet ours → BitNet paper.
    all_models = other_models + ["Qwen2.5 1.5B\n(paper FP16)"]
    if qwen_q8_acc is not None:
        all_models.append("Qwen2.5-1.5B\nQ8_0 (ours)")
    if qwen_q4_acc is not None:
        all_models.append("Qwen2.5-1.5B\nQ4_K_M (ours)")
    all_models += ["BitNet b1.58 2B4T\n(ours)", "BitNet b1.58 2B4T\n(paper)"]

    x = np.arange(len(all_models))
    n_tasks = len(tasks)
    width = 0.15
    offsets = np.linspace(-(n_tasks - 1) * width / 2, (n_tasks - 1) * width / 2, n_tasks)

    fig, ax = plt.subplots(figsize=(max(14, len(all_models) * 1.6), 6))
    for i, (task, label, color) in enumerate(zip(tasks, task_labels, task_colors)):
        vals = (
            [OTHER_BASELINES[m][task] for m in other_models]
            + [QWEN_PAPER[task]]
        )
        if qwen_q8_acc is not None:
            vals.append(qwen_q8_acc.get(task) or 0)
        if qwen_q4_acc is not None:
            vals.append(qwen_q4_acc.get(task) or 0)
        vals += [local_acc.get(task) or 0, BITNET_PAPER[task]]
        ax.bar(x + offsets[i], vals, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, ha="center")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Accuracy Comparison: BitNet b1.58 2B4T & Qwen2.5 1.5B vs FP16 Baselines\n"
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

    fig, ax = plt.subplots(figsize=(10, 6))

    QWEN_OURS_NAME    = "Qwen2.5-1.5B-Instruct Q8_0"
    QWEN_Q4_OURS_NAME = "Qwen2.5-1.5B-Instruct Q4_K_M"
    QWEN_PAPER_NAME   = "Qwen2.5 1.5B"
    BITNET_NAME       = "BitNet b1.58 2B4T"

    by_key = {
        (row["model"], row["source"]): (row[x_col], row["_y"])
        for _, row in plot_df.iterrows()
    }
    # Dotted connectors visualize the paper→ours or quant-chain delta.
    # For Qwen the chain is FP16 paper → Q8 ours → Q4 ours.
    for ours_key, from_key, color in [
        ((BITNET_NAME, "ours"),       (BITNET_NAME, "paper"),            BITNET_COLOR),
        ((QWEN_OURS_NAME, "ours"),    (QWEN_PAPER_NAME, "paper (FP16)"), QWEN_Q8_COLOR),
        ((QWEN_Q4_OURS_NAME, "ours"), (QWEN_OURS_NAME, "ours"),          QWEN_Q4_COLOR),
    ]:
        if ours_key in by_key and from_key in by_key:
            x0, y0 = by_key[from_key]
            x1, y1 = by_key[ours_key]
            ax.plot([x0, x1], [y0, y1], linestyle=":", color=color, alpha=0.55, zorder=2)

    for _, row in plot_df.iterrows():
        source, model = row["source"], row["model"]
        is_qwen_q8_ours    = source == "ours" and model == QWEN_OURS_NAME
        is_qwen_q4_ours = source == "ours" and model == QWEN_Q4_OURS_NAME
        is_ours = source == "ours"
        # Qwen FP16 paper collapses into the generic "FP16 baseline (paper)"
        # legend entry — same color, same marker, same label as the other
        # paper FP16 rows.
        if source == "paper (FP16)":
            color, label = OTHER_COLOR, "FP16 baseline (paper)"
        elif source == "paper":
            color, label = BITNET_COLOR, "BitNet b1.58 2B4T (paper)"
        elif is_qwen_q4_ours:
            color, label = QWEN_Q4_COLOR, "Qwen2.5-1.5B Q4_K_M (ours)"
        elif is_qwen_q8_ours:
            color, label = QWEN_Q8_COLOR, "Qwen2.5-1.5B Q8_0 (ours)"
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

        # Paper annotations use the full model name ("BitNet b1.58 2B4T (paper)",
        # "Qwen2.5 1.5B (paper)") to match the naming convention in the CSV and
        # other plots.  Our-measurement annotations stay abbreviated and keep the
        # quant suffix so Q8 and Q4 are distinguishable at a glance.
        if is_ours:
            ann = (model
                   .replace(" b1.58 2B4T", "")
                   .replace("Qwen2.5-1.5B-Instruct ", "Qwen ")) + " (ours)"
        else:
            ann = model.replace(" (FP16)", "") + " (paper)"
        # Stagger ours offsets so Q8 and Q4 don't overlap when their points sit close.
        if is_qwen_q4_ours:
            xy_offset = (8, -22)
        elif is_ours:
            xy_offset = (8, -10)
        else:
            xy_offset = (8, 6)
        ax.annotate(ann, (row[x_col], row["_y"]),
                    textcoords="offset points", xytext=xy_offset, fontsize=8)

    handles, labels_list = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels_list):
        if l and l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9, loc=legend_loc)

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
    """Cost vs accuracy: mean-of-5 plot plus one plot per task."""
    x_label = f"Cost per 1,000 tokens (USD, c5.xlarge @ ${hardware_rate:.3f}/hr)"
    _accuracy_scatter(
        df, out_dir,
        x_col="cost_per_1k_tokens", y_metric="mean_acc",
        x_label=x_label,
        y_label="Mean accuracy across 5 benchmarks (%)",
        title="Cost–Accuracy Trade-off: BitNet b1.58 2B4T & Qwen2.5 1.5B vs FP16 Baselines\n"
              "(mean of ARC-Easy, ARC-Challenge, WinoGrande, HellaSwag, MMLU;  "
              "hollow ○ = paper,  filled ♦ = ours)",
        filename="cost_accuracy.png",
    )
    for task, task_label in TASK_LABELS.items():
        _accuracy_scatter(
            df, out_dir,
            x_col="cost_per_1k_tokens", y_metric=task,
            x_label=x_label,
            y_label=f"{task_label} Accuracy (%)",
            title=f"Cost–{task_label} Trade-off: BitNet b1.58 2B4T & Qwen2.5 1.5B vs FP16 Baselines\n"
                  f"(hollow ○ = paper target,  filled ♦ = our measurement)",
            filename=f"{task}_cost_accuracy.png",
        )


def plot_energy_carbon(local_df: pd.DataFrame, qwen_q8_df: pd.DataFrame | None,
                       out_dir: Path, qwen_q4_df: pd.DataFrame | None = None,
                       electricity_rate: float = DEFAULT_ELECTRICITY_RATE):
    """
    Three-panel energy + carbon + dollar-cost footprint per 1,000 tokens
    at n_prompt=512, n_gen=128.

      Left:   Wh per 1k tokens     = energy_kwh × 1e6 / (n_prompt + n_gen)
      Middle: gCO₂ per 1k tokens   = co2_kg    × 1e6 / (n_prompt + n_gen)
      Right:  USD per 1k tokens    = energy_kwh × rate × 1000 / (n_prompt + n_gen)

    Carbon depends on the local grid's intensity (codecarbon resolves this from
    geolocation at run time); dollar cost depends on the local electricity rate.
    Both are properties of where the bench ran, not the model.

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
    bitnet_wh    = per_1k(local_df,   "energy_kwh", 1_000_000)
    qwen_q8_wh      = per_1k(qwen_q8_df,    "energy_kwh", 1_000_000)
    qwen_q4_wh   = per_1k(qwen_q4_df, "energy_kwh", 1_000_000)
    bitnet_gco2  = per_1k(local_df,   "co2_kg",     1_000_000)
    qwen_q8_gco2    = per_1k(qwen_q8_df,    "co2_kg",     1_000_000)
    qwen_q4_gco2 = per_1k(qwen_q4_df, "co2_kg",     1_000_000)
    bitnet_usd   = energy_cost_per_1k(local_df,   electricity_rate)
    qwen_q8_usd     = energy_cost_per_1k(qwen_q8_df,    electricity_rate)
    qwen_q4_usd  = energy_cost_per_1k(qwen_q4_df, electricity_rate)

    if bitnet_wh is None and qwen_q8_wh is None and qwen_q4_wh is None:
        print("Skipping energy/carbon plot: no energy_kwh data in benchmark CSVs.")
        return

    rows: list[tuple[str, str, float | None, float | None, float | None]] = []
    if qwen_q8_wh is not None:
        rows.append(("Qwen2.5-1.5B Q8_0 (ours)", QWEN_Q8_COLOR, qwen_q8_wh, qwen_q8_gco2, qwen_q8_usd))
    if qwen_q4_wh is not None:
        rows.append(("Qwen2.5-1.5B Q4_K_M (ours)", QWEN_Q4_COLOR, qwen_q4_wh, qwen_q4_gco2, qwen_q4_usd))
    if bitnet_wh is not None:
        rows.append(("BitNet b1.58 2B4T (ours)", BITNET_COLOR, bitnet_wh, bitnet_gco2, bitnet_usd))

    fig, (ax_e, ax_c, ax_d) = plt.subplots(
        1, 3, figsize=(17, max(3.5, len(rows) * 1.2)), sharey=True
    )

    def _draw_panel(ax, values: list[float | None], fmt: str):
        finite = [v for v in values if v is not None]
        max_val = max(finite) if finite else 1
        for i, val in enumerate(values):
            if val is None:
                ax.text(0, i, "  (no data)", va="center", fontsize=9, color="#888")
                continue
            ax.barh(i, val, color=rows[i][1], edgecolor="#444444", linewidth=0.5)
            ax.text(val + max_val * 0.02, i, fmt.format(val),
                    va="center", fontsize=10)
        ax.set_xlim(0, max_val * 1.3)

    ax_e.set_yticks(range(len(rows)))
    ax_e.set_yticklabels([r[0] for r in rows])
    ax_e.invert_yaxis()
    _draw_panel(ax_e, [r[2] for r in rows], "{:.2f} Wh")
    _draw_panel(ax_c, [r[3] for r in rows], "{:.3f} g")
    _draw_panel(ax_d, [r[4] for r in rows], "${:.5f}")

    ax_e.set_xlabel("Energy per 1,000 tokens (Wh)")
    ax_c.set_xlabel("Carbon emissions per 1,000 tokens (g CO₂)")
    ax_d.set_xlabel(f"Energy cost per 1,000 tokens (USD @ ${electricity_rate:.2f}/kWh)")
    ax_e.set_title("Energy")
    ax_c.set_title("Carbon")
    ax_d.set_title("Cost (electricity)")

    fig.suptitle(
        "Inference Energy, Carbon, and Cost per 1,000 tokens\n"
        "(BitNet vs Qwen on CPU at n_prompt=512, n_gen=128; "
        "CO₂ uses local grid intensity from CodeCarbon)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "energy_carbon_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cloud_cost_comparison(local_df: pd.DataFrame,
                               qwen_q8_df: pd.DataFrame | None,
                               qwen_q4_df: pd.DataFrame | None,
                               out_dir: Path,
                               hardware_rate: float,
                               electricity_rate: float):
    """
    Cost-per-1k-output-tokens across self-hosted (BitNet / Qwen) and cloud
    API services.  Each self-hosted model contributes two bars:
      - AWS c5.xlarge proxy (hatched) — what cloud rental would cost
      - local electricity (solid)     — marginal cost on hardware you own

    Cloud-API rows are taken from `CLOUD_API_PRICING` (output token rates).
    Log-x scale because prices span ~3 orders of magnitude.
    """
    entries: list[tuple[str, float, str, str]] = []  # (label, cost, color, hatch)
    for name, df, color in [
        ("Qwen2.5-1.5B Q8_0",   qwen_q8_df,    QWEN_Q8_COLOR),
        ("Qwen2.5-1.5B Q4_K_M", qwen_q4_df, QWEN_Q4_COLOR),
        ("BitNet b1.58 2B4T",   local_df,   BITNET_COLOR),
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

    fig, ax = plt.subplots(figsize=(12, max(5, len(entries) * 0.45)))
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
    have_bitnet = any(label.startswith("BitNet") for label, *_ in entries)
    have_qwen_q8   = any(label.startswith("Qwen2.5-1.5B Q8_0") for label, *_ in entries)
    have_q4     = any(label.startswith("Qwen2.5-1.5B Q4_K_M") for label, *_ in entries)
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
    ax.legend(handles=handles, loc="center right", fontsize=8)

    fig.tight_layout()
    path = out_dir / "cloud_cost_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    print(f"  Note: Cloud API prices hardcoded as of {CLOUD_API_PRICING_DATE}; "
          f"verify before publication (compare_runs.py:CLOUD_API_PRICING).")


def plot_memory_accuracy(df: pd.DataFrame, out_dir: Path):
    """
    Memory vs accuracy: mean-of-5 plot plus one plot per task.

    Lower-left is more efficient (less memory per accuracy point); BitNet b1.58
    typically sits in a corner the FP16 baselines can't reach.  Same hollow-vs-
    filled marker convention as plot_cost_accuracy.
    """
    _accuracy_scatter(
        df, out_dir,
        x_col="peak_rss_mb", y_metric="mean_acc",
        x_label="Peak RSS (MB) — lower is better",
        y_label="Mean accuracy across 5 benchmarks (%)",
        title="Memory–Accuracy: BitNet b1.58 2B4T & Qwen2.5 1.5B vs FP16 Baselines\n"
              "(mean of ARC-Easy, ARC-Challenge, WinoGrande, HellaSwag, MMLU;  "
              "hollow ○ = paper,  filled ♦ = ours)",
        filename="memory_accuracy.png",
        legend_loc="lower right",
    )
    for task, task_label in TASK_LABELS.items():
        _accuracy_scatter(
            df, out_dir,
            x_col="peak_rss_mb", y_metric=task,
            x_label="Peak RSS (MB) — lower is better",
            y_label=f"{task_label} Accuracy (%)",
            title=f"Memory–{task_label}: BitNet b1.58 2B4T & Qwen2.5 1.5B vs FP16 Baselines\n"
                  f"(hollow ○ = paper target,  filled ♦ = our measurement)",
            filename=f"{task}_memory_accuracy.png",
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
                                   qwen_q4_acc: dict | None = None):
    """
    Grouped-bar accuracy plot mirroring plot_accuracy(), but comparing our
    locally-measured models (BitNet, Qwen Q8/Q4) against cloud subscription
    APIs from CLOUD_API_PRICING.

    Cloud values come from CLOUD_API_ACCURACY.  Most providers only publish
    MMLU, so non-MMLU bars for cloud rows are typically empty (rendered as 0
    height — visually absent).  Cloud models appear in CLOUD_API_PRICING order
    (ascending output price).
    """
    tasks = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]
    task_labels = ["ARC-Easy", "ARC-Challenge", "WinoGrande", "HellaSwag", "MMLU"]
    task_colors = ["#4C72B0", "#55A868", "#8172B2", "#64B5CD", "#C44E52"]

    all_models: list[str] = []
    model_accs: dict[str, dict] = {}
    if qwen_q8_acc is not None:
        label = "Qwen2.5-1.5B\nQ8_0 (ours)"
        all_models.append(label); model_accs[label] = qwen_q8_acc
    if qwen_q4_acc is not None:
        label = "Qwen2.5-1.5B\nQ4_K_M (ours)"
        all_models.append(label); model_accs[label] = qwen_q4_acc
    label = "BitNet b1.58 2B4T\n(ours)"
    all_models.append(label); model_accs[label] = local_acc

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


def main():
    args = parse_args()
    local_df = load_local(Path(args.results))
    local_acc = load_accuracy(Path(args.accuracy))
    qwen_q8_df = load_qwen(Path(args.qwen_q8_results))
    qwen_q8_acc = load_accuracy(Path(args.qwen_q8_accuracy))
    qwen_q4_df = load_qwen(Path(args.qwen_q4_results))
    qwen_q4_acc = load_accuracy(Path(args.qwen_q4_accuracy))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df = build_comparison_df(
        local_df, local_acc, args.hardware_rate,
        qwen_q8_df, qwen_q8_acc,
        qwen_q4_df, qwen_q4_acc,
        electricity_rate=args.electricity_rate,
    )
    write_comparison_csv(comparison_df, Path(args.csv))

    plot_throughput(local_df, PLOTS_DIR, qwen_q8_df, qwen_q4_df)
    plot_thread_scaling(PLOTS_DIR)
    plot_cross_arch_throughput(PLOTS_DIR)
    plot_memory(local_df, PLOTS_DIR, qwen_q8_df, qwen_q4_df)
    plot_accuracy(local_acc, PLOTS_DIR, qwen_q8_acc, qwen_q4_acc)
    plot_cost_accuracy(comparison_df, PLOTS_DIR, args.hardware_rate)
    plot_energy_carbon(local_df, qwen_q8_df, PLOTS_DIR, qwen_q4_df,
                       electricity_rate=args.electricity_rate)
    plot_cloud_cost_comparison(local_df, qwen_q8_df, qwen_q4_df, PLOTS_DIR,
                                args.hardware_rate, args.electricity_rate)
    plot_memory_accuracy(comparison_df, PLOTS_DIR)
    plot_cloud_accuracy_comparison(local_acc, PLOTS_DIR, qwen_q8_acc, qwen_q4_acc)
    plot_cloud_cost_accuracy(comparison_df, PLOTS_DIR)

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
