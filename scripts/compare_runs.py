"""
Generates comparison plots and a comparison CSV for Phase 4 of the capstone.

Reads local benchmark CSVs and accuracy JSONs, overlays published FP16 baseline
numbers from arXiv:2504.12285 Table 1, and saves plots to results/plots/ plus a
summary CSV to --csv (default: results/comparison_table.csv).

  BitNet:  results/bitnet_step_metrics.csv  +  results/accuracy_results_bitnet.json
  Qwen:    results/qwen_step_metrics.csv    +  results/accuracy_results_qwen.json  (optional)

Hardware rate default: AWS c5.xlarge on-demand, us-east-1 ($0.170/hr, 4 vCPUs).
On-demand pricing is used rather than spot for reproducibility — spot prices
change hourly and would make cost numbers non-comparable across runs.
Override with --hardware-rate if running on different hardware.

Usage:
    python scripts/compare_runs.py
        [--results results/bitnet_step_metrics.csv]
        [--accuracy results/accuracy_results_bitnet.json]
        [--qwen-results results/qwen_step_metrics.csv]
        [--qwen-accuracy results/accuracy_results_qwen.json]
        [--csv results/comparison_table.csv]
        [--hardware-rate 0.170]
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
DEFAULT_QWEN_CSV = Path(__file__).parent.parent / "results" / "qwen_step_metrics.csv"
DEFAULT_QWEN_ACCURACY_JSON = Path(__file__).parent.parent / "results" / "accuracy_results_qwen.json"

# AWS c5.xlarge on-demand, us-east-1 (4 vCPUs — matches 4-thread benchmark condition)
# Source: https://instances.vantage.sh/aws/ec2/c5.xlarge (retrieved 2026-05-08)
DEFAULT_HARDWARE_RATE = 0.170

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
#   Other paper baselines  → blue
#   BitNet                 → orange (paper hatched, ours solid)
#   Qwen                   → green  (paper hatched, ours solid)
OTHER_COLOR  = "#4C72B0"
BITNET_COLOR = "#DD8452"
QWEN_COLOR   = "#55A868"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(DEFAULT_CSV))
    p.add_argument("--accuracy", default=str(DEFAULT_ACCURACY_JSON))
    p.add_argument("--qwen-results", default=str(DEFAULT_QWEN_CSV))
    p.add_argument("--qwen-accuracy", default=str(DEFAULT_QWEN_ACCURACY_JSON))
    p.add_argument("--csv", default=str(DEFAULT_COMPARISON_CSV))
    p.add_argument(
        "--hardware-rate",
        type=float,
        default=DEFAULT_HARDWARE_RATE,
        metavar="$/HR",
        help="On-demand $/hr for the target instance (default: %(default)s — AWS c5.xlarge us-east-1)",
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


def build_comparison_df(
    local_df: pd.DataFrame,
    local_acc: dict,
    hardware_rate: float,
    qwen_df: pd.DataFrame | None = None,
    qwen_acc: dict | None = None,
) -> pd.DataFrame:
    ACC_FIELDS = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]
    rows = []

    for name, b in OTHER_BASELINES.items():
        rows.append({
            "model": name,
            "source": "paper (FP16)",
            "throughput_tokens_s": b["throughput_tokens_s"],
            "peak_rss_mb": b["peak_rss_mb"],
            "cost_per_1k_tokens": round(cost_per_1k(b["throughput_tokens_s"], hardware_rate), 6),
            **{f: b[f] for f in ACC_FIELDS},
        })

    rows.append({
        "model": "BitNet b1.58 2B4T",
        "source": "paper",
        "throughput_tokens_s": BITNET_PAPER["throughput_tokens_s"],
        "peak_rss_mb": BITNET_PAPER["peak_rss_mb"],
        "cost_per_1k_tokens": round(cost_per_1k(BITNET_PAPER["throughput_tokens_s"], hardware_rate), 6),
        **{f: BITNET_PAPER[f] for f in ACC_FIELDS},
    })

    bitnet_tps, bitnet_rss = _bench_row(local_df)
    our_cost = round(cost_per_1k(bitnet_tps, hardware_rate), 6) if bitnet_tps else ""
    rows.append({
        "model": "BitNet b1.58 2B4T",
        "source": "ours",
        "throughput_tokens_s": round(bitnet_tps, 2) if bitnet_tps is not None else "",
        "peak_rss_mb": round(bitnet_rss, 0) if bitnet_rss is not None else "",
        "cost_per_1k_tokens": our_cost,
        **{f: (round(local_acc[f], 2) if local_acc.get(f) is not None else "") for f in ACC_FIELDS},
    })

    rows.append({
        "model": "Qwen2.5 1.5B",
        "source": "paper (FP16)",
        "throughput_tokens_s": QWEN_PAPER["throughput_tokens_s"],
        "peak_rss_mb": QWEN_PAPER["peak_rss_mb"],
        "cost_per_1k_tokens": round(cost_per_1k(QWEN_PAPER["throughput_tokens_s"], hardware_rate), 6),
        **{f: QWEN_PAPER[f] for f in ACC_FIELDS},
    })

    q_acc = qwen_acc or {}
    if qwen_df is not None or any(q_acc.get(f) is not None for f in ACC_FIELDS):
        q_tps, q_rss = _bench_row(qwen_df) if qwen_df is not None else (None, None)
        q_cost = round(cost_per_1k(q_tps, hardware_rate), 6) if q_tps else ""
        rows.append({
            "model": "Qwen2.5-1.5B-Instruct Q8_0",
            "source": "ours",
            "throughput_tokens_s": round(q_tps, 2) if q_tps is not None else "",
            "peak_rss_mb": round(q_rss, 0) if q_rss is not None else "",
            "cost_per_1k_tokens": q_cost,
            **{f: (round(q_acc[f], 2) if q_acc.get(f) is not None else "") for f in ACC_FIELDS},
        })

    return pd.DataFrame(rows)


def write_comparison_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def _bar_series(local_df: pd.DataFrame, qwen_df: pd.DataFrame | None,
                metric: str) -> tuple[list[str], list[float], list[str], list[str]]:
    """
    Build the (labels, values, colors, hatches) tuple for a horizontal bar chart.

    Order: other paper baselines → BitNet (paper, ours) → Qwen (paper, ours-if-available).
    `metric` selects which field to read from each paper dict / which bench column.
    """
    metric_col = "throughput_tokens_s" if metric == "throughput_tokens_s" else "peak_rss_mb"
    bitnet_tps, bitnet_rss = _bench_row(local_df)
    bitnet_local = bitnet_tps if metric_col == "throughput_tokens_s" else bitnet_rss

    labels = list(OTHER_BASELINES.keys()) + ["BitNet b1.58 2B4T (paper)"]
    values = [OTHER_BASELINES[m][metric_col] for m in OTHER_BASELINES] + [BITNET_PAPER[metric_col]]
    colors  = [OTHER_COLOR] * len(OTHER_BASELINES) + [BITNET_COLOR]
    hatches = [""]          * len(OTHER_BASELINES) + ["///"]

    if bitnet_local is not None:
        labels.append("BitNet b1.58 2B4T (ours)")
        values.append(bitnet_local)
        colors.append(BITNET_COLOR)
        hatches.append("")

    labels.append("Qwen2.5 1.5B (paper FP16)")
    values.append(QWEN_PAPER[metric_col])
    colors.append(QWEN_COLOR)
    hatches.append("///")

    q_tps, q_rss = _bench_row(qwen_df) if qwen_df is not None else (None, None)
    q_val = q_tps if metric_col == "throughput_tokens_s" else q_rss
    if q_val is not None:
        labels.append("Qwen2.5-1.5B-Instruct Q8_0 (ours)")
        values.append(q_val)
        colors.append(QWEN_COLOR)
        hatches.append("")

    return labels, values, colors, hatches


def _legend_handles(qwen_df: pd.DataFrame | None):
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=OTHER_COLOR,  edgecolor="#cccccc", label="Other FP16 baseline (paper)"),
        Patch(facecolor=BITNET_COLOR, hatch="///", edgecolor="#444444", label="BitNet b1.58 2B4T (paper)"),
        Patch(facecolor=BITNET_COLOR, edgecolor="#cccccc", label="BitNet b1.58 2B4T (ours)"),
        Patch(facecolor=QWEN_COLOR,   hatch="///", edgecolor="#444444", label="Qwen2.5 1.5B (paper FP16)"),
    ]
    if qwen_df is not None:
        handles.append(Patch(facecolor=QWEN_COLOR, edgecolor="#cccccc", label="Qwen2.5-1.5B Q8_0 (ours)"))
    return handles


def plot_throughput(local_df: pd.DataFrame, out_dir: Path, qwen_df: pd.DataFrame | None = None):
    if not _has_bench_data(local_df) and not _has_bench_data(qwen_df):
        print("Skipping throughput plot: no benchmark CSVs (run 'make benchmark').")
        return
    labels, values, colors, hatches = _bar_series(local_df, qwen_df, "throughput_tokens_s")

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.55)))
    max_val = max(v for v in values if v) or 1
    for i, (val, color, hatch) in enumerate(zip(values, colors, hatches)):
        ax.barh(i, val, color=color, hatch=hatch,
                edgecolor="#444444" if hatch else "#cccccc", linewidth=0.5)
        ax.text(val + max_val * 0.01, i, f"{val:.1f}", va="center", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Throughput (tokens/s)")
    ax.set_title("Inference Throughput: BitNet b1.58 2B4T & Qwen2.5 1.5B vs FP16 Baselines\n"
                 "(n_prompt=512, n_gen=128, CPU)")
    ax.set_xlim(0, max_val * 1.15)
    ax.invert_yaxis()
    ax.legend(handles=_legend_handles(qwen_df), loc="lower right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "throughput_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_memory(local_df: pd.DataFrame, out_dir: Path, qwen_df: pd.DataFrame | None = None):
    if not _has_bench_data(local_df) and not _has_bench_data(qwen_df):
        print("Skipping memory plot: no benchmark CSVs (run 'make benchmark').")
        return
    labels, values, colors, hatches = _bar_series(local_df, qwen_df, "peak_rss_mb")

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
    ax.legend(handles=_legend_handles(qwen_df), loc="lower right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "memory_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_throughput_by_config(local_df: pd.DataFrame, out_dir: Path):
    if local_df is None or local_df.empty:
        print("Skipping BitNet throughput-by-config plot: no benchmark CSV.")
        return
    configs = local_df.groupby(["n_prompt", "n_gen"])["throughput_tokens_s"].median().reset_index()
    labels = [f"p={int(row.n_prompt)} / g={int(row.n_gen)}" for _, row in configs.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, configs["throughput_tokens_s"], color="#DD8452")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("BitNet b1.58 2B4T Throughput by Prompt/Generation Length")
    fig.tight_layout()
    path = out_dir / "bitnet_throughput_configs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_accuracy(local_acc: dict, out_dir: Path, qwen_acc: dict | None = None):
    tasks = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]
    task_labels = ["ARC-Easy", "ARC-Challenge", "WinoGrande", "HellaSwag", "MMLU"]
    task_colors = ["#4C72B0", "#55A868", "#8172B2", "#64B5CD", "#C44E52"]

    other_models = list(OTHER_BASELINES.keys())
    # Column order: other paper baselines, BitNet (paper, ours), Qwen (paper, ours-if-available)
    all_models = other_models + [
        "BitNet 2B4T\n(paper)", "BitNet 2B4T\n(ours)",
        "Qwen2.5 1.5B\n(paper FP16)",
    ]
    if qwen_acc is not None:
        all_models.append("Qwen2.5-1.5B\nQ8_0 (ours)")

    x = np.arange(len(all_models))
    n_tasks = len(tasks)
    width = 0.15
    offsets = np.linspace(-(n_tasks - 1) * width / 2, (n_tasks - 1) * width / 2, n_tasks)

    fig, ax = plt.subplots(figsize=(max(14, len(all_models) * 1.6), 6))
    for i, (task, label, color) in enumerate(zip(tasks, task_labels, task_colors)):
        vals = (
            [OTHER_BASELINES[m][task] for m in other_models]
            + [BITNET_PAPER[task], local_acc.get(task) or 0, QWEN_PAPER[task]]
        )
        if qwen_acc is not None:
            vals.append(qwen_acc.get(task) or 0)
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


def plot_cost_accuracy(df: pd.DataFrame, out_dir: Path, hardware_rate: float):
    # Use MMLU as the accuracy axis — most general benchmark, reported by all models.
    # Hollow circles = paper-reported numbers; filled diamonds = our local measurements.
    # A dotted connector between each model's (paper, ours) pair makes the
    # paper-to-measurement delta visible at a glance.
    plot_df = df[df["cost_per_1k_tokens"].notna() & (df["cost_per_1k_tokens"] != "")].copy()
    plot_df = plot_df[plot_df["mmlu"].notna() & (plot_df["mmlu"] != "")].copy()
    plot_df["cost_per_1k_tokens"] = pd.to_numeric(plot_df["cost_per_1k_tokens"])
    plot_df["mmlu"] = pd.to_numeric(plot_df["mmlu"])

    fig, ax = plt.subplots(figsize=(10, 6))

    QWEN_OURS_NAME  = "Qwen2.5-1.5B-Instruct Q8_0"
    QWEN_PAPER_NAME = "Qwen2.5 1.5B"
    BITNET_NAME     = "BitNet b1.58 2B4T"

    by_key = {
        (row["model"], row["source"]): (row["cost_per_1k_tokens"], row["mmlu"])
        for _, row in plot_df.iterrows()
    }
    for ours_key, paper_key, color in [
        ((BITNET_NAME, "ours"),    (BITNET_NAME, "paper"),           BITNET_COLOR),
        ((QWEN_OURS_NAME, "ours"), (QWEN_PAPER_NAME, "paper (FP16)"), QWEN_COLOR),
    ]:
        if ours_key in by_key and paper_key in by_key:
            x0, y0 = by_key[paper_key]
            x1, y1 = by_key[ours_key]
            ax.plot([x0, x1], [y0, y1], linestyle=":", color=color, alpha=0.55, zorder=2)

    for _, row in plot_df.iterrows():
        source, model = row["source"], row["model"]
        is_qwen_ours  = source == "ours" and model == QWEN_OURS_NAME
        is_qwen_paper = source == "paper (FP16)" and model == QWEN_PAPER_NAME
        is_ours = source == "ours"
        if is_qwen_paper:
            color, label = QWEN_COLOR, "Qwen2.5 1.5B (paper FP16)"
        elif source == "paper (FP16)":
            color, label = OTHER_COLOR, "Other FP16 baseline (paper)"
        elif source == "paper":
            color, label = BITNET_COLOR, "BitNet b1.58 2B4T (paper)"
        elif is_qwen_ours:
            color, label = QWEN_COLOR, "Qwen2.5-1.5B Q8_0 (ours, 0-shot MMLU)"
        else:
            color, label = BITNET_COLOR, "BitNet b1.58 2B4T (ours, 0-shot MMLU)"

        if is_ours:
            ax.scatter(row["cost_per_1k_tokens"], row["mmlu"],
                       facecolors=color, edgecolors=color, marker="D",
                       s=160, linewidths=1.5, label=label, zorder=3)
        else:
            ax.scatter(row["cost_per_1k_tokens"], row["mmlu"],
                       facecolors="white", edgecolors=color, marker="o",
                       s=110, linewidths=2, label=label, zorder=3)

        ann_label = model.replace(" b1.58 2B4T", "").replace(" (FP16)", "").replace(" Q8_0", "")
        ann_label += " (ours)" if is_ours else " (paper)"
        # Offset paper labels above-right and ours labels below-right so each
        # model's paper/ours pair separates even when the points sit nearly
        # on top of each other (e.g. BitNet ours ≈ BitNet paper on MMLU/cost).
        xy_offset = (8, -10) if is_ours else (8, 6)
        ax.annotate(
            ann_label,
            (row["cost_per_1k_tokens"], row["mmlu"]),
            textcoords="offset points", xytext=xy_offset, fontsize=8,
        )

    # De-duplicate legend entries
    handles, labels_list = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels_list):
        if l and l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9)

    ax.set_xlabel(f"Cost per 1,000 tokens (USD, c5.xlarge @ ${hardware_rate:.3f}/hr)")
    ax.set_ylabel("MMLU Accuracy (%)")
    ax.set_title("Cost–Accuracy Trade-off: BitNet b1.58 2B4T & Qwen2.5 1.5B vs FP16 Baselines\n"
                 "(hollow ○ = paper target,  filled ♦ = our measurement)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "cost_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_energy_carbon(local_df: pd.DataFrame, qwen_df: pd.DataFrame | None, out_dir: Path):
    """
    Two-panel energy + carbon footprint per 1,000 tokens at n_prompt=512, n_gen=128.

      Left  panel: Wh per 1k tokens   = energy_kwh × 1e6 / (n_prompt + n_gen)
      Right panel: gCO₂ per 1k tokens = co2_kg    × 1e6 / (n_prompt + n_gen)

    Carbon depends on the local grid's intensity (codecarbon resolves this from
    geolocation at run time), so the per-region figure noted in the title is
    not a property of the model — it's a property of where the bench ran.

    FP16 baselines aren't shown — the paper doesn't report energy/CO₂.  Skips
    silently if neither metrics CSV has populated energy_kwh.
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
    bitnet_wh   = per_1k(local_df, "energy_kwh", 1_000_000)
    qwen_wh     = per_1k(qwen_df,  "energy_kwh", 1_000_000)
    bitnet_gco2 = per_1k(local_df, "co2_kg",     1_000_000)
    qwen_gco2   = per_1k(qwen_df,  "co2_kg",     1_000_000)

    if bitnet_wh is None and qwen_wh is None:
        print("Skipping energy/carbon plot: no energy_kwh data in benchmark CSVs.")
        return

    rows: list[tuple[str, str, float | None, float | None]] = []
    if bitnet_wh is not None:
        rows.append(("BitNet b1.58 2B4T (ours)", BITNET_COLOR, bitnet_wh, bitnet_gco2))
    if qwen_wh is not None:
        rows.append(("Qwen2.5-1.5B Q8_0 (ours)", QWEN_COLOR, qwen_wh, qwen_gco2))

    fig, (ax_e, ax_c) = plt.subplots(
        1, 2, figsize=(13, max(3.5, len(rows) * 1.2)), sharey=True
    )

    def _draw_panel(ax, values: list[float | None], unit: str):
        finite = [v for v in values if v is not None]
        max_val = max(finite) if finite else 1
        for i, val in enumerate(values):
            if val is None:
                ax.text(0, i, "  (no data)", va="center", fontsize=9, color="#888")
                continue
            ax.barh(i, val, color=rows[i][1], edgecolor="#444444", linewidth=0.5)
            ax.text(val + max_val * 0.02, i, f"{val:.2f} {unit}",
                    va="center", fontsize=10)
        ax.set_xlim(0, max_val * 1.25)

    ax_e.set_yticks(range(len(rows)))
    ax_e.set_yticklabels([r[0] for r in rows])
    ax_e.invert_yaxis()
    _draw_panel(ax_e, [r[2] for r in rows], "Wh")
    _draw_panel(ax_c, [r[3] for r in rows], "g")

    ax_e.set_xlabel("Energy per 1,000 tokens (Wh)")
    ax_c.set_xlabel("Carbon emissions per 1,000 tokens (g CO₂)")
    ax_e.set_title("Energy")
    ax_c.set_title("Carbon")

    fig.suptitle(
        "Inference Energy & Carbon Footprint per 1,000 tokens\n"
        "(BitNet vs Qwen on CPU at n_prompt=512, n_gen=128; CO₂ uses local grid intensity from CodeCarbon)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "energy_carbon_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_memory_accuracy(comparison_df: pd.DataFrame, out_dir: Path):
    """
    Memory–accuracy Pareto frontier: peak_rss_mb on x, mean-of-5-task accuracy on y.

    Lower-left is more efficient (less memory per accuracy point); BitNet b1.58
    typically sits in a corner the FP16 baselines can't reach.  Same hollow-vs-
    filled marker convention as plot_cost_accuracy.
    """
    ACC_FIELDS = ["arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu"]
    df = comparison_df.copy()
    df = df[df["peak_rss_mb"].notna() & (df["peak_rss_mb"] != "")].copy()
    df["peak_rss_mb"] = pd.to_numeric(df["peak_rss_mb"])

    def mean_acc(row) -> float | None:
        vals = []
        for f in ACC_FIELDS:
            v = row[f]
            if v == "" or pd.isna(v):
                return None
            vals.append(float(v))
        return sum(vals) / len(vals)

    df["mean_acc"] = df.apply(mean_acc, axis=1)
    df = df[df["mean_acc"].notna()]
    if df.empty:
        print("Skipping memory–accuracy plot: no rows with full accuracy + memory data.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    QWEN_OURS_NAME  = "Qwen2.5-1.5B-Instruct Q8_0"
    QWEN_PAPER_NAME = "Qwen2.5 1.5B"
    BITNET_NAME     = "BitNet b1.58 2B4T"

    by_key = {
        (row["model"], row["source"]): (row["peak_rss_mb"], row["mean_acc"])
        for _, row in df.iterrows()
    }
    for ours_key, paper_key, color in [
        ((BITNET_NAME, "ours"),    (BITNET_NAME, "paper"),           BITNET_COLOR),
        ((QWEN_OURS_NAME, "ours"), (QWEN_PAPER_NAME, "paper (FP16)"), QWEN_COLOR),
    ]:
        if ours_key in by_key and paper_key in by_key:
            x0, y0 = by_key[paper_key]
            x1, y1 = by_key[ours_key]
            ax.plot([x0, x1], [y0, y1], linestyle=":", color=color, alpha=0.55, zorder=2)

    for _, row in df.iterrows():
        source, model = row["source"], row["model"]
        is_qwen_ours  = source == "ours" and model == QWEN_OURS_NAME
        is_qwen_paper = source == "paper (FP16)" and model == QWEN_PAPER_NAME
        is_ours = source == "ours"
        if is_qwen_paper:
            color, label = QWEN_COLOR, "Qwen2.5 1.5B (paper FP16)"
        elif source == "paper (FP16)":
            color, label = OTHER_COLOR, "Other FP16 baseline (paper)"
        elif source == "paper":
            color, label = BITNET_COLOR, "BitNet b1.58 2B4T (paper)"
        elif is_qwen_ours:
            color, label = QWEN_COLOR, "Qwen2.5-1.5B Q8_0 (ours)"
        else:
            color, label = BITNET_COLOR, "BitNet b1.58 2B4T (ours)"

        if is_ours:
            ax.scatter(row["peak_rss_mb"], row["mean_acc"],
                       facecolors=color, edgecolors=color, marker="D",
                       s=160, linewidths=1.5, label=label, zorder=3)
        else:
            ax.scatter(row["peak_rss_mb"], row["mean_acc"],
                       facecolors="white", edgecolors=color, marker="o",
                       s=110, linewidths=2, label=label, zorder=3)

        ann = model.replace(" b1.58 2B4T", "").replace(" (FP16)", "").replace(" Q8_0", "")
        ann += " (ours)" if is_ours else " (paper)"
        xy_offset = (8, -10) if is_ours else (8, 6)
        ax.annotate(ann, (row["peak_rss_mb"], row["mean_acc"]),
                    textcoords="offset points", xytext=xy_offset, fontsize=8)

    handles, labels_list = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels_list):
        if l and l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9, loc="lower right")

    ax.set_xlabel("Peak RSS (MB) — lower is better")
    ax.set_ylabel("Mean accuracy across 5 benchmarks (%)")
    ax.set_title("Memory–Accuracy Pareto: BitNet b1.58 2B4T & Qwen2.5 1.5B vs FP16 Baselines\n"
                 "(hollow ○ = paper target,  filled ♦ = our measurement)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "memory_accuracy_pareto.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = parse_args()
    local_df = load_local(Path(args.results))
    local_acc = load_accuracy(Path(args.accuracy))
    qwen_df = load_qwen(Path(args.qwen_results))
    qwen_acc = load_accuracy(Path(args.qwen_accuracy))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df = build_comparison_df(local_df, local_acc, args.hardware_rate, qwen_df, qwen_acc)
    write_comparison_csv(comparison_df, Path(args.csv))

    plot_throughput(local_df, PLOTS_DIR, qwen_df)
    plot_memory(local_df, PLOTS_DIR, qwen_df)
    plot_throughput_by_config(local_df, PLOTS_DIR)
    plot_accuracy(local_acc, PLOTS_DIR, qwen_acc)
    plot_cost_accuracy(comparison_df, PLOTS_DIR, args.hardware_rate)
    plot_energy_carbon(local_df, qwen_df, PLOTS_DIR)
    plot_memory_accuracy(comparison_df, PLOTS_DIR)

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
