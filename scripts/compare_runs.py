"""
Generate latency, memory, throughput, and accuracy comparison plots.

Reads results/step_metrics.csv   — local BitNet benchmark results
Reads results/comparison_table.csv — published FP16 baseline accuracy figures

Produces PNG plots in results/plots/.

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --metrics results/step_metrics.csv \
                                   --comparison results/comparison_table.csv \
                                   --plots-dir results/plots
"""

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as e:
    import sys
    print(f"Missing dependency: {e}\nInstall: pip install matplotlib pandas", file=sys.stderr)
    sys.exit(1)

RESULTS_DIR = Path(__file__).parent.parent / "results"
METRICS_CSV = RESULTS_DIR / "step_metrics.csv"
COMPARISON_CSV = RESULTS_DIR / "comparison_table.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

ACCURACY_BENCHMARKS = ["mmlu", "arc_challenge", "hellaswag", "winogrande", "truthfulqa_mc2"]

BITNET_COLOR = "#00897b"   # teal — highlight BitNet
BASELINE_COLOR = "#5c85d6"  # blue  — FP16 baselines


# ── helpers ──────────────────────────────────────────────────────────────────

def _label_bars(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    for bar in ax.patches:
        h = bar.get_height()
        if h and h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                fmt.format(h),
                ha="center", va="bottom", fontsize=7.5,
            )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(RESULTS_DIR.parent)}")


# ── efficiency plots (from step_metrics.csv) ─────────────────────────────────

def load_local_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = ["tokens_per_sec", "eval_ms_per_token", "peak_memory_mb", "energy_kwh", "load_time_ms"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Aggregate repeated runs with median
    agg_cols = [c for c in numeric if c in df.columns]
    return df.groupby("model")[agg_cols].median().reset_index()


def plot_throughput(df: pd.DataFrame, plots_dir: Path) -> None:
    col = "tokens_per_sec"
    data = df[["model", col]].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.2), 4))
    ax.bar(data["model"], data[col], color=BITNET_COLOR, edgecolor="black", linewidth=0.4)
    _label_bars(ax)
    ax.set_title("Inference Throughput — BitNet b1.58 2B4T", fontweight="bold")
    ax.set_ylabel("Tokens / second  (higher is better)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, plots_dir / "throughput.png")


def plot_latency(df: pd.DataFrame, plots_dir: Path) -> None:
    col = "eval_ms_per_token"
    data = df[["model", col]].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.2), 4))
    ax.bar(data["model"], data[col], color="#e67e22", edgecolor="black", linewidth=0.4)
    _label_bars(ax)
    ax.set_title("Decode Latency — BitNet b1.58 2B4T", fontweight="bold")
    ax.set_ylabel("ms / token  (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, plots_dir / "latency.png")


def plot_memory(df: pd.DataFrame, plots_dir: Path) -> None:
    col = "peak_memory_mb"
    data = df[["model", col]].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.2), 4))
    ax.bar(data["model"], data[col], color="#e74c3c", edgecolor="black", linewidth=0.4)
    _label_bars(ax)
    ax.set_title("Peak Memory Usage — BitNet b1.58 2B4T", fontweight="bold")
    ax.set_ylabel("MB  (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, plots_dir / "memory.png")


def plot_energy(df: pd.DataFrame, plots_dir: Path) -> None:
    col = "energy_kwh"
    data = df[["model", col]].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.2), 4))
    ax.bar(data["model"], data[col], color="#8e44ad", edgecolor="black", linewidth=0.4)
    _label_bars(ax, fmt="{:.6f}")
    ax.set_title("Energy Consumption — BitNet b1.58 2B4T", fontweight="bold")
    ax.set_ylabel("kWh per run  (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, plots_dir / "energy.png")


# ── accuracy plots (from comparison_table.csv) ────────────────────────────────

def load_comparison(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ACCURACY_BENCHMARKS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_accuracy_per_benchmark(df: pd.DataFrame, plots_dir: Path) -> None:
    available = [b for b in ACCURACY_BENCHMARKS if b in df.columns]
    if not available:
        print("  No accuracy columns found — skipping per-benchmark accuracy plots.")
        return

    for bench in available:
        data = df[["model", bench, "type"]].dropna(subset=[bench])
        if data.empty:
            continue
        colors = [BITNET_COLOR if t == "bitnet" else BASELINE_COLOR for t in data["type"]]
        fig, ax = plt.subplots(figsize=(max(7, len(data) * 1.3), 4))
        bars = ax.bar(data["model"], data[bench], color=colors, edgecolor="black", linewidth=0.4)
        _label_bars(ax)
        ax.set_title(f"{bench.upper()} Accuracy", fontweight="bold")
        ax.set_ylabel("Accuracy %  (higher is better)")
        ax.tick_params(axis="x", rotation=25)
        # legend
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor=BITNET_COLOR, label="BitNet b1.58 2B4T"),
                Patch(facecolor=BASELINE_COLOR, label="FP16 baseline"),
            ],
            fontsize=8,
        )
        _save(fig, plots_dir / f"accuracy_{bench}.png")


def plot_accuracy_combined(df: pd.DataFrame, plots_dir: Path) -> None:
    available = [b for b in ACCURACY_BENCHMARKS if b in df.columns]
    if len(available) < 2:
        return

    n_models = len(df)
    n_bench = len(available)
    width = 0.8 / n_bench
    x = list(range(n_models))

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 5))
    cmap = plt.get_cmap("tab10")

    for i, bench in enumerate(available):
        offsets = [xi + i * width for xi in x]
        vals = df[bench].fillna(0).tolist()
        ax.bar(offsets, vals, width=width, label=bench.upper(),
               color=cmap(i), edgecolor="black", linewidth=0.3)

    center = width * (n_bench - 1) / 2
    ax.set_xticks([xi + center for xi in x])
    ax.set_xticklabels(df["model"], rotation=25, ha="right")
    ax.set_ylabel("Accuracy %")
    ax.set_title("Accuracy Across Benchmarks — BitNet vs FP16 Baselines", fontweight="bold")
    ax.legend(fontsize=8, ncol=len(available))

    # Vertical separator between BitNet and baselines
    bitnet_rows = (df["type"] == "bitnet").sum() if "type" in df.columns else 0
    if 0 < bitnet_rows < n_models:
        ax.axvline(x=bitnet_rows - 0.5 + center, color="gray", linestyle="--", linewidth=0.8)

    _save(fig, plots_dir / "accuracy_combined.png")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark comparison plots.")
    parser.add_argument("--metrics", default=str(METRICS_CSV))
    parser.add_argument("--comparison", default=str(COMPARISON_CSV))
    parser.add_argument("--plots-dir", default=str(PLOTS_DIR))
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    comparison_path = Path(args.comparison)
    plots_dir = Path(args.plots_dir)

    print("Generating plots...")

    if metrics_path.exists():
        metrics_df = load_local_metrics(metrics_path)
        if metrics_df.empty:
            print(f"  {metrics_path} is empty — run metrics_tracker.py first.")
        else:
            plot_throughput(metrics_df, plots_dir)
            plot_latency(metrics_df, plots_dir)
            plot_memory(metrics_df, plots_dir)
            plot_energy(metrics_df, plots_dir)
    else:
        print(f"  {metrics_path} not found — skipping efficiency plots.")

    if comparison_path.exists():
        comparison_df = load_comparison(comparison_path)
        plot_accuracy_per_benchmark(comparison_df, plots_dir)
        plot_accuracy_combined(comparison_df, plots_dir)
    else:
        print(f"  {comparison_path} not found — skipping accuracy plots.")

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
