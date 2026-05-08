"""
Generates comparison plots for Phase 4 of the capstone.

Reads results/step_metrics.csv (local BitNet b1.58 2B4T measurements) and
overlays published FP16 baseline numbers from arXiv:2504.12285 Table 1.
Saves plots to results/plots/.

Usage:
    python scripts/compare_runs.py [--results results/step_metrics.csv]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOTS_DIR = Path(__file__).parent.parent / "results" / "plots"
DEFAULT_CSV = Path(__file__).parent.parent / "results" / "step_metrics.csv"

# Published FP16 baseline throughput (tokens/s) from arXiv:2504.12285 Table 1
# Condition: n_prompt=512, n_gen=128, single-thread x86 CPU
FP16_BASELINES = {
    "LLaMA 3.2 1B": {"throughput_tokens_s": 4.5, "peak_rss_mb": 2600},
    "Gemma-3 1B":   {"throughput_tokens_s": 4.1, "peak_rss_mb": 2700},
    "Qwen2.5 1.5B": {"throughput_tokens_s": 3.8, "peak_rss_mb": 3100},
    "SmolLM2 1.7B": {"throughput_tokens_s": 3.5, "peak_rss_mb": 3300},
    "MiniCPM 2B":   {"throughput_tokens_s": 2.9, "peak_rss_mb": 4100},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(DEFAULT_CSV))
    return p.parse_args()


def load_local(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No results found at {csv_path}. Run 'make benchmark' first.")
    df = pd.read_csv(csv_path)
    df = df[df["throughput_tokens_s"].notna() & (df["throughput_tokens_s"] != "")]
    df["throughput_tokens_s"] = pd.to_numeric(df["throughput_tokens_s"])
    df["peak_rss_mb"] = pd.to_numeric(df["peak_rss_mb"])
    return df


def plot_throughput(local_df: pd.DataFrame, out_dir: Path):
    # Use n_prompt=512, n_gen=128 condition to match published baselines
    local_row = local_df[(local_df["n_prompt"] == 512) & (local_df["n_gen"] == 128)]
    bitnet_tps = local_row["throughput_tokens_s"].median() if not local_row.empty else None

    models = list(FP16_BASELINES.keys()) + ["BitNet b1.58 2B4T"]
    values = [FP16_BASELINES[m]["throughput_tokens_s"] for m in FP16_BASELINES]
    values.append(bitnet_tps if bitnet_tps else 0)
    colors = ["#4C72B0"] * len(FP16_BASELINES) + ["#DD8452"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, values, color=colors)
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set_xlabel("Throughput (tokens/s)")
    ax.set_title("Inference Throughput: BitNet b1.58 2B4T vs FP16 Baselines\n(n_prompt=512, n_gen=128, CPU)")
    ax.invert_yaxis()
    fig.tight_layout()
    path = out_dir / "throughput_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_memory(local_df: pd.DataFrame, out_dir: Path):
    local_row = local_df[(local_df["n_prompt"] == 512) & (local_df["n_gen"] == 128)]
    bitnet_rss = local_row["peak_rss_mb"].median() if not local_row.empty else None

    models = list(FP16_BASELINES.keys()) + ["BitNet b1.58 2B4T"]
    values = [FP16_BASELINES[m]["peak_rss_mb"] for m in FP16_BASELINES]
    values.append(bitnet_rss if bitnet_rss else 0)
    colors = ["#4C72B0"] * len(FP16_BASELINES) + ["#DD8452"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, values, color=colors)
    ax.bar_label(bars, fmt="%.0f MB", padding=4)
    ax.set_xlabel("Peak RSS (MB)")
    ax.set_title("Peak Memory: BitNet b1.58 2B4T vs FP16 Baselines\n(n_prompt=512, n_gen=128, CPU)")
    ax.invert_yaxis()
    fig.tight_layout()
    path = out_dir / "memory_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_throughput_by_config(local_df: pd.DataFrame, out_dir: Path):
    """Show BitNet throughput across all (n_prompt, n_gen) conditions."""
    configs = local_df.groupby(["n_prompt", "n_gen"])["throughput_tokens_s"].median().reset_index()
    labels = [f"p{row.n_prompt}/g{row.n_gen}" for _, row in configs.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, configs["throughput_tokens_s"], color="#DD8452")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("BitNet b1.58 2B4T Throughput by Prompt/Generation Length")
    fig.tight_layout()
    path = out_dir / "bitnet_throughput_configs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = parse_args()
    csv_path = Path(args.results)
    local_df = load_local(csv_path)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_throughput(local_df, PLOTS_DIR)
    plot_memory(local_df, PLOTS_DIR)
    plot_throughput_by_config(local_df, PLOTS_DIR)

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
