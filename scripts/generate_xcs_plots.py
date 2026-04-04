"""
Generate XCS distribution plots from v2 CSV files.

Produces:
- plots/xcs_{DS}_v2_correct_vs_wrong.png  — histogram of XCS for correct vs wrong predictions
- plots/xcs_{DS}_v2_distribution.png      — overall XCS distribution with threshold line
- plots/xcs_{DS}_v2_scatter.png           — XCS vs confidence scatter plot
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATASETS = ["CICIDS2017", "UNSWNB15", "CICIDS2018"]
XCS_THRESHOLD = 0.3
OUTPUT_DIR = Path("plots")

def plot_xcs_distribution(df: pd.DataFrame, dataset: str):
    """Overall XCS distribution with threshold line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["xcs"], bins=20, edgecolor="black", alpha=0.7, color="#2196F3")
    ax.axvline(XCS_THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"Threshold ({XCS_THRESHOLD})")
    ax.set_xlabel("XCS")
    ax.set_ylabel("Count")
    ax.set_title(f"XCS Distribution — {dataset} (v2)")
    ax.legend()
    out = OUTPUT_DIR / f"xcs_{dataset}_v2_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_xcs_correct_vs_wrong(df: pd.DataFrame, dataset: str):
    """Histogram of XCS for correct vs wrong predictions."""
    # Only meaningful when we have correct/wrong labels
    if "correct" not in df.columns or df["correct"].nunique() < 2:
        print(f"  Skipping correct vs wrong plot for {dataset} (single class)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [("True", "#4CAF50"), ("False", "#F44336")]:
        subset = df[df["correct"] == label]
        if len(subset) > 0:
            ax.hist(subset["xcs"], bins=15, alpha=0.6, edgecolor="black",
                    label=f"Correct={label}", color=color)
    ax.axvline(XCS_THRESHOLD, color="red", linestyle="--", linewidth=2,
               label=f"Threshold ({XCS_THRESHOLD})")
    ax.set_xlabel("XCS")
    ax.set_ylabel("Count")
    ax.set_title(f"XCS: Correct vs Wrong — {dataset} (v2)")
    ax.legend()
    out = OUTPUT_DIR / f"xcs_{dataset}_v2_correct_vs_wrong.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_xcs_scatter(df: pd.DataFrame, dataset: str):
    """XCS vs confidence scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["confidence"], df["xcs"], alpha=0.6, s=30, c="#2196F3", edgecolors="white", linewidth=0.5)
    ax.axhline(XCS_THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"XCS threshold ({XCS_THRESHOLD})")
    ax.set_xlabel("Model Confidence")
    ax.set_ylabel("XCS")
    ax.set_title(f"XCS vs Confidence — {dataset} (v2)")
    ax.legend()
    out = OUTPUT_DIR / f"xcs_{dataset}_v2_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def main():
    print("Generating XCS plots from v2 CSVs")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        csv_path = Path(f"explanations/xcs_{ds}_v2.csv")
        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        print(f"\n{ds}: {len(df)} samples")

        plot_xcs_distribution(df, ds)
        plot_xcs_correct_vs_wrong(df, ds)
        plot_xcs_scatter(df, ds)

    print(f"\n{'='*60}")
    print("Done. All plots saved to plots/")

if __name__ == "__main__":
    main()
