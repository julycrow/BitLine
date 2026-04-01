#!/usr/bin/env python3
"""Plot all metrics in one figure for method/timestep comparison table.

Example:
  python tools/analysis_tools/plot_method_timestep_metrics.py \
    --csv tools/analysis_tools/timestep_method_metrics.csv \
    --out tools/analysis_tools/timestep_method_metrics_onefig.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot all metrics in one figure for method/timestep table.")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path.")
    parser.add_argument(
        "--out",
        type=str,
        default="timestep_method_metrics_onefig.png",
        help="Output image path.")
    parser.add_argument(
        "--title",
        type=str,
        default="Method/Timesteps Metric Trends",
        help="Figure title.")
    parser.add_argument(
        "--value-mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="relative: plot percentage change vs first row (baseline).")
    return parser.parse_args()


def read_table(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty.")
    headers = reader.fieldnames or []
    if len(headers) < 2:
        raise ValueError("CSV must include method column + at least one metric.")

    method_key = headers[0]
    metric_keys = headers[1:]
    methods = [r[method_key] for r in rows]
    metrics = {}
    for k in metric_keys:
        metrics[k] = np.asarray([float(r[k]) for r in rows], dtype=np.float64)
    return methods, metrics


def get_colors(n):
    base = [
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
        "#56B4E9", "#F0E442", "#1B9E77", "#7570B3", "#E7298A",
        "#66A61E", "#A6761D", "#7F7F7F", "#000000"
    ]
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]


def plot_one_figure(methods, metrics, out_path, title, value_mode):
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    colors = get_colors(len(metrics))

    for i, (name, vals) in enumerate(metrics.items()):
        y = vals.copy()
        if value_mode == "relative":
            base = vals[0]
            denom = max(abs(base), 1e-12)
            y = (vals - base) / denom * 100.0
        ax.plot(
            x,
            y,
            label=name,
            color=colors[i],
            linewidth=2.2,
            marker="o",
            markersize=5,
            markerfacecolor=colors[i],
            markeredgecolor="white",
            markeredgewidth=0.7,
            alpha=0.95,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    if value_mode == "relative":
        ax.set_ylabel("Relative Change vs Baseline (%)")
        ax.set_title(f"{title} (Relative)")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.9)
    else:
        ax.set_ylabel("Metric Value (absolute)")
        ax.set_title(f"{title} (Absolute)")
    ax.grid(alpha=0.25, linestyle="--")

    # Mark baseline for easy comparison.
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.annotate("Baseline", (0, ax.get_ylim()[1]), textcoords="offset points", xytext=(6, -16), fontsize=9)

    # Put legend outside for readability.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


def main():
    args = parse_args()
    methods, metrics = read_table(args.csv)
    plot_one_figure(methods, metrics, args.out, args.title, args.value_mode)


if __name__ == "__main__":
    main()
