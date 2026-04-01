#!/usr/bin/env python3
"""Plot metric trends as timesteps increase.

Usage:
  python tools/analysis_tools/plot_timestep_metrics.py \
      --csv tools/analysis_tools/timestep_metrics_example.csv \
      --out tools/analysis_tools/timestep_metrics.png
"""

import argparse
import csv
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize metric changes along diffusion timesteps.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Input CSV path. First column must be timestep.")
    parser.add_argument(
        "--out",
        type=str,
        default="timestep_metrics.png",
        help="Output image path.")
    parser.add_argument(
        "--title",
        type=str,
        default="Metric Trends vs Timesteps",
        help="Figure title.")
    parser.add_argument(
        "--mode",
        type=str,
        default="overlay",
        choices=["both", "subplots", "overlay", "decision", "balanced"],
        help="decision: plots for timestep selection (gain-cost style).")
    parser.add_argument(
        "--decision-metric",
        type=str,
        default="TOPO_F1",
        help="Primary quality metric used in decision mode.")
    parser.add_argument(
        "--baseline-step",
        type=float,
        default=50.0,
        help="Reference timestep for relative-gain plot in decision mode.")
    return parser.parse_args()


def read_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty.")

    headers = reader.fieldnames or []
    if not headers:
        raise ValueError("CSV has no header.")
    x_key = headers[0]
    metric_keys = headers[1:]
    if not metric_keys:
        raise ValueError("CSV needs at least one metric column.")

    x_vals = [float(r[x_key]) for r in rows]
    metrics = {}
    for key in metric_keys:
        metrics[key] = [float(r[key]) for r in rows]
    return x_key, x_vals, metrics


def get_vivid_colors(n):
    # Paper-style, colorblind-friendly palette (Okabe-Ito first).
    okabe_ito = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # green
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#000000",  # black
    ]
    supplements = [
        "#7F7F7F", "#1B9E77", "#7570B3", "#E7298A", "#66A61E",
        "#E6AB02", "#A6761D", "#666666", "#A6CEE3", "#FB9A99",
        "#B2DF8A", "#FDBF6F", "#CAB2D6", "#B15928"
    ]

    base = okabe_ito + supplements
    if n <= len(base):
        return base[:n]

    colors = list(base)
    for i in range(n - len(base)):
        h = ((i * 0.618033988749895) % 1.0)
        s = 0.85
        v = 0.90
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append("#{0:02x}{1:02x}{2:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)))
    return colors


def plot_subplots(x_key, x_vals, metrics, out_path, title):
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colors = get_vivid_colors(n)

    for idx, (ax, (name, vals)) in enumerate(zip(axes, metrics.items())):
        color = colors[idx]
        ax.plot(
            x_vals,
            vals,
            marker="o",
            markersize=6,
            linewidth=2.8,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8)
        for x, y in zip(x_vals, vals):
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
                color=color)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)

    axes[-1].set_xlabel(x_key)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved subplot figure to: {out_path}")


def _minmax_norm(vals):
    arr = np.asarray(vals, dtype=np.float64)
    vmin, vmax = arr.min(), arr.max()
    if np.isclose(vmin, vmax):
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def _separate_overlapped_curves(curves, eps=0.012):
    """Add tiny vertical offsets to identical curves for visibility."""
    separated = {}
    groups = {}
    for name, vals in curves.items():
        key = tuple(np.round(np.asarray(vals, dtype=np.float64), 12))
        groups.setdefault(key, []).append(name)

    for _, names in groups.items():
        k = len(names)
        if k == 1:
            n = names[0]
            separated[n] = np.asarray(curves[n], dtype=np.float64)
            continue
        center = (k - 1) / 2.0
        for idx, n in enumerate(names):
            offset = (idx - center) * eps
            v = np.asarray(curves[n], dtype=np.float64) + offset
            separated[n] = np.clip(v, 0.0, 1.0)
    return separated


def plot_overlay_normalized(x_key, x_vals, metrics, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n = len(metrics)
    colors = get_vivid_colors(n)
    normalized = {}
    for name, vals in metrics.items():
        norm_vals = _minmax_norm(vals)
        # Constant series are meaningful; place them near zero for cleaner view.
        if np.allclose(norm_vals, 0):
            norm_vals = np.full_like(norm_vals, 0.03)
        normalized[name] = norm_vals

    normalized = _separate_overlapped_curves(normalized, eps=0.012)

    for idx, (name, vals) in enumerate(metrics.items()):
        color = colors[idx]
        norm_vals = normalized[name]
        ax.plot(
            x_vals,
            norm_vals,
            marker="o",
            markersize=6,
            linewidth=2.8,
            label=name,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8)

    ax.set_xlabel(x_key)
    ax.set_ylabel("normalized value [0, 1]")
    ax.set_title(f"{title} (All Metrics, Normalized)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved overlay figure to: {out_path}")


def plot_balanced_onefig(x_key, x_vals, metrics, out_path, title, baseline_step):
    x = np.asarray(x_vals, dtype=np.float64)
    cost_key = _pick_cost_key(metrics)
    ignore = {cost_key, "Params_M", "params_m", "Params"}
    quality_keys = [k for k in metrics.keys() if k not in ignore]
    if not quality_keys:
        raise ValueError("No quality metrics found for balanced mode.")

    # Normalize all quality metrics
    normalized = {}
    for k in quality_keys:
        v = _minmax_norm(metrics[k])
        if np.allclose(v, 0):
            v = np.full_like(v, 0.03)
        normalized[k] = v
    normalized = _separate_overlapped_curves(normalized, eps=0.012)

    # Aggregate quality and cost
    quality_stack = np.vstack([normalized[k] for k in quality_keys])
    quality_mean = quality_stack.mean(axis=0)
    cost_norm = _minmax_norm(metrics[cost_key])

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    colors = get_vivid_colors(len(quality_keys))
    for idx, k in enumerate(quality_keys):
        ax.plot(
            x,
            normalized[k],
            color=colors[idx],
            linewidth=1.5,
            alpha=0.85,
            marker="o",
            markersize=4,
            label=k,
        )

    # Emphasize "quality vs cost" tradeoff
    ax.plot(x, quality_mean, color="#009E73", linewidth=3.4, marker="o", markersize=6, label="Quality Mean")
    ax.plot(x, cost_norm, color="#D55E00", linewidth=3.4, marker="s", markersize=6, label=f"{cost_key} (norm)")

    base_idx = int(np.argmin(np.abs(x - baseline_step)))
    base_step = x[base_idx]
    ax.axvline(base_step, color="black", linestyle="--", linewidth=1.3, alpha=0.9)
    ax.annotate(
        f"Balanced choice: t={int(base_step)}",
        (base_step, 0.9),
        textcoords="offset points",
        xytext=(8, 0),
        fontsize=10,
        color="black",
    )

    ax.set_xlabel(x_key)
    ax.set_ylabel("normalized value [0, 1]")
    ax.set_title(f"{title} (All Trends + Balanced Point)")
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.8)
    ax.legend(ncol=3, fontsize=8, loc="upper left", frameon=True)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved balanced figure to: {out_path}")


def _pick_cost_key(metrics):
    for key in ("FLOPs_G", "flops_gflops", "FLOPs", "Latency_ms", "latency_ms"):
        if key in metrics:
            return key
    raise ValueError("Decision mode needs a cost column like FLOPs_G.")


def plot_decision_views(x_vals, metrics, out_path, title, metric_key, baseline_step):
    if metric_key not in metrics:
        raise ValueError(f"decision metric '{metric_key}' not found in CSV.")
    cost_key = _pick_cost_key(metrics)
    quality = np.asarray(metrics[metric_key], dtype=np.float64)
    cost = np.asarray(metrics[cost_key], dtype=np.float64)
    x = np.asarray(x_vals, dtype=np.float64)

    # 1) Gain-Cost scatter
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.5))
    ax1.plot(cost, quality, marker="o", linewidth=2.6, color="#0072B2")
    for xi, yi, ti in zip(cost, quality, x):
        ax1.annotate(f"t={int(ti)}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax1.set_xlabel(cost_key)
    ax1.set_ylabel(metric_key)
    ax1.set_title(f"{title} - Gain vs Cost")
    ax1.grid(alpha=0.25, linestyle="--")
    fig1.tight_layout()
    p1 = Path(out_path).with_name(Path(out_path).stem + "_decision_gain_cost" + Path(out_path).suffix)
    fig1.savefig(p1, dpi=220, bbox_inches="tight")
    print(f"Saved decision figure: {p1}")

    # 2) Marginal gain per cost
    d_metric = np.diff(quality)
    d_cost = np.diff(cost)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(d_cost) > 1e-12, d_metric / d_cost, 0.0)
    mid_steps = x[1:]
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
    ax2.bar(mid_steps, ratio, width=0.8 * np.min(np.diff(np.unique(x))), color="#E69F00", edgecolor="black", linewidth=0.5)
    for xi, yi in zip(mid_steps, ratio):
        ax2.annotate(f"{yi:.2e}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax2.set_xlabel("timestep")
    ax2.set_ylabel(f"Δ{metric_key} / Δ{cost_key}")
    ax2.set_title(f"{title} - Marginal Gain")
    ax2.grid(alpha=0.25, linestyle="--", axis="y")
    fig2.tight_layout()
    p2 = Path(out_path).with_name(Path(out_path).stem + "_decision_marginal" + Path(out_path).suffix)
    fig2.savefig(p2, dpi=220, bbox_inches="tight")
    print(f"Saved decision figure: {p2}")

    # 3) Relative to baseline step
    base_idx = int(np.argmin(np.abs(x - baseline_step)))
    base_step = x[base_idx]
    q_rel = (quality - quality[base_idx]) / max(abs(quality[base_idx]), 1e-12) * 100.0
    c_rel = (cost - cost[base_idx]) / max(abs(cost[base_idx]), 1e-12) * 100.0
    fig3, ax3 = plt.subplots(figsize=(8.0, 5.2))
    ax3.plot(x, q_rel, marker="o", linewidth=2.4, label=f"{metric_key} gain (%)", color="#009E73")
    ax3.plot(x, c_rel, marker="s", linewidth=2.4, label=f"{cost_key} increase (%)", color="#D55E00")
    ax3.axvline(base_step, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax3.annotate(f"baseline t={int(base_step)}", (base_step, 0), textcoords="offset points", xytext=(8, 8), fontsize=9, color="gray")
    ax3.set_xlabel("timestep")
    ax3.set_ylabel("relative change (%)")
    ax3.set_title(f"{title} - Relative to t={int(base_step)}")
    ax3.legend()
    ax3.grid(alpha=0.25, linestyle="--")
    fig3.tight_layout()
    p3 = Path(out_path).with_name(Path(out_path).stem + "_decision_relative" + Path(out_path).suffix)
    fig3.savefig(p3, dpi=220, bbox_inches="tight")
    print(f"Saved decision figure: {p3}")


def main():
    args = parse_args()
    x_key, x_vals, metrics = read_csv(args.csv)
    out_path = Path(args.out)

    if args.mode in ("both", "subplots"):
        subplot_out = out_path
        plot_subplots(x_key, x_vals, metrics, subplot_out, args.title)

    if args.mode in ("both", "overlay"):
        overlay_out = out_path if args.mode == "overlay" else out_path.with_name(
            out_path.stem + "_overlay_norm" + out_path.suffix)
        plot_overlay_normalized(x_key, x_vals, metrics, overlay_out, args.title)
    if args.mode == "balanced":
        plot_balanced_onefig(
            x_key=x_key,
            x_vals=x_vals,
            metrics=metrics,
            out_path=out_path,
            title=args.title,
            baseline_step=args.baseline_step,
        )
    if args.mode == "decision":
        plot_decision_views(
            x_vals=x_vals,
            metrics=metrics,
            out_path=out_path,
            title=args.title,
            metric_key=args.decision_metric,
            baseline_step=args.baseline_step,
        )


if __name__ == "__main__":
    main()
