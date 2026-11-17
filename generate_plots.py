"""Generates publication-quality figures for the CS6886W Assignment 2 report.

Produces:
  1. results/baseline_curves.png      - Train/Test Loss & Top-1 Accuracy curves (Q1c)
  2. results/parallel_coordinates.png - Wandb-style Parallel Coordinates plot across sweep runs (Q3b)
  3. results/compression_pareto.png   - Accuracy vs Model Size (MB) & Compression Ratio Pareto frontier (Q3, Q4)
"""

import csv
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, mode="r", newline="") as f:
        return list(csv.DictReader(f))


def plot_baseline_curves(history_csv: str, out_path: str) -> None:
    if not os.path.exists(history_csv):
        print(f"[Skip] {history_csv} not found.")
        return

    rows = load_csv(history_csv)
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    test_loss = [float(r["test_loss"]) for r in rows]
    train_acc = [float(r["train_acc"]) for r in rows]
    test_acc = [float(r["test_acc"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200)

    ax1.plot(epochs, train_loss, label="Train Loss (Label Smoothing=0.1)", color="#1f77b4", lw=1.8)
    ax1.plot(epochs, test_loss, label="Test Cross-Entropy Loss", color="#d62728", lw=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("MobileNet-v2 on CIFAR-10: Loss Curve")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.plot(epochs, train_acc, label="Train Top-1 Accuracy (%)", color="#1f77b4", lw=1.8)
    ax2.plot(epochs, test_acc, label="Test Top-1 Accuracy (%)", color="#2ca02c", lw=1.8)
    best_acc = max(test_acc)
    best_ep = epochs[test_acc.index(best_acc)]
    ax2.axhline(best_acc, color="#2ca02c", linestyle=":", alpha=0.8, label=f"Best Test: {best_acc:.2f}% (Ep {best_ep})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-1 Accuracy (%)")
    ax2.set_title("MobileNet-v2 on CIFAR-10: Accuracy Curve")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_parallel_coordinates(sweep_csv: str, out_path: str) -> None:
    """Renders a WandB-style Parallel Coordinates plot matching Figure 1 in the assignment."""
    if not os.path.exists(sweep_csv):
        print(f"[Skip] {sweep_csv} not found.")
        return

    rows = load_csv(sweep_csv)
    axes_keys = [
        "activation_quant_bits",
        "weight_quant_bits",
        "compression_ratio",
        "model_size_mb",
        "quantized_acc",
    ]
    axes_labels = [
        "activation_quant_bits",
        "weight_quant_bits",
        "compression_ratio",
        "model_size_mb",
        "quantized_acc",
    ]

    data = np.array([[float(r[k]) for k in axes_keys] for r in rows])
    num_runs, num_axes = data.shape

    mins = np.array([2.0, 2.0, np.floor(data[:, 2].min()), 0.5, max(0.0, np.floor(data[:, 4].min() / 10) * 10)])
    maxs = np.array([8.0, 8.0, np.ceil(data[:, 2].max()), np.ceil(data[:, 3].max() * 2) / 2, 100.0])
    spans = np.maximum(maxs - mins, 1e-6)
    norm_data = (data - mins) / spans

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
    x_positions = np.arange(num_axes)

    cmap = plt.colormaps["plasma"]
    acc_norm = mcolors.Normalize(vmin=data[:, 4].min(), vmax=data[:, 4].max())

    # Smooth cubic Bezier / cosine-interpolated curves like WandB parallel coordinates
    for i in range(num_runs):
        color = cmap(acc_norm(data[i, 4]))
        xs_dense = []
        ys_dense = []
        for j in range(num_axes - 1):
            t = np.linspace(0, 1, 30)
            # Smooth cosine interpolation between axis j and j+1
            s = 0.5 * (1 - np.cos(np.pi * t))
            xs_dense.extend(x_positions[j] + t)
            ys_dense.extend(norm_data[i, j] * (1 - s) + norm_data[i, j + 1] * s)
        ax.plot(xs_dense, ys_dense, color=color, alpha=0.82, lw=2.0)

    # Draw vertical axis lines and tick marks
    for j in range(num_axes):
        ax.axvline(x_positions[j], color="#555555", lw=1.2)
        ax.text(
            x_positions[j],
            1.05,
            axes_labels[j],
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )
        num_ticks = 7
        for tick_norm in np.linspace(0.0, 1.0, num_ticks):
            val = mins[j] + tick_norm * spans[j]
            fmt = f"{val:.1f}" if j < 4 else f"{val:.0f}"
            ax.plot([x_positions[j] - 0.03, x_positions[j]], [tick_norm, tick_norm], color="#555555", lw=1.0)
            ax.text(
                x_positions[j] - 0.05,
                tick_norm,
                fmt,
                ha="right",
                va="center",
                fontsize=8.5,
                color="#444444",
            )

    sm = cm.ScalarMappable(cmap=cmap, norm=acc_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=25)
    cbar.set_label("Quantized Test Top-1 Accuracy (%)", fontsize=10)

    ax.set_xlim(-0.35, num_axes - 0.8)
    ax.set_ylim(-0.05, 1.12)
    ax.axis("off")
    plt.title(
        "WandB Parallel Coordinates Plot: MobileNet-v2 CIFAR-10 Quantization Sweep (13 Runs)",
        fontsize=12,
        fontweight="bold",
        pad=18,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_pareto_frontier(sweep_csv: str, out_path: str) -> None:
    if not os.path.exists(sweep_csv):
        return

    rows = load_csv(sweep_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200)

    sizes = [float(r["model_size_mb"]) for r in rows]
    ratios = [float(r["compression_ratio"]) for r in rows]
    accs = [float(r["quantized_acc"]) for r in rows]
    labels = [r["run_id"] for r in rows]
    base_acc = float(rows[0]["baseline_acc"])

    ax1.axhline(base_acc, color="#2ca02c", linestyle="--", label=f"FP32 Baseline ({base_acc:.2f}%, 8.66 MB)")
    ax1.scatter(sizes, accs, c=ratios, cmap="viridis", s=75, edgecolors="black", zorder=3)
    for lbl, x, y in zip(labels, sizes, accs):
        ax1.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax1.set_xlabel("Compressed In-Memory Model Size (MB)")
    ax1.set_ylabel("Test Top-1 Accuracy (%)")
    ax1.set_title("Accuracy vs. In-Memory Model Size (MB)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")

    ax2.axhline(base_acc, color="#2ca02c", linestyle="--", label=f"FP32 Baseline ({base_acc:.2f}%)")
    ax2.scatter(ratios, accs, c=sizes, cmap="viridis_r", s=75, edgecolors="black", zorder=3)
    for lbl, x, y in zip(labels, ratios, accs):
        ax2.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax2.set_xlabel("In-Memory Model Compression Ratio (x)")
    ax2.set_ylabel("Test Top-1 Accuracy (%)")
    ax2.set_title("Accuracy vs. Model Compression Ratio")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    plot_baseline_curves("./results/training_history.csv", "./results/baseline_curves.png")
    plot_parallel_coordinates("./results/compression_sweep.csv", "./results/parallel_coordinates.png")
    plot_pareto_frontier("./results/compression_sweep.csv", "./results/compression_pareto.png")
