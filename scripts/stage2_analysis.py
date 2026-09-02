#!/usr/bin/env python3
"""Render Stage 2 training and geometry diagnostic plots from retained logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = _read(args.experiment_dir / "train_steps.csv")
    metrics = _read(args.experiment_dir / "geometry_metrics.csv")
    if not train:
        raise RuntimeError("train_steps.csv is empty")
    steps = [int(row["step"]) for row in train]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, [float(row["loss"]) for row in train], label="train loss")
    axes[0, 0].set_title("Flow loss")
    axes[0, 1].plot(steps, [float(row["sigma"]) for row in train], label="sigma", color="tab:orange")
    axes[0, 1].set_title("Training sigma")
    axes[1, 0].plot(steps, [float(row["gradient_norm"]) for row in train], label="gradient", color="tab:green")
    axes[1, 0].set_title("Geometry gradient norm")
    axes[1, 1].plot(steps, [float(row["peak_gpu_memory_mib"]) for row in train], label="MiB", color="tab:red")
    axes[1, 1].set_title("Peak GPU memory")
    for axis in axes.flat:
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
    fig.savefig(args.experiment_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    if metrics:
        eval_steps = [int(row["step"]) for row in metrics]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        axes[0].plot(eval_steps, [float(row["correct_loss"]) for row in metrics], label="correct")
        axes[0].plot(eval_steps, [float(row["shuffled_loss"]) for row in metrics], label="shuffled")
        axes[0].plot(eval_steps, [float(row["baseline_loss"]) for row in metrics], label="baseline")
        axes[0].set_title("Fixed-noise flow loss")
        axes[1].plot(eval_steps, [float(row["correct_shuffled_delta"]) for row in metrics], label="correct-shuffled")
        axes[1].plot(eval_steps, [float(row["correct_baseline_delta"]) for row in metrics], label="correct-baseline")
        axes[1].set_title("Camera intervention")
        if "correct_psnr" in metrics[0]:
            axes[2].plot(eval_steps, [float(row["correct_psnr"]) for row in metrics], label="geometry")
            axes[2].plot(eval_steps, [float(row["baseline_psnr"]) for row in metrics], label="baseline")
            axes[2].set_title("Decoded PSNR")
        else:
            axes[2].axis("off")
        for axis in axes[:2]:
            axis.set_xlabel("step")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.savefig(args.experiment_dir / "geometry_diagnostics.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
