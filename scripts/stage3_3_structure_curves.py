#!/usr/bin/env python3
"""Plot frozen per-step loss and gradient telemetry for one candidate cell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stage3_3_structure as campaign


def main(name):
    rows = [json.loads(line) for line in (
        campaign.CAMPAIGN / "train" / name / "train_steps.jsonl"
    ).read_text().splitlines()]
    if len(rows) != 1000 or [row["step"] for row in rows] != list(range(1, 1001)):
        raise ValueError("training curve requires complete 1000-step telemetry")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    groups = (
        ("loss", "main_flow_loss", "camera_rank_loss", "lr_rank_loss"),
        ("correct_image_ssim_loss",),
        ("gradient_norm", "fusion_gradient_norm", "bridge_gradient_norm"),
        ("camera_rank_active_fraction", "lr_rank_active_fraction"),
    )
    for axis, keys in zip(axes.flat, groups):
        for key in keys:
            if key in rows[0]:
                axis.plot([r["step"] for r in rows], [r[key] for r in rows],
                          label=key, linewidth=.8)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("step")
    fig.tight_layout()
    path = campaign.CAMPAIGN / "analysis" / name / "training_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=campaign.CELLS)
    main(parser.parse_args().cell)
