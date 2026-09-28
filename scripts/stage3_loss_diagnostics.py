#!/usr/bin/env python3
"""Paired full-trajectory RRE and fusion knockout diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

import stage3_experiment as st
import stage3_loss_experiment as experiment
from rl3dsr.validation.decoded_space import frame_metrics
from rl3dsr.validation.stage3_protocol import load_stage3_config


CONDITIONS = ("correct", "geometry_off", "fusion_off", "both_off")


def evaluate(config_path: Path, checkpoint: Path, manifest_path: Path,
             output: Path, seeds: tuple[int, ...]):
    if output.exists():
        raise RuntimeError(f"diagnostic output already exists: {output}")
    output.mkdir(parents=True)
    config = load_stage3_config(config_path)
    manifest, groups = st._load_seen_groups(manifest_path, config, "probe", None)
    st._seed_all(seeds[0])
    model = experiment.runtime(config, checkpoint)
    model.module.eval()
    model.dit.model.eval().requires_grad_(False)
    model.vae.model.model.eval().requires_grad_(False)
    metric = st._PerFrameLPIPS(model.device).eval().requires_grad_(False)
    rows = []
    for group in groups:
        _, hr, lr, camera = st._load_indices(
            experiment.RUNTIME_ROOT / "datasets/nerf_synthetic",
            group["scene"], "train", group["indices"], config, model.device,
        )
        with torch.no_grad():
            clean = model.vae.encode_multiview(hr)
        for seed in seeds:
            sample_seed = seed * 1_000_000 + group["anchor"]
            for condition in CONDITIONS:
                geometry, fusion = model.module.geometry, model.module.fusion
                try:
                    if condition in {"geometry_off", "both_off"}:
                        model.module.geometry = None
                    if condition in {"fusion_off", "both_off"}:
                        model.module.fusion = None
                    with torch.inference_mode():
                        latent = st.sample_latents(
                            model, lr, camera, tuple(clean.shape), config.sampling_steps,
                            config.image_size, seed=sample_seed,
                            sampling_shift=config.sampling_shift, dtype=torch.bfloat16,
                        )
                        decoded = model.vae.decode_multiview(latent)[:, :, :1]
                        values = frame_metrics(decoded, hr[:, :, :1],
                                               perceptual_metric=metric)[0]
                finally:
                    model.module.geometry, model.module.fusion = geometry, fusion
                rows.append({
                    "group_id": group["id"], "anchor": group["anchor"],
                    "seed": seed, "condition": condition,
                    **{key: float(values[key]) for key in ("psnr", "ssim", "lpips")},
                })
                image_path = output / "images" / group["scene"] / f"view_{group['anchor']:03d}" / f"seed_{seed}" / f"{condition}.png"
                st._save_frame(image_path, decoded)
    expected = len(groups) * len(seeds) * len(CONDITIONS)
    if len(rows) != expected:
        raise RuntimeError("diagnostic evaluation row count mismatch")
    with (output / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with (output / "rows.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {condition: {
        key: sum(row[key] for row in rows if row["condition"] == condition) /
             (len(groups) * len(seeds))
        for key in ("psnr", "ssim", "lpips")
    } for condition in CONDITIONS}
    experiment.frozen_json(output / "summary.json", {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": experiment.sha256(checkpoint),
        "config_sha256": experiment.sha256(config_path),
        "manifest_sha256": experiment.sha256(manifest_path),
        "seeds": list(seeds), "groups": len(groups), "rows": len(rows),
        "means": means,
        "correct_minus_knockout": {
            name: {key: means["correct"][key] - means[name][key]
                   for key in ("psnr", "ssim", "lpips")}
            for name in CONDITIONS if name != "correct"
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3302, 3303))
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.manifest, args.output, tuple(args.seeds))


if __name__ == "__main__":
    main()
