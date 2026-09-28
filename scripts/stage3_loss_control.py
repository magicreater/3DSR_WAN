#!/usr/bin/env python3
"""Matched 4000-step module-off training controls for the selected loss arm."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import stage3_experiment as st
import stage3_loss_campaign as campaign
import stage3_loss_experiment as experiment
from rl3dsr.validation.decoded_space import frame_metrics
from rl3dsr.validation.stage3_protocol import load_stage3_config


CONTROLS = {"geometry_off": "geometry", "fusion_off": "fusion"}


def selected() -> tuple[str, str, str]:
    review = json.loads((campaign.OUT / "full_review.json").read_text())
    screen = json.loads((campaign.OUT / "screen_review.json").read_text())
    arm = review["best_candidate"]
    if arm not in campaign.FULL:
        raise RuntimeError("invalid selected full arm")
    pair = {"full_core": "none", "full_ordinary": "ordinary",
            "full_challenge": "challenge"}[arm]
    return arm, screen["selected_loss"], pair


def train(control: str):
    campaign.assert_protocol()
    selected_arm, loss_mode, pair_mode = selected()
    path = campaign.OUT / "train" / f"control_{control}"
    checkpoint = campaign.OUT / "train" / selected_arm / "stage3_step_4000.pt"
    experiment.frozen_json(campaign.OUT / f"control_{control}_protocol.json", {
        "selected_arm": selected_arm,
        "selected_checkpoint_sha256": experiment.sha256(checkpoint),
        "loss_mode": loss_mode, "pair_mode": pair_mode,
        "disabled_module": control,
        "control_source_sha256": experiment.sha256(Path(__file__)),
        "parent_sha256": experiment.sha256(experiment.PARENT),
    })
    original_runtime = experiment.runtime
    original_save = experiment.save_stage3_checkpoint
    removed = {}

    def runtime_without_module(*args, **kwargs):
        model = original_runtime(*args, **kwargs)
        module = getattr(model.module, CONTROLS[control])
        if module is None:
            raise RuntimeError("module-off control expected an active module")
        removed[id(model.module)] = module
        setattr(model.module, CONTROLS[control], None)
        return model

    def save_with_full_structure(path, module, **kwargs):
        if getattr(module, CONTROLS[control]) is not None:
            raise RuntimeError("control module unexpectedly active during training")
        provenance = dict(kwargs["provenance"])
        provenance["disabled_module"] = control
        provenance["control_source_sha256"] = experiment.sha256(Path(__file__))
        kwargs["provenance"] = provenance
        setattr(module, CONTROLS[control], removed[id(module)])
        try:
            original_save(path, module, **kwargs)
        finally:
            setattr(module, CONTROLS[control], None)

    experiment.runtime = runtime_without_module
    experiment.save_stage3_checkpoint = save_with_full_structure
    try:
        experiment.train(SimpleNamespace(
            config=campaign.config_path("full"),
            calibration=campaign.OUT / "calibration.json",
            output=path, loss_mode=loss_mode, pair_mode=pair_mode,
            experiment=f"stage3_loss_{control}_20260926", seed=42,
        ))
    finally:
        experiment.runtime = original_runtime
        experiment.save_stage3_checkpoint = original_save


def evaluate(control: str):
    campaign.assert_protocol()
    selected_arm, _, _ = selected()
    checkpoint = campaign.OUT / "train" / f"control_{control}" / "stage3_step_4000.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["provenance"].get("disabled_module") != control:
        raise RuntimeError("control checkpoint provenance mismatch")
    output = campaign.OUT / "eval" / f"control_{control}"
    if output.exists():
        raise RuntimeError(f"control evaluation output already exists: {output}")
    output.mkdir(parents=True)
    config = load_stage3_config(campaign.config_path("full"))
    _, groups = st._load_seen_groups(campaign.manifest_path("full"), config, "probe", None)
    st._seed_all(3302)
    model = experiment.runtime(config, checkpoint)
    setattr(model.module, CONTROLS[control], None)
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
        with torch.inference_mode():
            clean = model.vae.encode_multiview(hr)
            for seed in (3302, 3303):
                latent = st.sample_latents(
                    model, lr, camera, tuple(clean.shape), config.sampling_steps,
                    config.image_size, seed=seed * 1_000_000 + group["anchor"],
                    sampling_shift=config.sampling_shift, dtype=torch.bfloat16,
                )
                decoded = model.vae.decode_multiview(latent)[:, :, :1]
                values = frame_metrics(decoded, hr[:, :, :1], perceptual_metric=metric)[0]
                rows.append({"group_id": group["id"], "seed": seed,
                             "control": control, "selected_arm": selected_arm,
                             **{key: float(values[key]) for key in ("psnr", "ssim", "lpips")}})
                st._save_frame(output / "images" / f"view_{group['anchor']:03d}"
                               / f"seed_{seed}.png", decoded)
    if len(rows) != 8:
        raise RuntimeError("control evaluation row count mismatch")
    with (output / "rows.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    experiment.frozen_json(output / "summary.json", {
        "checkpoint_sha256": experiment.sha256(checkpoint),
        "control": control, "selected_arm": selected_arm,
        "means": {key: sum(row[key] for row in rows) / 8
                  for key in ("psnr", "ssim", "lpips")},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval"))
    parser.add_argument("--control", required=True, choices=CONTROLS)
    args = parser.parse_args()
    {"train": train, "eval": evaluate}[args.command](args.control)


if __name__ == "__main__":
    main()
