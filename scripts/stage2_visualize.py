#!/usr/bin/env python3
"""Create a fixed-noise qualitative contact sheet for a Stage 2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from rl3dsr.models.wan import GeometryConditioner, load_geometry_checkpoint
from rl3dsr.models.wan.lq_conditioning import conditioned_prediction
from rl3dsr.models.wan.flow import flow_matching_pair
from rl3dsr.models.wan import FrozenLQConditioner, WanDiT, WanVAE, load_adapter_checkpoint, load_flashvsr_projector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_experiment import _load_item


def _to_image(value: torch.Tensor) -> Image.Image:
    array = value.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(np.ascontiguousarray(array), "RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--lq-source", type=Path, required=True)
    parser.add_argument("--lq-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    manifest = json.loads((args.experiment_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    class Args:
        pass
    run_args = Args()
    run_args.scene = args.scene
    run_args.view_indices = tuple(manifest["view_indices"])
    _, hr, lr, camera = _load_item(run_args, type("Config", (), {"hr_resolution": config["hr_resolution"], "scale": config["scale"], "reference_index": config["reference_index"]})(), device)
    projector = load_flashvsr_projector(args.lq_source, args.lq_checkpoint, device=device, dtype=torch.bfloat16)
    conditioner = FrozenLQConditioner(projector, bridge_blocks=(0, 1, 2, 3), bridge_time_conditioning=True).to(device)
    load_adapter_checkpoint(args.parent_checkpoint, conditioner)
    conditioner.bridge.requires_grad_(False)
    vae = WanVAE.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    geometry = GeometryConditioner(representation=config["representation"], hidden_dim=config["hidden_dim"], attention_heads=config["attention_heads"], blocks=tuple(config["blocks"])).to(device)
    load_geometry_checkpoint(args.experiment_dir / "geometry_final.pt", geometry, expected_parent_checkpoint=str(args.parent_checkpoint.resolve()))
    with torch.inference_mode():
        clean = vae.encode_multiview(hr)
        features = conditioner.multiview_features(lr, conditioning_size=(config["hr_resolution"], config["hr_resolution"]), latent_shape=tuple(clean.shape[2:]))
        sigma = torch.tensor([0.5], device=device)
        noise = torch.randn(clean.shape, generator=torch.Generator(device=device).manual_seed(9000), device=device, dtype=clean.dtype)
        noisy, timestep, _ = flow_matching_pair(clean, noise, sigma)
        correct = conditioned_prediction(dit, conditioner, noisy, timestep, None, features, geometry_adapter=geometry, camera=camera, latent_shape=tuple(clean.shape[2:]))
        shuffled = conditioned_prediction(dit, conditioner, noisy, timestep, None, features, geometry_adapter=geometry, camera=type(camera)(camera.K, torch.roll(camera.T_world_from_camera, 1, dims=1), camera.image_size, camera.sequence_kind, camera.reference_index), latent_shape=tuple(clean.shape[2:]))
        baseline = conditioned_prediction(dit, conditioner, noisy, timestep, None, features)
        outputs = {
            "HR target": hr,
            "baseline": vae.decode_multiview(noisy - sigma.reshape(1, 1, 1, 1, 1) * baseline),
            "geometry correct": vae.decode_multiview(noisy - sigma.reshape(1, 1, 1, 1, 1) * correct),
            "geometry shuffled": vae.decode_multiview(noisy - sigma.reshape(1, 1, 1, 1, 1) * shuffled),
        }
    tile_w, tile_h = config["hr_resolution"], config["hr_resolution"]
    canvas = Image.new("RGB", (tile_w * len(manifest["view_indices"]), (tile_h + 28) * len(outputs)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (label, value) in enumerate(outputs.items()):
        for column in range(value.shape[2]):
            canvas.paste(_to_image(value[0, :, column]), (column * tile_w, row * (tile_h + 28) + 28))
            draw.text((column * tile_w + 4, row * (tile_h + 28) + 6), f"{label} / view {manifest['view_indices'][column]}", fill="black")
    canvas.save(args.experiment_dir / "qualitative_contact.png")
    (args.experiment_dir / "qualitative_review.json").write_text(json.dumps({"sigma": 0.5, "noise_seed": 9000, "rows": list(outputs), "views": manifest["view_indices"]}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
