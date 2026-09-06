#!/usr/bin/env python3
"""Evaluate one Stage 2 checkpoint on the four memorized training views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rl3dsr.data import NeRFSyntheticAdapter, Split
from rl3dsr.data.tensor_prep import rgb_images_to_video
from rl3dsr.models.wan import (
    CameraBatch,
    FullRREConditioner,
    FrozenLQConditioner,
    GeometryConditioner,
    Stage1Degradation,
    WanDiT,
    WanVAE,
    flow_matching_loss,
    flow_matching_pair,
    load_adapter_checkpoint,
    load_flashvsr_projector,
    load_geometry_checkpoint,
)
from rl3dsr.models.wan.lq_conditioning import conditioned_prediction
from rl3dsr.models.wan.sampling import FlowSamplingConfig, sample_conditioned_flow
from rl3dsr.validation.decoded_space import frame_metrics, velocity_to_clean
from rl3dsr.validation.memorization import camera_usage_verdict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _camera(observations, resolution: int, device: torch.device) -> CameraBatch:
    intrinsics, transforms = [], []
    for observation in observations:
        K = torch.from_numpy(np.asarray(observation.K, dtype=np.float32)).clone()
        K[0] *= resolution / observation.width
        K[1] *= resolution / observation.height
        intrinsics.append(K)
        transforms.append(torch.from_numpy(np.asarray(observation.T_world_from_camera, dtype=np.float32)))
    return CameraBatch(
        torch.stack(intrinsics).unsqueeze(0).to(device),
        torch.stack(transforms).unsqueeze(0).to(device),
        (resolution, resolution),
        "multiview",
    )


def _load_scene(scene: Path, indices: tuple[int, ...], resolution: int, device: torch.device):
    adapter = NeRFSyntheticAdapter(scene)
    sequence = adapter.index(Split.TRAIN)
    observations = tuple(sequence.observations[index] for index in indices)
    hr = rgb_images_to_video([adapter.load_rgb(item) for item in observations], resolution).to(device)
    lr = Stage1Degradation(scale=4)(hr)
    return sequence.scene_id, hr, lr, _camera(observations, resolution, device)


def _upsample(lr: torch.Tensor, resolution: int) -> torch.Tensor:
    batch, channels, views, height, width = lr.shape
    flat = lr.permute(0, 2, 1, 3, 4).reshape(batch * views, channels, height, width)
    value = F.interpolate(
        flat,
        size=(resolution, resolution),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return value.reshape(batch, views, channels, resolution, resolution).permute(0, 2, 1, 3, 4)


def _perceptual(device: torch.device):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", reduction="mean", normalize=False
    ).to(device).eval().requires_grad_(False)
    return metric.net


def _metric_rows(
    value: torch.Tensor,
    target: torch.Tensor,
    metric,
    *,
    scene: str,
    condition: str,
    seed: int,
    indices: tuple[int, ...],
    checkpoint_step: int,
) -> list[dict]:
    rows = frame_metrics(value.detach().float().cpu(), target.detach().float().cpu(), perceptual_metric=metric)
    return [
        {
            "scene": scene,
            "evaluation_group": "seen",
            "condition": condition,
            "checkpoint_step": checkpoint_step,
            "inference_seed": seed,
            "view_position": int(row["frame_index"]),
            "view_index": indices[int(row["frame_index"])],
            **{key: float(value) for key, value in row.items() if key not in {"batch_index", "frame_index"}},
        }
        for row in rows
    ]


def _save_images(root: Path, condition: str, seed: int, value: torch.Tensor, indices: tuple[int, ...]) -> None:
    target = root / f"seed_{seed}" / condition
    target.mkdir(parents=True, exist_ok=True)
    frames = value.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5).round().byte()[0]
    for position, index in enumerate(indices):
        image = frames[:, position].permute(1, 2, 0).numpy()
        Image.fromarray(image, mode="RGB").save(target / f"view_{index:03d}.png")


def _roll(camera: CameraBatch) -> CameraBatch:
    order = torch.roll(torch.arange(camera.K.shape[1], device=camera.K.device), 1)
    return CameraBatch(
        camera.K,
        camera.T_world_from_camera[:, order],
        camera.image_size,
        camera.sequence_kind,
        camera.reference_index,
    )


def _perturb(camera: CameraBatch, amount: float) -> CameraBatch:
    transform = camera.T_world_from_camera.float().clone()
    transform[..., 0, 3] += amount
    return CameraBatch(camera.K.float(), transform, camera.image_size, camera.sequence_kind, camera.reference_index)


def _geometry(path: Path, representation: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    if representation == "rre_full":
        geometry = FullRREConditioner(
            hidden_dim=int(config.get("hidden_dim", 192)),
            attention_heads=int(config.get("attention_heads", 1)),
            branch_count=int(config.get("branch_count", 30)),
            compression=int(config.get("compression", 8)),
            absmap=bool(config.get("absmap", True)),
            world_up=tuple(config.get("world_up", (0.0, 0.0, 1.0))),
        )
    else:
        geometry = GeometryConditioner(
            representation=representation,
            hidden_dim=int(config.get("hidden_dim", 192)),
            attention_heads=int(config.get("attention_heads", 4)),
            blocks=tuple(config.get("blocks", (0, 1, 2, 3))),
        )
    return geometry.to(device).eval().requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--lq-source", type=Path, required=True)
    parser.add_argument("--lq-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path)
    parser.add_argument("--representation", choices=("stage1", "rre", "plucker", "rre_full"), required=True)
    parser.add_argument("--checkpoint-step", type=int, default=0)
    parser.add_argument("--view-indices", type=int, nargs=4, default=(0, 33, 66, 99))
    parser.add_argument("--inference-seeds", type=int, nargs="+", default=(2201,))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--sampling-shift", type=float, default=5.0)
    parser.add_argument("--pose-perturbation", type=float, default=0.02)
    parser.add_argument("--include-stage1", action="store_true")
    parser.add_argument("--pure-noise-camera-controls", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.representation == "stage1" and args.geometry_checkpoint is not None:
        raise ValueError("stage1 evaluation cannot load a geometry checkpoint")
    if args.representation != "stage1" and args.geometry_checkpoint is None:
        raise ValueError("geometry evaluation requires --geometry-checkpoint")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device(args.device)
    projector = load_flashvsr_projector(
        args.lq_source, args.lq_checkpoint, device=device, dtype=torch.bfloat16
    )
    conditioner = FrozenLQConditioner(
        projector, bridge_blocks=(0, 1, 2, 3), bridge_time_conditioning=True
    ).to(device).eval()
    load_adapter_checkpoint(args.parent_checkpoint, conditioner)
    conditioner.bridge.requires_grad_(False)
    vae = WanVAE.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    scene_id, hr, lr, camera = _load_scene(
        args.scene, tuple(args.view_indices), args.resolution, device
    )
    with torch.inference_mode():
        clean = vae.encode_multiview(hr)
        features = conditioner.multiview_features(
            lr,
            conditioning_size=(args.resolution, args.resolution),
            latent_shape=tuple(clean.shape[2:]),
        )
        vae_ceiling = vae.decode_multiview(clean)
    bicubic = _upsample(lr, args.resolution)
    geometry = None
    checkpoint_sha256 = None
    if args.geometry_checkpoint is not None:
        geometry = _geometry(args.geometry_checkpoint, args.representation, device)
        metadata = load_geometry_checkpoint(
            args.geometry_checkpoint,
            geometry,
            expected_parent_checkpoint=str(args.parent_checkpoint.resolve()),
        )
        if int(metadata.get("step", -1)) != args.checkpoint_step:
            raise RuntimeError("checkpoint step does not match --checkpoint-step")
        checkpoint_sha256 = _sha256(args.geometry_checkpoint)
    metric = _perceptual(device)
    sampling = FlowSamplingConfig(args.sampling_steps, args.sampling_shift)
    metric_rows: list[dict] = []
    one_step_rows: list[dict] = []
    control_rows: list[dict] = []

    def sample_with(current_geometry, current_camera, seed: int):
        initial_noise = torch.randn(
            clean.shape,
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
            dtype=torch.float32,
        )

        def predict(sample, timestep, context, supplied_features):
            return conditioned_prediction(
                dit,
                conditioner,
                sample,
                timestep,
                context,
                supplied_features,
                geometry_adapter=current_geometry,
                camera=current_camera,
                latent_shape=tuple(clean.shape[2:]),
            )

        latent = sample_conditioned_flow(
            initial_noise,
            features,
            torch.zeros(1, 512, 4096, device=device, dtype=torch.bfloat16),
            predict_velocity=predict,
            config=sampling,
        )
        return vae.decode_multiview(latent)

    with torch.inference_mode():
        if geometry is not None:
            sigma = torch.tensor([0.5], device=device)
            control_noise = torch.randn(
                clean.shape,
                generator=torch.Generator(device=device).manual_seed(9000),
                device=device,
                dtype=clean.dtype,
            )
            noisy, timestep, target = flow_matching_pair(clean, control_noise, sigma)
            velocities = {
                "correct": conditioned_prediction(dit, conditioner, noisy, timestep, None, features, geometry_adapter=geometry, camera=camera, latent_shape=tuple(clean.shape[2:])),
                "shuffled": conditioned_prediction(dit, conditioner, noisy, timestep, None, features, geometry_adapter=geometry, camera=_roll(camera), latent_shape=tuple(clean.shape[2:])),
                "disabled": conditioned_prediction(dit, conditioner, noisy, timestep, None, features),
                "perturbed_fp32": conditioned_prediction(dit, conditioner, noisy, timestep, None, features, geometry_adapter=geometry, camera=_perturb(camera, args.pose_perturbation), latent_shape=tuple(clean.shape[2:])),
            }
            control_rows.append({
                "scene": scene_id,
                "checkpoint_step": args.checkpoint_step,
                "correct_flow_loss": float(flow_matching_loss(velocities["correct"], target)),
                "shuffled_flow_loss": float(flow_matching_loss(velocities["shuffled"], target)),
                "disabled_flow_loss": float(flow_matching_loss(velocities["disabled"], target)),
                "perturbed_fp32_flow_loss": float(flow_matching_loss(velocities["perturbed_fp32"], target)),
                "correct_shuffled_delta": float((velocities["correct"] - velocities["shuffled"]).abs().float().mean()),
                "correct_disabled_delta": float((velocities["correct"] - velocities["disabled"]).abs().float().mean()),
                "correct_perturbed_fp32_delta": float((velocities["correct"] - velocities["perturbed_fp32"]).abs().float().mean()),
            })
            for condition, velocity in velocities.items():
                decoded = vae.decode_multiview(velocity_to_clean(noisy, velocity, sigma))
                one_step_rows.extend(_metric_rows(decoded, hr, metric, scene=scene_id, condition=condition, seed=9000, indices=tuple(args.view_indices), checkpoint_step=args.checkpoint_step))

        for seed in args.inference_seeds:
            primary_condition = "stage1" if geometry is None else "correct"
            primary = sample_with(geometry, camera, seed)
            metric_rows.extend(_metric_rows(primary, hr, metric, scene=scene_id, condition=primary_condition, seed=seed, indices=tuple(args.view_indices), checkpoint_step=args.checkpoint_step))
            outputs = {
                primary_condition: primary,
                "hr": hr,
                "lr": lr,
                "bicubic": bicubic,
                "vae_ceiling": vae_ceiling,
            }
            if geometry is not None and args.include_stage1:
                stage1 = sample_with(None, camera, seed)
                metric_rows.extend(_metric_rows(stage1, hr, metric, scene=scene_id, condition="stage1", seed=seed, indices=tuple(args.view_indices), checkpoint_step=args.checkpoint_step))
                outputs["stage1"] = stage1
            if geometry is not None and args.pure_noise_camera_controls and seed == args.inference_seeds[0]:
                for condition, control_geometry, control_camera in (
                    ("shuffled", geometry, _roll(camera)),
                    ("disabled", None, camera),
                    ("perturbed_fp32", geometry, _perturb(camera, args.pose_perturbation)),
                ):
                    value = sample_with(control_geometry, control_camera, seed)
                    metric_rows.extend(_metric_rows(value, hr, metric, scene=scene_id, condition=condition, seed=seed, indices=tuple(args.view_indices), checkpoint_step=args.checkpoint_step))
                    outputs[condition] = value
            for condition, baseline in (("bicubic", bicubic), ("vae_ceiling", vae_ceiling)):
                metric_rows.extend(_metric_rows(baseline, hr, metric, scene=scene_id, condition=condition, seed=seed, indices=tuple(args.view_indices), checkpoint_step=args.checkpoint_step))
            if args.save_images:
                for condition, value in outputs.items():
                    _save_images(args.output_dir / "images", condition, seed, value, tuple(args.view_indices))

    for path, rows in (
        (args.output_dir / "metric_rows.jsonl", metric_rows),
        (args.output_dir / "one_step_rows.jsonl", one_step_rows),
        (args.output_dir / "camera_controls.jsonl", control_rows),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    _csv(args.output_dir / "metric_rows.csv", metric_rows)
    _csv(args.output_dir / "one_step_rows.csv", one_step_rows)
    _csv(args.output_dir / "camera_controls.csv", control_rows)
    summary = {
        "scene": scene_id,
        "representation": args.representation,
        "checkpoint": None if args.geometry_checkpoint is None else str(args.geometry_checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": args.checkpoint_step,
        "view_indices": list(args.view_indices),
        "inference_seeds": list(args.inference_seeds),
        "sampling": {"steps": args.sampling_steps, "shift": args.sampling_shift},
        "camera_usage": camera_usage_verdict(control_rows),
        "metric_rows": len(metric_rows),
        "one_step_rows": len(one_step_rows),
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
