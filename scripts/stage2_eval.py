#!/usr/bin/env python3
"""Evaluate a Stage 2 checkpoint on train and held-out NeRF Synthetic views."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from rl3dsr.validation.decoded_space import frame_metrics, velocity_to_clean
from rl3dsr.validation.geometry import (
    evaluate_cross_view_reconstruction,
    evaluate_pose_sensitivity,
    evaluate_reprojection,
)
from rl3dsr.validation.stage2_protocol import evaluation_groups


def _camera(sequence, resolution: int, device: torch.device) -> CameraBatch:
    intrinsics, transforms = [], []
    for observation in sequence.observations:
        sx = resolution / observation.width
        sy = resolution / observation.height
        K = torch.from_numpy(np.asarray(observation.K, dtype=np.float32)).clone()
        K[0] *= sx
        K[1] *= sy
        intrinsics.append(K)
        transforms.append(torch.from_numpy(np.asarray(observation.T_world_from_camera, dtype=np.float32)))
    return CameraBatch(
        torch.stack(intrinsics).unsqueeze(0).to(device),
        torch.stack(transforms).unsqueeze(0).to(device),
        (resolution, resolution),
        "multiview",
        0,
    )


def _load_split(scene: Path, split: Split, indices: tuple[int, ...], resolution: int, device: torch.device):
    adapter = NeRFSyntheticAdapter(scene)
    sequence = adapter.index(split)
    selected = tuple(sequence.observations[index] for index in indices)
    images = [adapter.load_rgb(item) for item in selected]
    hr = rgb_images_to_video(images, resolution).to(device)
    camera = _camera(type("Selected", (), {"observations": selected})(), resolution, device)
    lr = Stage1Degradation(scale=4)(hr)
    return sequence.scene_id, hr, lr, camera


def load_depth_manifest(path: Path, scene_id: str, indices: tuple[int, ...], resolution: int, device: torch.device) -> torch.Tensor:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scene_id") != scene_id:
        raise RuntimeError("depth manifest scene_id mismatch")
    scale = payload.get("depth_scale")
    if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise RuntimeError("depth manifest requires a positive finite depth_scale")
    frames = payload.get("frames")
    if not isinstance(frames, dict):
        raise RuntimeError("depth manifest frames must be an object keyed by frame index")
    values = []
    for index in indices:
        raw = frames.get(str(index))
        if not isinstance(raw, str):
            raise RuntimeError(f"depth manifest has no frame {index}")
        path_value = Path(raw)
        if not path_value.is_absolute():
            path_value = path.parent / path_value
        with Image.open(path_value) as image:
            if image.width < 1 or image.height < 1:
                raise RuntimeError(f"invalid depth image: {path_value}")
            channel = np.asarray(image.convert("RGBA"), dtype=np.float32)[..., 0]
        tensor = torch.from_numpy(channel).unsqueeze(0).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(tensor, size=(resolution, resolution), mode="nearest")[0, 0]
        values.append(tensor * float(scale))
    depth = torch.stack(values).unsqueeze(0).to(device=device, dtype=torch.float32)
    if not torch.isfinite(depth).all() or bool((depth < 0).any()):
        raise RuntimeError("decoded depth must be finite and non-negative")
    return depth


def _perceptual(device: torch.device):
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", reduction="mean", normalize=False).to(device).eval()
    except Exception:
        return None
    class PerFrame(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.cat([self.inner(prediction[index:index + 1], target[index:index + 1]).reshape(1) for index in range(prediction.shape[0])])
    return PerFrame(metric).to(device).eval()


def _image_rows(
    value: torch.Tensor,
    target: torch.Tensor,
    condition: str,
    evaluation_group: str,
    dataset_split: str,
    view_indices: tuple[int, ...],
    checkpoint_step: int,
    metric,
) -> list[dict]:
    rows = frame_metrics(value.float().cpu(), target.float().cpu(), perceptual_metric=metric)
    result = []
    for row in rows:
        position = int(row["frame_index"])
        result.append({
            "condition": condition,
            "evaluation_group": evaluation_group,
            "dataset_split": dataset_split,
            "checkpoint_step": checkpoint_step,
            "view_position": position,
            "view_index": view_indices[position],
            **{
                key: float(val)
                for key, val in row.items()
                if key not in {"batch_index", "frame_index"}
            },
        })
    return result


def _upsample_lr(lr: torch.Tensor, resolution: int) -> torch.Tensor:
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


def _camera_roll(camera: CameraBatch) -> CameraBatch:
    order = torch.roll(torch.arange(camera.K.shape[1], device=camera.K.device), shifts=1)
    return CameraBatch(camera.K, camera.T_world_from_camera[:, order], camera.image_size, camera.sequence_kind, camera.reference_index)


def _camera_perturb(camera: CameraBatch, amount: float) -> CameraBatch:
    transform = camera.T_world_from_camera.clone()
    transform[..., 0, 3] += amount
    return CameraBatch(camera.K, transform, camera.image_size, camera.sequence_kind, camera.reference_index)


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--lq-source", type=Path, required=True)
    parser.add_argument("--lq-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path)
    parser.add_argument("--representation", choices=("rre", "plucker", "rre_full"), default="rre")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("selection", "final-test"), required=True)
    parser.add_argument("--seen-view-indices", type=int, nargs=4)
    parser.add_argument("--near-view-indices", type=int, nargs=4)
    parser.add_argument("--far-view-indices", type=int, nargs=4)
    parser.add_argument("--checkpoint-step", type=int, default=0)
    parser.add_argument("--depth-manifest", type=Path)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--pose-perturbation", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    group_specs = evaluation_groups(
        args.phase,
        seen=args.seen_view_indices,
        near=args.near_view_indices,
        far=args.far_view_indices,
    )
    groups = tuple(
        (
            label,
            Split.TRAIN if split == "train" else Split.TEST,
            indices,
        )
        for label, split, indices in group_specs
    )
    if args.geometry_checkpoint is None and args.checkpoint_step != 0:
        raise ValueError("baseline evaluation must use checkpoint step 0")
    if args.geometry_checkpoint is not None and args.checkpoint_step <= 0:
        raise ValueError("geometry evaluation requires a positive checkpoint step")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    projector = load_flashvsr_projector(args.lq_source, args.lq_checkpoint, device=device, dtype=torch.bfloat16)
    conditioner = FrozenLQConditioner(projector, bridge_blocks=(0, 1, 2, 3), bridge_time_conditioning=True).to(device).eval()
    load_adapter_checkpoint(args.parent_checkpoint, conditioner)
    conditioner.bridge.requires_grad_(False)
    vae = WanVAE.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    geometry = None
    if args.geometry_checkpoint is not None:
        geometry = (
            FullRREConditioner()
            if args.representation == "rre_full"
            else GeometryConditioner(
                representation=args.representation,
                hidden_dim=192,
                attention_heads=4,
                blocks=(0, 1, 2, 3),
            )
        ).to(device).eval()
        payload = load_geometry_checkpoint(args.geometry_checkpoint, geometry, expected_parent_checkpoint=str(args.parent_checkpoint.resolve()))
        if int(payload.get("step", -1)) != args.checkpoint_step:
            raise RuntimeError("checkpoint step does not match --checkpoint-step")
    metric = _perceptual(device)
    rows: list[dict] = []
    geometry_rows: list[dict] = []
    pose_rows: list[dict] = []
    evaluated_groups = []
    for evaluation_group, split, indices in groups:
        scene_id, hr, lr, camera = _load_split(args.scene, split, indices, args.resolution, device)
        bicubic = _upsample_lr(lr, args.resolution)
        with torch.inference_mode():
            clean = vae.encode_multiview(hr)
            features = conditioner.multiview_features(lr, conditioning_size=(args.resolution, args.resolution), latent_shape=tuple(clean.shape[2:]))
            sigma = torch.tensor([0.5], device=device)
            noise = torch.randn(clean.shape, generator=torch.Generator(device=device).manual_seed(9000), device=device, dtype=clean.dtype)
            noisy, timestep, flow_target = flow_matching_pair(clean, noise, sigma)
            baseline_velocity = conditioned_prediction(dit, conditioner, noisy, timestep, None, features)
            disabled_velocity = baseline_velocity
            if geometry is None:
                correct_velocity = baseline_velocity
                repeat_velocity = baseline_velocity
            else:
                correct_velocity = conditioned_prediction(dit, conditioner, noisy, timestep, None, features, geometry_adapter=geometry, camera=camera, latent_shape=tuple(clean.shape[2:]))
                jitter = max(args.pose_perturbation * 0.25, 0.02)
                repeat_velocity = conditioned_prediction(dit, conditioner, noisy, timestep, None, features, geometry_adapter=geometry, camera=_camera_perturb(camera, jitter), latent_shape=tuple(clean.shape[2:]))
            shuffled_velocity = baseline_velocity if geometry is None else conditioned_prediction(dit, conditioner, noisy, timestep, None, features, geometry_adapter=geometry, camera=_camera_roll(camera), latent_shape=tuple(clean.shape[2:]))
            perturbed_velocity = baseline_velocity if geometry is None else conditioned_prediction(dit, conditioner, noisy, timestep, None, features, geometry_adapter=geometry, camera=_camera_perturb(camera, args.pose_perturbation), latent_shape=tuple(clean.shape[2:]))
            predictions = {
                "baseline": vae.decode_multiview(velocity_to_clean(noisy, baseline_velocity, sigma)),
                "disabled": vae.decode_multiview(velocity_to_clean(noisy, disabled_velocity, sigma)),
                "correct": vae.decode_multiview(velocity_to_clean(noisy, correct_velocity, sigma)),
                "shuffled": vae.decode_multiview(velocity_to_clean(noisy, shuffled_velocity, sigma)),
                "perturbed": vae.decode_multiview(velocity_to_clean(noisy, perturbed_velocity, sigma)),
            }
        for condition, value in predictions.items():
            rows.extend(
                {"scene_id": scene_id, **row}
                for row in _image_rows(
                    value,
                    hr,
                    condition,
                    evaluation_group,
                    split.value,
                    indices,
                    args.checkpoint_step,
                    metric,
                )
            )
        pose = evaluate_pose_sensitivity(correct_velocity, shuffled_velocity, disabled_velocity, perturbed_velocity, repeat=repeat_velocity)
        pose_rows.append({
            "scene_id": scene_id,
            "evaluation_group": evaluation_group,
            "dataset_split": split.value,
            "checkpoint_step": args.checkpoint_step,
            "correct_flow_loss": float(flow_matching_loss(correct_velocity, flow_target)),
            "shuffled_flow_loss": float(flow_matching_loss(shuffled_velocity, flow_target)),
            "baseline_flow_loss": float(flow_matching_loss(baseline_velocity, flow_target)),
            **pose,
        })
        if evaluation_group == "far-held-out" and args.depth_manifest is not None:
            depth = load_depth_manifest(args.depth_manifest, scene_id, indices, args.resolution, device)
            for condition in ("baseline", "correct"):
                pair_rows = evaluate_reprojection(predictions[condition], hr, depth, camera.K, camera.T_world_from_camera)
                geometry_rows.extend({
                    "scene_id": scene_id,
                    "evaluation_group": evaluation_group,
                    "dataset_split": split.value,
                    "checkpoint_step": args.checkpoint_step,
                    "condition": condition,
                    "metric_group": "reprojection",
                    **row,
                } for row in pair_rows)
                fused_rows = evaluate_cross_view_reconstruction(predictions[condition], hr, depth, camera.K, camera.T_world_from_camera)
                geometry_rows.extend({
                    "scene_id": scene_id,
                    "evaluation_group": evaluation_group,
                    "dataset_split": split.value,
                    "checkpoint_step": args.checkpoint_step,
                    "condition": condition,
                    "metric_group": "cross_view_reconstruction",
                    **row,
                } for row in fused_rows)
        torch.save(
            {
                "scene_id": scene_id,
                "evaluation_group": evaluation_group,
                "dataset_split": split.value,
                "checkpoint_step": args.checkpoint_step,
                "view_indices": list(indices),
                "hr": hr.cpu(),
                "bicubic": bicubic.cpu(),
                **{key: value.cpu() for key, value in predictions.items()},
            },
            args.output_dir / f"qualitative_{evaluation_group}.pt",
        )
        evaluated_groups.append({
            "evaluation_group": evaluation_group,
            "dataset_split": split.value,
            "view_indices": list(indices),
        })
    with (args.output_dir / "evaluation_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    _csv(args.output_dir / "evaluation_rows.csv", rows)
    _csv(args.output_dir / "geometry_rows.csv", geometry_rows)
    _csv(args.output_dir / "pose_sensitivity.csv", pose_rows)
    summary = {
        "scene": str(args.scene.resolve()),
        "phase": args.phase,
        "checkpoint_step": args.checkpoint_step,
        "groups": evaluated_groups,
        "reprojection_available": bool(geometry_rows),
        "image_rows": len(rows),
        "geometry_rows": len(geometry_rows),
        "pose_rows": len(pose_rows),
    }
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
