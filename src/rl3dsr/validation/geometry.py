"""Deterministic geometry metrics for Stage 2 evaluation.

The model never receives depth from this module. Depth is an evaluation-only
input used to warp RGB predictions between canonical camera views.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def _check_rgb(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 5 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B,3,S,H,W]")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating RGB")


def _check_camera(K: Tensor, T: Tensor, batch: int, views: int, height: int, width: int) -> None:
    if K.shape != (batch, views, 3, 3) or T.shape != (batch, views, 4, 4):
        raise ValueError("K/T must have shapes [B,S,3,3] and [B,S,4,4]")
    if not torch.is_floating_point(K) or not torch.is_floating_point(T):
        raise ValueError("K/T must be floating point")
    if not torch.isfinite(K).all() or not torch.isfinite(T).all():
        raise ValueError("K/T must be finite")
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")


def _check_depth(depth: Tensor, batch: int, views: int, height: int, width: int) -> None:
    if depth.shape != (batch, views, height, width):
        raise ValueError("depth must have shape [B,S,H,W] matching RGB")
    if not torch.is_floating_point(depth) or not torch.isfinite(depth).all():
        raise ValueError("depth must be finite floating point")


def _pixel_grid(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((u.reshape(-1), v.reshape(-1), torch.ones(height * width, device=device, dtype=dtype)), dim=1)


def _forward_splat(
    source: Tensor,
    depth: Tensor,
    source_K: Tensor,
    source_T: Tensor,
    target_K: Tensor,
    target_T: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Warp one RGB image with nearest-pixel forward splatting and z-buffering."""

    if source.ndim != 3 or source.shape[0] != 3:
        raise ValueError("source must have shape [3,H,W]")
    height, width = source.shape[-2:]
    if depth.shape != (height, width):
        raise ValueError("depth must match source spatial dimensions")
    dtype = source.dtype
    device = source.device
    pixels = _pixel_grid(height, width, device=device, dtype=dtype)
    valid_source = depth.reshape(-1) > 1e-8
    inv_source = torch.linalg.inv(source_K.to(device=device, dtype=dtype))
    inv_target = torch.linalg.inv(target_T.to(device=device, dtype=dtype))
    world_from_camera = source_T.to(device=device, dtype=dtype)
    rays_camera = (inv_source @ pixels.T).T
    points_camera = rays_camera * depth.reshape(-1, 1)
    homogeneous = torch.cat((points_camera, torch.ones(points_camera.shape[0], 1, device=device, dtype=dtype)), dim=1)
    points_world = (world_from_camera @ homogeneous.T).T
    points_target = (inv_target @ points_world.T).T[:, :3]
    projected = (target_K.to(device=device, dtype=dtype) @ points_target.T).T
    z = points_target[:, 2]
    u = projected[:, 0] / z.clamp_min(1e-8)
    v = projected[:, 1] / z.clamp_min(1e-8)
    ui = torch.round(u).long()
    vi = torch.round(v).long()
    valid = valid_source & (z > 1e-8) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    output = torch.zeros_like(source)
    mask = torch.zeros(height * width, device=device, dtype=torch.bool)
    zbuffer = torch.full((height * width,), float("inf"), device=device, dtype=dtype)
    if not bool(valid.any()):
        return output, mask.reshape(height, width), zbuffer.reshape(height, width)
    target_index = vi[valid] * width + ui[valid]
    source_index = torch.arange(height * width, device=device)[valid]
    depths = z[valid]
    order = torch.argsort(depths)
    target_sorted = target_index[order]
    keep = torch.ones_like(target_sorted, dtype=torch.bool)
    keep[1:] = target_sorted[1:] != target_sorted[:-1]
    chosen = order[keep]
    selected_target = target_index[chosen]
    selected_source = source_index[chosen]
    zbuffer[selected_target] = depths[chosen]
    output.reshape(3, -1)[:, selected_target] = source.reshape(3, -1)[:, selected_source]
    mask[selected_target] = True
    return output, mask.reshape(height, width), zbuffer.reshape(height, width)


def reproject_rgb(
    source: Tensor,
    depth: Tensor,
    source_K: Tensor,
    source_T: Tensor,
    target_K: Tensor,
    target_T: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Public single-image reprojection primitive."""

    return _forward_splat(source, depth, source_K, source_T, target_K, target_T)


def _masked_mae(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    if not bool(mask.any()):
        return math.nan
    return float((prediction[:, mask] - target[:, mask]).abs().mean().detach().cpu())


def _masked_ssim(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    if not bool(mask.any()):
        return math.nan
    ys, xs = torch.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return math.nan
    pred = prediction[:, y0:y1, x0:x1].detach().float().cpu().clamp(-1, 1).add(1).mul(0.5)
    ref = target[:, y0:y1, x0:x1].detach().float().cpu().clamp(-1, 1).add(1).mul(0.5)
    side = min(y1 - y0, x1 - x0)
    if side < 3:
        return math.nan
    win_size = min(7, side if side % 2 else side - 1)
    value = structural_similarity(
        ref.permute(1, 2, 0).numpy(),
        pred.permute(1, 2, 0).numpy(),
        data_range=1.0,
        channel_axis=-1,
        win_size=win_size,
    )
    return float(value)


def _pair_rows(prediction: Tensor, target: Tensor, depth: Tensor, K: Tensor, T: Tensor) -> list[dict[str, Any]]:
    _check_rgb(prediction, "prediction")
    _check_rgb(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    batch, _, views, height, width = prediction.shape
    _check_camera(K, T, batch, views, height, width)
    _check_depth(depth, batch, views, height, width)
    rows: list[dict[str, Any]] = []
    for batch_index in range(batch):
        for source_index in range(views):
            for target_index in range(views):
                if source_index == target_index:
                    continue
                warped, valid, _ = _forward_splat(prediction[batch_index, :, source_index], depth[batch_index, source_index], K[batch_index, source_index], T[batch_index, source_index], K[batch_index, target_index], T[batch_index, target_index])
                gt_warped, gt_valid, _ = _forward_splat(target[batch_index, :, source_index], depth[batch_index, source_index], K[batch_index, source_index], T[batch_index, source_index], K[batch_index, target_index], T[batch_index, target_index])
                valid = valid & gt_valid
                target_frame = target[batch_index, :, target_index]
                pred_mae = _masked_mae(warped, target_frame, valid)
                ceiling_mae = _masked_mae(gt_warped, target_frame, valid)
                rows.append({
                    "batch_index": batch_index,
                    "source_index": source_index,
                    "target_index": target_index,
                    "coverage": float(valid.float().mean().detach().cpu()),
                    "photometric_mae": pred_mae,
                    "ssim": _masked_ssim(warped, target_frame, valid),
                    "gt_ceiling_mae": ceiling_mae,
                    "normalized_error": pred_mae / max(ceiling_mae, 1e-6) if math.isfinite(pred_mae) else math.nan,
                })
    return rows


def evaluate_reprojection(prediction: Tensor, target: Tensor, depth: Tensor, K: Tensor, T_world_from_camera: Tensor) -> list[dict[str, Any]]:
    """Return pairwise predicted-to-GT reprojection rows and GT ceiling rows."""

    return _pair_rows(prediction, target, depth, K, T_world_from_camera)


def evaluate_cross_view_reconstruction(prediction: Tensor, target: Tensor, depth: Tensor, K: Tensor, T_world_from_camera: Tensor) -> list[dict[str, Any]]:
    """Reconstruct each target from all other views using median fusion."""

    _check_rgb(prediction, "prediction")
    _check_rgb(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    batch, _, views, height, width = prediction.shape
    _check_camera(K, T_world_from_camera, batch, views, height, width)
    _check_depth(depth, batch, views, height, width)
    rows: list[dict[str, Any]] = []
    for batch_index in range(batch):
        for target_index in range(views):
            candidates: list[Tensor] = []
            masks: list[Tensor] = []
            for source_index in range(views):
                if source_index == target_index:
                    continue
                warped, valid, _ = _forward_splat(prediction[batch_index, :, source_index], depth[batch_index, source_index], K[batch_index, source_index], T_world_from_camera[batch_index, source_index], K[batch_index, target_index], T_world_from_camera[batch_index, target_index])
                candidates.append(warped)
                masks.append(valid)
            stack = torch.stack(candidates)
            mask_stack = torch.stack(masks)
            valid_count = mask_stack.sum(dim=0)
            valid = valid_count > 0
            masked_stack = stack.masked_fill(~mask_stack[:, None], float("nan"))
            fused = torch.nanmedian(masked_stack, dim=0).values
            fused = torch.nan_to_num(fused, nan=0.0)
            target_frame = target[batch_index, :, target_index]
            mae = _masked_mae(fused, target_frame, valid)
            rows.append({
                "batch_index": batch_index,
                "target_index": target_index,
                "source_count": views - 1,
                "coverage": float(valid.float().mean().detach().cpu()),
                "photometric_mae": mae,
                "psnr": float(10.0 * torch.log10(torch.tensor(1.0) / max(mae * mae, 1e-12))) if math.isfinite(mae) else math.nan,
                "ssim": _masked_ssim(fused, target_frame, valid),
            })
    return rows


def evaluate_pose_sensitivity(correct: Tensor, shuffled: Tensor, disabled: Tensor, perturbed: Tensor, *, repeat: Tensor | None = None) -> dict[str, float]:
    """Summarize fixed-noise camera interventions in latent/decoded space."""

    tensors = {"correct": correct, "shuffled": shuffled, "disabled": disabled, "perturbed": perturbed}
    if repeat is not None:
        tensors["repeat"] = repeat
    shape = correct.shape
    if any(value.shape != shape for value in tensors.values()):
        raise ValueError("pose sensitivity tensors must have identical shapes")
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("pose sensitivity tensors must be finite")
    baseline_jitter = float((repeat - correct).float().abs().mean().cpu()) if repeat is not None else 0.0
    intervention = float((perturbed - correct).float().abs().mean().cpu())
    shuffled_delta = float((shuffled - correct).float().abs().mean().cpu())
    disabled_delta = float((disabled - correct).float().abs().mean().cpu())
    return {
        "correct_shuffled_delta": shuffled_delta,
        "correct_disabled_delta": disabled_delta,
        "pose_perturbation_delta": intervention,
        "baseline_repeat_jitter": baseline_jitter,
        "pose_response_ratio": intervention / max(baseline_jitter, 1e-12),
        "shuffled_response_ratio": shuffled_delta / max(baseline_jitter, 1e-12),
    }


def summarize_geometry_rows(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Return descriptive mean/median/min/max for one finite metric column."""

    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    if not values:
        return {"count": 0, "mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    value = torch.tensor(values, dtype=torch.float64)
    return {"count": float(len(values)), "mean": float(value.mean()), "median": float(value.median()), "min": float(value.min()), "max": float(value.max())}
