"""Decoded-space evaluation utilities for Stage 1 LR conditioning.

All RGB tensors use the project layout ``[B,3,S,H,W]`` and range ``[-1,1]``.
The helpers are deliberately independent of Wan checkpoint loading so their
mathematics can be validated with lightweight CPU tests.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _sigma_for(value: Tensor, sigma: float | Tensor, *, name: str = "sigma") -> Tensor:
    sigma_tensor = torch.as_tensor(sigma, device=value.device, dtype=value.dtype)
    if sigma_tensor.ndim == 0:
        return sigma_tensor
    if sigma_tensor.ndim != 1 or sigma_tensor.shape[0] != value.shape[0]:
        raise ValueError(f"{name} must be scalar or have one value per batch item")
    return sigma_tensor.reshape(value.shape[0], *([1] * (value.ndim - 1)))


def velocity_to_clean(noisy: Tensor, velocity: Tensor, sigma: float | Tensor) -> Tensor:
    """Recover the clean latent from a flow-matching velocity prediction."""

    if noisy.shape != velocity.shape:
        raise ValueError("noisy and velocity must have the same shape")
    sigma_tensor = _sigma_for(noisy, sigma)
    return noisy - sigma_tensor * velocity


def shifted_flow_sigmas(
    num_steps: int,
    *,
    shift: float = 5.0,
    sigma_max: float = 1.0,
    sigma_min: float = 0.0,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return Wan FlowMatch sigmas plus the explicit final sigma zero/end point."""

    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if shift <= 0:
        raise ValueError("shift must be positive")
    if not 0 <= sigma_min < sigma_max <= 1:
        raise ValueError("sigmas must satisfy 0 <= sigma_min < sigma_max <= 1")
    base = torch.linspace(sigma_max, sigma_min, num_steps + 1, device=device, dtype=dtype)
    return shift * base / (1 + (shift - 1) * base)


def flow_euler_step(
    sample: Tensor,
    velocity: Tensor,
    *,
    sigma: float | Tensor,
    next_sigma: float | Tensor,
) -> Tensor:
    """Apply the Euler update used by Wan's FlowMatchScheduler."""

    if sample.shape != velocity.shape:
        raise ValueError("sample and velocity must have the same shape")
    sigma_tensor = _sigma_for(sample, sigma)
    next_sigma_tensor = _sigma_for(sample, next_sigma, name="next_sigma")
    return sample + velocity * (next_sigma_tensor - sigma_tensor)


def _validate_rgb_pair(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim != 5 or prediction.shape[1] != 3:
        raise ValueError("RGB tensors must have shape [B,3,S,H,W]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("RGB metrics require finite tensors")


def _unit_rgb(value: Tensor) -> Tensor:
    return value.float().clamp(-1, 1).add(1).mul(0.5)


def frame_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    perceptual_metric: torch.nn.Module | None = None,
) -> list[dict[str, float | int]]:
    """Compute clipped RGB MAE, PSNR, SSIM and optional LPIPS per view."""

    _validate_rgb_pair(prediction, target)
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:  # pragma: no cover - validated experiment dependency
        raise RuntimeError("frame_metrics requires scikit-image") from error

    predicted = _unit_rgb(prediction).detach().cpu()
    reference = _unit_rgb(target).detach().cpu()
    perceptual_values: list[float] | None = None
    if perceptual_metric is not None:
        batch, _, frames, height, width = prediction.shape
        pred_lpips = prediction.float().clamp(-1, 1).permute(0, 2, 1, 3, 4).reshape(
            batch * frames, 3, height, width
        )
        target_lpips = target.float().clamp(-1, 1).permute(0, 2, 1, 3, 4).reshape(
            batch * frames, 3, height, width
        )
        parameter = next(perceptual_metric.parameters(), None)
        metric_device = parameter.device if parameter is not None else pred_lpips.device
        with torch.inference_mode():
            scores = perceptual_metric(
                pred_lpips.to(metric_device), target_lpips.to(metric_device)
            ).detach().float().cpu().reshape(-1)
        if scores.numel() != batch * frames:
            raise RuntimeError("perceptual metric must return one value per frame/view")
        perceptual_values = [float(value) for value in scores]
    rows: list[dict[str, float | int]] = []
    for batch_index in range(predicted.shape[0]):
        for frame_index in range(predicted.shape[2]):
            pred = predicted[batch_index, :, frame_index].permute(1, 2, 0).numpy()
            ref = reference[batch_index, :, frame_index].permute(1, 2, 0).numpy()
            difference = pred.astype(np.float64) - ref.astype(np.float64)
            mae = float(np.mean(np.abs(difference)))
            mse = float(np.mean(np.square(difference)))
            psnr = math.inf if mse == 0 else float(10 * math.log10(1 / mse))
            ssim = float(structural_similarity(ref, pred, data_range=1.0, channel_axis=-1))
            row: dict[str, float | int] = {
                    "batch_index": batch_index,
                    "frame_index": frame_index,
                    "mae": mae,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            if perceptual_values is not None:
                row["lpips"] = perceptual_values[batch_index * predicted.shape[2] + frame_index]
            rows.append(row)
    return rows


def temporal_delta_mae(prediction: Tensor, target: Tensor) -> float:
    """Mean RGB error between adjacent-frame changes after display-range clipping."""

    _validate_rgb_pair(prediction, target)
    if prediction.shape[2] < 2:
        raise ValueError("temporal_delta_mae requires at least two frames")
    predicted = _unit_rgb(prediction)
    reference = _unit_rgb(target)
    predicted_delta = predicted[:, :, 1:] - predicted[:, :, :-1]
    reference_delta = reference[:, :, 1:] - reference[:, :, :-1]
    return float((predicted_delta - reference_delta).abs().mean())


def rgb_range_stats(value: Tensor) -> dict[str, Any]:
    """Describe finite values and display-range violations before clipping."""

    detached = value.detach().float().cpu()
    finite_mask = torch.isfinite(detached)
    finite = detached[finite_mask]
    out_of_range = (finite < -1) | (finite > 1)
    finite_count = int(finite.numel())
    return {
        "numel": int(detached.numel()),
        "finite_count": finite_count,
        "nonfinite_count": int((~finite_mask).sum()),
        "out_of_range_count": int(out_of_range.sum()),
        "out_of_range_ratio": float(out_of_range.float().mean()) if finite_count else math.nan,
        "finite_min": float(finite.min()) if finite_count else math.nan,
        "finite_max": float(finite.max()) if finite_count else math.nan,
        "finite_abs_max": float(finite.abs().max()) if finite_count else math.nan,
    }


def safe_artifact_name(value: str) -> str:
    """Convert an experiment item identifier into a deterministic portable stem."""

    stem = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return stem or "item"


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("metric distribution cannot be empty")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_condition_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-frame/view RGB rows for one mode, sigma and condition."""

    if not rows:
        raise ValueError("condition rows cannot be empty")
    metrics = {name: _distribution([float(row[name]) for row in rows]) for name in ("mae", "psnr", "ssim")}
    worst = max(rows, key=lambda row: float(row["mae"]))
    return {
        "count": len(rows),
        **metrics,
        "worst_mae": {
            "item": str(worst["item"]),
            "position": int(worst["position"]),
            "value": float(worst["mae"]),
        },
    }


def paired_condition_summary(rows: list[dict[str, Any]], *, control: str) -> dict[str, Any]:
    """Compare correct LR with one control using exact item/position pairs."""

    def indexed(condition: str) -> dict[tuple[str, int], dict[str, Any]]:
        return {
            (str(row["item"]), int(row["position"])): row
            for row in rows
            if row["condition"] == condition
        }

    correct = indexed("correct")
    comparison = indexed(control)
    if not correct or correct.keys() != comparison.keys():
        raise ValueError(f"correct and {control} rows must have identical paired keys")

    psnr_delta: list[float] = []
    ssim_delta: list[float] = []
    mae_improvement: list[float] = []
    wins = {"mae": 0, "psnr": 0, "ssim": 0, "all_three": 0}
    for key in sorted(correct):
        expected = correct[key]
        baseline = comparison[key]
        mae_win = float(expected["mae"]) < float(baseline["mae"])
        psnr_win = float(expected["psnr"]) > float(baseline["psnr"])
        ssim_win = float(expected["ssim"]) > float(baseline["ssim"])
        wins["mae"] += int(mae_win)
        wins["psnr"] += int(psnr_win)
        wins["ssim"] += int(ssim_win)
        wins["all_three"] += int(mae_win and psnr_win and ssim_win)
        psnr_delta.append(float(expected["psnr"]) - float(baseline["psnr"]))
        ssim_delta.append(float(expected["ssim"]) - float(baseline["ssim"]))
        mae_improvement.append(float(baseline["mae"]) - float(expected["mae"]))
    count = len(correct)
    return {
        "control": control,
        "count": count,
        "psnr_delta_correct_minus_control": _distribution(psnr_delta),
        "ssim_delta_correct_minus_control": _distribution(ssim_delta),
        "mae_improvement_control_minus_correct": _distribution(mae_improvement),
        "correct_win_count": wins,
        "correct_win_fraction": {name: value / count for name, value in wins.items()},
    }


def strict_3d_quality_verdict(rows: list[dict[str, Any]], *, development: bool = False) -> dict[str, Any]:
    """Apply the fixed pure-noise 3D overfit gates to per-view metric rows."""

    required = {"correct", "bicubic", "shuffled", "disabled"}
    indexed: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        condition = str(row["condition"])
        if condition not in required:
            continue
        key = (int(row["seed"]), str(row["item"]), int(row["position"]))
        if any(not math.isfinite(float(row[metric])) for metric in ("psnr", "ssim", "lpips")):
            raise ValueError("quality metrics must be finite")
        group = indexed.setdefault(key, {})
        if condition in group:
            raise ValueError(f"duplicate quality row: {key} {condition}")
        group[condition] = row
    if not indexed or any(set(group) != required for group in indexed.values()):
        raise ValueError("quality rows must contain paired correct/bicubic/shuffled/disabled conditions")

    def mean(condition: str, metric: str) -> float:
        return statistics.fmean(float(group[condition][metric]) for group in indexed.values())

    means = {
        condition: {metric: mean(condition, metric) for metric in ("psnr", "ssim", "lpips")}
        for condition in sorted(required)
    }
    bicubic_lpips_reduction = (
        means["bicubic"]["lpips"] - means["correct"]["lpips"]
    ) / means["bicubic"]["lpips"]
    shuffled_lpips_reduction = (
        means["shuffled"]["lpips"] - means["correct"]["lpips"]
    ) / means["shuffled"]["lpips"]

    def all_metrics_win(group: dict[str, dict[str, Any]], control: str) -> bool:
        return (
            float(group["correct"]["psnr"]) > float(group[control]["psnr"])
            and float(group["correct"]["ssim"]) > float(group[control]["ssim"])
            and float(group["correct"]["lpips"]) < float(group[control]["lpips"])
        )

    wins_bicubic = sum(all_metrics_win(group, "bicubic") for group in indexed.values())
    wins_shuffled = sum(all_metrics_win(group, "shuffled") for group in indexed.values())
    seeds = sorted({key[0] for key in indexed})
    if len({key[1] for key in indexed}) != 1 or any(
        {key[2] for key in indexed if key[0] == seed} != {0, 1, 2, 3}
        for seed in seeds
    ):
        raise ValueError("quality evaluation requires one sample and exactly four views per seed")

    def seed_wins(seed: int, control: str) -> bool:
        groups = [group for key, group in indexed.items() if key[0] == seed]
        return all(
            statistics.fmean(float(group["correct"][metric]) for group in groups)
            > statistics.fmean(float(group[control][metric]) for group in groups)
            if metric != "lpips"
            else statistics.fmean(float(group["correct"][metric]) for group in groups)
            < statistics.fmean(float(group[control][metric]) for group in groups)
            for metric in ("psnr", "ssim", "lpips")
        )

    checks = {
        "one_seed" if development else "four_seeds": len(seeds) == (1 if development else 4),
        "bicubic_each_seed": all(seed_wins(seed, "bicubic") for seed in seeds),
        "shuffled_each_seed": all(seed_wins(seed, "shuffled") for seed in seeds),
        "bicubic_margin": (
            means["correct"]["psnr"] - means["bicubic"]["psnr"] >= 0.25
            and means["correct"]["ssim"] - means["bicubic"]["ssim"] >= 0.005
            and bicubic_lpips_reduction >= 0.05
        ),
        "shuffled_margin": (
            means["correct"]["psnr"] - means["shuffled"]["psnr"] >= 1.0
            and means["correct"]["ssim"] - means["shuffled"]["ssim"] >= 0.02
            and shuffled_lpips_reduction >= 0.10
        ),
        "disabled_better": (
            means["correct"]["psnr"] > means["disabled"]["psnr"]
            and means["correct"]["ssim"] > means["disabled"]["ssim"]
            and means["correct"]["lpips"] < means["disabled"]["lpips"]
        ),
        "paired_wins": wins_bicubic >= (3 if development else 12) and wins_shuffled >= (3 if development else 12),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "means": means,
        "margins": {
            "bicubic": {
                "psnr": means["correct"]["psnr"] - means["bicubic"]["psnr"],
                "ssim": means["correct"]["ssim"] - means["bicubic"]["ssim"],
                "lpips_relative_reduction": bicubic_lpips_reduction,
            },
            "shuffled": {
                "psnr": means["correct"]["psnr"] - means["shuffled"]["psnr"],
                "ssim": means["correct"]["ssim"] - means["shuffled"]["ssim"],
                "lpips_relative_reduction": shuffled_lpips_reduction,
            },
        },
        "all_metric_wins_vs_bicubic": wins_bicubic,
        "all_metric_wins_vs_shuffled": wins_shuffled,
        "row_count": len(indexed),
        "seeds": seeds,
    }
