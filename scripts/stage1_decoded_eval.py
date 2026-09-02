#!/usr/bin/env python3
"""Real Wan decoded-space evaluation for trained Stage 1 LR adapters."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch import Tensor

from rl3dsr.models.wan import WanDiT, WanVAE
from rl3dsr.models.wan.lq_conditioning import (
    FrozenLQConditioner,
    conditioned_prediction,
    Stage1Degradation,
    derange_multiview_lr,
    load_adapter_checkpoint,
    load_flashvsr_projector,
)
from rl3dsr.models.wan.sampling import FlowSamplingConfig, sample_conditioned_flow
from rl3dsr.validation.decoded_space import (
    flow_euler_step,
    frame_metrics,
    paired_condition_summary,
    rgb_range_stats,
    safe_artifact_name,
    shifted_flow_sigmas,
    strict_3d_quality_verdict,
    summarize_condition_rows,
    temporal_delta_mae,
    velocity_to_clean,
)
from stage1_experiment import EXPECTED_LQ_SHA256, ExperimentConfig, _prepare_rgb, _seed_all

CONDITIONS = ("correct", "shuffled", "neutral", "disabled")
LABELS = {
    "correct": "Correct LR",
    "shuffled": "Shuffled LR",
    "neutral": "Neutral LR",
    "disabled": "Adapter disabled",
}


@dataclass(slots=True)
class EvalItem:
    item_id: str
    hr: Tensor
    lr: Tensor
    clean: Tensor
    features: Tensor
    shuffled_features: Tensor | None
    neutral_features: Tensor
    bicubic: Tensor
    vae_ceiling: Tensor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"format_version", "bridge", "config", "experiment"}
    if set(payload) != expected:
        raise RuntimeError(f"unexpected adapter checkpoint payload: {path}")
    return {"config": payload["config"], "experiment": payload["experiment"]}


def upsample_lr(lr: Tensor, size: tuple[int, int]) -> Tensor:
    b, c, sequence, h, w = lr.shape
    flat = lr.permute(0, 2, 1, 3, 4).reshape(b * sequence, c, h, w)
    value = F.interpolate(flat, size=size, mode="bicubic", align_corners=False, antialias=True)
    return value.reshape(b, sequence, c, *size).permute(0, 2, 1, 3, 4).contiguous()


def decode(vae: WanVAE, kind: str, latents: Tensor) -> Tensor:
    value = vae.decode_multiview(latents) if kind == "3d" else vae.decode_video(latents)
    return value.detach().float().cpu()


def cache_items(
    config: ExperimentConfig,
    scene: Path,
    vae: WanVAE,
    conditioner: FrozenLQConditioner,
    device: torch.device,
) -> list[EvalItem]:
    degradation = Stage1Degradation(scale=config.scale)
    result = []
    for item_id, hr in _prepare_rgb(config, scene):
        lr = degradation(hr)
        with torch.inference_mode():
            hr_device = hr.to(device)
            if config.kind == "3d":
                clean = vae.encode_multiview(hr_device)
                features = conditioner.multiview_features(
                    lr.to(device),
                    conditioning_size=(config.hr_resolution, config.hr_resolution),
                    latent_shape=tuple(clean.shape[2:]),
                )
                shuffled_lr, shuffled_indices = derange_multiview_lr(lr.to(device))
                shuffled = conditioner.multiview_features(
                    shuffled_lr,
                    conditioning_size=(config.hr_resolution, config.hr_resolution),
                    latent_shape=tuple(clean.shape[2:]),
                )
                neutral = conditioner.multiview_features(
                    torch.zeros_like(lr).to(device),
                    conditioning_size=(config.hr_resolution, config.hr_resolution),
                    latent_shape=tuple(clean.shape[2:]),
                )
            else:
                clean = vae.encode_video(hr_device)
                shuffled = None
                shuffled_indices = None
                features = conditioner.video_features(
                    lr.to(device),
                    conditioning_size=(config.hr_resolution, config.hr_resolution),
                    latent_shape=tuple(clean.shape[2:]),
                )
                neutral = conditioner.video_features(
                    torch.zeros_like(lr).to(device),
                    conditioning_size=(config.hr_resolution, config.hr_resolution),
                    latent_shape=tuple(clean.shape[2:]),
                )
            ceiling = decode(vae, config.kind, clean)
        if ceiling.shape != hr.shape:
            raise RuntimeError(f"VAE ceiling shape {tuple(ceiling.shape)} != HR {tuple(hr.shape)}")
        result.append(
            EvalItem(
                item_id=item_id,
                hr=hr.detach().float().cpu(),
                lr=lr.detach().float().cpu(),
                clean=clean.detach().cpu(),
                features=features.detach().cpu(),
                shuffled_features=None if shuffled is None else shuffled.detach().cpu(),
                neutral_features=neutral.detach().cpu(),
                bicubic=upsample_lr(lr, (config.hr_resolution, config.hr_resolution)).float().cpu(),
                vae_ceiling=ceiling,
            )
        )
        print(json.dumps({
            "event": "decoded_cache", "kind": config.kind, "item": item_id,
            "hr": list(hr.shape), "lr": list(lr.shape),
            "latent": list(clean.shape), "features": list(features.shape),
            "shuffled_view_indices": shuffled_indices,
        }), flush=True)
    return result


def condition_features(item: EvalItem, shuffled: EvalItem, condition: str) -> Tensor | None:
    if condition == "correct":
        return item.features
    if condition == "shuffled":
        return item.shuffled_features if item.shuffled_features is not None else shuffled.features
    if condition == "neutral":
        return item.neutral_features
    if condition == "disabled":
        return None
    raise ValueError(condition)


def predict(
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    sample: Tensor,
    sigma: float,
    context: Tensor,
    features: Tensor | None,
) -> Tensor:
    timestep = torch.tensor([1000.0 * sigma], device=dit.device, dtype=torch.float32)
    return conditioned_prediction(dit, conditioner, sample, timestep, context, features)


def metric_rows(
    decoded: Tensor,
    item: EvalItem,
    *,
    kind: str,
    mode: str,
    sigma: float | None,
    condition: str,
    seed: int | None = None,
    perceptual_metric: torch.nn.Module | None = None,
) -> list[dict[str, Any]]:
    rows = frame_metrics(decoded, item.hr, perceptual_metric=perceptual_metric)
    for row in rows:
        row.update({
            "kind": kind, "mode": mode, "sigma": sigma, "condition": condition,
            "item": item.item_id, "position": int(row.pop("frame_index")),
        })
        if seed is not None:
            row["seed"] = seed
    return rows


def evaluate_one_step(
    kind: str,
    items: list[EvalItem],
    vae: WanVAE,
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    context: Tensor,
    sigmas: tuple[float, ...],
) -> tuple[list, list, list, dict]:
    metrics, ranges, temporal, visuals = [], [], [], {}
    with torch.inference_mode():
        for item_index, item in enumerate(items):
            shuffled = items[(item_index + 1) % len(items)]
            clean = item.clean.to(dit.device)
            for sigma_index, sigma in enumerate(sigmas):
                noise_seed = 201 + item_index * 100 + sigma_index
                generator = torch.Generator(device=dit.device).manual_seed(noise_seed)
                noise = torch.randn(clean.shape, generator=generator, device=dit.device, dtype=clean.dtype)
                noisy = (1 - sigma) * clean + sigma * noise
                for condition in CONDITIONS:
                    features = condition_features(item, shuffled, condition)
                    velocity = predict(dit, conditioner, noisy, sigma, context, features)
                    estimate = velocity_to_clean(noisy, velocity, sigma)
                    value = decode(vae, kind, estimate)
                    if value.shape != item.hr.shape or not torch.isfinite(value).all():
                        raise RuntimeError(f"invalid decoded output: {kind}/{item.item_id}/{sigma}/{condition}")
                    metrics.extend(metric_rows(
                        value, item, kind=kind, mode="one_step", sigma=sigma, condition=condition
                    ))
                    ranges.append({
                        "kind": kind, "mode": "one_step", "sigma": sigma,
                        "condition": condition, "item": item.item_id, **rgb_range_stats(value),
                    })
                    if kind == "4d":
                        temporal.append({
                            "kind": kind, "mode": "one_step", "sigma": sigma,
                            "condition": condition, "item": item.item_id,
                            "temporal_delta_mae": temporal_delta_mae(value, item.hr),
                        })
                    if sigma in (0.5, 0.8, 0.95):
                        visuals[("one_step", item_index, sigma, condition)] = value
                print(json.dumps({
                    "event": "decoded_one_step", "kind": kind, "item": item.item_id,
                    "sigma": sigma, "noise_seed": noise_seed,
                }), flush=True)
    return metrics, ranges, temporal, visuals



def evaluate_trajectory(
    kind: str,
    items: list[EvalItem],
    vae: WanVAE,
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    context: Tensor,
    *,
    steps: int,
    shift: float,
) -> tuple[list, list, list, dict, list[float]]:
    metrics, ranges, temporal, visuals = [], [], [], {}
    sigmas = shifted_flow_sigmas(steps, shift=shift, device=dit.device)
    with torch.inference_mode():
        for item_index, item in enumerate(items):
            shuffled = items[(item_index + 1) % len(items)]
            generator = torch.Generator(device=dit.device).manual_seed(1201 + item_index)
            initial = torch.randn(
                item.clean.shape, generator=generator, device=dit.device, dtype=item.clean.dtype
            )
            for condition in CONDITIONS:
                sample = initial.clone()
                features = condition_features(item, shuffled, condition)
                for step_index in range(steps):
                    sigma = float(sigmas[step_index])
                    next_sigma = float(sigmas[step_index + 1])
                    velocity = predict(dit, conditioner, sample, sigma, context, features)
                    sample = flow_euler_step(
                        sample, velocity, sigma=sigma, next_sigma=next_sigma
                    )
                value = decode(vae, kind, sample)
                if value.shape != item.hr.shape or not torch.isfinite(value).all():
                    raise RuntimeError(f"invalid trajectory output: {kind}/{item.item_id}/{condition}")
                mode = f"trajectory_{steps}"
                metrics.extend(metric_rows(
                    value, item, kind=kind, mode=mode, sigma=None, condition=condition
                ))
                ranges.append({
                    "kind": kind, "mode": mode, "sigma": None,
                    "condition": condition, "item": item.item_id, **rgb_range_stats(value),
                })
                if kind == "4d":
                    temporal.append({
                        "kind": kind, "mode": mode, "sigma": None,
                        "condition": condition, "item": item.item_id,
                        "temporal_delta_mae": temporal_delta_mae(value, item.hr),
                    })
                visuals[(mode, item_index, None, condition)] = value
            print(json.dumps({
                "event": "decoded_trajectory", "kind": kind, "item": item.item_id,
                "steps": steps, "noise_seed": 1201 + item_index,
            }), flush=True)
    return metrics, ranges, temporal, visuals, [float(value) for value in sigmas.cpu()]


def evaluate_strict_3d_overfit(
    item: EvalItem,
    vae: WanVAE,
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    context: Tensor,
    perceptual_metric: torch.nn.Module,
    *,
    noise_seeds: tuple[int, ...],
    sampling: FlowSamplingConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate one V=4 sample from pure noise with paired LR interventions."""

    if item.hr.shape[:3] != (1, 3, 4):
        raise RuntimeError(f"strict 3D overfit requires one V=4 sample, got {tuple(item.hr.shape)}")
    rows: list[dict[str, Any]] = []
    ranges: list[dict[str, Any]] = []
    visuals: dict[tuple[int, str], Tensor] = {}
    conditions = ("correct", "shuffled", "neutral", "disabled")
    latent_shape = item.clean.shape

    def predict_velocity(sample, timestep, supplied_context, features):
        return conditioned_prediction(dit, conditioner, sample, timestep, supplied_context, features)

    with torch.inference_mode():
        for seed in noise_seeds:
            generator = torch.Generator(device=dit.device).manual_seed(seed)
            initial_noise = torch.randn(
                latent_shape,
                generator=generator,
                device=dit.device,
                dtype=torch.float32,
            )
            for condition in conditions:
                features = condition_features(item, item, condition)
                latent = sample_conditioned_flow(
                    initial_noise,
                    features,
                    context,
                    predict_velocity=predict_velocity,
                    config=sampling,
                )
                decoded = decode(vae, "3d", latent)
                if decoded.shape != item.hr.shape or not torch.isfinite(decoded).all():
                    raise RuntimeError(f"invalid strict decoded output: seed={seed} condition={condition}")
                rows.extend(metric_rows(
                    decoded,
                    item,
                    kind="3d",
                    mode=f"pure_noise_unipc_{sampling.steps}",
                    sigma=None,
                    condition=condition,
                    seed=seed,
                    perceptual_metric=perceptual_metric,
                ))
                ranges.append({
                    "kind": "3d", "mode": f"pure_noise_unipc_{sampling.steps}",
                    "seed": seed, "condition": condition, "item": item.item_id,
                    **rgb_range_stats(decoded),
                })
                visuals[(seed, condition)] = decoded

            for condition, baseline in (
                ("bicubic", item.bicubic),
                ("vae_ceiling", item.vae_ceiling),
            ):
                rows.extend(metric_rows(
                    baseline,
                    item,
                    kind="3d",
                    mode="baseline",
                    sigma=None,
                    condition=condition,
                    seed=seed,
                    perceptual_metric=perceptual_metric,
                ))
            print(json.dumps({
                "event": "strict_3d_seed", "seed": seed,
                "steps": sampling.steps, "shift": sampling.shift,
            }), flush=True)

    verdict = strict_3d_quality_verdict(rows)
    artifacts = write_strict_3d_visuals(item, visuals, output_dir, noise_seeds)
    result = {
        "mode": "pure_noise_unipc",
        "sampling": {"steps": sampling.steps, "shift": sampling.shift},
        "noise_seeds": list(noise_seeds),
        "initialization": "independent standard Gaussian latent; identical across conditions per seed",
        "metric_rows": rows,
        "range_rows": ranges,
        "verdict": verdict,
        "development_verdict": strict_3d_quality_verdict(rows, development=True) if len(noise_seeds) == 1 else None,
        "visual_review_required": True,
        "artifacts": artifacts,
    }
    output = output_dir / "3d_strict_metrics.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def add_baselines(kind: str, items: list[EvalItem]) -> tuple[list, list]:
    metrics, temporal = [], []
    for item in items:
        for condition, value in (("bicubic", item.bicubic), ("vae_ceiling", item.vae_ceiling)):
            metrics.extend(metric_rows(
                value, item, kind=kind, mode="baseline", sigma=None, condition=condition
            ))
            if kind == "4d":
                temporal.append({
                    "kind": kind, "mode": "baseline", "sigma": None,
                    "condition": condition, "item": item.item_id,
                    "temporal_delta_mae": temporal_delta_mae(value, item.hr),
                })
    return metrics, temporal


def group_key(row: dict[str, Any]) -> tuple[str, float | None]:
    return str(row["mode"]), None if row["sigma"] is None else float(row["sigma"])


def summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    paired_groups = defaultdict(list)
    for row in rows:
        mode, sigma = group_key(row)
        groups[(mode, sigma, str(row["condition"]))].append(row)
        if row["condition"] in CONDITIONS:
            paired_groups[(mode, sigma)].append(row)

    distributions = [
        {"mode": mode, "sigma": sigma, "condition": condition,
         **summarize_condition_rows(values)}
        for (mode, sigma, condition), values in sorted(groups.items(), key=lambda item: str(item[0]))
    ]
    paired_frames, paired_items = [], []
    for (mode, sigma), values in sorted(paired_groups.items(), key=lambda item: str(item[0])):
        for control in ("shuffled", "neutral", "disabled"):
            paired_frames.append({
                "mode": mode, "sigma": sigma,
                **paired_condition_summary(values, control=control),
            })
        item_groups = defaultdict(list)
        for row in values:
            item_groups[(str(row["item"]), str(row["condition"]))].append(row)
        averages = []
        for (item, condition), item_values in item_groups.items():
            averages.append({
                "item": item, "position": 0, "condition": condition,
                **{
                    metric: float(np.mean([float(value[metric]) for value in item_values]))
                    for metric in ("mae", "psnr", "ssim")
                },
            })
        for control in ("shuffled", "neutral", "disabled"):
            paired_items.append({
                "mode": mode, "sigma": sigma,
                **paired_condition_summary(averages, control=control),
            })
    return {
        "condition_distributions": distributions,
        "paired_frame_or_view": paired_frames,
        "paired_item": paired_items,
    }


def temporal_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        mode, sigma = group_key(row)
        groups[(mode, sigma, str(row["condition"]))].append(float(row["temporal_delta_mae"]))
    result = []
    for (mode, sigma, condition), values in sorted(groups.items(), key=lambda item: str(item[0])):
        array = np.asarray(values, dtype=np.float64)
        result.append({
            "mode": mode, "sigma": sigma, "condition": condition, "count": len(values),
            "mean": float(array.mean()), "median": float(np.median(array)),
            "std": float(array.std()), "min": float(array.min()), "max": float(array.max()),
        })
    return result


def tensor_image(value: Tensor, position: int) -> Image.Image:
    frame = value[0, :, position].float().clamp(-1, 1).add(1).mul(127.5)
    array = frame.permute(1, 2, 0).round().byte().numpy()
    return Image.fromarray(array, mode="RGB")


def error_image(value: Tensor, target: Tensor, position: int) -> Image.Image:
    pred = value[0, :, position].float().clamp(-1, 1).add(1).mul(0.5)
    ref = target[0, :, position].float().clamp(-1, 1).add(1).mul(0.5)
    error = (pred - ref).abs().mean(dim=0).clamp(0, 1)
    array = (error.numpy() * 255).round().astype(np.uint8)
    rgb = np.stack((array, np.zeros_like(array), 255 - array), axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def contact_sheet(
    rows: list[tuple[str, list[Image.Image]]],
    columns: list[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        canvas = Image.new("RGB", (640, 80), "white")
        ImageDraw.Draw(canvas).text((12, 28), "No matching rows", fill="black")
        canvas.save(path)
        return
    cell_w, cell_h = rows[0][1][0].size
    label_w, header_h, row_label_h = 210, 28, 22
    canvas = Image.new(
        "RGB",
        (label_w + len(columns) * cell_w, header_h + len(rows) * (cell_h + row_label_h)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(columns):
        draw.text((label_w + index * cell_w + 6, 7), label, fill="black")
    for row_index, (label, images) in enumerate(rows):
        y = header_h + row_index * (cell_h + row_label_h)
        draw.text((6, y + 6), label, fill="black")
        for column_index, image in enumerate(images):
            canvas.paste(image, (label_w + column_index * cell_w, y))
        draw.text((6, y + cell_h + 3), f"row {row_index}", fill=(80, 80, 80))
    canvas.save(path)


def _center_zoom(image: Image.Image) -> Image.Image:
    left, top = image.width // 4, image.height // 4
    crop = image.crop((left, top, image.width - left, image.height - top))
    return crop.resize(image.size, Image.Resampling.NEAREST)


def write_strict_3d_visuals(
    item: EvalItem,
    visuals: dict[tuple[int, str], Tensor],
    output_dir: Path,
    noise_seeds: tuple[int, ...],
) -> list[str]:
    produced: list[str] = []
    for seed in noise_seeds:
        rows: list[tuple[str, list[Image.Image]]] = []
        zoom_rows: list[tuple[str, list[Image.Image]]] = []
        for position in range(item.hr.shape[2]):
            individual_dir = output_dir / "images"
            individual_dir.mkdir(parents=True, exist_ok=True)
            for condition in ("correct", "shuffled", "neutral", "disabled"):
                tensor_image(visuals[(seed, condition)], position).save(individual_dir / f"seed_{seed}_view_{position}_{condition}.png")
            for condition, value in (("hr", item.hr), ("lr", item.lr), ("bicubic", item.bicubic), ("vae_ceiling", item.vae_ceiling)):
                tensor_image(value, position).save(individual_dir / f"view_{position}_{condition}.png")
            hr = tensor_image(item.hr, position)
            lr = tensor_image(item.lr, position).resize(hr.size, Image.Resampling.NEAREST)
            images = [
                hr,
                lr,
                tensor_image(item.bicubic, position),
                tensor_image(item.vae_ceiling, position),
                tensor_image(visuals[(seed, "correct")], position),
                tensor_image(visuals[(seed, "shuffled")], position),
                tensor_image(visuals[(seed, "neutral")], position),
                tensor_image(visuals[(seed, "disabled")], position),
            ]
            label = f"{safe_artifact_name(item.item_id)} / view {position} / seed {seed}"
            rows.append((label, images))
            zoom_rows.append((label, [_center_zoom(image) for image in images]))
        columns = ["HR", "LR nearest", "Bicubic", "VAE ceiling", "Correct", "Shuffled", "Neutral", "Disabled"]
        contact = output_dir / "3d" / f"strict_seed_{seed}_contact.png"
        zoom = output_dir / "3d" / f"strict_seed_{seed}_texture_zoom.png"
        contact_sheet(rows, columns, contact)
        contact_sheet(zoom_rows, columns, zoom)
        produced.extend((str(contact), str(zoom)))
    return produced


def labeled_video_frame(columns: list[tuple[str, Image.Image]]) -> np.ndarray:
    width = sum(image.width for _, image in columns)
    height = max(image.height for _, image in columns)
    canvas = Image.new("RGB", (width, height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in columns:
        canvas.paste(image, (x, 24))
        draw.text((x + 4, 6), label, fill="black")
        x += image.width
    return np.asarray(canvas)


def write_video(path: Path, value: Tensor, fps: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.asarray(tensor_image(value, position)) for position in range(value.shape[2])]
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def write_comparison_video(path: Path, values: list[tuple[str, Tensor]], fps: int = 4) -> None:
    frames = []
    for position in range(values[0][1].shape[2]):
        columns = [(label, tensor_image(value, position)) for label, value in values]
        frames.append(labeled_video_frame(columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)



def write_3d_visuals(
    items: list[EvalItem],
    visuals: dict,
    output_dir: Path,
    steps: int,
) -> list[str]:
    produced = []
    for mode, sigma in (("one_step", 0.5), ("one_step", 0.8), (f"trajectory_{steps}", None)):
        rows, error_rows = [], []
        for item_index, item in enumerate(items):
            for position in range(item.hr.shape[2]):
                outputs = [visuals[(mode, item_index, sigma, condition)] for condition in CONDITIONS]
                label = f"{safe_artifact_name(item.item_id)} / view {position}"
                rows.append((label, [
                    tensor_image(item.hr, position),
                    tensor_image(item.bicubic, position),
                    tensor_image(item.vae_ceiling, position),
                    *[tensor_image(value, position) for value in outputs],
                ]))
                error_rows.append((label, [
                    error_image(item.bicubic, item.hr, position),
                    error_image(item.vae_ceiling, item.hr, position),
                    *[error_image(value, item.hr, position) for value in outputs],
                ]))
        suffix = mode if sigma is None else f"{mode}_sigma_{sigma:.2f}".replace(".", "p")
        contact = output_dir / "3d" / f"3d_{suffix}_contact.png"
        errors = output_dir / "3d" / f"3d_{suffix}_errors.png"
        contact_sheet(rows, ["HR", "Bicubic", "VAE ceiling", *[LABELS[c] for c in CONDITIONS]], contact)
        contact_sheet(error_rows, ["Bicubic", "VAE ceiling", *[LABELS[c] for c in CONDITIONS]], errors)
        produced.extend((str(contact), str(errors)))
    return produced


def write_4d_visuals(
    items: list[EvalItem],
    visuals: dict,
    metric_values: list[dict[str, Any]],
    output_dir: Path,
    steps: int,
) -> list[str]:
    produced = []
    modes = (("one_step", 0.5), ("one_step", 0.8), (f"trajectory_{steps}", None))
    for item_index, item in enumerate(items):
        stem = safe_artifact_name(item.item_id)
        for mode, sigma in modes:
            outputs = {
                condition: visuals[(mode, item_index, sigma, condition)]
                for condition in CONDITIONS
            }
            rows = []
            for position in (0, 4, 8):
                rows.append((f"{item.item_id} / frame {position}", [
                    tensor_image(item.hr, position),
                    tensor_image(item.bicubic, position),
                    tensor_image(item.vae_ceiling, position),
                    *[tensor_image(outputs[c], position) for c in CONDITIONS],
                ]))
            suffix = mode if sigma is None else f"{mode}_sigma_{sigma:.2f}".replace(".", "p")
            contact = output_dir / "4d" / f"{stem}_{suffix}_contact.png"
            contact_sheet(rows, ["HR", "Bicubic", "VAE ceiling", *[LABELS[c] for c in CONDITIONS]], contact)
            produced.append(str(contact))
            for condition, value in outputs.items():
                video = output_dir / "4d" / f"{stem}_{suffix}_{condition}.mp4"
                write_video(video, value)
                produced.append(str(video))
            comparison = output_dir / "4d" / f"{stem}_{suffix}_comparison.mp4"
            write_comparison_video(comparison, [
                ("HR", item.hr), ("Correct", outputs["correct"]),
                ("Shuffled", outputs["shuffled"]), ("Disabled", outputs["disabled"]),
            ])
            produced.append(str(comparison))

    sigma_rows = [
        row for row in metric_values
        if row["mode"] == "one_step" and row["sigma"] == 0.95
        and row["condition"] in ("correct", "shuffled")
    ]
    indexed = {
        (row["item"], row["position"], row["condition"]): row
        for row in sigma_rows
    }
    failures = []
    for item_index, item in enumerate(items):
        for position in range(item.hr.shape[2]):
            correct = indexed[(item.item_id, position, "correct")]
            shuffled = indexed[(item.item_id, position, "shuffled")]
            if float(correct["psnr"]) <= float(shuffled["psnr"]):
                failures.append((item_index, position, correct, shuffled))

    for page in range(max(1, (len(failures) + 11) // 12)):
        rows = []
        for item_index, position, correct, shuffled in failures[page * 12:(page + 1) * 12]:
            item = items[item_index]
            correct_value = visuals[("one_step", item_index, 0.95, "correct")]
            shuffled_value = visuals[("one_step", item_index, 0.95, "shuffled")]
            label = (
                f"{item.item_id} f{position} | "
                f"PSNR c={correct['psnr']:.2f} s={shuffled['psnr']:.2f}"
            )
            rows.append((label, [
                tensor_image(item.hr, position),
                tensor_image(correct_value, position),
                tensor_image(shuffled_value, position),
                error_image(correct_value, item.hr, position),
                error_image(shuffled_value, item.hr, position),
            ]))
        gallery = output_dir / "4d" / f"4d_sigma_0p95_failure_gallery_{page + 1}.png"
        contact_sheet(rows, ["HR", "Correct", "Shuffled", "Correct error", "Shuffled error"], gallery)
        produced.append(str(gallery))
    return produced


def evaluate_kind(
    kind: str,
    adapter_path: Path,
    scene: Path,
    output_dir: Path,
    vae: WanVAE,
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    context: Tensor,
    *,
    model_dir: Path,
    sigmas: tuple[float, ...],
    trajectory_steps: int,
    trajectory_shift: float,
) -> dict[str, Any]:
    started = time.monotonic()
    metadata = checkpoint_metadata(adapter_path)
    config = ExperimentConfig(**metadata["config"])
    conditioner = FrozenLQConditioner(conditioner.projector, bridge_blocks=config.bridge_blocks, bridge_time_conditioning=config.bridge_time_conditioning).to(dit.device)
    if config.kind != kind:
        raise RuntimeError(f"checkpoint kind {config.kind} does not match {kind}")
    load_adapter_checkpoint(
        adapter_path,
        conditioner,
        expected_config=asdict(config),
        expected_experiment={
            "model_checkpoint": str(model_dir.resolve()),
            "lq_sha256": EXPECTED_LQ_SHA256,
        },
    )
    conditioner.eval()
    torch.cuda.reset_peak_memory_stats(dit.device)

    items = cache_items(config, scene, vae, conditioner, dit.device)
    baseline_metrics, baseline_temporal = add_baselines(kind, items)
    one_metrics, one_ranges, one_temporal, one_visuals = evaluate_one_step(
        kind, items, vae, dit, conditioner, context, sigmas
    )
    trajectory_metrics, trajectory_ranges, trajectory_temporal, trajectory_visuals, schedule = (
        evaluate_trajectory(
            kind, items, vae, dit, conditioner, context,
            steps=trajectory_steps, shift=trajectory_shift,
        )
    )
    all_metrics = baseline_metrics + one_metrics + trajectory_metrics
    all_temporal = baseline_temporal + one_temporal + trajectory_temporal
    all_visuals = {**one_visuals, **trajectory_visuals}
    if kind == "3d":
        artifacts = write_3d_visuals(items, all_visuals, output_dir, trajectory_steps)
    else:
        artifacts = write_4d_visuals(items, all_visuals, all_metrics, output_dir, trajectory_steps)

    result = {
        "kind": kind,
        "config": asdict(config),
        "adapter_checkpoint": str(adapter_path.resolve()),
        "adapter_sha256": sha256(adapter_path),
        "adapter_metadata": metadata,
        "items": [{
            "id": item.item_id, "hr_shape": list(item.hr.shape), "lr_shape": list(item.lr.shape),
            "latent_shape": list(item.clean.shape), "feature_shape": list(item.features.shape),
        } for item in items],
        "one_step": {
            "sigmas": list(sigmas),
            "noise_seed_rule": "201 + item_index*100 + sigma_index",
            "clean_estimate": "x_sigma - sigma * velocity_prediction",
        },
        "trajectory": {
            "steps": trajectory_steps, "shift": trajectory_shift, "sigmas": schedule,
            "noise_seed_rule": "1201 + item_index",
            "update": "x_next = x + (next_sigma - sigma) * velocity_prediction",
            "role": "supplementary; not a full quality-generation claim",
        },
        "metric_rows": all_metrics,
        "range_rows": one_ranges + trajectory_ranges,
        "temporal_rows": all_temporal,
        "summaries": summaries(all_metrics),
        "temporal_summaries": temporal_summaries(all_temporal),
        "artifacts": artifacts,
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(dit.device) / 2**20,
    }
    (output_dir / f"{kind}_decoded_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    del items, all_visuals
    gc.collect()
    torch.cuda.empty_cache()
    return result



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--lq-source", type=Path, required=True)
    parser.add_argument("--lq-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-3d", type=Path)
    parser.add_argument("--adapter-4d", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", choices=("3d", "4d"), default=("3d", "4d"))
    parser.add_argument("--sigmas", nargs="+", type=float, default=(0.2, 0.5, 0.8, 0.95))
    parser.add_argument("--trajectory-steps", type=int, default=8)
    parser.add_argument("--trajectory-shift", type=float, default=5.0)
    parser.add_argument("--strict-3d-overfit", action="store_true")
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=(2201, 2202, 2203, 2204))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.monotonic()
    _seed_all(42)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("decoded Stage 1 evaluation requires CUDA")
    if tuple(sorted(set(args.sigmas))) != tuple(args.sigmas):
        raise ValueError("sigmas must be unique and ascending")
    if any(not 0 < sigma < 1 for sigma in args.sigmas):
        raise ValueError("one-step sigmas must be inside (0,1)")
    if args.strict_3d_overfit and tuple(args.kinds) != ("3d",):
        raise ValueError("strict 3D overfit evaluation requires --kinds 3d")
    selected_paths = {"3d": args.adapter_3d, "4d": args.adapter_4d}
    missing = [kind for kind in args.kinds if selected_paths[kind] is None]
    if missing:
        raise ValueError(f"missing adapter checkpoint for selected kinds: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lq_actual_sha = sha256(args.lq_checkpoint)
    if lq_actual_sha != EXPECTED_LQ_SHA256:
        raise RuntimeError(f"LQ checkpoint SHA256 mismatch: {lq_actual_sha}")
    projector = load_flashvsr_projector(
        args.lq_source,
        args.lq_checkpoint,
        device=device,
        dtype=torch.bfloat16,
        expected_sha256=EXPECTED_LQ_SHA256,
    )
    conditioner = FrozenLQConditioner(projector).to(device).eval()
    vae = WanVAE.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    context = torch.zeros(1, 512, 4096, device=device, dtype=torch.bfloat16)
    if any(parameter.requires_grad for parameter in vae.model.model.parameters()):
        raise RuntimeError("Wan VAE must be frozen")
    if any(parameter.requires_grad for parameter in dit.model.parameters()):
        raise RuntimeError("Wan DiT must be frozen")
    if any(parameter.requires_grad for parameter in conditioner.projector.parameters()):
        raise RuntimeError("LQ projector must be frozen")

    paths = selected_paths
    results = {}
    if args.strict_3d_overfit:
        adapter_path = paths["3d"]
        metadata = checkpoint_metadata(adapter_path)
        config = ExperimentConfig(**metadata["config"])
        conditioner = FrozenLQConditioner(projector, bridge_blocks=config.bridge_blocks, bridge_time_conditioning=config.bridge_time_conditioning).to(device)
        if (
            config.kind != "3d" or config.hr_resolution != 256
            or config.views != 4 or config.sample_count != 1
        ):
            raise RuntimeError("strict checkpoint must be a 256x256 one-sample V=4 3D experiment")
        load_adapter_checkpoint(
            adapter_path,
            conditioner,
            expected_config=asdict(config),
            expected_experiment={
                "model_checkpoint": str(args.model_dir.resolve()),
                "lq_sha256": EXPECTED_LQ_SHA256,
            },
        )
        items = cache_items(config, args.scene, vae, conditioner, device)
        if len(items) != 1:
            raise RuntimeError("strict 3D overfit requires exactly one cached multiview sample")
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", reduction="mean", normalize=False
        ).to(device).eval().requires_grad_(False)
        strict = evaluate_strict_3d_overfit(
            items[0],
            vae,
            dit,
            conditioner,
            context,
            lpips_metric.net,
            noise_seeds=tuple(args.noise_seeds),
            sampling=FlowSamplingConfig(
                steps=config.sampling_steps,
                shift=config.sampling_shift,
            ),
            output_dir=args.output_dir,
        )
        report = {
            "schema_version": 1,
            "evaluation": "Stage 1 strict 3D pure-noise single-sample overfit",
            "created_unix_time": time.time(),
            "adapter_checkpoint": str(adapter_path.resolve()),
            "adapter_sha256": sha256(adapter_path),
            "adapter_metadata": metadata,
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "model_checkpoint": str(args.model_dir.resolve()),
                "lq_checkpoint": str(args.lq_checkpoint.resolve()),
                "lq_sha256": lq_actual_sha,
                "perceptual_metric": "LPIPS-VGG via torchmetrics 1.7.1; local weights only",
                "vgg16_sha256": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
                "lpips_vgg_sha256": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
                "text_condition": "deterministic all-zero [B,512,4096] BF16 context",
            },
            "result": strict,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "runtime_seconds": time.monotonic() - started,
        }
        output = args.output_dir / "decoded_metrics.json"
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "event": "strict_3d_complete", "output": str(output),
            "metrics_passed": strict["verdict"]["passed"],
            "visual_review_required": True,
        }), flush=True)
        return

    for kind in args.kinds:
        results[kind] = evaluate_kind(
            kind,
            paths[kind],
            args.scene,
            args.output_dir,
            vae,
            dit,
            conditioner,
            context,
            model_dir=args.model_dir,
            sigmas=tuple(args.sigmas),
            trajectory_steps=args.trajectory_steps,
            trajectory_shift=args.trajectory_shift,
        )

    report = {
        "schema_version": 1,
        "evaluation": "Stage 1 decoded-space LR-conditioning evaluation",
        "created_unix_time": time.time(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "model_checkpoint": str(args.model_dir.resolve()),
            "lq_checkpoint": str(args.lq_checkpoint.resolve()),
            "lq_sha256": lq_actual_sha,
            "text_condition": "deterministic all-zero [B,512,4096] BF16 context",
            "lpips": "not run; no LPIPS dependency or external perceptual weights added",
        },
        "controls": list(CONDITIONS),
        "results": results,
        "runtime_seconds": time.monotonic() - started,
    }
    output = args.output_dir / "decoded_metrics.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "decoded_complete", "output": str(output),
        "runtime_seconds": report["runtime_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
