#!/usr/bin/env python3
"""Run the Stage 1 frozen-Wan LR-conditioning micro-overfit experiments."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rl3dsr.data import NeRFSyntheticAdapter
from rl3dsr.data.temporal_fixture import make_motion_video
from rl3dsr.data.tensor_prep import rgb_images_to_video
from rl3dsr.models.wan import WanDiT, WanVAE
from rl3dsr.models.wan.flow import flow_matching_loss, flow_matching_pair
from rl3dsr.models.wan.lq_conditioning import (
    FrozenLQConditioner,
    conditioned_prediction,
    Stage1Degradation,
    derange_multiview_lr,
    evenly_spaced_indices,
    load_adapter_checkpoint,
    load_flashvsr_projector,
    save_adapter_checkpoint,
)
from rl3dsr.models.wan.sampling import FlowSamplingConfig, SigmaCycle, oracle_sampling_audit
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_reporting import atomic_json, snapshot_source, plot_training_curves, write_checkpoint_index


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    kind: str
    seed: int = 42
    hr_resolution: int = 128
    scale: int = 4
    views: int = 4
    sample_count: int = 2
    video_frames: int = 9
    max_steps: int = 150
    eval_interval: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    sampling_steps: int = 50
    sampling_shift: float = 5.0
    sampling_strategy: str = "permutation"
    checkpoint_interval: int = 250
    lr_decay_step: int = 750
    lr_decay_factor: float = 0.1
    max_hours: float = 2.0
    quality_overfit: bool = False
    bridge_blocks: tuple[int, ...] = (0,)
    bridge_time_conditioning: bool = False
    stop_on_dev_pass: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'bridge_blocks', tuple(self.bridge_blocks))


@dataclass(slots=True)
class CachedItem:
    item_id: str
    hr: Tensor
    lr: Tensor
    clean: Tensor
    features: Tensor
    shuffled_features: Tensor | None
    neutral_features: Tensor


EXPECTED_LQ_SHA256 = "d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"failed to persist experiment log {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"failed to persist experiment CSV {path}") from exc


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _plot_training_curves(output_dir, step_rows, eval_rows):
    plot_training_curves(output_dir, step_rows, eval_rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    """Hash a checkpoint/data tree without copying its contents."""
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _validate_step_history(rows: list[dict[str, Any]], expected_last: int | None = None) -> None:
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(1, len(steps) + 1)):
        raise RuntimeError("training log must contain contiguous, non-duplicated steps")
    if expected_last is not None and (steps[-1] if steps else 0) != expected_last:
        raise RuntimeError("training log and checkpoint step disagree")


def _quality_key(row: dict[str, Any]) -> tuple[int, int, float, float, float]:
    return (
        int(bool(row.get("dev_candidate_pass"))),
        int(bool(row.get("decoded_quality_pass"))),
        float(row.get("correct_psnr_margin", -1e9)),
        float(row.get("correct_ssim_margin", -1e9)),
        float(row.get("correct_lpips_reduction", -1e9)),
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _tensor_from_motion(seed: int, frames: int, resolution: int) -> Tensor:
    sample = make_motion_video(frames, height=resolution, width=resolution, seed=seed)
    return torch.from_numpy(sample.frames.copy()).permute(3, 0, 1, 2).unsqueeze(0).float().div(127.5).sub(1.0)


def _prepare_rgb(config: ExperimentConfig, scene: Path) -> list[tuple[str, Tensor]]:
    if config.kind == "3d":
        adapter = NeRFSyntheticAdapter(scene)
        sequence = adapter.index("train")
        count = config.views * config.sample_count
        indices = list(evenly_spaced_indices(len(sequence.observations), count))
        selected = [sequence.observations[index] for index in indices]
        result = []
        for group in range(config.sample_count):
            observations = selected[group * config.views : (group + 1) * config.views]
            rgb = rgb_images_to_video(
                [adapter.load_rgb(observation) for observation in observations],
                config.hr_resolution,
            )
            result.append((f"{sequence.scene_id}:views:{indices[group * config.views:(group + 1) * config.views]}", rgb))
        return result
    if config.kind == "4d":
        return [
            (f"stage1_motion_seed_{seed}", _tensor_from_motion(seed, config.video_frames, config.hr_resolution))
            for seed in range(4)
        ]
    raise ValueError("kind must be 3d or 4d")


def _cache_items(
    config: ExperimentConfig,
    rgb_items: list[tuple[str, Tensor]],
    vae: WanVAE,
    conditioner: FrozenLQConditioner,
    device: torch.device,
) -> list[CachedItem]:
    degradation = Stage1Degradation(scale=config.scale)
    cached: list[CachedItem] = []
    for item_id, hr_cpu in rgb_items:
        lr_cpu = degradation(hr_cpu)
        hr = hr_cpu.to(device)
        lr = lr_cpu.to(device)
        with torch.inference_mode():
            if config.kind == "3d":
                clean = vae.encode_multiview(hr)
                features = conditioner.multiview_features(
                    lr,
                    conditioning_size=(config.hr_resolution, config.hr_resolution),
                    latent_shape=(clean.shape[2], clean.shape[3], clean.shape[4]),
                )
                shuffled_lr, shuffled_indices = derange_multiview_lr(lr)
                shuffled = conditioner.multiview_features(
                    shuffled_lr,
                    conditioning_size=(config.hr_resolution, config.hr_resolution),
                    latent_shape=(clean.shape[2], clean.shape[3], clean.shape[4]),
                )
                neutral = conditioner.multiview_features(
                    torch.zeros_like(lr),
                    conditioning_size=(config.hr_resolution, config.hr_resolution),
                    latent_shape=(clean.shape[2], clean.shape[3], clean.shape[4]),
                )
            else:
                clean = vae.encode_video(hr)
                shuffled = None
                shuffled_indices = None
                features = conditioner.video_features(
                    lr,
                    conditioning_size=(config.hr_resolution, config.hr_resolution),
                    latent_shape=(clean.shape[2], clean.shape[3], clean.shape[4]),
                )
                neutral = conditioner.video_features(
                    torch.zeros_like(lr),
                    conditioning_size=(config.hr_resolution, config.hr_resolution),
                    latent_shape=(clean.shape[2], clean.shape[3], clean.shape[4]),
                )
        cached.append(
            CachedItem(
                item_id=item_id,
                hr=hr_cpu.detach().cpu(),
                lr=lr_cpu.detach().cpu(),
                clean=clean.detach().cpu(),
                features=features.detach().cpu(),
                shuffled_features=None if shuffled is None else shuffled.detach().cpu(),
                neutral_features=neutral.detach().cpu(),
            )
        )
        print(json.dumps({
            "event": "cached", "item": item_id, "latent": list(clean.shape),
            "tokens": list(features.shape), "shuffled_view_indices": shuffled_indices,
        }), flush=True)
    return cached


def _parameter_report(vae: WanVAE, dit: WanDiT, conditioner: FrozenLQConditioner) -> dict[str, Any]:
    groups = {
        "wan_vae": list(vae.model.model.parameters()),
        "wan_dit_including_text": list(dit.model.parameters()),
        "frozen_lq_projector": list(conditioner.projector.parameters()),
        "trainable_bridge": list(conditioner.bridge.parameters()),
    }
    counts = {name: sum(parameter.numel() for parameter in parameters) for name, parameters in groups.items()}
    total = sum(counts.values())
    trainable = sum(parameter.numel() for parameter in conditioner.parameters() if parameter.requires_grad)
    return {
        "groups": counts,
        "total": total,
        "trainable": trainable,
        "trainable_ratio": trainable / total,
        "trainable_names": [name for name, parameter in conditioner.named_parameters() if parameter.requires_grad],
        "frozen_modules": ["Wan VAE", "Wan DiT including text embedding", "FlashVSR LQ projector"],
    }


def _forward_case(
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    clean: Tensor,
    features: Tensor,
    *,
    sigma: float,
    noise_seed: int,
    context: Tensor,
    enabled: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    device = dit.device
    clean = clean.to(device)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
    noisy, timestep, target = flow_matching_pair(clean, noise, torch.tensor([sigma], device=device))
    prediction = conditioned_prediction(dit, conditioner, noisy, timestep, context, features if enabled else None)
    return prediction, target, noisy


def _baseline_and_gradient_audit(
    config: ExperimentConfig,
    items: list[CachedItem],
    vae: WanVAE,
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    context: Tensor,
) -> dict[str, Any]:
    item = items[0]
    clean = item.clean.to(dit.device)
    generator = torch.Generator(device=dit.device).manual_seed(17)
    noise = torch.randn(clean.shape, generator=generator, device=dit.device, dtype=clean.dtype)
    noisy, timestep, target = flow_matching_pair(clean, noise, torch.tensor([0.5], device=dit.device))
    zero_residual = conditioner.bridge_residuals(item.features.to(dit.device), timestep)
    with torch.no_grad():
        baseline = dit(noisy, timestep, context)
        conditioned = dit(noisy, timestep, context, block_token_residuals=zero_residual)
    delta = (conditioned - baseline).abs().float()
    preservation = {
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
        "max_tolerance": 1e-6,
        "mean_tolerance": 1e-7,
        "passed": bool(delta.max() <= 1e-6 and delta.mean() <= 1e-7),
    }
    if not preservation["passed"]:
        raise RuntimeError(f"baseline preservation failed: {preservation}")

    prediction = dit(noisy, timestep, context, block_token_residuals=zero_residual)
    loss = flow_matching_loss(prediction, target)
    loss.backward()
    bridge_gradients = {
        name: {
            "present": parameter.grad is not None,
            "finite": bool(parameter.grad is not None and torch.isfinite(parameter.grad).all()),
            "norm": float(parameter.grad.float().norm()) if parameter.grad is not None else 0.0,
        }
        for name, parameter in conditioner.bridge.named_parameters()
    }
    frozen_grad_counts = {
        "wan_vae": sum(parameter.grad is not None for parameter in vae.model.model.parameters()),
        "wan_dit": sum(parameter.grad is not None for parameter in dit.model.parameters()),
        "text_embedding": sum(parameter.grad is not None for parameter in dit.model.text_embedding.parameters()),
        "lq_projector": sum(parameter.grad is not None for parameter in conditioner.projector.parameters()),
    }
    gate_after_update = {}
    if conditioner.bridge_time_conditioning:
        saved = {name: value.detach().clone() for name, value in conditioner.bridge.state_dict().items()}
        try:
            with torch.no_grad():
                for parameter in conditioner.bridge.parameters():
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-config.learning_rate)
            conditioner.bridge.zero_grad(set_to_none=True)
            updated_prediction = conditioned_prediction(dit, conditioner, noisy, timestep, context, item.features)
            flow_matching_loss(updated_prediction, target).backward()
            gate_after_update = {
                name: float(parameter.grad.float().norm())
                for name, parameter in conditioner.bridge.named_parameters() if 'time_gates' in name
            }
            if not gate_after_update or not all(math.isfinite(value) and value > 0 for value in gate_after_update.values()):
                raise RuntimeError(f'time gates failed post-update gradient audit: {gate_after_update}')
        finally:
            conditioner.bridge.load_state_dict(saved, strict=True)
    conditioner.bridge.zero_grad(set_to_none=True)
    gradient_passed = all(value["present"] and value["finite"] and (value["norm"] > 0 or "time_gates" in name) for name, value in bridge_gradients.items()) and all(
        count == 0 for count in frozen_grad_counts.values()
    )
    if not gradient_passed:
        raise RuntimeError(f"gradient isolation failed: {bridge_gradients}, {frozen_grad_counts}")
    return {
        "baseline_preservation": preservation,
        "gradient_isolation": {
            "bridge": bridge_gradients,
            "time_gate_gradient_after_update": gate_after_update,
            "frozen_gradient_tensor_counts": frozen_grad_counts,
            "passed": gradient_passed,
        },
    }


def _evaluate(
    items: list[CachedItem],
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    context: Tensor,
    *,
    sigmas: tuple[float, ...],
    seed_base: int,
) -> dict[str, Any]:
    conditioner.eval()
    rows = []
    with torch.no_grad():
        for item_index, item in enumerate(items):
            shuffled = (
                item.shuffled_features
                if item.shuffled_features is not None
                else items[(item_index + 1) % len(items)].features
            )
            for sigma_index, sigma in enumerate(sigmas):
                seed = seed_base + item_index * 100 + sigma_index
                correct_prediction, target, _ = _forward_case(
                    dit, conditioner, item.clean, item.features, sigma=sigma, noise_seed=seed, context=context
                )
                shuffled_prediction, _, _ = _forward_case(
                    dit, conditioner, item.clean, shuffled, sigma=sigma, noise_seed=seed, context=context
                )
                neutral_prediction, _, _ = _forward_case(
                    dit, conditioner, item.clean, item.neutral_features, sigma=sigma, noise_seed=seed, context=context
                )
                disabled_prediction, _, _ = _forward_case(
                    dit,
                    conditioner,
                    item.clean,
                    item.features,
                    sigma=sigma,
                    noise_seed=seed,
                    context=context,
                    enabled=False,
                )
                difference = (correct_prediction - shuffled_prediction).float()
                correct_loss = float(flow_matching_loss(correct_prediction, target))
                shuffled_loss = float(flow_matching_loss(shuffled_prediction, target))
                neutral_loss = float(flow_matching_loss(neutral_prediction, target))
                disabled_loss = float(flow_matching_loss(disabled_prediction, target))
                rows.append(
                    {
                        "item": item.item_id,
                        "sigma": sigma,
                        "noise_seed": seed,
                        "correct_loss": correct_loss,
                        "shuffled_loss": shuffled_loss,
                        "neutral_loss": neutral_loss,
                        "disabled_loss": disabled_loss,
                        "intervention_mean_abs": float(difference.abs().mean()),
                        "intervention_max_abs": float(difference.abs().max()),
                        "intervention_relative_l2": float(difference.norm() / correct_prediction.float().norm().clamp_min(1e-12)),
                        "correct_beats_shuffled": correct_loss < shuffled_loss,
                        "correct_beats_neutral": correct_loss < neutral_loss,
                        "correct_beats_disabled": correct_loss < disabled_loss,
                    }
                )
    keys = ("correct_loss", "shuffled_loss", "neutral_loss", "disabled_loss")
    means = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    direction_fraction = float(
        np.mean(
            [
                row["correct_beats_shuffled"] and row["correct_beats_neutral"] and row["correct_beats_disabled"]
                for row in rows
            ]
        )
    )
    intervention_mean = float(np.mean([row["intervention_mean_abs"] for row in rows]))
    intervention_max = float(np.max([row["intervention_max_abs"] for row in rows]))
    intervention_relative = float(np.mean([row["intervention_relative_l2"] for row in rows]))
    passed = (
        means["correct_loss"] <= 0.95 * means["shuffled_loss"]
        and means["correct_loss"] <= 0.95 * means["neutral_loss"]
        and means["correct_loss"] <= 0.95 * means["disabled_loss"]
        and direction_fraction >= 0.75
        and intervention_mean > 1e-5
        and intervention_max > 1e-4
        and intervention_relative >= 1e-3
    )
    conditioner.train()
    return {
        "rows": rows,
        "means": means,
        "direction_fraction": direction_fraction,
        "intervention": {
            "mean_abs": intervention_mean,
            "max_abs": intervention_max,
            "relative_l2": intervention_relative,
        },
        "thresholds": {
            "correct_loss_ratio": 0.95,
            "direction_fraction": 0.75,
            "intervention_mean_abs": 1e-5,
            "intervention_max_abs": 1e-4,
            "intervention_relative_l2": 1e-3,
        },
        "passed": bool(passed),
    }


def _training_signature(config: ExperimentConfig) -> dict[str, Any]:
    return {
        key: getattr(config, key)
        for key in (
            "kind", "seed", "hr_resolution", "scale", "views", "sample_count",
            "sampling_steps", "sampling_shift", "sampling_strategy", "learning_rate",
            "weight_decay", "lr_decay_step", "lr_decay_factor", "bridge_blocks", "bridge_time_conditioning", "gradient_clip", "video_frames",
        )
    }


def _save_training_state(
    path: Path,
    *,
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer,
    step: int,
    sigma_cycle: SigmaCycle,
    noise_generator: torch.Generator,
    elapsed_seconds: float,
    adapter_checkpoint_sha256: str,
) -> None:
    torch.save(
        {
            "format_version": 2,
            "signature": _training_signature(config),
            "step": step,
            "optimizer": optimizer.state_dict(),
            "sigma_cycle": sigma_cycle.state_dict(),
            "noise_generator_state": noise_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "elapsed_seconds": elapsed_seconds,
            "adapter_checkpoint_sha256": adapter_checkpoint_sha256,
        },
        path,
    )


def _train_legacy(
    config: ExperimentConfig,
    items: list[CachedItem],
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    context: Tensor,
    output_dir: Path,
    *,
    resume_state: Path | None,
    experiment_identity: dict[str, Any],
    initial_checkpoint_sha256: str | None,
) -> tuple[dict[str, Any], torch.optim.Optimizer, int]:
    trainable = [parameter for parameter in conditioner.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != {id(parameter) for parameter in conditioner.bridge.parameters()}:
        raise RuntimeError("optimizer contains unexpected parameters")
    sigma_cycle = SigmaCycle(
        FlowSamplingConfig(steps=config.sampling_steps, shift=config.sampling_shift),
        seed=config.seed + 1000,
    )
    noise_generator = torch.Generator(device=dit.device).manual_seed(config.seed + 2000)
    losses: list[float] = []
    evaluations: list[dict[str, Any]] = []
    consecutive_passes = 0
    start_step = 0
    elapsed_prior = 0.0
    if resume_state is not None:
        state = torch.load(resume_state, map_location="cpu", weights_only=False)
        if state.get("format_version") != 2 or state.get("signature") != _training_signature(config):
            raise RuntimeError("training state is incompatible with the requested experiment")
        optimizer.load_state_dict(state["optimizer"])
        sigma_cycle.load_state_dict(state["sigma_cycle"])
        noise_generator.set_state(state["noise_generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        start_step = int(state["step"])
        elapsed_prior = float(state["elapsed_seconds"])
        if state.get("adapter_checkpoint_sha256") != initial_checkpoint_sha256:
            raise RuntimeError("training state does not match the loaded adapter checkpoint")

    step = start_step
    stop_reason = "max_steps"
    started = time.monotonic()
    stop_reason = "max_steps"
    conditioner.train()
    for step in range(start_step + 1, config.max_steps + 1):
        item = items[(step - 1) % len(items)]
        clean = item.clean.to(dit.device)
        features = item.features.to(dit.device)
        sigma = sigma_cycle.next().reshape(1).to(dit.device)
        noise = torch.randn(clean.shape, generator=noise_generator, device=dit.device, dtype=clean.dtype)
        noisy, timestep, target = flow_matching_pair(clean, noise, sigma)
        optimizer.zero_grad(set_to_none=True)
        prediction = conditioned_prediction(dit, conditioner, noisy, timestep, context, features)
        loss = flow_matching_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in trainable):
            raise RuntimeError(f"invalid bridge gradient at step {step}")
        if any(parameter.grad is not None for parameter in dit.model.parameters()):
            raise RuntimeError(f"frozen DiT gradient found at step {step}")
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 10 == 0:
            print(json.dumps({"event": "train", "kind": config.kind, "step": step, "loss": losses[-1], "sigma": float(sigma)}), flush=True)
        if step % config.eval_interval == 0:
            evaluation = _evaluate(items, dit, conditioner, context, sigmas=(0.2, 0.5, 0.8, 0.95), seed_base=101)
            evaluation["step"] = step
            evaluations.append(evaluation)
            consecutive_passes = consecutive_passes + 1 if evaluation["passed"] else 0
            print(json.dumps({"event": "eval", "kind": config.kind, "step": step, "means": evaluation["means"], "direction": evaluation["direction_fraction"], "intervention": evaluation["intervention"], "passed": evaluation["passed"]}), flush=True)
            if consecutive_passes >= 2 and not config.quality_overfit:
                stop_reason = "causal_early_stop"
                break
        elapsed = elapsed_prior + time.monotonic() - started
        if step % config.checkpoint_interval == 0:
            latest_path = output_dir / f"{config.kind}_latest.pt"
            save_adapter_checkpoint(
                latest_path,
                conditioner,
                config=asdict(config),
                experiment={
                    **experiment_identity,
                    "step": step,
                    "status": "running",
                    "parent_checkpoint_sha256": initial_checkpoint_sha256,
                },
            )
            _save_training_state(
                output_dir / f"{config.kind}_training_state.pt",
                config=config,
                optimizer=optimizer,
                step=step,
                sigma_cycle=sigma_cycle,
                noise_generator=noise_generator,
                elapsed_seconds=elapsed,
                adapter_checkpoint_sha256=_sha256(latest_path),
            )
        if elapsed >= config.max_hours * 3600:
            stop_reason = "max_hours"
            break

    elapsed = elapsed_prior + time.monotonic() - started
    latest_path = output_dir / f"{config.kind}_latest.pt"
    save_adapter_checkpoint(
        latest_path,
        conditioner,
        config=asdict(config),
        experiment={
            **experiment_identity,
            "step": step,
            "status": "stage_complete",
            "parent_checkpoint_sha256": initial_checkpoint_sha256,
        },
    )
    _save_training_state(
        output_dir / f"{config.kind}_training_state.pt",
        config=config,
        optimizer=optimizer,
        step=step,
        sigma_cycle=sigma_cycle,
        noise_generator=noise_generator,
        elapsed_seconds=elapsed,
        adapter_checkpoint_sha256=_sha256(latest_path),
    )
    return {
        "start_step": start_step,
        "steps": step,
        "losses": losses,
        "periodic_evaluations": evaluations,
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason,
        "sigma_cycle": {
            "cycle": sigma_cycle.cycle,
            "position": sigma_cycle.position,
            "values_per_cycle": len(sigma_cycle.values),
        },
    }, optimizer, step


def _save_adapter_atomic(path: Path, conditioner: FrozenLQConditioner, *, config: ExperimentConfig, experiment: dict[str, Any]) -> None:
    if '_step_' in path.name and path.exists():
        raise RuntimeError(f'refusing to overwrite immutable checkpoint: {path}')
    temporary = path.with_name(path.name + ".tmp")
    save_adapter_checkpoint(temporary, conditioner, config=asdict(config), experiment=experiment)
    os.replace(temporary, path)


def _save_observed_state(
    path: Path,
    *,
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer,
    step: int,
    sigma_cycle: SigmaCycle,
    noise_generator: torch.Generator,
    elapsed_seconds: float,
    adapter_checkpoint_sha256: str,
    log_path: Path,
    step_count: int,
    eval_count: int,
    parent_checkpoint_sha256: str | None,
) -> None:
    _atomic_torch_save(
        {
            "format_version": 4,
            "signature": _training_signature(config),
            "config_sha256": _config_hash(config),
            "step": step,
            "optimizer": optimizer.state_dict(),
            "sigma_cycle": sigma_cycle.state_dict(),
            "noise_generator_state": noise_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "elapsed_seconds": elapsed_seconds,
            "adapter_checkpoint_sha256": adapter_checkpoint_sha256,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "log_path": str(log_path.resolve()),
            "step_count": step_count,
            "eval_count": eval_count,
        },
        path,
    )


def _train(
    config: ExperimentConfig,
    items: list[CachedItem],
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    context: Tensor,
    output_dir: Path,
    *,
    resume_state: Path | None,
    experiment_identity: dict[str, Any],
    initial_checkpoint_sha256: str | None,
    decoded_evaluator: Any | None = None,
) -> tuple[dict[str, Any], torch.optim.Optimizer, int]:
    trainable = [parameter for parameter in conditioner.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    sigma_cycle = SigmaCycle(
        FlowSamplingConfig(steps=config.sampling_steps, shift=config.sampling_shift),
        seed=config.seed + 1000,
        strategy=config.sampling_strategy,
    )
    noise_generator = torch.Generator(device=dit.device).manual_seed(config.seed + 2000)
    log_path = output_dir / "train_steps.jsonl"
    csv_path = output_dir / "train_steps.csv"
    eval_path = output_dir / "checkpoint_metrics.jsonl"
    step_rows = _read_jsonl(log_path)
    eval_rows = _read_jsonl(eval_path)
    if resume_state is None and step_rows:
        raise RuntimeError(f"refusing to append to existing experiment log: {log_path}")
    _validate_step_history(step_rows)
    start_step = 0
    elapsed_prior = 0.0
    if resume_state is not None:
        state = torch.load(resume_state, map_location="cpu", weights_only=False)
        if state.get("format_version") not in {3, 4} or state.get("signature") != _training_signature(config):
            raise RuntimeError("training state is incompatible with the observed Stage 1 experiment")
        # The behavioral signature is strict; max_steps and reporting cadence may be
        # extended on resume, so the full manifest hash is retained for provenance
        # but is not used to reject a longer continuation.
        optimizer.load_state_dict(state["optimizer"])
        sigma_cycle.load_state_dict(state["sigma_cycle"])
        noise_generator.set_state(state["noise_generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        start_step = int(state["step"])
        elapsed_prior = float(state["elapsed_seconds"])
        if state.get("adapter_checkpoint_sha256") != initial_checkpoint_sha256:
            raise RuntimeError("training state does not match the loaded adapter checkpoint")
        if len(step_rows) != int(state.get("step_count", len(step_rows))):
            raise RuntimeError("training log is incomplete for the requested resume state")
        if state.get("format_version") == 4 and state.get("log_path") != str(log_path.resolve()):
            raise RuntimeError("training state points at a different experiment log")
        if state.get("format_version") == 4 and len(eval_rows) < int(state.get("eval_count", 0)):
            raise RuntimeError("decoded evaluation history is incomplete for the requested resume state")
    _validate_step_history(step_rows, start_step if step_rows else None)

    best_row = None
    best_key = None
    if eval_rows:
        for row in eval_rows:
            key = _quality_key(row)
            if best_key is None or key > best_key:
                best_key, best_row = key, row
    consecutive_passes = 0
    latent_evaluations: list[dict[str, Any]] = []
    started = time.monotonic()
    stop_reason = "max_steps"
    conditioner.train()
    for step in range(start_step + 1, config.max_steps + 1):
        if time.time() >= float(os.environ.get("RL3DSR_GPU_DEADLINE", "inf")):
            stop_reason = "gpu_budget"
            break
        step_started = time.monotonic()
        item = items[(step - 1) % len(items)]
        clean = item.clean.to(dit.device)
        features = item.features.to(dit.device)
        sigma = sigma_cycle.next().reshape(1).to(dit.device)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * (config.lr_decay_factor if step > config.lr_decay_step else 1.0)
        noise = torch.randn(clean.shape, generator=noise_generator, device=dit.device, dtype=clean.dtype)
        noisy, timestep, target = flow_matching_pair(clean, noise, sigma)
        optimizer.zero_grad(set_to_none=True)
        prediction = conditioned_prediction(dit, conditioner, noisy, timestep, context, features)
        loss = flow_matching_loss(prediction, target)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip))
        clipped = gradient_norm > config.gradient_clip
        diagnostics = {}
        for block, values in getattr(dit, "last_injection_stats", {}).items():
            for name, value in values.items():
                if name == "residual_to_input_rms":
                    diagnostics[f"block_{block}_{name}"] = float(value)
        for block, values in getattr(conditioner.bridge, "last_diagnostics", {}).items():
            for name, value in values.items():
                if name.startswith("gate_"):
                    diagnostics[f"block_{block}_{name}"] = float(value)
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in trainable):
            raise RuntimeError(f"invalid bridge gradient at step {step}")
        if any(parameter.grad is not None for parameter in dit.model.parameters()):
            raise RuntimeError(f"frozen DiT gradient found at step {step}")
        optimizer.step()
        elapsed = elapsed_prior + time.monotonic() - started
        row = {
            "step": step,
            "item_id": item.item_id,
            "loss": float(loss.detach()),
            "sigma": float(sigma),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": gradient_norm,
            "gradient_clip_applied": bool(clipped),
            "elapsed_seconds": elapsed,
            "step_seconds": time.monotonic() - step_started,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(dit.device) / 2**20,
            "sigma_cycle": sigma_cycle.cycle,
            "sigma_position": max(0, sigma_cycle.position - 1),
            **diagnostics,
        }
        step_rows.append(row)
        _append_jsonl(log_path, row)
        if step == 1 or step % 10 == 0:
            print(json.dumps({"event": "train", "kind": config.kind, **row}), flush=True)
        if step % config.eval_interval == 0:
            evaluation = _evaluate(items, dit, conditioner, context, sigmas=(0.2, 0.5, 0.8, 0.95), seed_base=101)
            evaluation["step"] = step
            latent_evaluations.append(evaluation)
            _append_jsonl(output_dir / "sigma_metrics.jsonl", evaluation)
            consecutive_passes = consecutive_passes + 1 if evaluation["passed"] else 0
            print(json.dumps({"event": "eval", "kind": config.kind, "step": step, "means": evaluation["means"], "direction": evaluation["direction_fraction"], "passed": evaluation["passed"]}), flush=True)
            if consecutive_passes >= 2 and not config.quality_overfit:
                break
        if step % config.checkpoint_interval == 0:
            step_path = output_dir / f"{config.kind}_step_{step:04d}.pt"
            metadata = {**experiment_identity, "step": step, "status": "running", "parent_checkpoint_sha256": initial_checkpoint_sha256}
            _save_adapter_atomic(step_path, conditioner, config=config, experiment=metadata)
            step_sha = _sha256(step_path)
            state_path = output_dir / f"{config.kind}_step_{step:04d}_training_state.pt"
            if decoded_evaluator is not None:
                quality = decoded_evaluator(step, step_path)
                eval_rows.append(quality)
                _append_jsonl(eval_path, quality)
                key = _quality_key(quality)
                if best_key is None or key > best_key:
                    best_key, best_row = key, quality
                    for prior in eval_rows:
                        prior["is_best"] = False
                    quality["is_best"] = True
                    _atomic_copy(step_path, output_dir / "best_dev.pt")
            _save_observed_state(state_path, config=config, optimizer=optimizer, step=step, sigma_cycle=sigma_cycle, noise_generator=noise_generator, elapsed_seconds=elapsed_prior + time.monotonic() - started, adapter_checkpoint_sha256=step_sha, log_path=log_path, step_count=len(step_rows), eval_count=len(eval_rows), parent_checkpoint_sha256=initial_checkpoint_sha256)
            if decoded_evaluator is not None and best_row is not None and int(best_row.get("step", -1)) == step:
                _atomic_copy(state_path, output_dir / "best_dev_training_state.pt")
            _write_csv(csv_path, step_rows)
            _plot_training_curves(output_dir, step_rows, eval_rows)
            write_checkpoint_index(output_dir, _config_hash(config), eval_rows, None if best_row is None else int(best_row['step']))
            if config.stop_on_dev_pass and quality.get("dev_candidate_pass"):
                stop_reason = "dev_candidate_pass"
                break
        if elapsed_prior + time.monotonic() - started >= config.max_hours * 3600:
            stop_reason = "time_budget"
            break
    step = step_rows[-1]["step"] if step_rows else start_step
    elapsed = elapsed_prior + time.monotonic() - started
    if step and not (output_dir / f"{config.kind}_step_{int(step):04d}.pt").exists():
        step_path = output_dir / f"{config.kind}_step_{int(step):04d}.pt"
        metadata = {**experiment_identity, "step": int(step), "status": "stage_complete", "parent_checkpoint_sha256": initial_checkpoint_sha256}
        _save_adapter_atomic(step_path, conditioner, config=config, experiment=metadata)
        step_sha = _sha256(step_path)
        state_path = output_dir / f"{config.kind}_step_{int(step):04d}_training_state.pt"
        if decoded_evaluator is not None:
            quality = decoded_evaluator(int(step), step_path)
            eval_rows.append(quality)
            _append_jsonl(eval_path, quality)
            key = _quality_key(quality)
            if best_key is None or key > best_key:
                best_key, best_row = key, quality
                for prior in eval_rows:
                    prior["is_best"] = False
                quality["is_best"] = True
                _atomic_copy(step_path, output_dir / "best_dev.pt")
        _save_observed_state(state_path, config=config, optimizer=optimizer, step=int(step), sigma_cycle=sigma_cycle, noise_generator=noise_generator, elapsed_seconds=elapsed_prior + time.monotonic() - started, adapter_checkpoint_sha256=step_sha, log_path=log_path, step_count=len(step_rows), eval_count=len(eval_rows), parent_checkpoint_sha256=initial_checkpoint_sha256)
        if decoded_evaluator is not None and best_row is not None and int(best_row.get("step", -1)) == int(step):
            _atomic_copy(state_path, output_dir / "best_dev_training_state.pt")
    _write_csv(csv_path, step_rows)
    _plot_training_curves(output_dir, step_rows, eval_rows)
    quality_by_step = {int(row["step"]): row for row in eval_rows}
    checkpoint_index = []
    for checkpoint in sorted(output_dir.glob(f"{config.kind}_step_[0-9][0-9][0-9][0-9].pt")):
        checkpoint_step = int(checkpoint.stem.rsplit("_", 1)[-1])
        checkpoint_index.append({
            "step": checkpoint_step,
            "adapter": str(checkpoint.resolve()),
            "training_state": str((output_dir / f"{config.kind}_step_{checkpoint_step:04d}_training_state.pt").resolve()),
            "sha256": _sha256(checkpoint),
            "config_sha256": _config_hash(config),
            "parent_checkpoint_sha256": initial_checkpoint_sha256,
            "quality": quality_by_step.get(checkpoint_step),
            "is_best": best_row is not None and int(best_row.get("step", -1)) == checkpoint_step,
        })
    write_checkpoint_index(output_dir, _config_hash(config), eval_rows, None if best_row is None else int(best_row["step"]))
    return {
        "start_step": start_step,
        "steps": int(step),
        "losses": [float(row["loss"]) for row in step_rows],
        "step_records": step_rows,
        "periodic_evaluations": latent_evaluations,
        "decoded_evaluations": eval_rows,
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason,
        "best_checkpoint": None if best_row is None else str((output_dir / "best_dev.pt").resolve()),
        "sigma_cycle": {"cycle": sigma_cycle.cycle, "position": sigma_cycle.position, "values_per_cycle": len(sigma_cycle.values), "strategy": config.sampling_strategy},
    }, optimizer, int(step)


def _build_decoded_evaluator(
    config: ExperimentConfig,
    items: list[CachedItem],
    vae: WanVAE,
    dit: WanDiT,
    conditioner: FrozenLQConditioner,
    context: Tensor,
    output_dir: Path,
) -> Any:
    if config.kind != "3d" or not config.quality_overfit:
        return None
    from stage1_decoded_eval import EvalItem, decode, evaluate_strict_3d_overfit, upsample_lr
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", reduction="mean", normalize=False).to(dit.device).eval().requires_grad_(False)
    eval_items = []
    for item in items:
        clean = item.clean.to(dit.device)
        with torch.inference_mode():
            ceiling = decode(vae, "3d", clean)
        eval_items.append(EvalItem(
            item_id=item.item_id,
            hr=item.hr,
            lr=item.lr,
            clean=item.clean,
            features=item.features,
            shuffled_features=item.shuffled_features,
            neutral_features=item.neutral_features,
            bicubic=upsample_lr(item.lr, (config.hr_resolution, config.hr_resolution)),
            vae_ceiling=ceiling,
        ))

    def evaluate(step: int, checkpoint_path: Path) -> dict[str, Any]:
        eval_dir = output_dir / f"eval_{step:04d}_dev"
        result = evaluate_strict_3d_overfit(
            eval_items[0], vae, dit, conditioner, context, metric.net,
            noise_seeds=(1201,),
            sampling=FlowSamplingConfig(steps=config.sampling_steps, shift=config.sampling_shift),
            output_dir=eval_dir,
        )
        rows = result["metric_rows"]
        means = {
            condition: {
                name: float(np.mean([float(row[name]) for row in rows if row["condition"] == condition]))
                for name in ("psnr", "ssim", "lpips")
            }
            for condition in ("correct", "bicubic", "vae_ceiling", "shuffled", "disabled")
        }
        quality = {
            "step": step,
            "checkpoint": str(checkpoint_path.resolve()),
            "correct_psnr": means["correct"]["psnr"],
            "correct_ssim": means["correct"]["ssim"],
            "correct_lpips": means["correct"]["lpips"],
            "bicubic_psnr": means["bicubic"]["psnr"],
            "bicubic_ssim": means["bicubic"]["ssim"],
            "bicubic_lpips": means["bicubic"]["lpips"],
            "vae_ceiling_psnr": means["vae_ceiling"]["psnr"],
            "correct_psnr_margin": means["correct"]["psnr"] - means["bicubic"]["psnr"],
            "correct_ssim_margin": means["correct"]["ssim"] - means["bicubic"]["ssim"],
            "correct_lpips_reduction": (means["bicubic"]["lpips"] - means["correct"]["lpips"]) / means["bicubic"]["lpips"],
            "correct_shuffled_psnr_gap": means["correct"]["psnr"] - means["shuffled"]["psnr"],
            "correct_shuffled_ssim_gap": means["correct"]["ssim"] - means["shuffled"]["ssim"],
            "correct_shuffled_lpips_reduction": (means["shuffled"]["lpips"] - means["correct"]["lpips"]) / means["shuffled"]["lpips"],
            "correct_disabled_psnr_gap": means["correct"]["psnr"] - means["disabled"]["psnr"],
            "correct_disabled_ssim_gap": means["correct"]["ssim"] - means["disabled"]["ssim"],
            "correct_disabled_lpips_reduction": (means["disabled"]["lpips"] - means["correct"]["lpips"]) / means["disabled"]["lpips"],
            "decoded_quality_pass": bool(
                means["correct"]["psnr"] > means["bicubic"]["psnr"]
                and means["correct"]["ssim"] > means["bicubic"]["ssim"]
                and means["correct"]["lpips"] < means["bicubic"]["lpips"]
            ),
            "dev_candidate_pass": bool(result["development_verdict"]["passed"]),
            "strict_overfit_pass": False,
            "final_evaluation_status": "not_run",
        }
        for condition, metrics in means.items():
            for metric_name, value in metrics.items():
                quality[f"{condition}_{metric_name}"] = value
        conditioner.train()
        return quality

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("3d", "4d"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--lq-source", type=Path, required=True)
    parser.add_argument("--lq-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--hr-resolution", type=int, default=128)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--sampling-shift", type=float, default=5.0)
    parser.add_argument("--sampling-strategy", choices=("permutation", "balanced"), default="permutation")
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--lr-decay-step", type=int, default=750)
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--quality-overfit", action="store_true")
    parser.add_argument("--bridge-blocks", type=int, nargs="+", default=(0,))
    parser.add_argument("--bridge-time-conditioning", action="store_true")
    parser.add_argument("--stop-on-dev-pass", action="store_true")
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = ExperimentConfig(
        kind=args.kind,
        hr_resolution=args.hr_resolution,
        views=args.views,
        sample_count=args.sample_count,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        sampling_steps=args.sampling_steps,
        sampling_shift=args.sampling_shift,
        sampling_strategy=args.sampling_strategy,
        checkpoint_interval=args.checkpoint_interval,
        lr_decay_step=args.lr_decay_step,
        lr_decay_factor=args.lr_decay_factor,
        max_hours=args.max_hours,
        quality_overfit=args.quality_overfit,
        bridge_blocks=tuple(args.bridge_blocks),
        bridge_time_conditioning=args.bridge_time_conditioning,
        stop_on_dev_pass=args.stop_on_dev_pass,
    )
    if config.quality_overfit and (
        config.kind != "3d" or config.hr_resolution != 256
        or config.views != 4 or config.sample_count != 1
    ):
        raise ValueError("quality overfit requires 3d, 256x256, V=4 and one sample")
    if args.resume_state is not None and args.init_adapter is None:
        raise ValueError("resume-state requires the matching init-adapter checkpoint")
    if min(config.eval_interval, config.checkpoint_interval, config.max_steps) < 1:
        raise ValueError("step counts must be positive")
    _seed_all(config.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage 1 real Wan experiments require CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_state is None and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty; use a new experiment directory: {args.output_dir}")
    torch.cuda.reset_peak_memory_stats(device)

    projector = load_flashvsr_projector(
        args.lq_source,
        args.lq_checkpoint,
        device=device,
        dtype=torch.bfloat16,
        expected_sha256=EXPECTED_LQ_SHA256,
    )
    conditioner = FrozenLQConditioner(projector, bridge_blocks=config.bridge_blocks, bridge_time_conditioning=config.bridge_time_conditioning).to(device)
    vae = WanVAE.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    context = torch.zeros(1, 512, 4096, device=device, dtype=torch.bfloat16)
    parameters = _parameter_report(vae, dit, conditioner)
    rgb_items = _prepare_rgb(config, args.scene)
    items = _cache_items(config, rgb_items, vae, conditioner, device)
    del rgb_items
    gc.collect()
    torch.cuda.empty_cache()

    config_sha256 = _config_hash(config)
    wan_sha256 = _sha256_tree(args.model_dir)
    scene_sha256 = _sha256_tree(args.scene)
    lq_sha256 = _sha256(args.lq_checkpoint)

    manifest = {
        "schema_version": 1,
        "created_unix_time": time.time(),
        "config": asdict(config),
        "config_sha256": config_sha256,
        "git_revision": _git_revision(),
        "software": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "experiment_identity": {
            "model_checkpoint": str(args.model_dir.resolve()),
            "model_checkpoint_sha256": wan_sha256,
            "lq_checkpoint": str(args.lq_checkpoint.resolve()),
            "lq_sha256": lq_sha256,
            "scene": str(args.scene.resolve()),
            "scene_sha256": scene_sha256,
        },
        "items": [{"id": item.item_id, "hr_shape": list(item.hr.shape), "lr_shape": list(item.lr.shape)} for item in items],
        "degradation": {"type": "bicubic", "scale": config.scale, "antialias": True},
        "data_policy": "source data remains in place; only identifiers and shapes are recorded",
    }
    source_identity = snapshot_source(args.output_dir)
    manifest["source_identity"] = source_identity
    manifest_path = args.output_dir / "run_manifest.json"
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    _append_jsonl(args.output_dir / "launches.jsonl", {
        "event": "start", "unix_time": time.time(), "config": asdict(config),
        "source_identity": source_identity,
        "parent_checkpoint": None if args.init_adapter is None else str(args.init_adapter.resolve()),
        "parent_checkpoint_sha256": None if args.init_adapter is None else _sha256(args.init_adapter),
        "resume": args.resume_state is not None,
    })

    pretraining = _baseline_and_gradient_audit(config, items, vae, dit, conditioner, context)
    if config.quality_overfit:
        oracle_generator = torch.Generator(device=device).manual_seed(1234)
        oracle_clean = items[0].clean.to(device).float()
        oracle_noise = torch.randn(oracle_clean.shape, device=device, generator=oracle_generator)
        oracle = oracle_sampling_audit(oracle_clean, oracle_noise, config=FlowSamplingConfig(steps=config.sampling_steps, shift=config.sampling_shift))
        oracle['passed'] = oracle['final_max_abs_error'] < 1e-3 and oracle['final_mean_abs_error'] < 3e-4
        atomic_json(args.output_dir / 'sampler_oracle.json', oracle)
        if not oracle['passed']:
            raise RuntimeError(f'Oracle sampler gate failed: {oracle}')
        pretraining['sampler_oracle'] = {key: value for key, value in oracle.items() if key != 'rows'}
    print(json.dumps({"event": "pretraining_audit", **pretraining}), flush=True)
    initial_checkpoint_sha256 = None
    initial_checkpoint_metadata = None
    if args.init_adapter is not None:
        initial_checkpoint_sha256 = _sha256(args.init_adapter)
        initial_checkpoint_metadata = load_adapter_checkpoint(
            args.init_adapter,
            conditioner,
            expected_experiment={
                "model_checkpoint": str(args.model_dir.resolve()),
                "lq_sha256": EXPECTED_LQ_SHA256,
                "model_checkpoint_sha256": wan_sha256,
                "scene_sha256": scene_sha256,
            },
        )
        if initial_checkpoint_metadata["config"].get("kind") != config.kind:
            raise RuntimeError("initial adapter kind does not match requested experiment")
        print(json.dumps({
            "event": "warm_start", "path": str(args.init_adapter.resolve()),
            "sha256": initial_checkpoint_sha256,
        }), flush=True)

    experiment_identity = {
        "seed": config.seed,
        "model_checkpoint": str(args.model_dir.resolve()),
        "model_checkpoint_sha256": wan_sha256,
        "scene_sha256": scene_sha256,
        "source_identity": source_identity,
        "lq_checkpoint": str(args.lq_checkpoint.resolve()),
        "lq_reference_source": str(args.lq_source.resolve()),
        "lq_sha256": EXPECTED_LQ_SHA256,
        "config_sha256": config_sha256,
    }
    decoded_evaluator = _build_decoded_evaluator(
        config, items, vae, dit, conditioner, context, args.output_dir
    )
    training, optimizer, step = _train(
        config, items, dit, conditioner, context, args.output_dir,
        resume_state=args.resume_state,
        experiment_identity=experiment_identity,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        decoded_evaluator=decoded_evaluator,
    )
    final_evaluation = _evaluate(items, dit, conditioner, context, sigmas=(0.2, 0.5, 0.8, 0.95), seed_base=201)

    checkpoint_path = args.output_dir / f"{config.kind}_adapter.pt"
    _save_adapter_atomic(
        checkpoint_path,
        conditioner,
        config=config,
        experiment={
            **experiment_identity,
            "step": step,
            "parent_checkpoint": None if args.init_adapter is None else str(args.init_adapter.resolve()),
            "parent_checkpoint_sha256": initial_checkpoint_sha256,
        },
    )

    item = items[0]
    with torch.no_grad():
        before, _, _ = _forward_case(dit, conditioner, item.clean, item.features, sigma=0.8, noise_seed=999, context=context)
        conditioner.reset_bridge()
        load_adapter_checkpoint(
            checkpoint_path,
            conditioner,
            expected_config=asdict(config),
            expected_experiment={
                "model_checkpoint": str(args.model_dir.resolve()),
                "lq_sha256": EXPECTED_LQ_SHA256,
            },
        )
        after, _, _ = _forward_case(dit, conditioner, item.clean, item.features, sigma=0.8, noise_seed=999, context=context)
    reload_delta = (before - after).abs().float()
    reload_equivalence = {
        "max_abs_diff": float(reload_delta.max()),
        "mean_abs_diff": float(reload_delta.mean()),
        "passed": bool(torch.equal(before, after)),
    }

    decoded_rows = training.get("decoded_evaluations", [])
    decoded_quality_pass = bool(decoded_rows and any(row.get("decoded_quality_pass") for row in decoded_rows))
    report = {
        "status": (
            "PASS"
            if decoded_rows and decoded_rows[-1].get("strict_overfit_pass") and reload_equivalence["passed"]
            else ("DEV_PASS_FINAL_PENDING" if any(r.get("dev_candidate_pass") for r in decoded_rows) else "CAUSAL_PASS_QUALITY_FAIL" if config.quality_overfit and decoded_rows and final_evaluation["passed"] else ("READY_FOR_QUALITY_EVAL" if final_evaluation["passed"] and reload_equivalence["passed"] else "PARTIAL"))
        ),
        "kind": config.kind,
        "config": asdict(config),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "model_checkpoint": str(args.model_dir.resolve()),
            "lq_checkpoint": str(args.lq_checkpoint.resolve()),
            "lq_reference_source": str(args.lq_source.resolve()),
            "lq_sha256": EXPECTED_LQ_SHA256,
            "text_condition": "deterministic all-zero [B,512,4096] BF16 context",
        },
        "architecture": {
            "degradation": "4x bicubic antialiased HR->LR",
            "conditioning_resolution": [config.hr_resolution, config.hr_resolution],
            "lq_encoder": "frozen FlashVSR Causal_LQ4x_Proj, layer_num=1",
            "temporal_alignment": "four outer first-frame warm-up copies; native sequence otherwise intact",
            "injection": {"blocks": list(config.bridge_blocks), "time_conditioning": config.bridge_time_conditioning, "projection": "FP32 zero Linear(1536,1536) per block"},
            "3d_alignment": "each LR view projected independently, regrouped as F/H/W Wan tokens",
            "4d_alignment": "native video projected jointly; no frame flattening",
        },
        "parameters": parameters,
        "items": [
            {"id": item.item_id, "latent_shape": list(item.clean.shape), "feature_shape": list(item.features.shape)}
            for item in items
        ],
        "pretraining": pretraining,
        "training": training,
        "initial_checkpoint": {
            "path": None if args.init_adapter is None else str(args.init_adapter.resolve()),
            "sha256": initial_checkpoint_sha256,
            "metadata": initial_checkpoint_metadata,
            "optimizer_reinitialized": args.resume_state is None,
        },
        "final_evaluation": final_evaluation,
        "quality_gates": {
            "causal_pass": bool(final_evaluation["passed"]),
            "decoded_quality_pass": decoded_quality_pass,
            "strict_overfit_pass": bool(decoded_rows and decoded_rows[-1].get("strict_overfit_pass")),
        },
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_reload": reload_equivalence,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    report_path = args.output_dir / f"{config.kind}_result.json"
    atomic_json(report_path, report)
    _append_jsonl(args.output_dir / "launches.jsonl", {
        "event": "complete", "unix_time": time.time(), "step": step,
        "stop_reason": training["stop_reason"], "elapsed_training_seconds": training["elapsed_seconds"],
        "status": report["status"], "source_identity": source_identity,
    })
    print(json.dumps({"event": "complete", "kind": config.kind, "status": report["status"], "result": str(report_path), "final": final_evaluation, "reload": reload_equivalence, "peak_mib": report["peak_gpu_memory_mib"]}), flush=True)


if __name__ == "__main__":
    main()
