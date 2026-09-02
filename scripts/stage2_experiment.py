#!/usr/bin/env python3
"""Run a bounded frozen-Wan Stage 2 geometry adapter experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rl3dsr.data import NeRFSyntheticAdapter, Split
from rl3dsr.data.tensor_prep import rgb_images_to_video
from rl3dsr.models.wan import (
    CameraBatch,
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
from rl3dsr.models.wan.sampling import SigmaCycle, FlowSamplingConfig
from rl3dsr.validation.decoded_space import frame_metrics


@dataclass(frozen=True, slots=True)
class Stage2Config:
    representation: str = "rre"
    seed: int = 42
    hr_resolution: int = 256
    scale: int = 4
    steps: int = 500
    eval_interval: int = 50
    checkpoint_interval: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    sampling_steps: int = 50
    sampling_shift: float = 5.0
    blocks: tuple[int, ...] = (0, 1, 2, 3)
    hidden_dim: int = 192
    attention_heads: int = 4
    reference_index: int = 0
    decoded_eval: bool = True


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _camera_from_sequence(sequence, resolution: int, device: torch.device, reference_index: int) -> CameraBatch:
    intrinsics = []
    transforms = []
    for observation in sequence.observations:
        scale_x = resolution / observation.width
        scale_y = resolution / observation.height
        k = torch.from_numpy(np.asarray(observation.K, dtype=np.float32)).clone()
        k[0] *= scale_x
        k[1] *= scale_y
        intrinsics.append(k)
        transforms.append(torch.from_numpy(np.asarray(observation.T_world_from_camera, dtype=np.float32)).clone())
    return CameraBatch(
        torch.stack(intrinsics).unsqueeze(0).to(device),
        torch.stack(transforms).unsqueeze(0).to(device),
        (resolution, resolution),
        "multiview",
        reference_index,
    )


def _load_item(args, config: Stage2Config, device: torch.device):
    adapter = NeRFSyntheticAdapter(args.scene)
    sequence = adapter.index(Split.TRAIN)
    selected = tuple(sequence.observations[index] for index in args.view_indices)
    images = [adapter.load_rgb(item) for item in selected]
    hr = rgb_images_to_video(images, config.hr_resolution).to(device)
    camera = _camera_from_sequence(
        type("SelectedSequence", (), {"observations": selected})(),
        config.hr_resolution,
        device,
        config.reference_index,
    )
    lr = Stage1Degradation(scale=config.scale)(hr)
    return sequence.scene_id, hr, lr, camera


def _permuted_camera(camera: CameraBatch) -> CameraBatch:
    order = torch.roll(torch.arange(camera.K.shape[1], device=camera.K.device), shifts=1)
    return CameraBatch(camera.K, camera.T_world_from_camera[:, order], camera.image_size, camera.sequence_kind, camera.reference_index)


def _save_checkpoint(path: Path, geometry: GeometryConditioner, config: Stage2Config, parent: Path, step: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "geometry": {name: value.detach().cpu() for name, value in geometry.state_dict().items()},
            "config": asdict(config),
            "parent_checkpoint": str(parent.resolve()),
            "parent_checkpoint_sha256": _sha256(parent),
            "step": step,
        },
        temporary,
    )
    temporary.replace(path)


def _mean_metrics(decoded: torch.Tensor, hr: torch.Tensor, perceptual_metric=None) -> dict[str, float]:
    rows = frame_metrics(decoded.detach().float().cpu(), hr.detach().float().cpu(), perceptual_metric=perceptual_metric)
    names = ("psnr", "ssim") + (("lpips",) if rows and "lpips" in rows[0] else ())
    return {name: float(np.mean([row[name] for row in rows])) for name in names}


class _PerFrameLPIPS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        self.metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", reduction="mean", normalize=False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.metric(prediction[index:index + 1], target[index:index + 1]).reshape(1) for index in range(prediction.shape[0])])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--lq-source", type=Path, required=True)
    parser.add_argument("--lq-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=("rre", "plucker"), default="rre")
    parser.add_argument("--view-indices", type=int, nargs=4, default=(0, 33, 66, 99))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--no-decoded-eval", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = Stage2Config(
        representation=args.representation,
        steps=args.steps,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        decoded_eval=not args.no_decoded_eval,
    )
    _seed(config.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage 2 experiments require CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError("output directory must be empty")

    projector = load_flashvsr_projector(args.lq_source, args.lq_checkpoint, device=device, dtype=torch.bfloat16)
    conditioner = FrozenLQConditioner(
        projector,
        bridge_blocks=(0, 1, 2, 3),
        bridge_time_conditioning=True,
    ).to(device)
    load_adapter_checkpoint(args.parent_checkpoint, conditioner)
    conditioner.bridge.requires_grad_(False)
    vae = WanVAE.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(args.model_dir, device=device, dtype=torch.bfloat16)
    scene_id, hr, lr, camera = _load_item(args, config, device)
    with torch.inference_mode():
        clean = vae.encode_multiview(hr)
        features = conditioner.multiview_features(
            lr,
            conditioning_size=(config.hr_resolution, config.hr_resolution),
            latent_shape=tuple(clean.shape[2:]),
        )
    geometry = GeometryConditioner(
        representation=config.representation,
        hidden_dim=config.hidden_dim,
        attention_heads=config.attention_heads,
        blocks=config.blocks,
    ).to(device)
    trainable = [parameter for parameter in geometry.parameters() if parameter.requires_grad]
    perceptual_metric = None
    if config.decoded_eval:
        try:
            perceptual_metric = _PerFrameLPIPS().to(device).eval().requires_grad_(False)
        except ImportError:
            perceptual_metric = None
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    sigma_cycle = SigmaCycle(FlowSamplingConfig(config.sampling_steps, config.sampling_shift), seed=config.seed + 1000, strategy="balanced")
    noise_generator = torch.Generator(device=device).manual_seed(config.seed + 2000)
    torch.save({"K": camera.K.cpu(), "T_world_from_camera": camera.T_world_from_camera.cpu(), "view_indices": list(args.view_indices), "scene": scene_id}, args.output_dir / "camera_cache.pt")
    manifest = {"config": asdict(config), "scene": str(args.scene.resolve()), "scene_id": scene_id, "view_indices": list(args.view_indices), "parent_checkpoint": str(args.parent_checkpoint.resolve()), "parent_checkpoint_sha256": _sha256(args.parent_checkpoint), "git_revision": _git_revision(), "python": platform.python_version(), "torch": torch.__version__, "latent_shape": list(clean.shape), "feature_shape": list(features.shape), "trainable_parameters": sum(parameter.numel() for parameter in trainable), "frozen_dit_parameters": sum(parameter.numel() for parameter in dit.model.parameters()), "frozen_vae_parameters": sum(parameter.numel() for parameter in vae.model.model.parameters())}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows: list[dict] = []
    eval_rows: list[dict] = []
    for step in range(1, config.steps + 1):
        clean_step = clean
        sigma = sigma_cycle.next().reshape(1).to(device)
        noise = torch.randn(clean_step.shape, generator=noise_generator, device=device, dtype=clean_step.dtype)
        noisy, timestep, target = flow_matching_pair(clean_step, noise, sigma)
        optimizer.zero_grad(set_to_none=True)
        prediction = conditioned_prediction(dit, conditioner, noisy, timestep, None, features, geometry_adapter=geometry, camera=camera, latent_shape=tuple(clean.shape[2:]))
        loss = flow_matching_loss(prediction, target)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip))
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in trainable):
            raise RuntimeError("geometry adapter gradient is missing or non-finite")
        if any(parameter.grad is not None for parameter in dit.model.parameters()):
            raise RuntimeError("frozen DiT received a gradient")
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()), "sigma": float(sigma), "gradient_norm": gradient_norm, "trainable_parameters": sum(parameter.numel() for parameter in trainable), "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        rows.append(row)
        _append_jsonl(args.output_dir / "train_steps.jsonl", row)
        if step % config.eval_interval == 0 or step == 1:
            with torch.inference_mode():
                eval_sigma = torch.tensor([0.5], device=device)
                eval_noise = torch.randn(clean.shape, generator=torch.Generator(device=device).manual_seed(9000), device=device, dtype=clean.dtype)
                eval_noisy, eval_timestep, eval_target = flow_matching_pair(clean, eval_noise, eval_sigma)
                correct = conditioned_prediction(dit, conditioner, eval_noisy, eval_timestep, None, features, geometry_adapter=geometry, camera=camera, latent_shape=tuple(clean.shape[2:]))
                shuffled = conditioned_prediction(dit, conditioner, eval_noisy, eval_timestep, None, features, geometry_adapter=geometry, camera=_permuted_camera(camera), latent_shape=tuple(clean.shape[2:]))
                baseline = conditioned_prediction(dit, conditioner, eval_noisy, eval_timestep, None, features)
                evaluation = {"step": step, "correct_loss": float(flow_matching_loss(correct, eval_target)), "shuffled_loss": float(flow_matching_loss(shuffled, eval_target)), "baseline_loss": float(flow_matching_loss(baseline, eval_target)), "correct_shuffled_delta": float((correct - shuffled).float().abs().mean()), "correct_baseline_delta": float((correct - baseline).float().abs().mean())}
                if config.decoded_eval:
                    estimate = eval_noisy - eval_sigma.reshape(1, 1, 1, 1, 1) * correct
                    baseline_estimate = eval_noisy - eval_sigma.reshape(1, 1, 1, 1, 1) * baseline
                    evaluation.update({f"correct_{key}": value for key, value in _mean_metrics(vae.decode_multiview(estimate), hr, perceptual_metric).items()})
                    evaluation.update({f"baseline_{key}": value for key, value in _mean_metrics(vae.decode_multiview(baseline_estimate), hr, perceptual_metric).items()})
                eval_rows.append(evaluation)
                _append_jsonl(args.output_dir / "geometry_metrics.jsonl", evaluation)
        if step % config.checkpoint_interval == 0 or step == config.steps:
            _save_checkpoint(args.output_dir / f"geometry_step_{step:04d}.pt", geometry, config, args.parent_checkpoint, step)
            torch.save({"step": step, "optimizer": optimizer.state_dict(), "sigma_cycle": sigma_cycle.state_dict(), "noise_generator_state": noise_generator.get_state()}, args.output_dir / f"geometry_step_{step:04d}_training_state.pt")
    _write_csv(args.output_dir / "train_steps.csv", rows)
    _write_csv(args.output_dir / "geometry_metrics.csv", eval_rows)
    _save_checkpoint(args.output_dir / "geometry_final.pt", geometry, config, args.parent_checkpoint, config.steps)
    reloaded = GeometryConditioner(representation=config.representation, hidden_dim=config.hidden_dim, attention_heads=config.attention_heads, blocks=config.blocks).to(device)
    load_geometry_checkpoint(args.output_dir / "geometry_final.pt", reloaded, expected_parent_checkpoint=str(args.parent_checkpoint.resolve()))
    reload_equal = all(torch.equal(first, second) for first, second in zip(geometry.parameters(), reloaded.parameters()))
    (args.output_dir / "result.json").write_text(json.dumps({"config": asdict(config), "steps": config.steps, "trainable_parameters": sum(parameter.numel() for parameter in trainable), "final_eval": eval_rows[-1] if eval_rows else None, "checkpoint_reload": {"parameter_exact": reload_equal}}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
