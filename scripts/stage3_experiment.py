#!/usr/bin/env python3
"""Stage 3 shared-training and evaluation entry points.

Nothing runs at import time. ``inspect`` and ``freeze`` do not load model
assets; the remaining commands require explicit paths and are intended to be
started later by the experiment operator.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from functools import lru_cache
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from rl3dsr.data import MipNeRF360Adapter, NeRFSyntheticAdapter, Split
from rl3dsr.data.tensor_prep import rgb_images_to_video
from rl3dsr.models.wan import (
    CameraBatch,
    FlowSamplingConfig,
    FrozenLQConditioner,
    FullRREConditioner,
    LRViewFusion,
    Stage1Degradation,
    Stage3Conditioning,
    WanDiT,
    WanVAE,
    flow_matching_loss,
    flow_matching_pair,
    load_adapter_checkpoint,
    load_flashvsr_projector,
    load_geometry_checkpoint,
    load_stage3_checkpoint,
    sample_conditioned_flow,
    save_stage3_checkpoint,
)
from rl3dsr.models.wan.stage3 import (
    load_stage3_initialization_checkpoint,
    validate_stage3_checkpoint_payload,
)
from rl3dsr.models.wan.lr_fusion import pairing_info_nce_loss
from rl3dsr.models.wan.sampling import SigmaCycle
from rl3dsr.validation.decoded_space import frame_metrics
from rl3dsr.validation.stage3_protocol import (
    Stage3Config,
    claim_final_evaluation,
    freeze_candidate,
    evenly_spaced_indices,
    intervene_lr,
    load_stage3_config,
    nearest_view_indices,
    sample_view_indices,
    shuffle_auxiliary_pairs,
    select_candidate,
)


class Runtime:
    def __init__(self, module, vae, dit, device, checkpoint=None):
        self.module = module
        self.vae = vae
        self.dit = dit
        self.device = device
        self.checkpoint = checkpoint


def scene_routes(config: Stage3Config, phase: str) -> list[tuple[str, str]]:
    if phase == "train":
        return [(scene, "train") for scene in config.train_scenes]
    if phase == "validate":
        return [(scene, "val") for scene in config.validation_scene_names]
    if phase == "test":
        return [(scene, "test") for scene in config.test_scenes]
    raise ValueError("phase must be train, validate or test")


def prepare_output(path: str | Path) -> Path:
    path = Path(path).resolve()
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("output directory must be empty")
    else:
        path.mkdir(parents=True)
    return path


def resume_rows(path: str | Path, step: int) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise ValueError("resume log is missing")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [row.get("step") for row in rows]
    if not rows or steps[-1] != step or steps != list(range(1, step + 1)):
        raise ValueError("resume log does not match checkpoint step")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value) -> str:
    """Hash JSON-compatible data using one stable canonical encoding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _initialization_module_policy(reset_fusion: bool) -> tuple[list[str], list[str]]:
    if reset_fusion:
        return ["bridge", "geometry"], ["fusion"]
    return ["bridge", "geometry", "fusion"], []


def _initialization_provenance(path: Path, payload: dict, *, reset_fusion: bool = False) -> dict:
    """Describe a model-only parent without carrying over its run state."""
    copied_modules, reset_modules = _initialization_module_policy(reset_fusion)
    return {
        "initialization_mode": "model_only",
        "parent_checkpoint": str(Path(path).resolve()),
        "parent_checkpoint_sha256": _sha256(Path(path)),
        "parent_saved_step": int(payload["step"]),
        "parent_saved_config_sha256": _json_sha256(payload.get("config") or {}),
        "reset_fusion": bool(reset_fusion),
        "copied_modules": copied_modules,
        "reset_modules": reset_modules,
    }


def _assert_manifest_initialization(provenance: dict, mode: str, reset_fusion: bool) -> None:
    """Fail before manifest write if its initialization declaration is ambiguous."""
    expected_copied, expected_reset = (
        _initialization_module_policy(reset_fusion)
        if mode == "model_only" else ([], [])
    )
    if provenance.get("initialization_mode") != mode:
        raise RuntimeError("manifest initialization mode mismatch")
    if provenance.get("copied_modules") != expected_copied:
        raise RuntimeError("manifest copied_modules mismatch")
    if provenance.get("reset_modules") != expected_reset:
        raise RuntimeError("manifest reset_modules mismatch")


def _initialization_mode(resume, init_checkpoint, reset_fusion: bool) -> str:
    if resume is not None and init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if reset_fusion and init_checkpoint is None:
        raise ValueError("--init-reset-fusion requires --init-checkpoint")
    return "resume" if resume is not None else "model_only" if init_checkpoint is not None else "scratch"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _module_grad_norm(module) -> float | None:
    if module is None:
        return None
    values = [parameter.grad.detach().float().square().sum()
              for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else None


def _sampling_signature(config: Stage3Config) -> dict:
    signature = {
        "train_scenes": list(config.train_scenes),
        "image_size": config.image_size,
        "scale": config.scale,
        "views": config.views,
    }
    if config.dataset_kind == "mipnerf360":
        signature.update(dataset_kind=config.dataset_kind, image_factor=config.image_factor)
    return signature


@lru_cache(maxsize=8)
def _mip_scene_data(root: Path, factor: int, split: Split):
    adapter = MipNeRF360Adapter(root, image_factor=factor)
    return adapter, adapter.index(split)


def _scene_data(root: Path, config: Stage3Config, split: Split):
    if config.dataset_kind == "mipnerf360":
        return _mip_scene_data(root, config.image_factor, split)
    adapter = NeRFSyntheticAdapter(root)
    return adapter, adapter.index(split)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        stream.flush()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({name for row in rows for name in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_runtime(
    config: Stage3Config,
    *,
    model_dir: Path,
    lq_source: Path,
    lq_checkpoint: Path,
    bridge_checkpoint: Path,
    device: str | torch.device,
    rre_checkpoint: Path | None = None,
    stage3_checkpoint: Path | None = None,
    model_only_initialization: bool = False,
    reset_initialization_fusion: bool = False,
) -> Runtime:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage 3 model execution requires CUDA")
    projector = load_flashvsr_projector(
        lq_source, lq_checkpoint, device=device, dtype=torch.bfloat16
    )
    conditioner = FrozenLQConditioner(
        projector,
        bridge_blocks=(0, 1, 2, 3),
        bridge_time_conditioning=True,
    ).to(device)
    load_adapter_checkpoint(bridge_checkpoint, conditioner)
    vae = WanVAE.from_checkpoint(model_dir, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(model_dir, device=device, dtype=torch.bfloat16)
    geometry = FullRREConditioner(branch_count=len(dit.model.blocks)).to(device)
    if rre_checkpoint is not None:
        load_geometry_checkpoint(
            rre_checkpoint,
            geometry,
            expected_parent_checkpoint=str(bridge_checkpoint.resolve()),
        )
    fusion = None
    if config.fusion_mode != "off":
        fusion_mode = config.fusion_mode
        if fusion_mode == "epipolar" and config.epipolar_attention == "local_band":
            fusion_mode = "epipolar_local"
        fusion = LRViewFusion(
            hidden_dim=config.fusion_dim,
            heads=config.fusion_heads,
            mode=fusion_mode,
            query_chunk_size=config.query_chunk_size,
            tau=config.epipolar_tau,
            epipolar_band=config.epipolar_band,
            allow_self_view_source=config.allow_self_view_source,
        ).to(device)
    module = Stage3Conditioning(conditioner, geometry, fusion).to(device)
    payload = None
    if stage3_checkpoint is not None:
        if model_only_initialization:
            payload = load_stage3_initialization_checkpoint(
                stage3_checkpoint, module, reset_fusion=reset_initialization_fusion
            )
        else:
            payload = load_stage3_checkpoint(
                stage3_checkpoint, module, expected_config=config.to_dict()
            )
    return Runtime(module, vae, dit, device, payload)


def _camera_digest(camera: CameraBatch) -> str:
    """Hash canonical camera tensors for intervention provenance."""
    camera.validate()
    digest = hashlib.sha256()
    for value in (camera.K, camera.T_world_from_camera):
        tensor = value.detach().float().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    digest.update(str(camera.image_size).encode("ascii"))
    digest.update(camera.sequence_kind.encode("ascii"))
    digest.update(str(camera.reference_index).encode("ascii"))
    digest.update(camera.camera_model.encode("ascii"))
    if camera.xi is not None:
        xi = camera.xi.detach().float().cpu().contiguous()
        digest.update(str(tuple(xi.shape)).encode("ascii"))
        digest.update(xi.numpy().tobytes())
    return digest.hexdigest()


def _scalar_diagnostics(payload) -> dict[str, dict[str, float]]:
    """Convert module diagnostics to JSON-safe scalar summaries."""
    result: dict[str, dict[str, float]] = {}
    if not isinstance(payload, dict):
        return result
    for block, values in payload.items():
        if not isinstance(values, dict):
            continue
        converted = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                converted[str(name)] = float(value)
        if converted:
            result[str(block)] = converted
    return result


def sample_latents(
    runtime: Runtime,
    lr: torch.Tensor,
    camera: CameraBatch,
    initial_shape: tuple[int, int, int, int, int],
    sampling_steps: int,
    image_size: int,
    *,
    fusion_camera: CameraBatch | None = None,
    geometry_camera: CameraBatch | None = None,
    source_mask: torch.Tensor | None = None,
    allow_self_view_source: bool | None = None,
    sampler=sample_conditioned_flow,
    seed: int = 0,
    sampling_shift: float = 5.0,
    dtype: torch.dtype = torch.bfloat16,
    return_diagnostics: bool = False,
    view_permutation: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    latent_shape = tuple(initial_shape[2:])
    fusion_camera = camera if fusion_camera is None else fusion_camera
    geometry_camera = camera if geometry_camera is None else geometry_camera
    fusion_module = getattr(runtime.module, "fusion", None)
    if fusion_module is not None:
        fusion_module.record_diagnostics = bool(return_diagnostics)
    prepared = runtime.module.prepare_multiview(
        lr,
        fusion_camera,
        latent_shape,
        (image_size, image_size),
        source_mask=source_mask,
        allow_self_view_source=allow_self_view_source,
    )
    generator = torch.Generator(device=runtime.device).manual_seed(seed)
    noise = torch.randn(initial_shape, generator=generator, device=runtime.device, dtype=dtype)
    inverse_permutation = None
    if view_permutation is not None:
        view_permutation = _validate_view_permutation(view_permutation, initial_shape[2])
        inverse_permutation = inverse_view_permutation(view_permutation)
        noise = permute_view_tensor(noise, view_permutation)
    context = torch.zeros(
        initial_shape[0], 512, 4096, device=runtime.device, dtype=dtype
    )
    velocity_trace: list[torch.Tensor] = []
    geometry_trace: list[dict[str, dict[str, float]]] = []
    injection_trace: list[dict[str, dict[str, float]]] = []

    def predict_velocity(sample, timestep, text_context, features):
        velocity = runtime.module.predict(
            runtime.dit,
            sample,
            timestep,
            text_context,
            features,
            geometry_camera,
            latent_shape,
        )
        if return_diagnostics:
            traced = velocity if inverse_permutation is None else permute_view_tensor(
                velocity, inverse_permutation
            )
            velocity_trace.append(traced.detach().float().cpu())
            geometry_trace.append(_scalar_diagnostics(getattr(runtime.module.geometry, "last_diagnostics", {})))
            injection_trace.append(_scalar_diagnostics(getattr(runtime.dit, "last_injection_stats", {})))
        return velocity

    sampled = sampler(
        noise,
        prepared,
        context,
        predict_velocity=predict_velocity,
        config=FlowSamplingConfig(sampling_steps, sampling_shift),
    )
    canonical_noise = noise
    if inverse_permutation is not None:
        canonical_noise = permute_view_tensor(noise, inverse_permutation)
    if not return_diagnostics:
        return sampled
    return sampled, {
        "prepared_features": prepared.detach().float().cpu(),
        "fusion_diagnostics": _scalar_diagnostics(
            getattr(fusion_module, "last_diagnostics", {})
        ),
        "velocity_trace": velocity_trace,
        "geometry_trace": geometry_trace,
        "injection_trace": injection_trace,
        "initial_noise": canonical_noise.detach().float().cpu(),
        "fusion_camera_sha256": _camera_digest(fusion_camera),
        "geometry_camera_sha256": _camera_digest(geometry_camera),
    }


def _camera(observations, resolution: int, device: torch.device) -> CameraBatch:
    intrinsics, transforms = [], []
    for observation in observations:
        k = torch.from_numpy(np.asarray(observation.K, dtype=np.float32)).clone()
        k[0] *= resolution / observation.width
        k[1] *= resolution / observation.height
        intrinsics.append(k)
        transforms.append(
            torch.from_numpy(np.asarray(observation.T_world_from_camera, dtype=np.float32)).clone()
        )
    return CameraBatch(
        torch.stack(intrinsics).unsqueeze(0).to(device),
        torch.stack(transforms).unsqueeze(0).to(device),
        (resolution, resolution),
        "multiview",
        0,
    )


def _load_group(
    dataset_root: Path,
    scene: str,
    route: str,
    config: Stage3Config,
    generator: torch.Generator,
    device: torch.device,
):
    adapter, sequence = _scene_data(
        dataset_root / scene, config, Split.TRAIN if route == "train" else Split.TEST
    )
    all_camera = _camera(sequence.observations, config.image_size, torch.device("cpu"))
    anchor = int(torch.randint(len(sequence.observations), (), generator=generator))
    indices = sample_view_indices(
        all_camera,
        anchor,
        generator,
        view_count=config.views,
        nearest=min(config.nearest_views, len(sequence.observations) - 1),
    )
    return _load_indices(dataset_root, scene, route, indices, config, device)


def _load_indices(
    dataset_root: Path,
    scene: str,
    route: str,
    indices: list[int],
    config: Stage3Config,
    device: torch.device,
):
    if len(indices) != config.views or len(set(indices)) != config.views:
        raise ValueError("view indices must contain exactly the configured unique views")
    adapter, sequence = _scene_data(
        dataset_root / scene, config, Split.TRAIN if route == "train" else Split.TEST
    )
    if min(indices) < 0 or max(indices) >= len(sequence.observations):
        raise ValueError("view index is outside the selected split")
    observations = tuple(sequence.observations[index] for index in indices)
    hr = rgb_images_to_video(
        [adapter.load_rgb(item) for item in observations], config.image_size
    ).to(device)
    camera = _camera(observations, config.image_size, device)
    lr = Stage1Degradation(scale=config.scale)(hr)
    return indices, hr, lr, camera


def _prepare_seen_manifest(args, config: Stage3Config) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    groups = []
    subsets = {"full": [], "probe": [], "intervention": []}
    datasets = {}
    for scene in config.train_scenes:
        scene_root = args.dataset_root / scene
        adapter, sequence = _scene_data(scene_root, config, Split.TRAIN)
        camera = _camera(sequence.observations, config.image_size, torch.device("cpu"))
        count = len(sequence.observations)
        probe = set(evenly_spaced_indices(count, 4))
        intervention = set(evenly_spaced_indices(count, 16))
        for anchor in range(count):
            group_id = f"{scene}:{anchor:03d}"
            groups.append({
                "id": group_id,
                "scene": scene,
                "anchor": anchor,
                "indices": nearest_view_indices(camera, anchor, config.views),
            })
            subsets["full"].append(group_id)
            if anchor in probe:
                subsets["probe"].append(group_id)
            if anchor in intervention:
                subsets["intervention"].append(group_id)
        if config.dataset_kind == "mipnerf360":
            image_root = scene_root / ("images" if config.image_factor == 1 else f"images_{config.image_factor}")
            datasets[scene] = {
                "views": count,
                "cameras_bin_sha256": _sha256(scene_root / "sparse/0/cameras.bin"),
                "images_bin_sha256": _sha256(scene_root / "sparse/0/images.bin"),
                "train_images_sha256": _sha256_tree(image_root),
            }
        else:
            datasets[scene] = {
                "views": count,
                "transforms_train_sha256": _sha256(scene_root / "transforms_train.json"),
                "train_images_sha256": _sha256_tree(scene_root / "train"),
            }
    payload = {
        "version": 1,
        "scope": "seen_train_sr",
        "sampling_signature": _sampling_signature(config),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_kind": config.dataset_kind,
        "image_factor": config.image_factor,
        "datasets": datasets,
        "groups": groups,
        "subsets": subsets,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def _load_seen_groups(path: Path, config: Stage3Config, subset: str,
                      group_ids: list[str] | None = None) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("scope") != "seen_train_sr":
        raise ValueError("unsupported seen-view manifest")
    if payload.get("sampling_signature") != _sampling_signature(config):
        raise ValueError("seen-view manifest does not match the Stage 3 sampling config")
    available = {group["id"]: group for group in payload.get("groups", [])}
    selected = list(payload.get("subsets", {}).get(subset, []))
    if group_ids:
        requested = set(group_ids)
        selected = [group_id for group_id in selected if group_id in requested]
        if set(selected) != requested:
            raise ValueError("requested group id is not present in the selected subset")
    if not selected or len(selected) != len(set(selected)) or any(item not in available for item in selected):
        raise ValueError("seen-view subset is empty, duplicated or incomplete")
    return payload, [available[item] for item in selected]


def _load_camera_donors(
    path: Path,
    seen_manifest: Path,
    config: Stage3Config,
    groups: list[dict],
) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("scope") != "stage3_2_far_camera_donors":
        raise ValueError("unsupported camera donor manifest")
    if payload.get("seen_manifest_sha256") != _sha256(seen_manifest):
        raise ValueError("camera donor manifest does not match seen manifest")
    if payload.get("sampling_signature") != _sampling_signature(config):
        raise ValueError("camera donor manifest does not match sampling config")
    items = payload.get("groups", [])
    available = {item.get("id"): item for item in items}
    if len(available) != len(items):
        raise ValueError("camera donor manifest has duplicate group ids")
    selected = {}
    for group in groups:
        item = available.get(group["id"])
        if item is None:
            raise ValueError(f"camera donor missing group {group['id']}")
        if item.get("scene") != group["scene"] or item.get("anchor") != group["anchor"]:
            raise ValueError(f"camera donor identity mismatch for {group['id']}")
        if item.get("source_indices") != group["indices"]:
            raise ValueError(f"camera donor source indices mismatch for {group['id']}")
        donor_indices = item.get("fusion_camera_indices")
        angles = item.get("donor_angles_deg")
        if (
            not isinstance(donor_indices, list)
            or len(donor_indices) != config.views
            or donor_indices[0] != group["anchor"]
            or len(set(donor_indices)) != len(donor_indices)
            or set(donor_indices[1:]) & set(group["indices"])
        ):
            raise ValueError(f"invalid camera donor indices for {group['id']}")
        if (
            not isinstance(angles, list)
            or len(angles) != config.views - 1
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
                for value in angles
            )
        ):
            raise ValueError(f"invalid camera donor angles for {group['id']}")
        selected[group["id"]] = item
    return selected


def _camera_for_indices(
    dataset_root: Path,
    scene: str,
    indices: list[int],
    config: Stage3Config,
    device: torch.device,
) -> CameraBatch:
    _, sequence = _scene_data(dataset_root / scene, config, Split.TRAIN)
    if any(type(index) is not int or not 0 <= index < len(sequence.observations) for index in indices):
        raise ValueError("camera donor index is out of range")
    observations = [sequence.observations[index] for index in indices]
    return _camera(observations, config.image_size, device)


def camera_pair_ranking_loss(prediction_correct, prediction_wrong, target, *, margin_ratio: float):
    """Return per-sample correct/wrong MSE and the preregistered ranking hinge."""
    if prediction_correct.shape != target.shape or prediction_wrong.shape != target.shape:
        raise ValueError("camera ranking predictions and target must have matching shapes")
    if isinstance(margin_ratio, bool) or not isinstance(margin_ratio, (int, float)) \
            or not np.isfinite(margin_ratio) or margin_ratio <= 0:
        raise ValueError("margin_ratio must be finite and positive")
    dimensions = tuple(range(1, target.ndim))
    e_correct = (prediction_correct - target).square().mean(dim=dimensions)
    e_wrong = (prediction_wrong - target).square().mean(dim=dimensions)
    rank = F.relu(float(margin_ratio) * e_correct.detach() + e_correct - e_wrong)
    return e_correct, e_wrong, rank


def per_view_flow_losses(prediction, target):
    """Return MSE for each [batch, view] without mixing target and auxiliaries."""
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError("prediction and target must have matching [B,C,V,...] shapes")
    dimensions = tuple(index for index in range(1, target.ndim) if index != 2)
    return (prediction - target).square().mean(dim=dimensions)


def decoded_target_ssim_loss(vae, noisy, prediction, sigma, hr):
    """Differentiable SSIM on the correctly paired target view only."""
    from torchmetrics.functional.image import structural_similarity_index_measure

    if noisy.shape != prediction.shape or noisy.shape[2] != hr.shape[2]:
        raise ValueError("latent and HR view dimensions must match")
    x0_hat = noisy[:, :, :1] - sigma.reshape(-1, 1, 1, 1, 1) * prediction[:, :, :1]
    decoded = vae.decode_multiview(x0_hat)[:, :, 0]
    reference = hr[:, :, 0]
    return 1 - structural_similarity_index_measure(
        (decoded.float() + 1) * 0.5, (reference.float() + 1) * 0.5,
        data_range=1.0,
    )


def _camera_pair_training_losses(
    prediction_correct,
    prediction_wrong,
    target,
    *,
    margin_ratio: float,
    target_view_only: bool,
    target_view_flow_fraction: float | None = None,
):
    """Keep the SR flow loss global while optionally ranking only view zero."""
    if target_view_flow_fraction is None:
        flow = flow_matching_loss(prediction_correct, target)
    else:
        per_view = per_view_flow_losses(prediction_correct, target)
        flow = (target_view_flow_fraction * per_view[:, 0]
                + (1 - target_view_flow_fraction) * per_view[:, 1:].mean(dim=1)).mean()
    if target_view_only:
        prediction_correct = prediction_correct[:, :, 0:1]
        prediction_wrong = prediction_wrong[:, :, 0:1]
        target = target[:, :, 0:1]
    e_correct, e_wrong, rank = camera_pair_ranking_loss(
        prediction_correct,
        prediction_wrong,
        target,
        margin_ratio=margin_ratio,
    )
    return flow, e_correct, e_wrong, rank


def _symmetric_correspondence_rank(camera_rank, lr_rank, camera_fraction=0.5):
    if camera_rank.shape != lr_rank.shape:
        raise ValueError("camera and LR ranking losses must have matching shapes")
    if not 0 < camera_fraction < 1:
        raise ValueError("camera_fraction must be in (0, 1)")
    return camera_fraction * camera_rank + (1 - camera_fraction) * lr_rank


def _camera_rank_gradient_groups(module, parameters, gradients):
    """Split A6 ranking gradients across the existing fusion/bridge path."""
    if len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must have equal lengths")
    if module.fusion is None:
        raise ValueError("camera ranking gradient telemetry requires LR fusion")
    by_parameter = {id(parameter): gradient for parameter, gradient in zip(parameters, gradients)}
    qkv = by_parameter.get(id(module.fusion.qkv.weight))
    hidden = module.fusion.hidden_dim
    qk = None if qkv is None else qkv[: 2 * hidden]
    value = None if qkv is None else qkv[2 * hidden :]

    def gradients_for(values):
        return tuple(by_parameter.get(id(parameter)) for parameter in values)

    return {
        "camera_rank_qk_gradient_norm": _tensor_gradient_norm((qk,)),
        "camera_rank_value_gradient_norm": _tensor_gradient_norm((value,)),
        "camera_rank_output_gradient_norm": _tensor_gradient_norm(
            gradients_for(module.fusion.output.parameters())
        ),
        "camera_rank_bridge_gradient_norm": _tensor_gradient_norm(
            gradients_for(module.conditioner.bridge.parameters())
        ),
    }


def _deranged_auxiliary_indices(views: int, generator: torch.Generator) -> torch.Tensor:
    if views < 3:
        raise ValueError("camera pairing requires at least two auxiliary views")
    auxiliary = list(range(1, views))
    shift = int(torch.randint(1, len(auxiliary), (), generator=generator))
    return torch.tensor([0, *(auxiliary[shift:] + auxiliary[:shift])], dtype=torch.long)


def _camera_index_select(camera: CameraBatch, indices: torch.Tensor) -> CameraBatch:
    camera.validate()
    indices = _validate_view_permutation(indices, camera.K.shape[1])
    indices = indices.to(device=camera.K.device)
    reference = int((indices == camera.reference_index).nonzero(as_tuple=False).item())
    xi = None if camera.xi is None else camera.xi.index_select(1, indices)
    return CameraBatch(
        camera.K.index_select(1, indices),
        camera.T_world_from_camera.index_select(1, indices),
        camera.image_size,
        camera.sequence_kind,
        reference_index=reference,
        camera_model=camera.camera_model,
        xi=xi,
    )


def _camera_with_auxiliary_permutation(
    camera: CameraBatch, indices: torch.Tensor
) -> CameraBatch:
    """Reassign auxiliary camera data while keeping its target slot fixed."""
    if int(indices[0]) != 0:
        raise ValueError("auxiliary permutation must preserve target slot zero")
    selected = _camera_index_select(camera, indices)
    return CameraBatch(
        selected.K,
        selected.T_world_from_camera,
        selected.image_size,
        selected.sequence_kind,
        reference_index=camera.reference_index,
        camera_model=selected.camera_model,
        xi=selected.xi,
    )


def derange_auxiliary_fusion_camera(camera: CameraBatch, generator: torch.Generator) -> CameraBatch:
    """Cyclically derange auxiliary camera slots while preserving target view zero."""
    camera.validate()
    if camera.sequence_kind != "multiview" or camera.K.shape[1] < 3:
        raise ValueError("camera ranking requires at least two auxiliary multiview cameras")
    return _camera_with_auxiliary_permutation(
        camera, _deranged_auxiliary_indices(camera.K.shape[1], generator)
    )


def _validate_view_permutation(permutation: torch.Tensor, views: int | None = None) -> torch.Tensor:
    permutation = torch.as_tensor(permutation, dtype=torch.long)
    if permutation.ndim != 1 or permutation.numel() < 1:
        raise ValueError("view permutation must be a non-empty vector")
    views = permutation.numel() if views is None else views
    if permutation.numel() != views or not torch.equal(
        torch.sort(permutation.cpu()).values, torch.arange(views)
    ):
        raise ValueError("invalid view permutation")
    return permutation


def inverse_view_permutation(permutation: torch.Tensor) -> torch.Tensor:
    permutation = _validate_view_permutation(permutation)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return inverse


def permute_view_tensor(value: torch.Tensor, permutation: torch.Tensor, *, view_dim: int = 2):
    permutation = _validate_view_permutation(permutation, value.shape[view_dim])
    return value.index_select(view_dim, permutation.to(value.device))


def _permutation_digest(permutation: torch.Tensor) -> str:
    permutation = _validate_view_permutation(permutation)
    return hashlib.sha256(permutation.cpu().numpy().tobytes()).hexdigest()


def apply_joint_view_permutation(
    permutation: torch.Tensor,
    *,
    lr: torch.Tensor,
    fusion_camera: CameraBatch,
    geometry_camera: CameraBatch,
    target: torch.Tensor | None = None,
    latent: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> dict:
    """Apply one view relabeling to every view-indexed model input."""
    permutation = _validate_view_permutation(permutation, lr.shape[2])
    result = {
        "lr": permute_view_tensor(lr, permutation),
        "fusion_camera": _camera_index_select(fusion_camera, permutation),
        "geometry_camera": _camera_index_select(geometry_camera, permutation),
        "target_index": int((permutation == 0).nonzero(as_tuple=False).item()),
        "permutation": permutation.clone(),
        "inverse_permutation": inverse_view_permutation(permutation),
        "permutation_sha256": _permutation_digest(permutation),
    }
    for name, value in (("target", target), ("latent", latent), ("noise", noise)):
        result[name] = None if value is None else permute_view_tensor(value, permutation)
    result["source_mask"] = (
        None if source_mask is None
        else source_mask.index_select(1, permutation.to(source_mask.device))
    )
    return result


def _rotation_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """Convert valid rotation matrices to normalized ``(w,x,y,z)`` quaternions."""
    m00, m11, m22 = rotation[..., 0, 0], rotation[..., 1, 1], rotation[..., 2, 2]
    magnitudes = torch.sqrt(torch.stack((
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ), dim=-1).clamp_min(0))
    candidates = torch.stack((
        torch.stack((
            magnitudes[..., 0].square(),
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ), dim=-1),
        torch.stack((
            rotation[..., 2, 1] - rotation[..., 1, 2],
            magnitudes[..., 1].square(),
            rotation[..., 1, 0] + rotation[..., 0, 1],
            rotation[..., 0, 2] + rotation[..., 2, 0],
        ), dim=-1),
        torch.stack((
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] + rotation[..., 0, 1],
            magnitudes[..., 2].square(),
            rotation[..., 2, 1] + rotation[..., 1, 2],
        ), dim=-1),
        torch.stack((
            rotation[..., 1, 0] - rotation[..., 0, 1],
            rotation[..., 0, 2] + rotation[..., 2, 0],
            rotation[..., 2, 1] + rotation[..., 1, 2],
            magnitudes[..., 3].square(),
        ), dim=-1),
    ), dim=-2)
    candidates = candidates / (2 * magnitudes[..., :, None].clamp_min(1e-7))
    index = magnitudes.argmax(dim=-1)
    selector = index[..., None, None].expand(*index.shape, 1, 4)
    quaternion = candidates.gather(-2, selector).squeeze(-2)
    return F.normalize(quaternion, dim=-1)


def _quaternion_to_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def _slerp_rotations(start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
    q0, q1 = _rotation_to_quaternion(start), _rotation_to_quaternion(end)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1)
    theta = torch.acos(dot)
    sine = torch.sin(theta)
    scalar = torch.as_tensor(alpha, dtype=start.dtype, device=start.device)
    slerp = (
        torch.sin((1 - scalar) * theta) / sine.clamp_min(1e-7) * q0
        + torch.sin(scalar * theta) / sine.clamp_min(1e-7) * q1
    )
    linear = F.normalize((1 - scalar) * q0 + scalar * q1, dim=-1)
    return _quaternion_to_rotation(torch.where(sine.abs() < 1e-6, linear, slerp))


def interpolate_fusion_camera_dose(
    correct: CameraBatch, wrong: CameraBatch, alpha: float
) -> CameraBatch:
    """Geodesically interpolate fusion cameras while preserving target slot zero."""
    correct.validate()
    wrong.validate()
    if not 0 <= alpha <= 1 or correct.camera_model != wrong.camera_model:
        raise ValueError("camera dose requires matching models and alpha in [0,1]")
    if correct.K.shape != wrong.K.shape or correct.image_size != wrong.image_size:
        raise ValueError("camera dose endpoints must have matching shapes and image size")
    if (
        correct.sequence_kind != wrong.sequence_kind
        or correct.reference_index != wrong.reference_index
        or correct.K.device != wrong.K.device
        or correct.K.dtype != wrong.K.dtype
    ):
        raise ValueError("camera dose endpoints must share metadata, device and dtype")
    target = correct.reference_index
    if not torch.equal(correct.K[:, target], wrong.K[:, target]) or not torch.equal(
        correct.T_world_from_camera[:, target], wrong.T_world_from_camera[:, target]
    ):
        raise ValueError("camera dose endpoints must preserve the target view")
    if correct.xi is not None and not torch.equal(correct.xi[:, target], wrong.xi[:, target]):
        raise ValueError("camera dose endpoints must preserve target xi")
    if alpha == 0:
        return correct
    if alpha == 1:
        return wrong
    transform = torch.zeros_like(correct.T_world_from_camera)
    transform[..., :3, :3] = _slerp_rotations(
        correct.T_world_from_camera[..., :3, :3],
        wrong.T_world_from_camera[..., :3, :3],
        alpha,
    )
    transform[..., :3, 3] = torch.lerp(
        correct.T_world_from_camera[..., :3, 3], wrong.T_world_from_camera[..., :3, 3], alpha
    )
    transform[..., 3, 3] = 1
    intrinsics = torch.lerp(correct.K, wrong.K, alpha)
    xi = None if correct.xi is None else torch.lerp(correct.xi, wrong.xi, alpha)
    intrinsics[:, target] = correct.K[:, target]
    transform[:, target] = correct.T_world_from_camera[:, target]
    if xi is not None:
        xi[:, target] = correct.xi[:, target]
    return CameraBatch(
        intrinsics,
        transform,
        correct.image_size,
        correct.sequence_kind,
        reference_index=correct.reference_index,
        camera_model=correct.camera_model,
        xi=xi,
    )


def _tensor_gradient_norm(gradients) -> float:
    values = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def _accumulate_gradient_vectors(accumulated, gradients):
    """Componentwise-add detached parameter gradients, preserving unused slots."""
    if accumulated is None:
        accumulated = [None] * len(gradients)
    if len(accumulated) != len(gradients):
        raise ValueError("gradient accumulator shape mismatch")
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        value = gradient.detach().float()
        accumulated[index] = (
            value.clone() if accumulated[index] is None
            else accumulated[index] + value
        )
    return accumulated


def _clone_preflight_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_preflight_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_preflight_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_preflight_value(item) for item in value)
    return value


def _pairing_preflight_objects(runtime: Runtime) -> list[object]:
    """Collect conditioning, VAE, and DiT wrappers/modules without broad graph walking."""
    queue = [runtime.module, runtime.vae, runtime.dit]
    result = []
    seen = set()
    while queue:
        value = queue.pop(0)
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        result.append(value)
        child_model = getattr(value, "model", None)
        if child_model is not None and child_model is not value:
            queue.append(child_model)
        if isinstance(value, torch.nn.Module):
            queue.extend(value.children())
    return result


@contextmanager
def _preserve_pairing_preflight_state(objects):
    """Restore RNG plus mutable conditioning/VAE/DiT state after A5 preflight."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    numpy_state = (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:])
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modules = [value for value in objects if isinstance(value, torch.nn.Module)]
    modes = [(module, module.training) for module in modules]
    buffers = [
        (module, name, value.detach().clone())
        for module in modules
        for name, value in module.named_buffers(recurse=False)
    ]
    parameter_versions = [
        (parameter, parameter._version)
        for module in modules
        for parameter in module.parameters(recurse=False)
    ]
    mutable_attributes = [
        (value, name, _clone_preflight_value(getattr(value, name)))
        for value in objects
        for name in ("cache", "last_diagnostics", "last_injection_stats")
        if hasattr(value, name)
    ]
    try:
        yield
        if any(parameter._version != version for parameter, version in parameter_versions):
            raise RuntimeError("A5 preflight mutated a model parameter")
    finally:
        restoration_error = None
        for module, name, value in buffers:
            current_buffers = dict(module.named_buffers(recurse=False))
            if name not in current_buffers or current_buffers[name].shape != value.shape:
                restoration_error = RuntimeError(
                    f"A5 preflight changed module buffer structure: {name}"
                )
            else:
                current_buffers[name].copy_(value)
        for value, name, state in mutable_attributes:
            setattr(value, name, _clone_preflight_value(state))
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if restoration_error is not None:
            raise restoration_error


def calibrate_pairing_weight(
    batch_factory,
    fusion_parameters,
    *,
    batch_count: int,
    target_gradient_ratio: float,
    weight_clip: tuple[float, float],
) -> dict:
    """Calibrate one fixed A5 weight from deterministic, side-effect-free losses."""
    parameters = tuple(fusion_parameters)
    rows = []
    flow_accumulated = pairing_accumulated = None
    for index in range(batch_count):
        batch = batch_factory(index)
        flow_gradients = torch.autograd.grad(
            batch["flow_loss"], parameters, retain_graph=True, allow_unused=True
        )
        pairing_gradients = torch.autograd.grad(
            batch["pairing_loss"], parameters, allow_unused=True
        )
        flow_norm = _tensor_gradient_norm(flow_gradients)
        pairing_norm = _tensor_gradient_norm(pairing_gradients)
        if not math.isfinite(flow_norm) or flow_norm <= 0:
            raise ValueError("A5 preflight main flow gradient is zero or nonfinite")
        if not math.isfinite(pairing_norm) or pairing_norm <= 0:
            raise ValueError("A5 preflight pairing gradient is zero or nonfinite")
        flow_accumulated = _accumulate_gradient_vectors(
            flow_accumulated, tuple(
                None if gradient is None else gradient / batch_count
                for gradient in flow_gradients
            )
        )
        pairing_accumulated = _accumulate_gradient_vectors(
            pairing_accumulated, tuple(
                None if gradient is None else gradient / batch_count
                for gradient in pairing_gradients
            )
        )
        metadata = dict(batch["metadata"])
        rows.append({**metadata, "flow_gradient_norm": flow_norm,
                     "pairing_gradient_norm": pairing_norm})
    flow_norm = _tensor_gradient_norm(flow_accumulated)
    pairing_norm = _tensor_gradient_norm(pairing_accumulated)
    if not math.isfinite(flow_norm) or flow_norm <= 0:
        raise ValueError("A5 reduced main flow gradient is zero or nonfinite")
    if not math.isfinite(pairing_norm) or pairing_norm <= 0:
        raise ValueError("A5 reduced pairing gradient is zero or nonfinite")
    raw = float(target_gradient_ratio * flow_norm / pairing_norm)
    weight = float(np.clip(raw, weight_clip[0], weight_clip[1]))
    return {
        "batches": rows,
        "gradient_reduction": "mean_gradient_over_8_batches_then_global_l2",
        "flow_gradient_norm": flow_norm,
        "pairing_gradient_norm": pairing_norm,
        "flow_batch_gradient_norm_mean": float(np.mean([
            row["flow_gradient_norm"] for row in rows
        ])),
        "pairing_batch_gradient_norm_mean": float(np.mean([
            row["pairing_gradient_norm"] for row in rows
        ])),
        "raw_weight": raw,
        "weight": weight,
    }


def resolve_pairing_weight(enabled, initialization_mode, manifest, checkpoint, calibrate):
    """Resume the frozen A5 scalar; every fresh/model-only run recalibrates."""
    if not enabled:
        return 0.0
    if initialization_mode == "resume":
        manifest_weight = (manifest.get("pairing") or {}).get("weight")
        state_weight = ((checkpoint or {}).get("training_state") or {}).get("pairing_weight")
        if (
            not isinstance(manifest_weight, (int, float))
            or not math.isfinite(manifest_weight)
            or manifest_weight != state_weight
        ):
            raise ValueError("resume pairing weight is missing or inconsistent")
        return float(manifest_weight)
    result = calibrate()
    weight = result.get("weight")
    if not isinstance(weight, (int, float)) or not math.isfinite(weight):
        raise ValueError("A5 calibration returned an invalid weight")
    return float(weight)


def validate_pairing_resume(manifest: dict, checkpoint: dict, config: Stage3Config) -> float:
    """Validate immutable A5 protocol/calibration provenance before state restore."""
    section = manifest.get("pairing")
    if not isinstance(section, dict) or section.get("enabled") is not True:
        raise ValueError("A5 resume pairing manifest is missing or disabled")
    if section.get("protocol") != config.pairing_protocol:
        raise ValueError("A5 resume pairing protocol mismatch")
    weight = section.get("weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or not config.pairing_weight_min <= weight <= config.pairing_weight_max
    ):
        raise ValueError("A5 resume pairing weight is invalid")
    path_value = section.get("preflight")
    path = Path(path_value) if isinstance(path_value, str) else None
    if path is None or not path.is_file():
        raise ValueError("A5 resume pairing preflight artifact is missing")
    if section.get("preflight_sha256") != _sha256(path):
        raise ValueError("A5 resume pairing preflight hash mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("A5 resume pairing preflight artifact is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("A5 resume pairing preflight artifact is invalid")
    if set(payload) != {"protocol", "calibration"}:
        raise ValueError("A5 resume pairing preflight schema mismatch")
    if payload.get("protocol") != config.pairing_protocol:
        raise ValueError("A5 resume preflight protocol mismatch")
    calibration = payload.get("calibration")
    calibration_keys = {
        "batches", "gradient_reduction", "flow_gradient_norm",
        "pairing_gradient_norm", "flow_batch_gradient_norm_mean",
        "pairing_batch_gradient_norm_mean", "raw_weight", "weight",
    }
    if not isinstance(calibration, dict) or set(calibration) != calibration_keys:
        raise ValueError("A5 resume pairing preflight calibration schema mismatch")
    reduction = "mean_gradient_over_8_batches_then_global_l2"
    if calibration["gradient_reduction"] != reduction:
        raise ValueError("A5 resume pairing gradient reduction mismatch")
    batches = calibration["batches"]
    if not isinstance(batches, list) or len(batches) != config.pairing_calibration_batches:
        raise ValueError("A5 resume pairing preflight requires exactly 8 batch records")
    training_seed = (manifest.get("provenance") or {}).get("training_seed")
    if type(training_seed) is not int or training_seed < 0:
        raise ValueError("A5 resume pairing training seed is invalid")

    def finite_positive(value) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0
        )

    def sha256_string(value) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    batch_keys = {
        "seed", "scene", "view_indices", "sigma", "target_lr_dropped",
        "view_count", "target_patch_count", "valid_pair_identities",
        "pair_count", "coverage_min", "coverage", "camera_sha256",
        "wrong_camera_sha256", "batch_sha256", "flow_gradient_norm",
        "pairing_gradient_norm",
    }
    identity_keys = {"batch_index", "target_view", "source_view"}
    coverage_keys = {
        "batch_index", "target_view", "source_view", "coverage", "pair_count",
        "target_patch_count",
    }
    for index, batch in enumerate(batches):
        if not isinstance(batch, dict) or set(batch) != batch_keys:
            raise ValueError("A5 resume pairing preflight batch schema mismatch")
        if batch["seed"] != training_seed + 6000 + index:
            raise ValueError("A5 resume pairing preflight seed mismatch")
        if batch["scene"] not in config.train_scenes:
            raise ValueError("A5 resume pairing preflight scene mismatch")
        view_indices = batch["view_indices"]
        if (
            not isinstance(view_indices, list)
            or len(view_indices) != config.views
            or any(type(value) is not int or value < 0 for value in view_indices)
            or len(set(view_indices)) != len(view_indices)
        ):
            raise ValueError("A5 resume pairing preflight view indices are invalid")
        sigma = batch["sigma"]
        if (
            isinstance(sigma, bool)
            or not isinstance(sigma, (int, float))
            or not math.isfinite(sigma)
            or not 0 <= sigma <= 1
        ):
            raise ValueError("A5 resume pairing preflight sigma is invalid")
        if type(batch["target_lr_dropped"]) is not bool:
            raise ValueError("A5 resume pairing preflight dropout flag is invalid")
        if type(batch["view_count"]) is not int or batch["view_count"] != config.views:
            raise ValueError("A5 resume pairing preflight view count is invalid")
        target_patch_count = batch["target_patch_count"]
        if type(target_patch_count) is not int or target_patch_count < 1:
            raise ValueError("A5 resume pairing target patch count is invalid")
        if not finite_positive(batch["flow_gradient_norm"]) or not finite_positive(
            batch["pairing_gradient_norm"]
        ):
            raise ValueError("A5 resume pairing per-batch gradient norm is invalid")
        if not all(sha256_string(batch[name]) for name in (
            "camera_sha256", "wrong_camera_sha256", "batch_sha256"
        )):
            raise ValueError("A5 resume pairing preflight hash is invalid")
        coverage_rows = batch["coverage"]
        if not isinstance(coverage_rows, list) or not coverage_rows:
            raise ValueError("A5 resume pairing coverage records are missing")
        identities = batch["valid_pair_identities"]
        if not isinstance(identities, list) or not identities:
            raise ValueError("A5 resume pairing valid identities are missing")
        recorded_expected_pairs = set()
        for identity_row in identities:
            if not isinstance(identity_row, dict) or set(identity_row) != identity_keys:
                raise ValueError("A5 resume pairing identity schema mismatch")
            identity = tuple(identity_row[name] for name in (
                "batch_index", "target_view", "source_view"
            ))
            if (
                any(type(value) is not int for value in identity)
                or identity[0] != 0
                or not 0 <= identity[1] < config.views
                or not 0 <= identity[2] < config.views
                or identity[1] == identity[2]
                or identity in recorded_expected_pairs
            ):
                raise ValueError("A5 resume pairing valid identity is invalid")
            recorded_expected_pairs.add(identity)
        expected_pairs = {
            (0, target_view, source_view)
            for target_view in range(config.views)
            for source_view in range(config.views)
            if target_view != source_view
        }
        if recorded_expected_pairs != expected_pairs:
            raise ValueError("A5 resume pairing valid identities are incomplete")
        seen_pairs = set()
        for row in coverage_rows:
            if not isinstance(row, dict) or set(row) != coverage_keys:
                raise ValueError("A5 resume pairing coverage schema mismatch")
            coverage = row["coverage"]
            if (
                isinstance(coverage, bool)
                or not isinstance(coverage, (int, float))
                or not math.isfinite(coverage)
                or not config.pairing_minimum_coverage <= coverage <= 1
            ):
                raise ValueError("A5 resume pairing per-pair coverage is invalid")
            pair_count = row["pair_count"]
            if type(pair_count) is not int or pair_count < 1:
                raise ValueError("A5 resume pairing pair count is invalid")
            if (
                type(row["target_patch_count"]) is not int
                or row["target_patch_count"] != target_patch_count
            ):
                raise ValueError("A5 resume pairing target patch count mismatch")
            if coverage != pair_count / target_patch_count:
                raise ValueError("A5 resume pairing coverage and pair count mismatch")
            identity = (row["batch_index"], row["target_view"], row["source_view"])
            if any(type(value) is not int for value in identity):
                raise ValueError("A5 resume pairing coverage identity is invalid")
            if identity[0] != 0:
                raise ValueError("A5 resume pairing coverage batch_index is invalid")
            if (
                not 0 <= identity[1] < config.views
                or not 0 <= identity[2] < config.views
                or identity[1] == identity[2]
                or identity in seen_pairs
            ):
                raise ValueError("A5 resume pairing coverage identity is invalid")
            seen_pairs.add(identity)
        if seen_pairs != expected_pairs:
            raise ValueError("A5 resume pairing coverage identities are incomplete")
        coverage_minimum = min(row["coverage"] for row in coverage_rows)
        stored_coverage_minimum = batch["coverage_min"]
        if (
            isinstance(stored_coverage_minimum, bool)
            or not isinstance(stored_coverage_minimum, (int, float))
            or not math.isfinite(stored_coverage_minimum)
            or stored_coverage_minimum < config.pairing_minimum_coverage
            or stored_coverage_minimum != coverage_minimum
        ):
            raise ValueError("A5 resume pairing coverage minimum mismatch")
        pair_count = batch["pair_count"]
        if type(pair_count) is not int or pair_count != sum(
            row["pair_count"] for row in coverage_rows
        ):
            raise ValueError("A5 resume pairing total pair count mismatch")

    norm_names = (
        "flow_gradient_norm", "pairing_gradient_norm",
        "flow_batch_gradient_norm_mean", "pairing_batch_gradient_norm_mean",
    )
    if any(not finite_positive(calibration[name]) for name in norm_names):
        raise ValueError("A5 resume pairing reduced or per-batch gradient norm is invalid")
    expected_flow_mean = float(np.mean([
        batch["flow_gradient_norm"] for batch in batches
    ]))
    expected_pairing_mean = float(np.mean([
        batch["pairing_gradient_norm"] for batch in batches
    ]))
    if not math.isclose(calibration["flow_batch_gradient_norm_mean"], expected_flow_mean):
        raise ValueError("A5 resume flow per-batch gradient norm mean mismatch")
    if not math.isclose(
        calibration["pairing_batch_gradient_norm_mean"], expected_pairing_mean
    ):
        raise ValueError("A5 resume pairing per-batch gradient norm mean mismatch")
    raw_weight = calibration["raw_weight"]
    calibrated_weight = calibration["weight"]
    if (
        isinstance(raw_weight, bool)
        or not isinstance(raw_weight, (int, float))
        or not math.isfinite(raw_weight)
        or isinstance(calibrated_weight, bool)
        or not isinstance(calibrated_weight, (int, float))
        or not math.isfinite(calibrated_weight)
    ):
        raise ValueError("A5 resume raw or clipped weight is invalid")
    expected_raw = (
        config.pairing_target_gradient_ratio
        * calibration["flow_gradient_norm"]
        / calibration["pairing_gradient_norm"]
    )
    if not math.isclose(raw_weight, expected_raw):
        raise ValueError("A5 resume raw pairing weight mismatch")
    expected_clipped = float(np.clip(
        raw_weight, config.pairing_weight_min, config.pairing_weight_max
    ))
    if calibrated_weight != expected_clipped:
        raise ValueError("A5 resume clipped weight mismatch")
    if calibrated_weight != weight:
        raise ValueError("A5 resume preflight weight mismatch")
    state_weight = ((checkpoint.get("training_state") or {}).get("pairing_weight"))
    if state_weight != weight:
        raise ValueError("A5 resume checkpoint pairing weight mismatch")
    return float(weight)


def persist_pairing_preflight(output: Path, config: Stage3Config, calibration: dict) -> dict:
    """Persist one immutable A5 calibration artifact and return its manifest section."""
    batches = calibration.get("batches") if isinstance(calibration, dict) else None
    if (
        not isinstance(batches, list)
        or len(batches) != config.pairing_calibration_batches
    ):
        raise ValueError("A5 calibration coverage requires exactly 8 batches")
    expected_identity_keys = {"batch_index", "target_view", "source_view"}
    for batch in batches:
        if not isinstance(batch, dict):
            raise ValueError("A5 calibration coverage batch is malformed")
        view_count = batch.get("view_count")
        target_patch_count = batch.get("target_patch_count")
        if view_count != config.views:
            raise ValueError("A5 calibration coverage has an invalid view count")
        expected_identities = _validate_calibration_coverage(
            batch.get("coverage"),
            view_count=view_count,
            target_patch_count=target_patch_count,
        )
        identities = batch.get("valid_pair_identities")
        if (
            not isinstance(identities, list)
            or len(identities) != len(expected_identities)
            or any(
                not isinstance(row, dict) or set(row) != expected_identity_keys
                for row in identities
            )
        ):
            raise ValueError("A5 calibration coverage identities are malformed")
        if any(
            any(type(row[name]) is not int for name in expected_identity_keys)
            or row["batch_index"] != 0
            or not 0 <= row["target_view"] < view_count
            or not 0 <= row["source_view"] < view_count
            or row["target_view"] == row["source_view"]
            for row in identities
        ):
            raise ValueError("A5 calibration coverage identities are malformed")
        identity_tuples = [
            (row["batch_index"], row["target_view"], row["source_view"])
            for row in identities
        ]
        expected_tuples = {
            (row["batch_index"], row["target_view"], row["source_view"])
            for row in expected_identities
        }
        if (
            len(set(identity_tuples)) != len(identity_tuples)
            or set(identity_tuples) != expected_tuples
        ):
            raise ValueError("A5 calibration coverage identities are incomplete")
        coverage_rows = batch["coverage"]
        if any(
            row["coverage"] < config.pairing_minimum_coverage
            for row in coverage_rows
        ):
            raise ValueError("A5 calibration coverage is below the required minimum")
        if (
            type(batch.get("pair_count")) is not int
            or batch["pair_count"] != sum(row["pair_count"] for row in coverage_rows)
            or isinstance(batch.get("coverage_min"), bool)
            or not isinstance(batch.get("coverage_min"), (int, float))
            or not math.isfinite(batch["coverage_min"])
            or batch["coverage_min"] != min(row["coverage"] for row in coverage_rows)
        ):
            raise ValueError("A5 calibration coverage aggregate is malformed")
    path = Path(output).resolve() / "pairing_preflight.json"
    payload = {"protocol": config.pairing_protocol, "calibration": calibration}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {
        "enabled": True,
        "weight": float(calibration["weight"]),
        "protocol": config.pairing_protocol,
        "preflight": str(path),
        "preflight_sha256": _sha256(path),
    }


def _validate_calibration_coverage(
    coverage_rows: list[dict], *, view_count: int, target_patch_count: int
) -> list[dict]:
    """Validate one pinhole batch's complete directed non-self pair coverage."""
    if type(view_count) is not int or view_count < 2:
        raise ValueError("A5 calibration coverage has an invalid view count")
    if type(target_patch_count) is not int or target_patch_count <= 0:
        raise ValueError("A5 calibration coverage has an invalid target patch count")
    expected = [
        {"batch_index": 0, "target_view": target, "source_view": source}
        for target in range(view_count)
        for source in range(view_count)
        if target != source
    ]
    if not isinstance(coverage_rows, list) or len(coverage_rows) != len(expected):
        raise ValueError("A5 calibration coverage is not the complete directed pair set")
    identities = []
    for row in coverage_rows:
        if (
            not isinstance(row, dict)
            or set(row) != {
                "batch_index", "target_view", "source_view", "coverage",
                "pair_count", "target_patch_count",
            }
        ):
            raise ValueError("A5 calibration coverage row is malformed")
        identity = (
            row.get("batch_index"), row.get("target_view"), row.get("source_view")
        )
        if (
            any(type(value) is not int for value in identity)
            or identity[0] != 0
            or not 0 <= identity[1] < view_count
            or not 0 <= identity[2] < view_count
            or identity[1] == identity[2]
        ):
            raise ValueError("A5 calibration coverage identity is malformed")
        pair_count = row.get("pair_count")
        row_patch_count = row.get("target_patch_count")
        coverage = row.get("coverage")
        if (
            type(pair_count) is not int
            or pair_count <= 0
            or pair_count > target_patch_count
            or type(row_patch_count) is not int
            or row_patch_count != target_patch_count
            or isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or not math.isfinite(coverage)
            or coverage != pair_count / target_patch_count
        ):
            raise ValueError("A5 calibration coverage value is malformed")
        identities.append(identity)
    expected_set = {
        (row["batch_index"], row["target_view"], row["source_view"])
        for row in expected
    }
    if len(set(identities)) != len(identities) or set(identities) != expected_set:
        raise ValueError("A5 calibration coverage is not the complete directed pair set")
    return expected


def _calibrate_a5_pairing(runtime: Runtime, args, config: Stage3Config) -> dict:
    """Run eight private deterministic A5 batches without touching training RNGs."""
    fusion_parameters = tuple(runtime.module.fusion.parameters())

    def batch_factory(index):
        seed = args.seed + 6000 + index
        view_generator = torch.Generator().manual_seed(seed)
        pairing_generator = torch.Generator().manual_seed(seed + 1000)
        noise_generator = torch.Generator(device=runtime.device).manual_seed(seed + 2000)
        sigma_generator = torch.Generator().manual_seed(seed + 3000)
        dropout_generator = torch.Generator().manual_seed(seed + 4000)
        scene = config.train_scenes[index % len(config.train_scenes)]
        indices, hr, lr, camera = _load_group(
            args.dataset_root, scene, "train", config, view_generator, runtime.device
        )
        pairing_lr = lr
        dropped = (
            config.target_lr_dropout > 0
            and float(torch.rand((), generator=dropout_generator)) < config.target_lr_dropout
        )
        if dropped:
            lr = lr.clone()
            lr[:, :, 0] = 0
        wrong_camera = derange_auxiliary_fusion_camera(camera, pairing_generator)
        with torch.no_grad():
            clean = runtime.vae.encode_multiview(hr)
        prepared, state = runtime.module.prepare_multiview(
            lr,
            camera,
            tuple(clean.shape[2:]),
            (config.image_size, config.image_size),
            pairing_camera=wrong_camera,
            pairing_minimum_coverage=config.pairing_minimum_coverage,
            pairing_lr=pairing_lr if dropped else None,
        )
        sigma = torch.rand(1, generator=sigma_generator).to(runtime.device)
        noise = torch.randn(
            clean.shape,
            generator=noise_generator,
            device=runtime.device,
            dtype=clean.dtype,
        )
        noisy, timestep, target = flow_matching_pair(clean, noise, sigma)
        prediction = runtime.module.predict(
            runtime.dit, noisy, timestep, None, prepared, camera, tuple(clean.shape[2:])
        )
        flow_loss = flow_matching_loss(prediction, target)
        pairing_loss = pairing_info_nce_loss(
            state, temperature=config.pairing_temperature
        )
        coverage_rows = [{
            "batch_index": row["batch_index"],
            "target_view": row["target_view"],
            "source_view": row["source_view"],
            "coverage": row["coverage"],
            "pair_count": int(row["target_patches"].numel()),
            "target_patch_count": int(state["target_patch_count"]),
        } for row in state["pairs"]]
        if int(state["view_count"]) != config.views:
            raise ValueError("A5 pairing state view count mismatch")
        expected_pair_identities = _validate_calibration_coverage(
            coverage_rows,
            view_count=config.views,
            target_patch_count=int(state["target_patch_count"]),
        )
        coverage = [row["coverage"] for row in coverage_rows]
        metadata = {
            "seed": seed,
            "scene": scene,
            "view_indices": indices,
            "sigma": float(sigma),
            "target_lr_dropped": bool(dropped),
            "view_count": int(state["view_count"]),
            "target_patch_count": int(state["target_patch_count"]),
            "valid_pair_identities": expected_pair_identities,
            "pair_count": int(sum(row["target_patches"].numel() for row in state["pairs"])),
            "coverage_min": float(min(coverage)),
            "coverage": coverage_rows,
            "camera_sha256": _camera_digest(camera),
            "wrong_camera_sha256": _camera_digest(wrong_camera),
        }
        digest = hashlib.sha256(_json_sha256(metadata).encode("ascii"))
        for tensor in (hr, pairing_lr, lr, clean, noise):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
        metadata["batch_sha256"] = digest.hexdigest()
        return {"flow_loss": flow_loss, "pairing_loss": pairing_loss, "metadata": metadata}

    with _preserve_pairing_preflight_state(_pairing_preflight_objects(runtime)):
        calibration = calibrate_pairing_weight(
            batch_factory,
            fusion_parameters,
            batch_count=config.pairing_calibration_batches,
            target_gradient_ratio=config.pairing_target_gradient_ratio,
            weight_clip=(config.pairing_weight_min, config.pairing_weight_max),
        )
    return calibration


def a5_step_telemetry(
    *,
    flow_losses,
    rank_losses,
    pairing_losses,
    pair_counts,
    coverage_minima,
    pairing_weight,
    flow_gradient_norm,
    rank_gradient_norm,
    pairing_gradient_norm,
    final_fusion_gradient_norm,
) -> dict:
    """Return explicit, non-aggregated A5 loss and fusion-gradient telemetry."""
    average = lambda values: float(np.mean(values)) if values else 0.0
    pairing_loss = average(pairing_losses)
    return {
        "main_flow_loss": average(flow_losses),
        "output_rank_loss": average(rank_losses),
        "pairing_loss": pairing_loss,
        "weighted_pairing_loss": float(pairing_weight * pairing_loss),
        "pair_count": int(sum(pair_counts)),
        "pairing_coverage_min": float(min(coverage_minima)),
        "pairing_weight": float(pairing_weight),
        "flow_fusion_gradient_norm": float(flow_gradient_norm),
        "rank_fusion_gradient_norm": float(rank_gradient_norm),
        "pairing_fusion_gradient_norm": float(pairing_gradient_norm),
        "final_fusion_gradient_norm": float(final_fusion_gradient_norm),
        "fusion_gradient_reduction": (
            "sum_of_microbatch_mean_loss_gradients_then_global_l2"
        ),
    }


def _training_state(optimizer, sigma_cycle, view_generator, noise_generator, step,
                    dropout_generator=None, pairing_generator=None, pairing_weight=None):
    numpy_state = np.random.get_state()
    state = {
        "format_version": 2,
        "step": step,
        "optimizer": optimizer.state_dict(),
        "sigma_cycle": sigma_cycle.state_dict(),
        "view_generator_state": view_generator.get_state(),
        "noise_generator_state": noise_generator.get_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_rng_state": torch.get_rng_state(),
    }
    if dropout_generator is not None:
        state["dropout_generator_state"] = dropout_generator.get_state()
    if pairing_generator is not None:
        state["pairing_generator_state"] = pairing_generator.get_state()
    if pairing_weight is not None:
        state["pairing_weight"] = float(pairing_weight)
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _validate_runtime_inputs(args, *, checkpoint: Path | None = None) -> None:
    for name in ("dataset_root", "model_dir"):
        path = Path(getattr(args, name))
        if not path.is_dir():
            raise ValueError(f"--{name.replace('_', '-')} must be an existing directory")
    for name in ("lq_source", "lq_checkpoint", "bridge_checkpoint"):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise ValueError(f"--{name.replace('_', '-')} must be an existing file")
    if args.rre_checkpoint is not None and not Path(args.rre_checkpoint).is_file():
        raise ValueError("--rre-checkpoint must be an existing file")
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise ValueError("Stage 3 checkpoint must be an existing file")


def _checkpoint_validation_module(config: Stage3Config) -> Stage3Conditioning:
    """Build a zero-storage meta schema matching the fixed Stage 3 runtime adapters."""
    # Construct directly on ``meta``: creating CPU layers first would consume the
    # global CPU RNG before a malformed resume checkpoint can be rejected.
    with torch.device("meta"):
        conditioner = FrozenLQConditioner(
            torch.nn.Identity(),
            bridge_blocks=(0, 1, 2, 3),
            bridge_time_conditioning=True,
        )
        geometry = FullRREConditioner(branch_count=30)
        fusion = None
        if config.fusion_mode != "off":
            fusion_mode = config.fusion_mode
            if fusion_mode == "epipolar" and config.epipolar_attention == "local_band":
                fusion_mode = "epipolar_local"
            fusion = LRViewFusion(
                hidden_dim=config.fusion_dim,
                heads=config.fusion_heads,
                mode=fusion_mode,
                query_chunk_size=config.query_chunk_size,
                tau=config.epipolar_tau,
                epipolar_band=config.epipolar_band,
                allow_self_view_source=config.allow_self_view_source,
            )
    return Stage3Conditioning(conditioner, geometry, fusion)


def _trial_generator_state(value: torch.Tensor, *, device: str, label: str) -> None:
    """Ask a private generator to parse state without touching a global RNG."""
    try:
        torch.Generator(device=device).set_state(value)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"resume checkpoint contains malformed {label} RNG state") from exc


def _balanced_sigma_cycle_length(config: Stage3Config) -> int:
    """Return the fixed Wan balanced-cycle length without importing the runtime scheduler."""
    training_alphas = np.linspace(1.0, 1.0 / 1000, 1000)[::-1].copy()
    base_sigmas = np.asarray(1.0 - training_alphas, dtype=np.float32)
    sigmas = np.linspace(
        float(base_sigmas[0]),
        float(base_sigmas[-1]),
        config.sampling_steps + 1,
    ).copy()[:-1]
    sigmas = config.sampling_shift * sigmas / (
        1 + (config.sampling_shift - 1) * sigmas
    )
    bucket_sizes = (
        int((sigmas < 0.25).sum()),
        int(((sigmas >= 0.25) & (sigmas < 0.5)).sum()),
        int(((sigmas >= 0.5) & (sigmas < 0.8)).sum()),
        int((sigmas >= 0.8).sum()),
    )
    if any(size == 0 for size in bucket_sizes):
        raise ValueError("inference schedule must populate all balanced sigma bins")
    return 1 + 4 * max(bucket_sizes)


def _validate_resume_optimizer_sigma(
    state: dict,
    module: Stage3Conditioning,
    config: Stage3Config,
) -> None:
    """Trial optimizer metadata and validate the fixed balanced SigmaCycle schema."""
    optimizer_state = state["optimizer"]
    if (
        not isinstance(optimizer_state, dict)
        or set(optimizer_state) != {"state", "param_groups"}
        or not isinstance(optimizer_state["state"], dict)
        or not optimizer_state["state"]
        or not isinstance(optimizer_state["param_groups"], list)
        or len(optimizer_state["param_groups"]) != 1
    ):
        raise ValueError("resume checkpoint contains malformed optimizer state")
    parameters = tuple(
        parameter for parameter in module.parameters() if parameter.requires_grad
    )
    try:
        trial_optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("resume checkpoint contains malformed optimizer state") from exc
    expected_group = trial_optimizer.state_dict()["param_groups"][0]
    group = optimizer_state["param_groups"][0]
    if not isinstance(group, dict) or set(group) != set(expected_group):
        raise ValueError("resume checkpoint contains malformed optimizer state")
    expected_ids = list(range(len(parameters)))
    if (
        group["params"] != expected_ids
        or any(type(value) is not int for value in group["params"])
    ):
        raise ValueError("resume checkpoint contains malformed optimizer state")

    def valid_hyperparameter(actual, expected) -> bool:
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, tuple):
            return (
                len(actual) == len(expected)
                and all(
                    valid_hyperparameter(actual_value, expected_value)
                    for actual_value, expected_value in zip(actual, expected)
                )
            )
        if type(expected) is float:
            return math.isfinite(actual) and actual == expected
        return actual == expected

    if any(
        not valid_hyperparameter(group[key], expected_group[key])
        for key in expected_group
        if key != "params"
    ):
        raise ValueError("resume checkpoint contains malformed optimizer state")
    valid_ids = set(expected_ids)
    if (
        any(type(parameter_id) is not int for parameter_id in optimizer_state["state"])
        or set(optimizer_state["state"]) != valid_ids
    ):
        raise ValueError("resume checkpoint contains malformed optimizer state")
    allowed_state_keys = {"step", "exp_avg", "exp_avg_sq"}
    if group["amsgrad"]:
        allowed_state_keys.add("max_exp_avg_sq")
    expected_optimizer_step = state.get("step")
    if type(expected_optimizer_step) is not int or expected_optimizer_step < 0:
        raise ValueError("resume checkpoint contains malformed optimizer state")
    for parameter_id, parameter_state in optimizer_state["state"].items():
        if (
            not isinstance(parameter_state, dict)
            or set(parameter_state) != allowed_state_keys
        ):
            raise ValueError("resume checkpoint contains malformed optimizer state")
        step = parameter_state["step"]
        if (
            type(step) is not torch.Tensor
            or step.ndim != 0
            or step.device.type != "cpu"
            or step.dtype != torch.float32
            or step.requires_grad
            or not torch.isfinite(step)
            or float(step) < 0
            or float(step) != int(float(step))
            or float(step) != expected_optimizer_step
        ):
            raise ValueError("resume checkpoint contains malformed optimizer state")
        parameter = parameters[parameter_id]
        for key in set(parameter_state) - {"step"}:
            moment = parameter_state[key]
            if (
                type(moment) is not torch.Tensor
                or moment.shape != parameter.shape
                or moment.device.type != "cpu"
                or moment.dtype != parameter.dtype
                or not moment.dtype.is_floating_point
                or moment.requires_grad
                or not torch.isfinite(moment).all()
            ):
                raise ValueError("resume checkpoint contains malformed optimizer state")
    try:
        trial_optimizer.load_state_dict(optimizer_state)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("resume checkpoint contains malformed optimizer state") from exc

    sigma_state = state["sigma_cycle"]
    sigma_keys = {
        "steps", "shift", "strategy", "generator_state", "order", "position", "cycle"
    }
    if not isinstance(sigma_state, dict) or set(sigma_state) != sigma_keys:
        raise ValueError("resume checkpoint contains malformed sigma cycle state")
    expected_length = _balanced_sigma_cycle_length(config)
    order = sigma_state["order"]
    if (
        type(sigma_state["steps"]) is not int
        or sigma_state["steps"] <= 0
        or sigma_state["steps"] != config.sampling_steps
        or type(sigma_state["shift"]) is not type(config.sampling_shift)
        or type(sigma_state["shift"]) not in {int, float}
        or not math.isfinite(sigma_state["shift"])
        or sigma_state["shift"] <= 0
        or sigma_state["shift"] != config.sampling_shift
        or type(sigma_state["strategy"]) is not str
        or sigma_state["strategy"] != "balanced"
        or type(sigma_state["generator_state"]) is not torch.Tensor
        or sigma_state["generator_state"].dtype != torch.uint8
        or sigma_state["generator_state"].ndim != 1
        or sigma_state["generator_state"].numel() == 0
        or sigma_state["generator_state"].device.type != "cpu"
        or sigma_state["generator_state"].requires_grad
        or type(order) is not torch.Tensor
        or order.dtype != torch.int64
        or order.device.type != "cpu"
        or order.requires_grad
        or order.shape != (expected_length,)
        or not torch.equal(torch.sort(order).values, torch.arange(expected_length))
        or type(sigma_state["position"]) is not int
        or not 0 <= sigma_state["position"] <= expected_length
        or type(sigma_state["cycle"]) is not int
        or sigma_state["cycle"] < 0
    ):
        raise ValueError("resume checkpoint contains malformed sigma cycle state")
    consumed = state["step"] * config.gradient_accumulation
    expected_cycle = 0 if consumed == 0 else (consumed - 1) // expected_length
    expected_position = 0 if consumed == 0 else (consumed - 1) % expected_length + 1
    if (
        sigma_state["cycle"] != expected_cycle
        or sigma_state["position"] != expected_position
    ):
        raise ValueError("resume checkpoint contains inconsistent sigma cycle state")
    try:
        # Reuse the real restore method without constructing its scheduler.
        trial_cycle = object.__new__(SigmaCycle)
        trial_cycle.config = FlowSamplingConfig(
            config.sampling_steps, config.sampling_shift
        )
        trial_cycle.strategy = "balanced"
        trial_cycle.values = torch.empty(expected_length)
        trial_cycle.generator = torch.Generator()
        trial_cycle.order = torch.arange(expected_length)
        trial_cycle.position = 0
        trial_cycle.cycle = 0
        trial_cycle.load_state_dict(sigma_state)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("resume checkpoint contains malformed sigma cycle state") from exc


def _validate_training_state_payload(
    state,
    *,
    expected_step: int,
    require_dropout: bool,
    require_pairing: bool,
    require_cuda: bool,
    noise_device: str = "cpu",
    require_pairing_weight: bool = False,
    require_version_two: bool = False,
) -> None:
    """Validate exact-resume fields without mutating optimizer or any RNG."""
    if not isinstance(state, dict):
        raise ValueError("resume checkpoint has no matching training state")
    version = state.get("format_version", 1)
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("unsupported training state format")
    if require_version_two and version != 2:
        raise ValueError("resume checkpoint requires training state format v2")
    required = {
        "step", "optimizer", "sigma_cycle", "view_generator_state",
        "noise_generator_state", "python_rng_state", "numpy_rng_state",
        "torch_rng_state",
    }
    if require_pairing:
        required.add("pairing_generator_state")
    if require_pairing_weight:
        required.add("pairing_weight")
    if version >= 2 and require_dropout:
        required.add("dropout_generator_state")
    if version >= 2 and require_cuda:
        required.add("cuda_rng_state_all")
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"resume checkpoint is missing required RNG/training state: {missing}")
    if type(state.get("step")) is not int or state["step"] < 0 or state["step"] != expected_step:
        raise ValueError("resume checkpoint has no matching training state")
    tensor_fields = ["view_generator_state", "noise_generator_state", "torch_rng_state"]
    if require_pairing:
        tensor_fields.append("pairing_generator_state")
    if version >= 2 and require_dropout:
        tensor_fields.append("dropout_generator_state")
    if any(
        not isinstance(state[name], torch.Tensor)
        or state[name].dtype != torch.uint8
        or state[name].ndim != 1
        or state[name].numel() == 0
        for name in tensor_fields
    ):
        raise ValueError("resume checkpoint contains malformed RNG state")
    cpu_generator_fields = ["view_generator_state", "torch_rng_state"]
    if require_pairing:
        cpu_generator_fields.append("pairing_generator_state")
    if version >= 2 and require_dropout:
        cpu_generator_fields.append("dropout_generator_state")
    for name in cpu_generator_fields:
        _trial_generator_state(state[name], device="cpu", label="torch")
    noise_state = state["noise_generator_state"]
    if torch.device(noise_device).type == "cuda":
        if noise_state.numel() != 16:
            raise ValueError("resume checkpoint contains malformed CUDA noise RNG state")
        if torch.cuda.is_available():
            _trial_generator_state(noise_state, device=noise_device, label="CUDA noise")
    else:
        _trial_generator_state(noise_state, device="cpu", label="torch noise")
    if not isinstance(state["optimizer"], dict) or not isinstance(state["sigma_cycle"], dict):
        raise ValueError("resume checkpoint contains malformed optimizer or sigma state")
    numpy_state = state["numpy_rng_state"]
    if (
        not isinstance(numpy_state, dict)
        or set(numpy_state) != {
            "bit_generator", "state", "position", "has_gauss", "cached_gaussian"
        }
        or not isinstance(numpy_state["bit_generator"], str)
        or not isinstance(numpy_state["state"], list)
        or not numpy_state["state"]
        or any(type(value) is not int for value in numpy_state["state"])
        or type(numpy_state["position"]) is not int
        or type(numpy_state["has_gauss"]) is not int
        or isinstance(numpy_state["cached_gaussian"], bool)
        or not isinstance(numpy_state["cached_gaussian"], (int, float))
        or not math.isfinite(numpy_state["cached_gaussian"])
    ):
        raise ValueError("resume checkpoint contains malformed NumPy RNG state")
    numpy_tuple = (
        numpy_state["bit_generator"],
        np.asarray(numpy_state["state"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    )
    try:
        np.random.RandomState(0).set_state(numpy_tuple)
    except (TypeError, ValueError) as exc:
        raise ValueError("resume checkpoint contains malformed NumPy RNG state") from exc
    try:
        probe = random.Random()
        probe.setstate(state["python_rng_state"])
    except (TypeError, ValueError) as exc:
        raise ValueError("resume checkpoint contains malformed Python RNG state") from exc
    if require_pairing_weight and (
        isinstance(state["pairing_weight"], bool)
        or not isinstance(state["pairing_weight"], (int, float))
        or not math.isfinite(state["pairing_weight"])
    ):
        raise ValueError("resume checkpoint contains malformed pairing weight")
    if require_cuda and version >= 2 and (
        not isinstance(state["cuda_rng_state_all"], list)
        or not state["cuda_rng_state_all"]
        or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            or value.numel() == 0
            for value in state["cuda_rng_state_all"]
        )
    ):
        raise ValueError("resume checkpoint contains malformed CUDA RNG state")
    if require_cuda and version >= 2:
        if len(state["cuda_rng_state_all"]) != max(torch.cuda.device_count(), 1):
            raise ValueError("resume checkpoint contains malformed CUDA RNG state")
        for value in state["cuda_rng_state_all"]:
            if value.numel() != 16:
                raise ValueError("resume checkpoint contains malformed CUDA RNG state")
            if torch.cuda.is_available():
                _trial_generator_state(value, device="cuda", label="CUDA")


def _prevalidate_resume_payload(payload: dict, config: Stage3Config, expected_step: int) -> None:
    """Reject config/adapter/training-state defects before runtime load or seeding."""
    module = _checkpoint_validation_module(config)
    validate_stage3_checkpoint_payload(
        payload, module, expected_config=config.to_dict()
    )
    _validate_training_state_payload(
        payload.get("training_state"),
        expected_step=expected_step,
        require_dropout=True,
        require_pairing=config.camera_rank_weight > 0 or config.pairing_supervision,
        require_pairing_weight=config.pairing_supervision,
        require_cuda=True,
        noise_device="cuda",
        require_version_two=config.pairing_supervision,
    )
    _validate_resume_optimizer_sigma(payload["training_state"], module, config)


def _restore_training_state(state, optimizer, sigma_cycle, view_generator, noise_generator,
                             dropout_generator=None, pairing_generator=None):
    _validate_training_state_payload(
        state,
        expected_step=state.get("step") if isinstance(state, dict) else -1,
        require_dropout=dropout_generator is not None,
        require_pairing=pairing_generator is not None,
        require_cuda=torch.cuda.is_available(),
        noise_device=str(noise_generator.device),
    )
    version = state.get("format_version", 1)
    optimizer.load_state_dict(state["optimizer"])
    sigma_cycle.load_state_dict(state["sigma_cycle"])
    view_generator.set_state(state["view_generator_state"])
    noise_generator.set_state(state["noise_generator_state"])
    if dropout_generator is not None and "dropout_generator_state" in state:
        dropout_generator.set_state(state["dropout_generator_state"])
    if pairing_generator is not None:
        pairing_generator.set_state(state["pairing_generator_state"])
    random.setstate(state["python_rng_state"])
    numpy_state = state["numpy_rng_state"]
    np.random.set_state((
        numpy_state["bit_generator"],
        np.asarray(numpy_state["state"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_rng_state"])
    if "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def _maybe_restore_training_progress(
    checkpoint,
    initialization_mode,
    optimizer,
    sigma_cycle,
    view_generator,
    noise_generator,
    dropout_generator=None,
    pairing_generator=None,
    *,
    expected_step=None,
):
    """Restore continuation state only for an explicit resume operation."""
    if initialization_mode in {"scratch", "model_only"}:
        return
    if initialization_mode != "resume":
        raise ValueError("unknown Stage 3 initialization mode")
    state = None if checkpoint is None else checkpoint.get("training_state")
    if not isinstance(state, dict) or (
        expected_step is not None and state.get("step") != expected_step
    ):
        raise ValueError("resume checkpoint has no matching training state")
    _restore_training_state(
        state,
        optimizer,
        sigma_cycle,
        view_generator,
        noise_generator,
        dropout_generator,
        pairing_generator,
    )


def _fresh_training_start(output: Path, init_checkpoint: Path | None):
    """Create an empty run boundary and inspect only parent model metadata."""
    prepare_output(output)
    if init_checkpoint is None:
        return 1, [], None
    payload = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "rl3dsr-stage3" or payload.get("format_version") != 1:
        raise ValueError("initialization checkpoint is not Stage 3")
    if type(payload.get("step")) is not int or payload["step"] < 0:
        raise ValueError("initialization checkpoint has invalid saved step")
    return 1, [], payload


def _train(args, config: Stage3Config) -> None:
    if args.seed not in config.training_seeds:
        raise ValueError("--seed must be listed in training_seeds")
    resume_checkpoint = getattr(args, "resume", None)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    init_reset_fusion = bool(getattr(args, "init_reset_fusion", False))
    parent_checkpoint = resume_checkpoint if resume_checkpoint is not None else init_checkpoint
    initialization_mode = _initialization_mode(
        resume_checkpoint, init_checkpoint, init_reset_fusion
    )
    _validate_runtime_inputs(args, checkpoint=parent_checkpoint)
    output = Path(args.output_dir).resolve()
    start_step, rows = 1, []
    resume_payload = None
    init_payload = None
    if resume_checkpoint is None:
        start_step, rows, init_payload = _fresh_training_start(output, init_checkpoint)
    else:
        resume_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
        if resume_payload.get("format") != "rl3dsr-stage3":
            raise ValueError("resume checkpoint is not Stage 3")
        resume_step = int(resume_payload.get("step", -1))
        _prevalidate_resume_payload(resume_payload, config, resume_step)
        rows = resume_rows(output / "train_steps.jsonl", resume_step)
        manifest_path = output / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("resume manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_config = json.loads(json.dumps(config.to_dict()))
        manifest_config = dict(manifest.get("config") or {})
        manifest_config.setdefault("target_lr_dropout", 0.0)
        manifest_config.setdefault("allow_self_view_source", True)
        manifest_config.setdefault("camera_rank_weight", 0.0)
        manifest_config.setdefault("camera_rank_margin_ratio", 0.05)
        manifest_config.setdefault("symmetric_correspondence_rank", False)
        manifest_config.setdefault("symmetric_camera_fraction", 0.5)
        manifest_config.setdefault("pairing_temperature", 0.07)
        manifest_config.setdefault("pairing_minimum_coverage", 0.05)
        manifest_config.setdefault("pairing_calibration_batches", 8)
        manifest_config.setdefault("pairing_target_gradient_ratio", 0.25)
        manifest_config.setdefault("pairing_weight_min", 0.01)
        manifest_config.setdefault("pairing_weight_max", 10.0)
        if manifest_config != expected_config or manifest.get("provenance", {}).get("training_seed") != args.seed:
            raise ValueError("resume manifest does not match config and training seed")
        if config.pairing_supervision:
            validate_pairing_resume(manifest, resume_payload, config)
        start_step = resume_step + 1
        if start_step > config.steps:
            raise ValueError("resume checkpoint already reached configured steps")
    _seed_all(args.seed)
    runtime = load_runtime(
        config,
        model_dir=args.model_dir,
        lq_source=args.lq_source,
        lq_checkpoint=args.lq_checkpoint,
        bridge_checkpoint=args.bridge_checkpoint,
        rre_checkpoint=args.rre_checkpoint,
        stage3_checkpoint=parent_checkpoint,
        model_only_initialization=initialization_mode == "model_only",
        reset_initialization_fusion=init_reset_fusion,
        device=args.device,
    )
    runtime.module.train()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    trainable = [parameter for parameter in runtime.module.parameters() if parameter.requires_grad]
    fusion_parameters = (
        tuple(runtime.module.fusion.parameters()) if config.pairing_supervision else ()
    )
    trainable_positions = {id(parameter): index for index, parameter in enumerate(trainable)}
    fusion_trainable_indices = tuple(
        trainable_positions[id(parameter)] for parameter in fusion_parameters
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    sigma_cycle = SigmaCycle(
        FlowSamplingConfig(config.sampling_steps, config.sampling_shift),
        seed=args.seed + 1000,
        strategy="balanced",
    )
    view_generator = torch.Generator().manual_seed(args.seed + 2000)
    noise_generator = torch.Generator(device=runtime.device).manual_seed(args.seed + 3000)
    dropout_generator = torch.Generator().manual_seed(args.seed + 4000)
    pairing_generator = (
        torch.Generator().manual_seed(args.seed + 5000)
        if config.camera_rank_weight > 0 or config.pairing_supervision else None
    )
    _maybe_restore_training_progress(
        runtime.checkpoint,
        initialization_mode,
        optimizer,
        sigma_cycle,
        view_generator,
        noise_generator,
        dropout_generator,
        pairing_generator,
        expected_step=start_step - 1,
    )
    pairing_calibration = None
    if config.pairing_supervision and initialization_mode != "resume":
        pairing_calibration = _calibrate_a5_pairing(runtime, args, config)
        if not math.isclose(pairing_calibration["raw_weight"], pairing_calibration["weight"], rel_tol=1e-9):
            persist_pairing_preflight(output, config, pairing_calibration)
            raise ValueError("A5 calibration weight clipped; stopping before training")
    pairing_weight = resolve_pairing_weight(
        config.pairing_supervision,
        initialization_mode,
        manifest if initialization_mode == "resume" else None,
        runtime.checkpoint,
        lambda: pairing_calibration,
    )
    provenance = {
        "training_seed": args.seed,
        "dataset_root": str(args.dataset_root.resolve()),
        "bridge_checkpoint": str(args.bridge_checkpoint.resolve()),
        "bridge_checkpoint_sha256": _sha256(args.bridge_checkpoint),
        "rre_checkpoint": None if args.rre_checkpoint is None else str(args.rre_checkpoint.resolve()),
        "rre_checkpoint_sha256": None if args.rre_checkpoint is None else _sha256(args.rre_checkpoint),
        "git_revision": _git_revision(),
        "initialization_mode": initialization_mode,
        "copied_modules": [],
        "reset_modules": [],
    }
    if init_checkpoint is not None:
        provenance.update(_initialization_provenance(
            init_checkpoint, init_payload, reset_fusion=init_reset_fusion
        ))
    if resume_checkpoint is None:
        _assert_manifest_initialization(provenance, initialization_mode, init_reset_fusion)
        pairing_manifest = (
            persist_pairing_preflight(output, config, pairing_calibration)
            if config.pairing_supervision else None
        )
        manifest = {
            "config": config.to_dict(),
            "provenance": provenance,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "frozen_wan_parameters": sum(p.numel() for p in runtime.dit.model.parameters()),
            "frozen_vae_parameters": sum(p.numel() for p in runtime.vae.model.model.parameters()),
        }
        if pairing_manifest is not None:
            manifest["pairing"] = pairing_manifest
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    elapsed_prior = float(rows[-1].get("elapsed_seconds", 0.0)) if rows else 0.0
    started = time.perf_counter()
    for step in range(start_step, config.steps + 1):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses, scenes, view_groups, sigmas, target_lr_dropped = [], [], [], [], []
        structure_losses = []
        paired_structure_losses = []
        main_flow_losses, correct_flow_losses, wrong_flow_losses = [], [], []
        rank_losses, rank_active_fractions, rank_gradient_norms = [], [], []
        lr_wrong_flow_losses, lr_rank_losses = [], []
        lr_rank_active_fractions, lr_rank_gradient_norms = [], []
        correspondence_rank_losses, correspondence_rank_gradient_norms = [], []
        per_view_correct_losses, per_view_wrong_losses = [], []
        per_view_lr_wrong_losses = []
        rank_gradient_groups: dict[str, list[float]] = {}
        lr_rank_gradient_groups: dict[str, list[float]] = {}
        pairing_losses, pair_counts, pairing_coverage_minima = [], [], []
        flow_fusion_gradients = rank_fusion_gradients = pairing_fusion_gradients = None
        for micro in range(config.gradient_accumulation):
            scene = config.train_scenes[
                ((step - 1) * config.gradient_accumulation + micro) % len(config.train_scenes)
            ]
            indices, hr, lr, camera = _load_group(
                args.dataset_root, scene, "train", config, view_generator, runtime.device
            )
            pairing_lr = lr
            dropped = (
                config.target_lr_dropout > 0
                and float(torch.rand((), generator=dropout_generator)) < config.target_lr_dropout
            )
            if dropped:
                lr = lr.clone()
                lr[:, :, 0] = 0
            with torch.no_grad():
                clean = runtime.vae.encode_multiview(hr)
            wrong_camera = wrong_permutation = None
            if config.camera_rank_weight > 0 or config.pairing_supervision:
                assert pairing_generator is not None
                if config.symmetric_correspondence_rank:
                    wrong_permutation = _deranged_auxiliary_indices(
                        camera.K.shape[1], pairing_generator
                    )
                    wrong_camera = _camera_with_auxiliary_permutation(
                        camera, wrong_permutation
                    )
                else:
                    wrong_camera = derange_auxiliary_fusion_camera(
                        camera, pairing_generator
                    )
            prepared_result = runtime.module.prepare_multiview(
                lr, camera, tuple(clean.shape[2:]), (config.image_size, config.image_size),
                **({
                    "pairing_camera": wrong_camera,
                    "pairing_minimum_coverage": config.pairing_minimum_coverage,
                    "pairing_lr": pairing_lr if dropped else None,
                } if config.pairing_supervision else {}),
            )
            if config.pairing_supervision:
                prepared, pairing_state = prepared_result
            else:
                prepared, pairing_state = prepared_result, None
            sigma = sigma_cycle.next().reshape(1).to(runtime.device)
            noise = torch.randn(
                clean.shape,
                generator=noise_generator,
                device=runtime.device,
                dtype=clean.dtype,
            )
            noisy, timestep, target = flow_matching_pair(clean, noise, sigma)
            prediction = runtime.module.predict(
                runtime.dit,
                noisy,
                timestep,
                None,
                prepared,
                camera,
                tuple(clean.shape[2:]),
            )
            if config.camera_rank_weight > 0:
                assert wrong_camera is not None
                wrong_prepared = runtime.module.prepare_multiview(
                    lr,
                    wrong_camera,
                    tuple(clean.shape[2:]),
                    (config.image_size, config.image_size),
                )
                prediction_wrong = runtime.module.predict(
                    runtime.dit,
                    noisy,
                    timestep,
                    None,
                    wrong_prepared,
                    camera,
                    tuple(clean.shape[2:]),
                )
                flow_term, e_correct, e_wrong, rank = _camera_pair_training_losses(
                    prediction,
                    prediction_wrong,
                    target,
                    margin_ratio=config.camera_rank_margin_ratio,
                    target_view_only=config.arm == "A6",
                    target_view_flow_fraction=config.target_view_flow_fraction,
                )
                camera_rank_scale = (
                    config.symmetric_camera_fraction
                    if config.symmetric_correspondence_rank else 1.0
                )
                camera_rank_term = config.camera_rank_weight * camera_rank_scale * rank.mean()
                camera_rank_gradients = torch.autograd.grad(
                    camera_rank_term / config.gradient_accumulation,
                    trainable,
                    retain_graph=True,
                    allow_unused=True,
                )
                rank_term = camera_rank_term
                rank_gradients = _accumulate_gradient_vectors(None, camera_rank_gradients)
                lr_rank = None
                if config.symmetric_correspondence_rank:
                    assert wrong_permutation is not None
                    wrong_lr = permute_view_tensor(lr, wrong_permutation)
                    wrong_lr_prepared = runtime.module.prepare_multiview(
                        wrong_lr,
                        camera,
                        tuple(clean.shape[2:]),
                        (config.image_size, config.image_size),
                    )
                    prediction_lr_wrong = runtime.module.predict(
                        runtime.dit,
                        noisy,
                        timestep,
                        None,
                        wrong_lr_prepared,
                        camera,
                        tuple(clean.shape[2:]),
                    )
                    _, _, e_lr_wrong, lr_rank = _camera_pair_training_losses(
                        prediction,
                        prediction_lr_wrong,
                        target,
                        margin_ratio=config.camera_rank_margin_ratio,
                        target_view_only=True,
                    )
                    lr_rank_term = (
                        config.camera_rank_weight
                        * (1 - config.symmetric_camera_fraction)
                        * lr_rank.mean()
                    )
                    lr_rank_gradients = torch.autograd.grad(
                        lr_rank_term / config.gradient_accumulation,
                        trainable,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    combined_rank = _symmetric_correspondence_rank(
                        rank, lr_rank, config.symmetric_camera_fraction
                    )
                    rank_term = config.camera_rank_weight * combined_rank.mean()
                    rank_gradients = _accumulate_gradient_vectors(
                        rank_gradients, lr_rank_gradients
                    )
                    lr_wrong_flow_losses.append(float(e_lr_wrong.mean().detach()))
                    lr_rank_losses.append(float(lr_rank.mean().detach()))
                    lr_rank_active_fractions.append(
                        float((lr_rank.detach() > 0).float().mean())
                    )
                    lr_rank_gradient_norms.append(
                        _tensor_gradient_norm(lr_rank_gradients)
                    )
                    correspondence_rank_losses.append(
                        float(combined_rank.mean().detach())
                    )
                    correspondence_rank_gradient_norms.append(
                        _tensor_gradient_norm(rank_gradients)
                    )
                    per_view_lr_wrong_losses.append(
                        per_view_flow_losses(prediction_lr_wrong, target)
                        .detach().mean(dim=0).float().cpu().tolist()
                    )
                    for name, value in _camera_rank_gradient_groups(
                        runtime.module, trainable, lr_rank_gradients
                    ).items():
                        lr_rank_gradient_groups.setdefault(
                            name.replace("camera_rank_", "lr_rank_"), []
                        ).append(value)
                loss = flow_term + rank_term
                if config.correct_image_ssim_weight is not None:
                    structure_term = decoded_target_ssim_loss(
                        runtime.vae, noisy, prediction, sigma, hr
                    )
                    loss = loss + config.correct_image_ssim_weight * structure_term
                    structure_losses.append(float(structure_term.detach()))
                    if config.paired_image_ssim_rank:
                        wrong_structure_term = decoded_target_ssim_loss(
                            runtime.vae, noisy, prediction_wrong, sigma, hr
                        )
                        paired_structure_term = F.relu(
                            0.0003 + structure_term - wrong_structure_term
                        )
                        loss = loss + config.correct_image_ssim_weight * paired_structure_term
                        paired_structure_losses.append(float(paired_structure_term.detach()))
                correct_flow_losses.append(float(e_correct.mean().detach()))
                wrong_flow_losses.append(float(e_wrong.mean().detach()))
                rank_losses.append(float(rank.mean().detach()))
                rank_active_fractions.append(float((rank.detach() > 0).float().mean()))
                rank_gradient_norms.append(_tensor_gradient_norm(camera_rank_gradients))
                if config.arm == "A6":
                    per_view_correct_losses.append(
                        per_view_flow_losses(prediction, target).detach().mean(dim=0).float().cpu().tolist()
                    )
                    per_view_wrong_losses.append(
                        per_view_flow_losses(prediction_wrong, target).detach().mean(dim=0).float().cpu().tolist()
                    )
                    for name, value in _camera_rank_gradient_groups(
                        runtime.module, trainable, camera_rank_gradients
                    ).items():
                        rank_gradient_groups.setdefault(name, []).append(value)
                micro_rank_fusion_gradients = tuple(
                    rank_gradients[index] for index in fusion_trainable_indices
                )
            else:
                flow_term = flow_matching_loss(prediction, target)
                rank_term = prediction.new_zeros(())
                loss = flow_term
                micro_rank_fusion_gradients = ()
            main_flow_losses.append(float(flow_term.detach()))
            if config.pairing_supervision:
                assert pairing_state is not None
                pairing_term = pairing_info_nce_loss(
                    pairing_state, temperature=config.pairing_temperature
                )
                micro_flow_gradients = torch.autograd.grad(
                    flow_term / config.gradient_accumulation,
                    fusion_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                micro_pairing_gradients = torch.autograd.grad(
                    pairing_term / config.gradient_accumulation,
                    fusion_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                loss = flow_term + rank_term + pairing_weight * pairing_term
                pairing_losses.append(float(pairing_term.detach()))
                pair_counts.append(sum(
                    int(row["target_patches"].numel()) for row in pairing_state["pairs"]
                ))
                pairing_coverage_minima.append(min(
                    float(row["coverage"]) for row in pairing_state["pairs"]
                ))
                flow_fusion_gradients = _accumulate_gradient_vectors(
                    flow_fusion_gradients, micro_flow_gradients
                )
                rank_fusion_gradients = _accumulate_gradient_vectors(
                    rank_fusion_gradients, micro_rank_fusion_gradients
                )
                pairing_fusion_gradients = _accumulate_gradient_vectors(
                    pairing_fusion_gradients, micro_pairing_gradients
                )
            (loss / config.gradient_accumulation).backward()
            losses.append(float(loss.detach()))
            scenes.append(scene)
            view_groups.append(indices)
            sigmas.append(float(sigma))
            target_lr_dropped.append(bool(dropped))
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
            raise RuntimeError("trainable adapter gradient is missing or non-finite")
        if any(p.grad is not None for p in runtime.dit.model.parameters()):
            raise RuntimeError("frozen Wan received a gradient")
        bridge_gradient_norm = _module_grad_norm(runtime.module.conditioner.bridge)
        rre_gradient_norm = _module_grad_norm(runtime.module.geometry)
        fusion_gradient_norm = _module_grad_norm(runtime.module.fusion)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip))
        optimizer.step()
        row = {
            "step": step,
            "loss": float(np.mean(losses)),
            "micro_losses": losses,
            "gradient_norm": gradient_norm,
            "bridge_gradient_norm": bridge_gradient_norm,
            "rre_gradient_norm": rre_gradient_norm,
            "fusion_gradient_norm": fusion_gradient_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "scenes": scenes,
            "view_indices": view_groups,
            "sigmas": sigmas,
            "target_lr_dropped": target_lr_dropped,
            "target_lr_drop_count": sum(target_lr_dropped),
            "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
            "step_seconds": time.perf_counter() - step_started,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(runtime.device) / 2**20,
        }
        if config.correct_image_ssim_weight is not None:
            row["correct_image_ssim_loss"] = float(np.mean(structure_losses))
            row["correct_image_ssim_weight"] = config.correct_image_ssim_weight
            if config.paired_image_ssim_rank:
                row["paired_image_ssim_rank_loss"] = float(np.mean(paired_structure_losses))
        if config.camera_rank_weight > 0:
            row.update({
                "correct_flow_loss": float(np.mean(correct_flow_losses)),
                "wrong_flow_loss": float(np.mean(wrong_flow_losses)),
                "camera_rank_loss": float(np.mean(rank_losses)),
                "camera_rank_active_fraction": float(np.mean(rank_active_fractions)),
                "camera_rank_gradient_norm": float(np.mean(rank_gradient_norms)),
            })
            if config.arm == "A6":
                row.update({
                    "camera_rank_scope": "target_view_0",
                    "per_view_correct_flow_loss": np.mean(per_view_correct_losses, axis=0).tolist(),
                    "per_view_wrong_flow_loss": np.mean(per_view_wrong_losses, axis=0).tolist(),
                    **{
                        name: float(np.mean(values))
                        for name, values in rank_gradient_groups.items()
                    },
                })
                if config.symmetric_correspondence_rank:
                    row.update({
                        "lr_rank_scope": "target_view_0",
                        "correspondence_rank_weight": config.camera_rank_weight,
                        "camera_rank_effective_weight": (
                            config.camera_rank_weight * config.symmetric_camera_fraction
                        ),
                        "lr_rank_effective_weight": (
                            config.camera_rank_weight * (1 - config.symmetric_camera_fraction)
                        ),
                        "lr_wrong_flow_loss": float(np.mean(lr_wrong_flow_losses)),
                        "lr_rank_loss": float(np.mean(lr_rank_losses)),
                        "lr_rank_active_fraction": float(
                            np.mean(lr_rank_active_fractions)
                        ),
                        "lr_rank_gradient_norm": float(
                            np.mean(lr_rank_gradient_norms)
                        ),
                        "correspondence_rank_loss": float(
                            np.mean(correspondence_rank_losses)
                        ),
                        "correspondence_rank_gradient_norm": float(
                            np.mean(correspondence_rank_gradient_norms)
                        ),
                        "per_view_lr_wrong_flow_loss": np.mean(
                            per_view_lr_wrong_losses, axis=0
                        ).tolist(),
                        **{
                            name: float(np.mean(values))
                            for name, values in lr_rank_gradient_groups.items()
                        },
                    })
        if config.pairing_supervision:
            row.update(a5_step_telemetry(
                flow_losses=main_flow_losses,
                rank_losses=rank_losses,
                pairing_losses=pairing_losses,
                pair_counts=pair_counts,
                coverage_minima=pairing_coverage_minima,
                pairing_weight=pairing_weight,
                flow_gradient_norm=_tensor_gradient_norm(flow_fusion_gradients),
                rank_gradient_norm=_tensor_gradient_norm(rank_fusion_gradients or ()),
                pairing_gradient_norm=_tensor_gradient_norm(pairing_fusion_gradients),
                final_fusion_gradient_norm=fusion_gradient_norm,
            ))
        rows.append(row)
        _append_jsonl(output / "train_steps.jsonl", row)
        _write_csv(output / "train_steps.csv", rows)
        if step % config.checkpoint_every == 0 or step == config.steps:
            save_stage3_checkpoint(
                output / f"stage3_step_{step:04d}.pt",
                runtime.module,
                config=config.to_dict(),
                step=step,
                provenance=provenance,
                training_state=_training_state(
                    optimizer, sigma_cycle, view_generator, noise_generator, step,
                    dropout_generator, pairing_generator,
                    pairing_weight=pairing_weight if config.pairing_supervision else None,
                ),
            )


class _PerFrameLPIPS(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", reduction="mean", normalize=False
        ).to(device).eval()

    def forward(self, prediction, target):
        return torch.cat(
            [
                self.metric(prediction[i : i + 1], target[i : i + 1]).reshape(1)
                for i in range(prediction.shape[0])
            ]
        )


def _intervention(lr, camera, mode, generator, *, far_camera=None, return_metadata=False):
    """Return LR, fusion camera, geometry camera and source mask.

    The old ``shuffle_camera`` name is retained as a compatibility alias for
    the historical fusion-only intervention. New audits should use the
    explicit ``shuffle_fusion``, ``shuffle_geometry`` or ``shuffle_all`` modes.
    ``shuffle_pair`` is likewise retained only as a historical corruption
    label; it is not a permutation-equivariance control.
    """
    metadata = {"requested_mode": mode}

    def finish(changed, fusion_camera, geometry_camera, mask):
        values = (changed, fusion_camera, geometry_camera, mask)
        return (*values, metadata) if return_metadata else values

    if mode.startswith("no_self_"):
        mode = mode.removeprefix("no_self_")
    if mode in {"correct", "correct_repeat"}:
        changed, changed_camera, mask = intervene_lr(
            lr, camera, "correct", target=0, generator=generator
        )
        return finish(changed, changed_camera, changed_camera, mask)
    if mode == "target_drop":
        changed = lr.clone()
        changed[:, :, 0] = 0
        mask = torch.ones(lr.shape[0], lr.shape[2], dtype=torch.bool, device=lr.device)
        return finish(changed, camera, camera, mask)
    if mode in {"mispaired_lr", "mispaired_camera", "target_drop_shuffle_fusion"}:
        indices = _deranged_auxiliary_indices(lr.shape[2], generator)
        metadata.update({
            "auxiliary_permutation": indices.tolist(),
            "auxiliary_permutation_sha256": _permutation_digest(indices),
        })
        changed = lr.clone()
        fusion_camera = camera
        if mode == "mispaired_lr":
            changed = permute_view_tensor(changed, indices)
        else:
            fusion_camera = _camera_with_auxiliary_permutation(camera, indices)
            if mode == "target_drop_shuffle_fusion":
                changed[:, :, 0] = 0
        mask = torch.ones(lr.shape[0], lr.shape[2], dtype=torch.bool, device=lr.device)
        return finish(changed, fusion_camera, camera, mask)
    if mode == "joint_permute":
        views = lr.shape[2]
        if views < 2:
            raise ValueError("joint permutation requires at least two views")
        permutation = torch.randperm(views, generator=generator)
        if torch.equal(permutation, torch.arange(views)):
            permutation = permutation.roll(1)
        mask = torch.ones(lr.shape[0], views, dtype=torch.bool, device=lr.device)
        result = apply_joint_view_permutation(
            permutation,
            lr=lr,
            fusion_camera=camera,
            geometry_camera=camera,
            source_mask=mask,
        )
        metadata.update({
            "permutation": permutation.tolist(),
            "inverse_permutation": result["inverse_permutation"].tolist(),
            "permutation_sha256": result["permutation_sha256"],
            "target_index": result["target_index"],
        })
        return finish(
            result["lr"], result["fusion_camera"], result["geometry_camera"],
            result["source_mask"],
        )
    if mode in {"fusion_camera_dose_0", "fusion_camera_dose_half", "fusion_camera_dose_full"}:
        wrong = derange_auxiliary_fusion_camera(camera, generator)
        alpha = {
            "fusion_camera_dose_0": 0.0,
            "fusion_camera_dose_half": 0.5,
            "fusion_camera_dose_full": 1.0,
        }[mode]
        metadata.update({"camera_dose_alpha": alpha, "wrong_camera_sha256": _camera_digest(wrong)})
        mask = torch.ones(lr.shape[0], lr.shape[2], dtype=torch.bool, device=lr.device)
        return finish(lr.clone(), interpolate_fusion_camera_dose(camera, wrong, alpha), camera, mask)
    if mode == "far_shuffle_fusion":
        if far_camera is None:
            raise ValueError("far_shuffle_fusion requires a frozen donor camera")
        far_camera.validate(batch=lr.shape[0])
        if far_camera.K.shape[1] != lr.shape[2]:
            raise ValueError("far donor camera view count does not match LR")
        if not torch.equal(far_camera.K[:, :1], camera.K[:, :1]) or not torch.equal(
            far_camera.T_world_from_camera[:, :1], camera.T_world_from_camera[:, :1]
        ):
            raise ValueError("far donor camera must preserve the target camera")
        mask = torch.ones(lr.shape[0], lr.shape[2], dtype=torch.bool, device=lr.device)
        return finish(lr.clone(), far_camera, camera, mask)
    if mode in {"shuffle_fusion", "shuffle_geometry", "shuffle_all", "shuffle_camera"}:
        _, shuffled_camera, mask = intervene_lr(
            lr, camera, "shuffle_camera", target=0, generator=generator
        )
        if mode in {"shuffle_geometry", "shuffle_all"}:
            geometry_camera = shuffled_camera
        else:
            geometry_camera = camera
        if mode in {"shuffle_fusion", "shuffle_all", "shuffle_camera"}:
            fusion_camera = shuffled_camera
        else:
            fusion_camera = camera
        return finish(lr.clone(), fusion_camera, geometry_camera, mask)
    if mode == "shuffle_pair":
        changed, shuffled_camera, mask = shuffle_auxiliary_pairs(
            lr, camera, target=0, generator=generator
        )
        return finish(changed, shuffled_camera, shuffled_camera, mask)
    if mode == "local_patch":
        h, w = lr.shape[-2:]
        changed, changed_camera, mask = intervene_lr(
            lr,
            camera,
            mode,
            target=0,
            source=1,
            box=(h // 4, w // 4, max(h // 4 + 1, h // 2), max(w // 4 + 1, w // 2)),
            generator=generator,
        )
        return finish(changed, changed_camera, changed_camera, mask)
    changed, changed_camera, mask = intervene_lr(
        lr, camera, mode, target=0, generator=generator
    )
    return finish(changed, changed_camera, changed_camera, mask)


def _unpack_intervention(result, mode: str):
    """Accept the historical four-tuple and the metadata-bearing five-tuple."""
    if len(result) == 4:
        return (*result, {"requested_mode": mode})
    if len(result) == 5:
        return result
    raise ValueError("intervention must return four or five values")


def inverse_permute_joint_output(output: torch.Tensor, metadata: dict) -> torch.Tensor:
    """Return a jointly permuted prediction to the original view labeling."""
    permutation = metadata.get("permutation")
    if permutation is None:
        return output
    return permute_view_tensor(output, inverse_view_permutation(permutation))


def _canonicalize_prepared_views(
    prepared: torch.Tensor, permutation, views: int
) -> torch.Tensor:
    if permutation is None:
        return prepared
    if prepared.ndim != 3 or prepared.shape[1] % views:
        raise ValueError("prepared features do not split into view slots")
    shaped = prepared.reshape(prepared.shape[0], views, -1, prepared.shape[-1])
    inverse = inverse_view_permutation(permutation).to(prepared.device)
    return shaped.index_select(1, inverse).reshape_as(prepared)


def _save_frame(path: Path, video: torch.Tensor) -> None:
    from PIL import Image

    frame = video[0, :, 0].detach().float().clamp(-1, 1).add(1).mul(127.5)
    array = frame.permute(1, 2, 0).round().byte().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _metric_row(metrics: dict, *, config: Stage3Config, training_seed, checkpoint: Path,
                step: int, inference_seed, group: dict, condition: str) -> dict:
    return {
        "arm": config.arm,
        "train_seed": training_seed,
        "inference_seed": inference_seed,
        "checkpoint": str(checkpoint.resolve()),
        "step": step,
        "scene": group["scene"],
        "split": "train",
        "scope": "seen_train_sr",
        "condition": condition,
        "group_id": group["id"],
        "view_index": group["anchor"],
        "view_indices": group["indices"],
        **{
            name: float(value)
            for name, value in metrics.items()
            if name not in {"batch_index", "frame_index"}
        },
    }


def _feature_diagnostics(runtime: Runtime, lr: torch.Tensor, camera: CameraBatch,
                         clean_shape: tuple[int, ...], config: Stage3Config,
                         changed_lr: torch.Tensor, fusion_camera: CameraBatch,
                         geometry_camera: CameraBatch,
                         source_mask: torch.Tensor, group: dict, condition: str,
                         allow_self_view_source: bool | None = None,
                         view_permutation=None) -> dict:
    """Measure how an intervention changes prepared and bridge residual features."""
    fusion_module = getattr(runtime.module, "fusion", None)
    if fusion_module is not None:
        fusion_module.record_diagnostics = True
    correct = runtime.module.prepare_multiview(
        lr,
        camera,
        tuple(clean_shape[2:]),
        (config.image_size, config.image_size),
        allow_self_view_source=allow_self_view_source,
    )
    correct_fusion = _scalar_diagnostics(
        getattr(fusion_module, "last_diagnostics", {})
    )
    changed = runtime.module.prepare_multiview(
        changed_lr,
        fusion_camera,
        tuple(clean_shape[2:]),
        (config.image_size, config.image_size),
        source_mask=source_mask,
        allow_self_view_source=allow_self_view_source,
    )
    changed = _canonicalize_prepared_views(
        changed, view_permutation, tuple(clean_shape[2:])[0]
    )
    changed_fusion = _scalar_diagnostics(
        getattr(fusion_module, "last_diagnostics", {})
    )
    delta = (correct.float() - changed.float()).reshape(
        correct.shape[0], tuple(clean_shape[2:])[0], -1, correct.shape[-1]
    )
    correct_views = correct.float().reshape_as(delta)
    timestep = torch.full(
        (correct.shape[0],), 0.5, device=correct.device, dtype=torch.float32
    )
    bridge_correct = runtime.module.conditioner.bridge_residuals(correct, timestep)
    bridge_changed = runtime.module.conditioner.bridge_residuals(changed, timestep)
    bridge = {}
    for block in sorted(bridge_correct):
        block_delta = bridge_correct[block].float() - bridge_changed[block].float()
        bridge["block_" + str(block)] = {
            "correct_norm": float(bridge_correct[block].float().norm()),
            "delta_norm": float(block_delta.norm()),
            "relative_delta": float(
                block_delta.norm() / bridge_correct[block].float().norm().clamp_min(1e-12)
            ),
        }
    return {
        "group_id": group["id"],
        "scene": group["scene"],
        "view_index": group["anchor"],
        "condition": condition,
        "correct_camera_sha256": _camera_digest(camera),
        "fusion_camera_sha256": _camera_digest(fusion_camera),
        "geometry_camera_sha256": _camera_digest(geometry_camera),
        "fusion_camera_changed": _camera_digest(camera) != _camera_digest(fusion_camera),
        "geometry_camera_changed": _camera_digest(camera) != _camera_digest(geometry_camera),
        "feature_norm": float(correct.float().norm()),
        "prepared_feature_delta": float(delta.norm()),
        "feature_delta_norm": float(delta.norm()),
        "feature_relative_delta": float(
            delta.norm() / correct.float().norm().clamp_min(1e-12)
        ),
        "view_relative_delta": [
            float(delta[:, view].norm() / correct_views[:, view].norm().clamp_min(1e-12))
            for view in range(delta.shape[1])
        ],
        "bridge": bridge,
        "fusion_correct": correct_fusion,
        "fusion_changed": changed_fusion,
    }


def _trace_delta(correct: dict, changed: dict) -> dict:
    """Compare per-step diagnostics captured with identical initial noise."""
    if not torch.equal(correct.get("initial_noise"), changed.get("initial_noise")):
        raise RuntimeError("paired diagnostics used different initial noise")
    correct_velocities = correct.get("velocity_trace", [])
    changed_velocities = changed.get("velocity_trace", [])
    if len(correct_velocities) != len(changed_velocities):
        raise RuntimeError("diagnostic velocity traces have different lengths")
    velocity_deltas = []
    for reference, value in zip(correct_velocities, changed_velocities):
        reference = reference.float()
        value = value.float()
        delta = (reference - value).norm()
        velocity_deltas.append({
            "absolute": float(delta),
            "relative": float(delta / reference.norm().clamp_min(1e-12)),
        })

    def block_delta(key: str) -> list[dict[str, float]]:
        reference_trace = correct.get(key, [])
        changed_trace = changed.get(key, [])
        if len(reference_trace) != len(changed_trace):
            raise RuntimeError(f"diagnostic {key} traces have different lengths")
        rows = []
        for reference, value in zip(reference_trace, changed_trace):
            blocks = sorted(set(reference) | set(value))
            row = {}
            for block in blocks:
                ref_values = reference.get(block, {})
                changed_values = value.get(block, {})
                ref_rms = float(ref_values.get("residual_rms", ref_values.get("camera_residual_rms", 0.0)))
                changed_rms = float(changed_values.get("residual_rms", changed_values.get("camera_residual_rms", 0.0)))
                row[str(block)] = {
                    "absolute_rms_delta": abs(ref_rms - changed_rms),
                    "relative_rms_delta": abs(ref_rms - changed_rms) / max(abs(ref_rms), 1e-12),
                }
            rows.append(row)
        return rows

    geometry_deltas = block_delta("geometry_trace")
    injection_deltas = block_delta("injection_trace")
    return {
        "per_step_velocity_delta": velocity_deltas,
        "per_step_velocity_delta_mean": float(np.mean([row["absolute"] for row in velocity_deltas])) if velocity_deltas else 0.0,
        "per_step_velocity_delta_relative_mean": float(np.mean([row["relative"] for row in velocity_deltas])) if velocity_deltas else 0.0,
        "per_block_geometry_residual_delta": geometry_deltas,
        "per_block_injection_delta": injection_deltas,
    }


def _write_evaluation(output: Path, rows: list[dict], baseline_rows: list[dict], summary: dict) -> None:
    for name, values in (("evaluation_rows", rows), ("baseline_rows", baseline_rows)):
        with (output / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        _write_csv(output / f"{name}.csv", values)
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _evaluate_seen(args, config: Stage3Config) -> None:
    _validate_runtime_inputs(args, checkpoint=args.checkpoint)
    if not args.inference_seeds or len(set(args.inference_seeds)) != len(args.inference_seeds):
        raise ValueError("inference seeds must be a nonempty unique list")
    if any(seed < 0 for seed in args.inference_seeds):
        raise ValueError("inference seeds must be nonnegative")
    manifest, groups = _load_seen_groups(
        args.seen_manifest, config, args.subset, args.group_ids
    )
    far_modes = {"far_shuffle_fusion", "no_self_far_shuffle_fusion"}
    needs_far_camera = bool(set(args.modes) & far_modes)
    donor_manifest = getattr(args, "camera_donor_manifest", None)
    if needs_far_camera and donor_manifest is None:
        raise ValueError("far fusion-camera modes require --camera-donor-manifest")
    camera_donors = (
        _load_camera_donors(donor_manifest, args.seen_manifest, config, groups)
        if donor_manifest is not None
        else {}
    )
    output = prepare_output(args.output_dir)
    _seed_all(args.inference_seeds[0])
    runtime = load_runtime(
        config,
        model_dir=args.model_dir,
        lq_source=args.lq_source,
        lq_checkpoint=args.lq_checkpoint,
        bridge_checkpoint=args.bridge_checkpoint,
        rre_checkpoint=args.rre_checkpoint,
        stage3_checkpoint=args.checkpoint,
        device=args.device,
    )
    runtime.module.eval()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    metric = _PerFrameLPIPS(runtime.device)
    payload = runtime.checkpoint or {}
    step = int(payload.get("step", -1))
    training_seed = payload.get("provenance", {}).get("training_seed")
    rows, baseline_rows, diagnostic_rows = [], [], []
    scene_order = {scene: index for index, scene in enumerate(config.train_scenes)}
    for group in groups:
        _, hr, lr, camera = _load_indices(
            args.dataset_root, group["scene"], "train", group["indices"], config, runtime.device
        )
        far_camera = None
        if group["id"] in camera_donors:
            far_camera = _camera_for_indices(
                args.dataset_root,
                group["scene"],
                camera_donors[group["id"]]["fusion_camera_indices"],
                config,
                runtime.device,
            )
        with torch.inference_mode():
            clean = runtime.vae.encode_multiview(hr)
            target = hr[:, :, :1]
            vae_ceiling = runtime.vae.decode_multiview(clean)[:, :, :1]
            bicubic = F.interpolate(
                lr[:, :, 0],
                size=(config.image_size, config.image_size),
                mode="bicubic",
                align_corners=False,
            ).unsqueeze(2).clamp(-1, 1)
            nearest = F.interpolate(
                lr[:, :, 0],
                size=(config.image_size, config.image_size),
                mode="nearest",
            ).unsqueeze(2)
            for condition, prediction in (("bicubic", bicubic), ("vae_ceiling", vae_ceiling)):
                item = frame_metrics(prediction, target, perceptual_metric=metric)[0]
                baseline_rows.append(_metric_row(
                    item,
                    config=config,
                    training_seed=training_seed,
                    checkpoint=args.checkpoint,
                    step=step,
                    inference_seed=None,
                    group=group,
                    condition=condition,
                ))
            if args.save_images:
                reference = output / "images" / group["scene"] / f"view_{group['anchor']:03d}" / "reference"
                _save_frame(reference / "hr.png", target)
                _save_frame(reference / "lr_nearest.png", nearest)
                _save_frame(reference / "bicubic.png", bicubic)
                _save_frame(reference / "vae_ceiling.png", vae_ceiling)
            for inference_seed in args.inference_seeds:
                sample_seed = (
                    inference_seed * 1_000_000
                    + scene_order[group["scene"]] * 1_000
                    + group["anchor"]
                )
                mode_diagnostics = {}
                mode_interventions = {}
                for mode_index, mode in enumerate(args.modes):
                    intervention_offset = 0 if mode.startswith("fusion_camera_dose_") else mode_index
                    intervention_generator = torch.Generator().manual_seed(
                        sample_seed * 10 + intervention_offset
                    )
                    changed_lr, fusion_camera, geometry_camera, source_mask, intervention_metadata = _unpack_intervention(
                        _intervention(
                            lr, camera, mode, intervention_generator, far_camera=far_camera,
                            return_metadata=True,
                        ),
                        mode,
                    )
                    self_source_override = False if mode.startswith("no_self_") else None
                    sampled = sample_latents(
                        runtime,
                        changed_lr,
                        camera,
                        tuple(clean.shape),
                        config.sampling_steps,
                        config.image_size,
                        fusion_camera=fusion_camera,
                        geometry_camera=geometry_camera,
                        source_mask=source_mask,
                        allow_self_view_source=self_source_override,
                        seed=sample_seed,
                        sampling_shift=config.sampling_shift,
                        dtype=torch.bfloat16,
                        return_diagnostics=getattr(args, "save_diagnostics", False),
                        view_permutation=intervention_metadata.get("permutation"),
                    )
                    if getattr(args, "save_diagnostics", False):
                        latent, mode_diagnostics[mode] = sampled
                        mode_interventions[mode] = (
                            changed_lr, fusion_camera, geometry_camera, source_mask,
                            self_source_override, intervention_metadata,
                        )
                    else:
                        latent = sampled
                    decoded = runtime.vae.decode_multiview(latent)
                    decoded = inverse_permute_joint_output(decoded, intervention_metadata)[:, :, :1]
                    item = frame_metrics(decoded, target, perceptual_metric=metric)[0]
                    metric_row = _metric_row(
                        item,
                        config=config,
                        training_seed=training_seed,
                        checkpoint=args.checkpoint,
                        step=step,
                        inference_seed=inference_seed,
                        group=group,
                        condition=mode,
                    )
                    metric_row["intervention"] = intervention_metadata
                    rows.append(metric_row)
                    if args.save_images:
                        image_path = (
                            output / "images" / group["scene"] / f"view_{group['anchor']:03d}"
                            / f"seed_{inference_seed}" / f"{mode}.png"
                        )
                        _save_frame(image_path, decoded)
                if getattr(args, "save_diagnostics", False) and "correct" in mode_diagnostics:
                    for mode, intervention in mode_interventions.items():
                        if mode in {"correct", "no_self_correct"}:
                            continue
                        changed_lr, fusion_camera, geometry_camera, source_mask, self_source_override, intervention_metadata = intervention
                        reference_mode = (
                            "no_self_correct"
                            if mode.startswith("no_self_") and "no_self_correct" in mode_diagnostics
                            else "correct"
                        )
                        diagnostic = _feature_diagnostics(
                            runtime,
                            lr,
                            camera,
                            tuple(clean.shape),
                            config,
                            changed_lr,
                            fusion_camera,
                            geometry_camera,
                            source_mask,
                            group,
                            mode,
                            self_source_override,
                            intervention_metadata.get("permutation"),
                        )
                        diagnostic.update(_trace_delta(mode_diagnostics[reference_mode], mode_diagnostics[mode]))
                        diagnostic.update({
                            "inference_seed": inference_seed,
                            "sample_seed": sample_seed,
                            "reference_condition": reference_mode,
                            "intervention": intervention_metadata,
                        })
                        diagnostic_rows.append(diagnostic)
    expected = len(groups) * len(args.inference_seeds) * len(args.modes)
    if len(rows) != expected or len(baseline_rows) != len(groups) * 2:
        raise RuntimeError("seen-view evaluation row count mismatch")
    _write_evaluation(
        output,
        rows,
        baseline_rows,
        {
            "scope": "seen_train_sr",
            "subset": args.subset,
            "arm": config.arm,
            "training_seed": training_seed,
            "checkpoint": str(args.checkpoint.resolve()),
            "step": step,
            "conditions": list(args.modes),
            "inference_seeds": list(args.inference_seeds),
            "groups": len(groups),
            "rows": len(rows),
            "baseline_rows": len(baseline_rows),
            "diagnostic_rows": len(diagnostic_rows),
            "seen_manifest": str(args.seen_manifest.resolve()),
            "seen_manifest_sha256": _sha256(args.seen_manifest),
            "camera_donor_manifest": None if donor_manifest is None else str(donor_manifest.resolve()),
            "camera_donor_manifest_sha256": None if donor_manifest is None else _sha256(donor_manifest),
            "dataset_manifest": manifest["datasets"],
            "diagnostics": bool(diagnostic_rows),
        },
    )
    if diagnostic_rows:
        with (output / "diagnostics.jsonl").open("x", encoding="utf-8") as stream:
            for row in diagnostic_rows:
                stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _evaluate(
    args,
    config: Stage3Config,
    *,
    phase: str,
    checkpoint: Path,
    modes: tuple[str, ...] = ("correct",),
) -> None:
    _validate_runtime_inputs(args, checkpoint=checkpoint)
    output = prepare_output(args.output_dir)
    runtime = load_runtime(
        config,
        model_dir=args.model_dir,
        lq_source=args.lq_source,
        lq_checkpoint=args.lq_checkpoint,
        bridge_checkpoint=args.bridge_checkpoint,
        rre_checkpoint=args.rre_checkpoint,
        stage3_checkpoint=checkpoint,
        device=args.device,
    )
    runtime.module.eval()
    metric = _PerFrameLPIPS(runtime.device)
    payload = runtime.checkpoint or {}
    step = int(payload.get("step", -1))
    training_seed = payload.get("provenance", {}).get("training_seed")
    routes = scene_routes(config, phase)
    groups = (
        config.final_groups_per_scene if phase == "test" else config.validation_groups_per_scene
    )
    inference_seeds = (
        config.final_inference_seeds if phase == "test" else (config.validation_inference_seed,)
    )
    rows = []
    for inference_seed in inference_seeds:
        view_generator = torch.Generator().manual_seed(inference_seed)
        for scene_index, (scene, route) in enumerate(routes):
            for group in range(groups):
                indices, hr, lr, camera = _load_group(
                    args.dataset_root, scene, route, config, view_generator, runtime.device
                )
                with torch.inference_mode():
                    latent_shape = tuple(runtime.vae.encode_multiview(hr).shape)
                    for mode_index, mode in enumerate(modes):
                        intervention_offset = 0 if mode.startswith("fusion_camera_dose_") else mode_index
                        intervention_generator = torch.Generator().manual_seed(
                            inference_seed * 1000000 + scene_index * 10000 + group * 100 + intervention_offset
                        )
                        changed_lr, fusion_camera, geometry_camera, source_mask, intervention_metadata = _unpack_intervention(
                            _intervention(
                                lr, camera, mode, intervention_generator, return_metadata=True
                            ),
                            mode,
                        )
                        latent = sample_latents(
                            runtime,
                            changed_lr,
                            camera,
                            latent_shape,
                            config.sampling_steps,
                            config.image_size,
                            fusion_camera=fusion_camera,
                            geometry_camera=geometry_camera,
                            source_mask=source_mask,
                            seed=inference_seed * 10000 + group,
                            sampling_shift=config.sampling_shift,
                            dtype=torch.bfloat16,
                            view_permutation=intervention_metadata.get("permutation"),
                        )
                        decoded = inverse_permute_joint_output(
                            runtime.vae.decode_multiview(latent), intervention_metadata
                        )
                        metrics = frame_metrics(decoded, hr, perceptual_metric=metric)
                        selected = metrics if modes == ("correct",) else metrics[:1]
                        for item in selected:
                            position = int(item["frame_index"])
                            rows.append(
                                {
                                    "arm": config.arm,
                                    "train_seed": training_seed,
                                    "inference_seed": inference_seed,
                                    "checkpoint": str(checkpoint.resolve()),
                                    "step": step,
                                    "scene": scene,
                                    "split": "test" if phase == "test" else "validation",
                                    "condition": mode,
                                    "intervention": intervention_metadata,
                                    "group": group,
                                    "view_position": position,
                                    "view_index": indices[position],
                                    **{
                                        name: float(value)
                                        for name, value in item.items()
                                        if name not in {"batch_index", "frame_index"}
                                    },
                                }
                            )
    with (output / "evaluation_rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    _write_csv(output / "evaluation_rows.csv", rows)
    (output / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "arm": config.arm,
                "checkpoint": str(checkpoint.resolve()),
                "step": step,
                "conditions": list(modes),
                "rows": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_rows(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _add_runtime_paths(parser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lq-source", type=Path, required=True)
    parser.add_argument("--lq-checkpoint", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--rre-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--config", type=Path, required=True)
    prepare_seen = subparsers.add_parser("prepare-seen")
    prepare_seen.add_argument("--config", type=Path, required=True)
    prepare_seen.add_argument("--dataset-root", type=Path, required=True)
    prepare_seen.add_argument("--manifest", type=Path, required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    _add_runtime_paths(train)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    restart = train.add_mutually_exclusive_group()
    restart.add_argument("--resume", type=Path)
    restart.add_argument("--init-checkpoint", type=Path)
    train.add_argument("--init-reset-fusion", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", type=Path, required=True)
    _add_runtime_paths(validate)
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    intervene = subparsers.add_parser("intervene")
    intervene.add_argument("--config", type=Path, required=True)
    _add_runtime_paths(intervene)
    intervene.add_argument("--checkpoint", type=Path, required=True)
    intervene.add_argument("--output-dir", type=Path, required=True)
    intervene.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "correct", "remove", "duplicate", "shuffle_camera", "shuffle_fusion",
            "shuffle_geometry", "shuffle_all", "shuffle_pair", "local_patch",
            "target_drop", "target_drop_shuffle_fusion", "mispaired_lr",
            "mispaired_camera", "joint_permute", "fusion_camera_dose_0",
            "fusion_camera_dose_half", "fusion_camera_dose_full",
        ),
        default=("correct", "remove", "duplicate", "shuffle_camera", "local_patch"),
    )
    intervene.add_argument("--save-diagnostics", action="store_true")
    seen = subparsers.add_parser("seen-eval")
    seen.add_argument("--config", type=Path, required=True)
    _add_runtime_paths(seen)
    seen.add_argument("--checkpoint", type=Path, required=True)
    seen.add_argument("--output-dir", type=Path, required=True)
    seen.add_argument("--seen-manifest", type=Path, required=True)
    seen.add_argument("--subset", choices=("probe", "full", "intervention"), required=True)
    seen.add_argument("--inference-seeds", type=int, nargs="+", required=True)
    seen.add_argument("--camera-donor-manifest", type=Path)
    seen.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "correct", "correct_repeat", "target_drop", "remove", "duplicate",
            "shuffle_camera", "shuffle_fusion", "shuffle_geometry", "shuffle_all",
            "shuffle_pair", "far_shuffle_fusion", "no_self_correct",
            "no_self_shuffle_fusion", "no_self_far_shuffle_fusion",
            "target_drop_shuffle_fusion", "mispaired_lr", "mispaired_camera",
            "joint_permute", "fusion_camera_dose_0", "fusion_camera_dose_half",
            "fusion_camera_dose_full",
        ),
        default=("correct",),
    )
    seen.add_argument("--save-diagnostics", action="store_true")
    seen.add_argument("--group-ids", nargs="+")
    seen.add_argument("--save-images", action="store_true")
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--rows", type=Path, nargs="+", required=True)
    freeze.add_argument("--manifest", type=Path, required=True)
    test = subparsers.add_parser("test")
    test.add_argument("--config", type=Path, required=True)
    _add_runtime_paths(test)
    test.add_argument("--manifest", type=Path, required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_stage3_config(args.config)
    if args.command == "inspect":
        print(json.dumps({**config.to_dict(), "fusion_mode": config.fusion_mode}, indent=2))
        return
    if args.command == "prepare-seen":
        _prepare_seen_manifest(args, config)
        return
    if args.command == "freeze":
        rows = [row for row in _read_rows(args.rows) if row.get("condition") == "correct"]
        candidate = select_candidate(rows, config.validation_scene_names)
        print(json.dumps(freeze_candidate(args.manifest, candidate, config), indent=2))
        return
    if args.command == "train":
        _train(args, config)
        return
    if args.command == "validate":
        _evaluate(args, config, phase="validate", checkpoint=args.checkpoint)
        return
    if args.command == "intervene":
        _evaluate(args, config, phase="validate", checkpoint=args.checkpoint, modes=tuple(args.modes))
        return
    if args.command == "seen-eval":
        _evaluate_seen(args, config)
        return
    _validate_runtime_inputs(args)
    prepare_output(args.output_dir)
    Path(args.output_dir).rmdir()
    frozen = claim_final_evaluation(args.manifest, config, args.output_dir)
    _evaluate(
        args,
        config,
        phase="test",
        checkpoint=Path(frozen["candidate"]["checkpoint"]),
    )


if __name__ == "__main__":
    main()
