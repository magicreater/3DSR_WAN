#!/usr/bin/env python3
"""Stage 3 shared-training and evaluation entry points.

Nothing runs at import time. ``inspect`` and ``freeze`` do not load model
assets; the remaining commands require explicit paths and are intended to be
started later by the experiment operator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from rl3dsr.data import NeRFSyntheticAdapter, Split
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
    return {
        "train_scenes": list(config.train_scenes),
        "image_size": config.image_size,
        "scale": config.scale,
        "views": config.views,
    }


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
            velocity_trace.append(velocity.detach().float().cpu())
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
        "initial_noise": noise.detach().float().cpu(),
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
    adapter = NeRFSyntheticAdapter(dataset_root / scene)
    sequence = adapter.index(Split.TRAIN if route == "train" else Split.TEST)
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
    adapter = NeRFSyntheticAdapter(dataset_root / scene)
    sequence = adapter.index(Split.TRAIN if route == "train" else Split.TEST)
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
        adapter = NeRFSyntheticAdapter(scene_root)
        sequence = adapter.index(Split.TRAIN)
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
        transforms = scene_root / "transforms_train.json"
        image_root = scene_root / "train"
        datasets[scene] = {
            "views": count,
            "transforms_train_sha256": _sha256(transforms),
            "train_images_sha256": _sha256_tree(image_root),
        }
    payload = {
        "version": 1,
        "scope": "seen_train_sr",
        "sampling_signature": _sampling_signature(config),
        "dataset_root": str(args.dataset_root.resolve()),
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
    sequence = NeRFSyntheticAdapter(dataset_root / scene).index(Split.TRAIN)
    if any(type(index) is not int or not 0 <= index < len(sequence.observations) for index in indices):
        raise ValueError("camera donor index is out of range")
    observations = [sequence.observations[index] for index in indices]
    return _camera(observations, config.image_size, device)


def _training_state(optimizer, sigma_cycle, view_generator, noise_generator, step,
                    dropout_generator=None):
    numpy_state = np.random.get_state()
    state = {
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


def _restore_training_state(state, optimizer, sigma_cycle, view_generator, noise_generator,
                            dropout_generator=None):
    optimizer.load_state_dict(state["optimizer"])
    sigma_cycle.load_state_dict(state["sigma_cycle"])
    view_generator.set_state(state["view_generator_state"])
    noise_generator.set_state(state["noise_generator_state"])
    if dropout_generator is not None and "dropout_generator_state" in state:
        dropout_generator.set_state(state["dropout_generator_state"])
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


def _train(args, config: Stage3Config) -> None:
    if args.seed not in config.training_seeds:
        raise ValueError("--seed must be listed in training_seeds")
    _validate_runtime_inputs(args, checkpoint=args.resume)
    _seed_all(args.seed)
    output = Path(args.output_dir).resolve()
    start_step, rows = 1, []
    resume_payload = None
    if args.resume is None:
        prepare_output(output)
    else:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume_payload.get("format") != "rl3dsr-stage3":
            raise ValueError("resume checkpoint is not Stage 3")
        resume_step = int(resume_payload.get("step", -1))
        rows = resume_rows(output / "train_steps.jsonl", resume_step)
        manifest_path = output / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("resume manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_config = json.loads(json.dumps(config.to_dict()))
        manifest_config = dict(manifest.get("config") or {})
        manifest_config.setdefault("target_lr_dropout", 0.0)
        manifest_config.setdefault("allow_self_view_source", True)
        if manifest_config != expected_config or manifest.get("provenance", {}).get("training_seed") != args.seed:
            raise ValueError("resume manifest does not match config and training seed")
        start_step = resume_step + 1
        if start_step > config.steps:
            raise ValueError("resume checkpoint already reached configured steps")
    runtime = load_runtime(
        config,
        model_dir=args.model_dir,
        lq_source=args.lq_source,
        lq_checkpoint=args.lq_checkpoint,
        bridge_checkpoint=args.bridge_checkpoint,
        rre_checkpoint=args.rre_checkpoint,
        stage3_checkpoint=args.resume,
        device=args.device,
    )
    runtime.module.train()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    trainable = [parameter for parameter in runtime.module.parameters() if parameter.requires_grad]
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
    if runtime.checkpoint is not None:
        state = runtime.checkpoint.get("training_state")
        if not isinstance(state, dict) or state.get("step") != start_step - 1:
            raise ValueError("resume checkpoint has no matching training state")
        _restore_training_state(
            state, optimizer, sigma_cycle, view_generator, noise_generator, dropout_generator
        )
    provenance = {
        "training_seed": args.seed,
        "dataset_root": str(args.dataset_root.resolve()),
        "bridge_checkpoint": str(args.bridge_checkpoint.resolve()),
        "bridge_checkpoint_sha256": _sha256(args.bridge_checkpoint),
        "rre_checkpoint": None if args.rre_checkpoint is None else str(args.rre_checkpoint.resolve()),
        "rre_checkpoint_sha256": None if args.rre_checkpoint is None else _sha256(args.rre_checkpoint),
        "git_revision": _git_revision(),
    }
    if args.resume is None:
        manifest = {
            "config": config.to_dict(),
            "provenance": provenance,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "frozen_wan_parameters": sum(p.numel() for p in runtime.dit.model.parameters()),
            "frozen_vae_parameters": sum(p.numel() for p in runtime.vae.model.model.parameters()),
        }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    elapsed_prior = float(rows[-1].get("elapsed_seconds", 0.0)) if rows else 0.0
    started = time.perf_counter()
    for step in range(start_step, config.steps + 1):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses, scenes, view_groups, sigmas, target_lr_dropped = [], [], [], [], []
        for micro in range(config.gradient_accumulation):
            scene = config.train_scenes[
                ((step - 1) * config.gradient_accumulation + micro) % len(config.train_scenes)
            ]
            indices, hr, lr, camera = _load_group(
                args.dataset_root, scene, "train", config, view_generator, runtime.device
            )
            dropped = (
                config.target_lr_dropout > 0
                and float(torch.rand((), generator=dropout_generator)) < config.target_lr_dropout
            )
            if dropped:
                lr = lr.clone()
                lr[:, :, 0] = 0
            with torch.no_grad():
                clean = runtime.vae.encode_multiview(hr)
            prepared = runtime.module.prepare_multiview(
                lr,
                camera,
                tuple(clean.shape[2:]),
                (config.image_size, config.image_size),
            )
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
            loss = flow_matching_loss(prediction, target)
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
                    dropout_generator,
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


def _intervention(lr, camera, mode, generator, *, far_camera=None):
    """Return LR, fusion camera, geometry camera and source mask.

    The old ``shuffle_camera`` name is retained as a compatibility alias for
    the historical fusion-only intervention. New audits should use the
    explicit ``shuffle_fusion``, ``shuffle_geometry`` or ``shuffle_all`` modes.
    """
    if mode.startswith("no_self_"):
        mode = mode.removeprefix("no_self_")
    if mode in {"correct", "correct_repeat"}:
        changed, changed_camera, mask = intervene_lr(
            lr, camera, "correct", target=0, generator=generator
        )
        return changed, changed_camera, changed_camera, mask
    if mode == "target_drop":
        changed = lr.clone()
        changed[:, :, 0] = 0
        mask = torch.ones(lr.shape[0], lr.shape[2], dtype=torch.bool, device=lr.device)
        return changed, camera, camera, mask
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
        return lr.clone(), far_camera, camera, mask
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
        return lr.clone(), fusion_camera, geometry_camera, mask
    if mode == "shuffle_pair":
        changed, shuffled_camera, mask = shuffle_auxiliary_pairs(
            lr, camera, target=0, generator=generator
        )
        return changed, shuffled_camera, shuffled_camera, mask
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
        return changed, changed_camera, changed_camera, mask
    changed, changed_camera, mask = intervene_lr(
        lr, camera, mode, target=0, generator=generator
    )
    return changed, changed_camera, changed_camera, mask


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
                         allow_self_view_source: bool | None = None) -> dict:
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
                    intervention_generator = torch.Generator().manual_seed(
                        sample_seed * 10 + mode_index
                    )
                    changed_lr, fusion_camera, geometry_camera, source_mask = _intervention(
                        lr, camera, mode, intervention_generator, far_camera=far_camera
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
                    )
                    if getattr(args, "save_diagnostics", False):
                        latent, mode_diagnostics[mode] = sampled
                        mode_interventions[mode] = (
                            changed_lr, fusion_camera, geometry_camera, source_mask,
                            self_source_override,
                        )
                    else:
                        latent = sampled
                    decoded = runtime.vae.decode_multiview(latent)[:, :, :1]
                    item = frame_metrics(decoded, target, perceptual_metric=metric)[0]
                    rows.append(_metric_row(
                        item,
                        config=config,
                        training_seed=training_seed,
                        checkpoint=args.checkpoint,
                        step=step,
                        inference_seed=inference_seed,
                        group=group,
                        condition=mode,
                    ))
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
                        changed_lr, fusion_camera, geometry_camera, source_mask, self_source_override = intervention
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
                        )
                        diagnostic.update(_trace_delta(mode_diagnostics[reference_mode], mode_diagnostics[mode]))
                        diagnostic.update({
                            "inference_seed": inference_seed,
                            "sample_seed": sample_seed,
                            "reference_condition": reference_mode,
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
                        intervention_generator = torch.Generator().manual_seed(
                            inference_seed * 1000000 + scene_index * 10000 + group * 100 + mode_index
                        )
                        changed_lr, fusion_camera, geometry_camera, source_mask = _intervention(
                            lr, camera, mode, intervention_generator
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
                        )
                        decoded = runtime.vae.decode_multiview(latent)
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
    train.add_argument("--resume", type=Path)
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
