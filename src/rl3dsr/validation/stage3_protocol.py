"""Stage 3 configuration, LR interventions and validation-only selection.

No function launches training or reads datasets. Camera transforms are OpenCV
camera-to-world; source masks use True for available LR evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch
from torch import Tensor

from rl3dsr.models.wan.geometry_conditioning import CameraBatch


ARM_MODES = {"A0": "off", "A1": "same_view", "A2": "visual", "A3": "epipolar"}


@dataclass(frozen=True)
class Stage3Config:
    arm: str = "A0"
    train_scenes: tuple[str, ...] = ("chair", "lego", "drums", "hotdog", "mic")
    validation_scenes: tuple[str, ...] = ("ficus",)
    test_scenes: tuple[str, ...] = ("materials", "ship")
    image_size: int = 256
    scale: int = 4
    views: int = 4
    steps: int = 4000
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    checkpoint_every: int = 500
    sampling_steps: int = 50
    sampling_shift: float = 5.0
    validation_groups_per_scene: int = 4
    final_groups_per_scene: int = 8
    training_seeds: tuple[int, ...] = (42, 43)
    validation_inference_seed: int = 3301
    final_inference_seeds: tuple[int, ...] = (3302, 3303, 3304)
    nearest_views: int = 12
    fusion_dim: int = 192
    fusion_heads: int = 3
    query_chunk_size: int = 128
    epipolar_tau: float = 1.0
    epipolar_attention: str = "global_bias"
    epipolar_band: float = 1.5
    target_lr_dropout: float = 0.0

    def __post_init__(self):
        if self.arm not in ARM_MODES:
            raise ValueError("arm must be A0, A1, A2 or A3")
        groups = (self.train_scenes, self.validation_scenes, self.test_scenes)
        for group in groups:
            if not isinstance(group, tuple) or not group or any(not isinstance(s, str) or not s for s in group):
                raise ValueError("scene lists must be nonempty tuples of names")
            if len(set(group)) != len(group):
                raise ValueError("scene names must be unique")
        if sum(map(len, groups)) != len(set().union(*groups)):
            raise ValueError("train, validation and test scenes must be disjoint")
        for name in ("image_size", "scale", "views", "steps", "gradient_accumulation", "checkpoint_every",
                     "sampling_steps", "validation_groups_per_scene", "final_groups_per_scene",
                     "nearest_views", "fusion_dim", "fusion_heads", "query_chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "weight_decay", "gradient_clip", "sampling_shift", "epipolar_tau", "epipolar_band"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.epipolar_attention not in {"global_bias", "local_band"}:
            raise ValueError("epipolar_attention must be global_bias or local_band")
        value = self.target_lr_dropout
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("target_lr_dropout must be finite and in [0, 1)")
        for group in (self.training_seeds, self.final_inference_seeds):
            if not isinstance(group, tuple) or not group or any(type(v) is not int or v < 0 for v in group) or len(set(group)) != len(group):
                raise ValueError("seed lists must contain distinct nonnegative integers")
        if type(self.validation_inference_seed) is not int or self.validation_inference_seed < 0:
            raise ValueError("validation_inference_seed must be a nonnegative integer")
        if self.validation_inference_seed in self.final_inference_seeds:
            raise ValueError("development and final inference seeds must be disjoint")
        if self.image_size % (8 * 2) or self.image_size % self.scale:
            raise ValueError("image_size must align with Wan VAE/patch grid and SR scale")
        if self.fusion_dim % self.fusion_heads:
            raise ValueError("fusion_dim must be divisible by fusion_heads")
        if self.nearest_views < self.views - 1:
            raise ValueError("nearest_views must cover all auxiliary views")

    @property
    def fusion_mode(self) -> str:
        return ARM_MODES[self.arm]

    @property
    def validation_scene_names(self) -> tuple[str, ...]:
        return self.train_scenes + self.validation_scenes

    def to_dict(self) -> dict:
        return asdict(self)


def load_stage3_config(path: str | Path) -> Stage3Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object")
    unknown = set(payload) - {f.name for f in fields(Stage3Config)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    for name in ("train_scenes", "validation_scenes", "test_scenes", "training_seeds", "final_inference_seeds"):
        if name in payload:
            if not isinstance(payload[name], list):
                raise ValueError(f"{name} must be a JSON array")
            payload[name] = tuple(payload[name])
    return Stage3Config(**payload)


def sample_view_indices(camera: CameraBatch, anchor: int, generator: torch.Generator,
                        view_count: int = 4, nearest: int = 12) -> list[int]:
    """Anchor then random auxiliaries among nearest optical-axis directions.

    CPU deterministic ordering breaks equal-angle ties by view index. This is
    a view-selection heuristic, not an overlap or visibility estimate.
    """
    camera.validate(batch=1)
    n = camera.K.shape[1]
    if camera.sequence_kind != "multiview" or not 0 <= anchor < n:
        raise ValueError("requires multiview cameras and an in-range anchor")
    if view_count < 1 or n < view_count or nearest < view_count - 1:
        raise ValueError("not enough candidate views")
    candidates = _nearest_candidates(camera, anchor)
    candidates = candidates[:nearest]
    chosen = torch.randperm(len(candidates), generator=generator).tolist()[:view_count - 1]
    return [anchor] + [candidates[i] for i in chosen]


def _nearest_candidates(camera: CameraBatch, anchor: int) -> list[int]:
    camera.validate(batch=1)
    n = camera.K.shape[1]
    if camera.sequence_kind != "multiview" or not 0 <= anchor < n:
        raise ValueError("requires multiview cameras and an in-range anchor")
    axes = camera.T_world_from_camera[0, :, :3, 2].detach().float().cpu()
    norms = axes.norm(dim=-1)
    if (norms < 1e-8).any():
        raise ValueError("invalid camera forward direction")
    axes = axes / norms[:, None]
    similarity = axes @ axes[anchor]
    return sorted((i for i in range(n) if i != anchor), key=lambda i: (-float(similarity[i]), i))


def nearest_view_indices(camera: CameraBatch, anchor: int, view_count: int = 4) -> list[int]:
    """Return an anchor and the closest deterministic optical-axis neighbours."""
    if type(view_count) is not int or view_count < 1 or camera.K.shape[1] < view_count:
        raise ValueError("not enough candidate views")
    return [anchor] + _nearest_candidates(camera, anchor)[:view_count - 1]


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Select deterministic inclusive endpoints without duplicate indices."""
    if type(total) is not int or type(count) is not int or total < 1 or not 1 <= count <= total:
        raise ValueError("count must be between one and total")
    if count == 1:
        return [0]
    return [index * (total - 1) // (count - 1) for index in range(count)]


def intervene_lr(lr: Tensor, camera: CameraBatch, mode: str, *, target: int = 0,
                 generator: torch.Generator | None = None, source: int | None = None,
                 box: tuple[int, int, int, int] | None = None) -> tuple[Tensor, CameraBatch, Tensor]:
    """Copy inputs and intervene on auxiliaries; box is (y0,x0,y1,x1) in LR pixels.

    Target pixels and pose always remain intact. Pass the returned source mask
    to LR fusion; removal alone is not an all-model view-attention mask.
    """
    if lr.ndim != 5 or lr.shape[1] != 3:
        raise ValueError("LR must be [B,3,V,H,W]")
    camera.validate(batch=lr.shape[0])
    if camera.sequence_kind != "multiview" or camera.K.shape[1] != lr.shape[2]:
        raise ValueError("LR views require aligned multiview cameras")
    if not 0 <= target < lr.shape[2]:
        raise ValueError("target is out of range")
    aux = [i for i in range(lr.shape[2]) if i != target]
    result = lr.clone()
    k, t = camera.K.clone(), camera.T_world_from_camera.clone()
    mask = torch.ones(lr.shape[0], lr.shape[2], dtype=torch.bool, device=lr.device)
    if mode == "correct":
        pass
    elif mode == "remove":
        result[:, :, aux] = 0
        mask[:, aux] = False
    elif mode == "duplicate":
        result[:, :, aux] = lr[:, :, target:target + 1]
        k[:, aux] = camera.K[:, target:target + 1]
        t[:, aux] = camera.T_world_from_camera[:, target:target + 1]
    elif mode == "shuffle_camera":
        if len(aux) < 2:
            raise ValueError("shuffle_camera requires at least two auxiliary views")
        # Nonzero cyclic shift guarantees a changed correspondence for each aux.
        shift = int(torch.randint(1, len(aux), (), generator=generator))
        shuffled = aux[shift:] + aux[:shift]
        k[:, aux], t[:, aux] = camera.K[:, shuffled], camera.T_world_from_camera[:, shuffled]
    elif mode == "local_patch":
        if source not in aux or box is None or len(box) != 4:
            raise ValueError("local_patch needs an auxiliary source and LR pixel box")
        y0, x0, y1, x1 = box
        if any(type(v) is not int for v in box) or not (0 <= y0 < y1 <= lr.shape[-2] and 0 <= x0 < x1 <= lr.shape[-1]):
            raise ValueError("local_patch box is invalid")
        result[:, :, source, y0:y1, x0:x1] = 0
    else:
        raise ValueError(f"unknown intervention {mode}")
    return result, replace(camera, K=k, T_world_from_camera=t), mask


def shuffle_auxiliary_pairs(
    lr: Tensor,
    camera: CameraBatch,
    *,
    target: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, CameraBatch, Tensor]:
    """Cyclically permute auxiliary LR/camera pairs while preserving target 0.

    The returned tensors keep the target view at ``target`` and apply the same
    non-zero auxiliary permutation to both the image evidence and its camera.
    This is the explicit pair-alignment intervention used by Stage 3.1.
    """
    if lr.ndim != 5 or lr.shape[1] != 3:
        raise ValueError("LR must be [B,3,V,H,W]")
    camera.validate(batch=lr.shape[0])
    if camera.sequence_kind != "multiview" or camera.K.shape[1] != lr.shape[2]:
        raise ValueError("LR views require aligned multiview cameras")
    if not 0 <= target < lr.shape[2]:
        raise ValueError("target is out of range")
    aux = [index for index in range(lr.shape[2]) if index != target]
    if len(aux) < 2:
        raise ValueError("pair shuffling requires at least two auxiliary views")
    shift = int(torch.randint(1, len(aux), (), generator=generator))
    shuffled = aux[shift:] + aux[:shift]
    result = lr.clone()
    result[:, :, aux] = lr[:, :, shuffled]
    k = camera.K.clone()
    t = camera.T_world_from_camera.clone()
    k[:, aux] = camera.K[:, shuffled]
    t[:, aux] = camera.T_world_from_camera[:, shuffled]
    mask = torch.ones(lr.shape[0], lr.shape[2], dtype=torch.bool, device=lr.device)
    return result, replace(camera, K=k, T_world_from_camera=t), mask


def select_candidate(rows: Iterable[dict], expected_scenes: Iterable[str]) -> dict:
    """Scene-equal validation means; within 0.01 PSNR use SSIM/LPIPS/earlier step."""
    expected = set(expected_scenes)
    if not expected:
        raise ValueError("expected scene coverage cannot be empty")
    grouped: dict[tuple[str, int], dict[str, list[dict]]] = {}
    checkpoint_steps: dict[str, int] = {}
    run_ids: set[tuple] = set()
    for row in rows:
        if row.get("split") != "validation":
            raise ValueError("selection accepts validation rows only")
        if row.get("scene") not in expected:
            raise ValueError("unexpected scene in validation coverage")
        for metric in ("psnr", "ssim", "lpips"):
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"nonfinite {metric}")
        if type(row["step"]) is not int or row["step"] < 0:
            raise ValueError("checkpoint step must be a nonnegative integer")
        key = (str(row["checkpoint"]), row["step"])
        previous_step = checkpoint_steps.setdefault(key[0], key[1])
        if previous_step != key[1]:
            raise ValueError("one checkpoint path cannot represent multiple steps")
        run_ids.add((row.get("arm"), row.get("train_seed", row.get("training_seed"))))
        if len(run_ids) > 1:
            raise ValueError("selection cannot mix arms or training seeds")
        grouped.setdefault(key, {}).setdefault(row["scene"], []).append(row)
    candidates = []
    for (checkpoint, step), scene_rows in grouped.items():
        if set(scene_rows) != expected:
            raise ValueError(f"incomplete validation scene coverage for {checkpoint}")
        candidate = {"checkpoint": checkpoint, "step": step, "scenes": sorted(expected)}
        for metric in ("psnr", "ssim", "lpips"):
            candidate[metric] = mean(mean(float(r[metric]) for r in values) for values in scene_rows.values())
        candidates.append(candidate)
    if not candidates:
        raise ValueError("no validation candidates")
    best_psnr = max(c["psnr"] for c in candidates)
    eligible = [c for c in candidates if best_psnr - c["psnr"] <= 0.01 + 1e-12]
    return min(eligible, key=lambda c: (-c["ssim"], c["lpips"], c["step"], c["checkpoint"]))


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def freeze_candidate(path: str | Path, candidate: dict, config: Stage3Config) -> dict:
    if set(candidate.get("scenes", ())) != set(config.validation_scene_names):
        raise ValueError("candidate must cover every configured validation scene")
    if type(candidate.get("step")) is not int or not 0 < candidate["step"] <= config.steps:
        raise ValueError("candidate step must be within the training protocol")
    for metric in ("psnr", "ssim", "lpips"):
        if not math.isfinite(float(candidate[metric])):
            raise ValueError(f"nonfinite candidate {metric}")
    checkpoint = Path(candidate["checkpoint"]).resolve(strict=True)
    payload = {"version": 1, "candidate": dict(candidate, checkpoint=str(checkpoint)),
               "checkpoint_sha256": _file_digest(checkpoint), "protocol": config.to_dict(),
               "protocol_sha256": _digest(config.to_dict())}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    return payload


def load_frozen_candidate(path: str | Path, config: Stage3Config) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("protocol_sha256") != _digest(config.to_dict()) or _digest(payload["protocol"]) != payload["protocol_sha256"]:
        raise ValueError("frozen protocol mismatch")
    if _file_digest(Path(payload["candidate"]["checkpoint"])) != payload["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash mismatch")
    return payload


def claim_final_evaluation(manifest_path: str | Path, config: Stage3Config,
                           output_dir: str | Path) -> dict:
    """Claim once before reading test data; failure also consumes this claim.

    A rerun requires explicit human review of the failed attempt. Changing the
    output directory cannot bypass the marker bound to this manifest.
    """
    payload = load_frozen_candidate(manifest_path, config)
    manifest_path = Path(manifest_path)
    claim = manifest_path.with_name(manifest_path.name + ".final-test-claimed")
    with claim.open("x", encoding="utf-8") as stream:
        json.dump({"output_dir": str(Path(output_dir).resolve()),
                   "manifest_sha256": _file_digest(manifest_path),
                   "protocol_sha256": payload["protocol_sha256"]}, stream, indent=2)
    return payload
