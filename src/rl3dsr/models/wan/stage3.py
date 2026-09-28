"""Stage 3 LR fusion preparation and small adapter-only checkpoints.

Preparation occurs outside the diffusion loop. The original temporal/video
entry points and Stage 1/2 checkpoint formats are deliberately unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .geometry_conditioning import CameraBatch
from .lq_conditioning import FrozenLQConditioner, conditioned_prediction
from .wan_lora import LORA_CONFIG, load_wan_lora_state, validate_wan_lora_state, wan_lora_state


@dataclass(frozen=True)
class DynamicPrepared:
    features: Tensor  # [B,V,P,D], frozen LR evidence before fusion
    camera: CameraBatch
    patch_grid: tuple[int, int]
    source_mask: Tensor | None
    allow_self_view_source: bool | None

    @property
    def shape(self):
        batch, views, patches, dim = self.features.shape
        return (batch, views * patches, dim)


class Stage3Conditioning(nn.Module):
    def __init__(self, conditioner: FrozenLQConditioner, geometry=None, fusion=None,
                 *, shared_multiview_rope: bool = False):
        super().__init__()
        self.conditioner = conditioner
        self.geometry = geometry
        self.fusion = fusion
        self.shared_multiview_rope = shared_multiview_rope

    def prepare_multiview(self, lr: Tensor, camera: CameraBatch,
                          latent_shape: tuple[int, int, int],
                          conditioning_size: tuple[int, int], *, source_mask=None,
                          allow_self_view_source=None, pairing_camera=None,
                          pairing_minimum_coverage=0.05,
                          pairing_lr: Tensor | None = None) -> Tensor | tuple[Tensor, dict]:
        """Encode independent LR views and fuse once, returning [B,V*P,D].

        Retain this tensor for the full inference trajectory. During training
        prepare again after each parameter update; never cache a learned graph.
        Supplying ``pairing_camera`` opts into a differentiable A5 Q/K state
        and returns ``(features, pairing_state)``. ``pairing_lr`` optionally
        supplies the original undropped LR solely for that state while ``lr``
        remains the flow-conditioning input.
        """
        if camera.sequence_kind != "multiview":
            raise ValueError("Stage 3 preparation requires multiview; use video_features for temporal input")
        camera.validate(lr.shape[0])
        if camera.K.shape[1] != lr.shape[2]:
            raise ValueError("camera view count does not match LR")
        features = self.conditioner.multiview_features(
            lr, conditioning_size=conditioning_size, latent_shape=latent_shape)
        if self.fusion is None:
            if pairing_camera is not None:
                raise ValueError("pairing supervision requires LR fusion")
            return features
        views, height, width = latent_shape
        shaped = features.reshape(features.shape[0], views, -1, features.shape[-1])
        if getattr(self.fusion, "dynamic", False):
            if pairing_camera is not None or pairing_lr is not None:
                raise ValueError("dynamic fusion does not use A5 pairing supervision")
            return DynamicPrepared(shaped, camera, (height // 2, width // 2),
                                   source_mask, allow_self_view_source)
        fused = self.fusion(
            shaped,
            camera,
            (height // 2, width // 2),
            source_mask=source_mask,
            allow_self_view_source=allow_self_view_source,
        )
        prepared = fused.reshape_as(features)
        if pairing_camera is None:
            if pairing_lr is not None:
                raise ValueError("pairing_lr requires pairing_camera")
            return prepared
        pairing_shaped = shaped
        if pairing_lr is not None:
            pairing_features = self.conditioner.multiview_features(
                pairing_lr,
                conditioning_size=conditioning_size,
                latent_shape=latent_shape,
            )
            pairing_shaped = pairing_features.reshape(
                pairing_features.shape[0], views, -1, pairing_features.shape[-1]
            )
        pairing = self.fusion.build_pairing_state(
            pairing_shaped,
            camera,
            pairing_camera,
            (height // 2, width // 2),
            minimum_coverage=pairing_minimum_coverage,
        )
        return prepared, pairing

    def predict(
        self,
        dit,
        sample,
        timestep,
        context,
        prepared_features,
        camera,
        latent_shape,
        *,
        geometry_camera=None,
    ):
        """Consume prepared features and use an explicit camera for geometry.

        ``camera`` remains the backwards-compatible positional argument. A
        separate ``geometry_camera`` makes intervention scope explicit: the
        camera used by LR fusion need not be the one seen by the Wan geometry
        adapter. When omitted, both paths intentionally use ``camera``.
        """
        geometry_camera = camera if geometry_camera is None else geometry_camera
        if isinstance(prepared_features, DynamicPrepared):
            state = prepared_features
            if tuple(sample.shape[2:]) != tuple(latent_shape) or sample.shape[2] != state.features.shape[1]:
                raise ValueError("dynamic fusion latent views do not match prepared LR views")
            batch, channels, views, height, width = sample.shape
            if channels != 16 or (height // 2, width // 2) != state.patch_grid:
                raise ValueError("dynamic fusion requires Wan latent channels and patch grid")
            pooled = F.avg_pool2d(
                sample.permute(0, 2, 1, 3, 4).reshape(batch * views, 16, height, width).float(),
                kernel_size=2,
            ).permute(0, 2, 3, 1).reshape(batch, views, -1, 16)
            prepared_features = self.fusion(
                state.features, state.camera, state.patch_grid,
                source_mask=state.source_mask,
                allow_self_view_source=state.allow_self_view_source,
                latent_query=pooled, timestep=timestep,
            ).reshape(batch, -1, state.features.shape[-1])
        return conditioned_prediction(
            dit, self.conditioner, sample, timestep, context, prepared_features,
            geometry_adapter=self.geometry, camera=geometry_camera, latent_shape=latent_shape,
            shared_view_rope=self.shared_multiview_rope and camera.sequence_kind == "multiview")


def _states(module: Stage3Conditioning):
    return {"bridge": module.conditioner.bridge, "geometry": module.geometry, "fusion": module.fusion}


def save_stage3_checkpoint(path, module: Stage3Conditioning, *, config: dict, step: int,
                           provenance: dict, training_state: dict | None = None,
                           wan_model: nn.Module | None = None):
    """Write immutable adapter/LoRA weights; never serialize Wan/VAE/projector."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    payload = {
        "format": "rl3dsr-stage3", "format_version": 2 if wan_model is not None else 1, "step": step,
        "config": json.loads(json.dumps(config)), "provenance": dict(provenance),
        "bridge_architecture": {"blocks": list(module.conditioner.bridge_blocks),
                                "time_conditioning": module.conditioner.bridge_time_conditioning},
        "adapters": {key: None if item is None else {
            name: value.detach().cpu().clone() for name, value in item.state_dict().items()
        } for key, item in _states(module).items()},
        "training_state": training_state,
    }
    if bool(config.get("wan_lora")) != (wan_model is not None):
        raise ValueError("Wan LoRA config and checkpoint model disagree")
    if wan_model is not None:
        state = wan_lora_state(wan_model)
        validate_wan_lora_state(state, wan_model)
        payload["wan_lora"] = {"architecture": dict(LORA_CONFIG), "state": state}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves historical candidates; incomplete writes are
    # unusable and must never be selected without successful load/hash checks.
    with path.open("xb") as stream:
        torch.save(payload, stream)


def validate_stage3_checkpoint_payload(
    payload: dict,
    module: Stage3Conditioning,
    *,
    expected_config: dict | None = None,
    wan_model: nn.Module | None = None,
) -> dict:
    """Validate a loaded checkpoint without mutating adapter parameters."""
    if not isinstance(payload, dict):
        raise ValueError("unsupported Stage 3 checkpoint; initialize old bridges with load_adapter_checkpoint")
    version = payload.get("format_version")
    if payload.get("format") != "rl3dsr-stage3" or version not in (1, 2):
        raise ValueError("unsupported Stage 3 checkpoint; initialize old bridges with load_adapter_checkpoint")
    if version == 2:
        lora = payload.get("wan_lora")
        if not isinstance(lora, dict) or lora.get("architecture") != LORA_CONFIG:
            raise ValueError("Wan LoRA checkpoint architecture mismatch")
        validate_wan_lora_state(lora.get("state"), wan_model)
    elif "wan_lora" in payload:
        raise ValueError("unexpected Wan LoRA state in legacy checkpoint")
    if expected_config is not None:
        if not isinstance(payload.get("config"), dict):
            raise ValueError("Stage 3 checkpoint config mismatch")
        saved_config = dict(payload["config"])
        expected = json.loads(json.dumps(expected_config))
        saved_config.setdefault("dataset_kind", "nerf_synthetic")
        saved_config.setdefault("image_factor", 1)
        expected.setdefault("dataset_kind", "nerf_synthetic")
        expected.setdefault("image_factor", 1)
        # Stage 3.1 adds default-off knobs; old Stage 3 bundles remain valid.
        saved_config.setdefault("target_lr_dropout", 0.0)
        saved_config.setdefault("epipolar_attention", "global_bias")
        saved_config.setdefault("epipolar_band", 1.5)
        saved_config.setdefault("allow_self_view_source", True)
        saved_config.setdefault("camera_rank_weight", 0.0)
        saved_config.setdefault("camera_rank_margin_ratio", 0.05)
        saved_config.setdefault("symmetric_correspondence_rank", False)
        saved_config.setdefault("symmetric_camera_fraction", 0.5)
        saved_config.setdefault("pairing_temperature", 0.07)
        saved_config.setdefault("pairing_minimum_coverage", 0.05)
        saved_config.setdefault("pairing_calibration_batches", 8)
        saved_config.setdefault("pairing_target_gradient_ratio", 0.25)
        saved_config.setdefault("pairing_weight_min", 0.01)
        saved_config.setdefault("pairing_weight_max", 10.0)
        saved_config.setdefault("dynamic_fusion", False)
        saved_config.setdefault("shared_multiview_rope", False)
        saved_config.setdefault("wan_lora", False)
        saved_config.setdefault("wan_lora_learning_rate", 1e-5)
        expected.setdefault("target_lr_dropout", 0.0)
        expected.setdefault("epipolar_attention", "global_bias")
        expected.setdefault("epipolar_band", 1.5)
        expected.setdefault("allow_self_view_source", True)
        expected.setdefault("camera_rank_weight", 0.0)
        expected.setdefault("camera_rank_margin_ratio", 0.05)
        expected.setdefault("symmetric_correspondence_rank", False)
        expected.setdefault("symmetric_camera_fraction", 0.5)
        expected.setdefault("pairing_temperature", 0.07)
        expected.setdefault("pairing_minimum_coverage", 0.05)
        expected.setdefault("pairing_calibration_batches", 8)
        expected.setdefault("pairing_target_gradient_ratio", 0.25)
        expected.setdefault("pairing_weight_min", 0.01)
        expected.setdefault("pairing_weight_max", 10.0)
        expected.setdefault("dynamic_fusion", False)
        expected.setdefault("shared_multiview_rope", False)
        expected.setdefault("wan_lora", False)
        expected.setdefault("wan_lora_learning_rate", 1e-5)
        if saved_config != expected:
            raise ValueError("Stage 3 checkpoint config mismatch")
        if bool(expected["wan_lora"]) != (version == 2):
            raise ValueError("Stage 3 checkpoint Wan LoRA presence mismatch")
    expected_arch = {"blocks": list(module.conditioner.bridge_blocks),
                     "time_conditioning": module.conditioner.bridge_time_conditioning}
    if payload.get("bridge_architecture") != expected_arch:
        raise ValueError("Stage 3 bridge architecture mismatch")
    # Validate all modules before mutating any parameter.
    states = payload.get("adapters")
    if not isinstance(states, dict) or set(states) != set(_states(module)):
        raise ValueError("Stage 3 checkpoint adapter set mismatch")
    for name, item in _states(module).items():
        saved = states.get(name)
        if (item is None) != (saved is None):
            raise ValueError(f"Stage 3 {name} presence mismatch")
        if item is not None:
            if not isinstance(saved, dict) or any(
                not isinstance(value, Tensor) for value in saved.values()
            ):
                raise ValueError(f"Stage 3 {name} architecture mismatch")
            current = item.state_dict()
            if current.keys() != saved.keys() or any(current[k].shape != saved[k].shape for k in current):
                raise ValueError(f"Stage 3 {name} architecture mismatch")
            if any(not torch.isfinite(v).all() for v in saved.values()):
                raise ValueError(f"Stage 3 {name} contains nonfinite parameters")
    return payload


def load_stage3_checkpoint(path, module: Stage3Conditioning, *, expected_config: dict | None = None,
                           wan_model: nn.Module | None = None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_stage3_checkpoint_payload(payload, module, expected_config=expected_config,
                                       wan_model=wan_model)
    if payload["format_version"] == 2 and wan_model is None:
        raise ValueError("Wan LoRA checkpoint requires an injected Wan model")
    states = payload["adapters"]
    for name, item in _states(module).items():
        if item is not None:
            item.load_state_dict(states[name], strict=True)
    if payload["format_version"] == 2:
        load_wan_lora_state(wan_model, payload["wan_lora"]["state"])
    return payload


def load_stage3_initialization_checkpoint(
    path, module: Stage3Conditioning, *, reset_fusion: bool = False,
    wan_model: nn.Module | None = None,
):
    """Load adapter parameters without restoring config or training progress.

    This is the model-only initialization path for a new experiment.  It
    deliberately permits compatible configuration changes, but requires the
    checkpoint to contain exactly the current Stage 3 adapter set.  The normal
    loader then validates every key, shape and value before mutating ``module``.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    states = payload.get("adapters")
    if not isinstance(states, dict) or set(states) != set(_states(module)):
        raise ValueError("Stage 3 checkpoint adapter set mismatch")
    for name, item in _states(module).items():
        saved = states[name]
        if item is None:
            if saved is not None:
                raise ValueError(f"Stage 3 {name} presence mismatch")
        elif not isinstance(saved, dict) or any(
            not isinstance(key, str) or not isinstance(value, Tensor)
            for key, value in saved.items()
        ):
            raise ValueError(f"Stage 3 {name} parameter state is malformed")
    if reset_fusion:
        expected_arch = {
            "blocks": list(module.conditioner.bridge_blocks),
            "time_conditioning": module.conditioner.bridge_time_conditioning,
        }
        if payload.get("format") != "rl3dsr-stage3" or payload.get("format_version") not in (1, 2):
            raise ValueError("unsupported Stage 3 checkpoint")
        if payload.get("bridge_architecture") != expected_arch:
            raise ValueError("Stage 3 bridge architecture mismatch")
        if payload["format_version"] == 2:
            if wan_model is None:
                raise ValueError("Wan LoRA checkpoint requires an injected Wan model")
            lora = payload.get("wan_lora")
            if not isinstance(lora, dict) or lora.get("architecture") != LORA_CONFIG:
                raise ValueError("Wan LoRA checkpoint architecture mismatch")
            validate_wan_lora_state(lora.get("state"), wan_model)
        selected = {"bridge": module.conditioner.bridge, "geometry": module.geometry}
        for name, item in selected.items():
            saved = states[name]
            if (item is None) != (saved is None):
                raise ValueError(f"Stage 3 {name} presence mismatch")
            if item is not None:
                current = item.state_dict()
                if current.keys() != saved.keys() or any(
                    current[key].shape != saved[key].shape for key in current
                ):
                    raise ValueError(f"Stage 3 {name} architecture mismatch")
                if any(not torch.isfinite(value).all() for value in saved.values()):
                    raise ValueError(f"Stage 3 {name} contains nonfinite parameters")
        for name, item in selected.items():
            if item is not None:
                item.load_state_dict(states[name], strict=True)
        if payload["format_version"] == 2:
            load_wan_lora_state(wan_model, payload["wan_lora"]["state"])
        return payload
    return load_stage3_checkpoint(path, module, expected_config=None, wan_model=wan_model)
