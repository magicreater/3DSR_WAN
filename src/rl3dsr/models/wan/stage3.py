"""Static-3D LR fusion preparation and small adapter-only checkpoints.

Preparation occurs outside the diffusion loop. The original temporal/video
entry points and Stage 1/2 checkpoint formats are deliberately unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn, Tensor

from .geometry_conditioning import CameraBatch
from .lq_conditioning import FrozenLQConditioner, conditioned_prediction


class Stage3Conditioning(nn.Module):
    def __init__(self, conditioner: FrozenLQConditioner, geometry=None, fusion=None):
        super().__init__()
        self.conditioner = conditioner
        self.geometry = geometry
        self.fusion = fusion

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
        return conditioned_prediction(
            dit, self.conditioner, sample, timestep, context, prepared_features,
            geometry_adapter=self.geometry, camera=geometry_camera, latent_shape=latent_shape)


def _states(module: Stage3Conditioning):
    return {"bridge": module.conditioner.bridge, "geometry": module.geometry, "fusion": module.fusion}


def save_stage3_checkpoint(path, module: Stage3Conditioning, *, config: dict, step: int,
                           provenance: dict, training_state: dict | None = None):
    """Write an immutable adapter bundle; never serialize Wan/VAE/projector."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    payload = {
        "format": "rl3dsr-stage3", "format_version": 1, "step": step,
        "config": json.loads(json.dumps(config)), "provenance": dict(provenance),
        "bridge_architecture": {"blocks": list(module.conditioner.bridge_blocks),
                                "time_conditioning": module.conditioner.bridge_time_conditioning},
        "adapters": {key: None if item is None else {
            name: value.detach().cpu().clone() for name, value in item.state_dict().items()
        } for key, item in _states(module).items()},
        "training_state": training_state,
    }
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
) -> dict:
    """Validate a loaded checkpoint without mutating adapter parameters."""
    if not isinstance(payload, dict):
        raise ValueError("unsupported Stage 3 checkpoint; initialize old bridges with load_adapter_checkpoint")
    if payload.get("format") != "rl3dsr-stage3" or payload.get("format_version") != 1:
        raise ValueError("unsupported Stage 3 checkpoint; initialize old bridges with load_adapter_checkpoint")
    if expected_config is not None:
        if not isinstance(payload.get("config"), dict):
            raise ValueError("Stage 3 checkpoint config mismatch")
        saved_config = dict(payload["config"])
        expected = json.loads(json.dumps(expected_config))
        # Stage 3.1 adds default-off knobs; old Stage 3 bundles remain valid.
        saved_config.setdefault("target_lr_dropout", 0.0)
        saved_config.setdefault("epipolar_attention", "global_bias")
        saved_config.setdefault("epipolar_band", 1.5)
        saved_config.setdefault("allow_self_view_source", True)
        saved_config.setdefault("camera_rank_weight", 0.0)
        saved_config.setdefault("camera_rank_margin_ratio", 0.05)
        saved_config.setdefault("pairing_temperature", 0.07)
        saved_config.setdefault("pairing_minimum_coverage", 0.05)
        saved_config.setdefault("pairing_calibration_batches", 8)
        saved_config.setdefault("pairing_target_gradient_ratio", 0.25)
        saved_config.setdefault("pairing_weight_min", 0.01)
        saved_config.setdefault("pairing_weight_max", 10.0)
        expected.setdefault("target_lr_dropout", 0.0)
        expected.setdefault("epipolar_attention", "global_bias")
        expected.setdefault("epipolar_band", 1.5)
        expected.setdefault("allow_self_view_source", True)
        expected.setdefault("camera_rank_weight", 0.0)
        expected.setdefault("camera_rank_margin_ratio", 0.05)
        expected.setdefault("pairing_temperature", 0.07)
        expected.setdefault("pairing_minimum_coverage", 0.05)
        expected.setdefault("pairing_calibration_batches", 8)
        expected.setdefault("pairing_target_gradient_ratio", 0.25)
        expected.setdefault("pairing_weight_min", 0.01)
        expected.setdefault("pairing_weight_max", 10.0)
        if saved_config != expected:
            raise ValueError("Stage 3 checkpoint config mismatch")
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


def load_stage3_checkpoint(path, module: Stage3Conditioning, *, expected_config: dict | None = None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_stage3_checkpoint_payload(payload, module, expected_config=expected_config)
    states = payload["adapters"]
    for name, item in _states(module).items():
        if item is not None:
            item.load_state_dict(states[name], strict=True)
    return payload


def load_stage3_initialization_checkpoint(
    path, module: Stage3Conditioning, *, reset_fusion: bool = False
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
        if payload.get("format") != "rl3dsr-stage3" or payload.get("format_version") != 1:
            raise ValueError("unsupported Stage 3 checkpoint")
        if payload.get("bridge_architecture") != expected_arch:
            raise ValueError("Stage 3 bridge architecture mismatch")
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
        return payload
    return load_stage3_checkpoint(path, module, expected_config=None)
