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
                          conditioning_size: tuple[int, int], *, source_mask=None) -> Tensor:
        """Encode independent LR views and fuse once, returning [B,V*P,D].

        Retain this tensor for the full inference trajectory. During training
        prepare again after each parameter update; never cache a learned graph.
        """
        if camera.sequence_kind != "multiview":
            raise ValueError("Stage 3 preparation requires multiview; use video_features for temporal input")
        camera.validate(lr.shape[0])
        if camera.K.shape[1] != lr.shape[2]:
            raise ValueError("camera view count does not match LR")
        features = self.conditioner.multiview_features(
            lr, conditioning_size=conditioning_size, latent_shape=latent_shape)
        if self.fusion is None:
            return features
        views, height, width = latent_shape
        shaped = features.reshape(features.shape[0], views, -1, features.shape[-1])
        fused = self.fusion(shaped, camera, (height // 2, width // 2), source_mask=source_mask)
        return fused.reshape_as(features)

    def predict(self, dit, sample, timestep, context, prepared_features, camera, latent_shape):
        """Consume prepared features without invoking LR encoding or fusion."""
        return conditioned_prediction(
            dit, self.conditioner, sample, timestep, context, prepared_features,
            geometry_adapter=self.geometry, camera=camera, latent_shape=latent_shape)


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


def load_stage3_checkpoint(path, module: Stage3Conditioning, *, expected_config: dict | None = None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "rl3dsr-stage3" or payload.get("format_version") != 1:
        raise ValueError("unsupported Stage 3 checkpoint; initialize old bridges with load_adapter_checkpoint")
    if expected_config is not None:
        saved_config = dict(payload.get("config") or {})
        expected = json.loads(json.dumps(expected_config))
        # Stage 3.1 adds a default-off training knob; old Stage 3 bundles remain valid.
        saved_config.setdefault("target_lr_dropout", 0.0)
        expected.setdefault("target_lr_dropout", 0.0)
        if saved_config != expected:
            raise ValueError("Stage 3 checkpoint config mismatch")
    expected_arch = {"blocks": list(module.conditioner.bridge_blocks),
                     "time_conditioning": module.conditioner.bridge_time_conditioning}
    if payload.get("bridge_architecture") != expected_arch:
        raise ValueError("Stage 3 bridge architecture mismatch")
    # Validate all modules before mutating any parameter.
    states = payload.get("adapters", {})
    for name, item in _states(module).items():
        saved = states.get(name)
        if (item is None) != (saved is None):
            raise ValueError(f"Stage 3 {name} presence mismatch")
        if item is not None:
            current = item.state_dict()
            if current.keys() != saved.keys() or any(current[k].shape != saved[k].shape for k in current):
                raise ValueError(f"Stage 3 {name} architecture mismatch")
            if any(not torch.isfinite(v).all() for v in saved.values()):
                raise ValueError(f"Stage 3 {name} contains nonfinite parameters")
    for name, item in _states(module).items():
        if item is not None:
            item.load_state_dict(states[name], strict=True)
    return payload
