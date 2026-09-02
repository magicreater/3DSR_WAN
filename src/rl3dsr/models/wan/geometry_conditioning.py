"""Camera geometry encoders for the frozen Wan Stage 2 adapter.

The module emits token-aligned residuals without changing Wan's VAE or DiT.
3D views and 4D frames share the implementation, while ``sequence_kind`` is
kept explicit in the camera representation and manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CameraBatch:
    """Canonical camera tensors for one 3D view set or 4D trajectory."""

    K: Tensor
    T_world_from_camera: Tensor
    image_size: tuple[int, int]
    sequence_kind: str
    reference_index: int = 0

    def validate(self, batch: int | None = None) -> None:
        if self.K.ndim != 4 or self.K.shape[-2:] != (3, 3):
            raise ValueError("K must have shape [B,S,3,3]")
        if self.T_world_from_camera.ndim != 4 or self.T_world_from_camera.shape[-2:] != (4, 4):
            raise ValueError("T_world_from_camera must have shape [B,S,4,4]")
        if self.K.shape[:2] != self.T_world_from_camera.shape[:2]:
            raise ValueError("K and T_world_from_camera must share [B,S]")
        if batch is not None and self.K.shape[0] != batch:
            raise ValueError(f"camera batch must have B={batch}")
        if not torch.is_floating_point(self.K) or not torch.is_floating_point(self.T_world_from_camera):
            raise ValueError("camera tensors must be floating point")
        if not torch.isfinite(self.K).all() or not torch.isfinite(self.T_world_from_camera).all():
            raise ValueError("camera tensors must be finite")
        height, width = self.image_size
        if height < 1 or width < 1:
            raise ValueError("image_size must be positive")
        if self.sequence_kind not in {"multiview", "temporal"}:
            raise ValueError("sequence_kind must be multiview or temporal")
        if not 0 <= self.reference_index < self.K.shape[1]:
            raise ValueError("reference_index is outside the sequence")
        if (self.K[..., 0, 0] <= 0).any() or (self.K[..., 1, 1] <= 0).any():
            raise ValueError("camera focal lengths must be positive")
        if not torch.allclose(
            self.T_world_from_camera[..., 3, :],
            torch.tensor([0, 0, 0, 1], device=self.T_world_from_camera.device, dtype=self.T_world_from_camera.dtype),
            atol=1e-3,
            rtol=0,
        ):
            raise ValueError("camera transforms must have homogeneous last row [0,0,0,1]")


def _normalize(value: Tensor, eps: float = 1e-6) -> Tensor:
    return value / value.square().sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()


def _ray_features(camera: CameraBatch, grid: tuple[int, int], representation: str) -> Tensor:
    camera.validate()
    batch, sequence = camera.K.shape[:2]
    grid_h, grid_w = grid
    if grid_h < 1 or grid_w < 1:
        raise ValueError("latent token grid must be positive")
    height, width = camera.image_size
    device = camera.K.device
    dtype = camera.K.dtype
    u = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) * (width / grid_w)
    v = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) * (height / grid_h)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    pixels = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1).reshape(1, 1, -1, 3)
    inverse_k = torch.linalg.inv(camera.K)
    rays_cam = _normalize(torch.einsum("bsij,bspj->bspi", inverse_k, pixels.expand(batch, sequence, -1, -1)))

    reference = camera.T_world_from_camera[:, camera.reference_index]
    relative = torch.linalg.inv(reference)[:, None] @ camera.T_world_from_camera
    relative_rotation = relative[..., :3, :3]
    origins = relative[..., :3, 3]
    rays_ref = _normalize(torch.einsum("bsij,bspj->bspi", relative_rotation, rays_cam))
    right_ref = torch.einsum("bsij,j->bsi", relative_rotation, torch.tensor([1, 0, 0], device=device, dtype=dtype))
    down_ref = torch.einsum("bsij,j->bsi", relative_rotation, torch.tensor([0, 1, 0], device=device, dtype=dtype))
    x_axis = torch.cross(down_ref[:, :, None, :].expand_as(rays_ref), rays_ref, dim=-1)
    fallback = torch.cross(right_ref[:, :, None, :].expand_as(rays_ref), rays_ref, dim=-1)
    x_axis = torch.where(x_axis.square().sum(-1, keepdim=True) < 1e-8, fallback, x_axis)
    x_axis = _normalize(x_axis)
    y_axis = _normalize(torch.cross(rays_ref, x_axis, dim=-1))
    ray_frame = torch.stack((x_axis, y_axis, rays_ref), dim=-1).reshape(batch, sequence, -1, 9)

    baseline = origins[:, :, 0].new_ones(batch)
    if sequence > 1:
        distances = origins.norm(dim=-1)
        mask = torch.ones(sequence, device=device, dtype=torch.bool)
        mask[camera.reference_index] = False
        baseline = distances[:, mask].mean(dim=1).clamp_min(1e-3)
    origin_norm = origins / baseline[:, None, None]
    origin_tokens = origin_norm[:, :, None, :].expand(-1, -1, grid_h * grid_w, -1)
    uv = torch.stack((uu / max(width, 1), vv / max(height, 1)), dim=-1).reshape(1, 1, -1, 2)
    uv = uv.expand(batch, sequence, -1, -1)
    intrinsics = torch.stack(
        (
            camera.K[..., 0, 0] / width,
            camera.K[..., 1, 1] / height,
            camera.K[..., 0, 2] / width,
            camera.K[..., 1, 2] / height,
        ),
        dim=-1,
    )[:, :, None, :].expand(-1, -1, grid_h * grid_w, -1)
    kind = torch.zeros(batch, sequence, grid_h * grid_w, 2, device=device, dtype=dtype)
    kind[..., 0 if camera.sequence_kind == "multiview" else 1] = 1

    if representation == "rre":
        return torch.cat((ray_frame, origin_tokens, rays_ref, uv, intrinsics, kind), dim=-1)
    if representation == "plucker":
        moment = torch.cross(origin_norm[:, :, None, :].expand_as(rays_ref), rays_ref, dim=-1)
        return torch.cat((rays_ref, moment, origin_tokens, uv, intrinsics, kind), dim=-1)
    raise ValueError("representation must be rre or plucker")


class GeometryConditioner(nn.Module):
    """Encode camera geometry and emit zero-initialized Wan block residuals."""

    def __init__(
        self,
        *,
        representation: str = "rre",
        feature_dim: int = 1536,
        hidden_dim: int = 192,
        attention_heads: int = 4,
        blocks: tuple[int, ...] = (0, 1, 2, 3),
        timestep_conditioning: bool = True,
    ) -> None:
        super().__init__()
        if representation not in {"rre", "plucker"}:
            raise ValueError("representation must be rre or plucker")
        if hidden_dim < 1 or feature_dim < 1 or hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if not blocks or len(set(blocks)) != len(blocks) or any(i < 0 for i in blocks):
            raise ValueError("blocks must contain unique non-negative indices")
        self.representation = representation
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.blocks = tuple(blocks)
        self.timestep_conditioning = bool(timestep_conditioning)
        input_dim = 23 if representation == "rre" else 17
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.attention = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.time = nn.Linear(256, hidden_dim) if timestep_conditioning else None
        self.output = nn.ModuleDict({str(i): nn.Linear(hidden_dim, feature_dim, dtype=torch.float32) for i in self.blocks})
        self.last_diagnostics: dict[int, dict[str, Tensor]] = {}
        for layer in self.output.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def input_dim(self) -> int:
        return 23 if self.representation == "rre" else 17

    def encode(self, camera: CameraBatch, latent_shape: tuple[int, int, int]) -> Tensor:
        sequence, latent_h, latent_w = latent_shape
        if camera.K.shape[1] != sequence:
            raise ValueError("camera sequence length must match latent_shape[0]")
        raw = _ray_features(camera, (latent_h // 2, latent_w // 2), self.representation)
        batch, views, patches, _ = raw.shape
        hidden = self.input(raw.float())
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch * patches, views, self.hidden_dim)
        attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        hidden = self.norm(hidden + attended)
        hidden = hidden + self.mlp(hidden)
        return hidden.reshape(batch, patches, views, self.hidden_dim).permute(0, 2, 1, 3).reshape(batch, sequence * patches, self.hidden_dim)

    def residuals(
        self,
        camera: CameraBatch,
        latent_shape: tuple[int, int, int],
        timesteps: Tensor,
        *,
        enabled: bool = True,
    ) -> dict[int, Tensor]:
        if not enabled:
            self.last_diagnostics = {}
            return {}
        batch = camera.K.shape[0]
        if timesteps.shape != (batch,) or not torch.isfinite(timesteps).all():
            raise ValueError("timesteps must have shape [B] and be finite")
        hidden = self.encode(camera, latent_shape)
        if self.time is not None:
            positions = timesteps.to(device=hidden.device, dtype=torch.float64)
            phases = torch.outer(
                positions,
                torch.pow(10000, -torch.arange(128, device=hidden.device, dtype=torch.float64) / 128),
            )
            embedding = F.silu(torch.cat((phases.cos(), phases.sin()), dim=1).float())
            hidden = hidden * (1 + torch.tanh(self.time(embedding))[:, None, :])
        result: dict[int, Tensor] = {}
        diagnostics: dict[int, dict[str, Tensor]] = {}
        for key, projection in self.output.items():
            residual = projection(hidden)
            result[int(key)] = residual
            diagnostics[int(key)] = {
                "residual_rms": residual.detach().float().square().mean().sqrt(),
                "hidden_rms": hidden.detach().float().square().mean().sqrt(),
            }
        self.last_diagnostics = diagnostics
        return result

    def forward(self, camera: CameraBatch, latent_shape: tuple[int, int, int], timesteps: Tensor) -> Mapping[int, Tensor]:
        return self.residuals(camera, latent_shape, timesteps)


def load_geometry_checkpoint(
    path: str | Path,
    geometry: GeometryConditioner,
    *,
    expected_parent_checkpoint: str | None = None,
) -> dict[str, object]:
    """Strictly load a Stage 2 geometry checkpoint and return its metadata."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or "geometry" not in payload or "config" not in payload:
        raise RuntimeError("unsupported Stage 2 geometry checkpoint")
    config = payload["config"]
    if config.get("representation") != geometry.representation:
        raise RuntimeError("geometry representation does not match checkpoint")
    if tuple(config.get("blocks", ())) != geometry.blocks:
        raise RuntimeError("geometry block configuration does not match checkpoint")
    if expected_parent_checkpoint is not None and payload.get("parent_checkpoint") != expected_parent_checkpoint:
        raise RuntimeError("geometry parent checkpoint mismatch")
    geometry.load_state_dict(payload["geometry"], strict=True)
    return {"config": config, "step": payload.get("step"), "parent_checkpoint": payload.get("parent_checkpoint")}
