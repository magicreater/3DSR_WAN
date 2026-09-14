"""Camera geometry encoders for the frozen Wan Stage 2 adapter.

The module emits token-aligned residuals without changing Wan's VAE or DiT.
3D views and 4D frames share the implementation, while ``sequence_kind`` is
kept explicit in the camera representation and manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


OFFICIAL_UCPE_COMMIT = "d992f1807803ba99331e807e8f018ed552886afd"


@dataclass(frozen=True, slots=True)
class CameraBatch:
    """Canonical camera tensors for one 3D view set or 4D trajectory."""

    K: Tensor
    T_world_from_camera: Tensor
    image_size: tuple[int, int]
    sequence_kind: str
    reference_index: int = 0
    camera_model: str = "pinhole"
    xi: Tensor | None = None

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
        if self.K.device != self.T_world_from_camera.device or self.K.dtype != self.T_world_from_camera.dtype:
            raise ValueError("K and T_world_from_camera must share device and dtype")
        if not torch.isfinite(self.K).all() or not torch.isfinite(self.T_world_from_camera).all():
            raise ValueError("camera tensors must be finite")
        height, width = self.image_size
        if height < 1 or width < 1:
            raise ValueError("image_size must be positive")
        if self.sequence_kind not in {"multiview", "temporal"}:
            raise ValueError("sequence_kind must be multiview or temporal")
        if self.camera_model not in {"pinhole", "ucm"}:
            raise ValueError("camera_model must be pinhole or ucm")
        if self.camera_model == "pinhole":
            if self.xi is not None:
                raise ValueError("xi must be omitted for pinhole cameras")
        else:
            if self.xi is None:
                raise ValueError("xi is required for ucm cameras")
            if not isinstance(self.xi, Tensor):
                raise ValueError("xi must be a tensor")
            if self.xi.shape != self.K.shape[:2]:
                raise ValueError("xi must have shape [B,S]")
            if not torch.is_floating_point(self.xi):
                raise ValueError("xi must be floating point")
            if not torch.isfinite(self.xi).all():
                raise ValueError("xi must be finite")
            if (self.xi < 0).any():
                raise ValueError("xi must be non-negative")
            if self.xi.device != self.K.device:
                raise ValueError("xi must be on the same device as K")
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


@dataclass(frozen=True, slots=True)
class FullRREContext:
    """Per-token camera state consumed by every full-RRE attention branch."""

    world_to_ray: Tensor
    absmap: Tensor
    patch_grid: tuple[int, int]


def _invert_se3(transform: Tensor) -> Tensor:
    if transform.shape[-2:] != (4, 4):
        raise ValueError("SE(3) tensors must end in [4,4]")
    rotation = transform[..., :3, :3].transpose(-1, -2)
    result = torch.zeros_like(transform)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i", rotation, transform[..., :3, 3]
    )
    result[..., 3, 3] = 1
    return result


def _camera_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype


def build_camera_rays(
    camera: CameraBatch,
    grid: tuple[int, int],
    *,
    return_validity: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Return canonical OpenCV camera rays as ``[B,S,P,3]``.

    ``K`` is interpreted in pixel coordinates at patch centers. Pinhole rays
    use ``K^-1 [u,v,1]``. UCM rays use the closed-form inverse unified camera
    model with per-observation ``xi``; ``xi=0`` is the pinhole limit.
    Half-precision inputs are evaluated in float32 and returned in their
    storage dtype so this geometry path remains usable on CPU. When requested,
    the second return value is a ``[B,S,P]`` inverse-domain validity mask.
    """

    camera.validate()

    grid_h, grid_w = grid
    if grid_h < 1 or grid_w < 1:
        raise ValueError("latent token grid must be positive")
    height, width = camera.image_size
    device = camera.K.device
    dtype = _camera_compute_dtype(camera.K.dtype)
    u = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) * (width / grid_w)
    v = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) * (height / grid_h)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    pixels = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1).reshape(1, 1, -1, 3)
    plane = torch.einsum(
        "bsij,bspj->bspi",
        torch.linalg.inv(camera.K.to(dtype=dtype)),
        pixels.expand(camera.K.shape[0], camera.K.shape[1], -1, -1),
    )
    valid = torch.isfinite(plane).all(dim=-1)
    if camera.camera_model == "pinhole":
        rays = _normalize(plane)
    else:
        normalized = plane[..., :2] / plane[..., 2:].clamp_min(torch.finfo(dtype).eps)
        radius_squared = normalized.square().sum(dim=-1, keepdim=True)
        xi = camera.xi.to(dtype=dtype)[..., None, None]
        radicand = 1 + (1 - xi.square()) * radius_squared
        valid = valid & torch.isfinite(radicand[..., 0]) & (radicand[..., 0] >= 0)
        scale = (
            xi + torch.sqrt(radicand.clamp_min(0))
        ) / (1 + radius_squared)
        rays = torch.cat((scale * normalized, scale - xi), dim=-1)
        rays = _normalize(rays)
    valid = valid & torch.isfinite(rays).all(dim=-1)
    rays = torch.where(valid[..., None], rays, torch.zeros_like(rays))
    rays = rays.to(dtype=camera.K.dtype)
    if return_validity:
        return rays, valid
    return rays


def project_camera_directions(camera: CameraBatch, directions: Tensor) -> tuple[Tensor, Tensor]:
    """Project camera-frame directions and return pixels plus denominators.

    This is the forward model paired with :func:`build_camera_rays` and is
    shared by the UCPE absolute up-map construction and downstream fusion.
    The denominator is ``[B,S,P]`` and belongs to the active camera model,
    allowing callers to reject singular projections without assuming pinhole
    ``z``. Pixels at non-finite or singular denominators are returned as zero.
    """

    camera.validate()
    if directions.ndim != 4 or directions.shape[:2] != camera.K.shape[:2] or directions.shape[-1] != 3:
        raise ValueError("directions must have shape [B,S,P,3] matching the camera")
    dtype = directions.dtype
    intrinsics = camera.K.to(dtype=dtype)
    x, y, z = directions.unbind(dim=-1)
    if camera.camera_model == "pinhole":
        denominator = z
    else:
        xi = camera.xi.to(dtype=dtype)[..., None]
        denominator = z + xi * directions.norm(dim=-1)
    valid = torch.isfinite(directions).all(dim=-1) & torch.isfinite(denominator) & (denominator.abs() > 1e-8)
    safe = torch.where(valid, denominator, torch.ones_like(denominator))
    u = intrinsics[..., 0, 0, None] * x / safe + intrinsics[..., 0, 2, None]
    v = intrinsics[..., 1, 1, None] * y / safe + intrinsics[..., 1, 2, None]
    pixels = torch.stack((u, v), dim=-1)
    valid = valid & torch.isfinite(pixels).all(dim=-1)
    return torch.where(valid[..., None], pixels, torch.zeros_like(pixels)), denominator


def _canonical_patch_rays(camera: CameraBatch, grid: tuple[int, int]) -> Tensor:
    """Compatibility alias for the former private pinhole ray builder."""

    return build_camera_rays(camera, grid)


def build_world_to_ray(
    camera: CameraBatch,
    grid: tuple[int, int],
    *,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> FullRREContext:
    """Build UCPE relray matrices and a world-anchored up/latitude map."""

    camera.validate()
    grid_h, grid_w = grid
    if grid_h < 1 or grid_w < 1:
        raise ValueError("latent token grid must be positive")
    output_dtype = camera.K.dtype
    compute_dtype = _camera_compute_dtype(output_dtype)
    up = torch.tensor(world_up, device=camera.K.device, dtype=compute_dtype)
    if up.shape != (3,) or not torch.isfinite(up).all() or float(up.norm()) < 1e-6:
        raise ValueError("world_up must be a finite non-zero 3-vector")
    up = _normalize(up)

    rays_camera, ray_valid = build_camera_rays(camera, grid, return_validity=True)
    rays_camera = rays_camera.to(dtype=compute_dtype)
    c2w = camera.T_world_from_camera.to(dtype=compute_dtype)
    rotation = c2w[..., :3, :3]
    centers = c2w[..., :3, 3]
    rays_world = _normalize(torch.einsum("bsij,bspj->bspi", rotation, rays_camera))
    camera_down = rotation[..., :, 1][:, :, None, :].expand_as(rays_world)
    x_axis = torch.cross(camera_down, rays_world, dim=-1)
    camera_right = rotation[..., :, 0][:, :, None, :].expand_as(rays_world)
    fallback = torch.cross(camera_right, rays_world, dim=-1)
    x_axis = torch.where(x_axis.square().sum(-1, keepdim=True) < 1e-8, fallback, x_axis)
    x_axis = _normalize(x_axis)
    y_axis = _normalize(torch.cross(rays_world, x_axis, dim=-1))
    ray_to_world_rotation = torch.stack((x_axis, y_axis, rays_world), dim=-1)
    world_to_ray_rotation = ray_to_world_rotation.transpose(-1, -2)
    expanded_centers = centers[:, :, None, :].expand_as(rays_world)

    world_to_ray = torch.zeros(
        (*rays_world.shape[:-1], 4, 4),
        device=camera.K.device,
        dtype=compute_dtype,
    )
    world_to_ray[..., :3, :3] = world_to_ray_rotation
    world_to_ray[..., :3, 3] = -torch.einsum(
        "bspij,bspj->bspi", world_to_ray_rotation, expanded_centers
    )
    world_to_ray[..., 3, 3] = 1
    identity = torch.eye(4, device=camera.K.device, dtype=compute_dtype).reshape(1, 1, 1, 4, 4)
    world_to_ray = torch.where(ray_valid[..., None, None], world_to_ray, identity)

    latitude = torch.asin(torch.einsum("bspj,j->bsp", rays_world, up).clamp(-1, 1))[..., None]
    expanded_up = up.reshape(1, 1, 1, 3).expand_as(rays_world)
    axis = torch.cross(rays_world, expanded_up, dim=-1)
    axis_norm = axis.norm(dim=-1, keepdim=True)
    axis = axis / axis_norm.clamp_min(1e-8)
    delta = torch.tensor(0.1, device=camera.K.device, dtype=compute_dtype)
    rotated_world = (
        rays_world * delta.cos()
        + torch.cross(axis, rays_world, dim=-1) * delta.sin()
        + axis * (axis * rays_world).sum(dim=-1, keepdim=True) * (1 - delta.cos())
    )
    rotated_camera = torch.einsum(
        "bsji,bspj->bspi", rotation, rotated_world
    )
    projected, projection_denominator = project_camera_directions(camera, rotated_camera)
    u = (torch.arange(grid_w, device=camera.K.device, dtype=compute_dtype) + 0.5) * (camera.image_size[1] / grid_w)
    v = (torch.arange(grid_h, device=camera.K.device, dtype=compute_dtype) + 0.5) * (camera.image_size[0] / grid_h)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    image_up = torch.stack(
        (projected[..., 0] - uu.reshape(1, 1, -1), projected[..., 1] - vv.reshape(1, 1, -1)),
        dim=-1,
    )
    valid = (
        ray_valid[..., None]
        & (axis_norm > 1e-8)
        & torch.isfinite(projection_denominator[..., None])
        & (projection_denominator.abs()[..., None] > 1e-8)
        & torch.isfinite(image_up).all(dim=-1, keepdim=True)
    )
    image_up = torch.where(valid, _normalize(image_up), torch.zeros_like(image_up))
    absmap = torch.cat((image_up, latitude), dim=-1)
    absmap = torch.where(ray_valid[..., None], absmap, torch.zeros_like(absmap))
    batch, sequence, patches = rays_world.shape[:3]
    return FullRREContext(
        world_to_ray.to(dtype=output_dtype).reshape(batch, sequence * patches, 4, 4).contiguous(),
        absmap.to(dtype=output_dtype).reshape(batch, sequence * patches, 3).contiguous(),
        (grid_h, grid_w),
    )


def _rope_coefficients(positions: Tensor, feature_dim: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    if positions.ndim != 1 or feature_dim % 2:
        raise ValueError("RoPE positions must be 1-D and feature_dim must be even")
    count = feature_dim // 2
    frequencies = 100.0 ** (
        -torch.arange(count, device=positions.device, dtype=torch.float32) / count
    )
    angles = positions.float()[:, None] * frequencies[None, :]
    return angles.cos().to(dtype)[None, None], angles.sin().to(dtype)[None, None]


def _apply_rope(value: Tensor, coefficients: tuple[Tensor, Tensor], *, inverse: bool = False) -> Tensor:
    cosine, sine = coefficients
    if cosine.shape[2] != value.shape[2]:
        if value.shape[2] % cosine.shape[2]:
            raise ValueError("RoPE coefficients do not tile to the token count")
        repeats = value.shape[2] // cosine.shape[2]
        cosine = cosine.repeat(1, 1, repeats, 1)
        sine = sine.repeat(1, 1, repeats, 1)
    first, second = value.chunk(2, dim=-1)
    if inverse:
        return torch.cat((cosine * first - sine * second, sine * first + cosine * second), dim=-1)
    return torch.cat((cosine * first + sine * second, -sine * first + cosine * second), dim=-1)


def _apply_token_matrices(value: Tensor, matrices: Tensor) -> Tensor:
    batch, heads, tokens, feature_dim = value.shape
    if matrices.shape != (batch, tokens, 4, 4) or feature_dim % 4:
        raise ValueError("per-token matrices and projective feature dimensions do not align")
    reshaped = value.reshape(batch, heads, tokens, feature_dim // 4, 4)
    return torch.einsum("btij,bntpj->bntpi", matrices, reshaped).reshape(value.shape)


def _apply_prope(
    value: Tensor,
    matrix: Tensor,
    coeff_x: tuple[Tensor, Tensor],
    coeff_y: tuple[Tensor, Tensor],
    *,
    inverse_rope: bool = False,
) -> Tensor:
    projective, horizontal, vertical = torch.split(
        value, (value.shape[-1] // 2, value.shape[-1] // 4, value.shape[-1] // 4), dim=-1
    )
    return torch.cat(
        (
            _apply_token_matrices(projective, matrix),
            _apply_rope(horizontal, coeff_x, inverse=inverse_rope),
            _apply_rope(vertical, coeff_y, inverse=inverse_rope),
        ),
        dim=-1,
    )


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
    rays_cam = build_camera_rays(camera, grid)

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
                "residual_abs_mean": residual.detach().float().abs().mean(),
                "hidden_rms": hidden.detach().float().square().mean().sqrt(),
            }
        self.last_diagnostics = diagnostics
        return result

    def forward(self, camera: CameraBatch, latent_shape: tuple[int, int, int], timesteps: Tensor) -> Mapping[int, Tensor]:
        return self.residuals(camera, latent_shape, timesteps)


class FullRREAttentionBlock(nn.Module):
    """Compressed UCPE/PRoPE camera attention running beside Wan self-attention."""

    def __init__(self, feature_dim: int, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        if hidden_dim % attention_heads or (hidden_dim // attention_heads) % 8:
            raise ValueError("full RRE head dimensions must be divisible by 8")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.head_dim = self.hidden_dim // self.attention_heads
        self.camera_encoder = nn.Linear(3, self.feature_dim)
        self.q = nn.Linear(self.feature_dim, self.hidden_dim)
        self.k = nn.Linear(self.feature_dim, self.hidden_dim)
        self.v = nn.Linear(self.feature_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.feature_dim, dtype=torch.float32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens: Tensor, context: FullRREContext) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.feature_dim:
            raise ValueError(f"full RRE tokens must have shape [B,L,{self.feature_dim}]")
        batch, token_count, _ = tokens.shape
        if context.world_to_ray.shape != (batch, token_count, 4, 4):
            raise ValueError("world-to-ray matrices do not match Wan tokens")
        if context.absmap.shape != (batch, token_count, 3):
            raise ValueError("absolute camera map does not match Wan tokens")
        grid_h, grid_w = context.patch_grid
        patches = grid_h * grid_w
        if token_count % patches:
            raise ValueError("Wan token count is not divisible by the camera patch grid")

        value = tokens + self.camera_encoder(context.absmap.to(dtype=tokens.dtype))
        q = self.q(value).reshape(batch, token_count, self.attention_heads, self.head_dim).transpose(1, 2)
        k = self.k(value).reshape(batch, token_count, self.attention_heads, self.head_dim).transpose(1, 2)
        v = self.v(value).reshape(batch, token_count, self.attention_heads, self.head_dim).transpose(1, 2)
        matrices = context.world_to_ray.to(device=tokens.device, dtype=q.dtype)
        inverse = _invert_se3(matrices)
        sequence = token_count // patches
        coeff_x = _rope_coefficients(
            torch.tile(torch.arange(grid_w, device=tokens.device), (grid_h * sequence,)),
            self.head_dim // 4,
            q.dtype,
        )
        coeff_y = _rope_coefficients(
            torch.tile(
                torch.repeat_interleave(torch.arange(grid_h, device=tokens.device), grid_w),
                (sequence,),
            ),
            self.head_dim // 4,
            q.dtype,
        )
        q = _apply_prope(q, matrices.transpose(-1, -2), coeff_x, coeff_y)
        k = _apply_prope(k, inverse, coeff_x, coeff_y)
        v = _apply_prope(v, inverse, coeff_x, coeff_y)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attended = _apply_prope(attended, matrices, coeff_x, coeff_y, inverse_rope=True)
        attended = attended.transpose(1, 2).reshape(batch, token_count, self.hidden_dim)
        return self.output(attended)


class FullRREConditioner(nn.Module):
    """Thirty independent relray_absmap branches for the official Wan blocks."""

    injection_mode = "self_attention"

    def __init__(
        self,
        *,
        feature_dim: int = 1536,
        hidden_dim: int = 192,
        attention_heads: int = 1,
        branch_count: int = 30,
        compression: int = 8,
        absmap: bool = True,
        world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        super().__init__()
        if branch_count < 1 or compression < 1:
            raise ValueError("branch_count and compression must be positive")
        if hidden_dim != feature_dim // compression:
            raise ValueError("hidden_dim must equal feature_dim // compression")
        if absmap is not True:
            raise ValueError("full RRE requires relray_absmap")
        if len(world_up) != 3:
            raise ValueError("world_up must contain three values")
        self.representation = "rre_full"
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.branch_count = int(branch_count)
        self.compression = int(compression)
        self.absmap = True
        self.world_up = tuple(float(value) for value in world_up)
        self.branches = nn.ModuleList(
            FullRREAttentionBlock(self.feature_dim, self.hidden_dim, self.attention_heads)
            for _ in range(self.branch_count)
        )
        self.last_diagnostics: dict[int, dict[str, Tensor]] = {}

    def prepare(self, camera: CameraBatch, latent_shape: tuple[int, int, int]) -> FullRREContext:
        sequence, latent_h, latent_w = latent_shape
        if camera.K.shape[1] != sequence:
            raise ValueError("camera sequence length must match latent_shape[0]")
        if latent_h % 2 or latent_w % 2:
            raise ValueError("latent spatial dimensions must be divisible by Wan patch size 2")
        return build_world_to_ray(
            camera,
            (latent_h // 2, latent_w // 2),
            world_up=self.world_up,
        )

    def attention_residual(
        self,
        block_index: int,
        tokens: Tensor,
        context: FullRREContext,
        *,
        enabled: bool = True,
    ) -> Tensor:
        if not 0 <= block_index < self.branch_count:
            raise ValueError("full RRE block index is outside configured branches")
        if not enabled:
            self.last_diagnostics.pop(block_index, None)
            return torch.zeros_like(tokens)
        residual = self.branches[block_index](tokens, context)
        self.last_diagnostics[block_index] = {
            "input_rms": tokens.detach().float().square().mean().sqrt(),
            "residual_rms": residual.detach().float().square().mean().sqrt(),
            "residual_abs_mean": residual.detach().float().abs().mean(),
        }
        return residual

    def reset_output(self) -> None:
        for branch in self.branches:
            nn.init.zeros_(branch.output.weight)
            nn.init.zeros_(branch.output.bias)
        self.last_diagnostics = {}


_OFFICIAL_UCPE_LAYER_NAMES = {
    "cam_encoder": "camera_encoder",
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "out_proj": "output",
}

_CANONICAL_UCPE_ARCHITECTURE = {
    "feature_dim": 1536,
    "hidden_dim": 192,
    "attention_heads": 1,
    "branch_count": 30,
    "compression": 8,
}


def _official_ucpe_key_map(module: FullRREConditioner) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for branch in range(module.branch_count):
        for source_layer, target_layer in _OFFICIAL_UCPE_LAYER_NAMES.items():
            for suffix in ("weight", "bias"):
                source = f"pipe.dit.blocks.{branch}.cam_self_attn.{source_layer}.{suffix}"
                target = f"branches.{branch}.{target_layer}.{suffix}"
                mapping[source] = target
    return mapping


def _canonical_ucpe_target_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    layer_shapes = {
        "camera_encoder": {"weight": (1536, 3), "bias": (1536,)},
        "q": {"weight": (192, 1536), "bias": (192,)},
        "k": {"weight": (192, 1536), "bias": (192,)},
        "v": {"weight": (192, 1536), "bias": (192,)},
        "output": {"weight": (1536, 192), "bias": (1536,)},
    }
    for branch in range(30):
        for layer, parameters in layer_shapes.items():
            for suffix, shape in parameters.items():
                shapes[f"branches.{branch}.{layer}.{suffix}"] = shape
    return shapes


def _validate_canonical_ucpe_module(module: FullRREConditioner) -> None:
    actual_architecture = {
        key: getattr(module, key) for key in _CANONICAL_UCPE_ARCHITECTURE
    }
    expected_shapes = _canonical_ucpe_target_shapes()
    actual_state = module.state_dict()
    actual_shapes = {name: tuple(value.shape) for name, value in actual_state.items()}
    if (
        actual_architecture != _CANONICAL_UCPE_ARCHITECTURE
        or module.representation != "rre_full"
        or module.absmap is not True
        or set(actual_shapes) != set(expected_shapes)
        or any(actual_shapes[name] != shape for name, shape in expected_shapes.items())
    ):
        raise RuntimeError(
            "canonical official UCPE conversion requires 30 branches with exact "
            "1536-to-192 relray_absmap tensor shapes, one attention head, and compression 8"
        )


def convert_official_ucpe_checkpoint(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    asserted_source_commit: str,
    module: FullRREConditioner | None = None,
) -> dict[str, str]:
    """Convert one pinned official UCPE adapter checkpoint to format v2.

    The input contract is the adapter-only Lightning ``state_dict`` emitted by
    UCPE commit :data:`OFFICIAL_UCPE_COMMIT`. Every tensor is mapped by name
    and checked against the destination module before any file is written.
    Provenance, including the hash of the completed output, is stored in an
    adjacent ``.provenance.json`` file because a file cannot contain its own
    cryptographic digest.
    """

    source = Path(source_path)
    destination = Path(destination_path)
    provenance_path = destination.with_suffix(destination.suffix + ".provenance.json")
    if asserted_source_commit != OFFICIAL_UCPE_COMMIT:
        raise RuntimeError("official UCPE source commit assertion does not match pinned contract")
    if not source.is_file():
        raise FileNotFoundError(f"official UCPE checkpoint does not exist: {source}")
    if destination.exists() or provenance_path.exists():
        raise FileExistsError("converted checkpoint or provenance file already exists")
    if module is None:
        module = FullRREConditioner()
    if not isinstance(module, FullRREConditioner):
        raise TypeError("module must be FullRREConditioner")
    _validate_canonical_ucpe_module(module)

    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, Mapping) and "state_dict" in payload else payload
    if not isinstance(state, Mapping) or not all(isinstance(name, str) for name in state):
        raise RuntimeError("official UCPE checkpoint must contain a tensor state_dict")

    key_map = _official_ucpe_key_map(module)
    actual_keys = set(state)
    expected_keys = set(key_map)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        raise RuntimeError(f"missing official UCPE branch tensors: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected official UCPE checkpoint keys: {unexpected}")

    expected_state = module.state_dict()
    converted: dict[str, Tensor] = {}
    for source_name, target_name in key_map.items():
        value = state[source_name]
        if not isinstance(value, Tensor):
            raise RuntimeError(f"official UCPE value is not a tensor: {source_name}")
        expected_shape = expected_state[target_name].shape
        if value.shape != expected_shape:
            raise RuntimeError(
                f"shape mismatch for {source_name}: expected {tuple(expected_shape)}, got {tuple(value.shape)}"
            )
        converted[target_name] = value.detach().cpu()

    input_sha256 = _sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    provenance_temporary = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    try:
        torch.save(
            {
                "format_version": 2,
                "geometry": converted,
                "config": {
                    "representation": module.representation,
                    "feature_dim": module.feature_dim,
                    "hidden_dim": module.hidden_dim,
                    "attention_heads": module.attention_heads,
                    "branch_count": module.branch_count,
                    "compression": module.compression,
                    "absmap": module.absmap,
                    "world_up": list(module.world_up),
                },
                "source_commit": OFFICIAL_UCPE_COMMIT,
                "asserted_source_commit": asserted_source_commit,
                "source_checkpoint_sha256": input_sha256,
                "trainable_parameters": sum(value.numel() for value in module.parameters() if value.requires_grad),
            },
            temporary,
        )
        output_sha256 = _sha256_file(temporary)
        provenance = {
            "asserted_source_commit": asserted_source_commit,
            "input_sha256": input_sha256,
            "output_sha256": output_sha256,
        }
        provenance_temporary.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        provenance_temporary.replace(provenance_path)
    finally:
        temporary.unlink(missing_ok=True)
        provenance_temporary.unlink(missing_ok=True)
    return provenance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_geometry_checkpoint(
    path: str | Path,
    geometry: GeometryConditioner | FullRREConditioner,
    *,
    config: Mapping[str, Any],
    parent_checkpoint: str | Path,
    step: int,
) -> None:
    """Save legacy token adapters as v1 and full-RRE attention adapters as v2."""

    path = Path(path)
    parent = Path(parent_checkpoint).resolve()
    saved_config = dict(config)
    if isinstance(geometry, FullRREConditioner):
        version = 2
        saved_config.update(
            representation="rre_full",
            branch_count=geometry.branch_count,
            compression=geometry.compression,
            hidden_dim=geometry.hidden_dim,
            attention_heads=geometry.attention_heads,
            absmap=geometry.absmap,
            world_up=list(geometry.world_up),
        )
    elif isinstance(geometry, GeometryConditioner):
        version = 1
    else:
        raise TypeError("unsupported geometry conditioner")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": version,
            "geometry": {name: value.detach().cpu() for name, value in geometry.state_dict().items()},
            "config": saved_config,
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": _sha256_file(parent),
            "step": int(step),
            "trainable_parameters": sum(value.numel() for value in geometry.parameters() if value.requires_grad),
        },
        temporary,
    )
    temporary.replace(path)


def load_geometry_checkpoint(
    path: str | Path,
    geometry: GeometryConditioner | FullRREConditioner,
    *,
    expected_parent_checkpoint: str | None = None,
) -> dict[str, object]:
    """Strictly load a Stage 2 geometry checkpoint and return its metadata."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    version = payload.get("format_version")
    if version not in (1, 2) or "geometry" not in payload or "config" not in payload:
        raise RuntimeError("unsupported Stage 2 geometry checkpoint")
    config = payload["config"]
    if config.get("representation") != geometry.representation:
        raise RuntimeError("geometry representation does not match checkpoint")
    if version == 1:
        if not isinstance(geometry, GeometryConditioner):
            raise RuntimeError("format-v1 checkpoints require the simplified geometry adapter")
        if tuple(config.get("blocks", ())) != geometry.blocks:
            raise RuntimeError("geometry block configuration does not match checkpoint")
    else:
        if not isinstance(geometry, FullRREConditioner):
            raise RuntimeError("format-v2 checkpoints require FullRREConditioner")
        expected = {
            "branch_count": geometry.branch_count,
            "compression": geometry.compression,
            "hidden_dim": geometry.hidden_dim,
            "attention_heads": geometry.attention_heads,
            "absmap": geometry.absmap,
            "world_up": list(geometry.world_up),
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise RuntimeError("full RRE architecture does not match checkpoint")
    if expected_parent_checkpoint is not None and payload.get("parent_checkpoint") != expected_parent_checkpoint:
        raise RuntimeError("geometry parent checkpoint mismatch")
    geometry.load_state_dict(payload["geometry"], strict=True)
    return {
        "format_version": version,
        "config": config,
        "step": payload.get("step"),
        "parent_checkpoint": payload.get("parent_checkpoint"),
        "parent_checkpoint_sha256": payload.get("parent_checkpoint_sha256"),
        "trainable_parameters": payload.get("trainable_parameters"),
    }
