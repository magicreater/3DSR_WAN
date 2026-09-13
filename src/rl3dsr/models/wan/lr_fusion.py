"""Static-3D LR evidence fusion before the existing Wan LR bridge.

Inputs/outputs are [B,V,P,D], with row-major patch order within each view.
Only this residual branch is trainable here; its zero output projection initially
preserves the LR encoder features exactly. Temporal sequences bypass the branch.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .geometry_conditioning import CameraBatch


def patch_fundamental_matrices(
    camera: CameraBatch, patch_grid: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """Return F[B,target,source,3,3] and nondegenerate-pair flags.

    F maps a target homogeneous patch coordinate to an epipolar line in the
    source image. Coordinates are feature-cell coordinates, with centers at
    (column + .5, row + .5), matching the existing RRE ray convention. K is in
    pixels of camera.image_size; T_world_from_camera uses OpenCV camera axes.
    """
    camera.validate()
    if len(patch_grid) != 2 or any(type(value) is not int or value < 1 for value in patch_grid):
        raise ValueError('patch_grid must contain two positive integers')
    gh, gw = patch_grid
    if camera.K.device != camera.T_world_from_camera.device:
        raise ValueError('camera tensors must share a device')
    with torch.autocast(device_type=camera.K.device.type, enabled=False):
        k = camera.K.float()
        scale = k.new_tensor([gw / camera.image_size[1], gh / camera.image_size[0], 1])
        k = k * scale.view(1, 1, 3, 1)
        try:
            inv_k = torch.linalg.inv(k)
        except RuntimeError as exc:
            raise ValueError('camera intrinsics must be invertible') from exc
        c2w = camera.T_world_from_camera.float()
        rotation = c2w[..., :3, :3]
        centers = c2w[..., :3, 3]
        # Target i -> source j: R_j^T R_i, R_j^T (c_i - c_j).
        source_rt = rotation.transpose(-1, -2)[:, None, :]
        relative_r = source_rt @ rotation[:, :, None]
        relative_t = (source_rt @ (centers[:, :, None] - centers[:, None, :]).unsqueeze(-1)).squeeze(-1)
        x, y, z = relative_t.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*x.shape, 3, 3)
        fundamental = inv_k.transpose(-1, -2)[:, None, :] @ skew @ relative_r @ inv_k[:, :, None]
        valid = relative_t.square().sum(-1) > 1e-14
        # Normalize F to keep point-line divisions well-scaled for small baselines.
        norm = fundamental.square().sum((-1, -2), keepdim=True).sqrt()
        fundamental = fundamental / norm.clamp_min(1e-12)
        fundamental = torch.where(valid[..., None, None], fundamental, torch.zeros_like(fundamental))
        if not torch.isfinite(fundamental).all():
            raise ValueError('camera geometry produced non-finite fundamental matrices')
        return fundamental, valid


def _patch_centers(device: torch.device, patch_grid: tuple[int, int]) -> Tensor:
    gh, gw = patch_grid
    yy, xx = torch.meshgrid(
        torch.arange(gh, device=device, dtype=torch.float32) + .5,
        torch.arange(gw, device=device, dtype=torch.float32) + .5,
        indexing='ij',
    )
    return torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(gh * gw, 3)


def _epipolar_chunk(
    matrices: Tensor,
    valid: Tensor,
    pixels: Tensor,
    query_views: Tensor,
    query_patches: Tensor,
    *,
    tau: float,
    band: float,
) -> tuple[Tensor, Tensor, Tensor]:
    query_pixels = pixels[query_patches]
    lines = torch.einsum('bqvij,qj->bqvi', matrices[:, query_views], query_pixels)
    numer = torch.einsum('bqvi,pi->bqvp', lines, pixels).square()
    denom = lines[..., :2].square().sum(-1)
    usable = valid[:, query_views] & (denom > 1e-12)
    dist2 = numer / denom.clamp_min(1e-12).unsqueeze(-1)
    bias = (-dist2 / (2 * tau**2)).clamp(-20, 0)
    bias = torch.where(usable[..., None], bias, torch.zeros_like(bias))
    allowed = (~usable[..., None]) | (dist2 <= band**2)
    return bias, allowed, usable


def epipolar_local_key_mask(
    camera: CameraBatch,
    patch_grid: tuple[int, int],
    *,
    band: float = 1.5,
) -> tuple[Tensor, Tensor]:
    """Return local-band key mask [B,V*P,V*P] and usable pairs [B,V*P,V]."""
    if isinstance(band, bool) or not isinstance(band, (int, float)) or not math.isfinite(band) or band <= 0:
        raise ValueError('band must be finite and positive')
    matrices, valid = patch_fundamental_matrices(camera, patch_grid)
    patches = patch_grid[0] * patch_grid[1]
    tokens = camera.K.shape[1] * patches
    ids = torch.arange(tokens, device=camera.K.device)
    _, allowed, usable = _epipolar_chunk(
        matrices,
        valid,
        _patch_centers(camera.K.device, patch_grid),
        ids // patches,
        ids % patches,
        tau=1.0,
        band=float(band),
    )
    return allowed.reshape(camera.K.shape[0], tokens, tokens), usable


class LRViewFusion(nn.Module):
    """One zero-initialized LR attention residual; never encodes views as time.

    ``source_mask[B,V]`` is boolean and filters K/V only: it cannot remove a
    target query or its own LR residual. A zero K/V null candidate is always
    available, including when every source is masked. No checkpoint or model
    loading takes place in this module.
    """

    def __init__(
        self,
        feature_dim: int = 1536,
        hidden_dim: int = 192,
        heads: int = 3,
        mode: str = 'epipolar',
        query_chunk_size: int = 128,
        tau: float = 1.0,
        epipolar_band: float = 1.5,
        allow_self_view_source: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {'off', 'same_view', 'visual', 'epipolar', 'epipolar_local'}:
            raise ValueError('invalid LR fusion mode')
        dimensions = (feature_dim, hidden_dim, heads, query_chunk_size)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError('feature_dim, hidden_dim, heads and query_chunk_size must be positive integers')
        if hidden_dim % heads:
            raise ValueError('positive dimensions required and hidden_dim must divide heads')
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau <= 0:
            raise ValueError('query_chunk_size and finite tau must be positive')
        if isinstance(epipolar_band, bool) or not isinstance(epipolar_band, (int, float)) or not math.isfinite(epipolar_band) or epipolar_band <= 0:
            raise ValueError('epipolar_band must be finite and positive')
        if type(allow_self_view_source) is not bool:
            raise ValueError('allow_self_view_source must be boolean')
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.mode = mode
        self.query_chunk_size = query_chunk_size
        self.tau = float(tau)
        self.epipolar_band = float(epipolar_band)
        self.allow_self_view_source = allow_self_view_source
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, dict[str, Tensor]] = {}
        self.qkv = nn.Linear(feature_dim, hidden_dim * 3, bias=False)
        self.output = nn.Linear(hidden_dim, feature_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        features: Tensor,
        camera: CameraBatch | None,
        patch_grid: tuple[int, int],
        *,
        source_mask: Tensor | None = None,
        allow_self_view_source: bool | None = None,
    ) -> Tensor:
        self.last_diagnostics = {}
        if features.ndim != 4 or features.shape[-1] != self.feature_dim:
            raise ValueError('features must have shape [B,V,P,feature_dim]')
        b, views, patches, _ = features.shape
        if len(patch_grid) != 2 or any(type(value) is not int or value < 1 for value in patch_grid):
            raise ValueError('patch_grid must contain two positive integers')
        gh, gw = patch_grid
        if min(b, views) < 1 or patches != gh * gw:
            raise ValueError('patch_grid must match the nonempty LR token grid')
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError('features must be finite floating point')
        if source_mask is not None:
            if source_mask.shape != (b, views) or source_mask.dtype != torch.bool:
                raise ValueError('source_mask must be bool [B,V]')
            if source_mask.device != features.device:
                raise ValueError('source_mask must share the feature device')
        if allow_self_view_source is not None and type(allow_self_view_source) is not bool:
            raise ValueError('allow_self_view_source override must be boolean')
        allow_self = (
            self.allow_self_view_source
            if allow_self_view_source is None
            else allow_self_view_source
        )
        if self.mode == 'off':
            return features
        if camera is not None:
            camera.validate(batch=b)
            if camera.K.shape[1] != views:
                raise ValueError('camera views must match LR views')
            if camera.sequence_kind == 'temporal':
                return features
        if views == 1:
            return features
        if self.mode == 'epipolar' and camera is None:
            raise ValueError('epipolar fusion requires cameras')
        if camera is not None and (camera.K.device != features.device or camera.T_world_from_camera.device != features.device):
            raise ValueError('camera and LR features must share a device')

        tokens = views * patches
        head_dim = self.hidden_dim // self.heads
        qkv = self.qkv(features.to(self.qkv.weight.dtype)).reshape(
            b, tokens, 3, self.heads, head_dim
        )
        q, k, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        with torch.autocast(device_type=features.device.type, enabled=False):
            q, k, value = q.float(), k.float(), value.float()
            # A fixed null key has logit zero and contributes no evidence.
            null = k.new_zeros((b, self.heads, 1, head_dim))
            k = torch.cat((k, null), dim=2)
            value = torch.cat((value, null), dim=2)
            view_ids = torch.arange(tokens, device=features.device) // patches
            record = bool(self.record_diagnostics)
            same_mass = features.new_zeros((), dtype=torch.float32)
            cross_mass = features.new_zeros((), dtype=torch.float32)
            null_mass = features.new_zeros((), dtype=torch.float32)
            retained_key_ratio = features.new_zeros((), dtype=torch.float32)
            attention_count = 0
            key_count = 0
            aux_source_ratio = features.new_tensor(1.0, dtype=torch.float32)
            if record and source_mask is not None and views > 1:
                aux_source_ratio = source_mask[:, 1:].float().mean()
            pixels = _patch_centers(features.device, patch_grid)
            matrices, valid = (None, None)
            if self.mode in {'epipolar', 'epipolar_local'}:
                matrices, valid = patch_fundamental_matrices(camera, patch_grid)
            key_allowed = None
            if source_mask is not None:
                key_allowed = source_mask.repeat_interleave(patches, dim=1)
            chunks = []
            for start in range(0, tokens, self.query_chunk_size):
                stop = min(tokens, start + self.query_chunk_size)
                query_views = view_ids[start:stop]
                logits = (q[:, :, start:stop] @ k.transpose(-1, -2)) / math.sqrt(head_dim)
                allowed_keys = None
                if record:
                    allowed_keys = torch.ones(
                        (b, stop - start, tokens), dtype=torch.bool, device=features.device
                    )
                if matrices is not None:
                    bias, allowed, _ = _epipolar_chunk(
                        matrices,
                        valid,
                        pixels,
                        query_views,
                        torch.arange(start, stop, device=features.device) % patches,
                        tau=self.tau,
                        band=self.epipolar_band,
                    )
                    logits[..., :tokens] = logits[..., :tokens] + bias.reshape(b, stop - start, tokens)[:, None]
                    if self.mode == 'epipolar_local':
                        # Keep a null candidate and fall back to global keys for
                        # degenerate camera pairs; otherwise restrict each query
                        # to a finite epipolar band in source patch coordinates.
                        logits[..., :tokens] = logits[..., :tokens].masked_fill(
                            ~allowed.reshape(b, stop - start, tokens)[:, None], -torch.inf
                        )
                        if record:
                            allowed_keys = allowed_keys & allowed.reshape(b, stop - start, tokens)
                if self.mode == 'same_view':
                    allowed = query_views[:, None] == view_ids[None, :]
                    logits[..., :tokens] = logits[..., :tokens].masked_fill(~allowed[None, None], -torch.inf)
                    if record:
                        allowed_keys = allowed_keys & allowed[None].expand(b, -1, -1)
                if not allow_self:
                    allowed = query_views[:, None] != view_ids[None, :]
                    logits[..., :tokens] = logits[..., :tokens].masked_fill(
                        ~allowed[None, None], -torch.inf
                    )
                    if record:
                        allowed_keys = allowed_keys & allowed[None].expand(b, -1, -1)
                if key_allowed is not None:
                    logits[..., :tokens] = logits[..., :tokens].masked_fill(~key_allowed[:, None, None, :], -torch.inf)
                    if record:
                        allowed_keys = allowed_keys & key_allowed[:, None, :]
                weights = torch.softmax(logits, dim=-1)
                if record:
                    same = query_views[:, None] == view_ids[None, :]
                    non_null = weights[..., :tokens]
                    same_mass = same_mass + (non_null * same[None, None]).sum()
                    cross_mass = cross_mass + (non_null * (~same)[None, None]).sum()
                    null_mass = null_mass + weights[..., tokens].sum()
                    retained_key_ratio = retained_key_ratio + allowed_keys.float().mean() * (b * (stop - start))
                    attention_count += b * self.heads * (stop - start)
                    key_count += b * (stop - start)
                chunks.append(weights @ value)
            fused = torch.cat(chunks, dim=2).transpose(1, 2).reshape(b, views, patches, self.hidden_dim)
            if record:
                self.last_diagnostics = {
                    'fusion': {
                        'same_view_attention_mass': (same_mass / attention_count).detach(),
                        'cross_view_attention_mass': (cross_mass / attention_count).detach(),
                        'null_attention_mass': (null_mass / attention_count).detach(),
                        'attention_mass_total': ((same_mass + cross_mass + null_mass) / attention_count).detach(),
                        'retained_key_ratio': (retained_key_ratio / max(key_count, 1)).detach(),
                        'active_auxiliary_source_ratio': aux_source_ratio.detach(),
                    }
                }
        residual = self.output(fused.to(self.output.weight.dtype)).to(features.dtype)
        return features + residual
