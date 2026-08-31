"""Shape-safe wrappers around the official Wan2.1 temporal VAE."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

from rl3dsr.models.wan.checkpoint import WanCheckpoint


def _official_root() -> Path:
    return Path(__file__).resolve().parents[4] / "third_party" / "wan2_1"


def _load_official_vae():
    root = _official_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from wan.modules.vae import WanVAE as OfficialWanVAE

    return OfficialWanVAE


class WanVAE:
    """Frozen Wan VAE with explicit multiview and temporal entry points."""

    def __init__(self, model: object, *, device: str | torch.device, channels: int = 16):
        self.model = model
        self.device = torch.device(device)
        self.channels = channels
        self.model.model.eval().requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "WanVAE":
        checkpoint = WanCheckpoint.from_dir(model_dir)
        official = _load_official_vae()
        model = official(z_dim=16, vae_pth=str(checkpoint.vae_path), dtype=dtype, device=str(device))
        return cls(model, device=device)

    def encode_multiview(self, rgb: Tensor) -> Tensor:
        """Encode ``[B,3,V,H,W]`` as independent one-frame videos."""

        _validate_rgb(rgb, name="multiview rgb")
        b, c, views, height, width = rgb.shape
        flat = rgb.permute(0, 2, 1, 3, 4).reshape(b * views, c, 1, height, width)
        videos = [item.to(self.device) for item in flat]
        latents = self.model.encode(videos)
        stacked = torch.stack(latents, dim=0)
        if stacked.ndim != 5 or stacked.shape[1] != self.channels or stacked.shape[2] != 1:
            raise RuntimeError(f"official Wan VAE returned unexpected per-view shape: {tuple(stacked.shape)}")
        _, cz, _, h, w = stacked.shape
        return stacked.reshape(b, views, cz, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def decode_multiview(self, latents: Tensor) -> Tensor:
        """Decode ``[B,16,V,h,w]`` symmetrically to ``[B,3,V,H,W]``."""

        _validate_latents(latents, name="multiview latents")
        b, channels, views, height, width = latents.shape
        flat = latents.permute(0, 2, 1, 3, 4).reshape(b * views, channels, 1, height, width)
        decoded = torch.stack(self.model.decode([item.to(self.device) for item in flat]), dim=0)
        if decoded.ndim != 5 or decoded.shape[1] != 3 or decoded.shape[2] != 1:
            raise RuntimeError(f"official Wan VAE returned unexpected decoded shape: {tuple(decoded.shape)}")
        _, c, _, h, w = decoded.shape
        return decoded.reshape(b, views, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def encode_video(self, rgb: Tensor) -> Tensor:
        """Encode ``[B,3,T,H,W]`` using native Wan temporal processing."""

        _validate_rgb(rgb, name="video rgb")
        videos = [item.to(self.device) for item in rgb]
        latents = torch.stack(self.model.encode(videos), dim=0)
        if latents.ndim != 5 or latents.shape[1] != self.channels:
            raise RuntimeError(f"official Wan VAE returned unexpected video shape: {tuple(latents.shape)}")
        return latents.contiguous()

    def decode_video(self, latents: Tensor) -> Tensor:
        """Decode ``[B,16,T',h,w]`` using native Wan temporal processing."""

        _validate_latents(latents, name="video latents")
        decoded = torch.stack(self.model.decode([item.to(self.device) for item in latents]), dim=0)
        if decoded.ndim != 5 or decoded.shape[1] != 3:
            raise RuntimeError(f"official Wan VAE returned unexpected decoded video shape: {tuple(decoded.shape)}")
        return decoded.contiguous()


def _validate_rgb(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 5 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B,3,S,H,W]")
    if any(int(size) < 1 for size in value.shape) or value.shape[-2] % 8 or value.shape[-1] % 8:
        raise ValueError(f"{name} must have positive dimensions and H/W divisible by 8")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating tensor")
    if float(value.detach().amin()) < -1.0001 or float(value.detach().amax()) > 1.0001:
        raise ValueError(f"{name} must be in [-1,1]")


def _validate_latents(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 5 or value.shape[1] != 16:
        raise ValueError(f"{name} must have shape [B,16,S,h,w]")
    if any(int(size) < 1 for size in value.shape) or value.shape[-2] % 1 or value.shape[-1] % 1:
        raise ValueError(f"{name} must have positive dimensions")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating tensor")
