"""Frozen wrapper around the official Wan2.1 DiT forward."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor

from rl3dsr.models.wan.checkpoint import WanCheckpoint


def _load_official_model():
    root = Path(__file__).resolve().parents[4] / "third_party" / "wan2_1"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from wan.modules.model import WanModel

    return WanModel


class WanDiT:
    """Real Wan DiT accepting already-encoded 3D or 4D latents."""

    def __init__(self, model: object, *, device: str | torch.device):
        self.model = model.eval().requires_grad_(False)
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "WanDiT":
        checkpoint = WanCheckpoint.from_dir(model_dir)
        official = _load_official_model()
        model = official.from_pretrained(str(checkpoint.model_dir), torch_dtype=dtype)
        model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        return cls(model, device=device)

    def __call__(self, latents: Tensor, timesteps: Tensor, context: Tensor | None = None) -> Tensor:
        if not isinstance(latents, Tensor) or latents.ndim != 5 or latents.shape[1] != 16:
            raise ValueError("latents must have shape [B,16,F,h,w]")
        if not torch.is_floating_point(latents) or not torch.isfinite(latents).all():
            raise ValueError("latents must be finite floating values")
        batch, _, frames, height, width = latents.shape
        if height % 2 or width % 2:
            raise ValueError("latent height and width must be divisible by DiT patch size 2")
        if timesteps.shape != (batch,):
            raise ValueError(f"timesteps must have shape [{batch}]")
        if context is None:
            context = torch.zeros(batch, 512, 4096, device=self.device, dtype=latents.dtype)
        if context.shape != (batch, 512, 4096):
            raise ValueError(f"context must have shape [{batch},512,4096]")
        model_dtype = next(self.model.parameters()).dtype
        samples = [item.to(device=self.device, dtype=model_dtype) for item in latents]
        contexts = [item.to(device=self.device, dtype=model_dtype) for item in context]
        seq_len = frames * (height // 2) * (width // 2)
        autocast = (
            torch.autocast(device_type=self.device.type, dtype=model_dtype)
            if self.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast:
            outputs = self.model(samples, timesteps.to(self.device), contexts, seq_len)
        output = torch.stack(outputs, dim=0)
        if output.shape != latents.shape:
            raise RuntimeError(f"Wan DiT shape mismatch: expected {tuple(latents.shape)}, got {tuple(output.shape)}")
        return output.contiguous()
