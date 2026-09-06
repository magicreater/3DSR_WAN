"""Frozen wrapper around the official Wan2.1 DiT forward."""

from __future__ import annotations

import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

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
    """Real frozen Wan DiT accepting already-encoded 3D or 4D latents.

    Stage 1 adds token residuals before selected blocks (legacy: block 0).
    Scoped pre-hooks keep the official forward unchanged and are always removed
    before this call returns, including failures during hook registration.
    """

    def __init__(self, model: object, *, device: str | torch.device):
        self.model = model.eval().requires_grad_(False)
        self.device = torch.device(device)
        self._injection_lock = threading.RLock()
        self.last_injection_stats: dict[int, dict[str, Tensor]] = {}

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

    def __call__(
        self,
        latents: Tensor,
        timesteps: Tensor,
        context: Tensor | None = None,
        *,
        token_residual: Tensor | None = None,
        block_token_residuals: Mapping[int, Tensor] | None = None,
        camera_attention: tuple[object, object] | None = None,
    ) -> Tensor:
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
        if token_residual is not None and block_token_residuals is not None:
            raise ValueError("token_residual and block_token_residuals are mutually exclusive")
        if block_token_residuals is not None and not isinstance(block_token_residuals, Mapping):
            raise ValueError("block_token_residuals must be a mapping")
        residuals = {0: token_residual} if token_residual is not None else dict(block_token_residuals or {})
        for block_index, residual in residuals.items():
            if isinstance(block_index, bool) or not isinstance(block_index, int) or not 0 <= block_index < len(self.model.blocks):
                raise ValueError("residual block index must be an integer within model.blocks")
            dim = int(getattr(self.model, "dim"))
            expected = (batch, seq_len, dim)
            if not isinstance(residual, Tensor) or residual.shape != expected:
                raise ValueError(f"token_residual must have shape {expected}")
            if not torch.is_floating_point(residual) or not torch.isfinite(residual).all():
                raise ValueError("token_residual must be finite floating values")
            residuals[block_index] = residual.to(device=self.device)
        camera_adapter = camera_context = None
        if camera_attention is not None:
            if not isinstance(camera_attention, tuple) or len(camera_attention) != 2:
                raise ValueError("camera_attention must be an (adapter, context) tuple")
            camera_adapter, camera_context = camera_attention
            if getattr(camera_adapter, "injection_mode", None) != "self_attention":
                raise ValueError("camera attention adapter has an unsupported injection mode")
            if int(getattr(camera_adapter, "branch_count", -1)) != len(self.model.blocks):
                raise ValueError("camera attention branch count must match Wan blocks")
        autocast = (
            torch.autocast(device_type=self.device.type, dtype=model_dtype)
            if self.device.type in {"cuda", "cpu"} and model_dtype in {torch.float16, torch.bfloat16}
            else nullcontext()
        )
        with self._injection_lock:
            handles = []
            self.last_injection_stats = {}
            try:
                for block_index, residual in residuals.items():
                    def inject_tokens(_module, args, residual=residual, block_index=block_index):
                        embedded = args[0]
                        if embedded.shape != residual.shape:
                            raise RuntimeError(
                                f"Wan patch tokens {tuple(embedded.shape)} do not match LR residual {tuple(residual.shape)}"
                            )
                        cast_residual = residual.to(dtype=embedded.dtype)
                        input_rms = embedded.detach().float().square().mean().sqrt()
                        residual_rms = cast_residual.detach().float().square().mean().sqrt()
                        self.last_injection_stats[block_index] = {
                            "input_rms": input_rms, "residual_rms": residual_rms,
                            "residual_to_input_rms": residual_rms / input_rms.clamp_min(1e-12),
                        }
                        return (embedded + cast_residual, *args[1:])
                    handles.append(self.model.blocks[block_index].register_forward_pre_hook(inject_tokens))
                if camera_adapter is not None:
                    for block_index, block in enumerate(self.model.blocks):
                        def inject_camera(
                            _module,
                            args,
                            output,
                            block_index=block_index,
                        ):
                            if not isinstance(output, Tensor) or not args or not isinstance(args[0], Tensor):
                                raise RuntimeError("Wan self-attention hook received an invalid input/output")
                            residual = camera_adapter.attention_residual(
                                block_index,
                                args[0],
                                camera_context,
                            )
                            if residual.shape != output.shape or not torch.isfinite(residual).all():
                                raise RuntimeError("full RRE residual must be finite and match self-attention output")
                            cast_residual = residual.to(dtype=output.dtype)
                            output_rms = output.detach().float().square().mean().sqrt()
                            residual_rms = cast_residual.detach().float().square().mean().sqrt()
                            stats = self.last_injection_stats.setdefault(block_index, {})
                            stats.update({
                                "attention_output_rms": output_rms,
                                "camera_residual_rms": residual_rms,
                                "camera_to_attention_rms": residual_rms / output_rms.clamp_min(1e-12),
                            })
                            return output + cast_residual
                        handles.append(block.self_attn.register_forward_hook(inject_camera))
                with autocast:
                    outputs = self.model(samples, timesteps.to(self.device), contexts, seq_len)
            finally:
                for handle in handles:
                    handle.remove()
        output = torch.stack(outputs, dim=0)
        if output.shape != latents.shape:
            raise RuntimeError(f"Wan DiT shape mismatch: expected {tuple(latents.shape)}, got {tuple(output.shape)}")
        return output.contiguous()
