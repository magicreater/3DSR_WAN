"""Reusable frozen FlashVSR LQ projector and a zero-initialized Wan bridge."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class Stage1Degradation:
    """Deterministic HR-to-LR construction for Stage 1 only."""

    scale: int = 4
    mode: str = "bicubic"
    antialias: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, int) or self.scale < 2:
            raise ValueError("scale must be an integer >= 2")
        if self.mode not in {"bicubic", "bilinear"}:
            raise ValueError("mode must be bicubic or bilinear")

    def __call__(self, hr: Tensor) -> Tensor:
        _validate_rgb(hr, "hr")
        b, c, positions, height, width = hr.shape
        if height % self.scale or width % self.scale:
            raise ValueError("HR height and width must be divisible by scale")
        flat = hr.permute(0, 2, 1, 3, 4).reshape(b * positions, c, height, width)
        lr = F.interpolate(
            flat,
            size=(height // self.scale, width // self.scale),
            mode=self.mode,
            align_corners=False,
            antialias=self.antialias,
        )
        return lr.reshape(b, positions, c, height // self.scale, width // self.scale).permute(0, 2, 1, 3, 4).contiguous()

    def conditioning_size(self, lr_size: tuple[int, int]) -> tuple[int, int]:
        return lr_size[0] * self.scale, lr_size[1] * self.scale


def derange_multiview_lr(lr: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Rotate raw LR views once so every target receives a different view."""

    _validate_rgb(lr, "lr")
    views = lr.shape[2]
    if views < 2:
        raise ValueError("multiview derangement requires at least two views")
    indices = tuple(range(1, views)) + (0,)
    return lr[:, :, indices].contiguous(), indices


def evenly_spaced_indices(total: int, count: int) -> tuple[int, ...]:
    """Select deterministic endpoints and floor-spaced interior observations."""

    if isinstance(total, bool) or isinstance(count, bool) or total < 1 or not 1 <= count <= total:
        raise ValueError("count must satisfy 1 <= count <= total")
    if count == 1:
        return (0,)
    return tuple(index * (total - 1) // (count - 1) for index in range(count))

class _RMSNorm(nn.Module):
    """FlashVSR-compatible channel-first RMS normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, value: Tensor) -> Tensor:
        return F.normalize(value, dim=1) * self.scale * self.gamma


class _CausalConv3d(nn.Conv3d):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._causal_padding = (
            self.padding[2], self.padding[2],
            self.padding[1], self.padding[1],
            2 * self.padding[0], 0,
        )
        self.padding = (0, 0, 0)

    def forward(self, value: Tensor, cache: Tensor | None = None) -> Tensor:
        padding = list(self._causal_padding)
        if cache is not None and padding[4] > 0:
            value = torch.cat([cache.to(value.device), value], dim=2)
            padding[4] -= cache.shape[2]
        return super().forward(F.pad(value, padding, mode="replicate"))


class _PixelUnshuffle16(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        batch, channels, frames, height, width = value.shape
        if height % 16 or width % 16:
            raise ValueError("FlashVSR LQ input H/W must be divisible by 16")
        value = value.reshape(batch, channels, frames, height // 16, 16, width // 16, 16)
        return value.permute(0, 1, 4, 6, 2, 3, 5).reshape(
            batch, channels * 16 * 16, frames, height // 16, width // 16
        )


class CausalLQ4xProjector(nn.Module):
    """Checkpoint-compatible one-layer FlashVSR ``Causal_LQ4x_Proj``.

    This local definition keeps Stage 1 runnable without importing the full
    FlashVSR repository. Parameter names and operations match its published
    projector, so the existing ``LQ_proj_in.ckpt`` loads strictly.
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 1536, layer_num: int = 1):
        super().__init__()
        self.layer_num = layer_num
        self.pixel_shuffle = _PixelUnshuffle16()
        self.conv1 = _CausalConv3d(in_dim * 16 * 16, 2048, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.norm1 = _RMSNorm(2048)
        self.act1 = nn.SiLU()
        self.conv2 = _CausalConv3d(2048, 3072, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.norm2 = _RMSNorm(3072)
        self.act2 = nn.SiLU()
        self.linear_layers = nn.ModuleList([nn.Linear(3072, out_dim) for _ in range(layer_num)])
        self.cache: dict[str, Tensor | None] = {}
        self.clear_cache()

    def clear_cache(self) -> None:
        self.cache = {"conv1": None, "conv2": None}

    def forward(self, video: Tensor) -> list[Tensor]:
        self.clear_cache()
        iterations = 1 + (video.shape[2] - 1) // 4
        first = video[:, :, :1].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first, video], dim=2)
        outputs = []
        for index in range(iterations):
            value = self.pixel_shuffle(video[:, :, index * 4 : (index + 1) * 4])
            cache1 = value[:, :, -2:].clone()
            value = self.conv1(value, self.cache["conv1"])
            self.cache["conv1"] = cache1
            value = self.act1(self.norm1(value))
            cache2 = value[:, :, -2:].clone()
            if index == 0:
                self.cache["conv2"] = cache2
                continue
            value = self.conv2(value, self.cache["conv2"])
            self.cache["conv2"] = cache2
            outputs.append(self.act2(self.norm2(value)))
        if not outputs:
            raise ValueError("LQ projector needs the Stage 1 warm-up prefix")
        value = torch.cat(outputs, dim=2)
        value = value.permute(0, 2, 3, 4, 1).reshape(value.shape[0], -1, value.shape[1])
        return [layer(value) for layer in self.linear_layers]

def _bridge_blocks(blocks) -> tuple[int, ...]:
    blocks = tuple(blocks)
    if not blocks or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in blocks):
        raise ValueError("bridge_blocks must contain non-negative integer block indices")
    if len(set(blocks)) != len(blocks):
        raise ValueError("bridge_blocks must not contain duplicates")
    return blocks


class _MultiBlockBridge(nn.Module):
    """Independent FP32 zero projections and optional identity-centered time gates.

    Features are [B,L,D], timesteps [B] in Wan units (1000 * sigma).
    Only this module is trainable; no temporal or view reshaping occurs here.
    """

    def __init__(self, dim: int, blocks: tuple[int, ...], time_conditioning: bool):
        super().__init__()
        self.projections = nn.ModuleDict({str(i): nn.Linear(dim, dim, dtype=torch.float32) for i in blocks})
        self.time_gates = nn.ModuleDict({str(i): nn.Linear(256, dim, dtype=torch.float32) for i in blocks} if time_conditioning else {})
        self.last_diagnostics: dict[int, dict[str, Tensor]] = {}
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.parameters():
            nn.init.zeros_(parameter)
        self.last_diagnostics = {}

    def forward(self, features: Tensor, timesteps: Tensor) -> dict[int, Tensor]:
        anchor = next(self.parameters())
        if anchor.dtype != torch.float32:
            raise ValueError("bridge parameters must remain FP32")
        if timesteps.shape != (features.shape[0],) or not torch.isfinite(timesteps).all():
            raise ValueError("timesteps must be finite [B] in Wan units (1000 * sigma)")
        # Match official Wan sinusoidal_embedding_1d: float64 phases, cos then sin.
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            features = features.to(device=anchor.device, dtype=torch.float32)
            embedding = None
            if self.time_gates:
                positions = timesteps.to(device=anchor.device, dtype=torch.float64)
                phases = torch.outer(positions, torch.pow(10000, -torch.arange(128, device=anchor.device, dtype=torch.float64) / 128))
                embedding = F.silu(torch.cat([phases.cos(), phases.sin()], dim=1).float())
            result = {}
            diagnostics = {}
            for key, projection in self.projections.items():
                gate = 1 + torch.tanh(self.time_gates[key](embedding)) if embedding is not None else torch.ones(features.shape[0], features.shape[-1], device=anchor.device)
                residual = gate[:, None, :] * projection(features)
                result[int(key)] = residual
                detached_gate = gate.detach()
                diagnostics[int(key)] = {"gate_min": detached_gate.amin(), "gate_max": detached_gate.amax(), "gate_mean": detached_gate.mean(), "residual_rms": residual.detach().square().mean().sqrt()}
            self.last_diagnostics = diagnostics
            return result


class FrozenLQConditioner(nn.Module):
    """Frozen causal LQ encoder plus the only trainable Stage 1 bridge.

    The FlashVSR projector receives RGB in ``[B,3,T,H,W]`` and returns Wan
    patch tokens ``[B,L,D]``. Four outer first-frame copies compensate for its
    warm-up chunk so positions align with Wan VAE's ``1 + (T-1)//4`` layout.
    """

    def __init__(self, projector: nn.Module, *, feature_dim: int = 1536, prefix_frames: int = 4,
                 bridge_blocks: tuple[int, ...] = (0,), bridge_time_conditioning: bool = False):
        super().__init__()
        if feature_dim < 1 or prefix_frames < 0:
            raise ValueError("feature_dim must be positive and prefix_frames non-negative")
        self.projector = projector.eval().requires_grad_(False)
        self.feature_dim = int(feature_dim)
        self.prefix_frames = int(prefix_frames)
        self.bridge_blocks = _bridge_blocks(bridge_blocks)
        if not isinstance(bridge_time_conditioning, bool):
            raise ValueError("bridge_time_conditioning must be boolean")
        self.bridge_time_conditioning = bridge_time_conditioning
        self.legacy_bridge = self.bridge_blocks == (0,) and not bridge_time_conditioning
        self.bridge = (nn.Linear(self.feature_dim, self.feature_dim, bias=True, dtype=torch.float32)
                       if self.legacy_bridge else _MultiBlockBridge(self.feature_dim, self.bridge_blocks, bridge_time_conditioning))
        self.reset_bridge()

    def reset_bridge(self) -> None:
        """Reset every trainable bridge parameter to its identity/no-op initialization."""
        for parameter in self.bridge.parameters():
            nn.init.zeros_(parameter)
        if not self.legacy_bridge:
            self.bridge.last_diagnostics = {}

    def bridge_residuals(self, features: Tensor, timesteps: Tensor, *, enabled: bool = True) -> dict[int, Tensor]:
        if not enabled:
            if not self.legacy_bridge:
                self.bridge.last_diagnostics = {}
            return {}
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must have shape [B,L,{self.feature_dim}]")
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError("features must be finite floating values")
        if self.legacy_bridge:
            return {0: self.bridge_tokens(features)}
        return self.bridge(features, timesteps)

    def train(self, mode: bool = True):
        super().train(mode)
        self.projector.eval()
        return self

    def bridge_tokens(self, features: Tensor, *, enabled: bool = True) -> Tensor:
        if not self.legacy_bridge:
            raise ValueError("multi-block/time-conditioned bridges require bridge_residuals(features, timesteps)")
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must have shape [B,L,{self.feature_dim}]")
        if not enabled:
            return torch.zeros_like(features, dtype=self.bridge.weight.dtype)
        return self.bridge(features.to(device=self.bridge.weight.device, dtype=self.bridge.weight.dtype))

    def multiview_features(
        self,
        lr: Tensor,
        *,
        conditioning_size: tuple[int, int],
        latent_shape: tuple[int, int, int],
    ) -> Tensor:
        """Encode views independently and regroup in Wan F/H/W token order."""

        _validate_rgb(lr, "multiview LR")
        batch, channels, views, height, width = lr.shape
        latent_views, latent_h, latent_w = latent_shape
        if views != latent_views:
            raise ValueError("LR views must match latent views")
        flat = lr.permute(0, 2, 1, 3, 4).reshape(batch * views, channels, 1, height, width)
        prepared = _resize_spatial(flat, conditioning_size)
        prepared = _prefix_first(prepared, self.prefix_frames)
        features = self._project(prepared)
        expected_spatial = (latent_h // 2) * (latent_w // 2)
        if features.shape != (batch * views, expected_spatial, self.feature_dim):
            raise RuntimeError(
                "LQ multiview tokens do not align with Wan patches: "
                f"expected {(batch * views, expected_spatial, self.feature_dim)}, got {tuple(features.shape)}"
            )
        return features.reshape(batch, views * expected_spatial, self.feature_dim).contiguous()

    def multiview_tokens(self, lr: Tensor, **kwargs: Any) -> Tensor:
        return self.bridge_tokens(self.multiview_features(lr, **kwargs))

    def video_features(
        self,
        lr: Tensor,
        *,
        conditioning_size: tuple[int, int],
        latent_shape: tuple[int, int, int],
    ) -> Tensor:
        """Encode native video without flattening its temporal dimension."""

        _validate_rgb(lr, "video LR")
        latent_frames, latent_h, latent_w = latent_shape
        prepared = _resize_spatial(lr, conditioning_size)
        prepared = _prefix_first(prepared, self.prefix_frames)
        features = self._project(prepared)
        expected = latent_frames * (latent_h // 2) * (latent_w // 2)
        if features.shape != (lr.shape[0], expected, self.feature_dim):
            raise RuntimeError(
                "LQ video tokens do not align with Wan patches: "
                f"expected {(lr.shape[0], expected, self.feature_dim)}, got {tuple(features.shape)}"
            )
        return features.contiguous()

    def video_tokens(self, lr: Tensor, **kwargs: Any) -> Tensor:
        return self.bridge_tokens(self.video_features(lr, **kwargs))

    def _project(self, prepared: Tensor) -> Tensor:
        try:
            anchor = next(self.projector.parameters())
            prepared = prepared.to(device=anchor.device, dtype=anchor.dtype)
        except StopIteration:
            pass
        with torch.no_grad():
            output = self.projector(prepared)
        if not isinstance(output, (list, tuple)) or len(output) != 1:
            raise RuntimeError("Stage 1 requires a one-layer FlashVSR LQ projector")
        features = output[0]
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise RuntimeError(f"projector must return [B,L,{self.feature_dim}] tokens")
        return features.detach()


def conditioned_prediction(
    dit,
    conditioner: FrozenLQConditioner,
    sample: Tensor,
    timestep: Tensor,
    context: Tensor | None,
    features: Tensor | None,
    *,
    geometry_adapter=None,
    camera=None,
    latent_shape: tuple[int, int, int] | None = None,
    geometry_enabled: bool = True,
) -> Tensor:
    """Run the shared Stage 1/2 path and sum residuals at matching blocks."""
    residuals: dict[int, Tensor] = {}
    camera_attention = None
    if features is not None:
        for block, residual in conditioner.bridge_residuals(features, timestep).items():
            residuals[block] = residual
    if geometry_adapter is not None and geometry_enabled:
        if camera is None or latent_shape is None:
            raise ValueError("camera and latent_shape are required for geometry conditioning")
        if getattr(geometry_adapter, "injection_mode", None) == "self_attention":
            camera_attention = (geometry_adapter, geometry_adapter.prepare(camera, latent_shape))
        else:
            for block, residual in geometry_adapter.residuals(camera, latent_shape, timestep).items():
                residuals[block] = residual if block not in residuals else residuals[block] + residual
    if not residuals and camera_attention is None:
        return dit(sample, timestep, context)
    return dit(
        sample,
        timestep,
        context,
        block_token_residuals=residuals or None,
        camera_attention=camera_attention,
    )


def _prefix_first(video: Tensor, count: int) -> Tensor:
    if count == 0:
        return video
    return torch.cat([video[:, :, :1].expand(-1, -1, count, -1, -1), video], dim=2).contiguous()


def _resize_spatial(video: Tensor, size: tuple[int, int]) -> Tensor:
    if len(size) != 2 or min(size) < 16 or size[0] % 16 or size[1] % 16:
        raise ValueError("conditioning size must be positive and divisible by 16")
    b, c, frames, height, width = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(b * frames, c, height, width)
    resized = F.interpolate(flat, size=size, mode="bicubic", align_corners=False, antialias=True)
    return resized.reshape(b, frames, c, *size).permute(0, 2, 1, 3, 4).contiguous()


def _validate_rgb(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 5 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B,3,S,H,W]")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating RGB")
    if min(value.shape) < 1:
        raise ValueError(f"{name} dimensions must be positive")


def load_flashvsr_projector(
    source_file: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    expected_sha256: str | None = None,
) -> nn.Module:
    """Load the FlashVSR checkpoint strictly into the local compatible projector."""

    source_file = Path(source_file).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not source_file.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("FlashVSR projector source/checkpoint is missing")
    if expected_sha256 is not None:
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(f"LQ projector SHA256 mismatch: {digest}")
    with torch.device("meta"):
        projector = CausalLQ4xProjector(in_dim=3, out_dim=1536, layer_num=1)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    projector.load_state_dict(state, strict=True, assign=True)
    projector.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return projector


def save_adapter_checkpoint(
    path: str | Path,
    conditioner: FrozenLQConditioner,
    *,
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> None:
    """Save only the small trainable bridge and reproducibility metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_config = dict(config)
    if not conditioner.legacy_bridge:
        supplied_blocks = saved_config.get("bridge_blocks", conditioner.bridge_blocks)
        supplied_time = saved_config.get("bridge_time_conditioning", conditioner.bridge_time_conditioning)
        if tuple(supplied_blocks) != conditioner.bridge_blocks or supplied_time != conditioner.bridge_time_conditioning:
            raise RuntimeError("adapter config architecture does not match conditioner")
        saved_config.update(bridge_blocks=list(conditioner.bridge_blocks),
                            bridge_time_conditioning=conditioner.bridge_time_conditioning)
    elif tuple(saved_config.get("bridge_blocks", (0,))) != (0,) or saved_config.get("bridge_time_conditioning", False) is not False:
        raise RuntimeError("adapter config architecture does not match conditioner")
    torch.save(
        {
            "format_version": 1 if conditioner.legacy_bridge else 2,
            "bridge": {name: value.detach().cpu() for name, value in conditioner.bridge.state_dict().items()},
            "config": saved_config,
            "experiment": dict(experiment),
        },
        path,
    )


def load_adapter_checkpoint(
    path: str | Path,
    conditioner: FrozenLQConditioner,
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_experiment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the bridge and optionally validate its runtime identity metadata."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    version = payload.get("format_version")
    if version not in (1, 2):
        raise RuntimeError("unsupported Stage 1 adapter checkpoint")
    config = payload["config"]
    experiment = payload["experiment"]
    try:
        blocks = _bridge_blocks(config.get("bridge_blocks", (0,)))
        time_conditioning = config.get("bridge_time_conditioning", False)
        if not isinstance(time_conditioning, bool):
            raise ValueError("bridge_time_conditioning must be boolean")
    except (TypeError, ValueError) as error:
        raise RuntimeError("invalid adapter architecture config") from error
    if version == 1 and (blocks != (0,) or time_conditioning):
        raise RuntimeError("v1 adapter architecture must be legacy block 0 without time gating")
    if version == 2 and not {"bridge_blocks", "bridge_time_conditioning"}.issubset(config):
        raise RuntimeError("v2 adapter architecture config is missing")
    if blocks != conditioner.bridge_blocks or time_conditioning != conditioner.bridge_time_conditioning:
        raise RuntimeError("adapter architecture mismatch with runtime conditioner")
    def normalized(value):
        result = dict(value)
        result["bridge_blocks"] = tuple(result.get("bridge_blocks", (0,)))
        result["bridge_time_conditioning"] = result.get("bridge_time_conditioning", False)
        result["stop_on_dev_pass"] = result.get("stop_on_dev_pass", False)
        return result
    if expected_config is not None and normalized(config) != normalized(expected_config):
        raise RuntimeError(f"adapter config mismatch: expected {dict(expected_config)!r}, got {config!r}")
    if expected_experiment is not None:
        mismatched = {
            key: (expected, experiment.get(key))
            for key, expected in expected_experiment.items()
            if experiment.get(key) != expected
        }
        if mismatched:
            raise RuntimeError(f"adapter experiment metadata mismatch: {mismatched!r}")
    conditioner.bridge.load_state_dict(payload["bridge"], strict=True)
    return {"config": config, "experiment": experiment}
