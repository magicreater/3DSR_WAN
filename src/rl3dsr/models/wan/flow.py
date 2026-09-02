"""Minimal Wan flow-matching training semantics used by Stage 1."""

from __future__ import annotations

import torch
from torch import Tensor


def flow_matching_pair(clean: Tensor, noise: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return Wan's noisy latent, 0..1000 timestep, and velocity target.

    Wan/FlowMatch uses ``x_sigma=(1-sigma)*x_0+sigma*epsilon`` and predicts
    ``epsilon-x_0``. ``sigma`` is one scalar per batch item.
    """

    if clean.shape != noise.shape or clean.ndim < 2:
        raise ValueError("clean and noise must have the same batched shape")
    if sigma.shape != (clean.shape[0],):
        raise ValueError(f"sigma must have shape [{clean.shape[0]}]")
    if not torch.is_floating_point(clean) or not torch.is_floating_point(noise):
        raise ValueError("clean and noise must be floating tensors")
    sigma = sigma.to(device=clean.device, dtype=clean.dtype)
    if not torch.isfinite(sigma).all() or bool((sigma < 0).any()) or bool((sigma > 1).any()):
        raise ValueError("sigma must be finite and in [0,1]")
    broadcast = sigma.reshape(sigma.shape[0], *([1] * (clean.ndim - 1)))
    noisy = (1.0 - broadcast) * clean + broadcast * noise
    timestep = sigma.float() * 1000.0
    target = noise - clean
    return noisy, timestep, target


def flow_matching_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """FP32 mean-squared velocity loss."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    return torch.nn.functional.mse_loss(prediction.float(), target.float())