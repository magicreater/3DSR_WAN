"""Opt-in real Wan smoke: WAN_MODEL_DIR and one free CUDA GPU are required."""

import os

import pytest
import torch

from rl3dsr.models.wan.dit import WanDiT
from rl3dsr.models.wan.wan_lora import inject_wan_lora


@pytest.mark.wan
def test_real_wan_lora_is_noop_then_backpropagates():
    model_dir = os.environ.get("WAN_MODEL_DIR")
    if not model_dir or not torch.cuda.is_available():
        pytest.skip("WAN_MODEL_DIR and CUDA are required")
    dit = WanDiT.from_checkpoint(model_dir, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(7)
    latents = torch.randn(1, 16, 2, 4, 4, device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device="cuda")
    with torch.no_grad():
        before = dit(latents, timestep)
    parameters = inject_wan_lora(dit.model)
    with torch.no_grad():
        after = dit(latents, timestep)
    assert torch.equal(before, after)
    dit(latents, timestep).float().square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in parameters)
    assert all(parameter.grad is None for name, parameter in dit.model.named_parameters()
               if "lora_" not in name)
