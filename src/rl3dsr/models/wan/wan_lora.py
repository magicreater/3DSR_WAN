"""Optional A5-only LoRA on the first four Wan self-attention blocks.

The pretrained Wan weights stay frozen.  This changes no input, camera, or
3D/4D tensor semantics; disabled configurations never import PEFT.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor, nn


LORA_CONFIG = {
    "blocks": [0, 1, 2, 3],
    "projections": ["q", "k", "v", "o"],
    "rank": 8,
    "alpha": 8,
    "dropout": 0.0,
    "bias": "none",
}
_TARGET = re.compile(r"blocks\.[0-3]\.self_attn\.[qkvo]")
_PARAMETER = re.compile(
    r"blocks\.([0-3])\.self_attn\.([qkvo])\.lora_([AB])\.default\.weight"
)


def inject_wan_lora(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Inject zero-output LoRA into exactly 16 Wan ``nn.Linear`` projections."""
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise ValueError("Wan LoRA is already installed")
    targets = [(name, module) for name, module in model.named_modules() if _TARGET.fullmatch(name)]
    if len(targets) != 16 or any(not isinstance(module, nn.Linear) for _, module in targets):
        raise ValueError("Wan LoRA requires q/k/v/o linear layers in blocks 0-3")
    from peft import LoraConfig, inject_adapter_in_model

    inject_adapter_in_model(
        LoraConfig(
            r=8, lora_alpha=8, target_modules=_TARGET.pattern,
            lora_dropout=0.0, bias="none", init_lora_weights=True,
        ), model,
    )
    parameters = wan_lora_parameters(model)
    if len(parameters) != 32:
        raise RuntimeError("Wan LoRA injection did not create the expected 32 matrices")
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name):
        raise RuntimeError("Wan pretrained weights must remain frozen")
    return parameters


def wan_lora_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    return tuple(parameter for name, parameter in model.named_parameters() if _PARAMETER.fullmatch(name))


def wan_lora_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if _PARAMETER.fullmatch(name)
    }


def validate_wan_lora_state(state: object, model: nn.Module | None = None) -> None:
    if not isinstance(state, dict) or len(state) != 32:
        raise ValueError("Wan LoRA state must contain 32 matrices")
    actual = {}
    for name, value in state.items():
        match = _PARAMETER.fullmatch(name) if isinstance(name, str) else None
        if match is None or not isinstance(value, Tensor) or value.ndim != 2 or not torch.isfinite(value).all():
            raise ValueError("Wan LoRA state has malformed or nonfinite parameters")
        actual[(int(match[1]), match[2], match[3])] = value
    expected = {(block, projection, side) for block in range(4)
                for projection in "qkvo" for side in "AB"}
    if set(actual) != expected:
        raise ValueError("Wan LoRA state has missing or unexpected parameters")
    for block in range(4):
        for projection in "qkvo":
            a = actual[(block, projection, "A")]
            b = actual[(block, projection, "B")]
            if a.shape[0] != 8 or b.shape[1] != 8 or a.shape[1] != b.shape[0]:
                raise ValueError("Wan LoRA state has incompatible matrix shapes")
    if model is not None:
        current = {name: parameter for name, parameter in model.named_parameters()
                   if _PARAMETER.fullmatch(name)}
        if current.keys() != state.keys() or any(
            current[name].shape != value.shape or current[name].dtype != value.dtype
            for name, value in state.items()
        ):
            raise ValueError("Wan LoRA state does not match the loaded backbone")


def load_wan_lora_state(model: nn.Module, state: object) -> None:
    validate_wan_lora_state(state, model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name].to(device=parameter.device))
