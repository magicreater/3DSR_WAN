"""Small CPU checks for the optional Wan LoRA path."""

import pytest
import torch
from torch import nn

pytest.importorskip("peft")

from rl3dsr.models.wan.stage3 import (
    load_stage3_checkpoint, load_stage3_initialization_checkpoint,
    save_stage3_checkpoint,
)
from rl3dsr.models.wan.wan_lora import (
    inject_wan_lora, wan_lora_parameters, wan_lora_state,
)
from test_stage3_conditioning import bundle


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16)
        self.k = nn.Linear(16, 16)
        self.v = nn.Linear(16, 16)
        self.o = nn.Linear(16, 16)

    def forward(self, x):
        return self.o(self.q(x) + self.k(x) + self.v(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()

    def forward(self, x):
        return x + self.self_attn(x)


class TinyWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(16))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@pytest.mark.parametrize("blocks", [(0, 1, 2, 3), (12, 13, 14, 15)])
def test_selected_lora_is_initially_noop_and_only_it_gets_wan_gradients(blocks):
    model = TinyWan().eval().requires_grad_(False)
    x = torch.randn(2, 3, 16)
    before = model(x).detach()
    parameters = inject_wan_lora(model, blocks)
    assert len(parameters) == 32
    assert {int(name.split(".")[1]) for name, _ in model.named_parameters()
            if "lora_" in name} == set(blocks)
    assert torch.equal(model(x).detach(), before)
    assert not any(parameter.requires_grad for name, parameter in model.named_parameters()
                   if "lora_" not in name)
    model(x).square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in parameters)
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if "lora_" not in name)
    with pytest.raises(ValueError, match="already installed"):
        inject_wan_lora(model)


@pytest.mark.parametrize("blocks", [(0, 1, 2, 3), (12, 13, 14, 15)])
def test_lora_checkpoint_roundtrip_and_legacy_parent(tmp_path, blocks):
    source = TinyWan().eval().requires_grad_(False)
    inject_wan_lora(source, blocks)
    with torch.no_grad():
        for parameter in wan_lora_parameters(source):
            parameter.add_(0.01)
    module = bundle()
    config = {"arm": "A5", "wan_lora": True, "wan_lora_learning_rate": 1e-5}
    if blocks[0] == 12:
        config["wan_lora_blocks"] = list(blocks)
    path = tmp_path / "lora.pt"
    save_stage3_checkpoint(path, module, config=config, step=3, provenance={"wan_checkpoint_sha256": "abc"},
                           wan_model=source)
    target = TinyWan().eval().requires_grad_(False)
    inject_wan_lora(target, blocks)
    loaded = load_stage3_checkpoint(path, bundle(), expected_config=config, wan_model=target)
    assert loaded["format_version"] == 2
    assert all(torch.equal(value, wan_lora_state(target)[name])
               for name, value in wan_lora_state(source).items())
    wrong = TinyWan().eval().requires_grad_(False)
    inject_wan_lora(wrong, (12, 13, 14, 15) if blocks[0] == 0 else (0, 1, 2, 3))
    with pytest.raises(ValueError, match="architecture mismatch"):
        load_stage3_checkpoint(path, bundle(), expected_config=config, wan_model=wrong)
    legacy = tmp_path / "legacy.pt"
    save_stage3_checkpoint(legacy, module, config={"arm": "A5"}, step=1, provenance={})
    fresh = TinyWan().eval().requires_grad_(False)
    inject_wan_lora(fresh, blocks)
    initial = wan_lora_state(fresh)
    load_stage3_initialization_checkpoint(legacy, bundle(), wan_model=fresh)
    assert all(torch.equal(value, wan_lora_state(fresh)[name]) for name, value in initial.items())


def test_bad_lora_initialization_does_not_change_adapters(tmp_path):
    source = TinyWan().eval().requires_grad_(False)
    inject_wan_lora(source)
    path = tmp_path / "bad.pt"
    save_stage3_checkpoint(path, bundle(), config={"arm": "A5", "wan_lora": True},
                           step=1, provenance={}, wan_model=source)
    payload = torch.load(path, weights_only=True)
    payload["wan_lora"]["state"].pop(next(iter(payload["wan_lora"]["state"])))
    torch.save(payload, path)
    target = bundle()
    before = {key: value.clone() for key, value in target.conditioner.bridge.state_dict().items()}
    with pytest.raises(ValueError, match="32 matrices"):
        load_stage3_initialization_checkpoint(path, target, reset_fusion=True, wan_model=source)
    assert all(torch.equal(value, target.conditioner.bridge.state_dict()[key])
               for key, value in before.items())
