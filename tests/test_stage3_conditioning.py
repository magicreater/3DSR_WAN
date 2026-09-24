"""Synthetic CPU checks only: no Wan checkpoint, dataset or sampler is loaded."""
import pytest
import torch
from torch import nn

from rl3dsr.models.wan.lq_conditioning import FrozenLQConditioner, save_adapter_checkpoint, load_adapter_checkpoint
from rl3dsr.models.wan.geometry_conditioning import CameraBatch
from rl3dsr.models.wan.stage3 import (
    Stage3Conditioning,
    load_stage3_checkpoint,
    load_stage3_initialization_checkpoint,
    save_stage3_checkpoint,
)


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, rgb):
        self.calls += 1
        return [rgb.mean((1, 2, 3, 4))[:, None, None].expand(-1, 4, 8)]


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, features, camera, patch_grid, *, source_mask=None,
                allow_self_view_source=None):
        self.calls += 1
        self.allow_self_view_source = allow_self_view_source
        return features + self.offset


def bundle(fusion=True):
    lq = FrozenLQConditioner(Projector(), feature_dim=8)
    with torch.no_grad():
        lq.bridge.weight.copy_(torch.eye(8))
    return Stage3Conditioning(lq, geometry=None, fusion=Fusion() if fusion else None)


def camera(kind="multiview"):
    return CameraBatch(torch.eye(3).repeat(1, 2, 1, 1), torch.eye(4).repeat(1, 2, 1, 1), (32, 32), kind)


def test_preparation_runs_fusion_once_and_backpropagates_through_frozen_encoder():
    module = bundle()
    lr = torch.rand(1, 3, 2, 8, 8)
    prepared = module.prepare_multiview(lr, camera(), (2, 4, 4), (32, 32))
    assert prepared.shape == (1, 8, 8)
    sample = torch.zeros(1, 1, 2, 2, 2)
    def dit(x, t, context, **kw):
        return kw["block_token_residuals"][0].sum(-1).reshape_as(x)
    outputs = [module.predict(dit, sample, torch.ones(1), None, prepared, camera(), (2, 4, 4)) for _ in range(3)]
    sum(out.sum() for out in outputs).backward()
    assert module.fusion.calls == 1
    assert module.conditioner.projector.calls == 1
    assert module.fusion.offset.grad.abs() > 0
    assert module.conditioner.bridge.weight.grad is not None
    assert module.conditioner.projector.anchor.grad is None


def test_legacy_bridge_can_load_and_fusion_disabled_is_exact(tmp_path):
    original = bundle(False)
    path = tmp_path / "legacy.pt"
    save_adapter_checkpoint(path, original.conditioner, config={}, experiment={})
    restored = bundle(False)
    load_adapter_checkpoint(path, restored.conditioner)
    lr = torch.rand(1, 3, 2, 8, 8)
    direct = original.conditioner.multiview_features(lr, conditioning_size=(32, 32), latent_shape=(2, 4, 4))
    assert torch.equal(restored.prepare_multiview(lr, camera(), (2, 4, 4), (32, 32)), direct)


def test_checkpoint_roundtrip_and_mismatch_rejected_before_loading(tmp_path):
    module = bundle()
    path = tmp_path / "stage3.pt"
    save_stage3_checkpoint(path, module, config={"arm": "A3"}, step=8, provenance={"parent_sha256": "abc"}, training_state={"rng": torch.arange(4)})
    target = bundle()
    with torch.no_grad():
        target.fusion.offset.fill_(17)
    with pytest.raises(ValueError, match="config"):
        load_stage3_checkpoint(path, target, expected_config={"arm": "A2"})
    assert target.fusion.offset.item() == 17
    payload = load_stage3_checkpoint(path, target, expected_config={"arm": "A3"})
    assert payload["step"] == 8
    assert torch.equal(payload["training_state"]["rng"], torch.arange(4))
    assert torch.equal(target.fusion.offset, module.fusion.offset)
    with pytest.raises(FileExistsError):
        save_stage3_checkpoint(path, module, config={}, step=9, provenance={})


def test_model_only_initialization_permits_config_change_and_validates_exact_state(tmp_path):
    parent = bundle()
    path = tmp_path / "parent.pt"
    save_stage3_checkpoint(
        path,
        parent,
        config={"arm": "A3", "camera_rank_weight": 0.1},
        step=1000,
        provenance={},
        training_state={"optimizer": {"poison": True}, "torch_rng_state": torch.tensor([255])},
    )
    target = bundle()
    with torch.no_grad():
        target.fusion.offset.fill_(17)
    payload = load_stage3_initialization_checkpoint(path, target)
    assert payload["step"] == 1000
    assert payload["config"]["arm"] == "A3"
    assert torch.equal(target.fusion.offset, parent.fusion.offset)

    poisoned = torch.load(path, map_location="cpu", weights_only=True)
    poisoned["adapters"]["unexpected"] = None
    bad = tmp_path / "extra.pt"
    torch.save(poisoned, bad)
    untouched = bundle()
    with torch.no_grad():
        untouched.fusion.offset.fill_(19)
    with pytest.raises(ValueError, match="adapter set"):
        load_stage3_initialization_checkpoint(bad, untouched)
    assert untouched.fusion.offset.item() == 19

    poisoned = torch.load(path, map_location="cpu", weights_only=True)
    poisoned["adapters"]["fusion"]["offset"] = torch.tensor(float("nan"))
    bad = tmp_path / "nonfinite.pt"
    torch.save(poisoned, bad)
    with pytest.raises(ValueError, match="nonfinite"):
        load_stage3_initialization_checkpoint(bad, bundle())


def test_model_only_initialization_can_explicitly_reset_fusion_only(tmp_path):
    parent = bundle()
    parent.geometry = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        parent.conditioner.bridge.weight.fill_(2)
        parent.geometry.weight.fill_(3)
        parent.fusion.offset.fill_(4)
    path = tmp_path / "a3-parent.pt"
    save_stage3_checkpoint(path, parent, config={"arm": "A3"}, step=1000, provenance={})

    target = bundle()
    target.geometry = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        target.conditioner.bridge.weight.fill_(7)
        target.geometry.weight.fill_(8)
        target.fusion.offset.fill_(0)
    load_stage3_initialization_checkpoint(path, target, reset_fusion=True)
    assert torch.all(target.conditioner.bridge.weight == 2)
    assert torch.all(target.geometry.weight == 3)
    assert target.fusion.offset.item() == 0


def test_old_checkpoint_config_defaults_new_dropout_field(tmp_path):
    module = bundle()
    path = tmp_path / "stage3-old.pt"
    save_stage3_checkpoint(path, module, config={"arm": "A3"}, step=8, provenance={})
    target = bundle()
    payload = load_stage3_checkpoint(
        path, target, expected_config={
            "arm": "A3", "target_lr_dropout": 0.0, "allow_self_view_source": True,
            "camera_rank_weight": 0.0, "camera_rank_margin_ratio": 0.05,
            "symmetric_camera_fraction": 0.5,
        }
    )
    assert payload["config"] == {"arm": "A3"}


def test_prepare_multiview_propagates_self_source_override():
    module = bundle()
    module.prepare_multiview(
        torch.rand(1, 3, 2, 8, 8), camera(), (2, 4, 4), (32, 32),
        allow_self_view_source=False,
    )
    assert module.fusion.allow_self_view_source is False


def test_multiview_entry_rejects_temporal_instead_of_flattening_it():
    with pytest.raises(ValueError, match="multiview"):
        bundle().prepare_multiview(torch.rand(1, 3, 2, 8, 8), camera("temporal"), (2, 4, 4), (32, 32))
