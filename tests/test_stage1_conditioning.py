from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from rl3dsr.models.wan.flow import flow_matching_pair, flow_matching_loss
from rl3dsr.models.wan.lq_conditioning import (
    FrozenLQConditioner,
    Stage1Degradation,
    derange_multiview_lr,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


class FakeProjector(nn.Module):
    """Cheap stand-in with the FlashVSR projector's list-of-token output."""

    def __init__(self, dim: int = 6):
        super().__init__()
        self.proj = nn.Linear(3, dim, bias=False)

    def forward(self, video):
        temporal = (video.shape[2] - 1) // 4
        spatial_h = video.shape[-2] // 16
        spatial_w = video.shape[-1] // 16
        frames = video[:, :, 4 : 4 + 4 * temporal : 4]
        pooled = torch.nn.functional.adaptive_avg_pool3d(frames, (temporal, spatial_h, spatial_w))
        tokens = pooled.permute(0, 2, 3, 4, 1).reshape(video.shape[0], -1, 3)
        return [self.proj(tokens)]


def test_stage1_degradation_is_deterministic_and_separate_storage():
    config = Stage1Degradation(scale=4, mode="bicubic", antialias=True)
    hr = torch.linspace(-1, 1, 2 * 3 * 5 * 64 * 80).reshape(2, 3, 5, 64, 80)
    lr_a = config(hr)
    lr_b = config(hr)
    assert lr_a.shape == (2, 3, 5, 16, 20)
    assert torch.equal(lr_a, lr_b)
    assert lr_a.untyped_storage().data_ptr() != hr.untyped_storage().data_ptr()
    assert config.conditioning_size((16, 20)) == (64, 80)


def test_zero_bridge_preserves_baseline_and_only_bridge_is_trainable():
    conditioner = FrozenLQConditioner(FakeProjector(), feature_dim=6, prefix_frames=4)
    assert all(not parameter.requires_grad for parameter in conditioner.projector.parameters())
    assert {name for name, p in conditioner.named_parameters() if p.requires_grad} == {
        "bridge.weight",
        "bridge.bias",
    }
    features = torch.randn(2, 12, 6)
    residual = conditioner.bridge_tokens(features)
    assert torch.count_nonzero(residual) == 0


def test_multiview_alignment_keeps_views_independent_and_matches_wan_tokens():
    conditioner = FrozenLQConditioner(FakeProjector(), feature_dim=6, prefix_frames=4)
    lr = torch.randn(1, 3, 4, 16, 16)
    tokens = conditioner.multiview_tokens(lr, conditioning_size=(64, 64), latent_shape=(4, 8, 8))
    assert tokens.shape == (1, 4 * 4 * 4, 6)
    changed = lr.clone()
    changed[:, :, 2] += 1
    changed_tokens = conditioner.multiview_features(
        changed, conditioning_size=(64, 64), latent_shape=(4, 8, 8)
    ).reshape(1, 4, 16, 6)
    base_tokens = conditioner.multiview_features(
        lr, conditioning_size=(64, 64), latent_shape=(4, 8, 8)
    ).reshape(1, 4, 16, 6)
    assert (changed_tokens[:, 2] - base_tokens[:, 2]).abs().max() > 0
    assert torch.equal(changed_tokens[:, [0, 1, 3]], base_tokens[:, [0, 1, 3]])


def test_multiview_derangement_rotates_raw_lr_without_fixed_views():
    lr = torch.arange(4).reshape(1, 1, 4, 1, 1).expand(1, 3, 4, 2, 2).float()
    shuffled, indices = derange_multiview_lr(lr)
    assert indices == (1, 2, 3, 0)
    assert torch.equal(shuffled[:, :, 0], lr[:, :, 1])
    assert torch.equal(shuffled[:, :, 3], lr[:, :, 0])
    assert all(source != target for target, source in enumerate(indices))


def test_multiview_derangement_requires_multiple_views():
    with pytest.raises(ValueError, match="at least two"):
        derange_multiview_lr(torch.zeros(1, 3, 1, 8, 8))


def test_video_alignment_matches_native_wan_latent_positions():
    conditioner = FrozenLQConditioner(FakeProjector(), feature_dim=6, prefix_frames=4)
    for frames, latent_frames in ((1, 1), (5, 2), (9, 3), (17, 5)):
        lr = torch.randn(1, 3, frames, 16, 16)
        features = conditioner.video_features(
            lr, conditioning_size=(64, 64), latent_shape=(latent_frames, 8, 8)
        )
        assert features.shape == (1, latent_frames * 4 * 4, 6)


def test_bridge_receives_gradient_while_projector_stays_frozen():
    conditioner = FrozenLQConditioner(FakeProjector(), feature_dim=6, prefix_frames=4)
    nn.init.eye_(conditioner.bridge.weight)
    lr = torch.randn(1, 3, 5, 16, 16)
    loss = conditioner.video_tokens(
        lr, conditioning_size=(64, 64), latent_shape=(2, 8, 8)
    ).square().mean()
    loss.backward()
    assert conditioner.bridge.weight.grad is not None
    assert torch.isfinite(conditioner.bridge.weight.grad).all()
    assert all(parameter.grad is None for parameter in conditioner.projector.parameters())


def test_flow_pair_and_target_match_wan_training_formulation():
    clean = torch.tensor([[[[[2.0]]]]])
    noise = torch.tensor([[[[[-1.0]]]]])
    sigma = torch.tensor([0.25])
    noisy, timestep, target = flow_matching_pair(clean, noise, sigma)
    assert torch.equal(noisy, torch.tensor([[[[[1.25]]]]]))
    assert torch.equal(timestep, torch.tensor([250.0]))
    assert torch.equal(target, torch.tensor([[[[[-3.0]]]]]))
    assert flow_matching_loss(target, target).item() == 0.0


def test_adapter_checkpoint_roundtrip_excludes_frozen_projector(tmp_path):
    conditioner = FrozenLQConditioner(FakeProjector(), feature_dim=6, prefix_frames=4)
    nn.init.normal_(conditioner.bridge.weight)
    nn.init.normal_(conditioner.bridge.bias)
    before = copy.deepcopy(conditioner.bridge.state_dict())
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(
        path,
        conditioner,
        config={"scale": 4},
        experiment={"seed": 42, "kind": "unit"},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {"format_version", "bridge", "config", "experiment"}
    assert all("projector" not in key for key in payload["bridge"])
    nn.init.zeros_(conditioner.bridge.weight)
    nn.init.zeros_(conditioner.bridge.bias)
    metadata = load_adapter_checkpoint(path, conditioner)
    assert metadata["config"] == {"scale": 4}
    for name, value in conditioner.bridge.state_dict().items():
        assert torch.equal(value, before[name])
class DtypeTrackingProjector(FakeProjector):
    def __init__(self):
        super().__init__()
        self.last_dtype = None

    def forward(self, video):
        self.last_dtype = video.dtype
        return super().forward(video)


def test_projector_boundary_casts_lr_to_frozen_projector_dtype():
    projector = DtypeTrackingProjector().to(dtype=torch.bfloat16)
    conditioner = FrozenLQConditioner(projector, feature_dim=6, prefix_frames=4)
    lr = torch.randn(1, 3, 1, 16, 16, dtype=torch.float32)
    conditioner.video_features(lr, conditioning_size=(64, 64), latent_shape=(1, 8, 8))
    assert projector.last_dtype == torch.bfloat16

def test_local_flashvsr_projector_matches_checkpoint_parameter_contract():
    from rl3dsr.models.wan.lq_conditioning import CausalLQ4xProjector

    with torch.device("meta"):
        projector = CausalLQ4xProjector(in_dim=3, out_dim=1536, layer_num=1)
    assert sum(parameter.numel() for parameter in projector.parameters()) == 287_845_888
    assert set(projector.state_dict()) == {
        "conv1.weight", "conv1.bias", "norm1.gamma",
        "conv2.weight", "conv2.bias", "norm2.gamma",
        "linear_layers.0.weight", "linear_layers.0.bias",
    }

def test_adapter_checkpoint_rejects_mismatched_expected_metadata(tmp_path):
    conditioner = FrozenLQConditioner(FakeProjector(), feature_dim=6, prefix_frames=4)
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(
        path,
        conditioner,
        config={"scale": 4, "kind": "3d"},
        experiment={"model_checkpoint": "wan-A", "lq_sha256": "abc"},
    )
    with pytest.raises(RuntimeError, match="config mismatch"):
        load_adapter_checkpoint(path, conditioner, expected_config={"scale": 2, "kind": "3d"})
    with pytest.raises(RuntimeError, match="experiment metadata mismatch"):
        load_adapter_checkpoint(path, conditioner, expected_experiment={"model_checkpoint": "wan-B"})
