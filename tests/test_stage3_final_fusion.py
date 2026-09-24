"""Small CPU checks for the opt-in final Stage 3 candidate."""

import pytest
import torch
from torch import nn

from rl3dsr.models.wan.dit import WanDiT
from rl3dsr.models.wan.geometry_conditioning import CameraBatch
from rl3dsr.models.wan.lr_fusion import LRViewFusion
from rl3dsr.validation.stage3_protocol import Stage3Config


def _camera(views):
    k = torch.eye(3).repeat(1, views, 1, 1)
    k[..., 0, 0] = k[..., 1, 1] = 28
    k[..., 0, 2] = k[..., 1, 2] = 16
    pose = torch.eye(4).repeat(1, views, 1, 1)
    pose[0, :, 0, 3] = torch.arange(views) * 0.2
    return CameraBatch(k, pose, (32, 32), "multiview")


@pytest.mark.parametrize("views", [4, 8])
def test_dynamic_fusion_query_null_and_auxiliary_order(views):
    torch.manual_seed(12)
    fusion = LRViewFusion(feature_dim=8, hidden_dim=192, heads=1,
                          mode="rre_epipolar", allow_self_view_source=False,
                          dynamic=True, query_chunk_size=4)
    features = torch.randn(1, views, 4, 8)
    features[:, 0] = 0  # target LR dropout still leaves a noisy-latent query
    latent = torch.randn(1, views, 4, 16)
    camera = _camera(views)
    first = fusion(features, camera, (2, 2), latent_query=latent,
                   timestep=torch.tensor([500.0]))
    assert torch.equal(first, features)  # zero-init residual
    with torch.no_grad():
        fusion.output.weight.normal_(std=0.01)
    first = fusion(features, camera, (2, 2), latent_query=latent,
                   timestep=torch.tensor([500.0]))
    changed = latent.clone()
    changed[:, 0] += 2
    second = fusion(features, camera, (2, 2), latent_query=changed,
                    timestep=torch.tensor([500.0]))
    assert not torch.allclose(first[:, 0], second[:, 0])
    empty = fusion(features, camera, (2, 2), latent_query=latent,
                   timestep=torch.tensor([500.0]),
                   source_mask=torch.zeros(1, views, dtype=torch.bool))
    assert torch.allclose(empty, features, atol=1e-6)
    permutation = torch.tensor([0, *range(2, views), 1])
    permuted_camera = CameraBatch(
        camera.K[:, permutation], camera.T_world_from_camera[:, permutation],
        camera.image_size, "multiview",
    )
    reordered = fusion(features[:, permutation], permuted_camera, (2, 2),
                       latent_query=latent[:, permutation], timestep=torch.tensor([500.0]))
    assert torch.allclose(first, reordered[:, torch.argsort(permutation)], atol=2e-5)
    first.square().mean().backward()
    assert fusion.output.weight.grad is not None
    assert fusion.latent_query.weight.grad is not None


def test_shared_rope_restores_on_failure_and_config_defaults():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.blocks = nn.ModuleList([nn.Identity()])
            self.dim, self.num_heads = 12, 1
            self.register_buffer("freqs", torch.arange(30.).reshape(5, 6))

        def forward(self, samples, timestep, contexts, seq_len):
            assert torch.equal(self.freqs[:, :2], self.freqs[:1, :2].expand(5, -1))
            raise RuntimeError("planned failure")

    model = Model()
    original = model.freqs
    dit = WanDiT(model, device="cpu")
    with pytest.raises(RuntimeError, match="planned failure"):
        dit(torch.zeros(1, 16, 3, 4, 4), torch.ones(1), shared_view_rope=True)
    assert model.freqs is original
    assert Stage3Config().dynamic_fusion is False
    assert Stage3Config().shared_multiview_rope is False
    with pytest.raises(ValueError, match="require A6"):
        Stage3Config(arm="A5", dynamic_fusion=True)
