from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from rl3dsr.models.wan.dit import WanDiT
from rl3dsr.models.wan.lq_conditioning import FrozenLQConditioner


class FakeBlock(nn.Module):
    def forward(self, x, **kwargs):
        return x


class TinyWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.dim = 6
        self.blocks = nn.ModuleList([FakeBlock()])

    def forward(self, samples, timesteps, contexts, seq_len):
        batch = len(samples)
        x = torch.zeros(batch, seq_len, self.dim, device=samples[0].device, dtype=samples[0].dtype)
        x = self.blocks[0](x)
        values = x.mean(dim=-1)
        outputs = []
        for index, sample in enumerate(samples):
            channels, frames, height, width = sample.shape
            patch_values = values[index].reshape(frames, height // 2, width // 2)
            output = patch_values.repeat_interleave(2, 1).repeat_interleave(2, 2)
            outputs.append(output.unsqueeze(0).expand(channels, -1, -1, -1).contiguous())
        return outputs


class ZeroProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, video):
        temporal = (video.shape[2] - 1) // 4
        spatial = (video.shape[-2] // 16) * (video.shape[-1] // 16)
        return [torch.ones(video.shape[0], temporal * spatial, 6, device=video.device)]


def test_zero_conditioned_forward_is_exact_baseline_and_nonzero_changes_prediction():
    official = TinyWan().eval().requires_grad_(False)
    dit = WanDiT(official, device="cpu")
    latents = torch.randn(1, 16, 2, 8, 8)
    timestep = torch.tensor([500.0])
    context = torch.zeros(1, 512, 4096)
    baseline = dit(latents, timestep, context)
    zero = torch.zeros(1, 2 * 4 * 4, 6)
    conditioned_zero = dit(latents, timestep, context, token_residual=zero)
    changed = dit(latents, timestep, context, token_residual=torch.ones_like(zero))
    assert torch.equal(baseline, conditioned_zero)
    assert (changed - baseline).abs().max() > 0


def test_conditioned_forward_backpropagates_only_to_bridge():
    official = TinyWan().eval().requires_grad_(False)
    dit = WanDiT(official, device="cpu")
    conditioner = FrozenLQConditioner(ZeroProjector(), feature_dim=6, prefix_frames=4)
    nn.init.eye_(conditioner.bridge.weight)
    lr = torch.randn(1, 3, 5, 16, 16)
    residual = conditioner.video_tokens(lr, conditioning_size=(64, 64), latent_shape=(2, 8, 8))
    prediction = dit(
        torch.randn(1, 16, 2, 8, 8),
        torch.tensor([500.0]),
        torch.zeros(1, 512, 4096),
        token_residual=residual,
    )
    prediction.square().mean().backward()
    assert conditioner.bridge.weight.grad is not None
    assert all(parameter.grad is None for parameter in official.parameters())
    assert all(parameter.grad is None for parameter in conditioner.projector.parameters())