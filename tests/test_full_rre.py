from __future__ import annotations

import pytest
import torch
from torch import nn

from rl3dsr.models.wan.dit import WanDiT
from rl3dsr.models.wan.geometry_conditioning import (
    CameraBatch,
    FullRREConditioner,
    FullRREContext,
    GeometryConditioner,
    _apply_prope,
    _invert_se3,
    _rope_coefficients,
    build_world_to_ray,
    load_geometry_checkpoint,
    save_geometry_checkpoint,
)


def _camera(sequence: int = 2) -> CameraBatch:
    intrinsics = torch.eye(3).repeat(1, sequence, 1, 1)
    intrinsics[..., 0, 0] = 16
    intrinsics[..., 1, 1] = 16
    intrinsics[..., 0, 2] = 8
    intrinsics[..., 1, 2] = 8
    transforms = torch.eye(4).repeat(1, sequence, 1, 1)
    transforms[..., 0, 3] = torch.arange(sequence)
    return CameraBatch(intrinsics, transforms, (16, 16), "multiview")


def _reference_projective(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(value)
    for batch in range(value.shape[0]):
        for head in range(value.shape[1]):
            for token in range(value.shape[2]):
                groups = value[batch, head, token].reshape(-1, 4)
                result[batch, head, token] = (matrix[batch, token] @ groups.T).T.reshape(-1)
    return result


def test_prope_projective_block_matches_reference():
    torch.manual_seed(3)
    value = torch.randn(1, 1, 4, 32)
    matrix = torch.eye(4).repeat(1, 4, 1, 1)
    matrix[:, :, 0, 3] = torch.arange(4)
    coeff_x = _rope_coefficients(torch.tensor([0, 1, 0, 1]), 8, value.dtype)
    coeff_y = _rope_coefficients(torch.tensor([0, 0, 1, 1]), 8, value.dtype)
    actual = _apply_prope(value, matrix, coeff_x, coeff_y)
    expected_first = _reference_projective(value[..., :16], matrix)
    assert torch.allclose(actual[..., :16], expected_first, atol=1e-6, rtol=1e-6)


def test_relative_ray_pairwise_transform_is_world_translation_invariant():
    camera = _camera()
    first = build_world_to_ray(camera, (2, 2)).world_to_ray
    translated = camera.T_world_from_camera.clone()
    translated[..., :3, 3] += torch.tensor([4.0, -2.0, 7.0])
    second = build_world_to_ray(
        CameraBatch(camera.K, translated, camera.image_size, camera.sequence_kind),
        (2, 2),
    ).world_to_ray
    first_relative = first[:, :, None] @ _invert_se3(first[:, None, :])
    second_relative = second[:, :, None] @ _invert_se3(second[:, None, :])
    assert torch.allclose(first_relative, second_relative, atol=2e-5, rtol=2e-5)


def test_nerf_world_up_is_encoded_as_up_and_latitude():
    camera = _camera(sequence=1)
    transforms = camera.T_world_from_camera.clone()
    transforms[..., :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    )
    context = build_world_to_ray(
        CameraBatch(camera.K, transforms, camera.image_size, camera.sequence_kind),
        (1, 1),
        world_up=(0.0, 0.0, 1.0),
    )
    assert torch.allclose(context.absmap[0, 0, :2], torch.tensor([0.0, -1.0]), atol=1e-6)
    assert abs(float(context.absmap[0, 0, 2])) < 1e-6


def test_full_rre_zero_init_and_branch_gradient_isolation():
    module = FullRREConditioner(
        feature_dim=32,
        hidden_dim=8,
        attention_heads=1,
        branch_count=30,
        compression=4,
    )
    context = FullRREContext(
        torch.eye(4).repeat(1, 4, 1, 1),
        torch.zeros(1, 4, 3),
        (2, 2),
    )
    tokens = torch.randn(1, 4, 32)
    assert torch.equal(module.attention_residual(7, tokens, context), torch.zeros_like(tokens))
    with torch.no_grad():
        module.branches[7].output.weight.normal_(std=1e-3)
    module.attention_residual(7, tokens, context).square().mean().backward()
    assert any(parameter.grad is not None for parameter in module.branches[7].parameters())
    assert all(parameter.grad is None for parameter in module.branches[6].parameters())
    assert all(parameter.grad is None for parameter in module.branches[8].parameters())


def test_full_rre_default_parameter_count_matches_ucpe_compression():
    module = FullRREConditioner()
    assert sum(parameter.numel() for parameter in module.parameters()) == 35_637_120


class _SelfAttention(nn.Module):
    def __init__(self, fail: bool = False):
        super().__init__()
        self.fail = fail

    def forward(self, value, *_args):
        if self.fail:
            raise RuntimeError("expected failure")
        return value


class _Block(nn.Module):
    def __init__(self, fail: bool = False):
        super().__init__()
        self.self_attn = _SelfAttention(fail)


class _Model(nn.Module):
    dim = 32

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.blocks = nn.ModuleList(_Block(index == 4) for index in range(30))

    def forward(self, samples, _timesteps, _contexts, seq_len):
        value = torch.zeros(len(samples), seq_len, self.dim)
        for block in self.blocks:
            value = block.self_attn(value, None, None, None)
        return samples


def test_wan_full_rre_hooks_are_removed_after_failure():
    model = _Model()
    dit = WanDiT(model, device="cpu")
    adapter = FullRREConditioner(
        feature_dim=32,
        hidden_dim=8,
        attention_heads=1,
        branch_count=30,
        compression=4,
    )
    context = FullRREContext(torch.eye(4).reshape(1, 1, 4, 4), torch.zeros(1, 1, 3), (1, 1))
    with pytest.raises(RuntimeError, match="expected failure"):
        dit(
            torch.zeros(1, 16, 1, 2, 2),
            torch.tensor([500.0]),
            camera_attention=(adapter, context),
        )
    assert all(not block.self_attn._forward_hooks for block in model.blocks)


def test_geometry_checkpoint_v1_and_v2_roundtrip(tmp_path):
    parent = tmp_path / "parent.pt"
    torch.save({"parent": True}, parent)
    legacy = GeometryConditioner(hidden_dim=16, attention_heads=4)
    legacy_path = tmp_path / "legacy.pt"
    save_geometry_checkpoint(
        legacy_path,
        legacy,
        config={"representation": "rre", "blocks": [0, 1, 2, 3]},
        parent_checkpoint=parent,
        step=5,
    )
    legacy_copy = GeometryConditioner(hidden_dim=16, attention_heads=4)
    assert load_geometry_checkpoint(legacy_path, legacy_copy)["format_version"] == 1

    full = FullRREConditioner(feature_dim=32, hidden_dim=8, branch_count=2, compression=4)
    full_path = tmp_path / "full.pt"
    save_geometry_checkpoint(
        full_path,
        full,
        config={"representation": "rre_full"},
        parent_checkpoint=parent,
        step=9,
    )
    full_copy = FullRREConditioner(feature_dim=32, hidden_dim=8, branch_count=2, compression=4)
    metadata = load_geometry_checkpoint(full_path, full_copy)
    assert metadata["format_version"] == 2
    assert metadata["step"] == 9
    assert all(torch.equal(a, b) for a, b in zip(full.parameters(), full_copy.parameters()))
