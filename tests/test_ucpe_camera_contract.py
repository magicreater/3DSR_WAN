from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl3dsr.models.wan.geometry_conditioning import (
    CameraBatch,
    FullRREConditioner,
    build_camera_rays,
    build_world_to_ray,
)


def _camera(*, dtype: torch.dtype = torch.float32) -> CameraBatch:
    intrinsics = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    ).reshape(1, 1, 3, 3)
    transforms = torch.eye(4, dtype=dtype).reshape(1, 1, 4, 4)
    return CameraBatch(intrinsics, transforms, (2, 2), "multiview")


def test_pinhole_patch_rays_match_hand_derived_literals():
    actual = build_camera_rays(_camera(), (2, 2))
    expected = torch.tensor(
        [
            [-0.4082483, -0.4082483, 0.8164966],
            [0.4082483, -0.4082483, 0.8164966],
            [-0.4082483, 0.4082483, 0.8164966],
            [0.4082483, 0.4082483, 0.8164966],
        ]
    ).reshape(1, 1, 4, 3)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_ucm_xi_zero_is_the_pinhole_limit():
    pinhole = _camera()
    ucm = replace(pinhole, camera_model="ucm", xi=torch.zeros(1, 1))
    assert torch.allclose(
        build_camera_rays(ucm, (2, 2)),
        build_camera_rays(pinhole, (2, 2)),
        atol=1e-6,
        rtol=0,
    )


def test_ucm_nonzero_xi_changes_off_axis_ray_by_the_literal_inverse_model():
    intrinsics = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    camera = CameraBatch(
        intrinsics,
        torch.eye(4).reshape(1, 1, 4, 4),
        (1, 2),
        "multiview",
        camera_model="ucm",
        xi=torch.tensor([[0.5]]),
    )
    actual = build_camera_rays(camera, (1, 2))
    expected = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.9114378, 0.0, 0.4114378]]]
    ).reshape(1, 1, 2, 3)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("camera_model", "xi", "message"),
    [
        ("ucm", None, "xi is required"),
        ("ucm", 0.5, "xi must be a tensor"),
        ("ucm", torch.zeros(1), "xi must have shape"),
        ("ucm", torch.zeros(1, 1, dtype=torch.int64), "xi must be floating point"),
        ("ucm", torch.tensor([[float("nan")]]), "xi must be finite"),
        ("ucm", torch.tensor([[-0.1]]), "xi must be non-negative"),
        ("pinhole", torch.zeros(1, 1), "xi must be omitted"),
        ("fisheye", None, "camera_model must be"),
    ],
)
def test_camera_model_specific_xi_validation(camera_model, xi, message):
    with pytest.raises(ValueError, match=message):
        replace(_camera(), camera_model=camera_model, xi=xi).validate()


def test_ucm_accepts_official_wide_angle_xi_above_one():
    replace(
        _camera(), camera_model="ucm", xi=torch.tensor([[1.5]])
    ).validate()


def test_world_to_ray_matches_literal_center_ray_transform():
    intrinsics = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    transform = torch.eye(4).reshape(1, 1, 4, 4)
    transform[..., :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    context = build_world_to_ray(
        CameraBatch(intrinsics, transform, (1, 1), "multiview"),
        (1, 1),
    )
    expected = torch.tensor(
        [[1.0, 0.0, 0.0, -1.0],
         [0.0, 1.0, 0.0, -2.0],
         [0.0, 0.0, 1.0, -3.0],
         [0.0, 0.0, 0.0, 1.0]]
    )
    assert torch.allclose(context.world_to_ray[0, 0], expected, atol=1e-6, rtol=0)


def test_camera_context_preserves_bfloat16_storage_dtype():
    camera = _camera(dtype=torch.bfloat16)
    rays = build_camera_rays(camera, (2, 2))
    context = build_world_to_ray(camera, (2, 2))
    assert rays.dtype == torch.bfloat16
    assert context.world_to_ray.dtype == torch.bfloat16
    assert context.absmap.dtype == torch.bfloat16
    assert torch.isfinite(rays).all()
    assert torch.isfinite(context.world_to_ray).all()
    assert torch.isfinite(context.absmap).all()


def test_full_rre_disabled_is_noop_even_with_nonzero_output_weights():
    module = FullRREConditioner(
        feature_dim=32,
        hidden_dim=8,
        attention_heads=1,
        branch_count=1,
        compression=4,
    )
    with torch.no_grad():
        module.branches[0].output.weight.fill_(0.25)
        module.branches[0].output.bias.fill_(1.0)
    tokens = torch.randn(1, 1, 32)
    context = build_world_to_ray(
        replace(_camera(), image_size=(1, 1), K=torch.eye(3).reshape(1, 1, 3, 3)),
        (1, 1),
    )
    module.last_diagnostics[0] = {"stale": torch.tensor(1.0)}
    assert torch.equal(
        module.attention_residual(0, tokens, context, enabled=False),
        torch.zeros_like(tokens),
    )
    assert module.last_diagnostics == {}
