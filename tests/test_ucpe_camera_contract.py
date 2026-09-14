from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl3dsr.models.wan.geometry_conditioning import (
    CameraBatch,
    FullRREConditioner,
    build_camera_rays,
    build_world_to_ray,
    project_camera_directions,
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


def test_ucm_projection_round_trips_patch_centers_and_reports_validity():
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
    pixels, denominator = project_camera_directions(camera, build_camera_rays(camera, (1, 2)))
    assert torch.allclose(
        pixels,
        torch.tensor([[[[0.5, 0.5], [1.5, 0.5]]]]),
        atol=1e-6,
        rtol=0,
    )
    assert torch.isfinite(denominator).all()
    assert torch.all(denominator.abs() > 1e-8)


def test_ucm_projection_masks_its_model_denominator_singularity():
    camera = replace(
        _camera(), camera_model="ucm", xi=torch.ones(1, 1)
    )
    pixels, denominator = project_camera_directions(
        camera, torch.tensor([[[[0.0, 0.0, -1.0]]]])
    )
    assert torch.equal(denominator, torch.zeros_like(denominator))
    assert torch.equal(pixels, torch.zeros_like(pixels))


def test_projection_invalidates_finite_denominator_when_calibration_overflows_pixels():
    camera = _camera()
    intrinsics = camera.K.clone()
    intrinsics[..., 0, 0] = torch.finfo(torch.float32).max
    camera = replace(camera, K=intrinsics)
    pixels, denominator = project_camera_directions(
        camera, torch.tensor([[[[2.0, 0.0, 1.0]]]])
    )
    assert torch.equal(pixels, torch.zeros_like(pixels))
    assert torch.equal(denominator, torch.zeros_like(denominator))


def test_world_to_ray_uses_ucm_denominator_for_lat_up_projection():
    intrinsics = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    camera = CameraBatch(
        intrinsics,
        torch.eye(4).reshape(1, 1, 4, 4),
        (1, 2),
        "multiview",
        camera_model="ucm",
        xi=torch.ones(1, 1),
    )
    context = build_world_to_ray(camera, (1, 2), world_up=(0.0, 1.0, 0.0))
    assert torch.allclose(
        context.absmap[0, 1, :2],
        torch.tensor([-0.0499792, 0.9987503]),
        atol=1e-5,
        rtol=0,
    )


def test_ucm_out_of_inverse_domain_is_finitely_masked_from_context():
    intrinsics = torch.tensor(
        [[0.25, 0.0, 0.5], [0.0, 0.25, 0.5], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    camera = CameraBatch(
        intrinsics,
        torch.eye(4).reshape(1, 1, 4, 4),
        (1, 2),
        "multiview",
        camera_model="ucm",
        xi=torch.tensor([[2.0]]),
    )
    rays, valid = build_camera_rays(camera, (1, 2), return_validity=True)
    context = build_world_to_ray(camera, (1, 2))
    assert torch.equal(valid, torch.tensor([[[True, False]]]))
    assert torch.equal(rays[0, 0, 1], torch.zeros(3))
    assert torch.equal(context.world_to_ray[0, 1], torch.eye(4))
    assert torch.equal(context.absmap[0, 1], torch.zeros(3))
    assert torch.isfinite(context.world_to_ray).all()
    assert torch.isfinite(context.absmap).all()


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


def test_ucm_camera_rays_and_context_are_finite_in_bfloat16():
    camera = replace(
        _camera(dtype=torch.bfloat16),
        camera_model="ucm",
        xi=torch.tensor([[0.5]], dtype=torch.bfloat16),
    )
    rays, valid = build_camera_rays(camera, (2, 2), return_validity=True)
    context = build_world_to_ray(camera, (2, 2))
    assert rays.dtype == torch.bfloat16
    assert context.world_to_ray.dtype == torch.bfloat16
    assert context.absmap.dtype == torch.bfloat16
    assert valid.all()
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


def test_public_wan_package_exports_ucpe_contract_surface():
    from rl3dsr.models import wan

    assert wan.build_camera_rays is build_camera_rays
    assert wan.project_camera_directions is project_camera_directions
    assert callable(wan.convert_official_ucpe_checkpoint)
