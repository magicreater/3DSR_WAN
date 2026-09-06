import math

import torch

from rl3dsr.validation.geometry import (
    evaluate_cross_view_reconstruction,
    evaluate_pose_sensitivity,
    evaluate_reprojection,
    reproject_rgb,
)


def _camera(views=2, height=4, width=4):
    K = torch.eye(3).repeat(1, views, 1, 1)
    K[..., 0, 0] = 1
    K[..., 1, 1] = 1
    K[..., 0, 2] = 0
    K[..., 1, 2] = 0
    T = torch.eye(4).repeat(1, views, 1, 1)
    return K, T


def test_identity_reprojection_is_exact():
    K, T = _camera()
    source = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
    depth = torch.ones(4, 4)
    warped, mask, _ = reproject_rgb(source, depth, K[0, 0], T[0, 0], K[0, 1], T[0, 1])
    assert bool(mask.all())
    torch.testing.assert_close(warped, source)


def test_reprojection_rows_include_gt_ceiling_and_coverage():
    K, T = _camera()
    frame = torch.rand(1, 3, 1, 4, 4)
    rgb = frame.repeat(1, 1, 2, 1, 1)
    depth = torch.ones(1, 2, 4, 4)
    rows = evaluate_reprojection(rgb, rgb, depth, K, T)
    assert len(rows) == 2
    assert all(row["coverage"] == 1.0 for row in rows)
    assert all(row["photometric_mae"] == 0.0 for row in rows)
    assert all(row["gt_ceiling_mae"] == 0.0 for row in rows)
    assert all(row["normalized_error"] == 0.0 for row in rows)


def test_cross_view_reconstruction_excludes_target_view():
    K, T = _camera(views=3)
    rgb = torch.zeros(1, 3, 3, 4, 4)
    rgb[:, :, 0] = 0.1
    rgb[:, :, 1] = 0.2
    rgb[:, :, 2] = 0.3
    depth = torch.ones(1, 3, 4, 4)
    rows = evaluate_cross_view_reconstruction(rgb, rgb, depth, K, T)
    assert [row["target_index"] for row in rows] == [0, 1, 2]
    assert all(row["source_count"] == 2 for row in rows)
    assert all(row["photometric_mae"] > 0 for row in rows)


def test_pose_sensitivity_reports_repeat_jitter_ratio():
    correct = torch.zeros(1, 3, 2, 2, 2)
    repeat = correct + 0.001
    perturbed = correct + 0.1
    shuffled = correct + 0.2
    disabled = correct + 0.3
    result = evaluate_pose_sensitivity(correct, shuffled, disabled, perturbed, repeat=repeat)
    assert result["pose_response_ratio"] > 90
    assert result["correct_disabled_delta"] > result["correct_shuffled_delta"]


def test_invalid_depth_is_rejected():
    K, T = _camera()
    rgb = torch.rand(1, 3, 2, 4, 4)
    with torch.no_grad():
        bad_depth = torch.ones(1, 2, 3, 4)
    try:
        evaluate_reprojection(rgb, rgb, bad_depth, K, T)
    except ValueError as error:
        assert "depth" in str(error)
    else:
        raise AssertionError("invalid depth shape was accepted")
