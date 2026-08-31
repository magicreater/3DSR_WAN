"""Analytical camera tests independent of binary fixture generation."""

import numpy as np
import pytest

from rl3dsr.camera import convert_colmap_w2c, scale_intrinsics


def test_colmap_inverse_preserves_opencv_axes_and_world_units():
    # 90 degrees about Z: a nontrivial inverse with a known camera center.
    q = np.array([2**-0.5, 0.0, 0.0, 2**-0.5])
    t = np.array([2.0, 3.0, 4.0])
    original_q, original_t = q.copy(), t.copy()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected = np.array(
        [[0.0, 1.0, 0.0, -3.0], [-1.0, 0.0, 0.0, 2.0],
         [0.0, 0.0, 1.0, -4.0], [0.0, 0.0, 0.0, 1.0]]
    )
    canonical = convert_colmap_w2c(q, t)
    np.testing.assert_allclose(canonical, expected, atol=1e-12)
    np.testing.assert_allclose(convert_colmap_w2c(-q, t), expected, atol=1e-12)
    np.testing.assert_array_equal(q, original_q)
    np.testing.assert_array_equal(t, original_t)
    assert canonical.dtype == np.float64

    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = rotation, t
    np.testing.assert_allclose(w2c @ canonical, np.eye(4), atol=1e-12)
    world = np.array([5.0, 7.0, 9.0, 1.0])
    np.testing.assert_allclose(canonical @ (w2c @ world), world, atol=1e-12)
    # Identity COLMAP pose must stay identity, not acquire the NeRF Y/Z flip.
    np.testing.assert_array_equal(convert_colmap_w2c([1, 0, 0, 0], [0, 0, 0]), np.eye(4))


def test_near_unit_quaternion_is_normalized_without_modifying_input():
    q = np.array([1.0 + 5e-5, 0.0, 0.0, 0.0])
    before = q.copy()
    np.testing.assert_array_equal(convert_colmap_w2c(q, [0, 0, 0]), np.eye(4))
    np.testing.assert_array_equal(q, before)


@pytest.mark.parametrize("q,t", [
    ([0, 0, 0, 0], [0, 0, 0]),
    ([2, 0, 0, 0], [0, 0, 0]),
    ([1.001, 0, 0, 0], [0, 0, 0]),
    ([1, 0, 0], [0, 0, 0]),
    ([[1, 0, 0, 0]], [0, 0, 0]),
    ([np.nan, 0, 0, 0], [0, 0, 0]),
    ([1, 0, 0, 0], [0, 0]),
    ([1, 0, 0, 0], [0, np.inf, 0]),
])
def test_invalid_colmap_pose_is_rejected(q, t):
    with pytest.raises(ValueError):
        convert_colmap_w2c(q, t)


def test_intrinsics_scale_actual_width_and_height_and_preserve_projection():
    source = np.array([[100.0, 2.0, 5.5], [0.0, 120.0, 3.5], [0.0, 0.0, 1.0]])
    before = source.copy()
    source.setflags(write=False)
    scaled = scale_intrinsics(
        source, source_width=11, source_height=7, target_width=6, target_height=3
    )
    expected = np.diag([6 / 11, 3 / 7, 1.0]) @ source
    np.testing.assert_allclose(scaled, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(source, before)
    assert not np.shares_memory(source, scaled)
    assert scaled.dtype == np.float64
    ray = np.array([0.3, -0.2, 1.0])
    np.testing.assert_allclose(scaled @ ray, (source @ ray) * [6 / 11, 3 / 7, 1])
    recovered = scale_intrinsics(
        scaled, source_width=6, source_height=3, target_width=11, target_height=7
    )
    np.testing.assert_allclose(recovered, source)


@pytest.mark.parametrize("key,value", [
    ("source_width", 0), ("source_height", -1), ("target_width", True),
    ("target_height", 2.5), ("source_width", float("inf")),
])
def test_scale_rejects_invalid_dimensions(key, value):
    dimensions = dict(source_width=11, source_height=7, target_width=6, target_height=3)
    dimensions[key] = value
    with pytest.raises(ValueError):
        scale_intrinsics(np.eye(3), **dimensions)


@pytest.mark.parametrize("matrix", [
    np.eye(4), [[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 1, 0], [0, 1, 1]],
])
def test_scale_rejects_invalid_intrinsics(matrix):
    with pytest.raises(ValueError):
        scale_intrinsics(
            matrix, source_width=11, source_height=7, target_width=6, target_height=3
        )
