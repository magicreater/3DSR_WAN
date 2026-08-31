"""Camera conversions for the canonical RL3dSR data contract.

The canonical camera frame follows OpenCV axes: +X right, +Y down, and
+Z forward. Extrinsics are camera-to-world transforms.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray


_OPENGL_TO_OPENCV_CAMERA_BASIS = np.diag(
    np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
)


def build_pinhole_intrinsics(
    width: int,
    height: int,
    camera_angle_x: float,
) -> NDArray[np.float32]:
    """Build centered square-pixel intrinsics from a horizontal field of view."""

    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    if isinstance(camera_angle_x, bool):
        raise ValueError("camera_angle_x must be a finite number in (0, pi)")
    try:
        angle = float(camera_angle_x)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera_angle_x must be a finite number in (0, pi)") from exc
    if not math.isfinite(angle) or not 0.0 < angle < math.pi:
        raise ValueError(
            f"camera_angle_x must be a finite number in (0, pi), got {angle!r}"
        )

    focal = 0.5 * width / math.tan(0.5 * angle)
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def convert_nerf_synthetic_c2w(transform: Any) -> NDArray[np.float32]:
    """Convert Blender/OpenGL camera-to-world axes to canonical OpenCV axes.

    The world frame and camera origin are preserved. Only the local camera basis
    changes, by right-multiplying ``diag(1, -1, -1, 1)``.
    """

    try:
        source = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("transform_matrix must contain numeric values") from exc
    _validate_rigid_transform(source, name="transform_matrix")

    canonical = source @ _OPENGL_TO_OPENCV_CAMERA_BASIS
    _validate_rigid_transform(canonical, name="canonical transform")
    return canonical.astype(np.float32)


def convert_colmap_w2c(qvec: Any, tvec: Any) -> NDArray[np.float64]:
    """Invert COLMAP's scalar-first quaternion world-to-camera pose.

    ``qvec`` is ``[qw, qx, qy, qz]`` and ``tvec`` is ``[tx, ty, tz]``.
    The float64 ``[4, 4]`` result retains OpenCV camera axes and world units.
    The metadata contract performs the final conversion to read-only float32.
    """
    quaternion = np.asarray(qvec, dtype=np.float64)
    translation = np.asarray(tvec, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("qvec must be finite with shape (4,) in (qw, qx, qy, qz) order")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("tvec must be finite with shape (3,)")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, abs_tol=1e-4, rel_tol=0.0):
        raise ValueError(f"qvec must have unit norm within 1e-4, got {norm}")
    w, x, y, z = quaternion / norm
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)
    canonical = np.eye(4, dtype=np.float64)
    canonical[:3, :3] = rotation.T
    canonical[:3, 3] = -(rotation.T @ translation)
    _validate_rigid_transform(canonical, name="COLMAP camera-to-world transform")
    return canonical


def scale_intrinsics(
    K: Any,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> NDArray[np.float64]:
    """Scale pixel intrinsics by actual dimensions, without crop/half-pixel shift.

    Returns a new float64 ``[3, 3]`` array. Never modifies ``K``. Width and
    height are scaled independently to account for rounded image dimensions.
    """
    for name, value in (
        ("source_width", source_width), ("source_height", source_height),
        ("target_width", target_width), ("target_height", target_height),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    intrinsics = np.asarray(K, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("K must be finite with shape (3, 3)")
    if not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-5, rtol=0):
        raise ValueError("K must have homogeneous last row [0, 0, 1]")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("K focal lengths must be positive")
    scaled = np.diag([
        target_width / source_width, target_height / source_height, 1.0
    ]) @ intrinsics
    if not np.isfinite(scaled).all():
        raise ValueError("scaled K must contain only finite values")
    return scaled


def _validate_rigid_transform(matrix: NDArray[np.float64], *, name: str) -> None:
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")

    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        atol=1e-4,
        rtol=0.0,
    ):
        raise ValueError(f"{name} rotation must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=1e-4, rel_tol=0.0):
        raise ValueError(f"{name} rotation must have determinant +1, got {determinant}")
