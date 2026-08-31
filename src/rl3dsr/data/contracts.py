"""Framework-independent metadata contracts for RL3dSR sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class SequenceKind(str, Enum):
    """Semantic meaning of the sequence axis."""

    MULTIVIEW = "multiview"
    TEMPORAL = "temporal"


class Split(str, Enum):
    """Supported dataset splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class AlphaBackground(str, Enum):
    """Deterministic background used to composite RGBA observations."""

    WHITE = "white"
    BLACK = "black"


class DatasetFormatError(ValueError):
    """Raised when dataset metadata violates the canonical contract."""


Float32Matrix = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ObservationMetadata:
    """Metadata for one image observation.

    ``K`` has shape ``[3, 3]``. ``T_world_from_camera`` has shape ``[4, 4]``
    and maps canonical OpenCV camera coordinates into the source world frame.
    """

    observation_id: str
    frame_index: int
    rgb_path: Path
    width: int
    height: int
    K: Float32Matrix
    T_world_from_camera: Float32Matrix

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise DatasetFormatError("observation_id must be a non-empty string")
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise DatasetFormatError("frame_index must be a non-negative integer")

        path = Path(self.rgb_path)
        if not path.is_absolute():
            raise DatasetFormatError(f"rgb_path must be absolute, got {path}")
        object.__setattr__(self, "rgb_path", path)

        for name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DatasetFormatError(f"{name} must be a positive integer")

        intrinsics = _readonly_matrix(self.K, shape=(3, 3), name="K")
        if not np.allclose(
            intrinsics[2], [0.0, 0.0, 1.0], atol=1e-5, rtol=0.0
        ):
            raise DatasetFormatError("K must have homogeneous last row [0, 0, 1]")
        if not math.isfinite(float(intrinsics[0, 0])) or intrinsics[0, 0] <= 0:
            raise DatasetFormatError("K fx must be positive")
        if not math.isfinite(float(intrinsics[1, 1])) or intrinsics[1, 1] <= 0:
            raise DatasetFormatError("K fy must be positive")

        transform = _readonly_matrix(
            self.T_world_from_camera,
            shape=(4, 4),
            name="T_world_from_camera",
        )
        _validate_canonical_transform(transform)

        object.__setattr__(self, "K", intrinsics)
        object.__setattr__(self, "T_world_from_camera", transform)


@dataclass(frozen=True, slots=True)
class SequenceMetadata:
    """An ordered observation sequence with explicit multiview/temporal meaning."""

    scene_id: str
    split: Split
    sequence_kind: SequenceKind
    observations: tuple[ObservationMetadata, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise DatasetFormatError("scene_id must be a non-empty string")
        try:
            split = Split(self.split)
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(f"unsupported split: {self.split!r}") from exc
        try:
            sequence_kind = SequenceKind(self.sequence_kind)
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(
                f"unsupported sequence_kind: {self.sequence_kind!r}"
            ) from exc

        observations = tuple(self.observations)
        if not observations:
            raise DatasetFormatError("a sequence must contain at least one observation")
        if not all(isinstance(item, ObservationMetadata) for item in observations):
            raise DatasetFormatError(
                "observations must contain only ObservationMetadata values"
            )

        observation_ids = [item.observation_id for item in observations]
        frame_indices = [item.frame_index for item in observations]
        rgb_paths = [item.rgb_path for item in observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise DatasetFormatError("observation_id values must be unique")
        if len(set(frame_indices)) != len(frame_indices):
            raise DatasetFormatError("frame_index values must be unique")
        if len(set(rgb_paths)) != len(rgb_paths):
            raise DatasetFormatError("rgb_path values must be unique")

        object.__setattr__(self, "split", split)
        object.__setattr__(self, "sequence_kind", sequence_kind)
        object.__setattr__(self, "observations", observations)


def _readonly_matrix(value: Any, *, shape: tuple[int, int], name: str) -> Float32Matrix:
    try:
        matrix = np.array(value, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise DatasetFormatError(f"{name} must contain numeric values") from exc
    if matrix.shape != shape:
        raise DatasetFormatError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise DatasetFormatError(f"{name} must contain only finite values")
    matrix.setflags(write=False)
    return matrix


def _validate_canonical_transform(transform: Float32Matrix) -> None:
    if not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4, rtol=0.0
    ):
        raise DatasetFormatError(
            "T_world_from_camera must have homogeneous last row [0, 0, 0, 1]"
        )
    rotation = transform[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float32),
        atol=1e-4,
        rtol=0.0,
    ):
        raise DatasetFormatError("T_world_from_camera rotation must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=1e-4, rel_tol=0.0):
        raise DatasetFormatError(
            "T_world_from_camera rotation must have determinant +1, "
            f"got {determinant}"
        )

