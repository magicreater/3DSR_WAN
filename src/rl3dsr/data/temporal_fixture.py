"""Deterministic temporal samples used by the Stage 0 native-video path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TemporalVideoSample:
    """A small temporal sample with per-frame camera metadata.

    ``frames`` is ``uint8[T,H,W,3]``. ``intrinsics`` and ``extrinsics`` are
    ``float32[T,3,3]`` and ``float32[T,4,4]`` respectively. Timestamps are
    strictly increasing and expressed in arbitrary fixture time units.
    """

    scene_id: str
    frames: NDArray[np.uint8]
    intrinsics: NDArray[np.float32]
    extrinsics: NDArray[np.float32]
    timestamps: NDArray[np.float64]

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames, dtype=np.uint8)
        intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        extrinsics = np.asarray(self.extrinsics, dtype=np.float32)
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must be non-empty")
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape [T,H,W,3]")
        t, h, w, _ = frames.shape
        if t < 1 or h < 1 or w < 1:
            raise ValueError("frames dimensions must be positive")
        if intrinsics.shape != (t, 3, 3):
            raise ValueError("intrinsics must have shape [T,3,3]")
        if extrinsics.shape != (t, 4, 4):
            raise ValueError("extrinsics must have shape [T,4,4]")
        if timestamps.shape != (t,) or not np.isfinite(timestamps).all():
            raise ValueError("timestamps must have shape [T] and be finite")
        if t > 1 and not np.all(np.diff(timestamps) > 0):
            raise ValueError("timestamps must be strictly increasing")
        if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
            raise ValueError("camera metadata must be finite")
        for value, name, shape in (
            (frames, "frames", frames.shape),
            (intrinsics, "intrinsics", intrinsics.shape),
            (extrinsics, "extrinsics", extrinsics.shape),
            (timestamps, "timestamps", timestamps.shape),
        ):
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def make_synthetic_video(
    frame_count: int,
    *,
    height: int = 64,
    width: int = 64,
    seed: int = 0,
) -> TemporalVideoSample:
    """Create a deterministic RGB video and aligned moving-camera metadata."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(frame_count, height, width, 3), dtype=np.uint8)
    focal = float(width)
    intrinsics = np.repeat(
        np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)[None],
        frame_count,
        axis=0,
    )
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    extrinsics[:, 0, 3] = np.arange(frame_count, dtype=np.float32) * 0.01
    timestamps = np.arange(frame_count, dtype=np.float64) / 16.0
    return TemporalVideoSample("synthetic_temporal", frames, intrinsics, extrinsics, timestamps)
