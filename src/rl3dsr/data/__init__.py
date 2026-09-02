"""Dataset contracts and adapters."""

from rl3dsr.data.contracts import (
    AlphaBackground,
    DatasetFormatError,
    ObservationMetadata,
    SequenceKind,
    SequenceMetadata,
    Split,
)
from rl3dsr.data.mipnerf360 import MipNeRF360Adapter
from rl3dsr.data.nerf_synthetic import NeRFSyntheticAdapter
from rl3dsr.data.temporal_fixture import TemporalVideoSample, make_motion_video, make_synthetic_video

__all__ = [
    "AlphaBackground",
    "DatasetFormatError",
    "MipNeRF360Adapter",
    "NeRFSyntheticAdapter",
    "ObservationMetadata",
    "SequenceKind",
    "SequenceMetadata",
    "Split",
    "TemporalVideoSample",
    "make_motion_video",
    "make_synthetic_video",
]