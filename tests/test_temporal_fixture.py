import numpy as np
import pytest

from rl3dsr.data.temporal_fixture import TemporalVideoSample, make_synthetic_video


@pytest.mark.parametrize("frame_count", [1, 5, 9, 17])
def test_synthetic_video_has_explicit_temporal_shapes(frame_count):
    sample = make_synthetic_video(frame_count, seed=123)
    assert sample.scene_id == "synthetic_temporal"
    assert sample.frames.shape == (frame_count, 64, 64, 3)
    assert sample.intrinsics.shape == (frame_count, 3, 3)
    assert sample.extrinsics.shape == (frame_count, 4, 4)
    assert sample.timestamps.shape == (frame_count,)
    assert np.all(np.diff(sample.timestamps) > 0) if frame_count > 1 else True
    assert not sample.frames.flags.writeable


def test_synthetic_video_is_reproducible():
    first = make_synthetic_video(5, seed=9)
    second = make_synthetic_video(5, seed=9)
    np.testing.assert_array_equal(first.frames, second.frames)
    np.testing.assert_array_equal(first.intrinsics, second.intrinsics)
    np.testing.assert_array_equal(first.extrinsics, second.extrinsics)


def test_temporal_video_rejects_misaligned_metadata():
    sample = make_synthetic_video(3)
    with pytest.raises(ValueError, match="intrinsics"):
        TemporalVideoSample(sample.scene_id, sample.frames, sample.intrinsics[:2], sample.extrinsics, sample.timestamps)
