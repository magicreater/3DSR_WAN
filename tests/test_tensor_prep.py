import numpy as np
import torch

from rl3dsr.data.tensor_prep import rgb_images_to_video


def test_rgb_images_to_video_uses_channel_first_temporal_contract():
    images = [np.full((8, 6, 3), value, dtype=np.uint8) for value in (0, 255, 128)]
    result = rgb_images_to_video(images, 8)
    assert result.shape == (1, 3, 3, 8, 8)
    assert result.dtype == torch.float32
    assert float(result[0, :, 0].mean()) == -1.0
    assert float(result[0, :, 1].mean()) == 1.0
