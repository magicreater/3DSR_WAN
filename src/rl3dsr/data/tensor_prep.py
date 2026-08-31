"""Small deterministic RGB-to-tensor helpers for Stage 0 smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def resize_rgb(rgb: np.ndarray, resolution: int) -> np.ndarray:
    """Resize ``uint8[H,W,3]`` to a square deterministic RGB image."""

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have dtype uint8 and shape [H,W,3]")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 8:
        raise ValueError("resolution must be an integer >= 8")
    image = Image.fromarray(np.ascontiguousarray(rgb))
    return np.asarray(image.resize((resolution, resolution), Image.Resampling.BICUBIC), dtype=np.uint8).copy()


def rgb_images_to_video(images: list[np.ndarray], resolution: int) -> Tensor:
    """Convert ordered RGB images to float ``[1,3,T,H,W]`` in ``[-1,1]``."""

    if not images:
        raise ValueError("images must not be empty")
    resized = np.stack([resize_rgb(image, resolution) for image in images], axis=0)
    return torch.from_numpy(resized).permute(3, 0, 1, 2).unsqueeze(0).float().div(127.5).sub(1.0)


def load_observations(adapter, sequence, resolution: int) -> Tensor:
    """Decode a metadata sequence through its adapter and prepare RGB tensor."""

    return rgb_images_to_video([adapter.load_rgb(item) for item in sequence.observations], resolution)
