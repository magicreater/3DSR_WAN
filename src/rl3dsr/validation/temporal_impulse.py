"""Reproducible temporal receptive-field observations for Wan VAE."""

from __future__ import annotations

from typing import Any

import torch

from rl3dsr.data.temporal_fixture import make_synthetic_video


def sample_to_tensor(sample, *, device: str | torch.device) -> torch.Tensor:
    frames = sample.frames.copy()
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().div(127.5).sub(1.0).to(device)


def run_temporal_impulse(vae, *, lengths=(1, 5, 9, 17), seed=0) -> dict[str, Any]:
    """Return latent temporal deltas for every input-frame impulse."""

    results: dict[str, Any] = {"seed": seed, "lengths": {}}
    for frame_count in lengths:
        sample = make_synthetic_video(frame_count, height=64, width=64, seed=seed)
        baseline = sample_to_tensor(sample, device=vae.device)
        with torch.inference_mode():
            reference = vae.encode_video(baseline)
        length_result = {"latent_length": int(reference.shape[2]), "positions": {}}
        for position in range(frame_count):
            modified = baseline.clone()
            modified[:, :, position, 16:48, 16:48] = 1.0
            with torch.inference_mode():
                delta = (vae.encode_video(modified) - reference).abs()
            norms = delta.mean(dim=(0, 1, 3, 4)).detach().cpu()
            threshold = max(1e-6, float(norms.max()) * 1e-4)
            affected = [index for index, value in enumerate(norms.tolist()) if value > threshold]
            length_result["positions"][str(position)] = {
                "affected_latent_positions": affected,
                "mean_abs_delta": [float(value) for value in norms.tolist()],
                "threshold": threshold,
            }
        results["lengths"][str(frame_count)] = length_result
    return results
