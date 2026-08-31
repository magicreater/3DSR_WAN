import os

import pytest

torch = pytest.importorskip("torch")

from rl3dsr.models.wan import WanCheckpoint, WanDiT, WanVAE

pytestmark = pytest.mark.wan


MODEL_DIR = os.environ.get("RL3DSR_WAN_MODEL_DIR")
requires_wan = pytest.mark.skipif(not MODEL_DIR, reason="RL3DSR_WAN_MODEL_DIR is not set")


def test_checkpoint_paths_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        WanCheckpoint.from_dir(tmp_path)


@requires_wan
def test_3d_vae_isolation_and_roundtrip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = WanVAE.from_checkpoint(MODEL_DIR, device=device, dtype=torch.bfloat16)
    rgb = torch.linspace(-1, 1, 1 * 3 * 4 * 64 * 64, device=device).reshape(1, 3, 4, 64, 64)
    with torch.inference_mode():
        latents = vae.encode_multiview(rgb)
        changed = rgb.clone()
        changed[:, :, 2] += 0.1
        changed[:, :, 2].clamp_(-1, 1)
        changed_latents = vae.encode_multiview(changed)
        reconstruction = vae.decode_multiview(latents)
    assert latents.shape[:3] == (1, 16, 4)
    assert torch.max(torch.abs(changed_latents[:, :, 2] - latents[:, :, 2])) > 1e-6
    assert torch.max(torch.abs(changed_latents[:, :, [0, 1, 3]] - latents[:, :, [0, 1, 3]])) < 1e-5
    assert reconstruction.shape == rgb.shape
    assert torch.isfinite(reconstruction).all()


@requires_wan
@pytest.mark.parametrize("frame_count", [1, 5, 9, 17])
def test_native_video_vae_and_dit(frame_count):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = WanVAE.from_checkpoint(MODEL_DIR, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(MODEL_DIR, device=device, dtype=torch.bfloat16)
    rgb = torch.zeros(1, 3, frame_count, 64, 64, device=device)
    with torch.inference_mode():
        latents = vae.encode_video(rgb)
        reconstruction = vae.decode_video(latents)
        output = dit(latents, torch.tensor([500.0], device=device))
    assert latents.shape[0] == 1 and latents.shape[1] == 16
    assert reconstruction.shape == rgb.shape
    assert output.shape == latents.shape
    assert torch.isfinite(output).all()
