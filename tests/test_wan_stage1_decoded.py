from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from rl3dsr.data.temporal_fixture import make_motion_video
from rl3dsr.models.wan import WanDiT, WanVAE
from rl3dsr.models.wan.flow import flow_matching_pair
from rl3dsr.models.wan.lq_conditioning import (
    FrozenLQConditioner,
    Stage1Degradation,
    load_adapter_checkpoint,
    load_flashvsr_projector,
)
from rl3dsr.validation.decoded_space import velocity_to_clean

pytestmark = pytest.mark.wan

MODEL_DIR = os.environ.get("RL3DSR_WAN_MODEL_DIR")
LQ_SOURCE = os.environ.get("RL3DSR_LQ_SOURCE")
LQ_CHECKPOINT = os.environ.get("RL3DSR_LQ_CHECKPOINT")
ADAPTER_3D = os.environ.get("RL3DSR_ADAPTER_3D")
ADAPTER_4D = os.environ.get("RL3DSR_ADAPTER_4D")
requires_decoded = pytest.mark.skipif(
    not all((MODEL_DIR, LQ_SOURCE, LQ_CHECKPOINT, ADAPTER_3D, ADAPTER_4D)),
    reason="real Wan, LQ, and both Stage 1 adapter checkpoints are required",
)


@requires_decoded
def test_trained_stage1_adapters_decode_through_real_3d_and_native_4d_paths():
    device = torch.device("cuda")
    projector = load_flashvsr_projector(
        LQ_SOURCE, LQ_CHECKPOINT, device=device, dtype=torch.bfloat16
    )
    adapter_payload = torch.load(ADAPTER_3D, map_location="cpu", weights_only=True)
    adapter_config = adapter_payload["config"]
    conditioner = FrozenLQConditioner(
        projector,
        bridge_blocks=tuple(adapter_config.get("bridge_blocks", (0,))),
        bridge_time_conditioning=adapter_config.get("bridge_time_conditioning", False),
    ).to(device).eval()
    vae = WanVAE.from_checkpoint(MODEL_DIR, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(MODEL_DIR, device=device, dtype=torch.bfloat16)
    context = torch.zeros(1, 512, 4096, device=device, dtype=torch.bfloat16)
    degradation = Stage1Degradation(scale=4)

    multiview = torch.linspace(-1, 1, 1 * 3 * 4 * 64 * 64).reshape(1, 3, 4, 64, 64)
    video_np = make_motion_video(9, height=64, width=64, seed=23).frames.copy()
    video = torch.from_numpy(video_np).permute(3, 0, 1, 2).unsqueeze(0).float().div(127.5).sub(1)

    with torch.inference_mode():
        metadata_3d = load_adapter_checkpoint(ADAPTER_3D, conditioner)
        multiview_z = vae.encode_multiview(multiview.to(device))
        multiview_features = conditioner.multiview_features(
            degradation(multiview).to(device),
            conditioning_size=(64, 64),
            latent_shape=tuple(multiview_z.shape[2:]),
        )
        generator = torch.Generator(device=device).manual_seed(701)
        noise = torch.randn(multiview_z.shape, generator=generator, device=device, dtype=multiview_z.dtype)
        noisy, timestep, _ = flow_matching_pair(multiview_z, noise, torch.tensor([0.5], device=device))
        correct_prediction = dit(
            noisy, timestep, context,
            block_token_residuals=conditioner.bridge_residuals(multiview_features, timestep),
        )
        disabled_prediction = dit(noisy, timestep, context)
        multiview_rgb = vae.decode_multiview(
            velocity_to_clean(noisy, correct_prediction, 0.5)
        )

        adapter_4d_payload = torch.load(ADAPTER_4D, map_location="cpu", weights_only=True)
        adapter_4d_config = adapter_4d_payload["config"]
        conditioner_4d = FrozenLQConditioner(
            projector,
            bridge_blocks=tuple(adapter_4d_config.get("bridge_blocks", (0,))),
            bridge_time_conditioning=adapter_4d_config.get("bridge_time_conditioning", False),
        ).to(device).eval()
        metadata_4d = load_adapter_checkpoint(ADAPTER_4D, conditioner_4d)
        video_z = vae.encode_video(video.to(device))
        video_features = conditioner.video_features(
            degradation(video).to(device),
            conditioning_size=(64, 64),
            latent_shape=tuple(video_z.shape[2:]),
        )
        generator = torch.Generator(device=device).manual_seed(702)
        noise = torch.randn(video_z.shape, generator=generator, device=device, dtype=video_z.dtype)
        noisy, timestep, _ = flow_matching_pair(video_z, noise, torch.tensor([0.5], device=device))
        video_prediction = dit(
            noisy, timestep, context,
            token_residual=conditioner_4d.bridge_tokens(video_features),
        )
        video_rgb = vae.decode_video(velocity_to_clean(noisy, video_prediction, 0.5))

    assert metadata_3d["config"]["kind"] == "3d"
    assert metadata_4d["config"]["kind"] == "4d"
    assert multiview_z.shape == (1, 16, 4, 8, 8)
    assert multiview_rgb.shape == multiview.shape
    assert video_z.shape == (1, 16, 3, 8, 8)
    assert video_rgb.shape == video.shape
    assert torch.isfinite(multiview_rgb).all()
    assert torch.isfinite(video_rgb).all()
    assert (correct_prediction - disabled_prediction).abs().float().mean() > 1e-5
    assert all(not parameter.requires_grad for parameter in vae.model.model.parameters())
    assert all(not parameter.requires_grad for parameter in dit.model.parameters())
    assert all(not parameter.requires_grad for parameter in conditioner.projector.parameters())
