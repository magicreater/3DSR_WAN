import os

import pytest

torch = pytest.importorskip("torch")

from rl3dsr.data.temporal_fixture import make_motion_video
from rl3dsr.models.wan import WanDiT, WanVAE
from rl3dsr.models.wan.lq_conditioning import (
    FrozenLQConditioner,
    Stage1Degradation,
    load_flashvsr_projector,
)

pytestmark = pytest.mark.wan

MODEL_DIR = os.environ.get("RL3DSR_WAN_MODEL_DIR")
LQ_SOURCE = os.environ.get("RL3DSR_LQ_SOURCE")
LQ_CHECKPOINT = os.environ.get("RL3DSR_LQ_CHECKPOINT")
requires_stage1 = pytest.mark.skipif(
    not all((MODEL_DIR, LQ_SOURCE, LQ_CHECKPOINT)),
    reason="real Wan and LQ checkpoint environment variables are required",
)


@requires_stage1
def test_real_lq_alignment_and_zero_baseline_for_3d_and_4d():
    device = torch.device("cuda")
    projector = load_flashvsr_projector(
        LQ_SOURCE,
        LQ_CHECKPOINT,
        device=device,
        dtype=torch.bfloat16,
    )
    conditioner = FrozenLQConditioner(projector).to(device)
    vae = WanVAE.from_checkpoint(MODEL_DIR, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(MODEL_DIR, device=device, dtype=torch.bfloat16)
    degradation = Stage1Degradation(scale=4)
    context = torch.zeros(1, 512, 4096, device=device, dtype=torch.bfloat16)

    multiview = torch.linspace(-1, 1, 1 * 3 * 4 * 64 * 64).reshape(1, 3, 4, 64, 64)
    video_np = make_motion_video(9, height=64, width=64, seed=19).frames.copy()
    video = torch.from_numpy(video_np).permute(3, 0, 1, 2).unsqueeze(0).float().div(127.5).sub(1)

    with torch.inference_mode():
        multiview_z = vae.encode_multiview(multiview.to(device))
        multiview_features = conditioner.multiview_features(
            degradation(multiview).to(device),
            conditioning_size=(64, 64),
            latent_shape=tuple(multiview_z.shape[2:]),
        )
        baseline = dit(multiview_z, torch.tensor([500.0], device=device), context)
        conditioned = dit(
            multiview_z,
            torch.tensor([500.0], device=device),
            context,
            token_residual=conditioner.bridge_tokens(multiview_features),
        )
        video_z = vae.encode_video(video.to(device))
        video_features = conditioner.video_features(
            degradation(video).to(device),
            conditioning_size=(64, 64),
            latent_shape=tuple(video_z.shape[2:]),
        )

    assert multiview_z.shape == (1, 16, 4, 8, 8)
    assert multiview_features.shape == (1, 64, 1536)
    assert torch.equal(baseline, conditioned)
    assert video_z.shape == (1, 16, 3, 8, 8)
    assert video_features.shape == (1, 48, 1536)
    assert all(not parameter.requires_grad for parameter in vae.model.model.parameters())
    assert all(not parameter.requires_grad for parameter in dit.model.parameters())
    assert all(not parameter.requires_grad for parameter in conditioner.projector.parameters())

@requires_stage1
def test_local_lq_projector_is_exact_flashvsr_reference():
    import importlib.util

    from rl3dsr.models.wan.lq_conditioning import load_flashvsr_projector

    state = torch.load(LQ_CHECKPOINT, map_location="cpu", weights_only=True)
    spec = importlib.util.spec_from_file_location("stage1_flashvsr_reference", LQ_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.device("meta"):
        reference = module.Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    reference.load_state_dict(state, strict=True, assign=True)
    reference.to("cuda", dtype=torch.bfloat16).eval()
    local = load_flashvsr_projector(
        LQ_SOURCE,
        LQ_CHECKPOINT,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.linspace(-1, 1, 3 * 5 * 64 * 64, device="cuda", dtype=torch.bfloat16).reshape(1, 3, 5, 64, 64)
    with torch.inference_mode():
        expected = reference(value)[0]
        actual = local(value)[0]
    assert expected.shape == actual.shape == (1, 16, 1536)
    assert torch.equal(expected, actual)