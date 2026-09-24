import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage3_experiment as stage3
from rl3dsr.validation.stage3_protocol import load_stage3_config


def test_target_ssim_is_differentiable_and_config_is_opt_in():
    base = load_stage3_config(
        Path(__file__).resolve().parents[1] / "configs/stage3_3/A6_target_rank_lego_1000.json"
    )
    assert "correct_image_ssim_weight" not in base.to_dict()
    candidate = replace(base, correct_image_ssim_weight=0.05)
    assert candidate.to_dict()["correct_image_ssim_weight"] == 0.05
    with pytest.raises(ValueError):
        replace(candidate, target_view_flow_fraction=0.5)

    class IdentityVAE:
        def decode_multiview(self, x):
            return x

    noisy = torch.zeros(1, 3, 2, 16, 16)
    prediction = torch.full_like(noisy, 0.2, requires_grad=True)
    reference = torch.zeros_like(noisy)
    loss = stage3.decoded_target_ssim_loss(
        IdentityVAE(), noisy, prediction, torch.tensor([0.5]), reference
    )
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert prediction.grad[:, :, 0].abs().sum() > 0
    assert prediction.grad[:, :, 1].abs().sum() == 0
