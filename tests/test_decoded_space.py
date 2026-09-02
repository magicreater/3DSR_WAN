from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from rl3dsr.validation.decoded_space import (
    flow_euler_step,
    frame_metrics,
    paired_condition_summary,
    rgb_range_stats,
    summarize_condition_rows,
    safe_artifact_name,
    shifted_flow_sigmas,
    temporal_delta_mae,
    velocity_to_clean,
)


def test_velocity_to_clean_exactly_recovers_constructed_clean_latent():
    clean = torch.tensor([[[[[2.0]]]], [[[[4.0]]]]])
    noise = torch.tensor([[[[[-1.0]]]], [[[[1.0]]]]])
    sigma = torch.tensor([0.25, 0.75])
    noisy = (1 - sigma[:, None, None, None, None]) * clean + sigma[:, None, None, None, None] * noise
    velocity = noise - clean
    recovered = velocity_to_clean(noisy, velocity, sigma)
    assert torch.equal(recovered, clean)


def test_velocity_to_clean_rejects_incompatible_inputs():
    value = torch.zeros(2, 3, 1, 4, 4)
    with pytest.raises(ValueError, match="same shape"):
        velocity_to_clean(value, value[:, :, :, :, :2], 0.5)
    with pytest.raises(ValueError, match="sigma"):
        velocity_to_clean(value, value, torch.tensor([0.1, 0.2, 0.3]))


def test_shifted_flow_sigmas_match_wan_schedule_and_end_at_zero():
    sigmas = shifted_flow_sigmas(8, shift=5.0)
    assert sigmas.shape == (9,)
    assert sigmas[0].item() == pytest.approx(1.0)
    assert sigmas[-1].item() == pytest.approx(0.0)
    assert torch.all(sigmas[:-1] > sigmas[1:])
    base_second = 0.875
    assert sigmas[1].item() == pytest.approx(5 * base_second / (1 + 4 * base_second))


def test_flow_euler_step_uses_velocity_and_sigma_delta():
    sample = torch.tensor([1.0, 2.0])
    velocity = torch.tensor([3.0, -4.0])
    actual = flow_euler_step(sample, velocity, sigma=0.8, next_sigma=0.5)
    assert torch.allclose(actual, torch.tensor([0.1, 3.2]))


def test_frame_metrics_identity_and_known_mae():
    target = torch.zeros(1, 3, 2, 8, 8)
    identical = frame_metrics(target, target)
    assert len(identical) == 2
    assert all(math.isinf(row["psnr"]) for row in identical)
    assert all(row["ssim"] == pytest.approx(1.0) for row in identical)
    assert all(row["mae"] == 0.0 for row in identical)

    prediction = target.clone()
    prediction[:, :, 1] = 1.0
    rows = frame_metrics(prediction, target)
    assert rows[0]["mae"] == 0.0
    assert rows[1]["mae"] == pytest.approx(0.5)


def test_frame_metrics_adds_one_lpips_value_per_view():
    class MeanDistance(torch.nn.Module):
        def forward(self, prediction, target):
            return (prediction - target).square().mean(dim=(1, 2, 3), keepdim=True)

    target = torch.zeros(1, 3, 2, 8, 8)
    prediction = target.clone()
    prediction[:, :, 1] = 1.0
    rows = frame_metrics(prediction, target, perceptual_metric=MeanDistance())
    assert rows[0]["lpips"] == 0.0
    assert rows[1]["lpips"] == 1.0


def test_temporal_delta_mae_detects_wrong_frame_change():
    target = torch.zeros(1, 3, 3, 4, 4)
    target[:, :, 1:] = 1.0
    assert temporal_delta_mae(target, target) == 0.0
    static = torch.zeros_like(target)
    assert temporal_delta_mae(static, target) == pytest.approx(0.25)


def test_rgb_range_stats_reports_nonfinite_and_out_of_range_values():
    value = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, float("nan")])
    stats = rgb_range_stats(value)
    assert stats["numel"] == 6
    assert stats["nonfinite_count"] == 1
    assert stats["out_of_range_count"] == 2
    assert stats["out_of_range_ratio"] == pytest.approx(2 / 5)
    assert stats["finite_min"] == -2.0
    assert stats["finite_max"] == 2.0


def test_safe_artifact_name_is_deterministic_and_portable():
    assert safe_artifact_name("chair:views:[0, 99, 199]") == "chair_views_0_99_199"
    assert safe_artifact_name("stage1 motion seed 0") == "stage1_motion_seed_0"
    assert safe_artifact_name("chair:views:[0, 99, 199]") == safe_artifact_name("chair:views:[0, 99, 199]")



def test_condition_summary_reports_distribution_and_worst_sample():
    rows = [
        {"item": "a", "position": 0, "mae": 0.1, "psnr": 20.0, "ssim": 0.8},
        {"item": "b", "position": 1, "mae": 0.3, "psnr": 10.0, "ssim": 0.4},
    ]
    summary = summarize_condition_rows(rows)
    assert summary["count"] == 2
    assert summary["mae"]["mean"] == pytest.approx(0.2)
    assert summary["psnr"]["median"] == pytest.approx(15.0)
    assert summary["ssim"]["std"] == pytest.approx(0.2)
    assert summary["worst_mae"] == {"item": "b", "position": 1, "value": 0.3}


def test_paired_condition_summary_uses_exact_item_and_position_pairs():
    rows = [
        {"item": "a", "position": 0, "condition": "correct", "mae": 0.1, "psnr": 20.0, "ssim": 0.8},
        {"item": "a", "position": 0, "condition": "shuffled", "mae": 0.2, "psnr": 18.0, "ssim": 0.7},
        {"item": "b", "position": 1, "condition": "correct", "mae": 0.3, "psnr": 10.0, "ssim": 0.4},
        {"item": "b", "position": 1, "condition": "shuffled", "mae": 0.2, "psnr": 12.0, "ssim": 0.5},
    ]
    summary = paired_condition_summary(rows, control="shuffled")
    assert summary["count"] == 2
    assert summary["psnr_delta_correct_minus_control"]["mean"] == pytest.approx(0.0)
    assert summary["ssim_delta_correct_minus_control"]["mean"] == pytest.approx(0.0)
    assert summary["mae_improvement_control_minus_correct"]["mean"] == pytest.approx(0.0)
    assert summary["correct_win_count"] == {"mae": 1, "psnr": 1, "ssim": 1, "all_three": 1}
    assert summary["correct_win_fraction"]["all_three"] == pytest.approx(0.5)


def test_paired_condition_summary_rejects_missing_pairs():
    rows = [
        {"item": "a", "position": 0, "condition": "correct", "mae": 0.1, "psnr": 20.0, "ssim": 0.8},
    ]
    with pytest.raises(ValueError, match="paired keys"):
        paired_condition_summary(rows, control="shuffled")
