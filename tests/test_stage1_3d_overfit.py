from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from rl3dsr.models.wan.sampling import (
    FlowSamplingConfig,
    SigmaCycle,
    sample_conditioned_flow,
    training_sigmas,
    unipc_schedule,
)
from rl3dsr.models.wan.lq_conditioning import evenly_spaced_indices
from rl3dsr.validation.decoded_space import strict_3d_quality_verdict


def test_official_unipc_schedule_has_50_steps_and_explicit_zero_endpoint():
    timesteps, sigmas = unipc_schedule(FlowSamplingConfig(steps=50, shift=5.0), device="cpu")
    assert timesteps.shape == (50,)
    assert sigmas.shape == (51,)
    assert timesteps[:3].tolist() == [999, 995, 991]
    assert timesteps[-3:].tolist() == [241, 172, 92]
    assert sigmas[-1].item() == 0.0
    assert torch.all(sigmas[:-1] > sigmas[1:])


def test_training_sigmas_include_pure_noise_and_every_inference_sigma():
    _, inference_sigmas = unipc_schedule(FlowSamplingConfig(), device="cpu")
    values = training_sigmas(FlowSamplingConfig(), device="cpu")
    assert values[0].item() == 1.0
    assert torch.equal(values[1:], inference_sigmas[:-1])
    assert not bool((values == 0).any())


def test_single_sample_uses_four_fixed_evenly_spaced_views():
    assert evenly_spaced_indices(100, 4) == (0, 33, 66, 99)


def test_sigma_cycle_resume_reproduces_the_next_value():
    cycle = SigmaCycle(FlowSamplingConfig(steps=4), seed=42)
    first_cycle = [cycle.next().item() for _ in range(5)]
    expected = training_sigmas(FlowSamplingConfig(steps=4), device="cpu")
    assert sorted(first_cycle) == pytest.approx(sorted(expected.tolist()))
    state = cycle.state_dict()
    expected_next = cycle.next()

    restored = SigmaCycle(FlowSamplingConfig(steps=4), seed=999)
    restored.load_state_dict(state)
    assert torch.equal(restored.next(), expected_next)


def test_sampler_api_cannot_receive_clean_or_target_latent():
    parameters = inspect.signature(sample_conditioned_flow).parameters
    assert "clean" not in parameters
    assert "target" not in parameters
    assert "initial_noise" in parameters
    assert "condition_features" in parameters


def test_sampler_reuses_condition_and_reaches_finite_output():
    initial = torch.zeros(1, 16, 1, 2, 2)
    features = torch.ones(1, 1, 4)
    context = torch.zeros(1, 2, 3)
    seen_features = []

    def predict(sample, timestep, supplied_context, supplied_features):
        assert supplied_context is context
        seen_features.append(supplied_features)
        return torch.ones_like(sample)

    result = sample_conditioned_flow(
        initial,
        features,
        context,
        predict_velocity=predict,
        config=FlowSamplingConfig(steps=4, shift=5.0),
    )
    assert result.shape == initial.shape
    assert torch.isfinite(result).all()
    assert len(seen_features) == 4
    assert all(value is features for value in seen_features)


def _quality_rows(correct, bicubic, shuffled, disabled):
    rows = []
    for seed in (2201, 2202, 2203, 2204):
        for position in range(4):
            for condition, metrics in (
                ("correct", correct),
                ("bicubic", bicubic),
                ("shuffled", shuffled),
                ("disabled", disabled),
            ):
                rows.append({
                    "seed": seed,
                    "item": "chair",
                    "position": position,
                    "condition": condition,
                    **metrics,
                })
    return rows


def test_strict_quality_verdict_requires_bicubic_and_shuffled_margins():
    passing = _quality_rows(
        {"psnr": 26.5, "ssim": 0.86, "lpips": 0.18},
        {"psnr": 26.0, "ssim": 0.85, "lpips": 0.20},
        {"psnr": 25.0, "ssim": 0.82, "lpips": 0.22},
        {"psnr": 24.0, "ssim": 0.80, "lpips": 0.25},
    )
    verdict = strict_3d_quality_verdict(passing)
    assert verdict["passed"] is True
    assert verdict["all_metric_wins_vs_bicubic"] == 16
    assert verdict["all_metric_wins_vs_shuffled"] == 16

    failing = _quality_rows(
        {"psnr": 26.1, "ssim": 0.852, "lpips": 0.195},
        {"psnr": 26.0, "ssim": 0.85, "lpips": 0.20},
        {"psnr": 25.0, "ssim": 0.82, "lpips": 0.22},
        {"psnr": 24.0, "ssim": 0.80, "lpips": 0.25},
    )
    verdict = strict_3d_quality_verdict(failing)
    assert verdict["passed"] is False
    assert verdict["checks"]["bicubic_margin"] is False
