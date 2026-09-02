from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from rl3dsr.models.wan.sampling import FlowSamplingConfig, SigmaCycle, oracle_sampling_audit, sample_conditioned_flow, training_sigmas

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage1_experiment.py"
_SPEC = importlib.util.spec_from_file_location("stage1_experiment", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["stage1_experiment"] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_append_jsonl = _MODULE._append_jsonl
_plot_training_curves = _MODULE._plot_training_curves
_read_jsonl = _MODULE._read_jsonl
_write_csv = _MODULE._write_csv


def test_observation_files_round_trip_and_curve(tmp_path):
    rows = [{
        "step": 1,
        "loss": 0.5,
        "sigma": 1.0,
        "learning_rate": 3e-4,
        "gradient_norm": 2.0,
        "peak_gpu_memory_mib": 10.0,
        "step_seconds": 0.1,
    }]
    jsonl = tmp_path / "train_steps.jsonl"
    _append_jsonl(jsonl, rows[0])
    assert _read_jsonl(jsonl) == rows
    _write_csv(tmp_path / "train_steps.csv", rows)
    assert (tmp_path / "train_steps.csv").read_text(encoding="utf-8").count("step") >= 2
    _plot_training_curves(tmp_path, rows, [{
        "step": 1,
        "correct_psnr": 20.0,
        "bicubic_psnr": 21.0,
        "correct_shuffled_psnr_gap": 1.0,
        "correct_disabled_psnr_gap": 2.0,
    }])
    assert (tmp_path / "training_curves.png").stat().st_size > 0


def test_balanced_sigma_cycle_populates_all_bins():
    values = training_sigmas(FlowSamplingConfig(steps=50), device="cpu", strategy="balanced")
    assert values[0].item() == 1.0
    assert values.numel() > 1
    assert all(bool(((values[1:] >= low) & (values[1:] < high)).any()) for low, high in ((0, .25), (.25, .5), (.5, .8)))
    assert bool((values[1:] >= .8).any())
    cycle = SigmaCycle(FlowSamplingConfig(steps=50), seed=7, strategy="balanced")
    state = cycle.state_dict()
    restored = SigmaCycle(FlowSamplingConfig(steps=50), seed=8, strategy="balanced")
    restored.load_state_dict(state)
    assert torch.equal(cycle.next(), restored.next())


def test_oracle_unipc_returns_clean_latent():
    clean = torch.randn(1, 2, 1, 4, 4)
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(clean.shape, generator=generator)

    def oracle(sample, timestep, context, features):
        return (noise - clean).expand_as(sample)

    result = sample_conditioned_flow(
        noise, None, torch.zeros(1, 1, 1), predict_velocity=oracle,
        config=FlowSamplingConfig(steps=50, shift=5),
    )
    assert float((result - clean).abs().max()) < 1e-3
    assert float((result - clean).abs().mean()) < 3e-4
    audit = oracle_sampling_audit(clean, noise, config=FlowSamplingConfig(steps=50, shift=5))
    assert len(audit["rows"]) == 50
    assert audit["rows"][-1]["post_max_abs_error"] < 1e-3
