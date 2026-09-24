import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "stage3_3_rank_weight_ablation.py"
    spec = importlib.util.spec_from_file_location("stage3_3_rank_weight_ablation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_weight_candidates_are_exact(driver):
    w1 = driver.VARIANTS["w1_lego_seed42"]
    w2 = driver.VARIANTS["w2_lego_seed42"]
    assert w1["rank_weight"] * w1["camera_fraction"] == pytest.approx(0.5)
    assert w1["rank_weight"] * (1 - w1["camera_fraction"]) == pytest.approx(0.125)
    assert w2["rank_weight"] * w2["camera_fraction"] == pytest.approx(0.75)
    assert w2["rank_weight"] * (1 - w2["camera_fraction"]) == pytest.approx(0.125)


def test_core_paired_gate_requires_mean_and_direction(driver):
    def gate(psnr, ssim):
        return {"deltas": {"shuffle_fusion": {
            "psnr": {"mean": sum(psnr) / 4, "values": psnr},
            "ssim": {"mean": sum(ssim) / 4, "values": ssim},
        }}}

    assert driver._paired_gate(
        gate([0.04] * 4, [0.0004] * 4), "shuffle_fusion"
    )["pass"]
    result = driver._paired_gate(
        gate([0.10, 0.10, -0.01, -0.01], [0.0004] * 4),
        "shuffle_fusion",
    )
    assert not result["pass"]
    assert not result["checks"]["psnr_direction"]


@pytest.mark.parametrize("failure,expected", [(None, 1), ("LR_FAIL", 1), ("CAMERA_FAIL", 2)])
def test_staged_stop_only_runs_w2_for_camera_failure(driver, monkeypatch, tmp_path, failure, expected):
    calls = []

    def run_cell(_args, name):
        calls.append(name)
        return {
            "pass": failure is None,
            "failure_category": failure,
            "status": "WEIGHT_CANDIDATE" if failure is None else "HOLD",
            "weights": {},
            "hard_gate": {},
            "next": "NEXT",
        }

    monkeypatch.setattr(driver, "run_cell", run_cell)
    monkeypatch.setattr(driver, "_write_final", lambda _args, results: results)
    result = driver.run(SimpleNamespace(campaign_root=tmp_path))
    assert len(calls) == expected
    assert calls[0] == "w1_lego_seed42"
    assert (result["w2_lego_seed42"] is not None) == (expected == 2)
