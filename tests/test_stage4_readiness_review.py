import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def review():
    path = Path(__file__).resolve().parents[1] / "scripts/stage4_readiness_review.py"
    spec = importlib.util.spec_from_file_location("stage4_readiness_review", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gate(psnr=(0.04, 0.04, 0.04, 0.04), ssim=(0.0004, 0.0004, 0.0004, 0.0004)):
    return {
        "pass": True,
        "deltas": {"mispaired_lr": {
            "psnr": {"mean": sum(psnr) / 4, "values": list(psnr)},
            "ssim": {"mean": sum(ssim) / 4, "values": list(ssim)},
        }},
    }


def test_frozen_metric_gates_can_pass(review):
    matched = review.nonregression(
        {"psnr": 30.0, "ssim": 0.95}, {"psnr": 29.90, "ssim": 0.949}
    )
    paired = review.correspondence(gate())
    assert matched["pass"]
    assert paired["pass"]


def test_nonregression_fails_below_either_boundary(review):
    assert not review.nonregression(
        {"psnr": 30.0, "ssim": 0.95}, {"psnr": 29.899, "ssim": 0.949}
    )["pass"]
    assert not review.nonregression(
        {"psnr": 30.0, "ssim": 0.95}, {"psnr": 29.90, "ssim": 0.9489}
    )["pass"]


@pytest.mark.parametrize(
    "psnr,ssim,failed",
    [
        ((0.02, 0.02, 0.02, 0.02), (0.0004,) * 4, "psnr_mean"),
        ((0.10, 0.10, -0.01, -0.01), (0.0004,) * 4, "psnr_direction"),
        ((0.04,) * 4, (0.0002,) * 4, "ssim_mean"),
        ((0.04,) * 4, (0.001, 0.001, -0.0001, -0.0001), "ssim_direction"),
    ],
)
def test_correspondence_fails_mean_or_direction(review, psnr, ssim, failed):
    result = review.correspondence(gate(psnr, ssim))
    assert not result["pass"]
    assert not result["checks"][failed]


def test_missing_inputs_fail_closed(review, tmp_path):
    result = review.build_review(tmp_path, tmp_path / "chair.json", tmp_path / "lego.json")
    assert result["verdict"] == "AUDIT_INVALID"
    assert result["STAGE4_READY"] is False
    assert result["next"] == "FIX_AUDIT_INPUTS"


def test_protocol_revision_mismatch_fails_closed(review, tmp_path):
    (tmp_path / "protocol.json").write_text(
        '{"source_revision":"wrong","stage4_ready":false}', encoding="utf-8"
    )
    (tmp_path / "machine_verdict.json").write_text(
        '{"STAGE4_READY":false}', encoding="utf-8"
    )
    result = review.build_review(tmp_path, tmp_path / "chair.json", tmp_path / "lego.json")
    assert result["verdict"] == "AUDIT_INVALID"
    assert "source revision mismatch" in result["errors"][0]
