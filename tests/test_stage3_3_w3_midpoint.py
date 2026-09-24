import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage3_3_w3_midpoint as w3
import stage3_3_weight_gradient_audit as audit


def test_target_flow_is_the_single_registered_candidate():
    assert w3.TOTAL_WEIGHT == 0.75
    assert math.isclose(w3.TOTAL_WEIGHT * w3.CAMERA_FRACTION, 0.625)
    assert math.isclose(w3.TOTAL_WEIGHT * (1 - w3.CAMERA_FRACTION), 0.125)
    assert w3.TARGET_FLOW_FRACTION == 0.5
    assert w3.ORDER == ("f1_lego_seed42", "f1_lego_seed43", "f1_chair_seed42")


def test_gradient_statistics_preserve_negative_and_zero_cases():
    gradients = {
        "flow": (torch.tensor([1., 2., 3.]), torch.tensor([1.]), torch.tensor([1.])),
        "camera_rank": (torch.tensor([-1., -2., 3.]), torch.tensor([-1.]), torch.tensor([-1.])),
        "lr_rank": (torch.zeros(3), torch.tensor([0.]), torch.tensor([0.])),
    }
    result = audit.gradient_statistics(gradients, hidden=1)
    assert result["qk"]["cosines"]["flow_vs_camera_rank"] == pytest.approx(-1)
    assert result["value"]["cosines"]["flow_vs_camera_rank"] == pytest.approx(1)
    assert result["shared"]["zero_gradient"]["lr_rank"]
    assert result["shared"]["cosines"]["flow_vs_lr_rank"] is None
    summary = audit.summarize([{"gradients": result}])
    assert summary["zero_gradient_count"]["lr_rank"] == 1
    assert summary["cosine"]["flow_vs_lr_rank"]["valid"] == 0


@pytest.mark.parametrize("first_pass,expected", [(False, ["f1_lego_seed42"]),
                                                 (True, list(w3.ORDER))])
def test_w3_stops_on_first_hard_gate_failure(monkeypatch, first_pass, expected):
    called = []
    monkeypatch.setattr(w3, "run_cell", lambda args, name: called.append(name) or
                        {"pass": first_pass if name == w3.ORDER[0] else True})
    monkeypatch.setattr(w3, "verdict", lambda args, results: dict(results))
    w3.run(SimpleNamespace())
    assert called == expected


def test_all_passes_still_hold_stage4(monkeypatch):
    monkeypatch.setattr(w3.prior, "sha256_file", lambda path: "hash")
    monkeypatch.setattr(w3.prior, "write_frozen_json", lambda *args: None)
    monkeypatch.setattr(w3.prior, "write_frozen_text", lambda *args: None)
    args = SimpleNamespace(campaign_root=Path("unused"), gradient_audit=Path("unused"))
    results = {name: {"pass": True, "hard_gate": {"checks": {
        "quality": True, "mispaired_lr": True,
        **{key: True for key in w3.previous.CAMERA_CONDITIONS}}},
        "checkpoint_sha256": "hash", "failure_category": None} for name in w3.ORDER}
    verdict = w3.verdict(args, results)
    assert verdict["verdict"] == "REVIEW_STAGE4"
    assert verdict["STAGE4_READY"] is False
