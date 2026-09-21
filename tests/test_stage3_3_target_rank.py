import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rl3dsr.validation.stage3_protocol import load_stage3_config


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/stage3_3_target_rank.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("stage3_3_target_rank", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a6_template_is_target_rank_only(driver):
    config = load_stage3_config(driver.CONFIG_TEMPLATE)
    assert config.arm == "A6"
    assert config.train_scenes == ("lego",)
    assert config.camera_rank_weight == 1.0
    assert config.camera_rank_margin_ratio == 0.05
    assert config.target_lr_dropout == 0.5
    assert config.pairing_supervision is False


def test_a6_run_stops_at_first_hold(driver, tmp_path, monkeypatch):
    calls = []

    def run_cell(_args, name):
        calls.append(name)
        return {"status": "HOLD", "pass": False, "next": "A7_REQUIRED"}

    monkeypatch.setattr(driver, "run_cell", run_cell)
    result = driver.run(SimpleNamespace(campaign_root=tmp_path))
    assert calls == ["a6_lego_seed42"]
    assert result["next"] == "A7_REQUIRED"
    assert result["a6_lego_seed43"] is None
    assert result["a6_chair_seed42"] is None
    assert result["STAGE4_READY"] is False


def test_a6_run_replicates_only_after_each_pass(driver, tmp_path, monkeypatch):
    calls = []

    def run_cell(_args, name):
        calls.append(name)
        next_action = {
            "a6_lego_seed42": "RUN_LEGO_SEED43",
            "a6_lego_seed43": "RUN_CHAIR_SEED42",
            "a6_chair_seed42": "REVIEW_STAGE4",
        }[name]
        return {"status": "PILOT_PASS", "pass": True, "next": next_action}

    monkeypatch.setattr(driver, "run_cell", run_cell)
    result = driver.run(SimpleNamespace(campaign_root=tmp_path))
    assert calls == ["a6_lego_seed42", "a6_lego_seed43", "a6_chair_seed42"]
    assert result["next"] == "REVIEW_STAGE4"
    assert result["STAGE4_READY"] is False


def test_a6_cli_has_no_forbidden_execution_commands(driver):
    parser = driver.build_parser()
    command = next(action for action in parser._actions if action.dest == "command")
    assert not {"counter", "stage4", "4dsr"} & set(command.choices)
