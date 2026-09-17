import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rl3dsr.validation.stage3_protocol import Stage3Config, load_stage3_config


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/stage3_3_ucpe_rre_fusion.py"
    spec = importlib.util.spec_from_file_location("stage3_3_driver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def passing_rows(driver):
    rows = []
    conditions = {
        "correct": (30.0, 0.95),
        "correct_repeat": (30.0, 0.95),
        "remove": (28.5, 0.94),
        "target_drop": (26.0, 0.90),
        "shuffle_fusion": (29.9, 0.949),
        "target_drop_shuffle_fusion": (29.8, 0.948),
        "mispaired_lr": (29.7, 0.947),
        "mispaired_camera": (29.7, 0.947),
        "joint_permute": (30.0, 0.95),
        "fusion_camera_dose_half": (29.95, 0.9495),
        "fusion_camera_dose_full": (29.9, 0.949),
        "shuffle_geometry": (29.0, 0.94),
    }
    for group in driver.PROBE_IDS:
        for condition in driver.MODES:
            psnr, ssim = conditions[condition]
            rows.append({
                "group_id": group,
                "condition": condition,
                "psnr": psnr,
                "ssim": ssim,
            })
    return rows


def passing_train_rows(steps=200):
    return [
        {
            "step": step,
            "correct_flow_loss": 1.0,
            "wrong_flow_loss": 1.1,
            "camera_rank_active_fraction": 0.4,
        }
        for step in range(1, steps + 1)
    ]


def test_candidate_gate_requires_every_frozen_threshold(driver):
    result = driver.candidate_gate(
        passing_rows(driver), passing_train_rows(), expected_steps=200
    )
    assert result["pass"] is True
    assert result["camera_dose_monotonic_probes"] == {"psnr": 4, "ssim": 4}


def test_candidate_gate_does_not_substitute_geometry_for_fusion(driver):
    rows = passing_rows(driver)
    for row in rows:
        if row["condition"] == "shuffle_fusion":
            row["psnr"] = 30.0
            row["ssim"] = 0.95
    result = driver.candidate_gate(rows, passing_train_rows(), expected_steps=200)
    assert result["pass"] is False
    assert result["checks"]["shuffle_fusion_psnr"] is False
    assert result["deltas"]["shuffle_geometry"]["psnr"]["mean"] > 0


def test_candidate_gate_rejects_nonseparating_last_100_steps(driver):
    train = passing_train_rows()
    for row in train[-100:]:
        row["wrong_flow_loss"] = 1.01
        row["camera_rank_active_fraction"] = 1.0
    result = driver.candidate_gate(passing_rows(driver), train, expected_steps=200)
    assert result["checks"]["last100_wrong_correct_margin"] is False
    assert result["checks"]["last100_rank_hinge_activity"] is False


def test_evaluation_key_set_is_exact(driver):
    rows = passing_rows(driver)
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        driver.evaluation_index(rows)
    with pytest.raises(ValueError, match="mismatch"):
        driver.evaluation_index(passing_rows(driver)[:-1])


def test_camera_dose_must_be_monotonic_in_both_metrics(driver):
    rows = passing_rows(driver)
    for row in rows:
        if row["group_id"] in driver.PROBE_IDS[:2] and row["condition"] == "fusion_camera_dose_full":
            row["ssim"] = 0.9505
    result = driver.candidate_gate(rows, passing_train_rows(), expected_steps=200)
    assert result["checks"]["camera_dose_psnr_monotonic"] is True
    assert result["checks"]["camera_dose_ssim_monotonic"] is False


def test_h0_routes_to_fresh_a3_1000_or_a4(driver):
    assert driver.next_action("h0", True) == "RUN_A3_1000"
    assert driver.next_action("h0", False) == "RUN_A4_1000"
    assert driver.next_action("a4_pilot", False) == "RUN_A5_1000"
    assert driver.next_action("a5_pilot", False) == "FINAL_HOLD"


def test_replication_configs_keep_the_winning_arm_and_frozen_scale(driver):
    base = Stage3Config(
        arm="A4",
        train_scenes=("chair",),
        validation_scenes=("lego",),
        test_scenes=("drums",),
        fusion_dim=192,
        fusion_heads=1,
        epipolar_attention="local_band",
        allow_self_view_source=False,
        camera_rank_weight=1.0,
    )
    seed43 = driver.derive_replication_config(base, "seed43")
    v8 = driver.derive_replication_config(base, "v8")
    five = driver.derive_replication_config(base, "five_scene")
    assert seed43.arm == v8.arm == five.arm == "A4"
    assert seed43.steps == v8.steps == five.steps == 2000
    assert v8.views == 8
    assert five.train_scenes == driver.FIVE_TRAIN_SCENES


def test_all_fixed_templates_are_valid_and_bounded(driver):
    root = Path(__file__).resolve().parents[1]
    for cell in driver.FIXED_CELLS.values():
        config = load_stage3_config(root / "configs" / "stage3_3" / cell.config_name)
        assert config.steps in {200, 1000}
        assert config.camera_rank_weight == 1.0
        assert config.allow_self_view_source is False


def test_complete_training_uses_the_runner_four_digit_checkpoint_name(driver, tmp_path):
    output = tmp_path / "train"
    output.mkdir()
    (output / "stage3_step_0200.pt").write_bytes(b"checkpoint")
    (output / "train_steps.jsonl").write_text(
        "".join(driver._json_text({"step": step}).replace("\n", "") + "\n" for step in range(1, 201))
    )
    assert driver._complete_training(output, 200)


def test_cli_has_no_stage4_or_4dsr_execution_command(driver):
    parser = driver.build_parser()
    command = next(action for action in parser._actions if action.dest == "command")
    assert "stage4" not in command.choices
    assert "4dsr" not in command.choices


def test_gpu_guard_is_uuid_scoped_and_rejects_selected_process(driver, monkeypatch):
    monkeypatch.setattr(
        driver.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "0, GPU-A, 2, 0\n1, GPU-B, 2, 0\n",
    )
    monkeypatch.setattr(
        driver.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="GPU-B, 123\n"),
    )
    assert driver._gpu_record(0)["uuid"] == "GPU-A"
    with pytest.raises(RuntimeError, match="exclusively idle"):
        driver._gpu_record(1)


def test_readiness_never_relabels_new_four_probe_gates_as_frozen_legacy_gates(driver, tmp_path):
    winner = driver.FIXED_CELLS["a4_pilot"]
    names = [winner.name, f"{winner.name}_seed43_2000"]
    base = driver.candidate_gate(
        passing_rows(driver), passing_train_rows(200), expected_steps=200
    )
    for name in names:
        path = tmp_path / "analysis" / name / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(driver._json_text({"gate": base}))
    args = SimpleNamespace(campaign_root=tmp_path)
    result = driver.recompute_readiness(args, winner, names)
    assert result["recomputed"] is False
    assert result["verdicts"]["CROSS_VIEW_PASS"] is False
    assert result["verdicts"]["GEOMETRY_PASS"] is False
    assert result["verdicts"]["SEEN_SR_EFFECTIVE"] is False


def test_run_stops_after_failed_a5_pilot_without_replication(driver, tmp_path, monkeypatch):
    args = SimpleNamespace(campaign_root=tmp_path)
    calls = []
    monkeypatch.setattr(driver, "prepare", lambda _args: calls.append("prepare"))
    monkeypatch.setattr(driver, "cpu_tests", lambda _args: calls.append("cpu"))
    monkeypatch.setattr(driver, "smoke", lambda _args: calls.append("smoke"))
    monkeypatch.setattr(driver, "a5_probe_coverage", lambda _args: calls.append("a5_preflight"))
    monkeypatch.setattr(
        driver, "analyze_cell",
        lambda _args, cell: calls.append(cell.name) or {"pass": False},
    )
    monkeypatch.setattr(
        driver, "_replication_cell",
        lambda *_args: pytest.fail("replication must not start after failed pilot"),
    )

    result = driver.run(args)

    assert calls == ["prepare", "cpu", "smoke", "h0", "a4_pilot", "a5_preflight", "a5_pilot"]
    assert result["verdict"] == "HOLD"
    assert result["STAGE4_READY"] is False


def test_run_requires_fresh_a3_1000_after_h0_screen(driver, tmp_path, monkeypatch):
    args = SimpleNamespace(campaign_root=tmp_path)
    calls = []
    monkeypatch.setattr(driver, "prepare", lambda _args: None)
    monkeypatch.setattr(driver, "cpu_tests", lambda _args: None)
    monkeypatch.setattr(driver, "smoke", lambda _args: None)

    def analyze(_args, cell):
        calls.append(cell.name)
        return {"pass": cell.name == "h0"}

    monkeypatch.setattr(driver, "analyze_cell", analyze)
    result = driver.run(args)
    assert calls == ["h0", "a3_pilot"]
    assert result["verdict"] == "HOLD"


def test_a5_probe_coverage_failure_stops_before_training(driver, tmp_path, monkeypatch):
    args = SimpleNamespace(campaign_root=tmp_path)
    calls = []
    monkeypatch.setattr(driver, "prepare", lambda _args: None)
    monkeypatch.setattr(driver, "cpu_tests", lambda _args: None)
    monkeypatch.setattr(driver, "smoke", lambda _args: None)

    def analyze(_args, cell):
        calls.append(cell.name)
        return {"pass": False}

    monkeypatch.setattr(driver, "analyze_cell", analyze)
    monkeypatch.setattr(
        driver, "a5_probe_coverage",
        lambda _args: (_ for _ in ()).throw(
            ValueError("A5 four-probe coverage preflight failed: chair:033")
        ),
    )
    result = driver.run(args)
    assert calls == ["h0", "a4_pilot"]
    assert result["verdict"] == "HOLD"
    assert driver.read_json(tmp_path / "analysis" / "a5_preflight_hold.json")["verdict"] == "HOLD"
