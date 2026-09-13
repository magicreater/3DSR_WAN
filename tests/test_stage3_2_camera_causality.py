"""CPU-only contracts for the staged camera-causality audit."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from rl3dsr.models.wan.geometry_conditioning import CameraBatch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage3_2_camera_causality.py"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "stage3_2" / "A3_no_self_local_band_chair_1000.json"
RANK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "stage3_2" / "A3_no_self_rank_local_band_chair_1000.json"


def load_driver():
    assert SCRIPT.is_file(), "Stage 3.2 camera-causality driver is missing"
    spec = importlib.util.spec_from_file_location("stage3_2_camera_causality", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def yaw_camera(degrees):
    rotations = []
    for value in degrees:
        angle = torch.tensor(value * torch.pi / 180)
        rotations.append(torch.tensor([
            [angle.cos(), 0, angle.sin()],
            [0, 1, 0],
            [-angle.sin(), 0, angle.cos()],
        ]))
    transform = torch.eye(4).repeat(1, len(degrees), 1, 1)
    transform[0, :, :3, :3] = torch.stack(rotations)
    intrinsic = torch.eye(3).repeat(1, len(degrees), 1, 1)
    return CameraBatch(intrinsic, transform, (32, 32), "multiview")


def test_gpu_idle_guard_ignores_processes_on_other_gpus(monkeypatch):
    driver = load_driver()

    def fake_check_output(command, **_kwargs):
        query = " ".join(command)
        if "query-compute-apps" in query:
            return "GPU-other, 123, python, 4096 MiB\n"
        return "0, GPU-selected, 2, 0\n1, GPU-other, 4098, 95\n"

    monkeypatch.setattr(driver.subprocess, "check_output", fake_check_output)
    driver._gpu_is_idle(0)


def test_gpu_idle_guard_rejects_process_on_selected_gpu(monkeypatch):
    driver = load_driver()

    def fake_check_output(command, **_kwargs):
        query = " ".join(command)
        if "query-compute-apps" in query:
            return "GPU-selected, 456, python, 1024 MiB\n"
        return "0, GPU-selected, 1026, 90\n1, GPU-other, 2, 0\n"

    monkeypatch.setattr(driver.subprocess, "check_output", fake_check_output)
    with pytest.raises(RuntimeError, match="GPU work is already running"):
        driver._gpu_is_idle(0)


def test_far_donors_are_deterministic_unique_and_outside_group():
    driver = load_driver()
    camera = yaw_camera([0, 10, 20, 170, 180, -170, 90])
    first = driver.select_far_donors(camera, [0, 1, 2])
    second = driver.select_far_donors(camera, [0, 1, 2])
    assert first == second
    assert len(first["indices"]) == 2
    assert len(set(first["indices"])) == 2
    assert not set(first["indices"]) & {0, 1, 2}
    assert min(first["angles_deg"]) >= 150


def test_mask_comparison_uses_cross_view_union():
    driver = load_driver()
    correct = torch.tensor([[[True, True, False, False]]])
    wrong = torch.tensor([[[True, False, True, False]]])
    result = driver.mask_comparison(correct, wrong)
    assert result["mask_jaccard"] == pytest.approx(1 / 3)
    assert result["mask_churn"] == pytest.approx(2 / 3)


def test_mask_statistics_conserve_cross_key_counts():
    driver = load_driver()
    camera = yaw_camera([0, 30, 60])
    camera.T_world_from_camera[0, :, 0, 3] = torch.tensor([0.0, 0.25, 0.5])
    mask, stats = driver._mask_stats(camera, (2, 3), 1.5)
    batch, queries, _ = mask.shape
    expected = int(mask.sum()) / (batch * queries)
    assert stats["cross_keys_per_query"] == pytest.approx(expected)
    assert 0 <= stats["cross_key_retention"] <= 1
    assert stats["valid_pair_ratio"] == 1
    assert 0 <= stats["usable_query_pair_ratio"] <= 1
    assert 0 <= stats["in_bounds_query_pair_ratio"] <= 1


def test_mask_statistics_normalize_usable_pairs_across_batch():
    driver = load_driver()
    single = yaw_camera([0, 30, 60])
    single.T_world_from_camera[0, :, 0, 3] = torch.tensor([0.0, 0.25, 0.5])
    batched = CameraBatch(
        single.K.repeat(2, 1, 1, 1),
        single.T_world_from_camera.repeat(2, 1, 1, 1),
        single.image_size,
        single.sequence_kind,
    )
    _, single_stats = driver._mask_stats(single, (2, 3), 1.5)
    _, batched_stats = driver._mask_stats(batched, (2, 3), 1.5)
    assert batched_stats["usable_query_pair_ratio"] == pytest.approx(
        single_stats["usable_query_pair_ratio"]
    )


def test_mask_statistics_do_not_call_out_of_frame_lines_zero_key_queries(monkeypatch):
    driver = load_driver()
    camera = yaw_camera([0, 30])
    matrices = torch.zeros(1, 2, 2, 3, 3)
    matrices[..., 0] = 1
    matrices[..., 2] = 100  # x + 100 = 0 never intersects the patch extent.
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    monkeypatch.setattr(driver, "patch_fundamental_matrices", lambda *_: (matrices, valid))
    monkeypatch.setattr(
        driver,
        "epipolar_local_key_mask",
        lambda *_args, **_kwargs: (
            torch.zeros(1, 2, 2, dtype=torch.bool),
            torch.ones(1, 2, 2, dtype=torch.bool),
        ),
    )
    _, stats = driver._mask_stats(camera, (1, 1), 1.5)
    assert stats["in_bounds_query_nonempty_ratio"] == 0
    assert stats["zero_key_query_pair_count"] == 0


def test_output_gate_requires_thresholds_and_three_probe_directions():
    driver = load_driver()
    deltas = [
        {"psnr": 0.06, "ssim": 0.0006, "lpips": 0.001, "mae": 0.001},
        {"psnr": 0.07, "ssim": 0.0007, "lpips": 0.001, "mae": 0.001},
        {"psnr": 0.08, "ssim": 0.0008, "lpips": 0.001, "mae": 0.001},
        {"psnr": 0.04, "ssim": 0.0004, "lpips": 0.001, "mae": 0.001},
    ]
    assert driver.output_gate(deltas)["pass"]
    deltas[2]["lpips"] = -0.01
    assert not driver.output_gate(deltas)["pass"]


def test_phase_a_decision_stops_on_invalid_geometry_and_routes_insensitive_model():
    driver = load_driver()
    gate = {"pass": False}
    invalid = driver.phase_a_decision(
        ordinary_churn=0.1, far_churn=0.1, min_far_angle=90,
        in_bounds_nonempty=1.0, far_output_gate=gate,
    )
    assert invalid["verdict"] == "AUDIT_INVALID"
    next_phase = driver.phase_a_decision(
        ordinary_churn=0.1, far_churn=0.4, min_far_angle=90,
        in_bounds_nonempty=1.0, far_output_gate=gate,
    )
    assert next_phase["verdict"] == "PROCEED_PHASE_B"


def test_phase_a_decision_rejects_failed_integrity():
    driver = load_driver()
    decision = driver.phase_a_decision(
        ordinary_churn=0.1,
        far_churn=0.4,
        min_far_angle=90,
        in_bounds_nonempty=1.0,
        far_output_gate={"pass": False},
        integrity_pass=False,
    )
    assert decision == {"verdict": "AUDIT_INVALID", "proceed_phase_b": False}


def test_phase_b_next_action_stops_on_failed_integrity():
    driver = load_driver()
    assert driver.phase_b_next_action({"pass": False}, {"pass": False}) == "AUDIT_INVALID"
    assert driver.phase_b_next_action({"pass": False}, {"pass": True}) == "PROCEED_PHASE_C"
    assert driver.phase_b_next_action({"pass": True}, {"pass": True}) == "RUN_2000_STEP_REPLICATION"


def test_phase_b_frozen_hash_checks_detect_upstream_drift(tmp_path):
    driver = load_driver()
    campaign = tmp_path / "campaign"
    paths = {
        "seen_manifest": tmp_path / "seen.json",
        "baseline_eval": tmp_path / "baseline",
        "baseline_checkpoint": tmp_path / "baseline.pt",
        "bridge_checkpoint": tmp_path / "bridge.pt",
        "phase_a_summary": campaign / "analysis" / "phase_a_summary.json",
    }
    for path in paths.values():
        target = path / "evaluation_rows.jsonl" if path == paths["baseline_eval"] else path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(target), encoding="utf-8")
    protocol = {
        "seen_manifest": str(paths["seen_manifest"]),
        "seen_manifest_sha256": driver.sha256_file(paths["seen_manifest"]),
        "baseline_eval": str(paths["baseline_eval"]),
        "baseline_evaluation_rows_sha256": driver.sha256_file(paths["baseline_eval"] / "evaluation_rows.jsonl"),
        "baseline_checkpoint": str(paths["baseline_checkpoint"]),
        "baseline_checkpoint_sha256": driver.sha256_file(paths["baseline_checkpoint"]),
        "bridge_checkpoint": str(paths["bridge_checkpoint"]),
        "bridge_checkpoint_sha256": driver.sha256_file(paths["bridge_checkpoint"]),
        "phase_a_summary_sha256": driver.sha256_file(paths["phase_a_summary"]),
    }
    assert all(driver.phase_b_frozen_hash_checks(campaign, protocol).values())
    (paths["baseline_eval"] / "evaluation_rows.jsonl").write_text("drift", encoding="utf-8")
    checks = driver.phase_b_frozen_hash_checks(campaign, protocol)
    assert checks["baseline_evaluation_rows_hash"] is False


def test_phase_c_frozen_hash_checks_cover_both_baselines(tmp_path):
    driver = load_driver()
    campaign = tmp_path / "campaign"
    files = {
        "seen_manifest": tmp_path / "seen.json",
        "no_self_rows": tmp_path / "no_self" / "evaluation_rows.jsonl",
        "no_self_checkpoint": tmp_path / "no_self.pt",
        "self_allowed_rows": tmp_path / "self_allowed" / "evaluation_rows.jsonl",
        "bridge_checkpoint": tmp_path / "bridge.pt",
        "phase_b_summary": campaign / "analysis" / "phase_b_pilot_summary.json",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(path), encoding="utf-8")
    protocol = {
        "seen_manifest": str(files["seen_manifest"]),
        "seen_manifest_sha256": driver.sha256_file(files["seen_manifest"]),
        "no_self_baseline_eval": str(files["no_self_rows"].parent),
        "no_self_baseline_rows_sha256": driver.sha256_file(files["no_self_rows"]),
        "no_self_baseline_checkpoint": str(files["no_self_checkpoint"]),
        "no_self_baseline_checkpoint_sha256": driver.sha256_file(files["no_self_checkpoint"]),
        "self_allowed_baseline_eval": str(files["self_allowed_rows"].parent),
        "self_allowed_baseline_rows_sha256": driver.sha256_file(files["self_allowed_rows"]),
        "bridge_checkpoint": str(files["bridge_checkpoint"]),
        "bridge_checkpoint_sha256": driver.sha256_file(files["bridge_checkpoint"]),
        "phase_b_summary_sha256": driver.sha256_file(files["phase_b_summary"]),
    }
    checks = driver.phase_c_frozen_hash_checks(campaign, protocol)
    assert set(checks) == {
        "seen_manifest_hash", "no_self_baseline_rows_hash", "no_self_baseline_checkpoint_hash",
        "self_allowed_baseline_rows_hash", "bridge_checkpoint_hash", "phase_b_summary_hash",
    }
    assert all(checks.values())


def test_phase_b_config_changes_only_self_source_switch():
    driver = load_driver()
    config = driver.load_stage3_config(CONFIG)
    assert config.allow_self_view_source is False
    assert config.steps == 1000
    assert config.views == 4
    assert config.epipolar_attention == "local_band"
    assert config.epipolar_band == 1.5
    assert config.target_lr_dropout == 0.5


def test_phase_c_config_only_adds_preregistered_ranking_loss():
    driver = load_driver()
    no_self = driver.load_stage3_config(CONFIG).to_dict()
    ranked = driver.load_stage3_config(RANK_CONFIG).to_dict()
    differences = {
        key: (no_self[key], ranked[key])
        for key in no_self
        if no_self[key] != ranked[key]
    }
    assert differences == {
        "camera_rank_weight": (0.0, 0.1),
    }
    assert ranked["camera_rank_margin_ratio"] == 0.05


def test_phase_b_pilot_gate_enforces_camera_gain_and_no_self_attention():
    driver = load_driver()
    positive = [
        {"psnr": 0.10, "ssim": 0.001, "lpips": 0.001, "mae": 0.001}
        for _ in range(4)
    ]
    candidate = {
        "means": {"correct": {"psnr": 30.0, "ssim": 0.95, "lpips": 0.03, "mae": 0.01}},
        "deltas": {
            "target_drop": {"psnr": 5.0, "ssim": 0.06, "lpips": 0.02, "mae": 0.01},
            "shuffle_fusion": {"psnr": 0.10, "ssim": 0.001, "lpips": 0.001, "mae": 0.001},
            "remove": {"psnr": 0.10, "ssim": 0.001, "lpips": 0.001, "mae": 0.001},
        },
        "delta_rows": {"shuffle_fusion": positive, "remove": positive},
        "repeat_jitter": {metric: 0.0 for metric in driver.METRICS},
        "same_view_attention_mass": 0.0,
    }
    baseline = {
        "means": {"correct": {"psnr": 30.0, "ssim": 0.95, "lpips": 0.03, "mae": 0.01}},
        "deltas": {"shuffle_fusion": {"psnr": 0.01, "ssim": 0.0001, "lpips": 0.0, "mae": 0.0}},
    }
    assert driver.phase_b_pilot_gate(candidate, baseline, {"pass": True})["pass"]
    candidate["same_view_attention_mass"] = 1e-4
    assert not driver.phase_b_pilot_gate(candidate, baseline, {"pass": True})["pass"]


def test_phase_b_payload_accepts_legacy_baseline_condition_set(tmp_path):
    driver = load_driver()
    conditions = (
        "correct", "correct_repeat", "target_drop", "shuffle_fusion",
        "shuffle_geometry", "shuffle_all", "shuffle_pair",
    )
    rows = [
        {
            "condition": condition,
            "group_id": group,
            "psnr": 30.0,
            "ssim": 0.95,
            "lpips": 0.03,
            "mae": 0.01,
        }
        for condition in conditions
        for group in driver.PROBE_IDS
    ]
    (tmp_path / "evaluation_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    diagnostic = {"condition": "shuffle_fusion"}
    (tmp_path / "diagnostics.jsonl").write_text(json.dumps(diagnostic) + "\n", encoding="utf-8")
    payload = driver._phase_b_eval_payload(
        tmp_path,
        required_modes=("correct", "correct_repeat", "target_drop", "shuffle_fusion"),
    )
    assert set(payload["means"]) == {"correct", "correct_repeat", "target_drop", "shuffle_fusion"}
    assert "remove" not in payload["deltas"]
    assert payload["same_view_attention_mass"] is None
