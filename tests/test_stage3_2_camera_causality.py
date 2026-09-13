"""CPU-only contracts for the staged camera-causality audit."""
import importlib.util
from pathlib import Path

import pytest
import torch

from rl3dsr.models.wan.geometry_conditioning import CameraBatch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage3_2_camera_causality.py"


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
