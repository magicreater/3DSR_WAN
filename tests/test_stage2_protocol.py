import json
import math

import pytest

from rl3dsr.validation.stage2_protocol import (
    evaluation_groups,
    protocol_digest,
    select_checkpoint,
    select_near_views,
)


def _frame(x, y, z=0.0):
    return {
        "transform_matrix": [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }


def test_near_views_use_camera_angle_then_translation_and_index(tmp_path):
    path = tmp_path / "transforms_train.json"
    frames = [
        _frame(1, 0),
        _frame(0, 1),
        _frame(-1, 0),
        _frame(0, -1),
        _frame(2, 0.1),
        _frame(-0.1, 2),
        _frame(-2, -0.1),
        _frame(0.1, -2),
    ]
    path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    selected, details = select_near_views(path, (0, 1, 2, 3))
    assert selected == (4, 5, 6, 7)
    assert [row["train_view_index"] for row in details] == [0, 1, 2, 3]
    assert all(row["angular_distance_radians"] < 0.06 for row in details)


def test_near_views_reject_duplicate_independent_matches(tmp_path):
    path = tmp_path / "transforms_train.json"
    frames = [
        _frame(1, 0),
        _frame(1, 0.01),
        _frame(-1, 0),
        _frame(0, -1),
        _frame(2, 0.005),
        _frame(-2, 0.1),
        _frame(0.1, -2),
        _frame(0, 2),
    ]
    path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        select_near_views(path, (0, 1, 2, 3))


def _metric_rows(step, psnr, ssim, lpips, *, group="near-held-out"):
    return [
        {
            "checkpoint_step": step,
            "evaluation_group": group,
            "condition": "correct",
            "view_index": view,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "mae": 0.1,
        }
        for view in (10, 11, 12, 13)
    ]


def test_checkpoint_selection_uses_only_near_rows_and_psnr_tie_breaks():
    rows = []
    rows += _metric_rows(500, 20.000, 0.80, 0.20)
    rows += _metric_rows(1000, 20.009, 0.79, 0.10)
    rows += _metric_rows(1500, 20.005, 0.82, 0.30)
    rows += _metric_rows(2000, 19.0, 0.99, 0.01)
    rows += _metric_rows(2000, 99.0, 0.99, 0.01, group="far-held-out")
    chosen, audit = select_checkpoint(rows, (10, 11, 12, 13))
    assert chosen["step"] == 1500
    assert [row["step"] for row in audit if row["psnr_tie_eligible"]] == [500, 1000, 1500]
    assert [row["step"] for row in audit if row["selected"]] == [1500]


def test_checkpoint_selection_uses_lpips_then_earlier_step():
    rows = []
    rows += _metric_rows(500, 20.0, 0.8, 0.15)
    rows += _metric_rows(1000, 20.0, 0.8, 0.10)
    rows += _metric_rows(1500, 20.0, 0.8, 0.10)
    rows += _metric_rows(2000, 19.0, 0.9, 0.01)
    chosen, _ = select_checkpoint(rows, (10, 11, 12, 13))
    assert chosen["step"] == 1000


def test_checkpoint_selection_requires_all_near_views():
    rows = _metric_rows(500, 20.0, 0.8, 0.1)[:-1]
    rows += _metric_rows(1000, 20.0, 0.8, 0.1)
    rows += _metric_rows(1500, 20.0, 0.8, 0.1)
    rows += _metric_rows(2000, 20.0, 0.8, 0.1)
    with pytest.raises(RuntimeError, match="near views"):
        select_checkpoint(rows, (10, 11, 12, 13))


def test_protocol_digest_detects_dataset_hash_drift():
    payload = {"protocol_version": 1, "scenes": {"chair": {"hash": "a"}}}
    first = protocol_digest(payload)
    payload["scenes"]["chair"]["hash"] = "b"
    second = protocol_digest(payload)
    assert first != second
    assert all(math.isfinite(float(int(value, 16))) for value in (first, second))


def test_selection_groups_cannot_include_far_views():
    groups = evaluation_groups(
        "selection",
        seen=(0, 33, 66, 99),
        near=(30, 7, 14, 41),
    )
    assert [group[0] for group in groups] == ["seen", "near-held-out"]
    with pytest.raises(ValueError, match="must not receive far"):
        evaluation_groups(
            "selection",
            seen=(0, 33, 66, 99),
            near=(30, 7, 14, 41),
            far=(0, 50, 100, 150),
        )


def test_final_test_groups_cannot_include_seen_or_near_views():
    groups = evaluation_groups("final-test", far=(0, 50, 100, 150))
    assert groups == (("far-held-out", "test", (0, 50, 100, 150)),)
    with pytest.raises(ValueError, match="must not receive seen or near"):
        evaluation_groups(
            "final-test",
            seen=(0, 33, 66, 99),
            far=(0, 50, 100, 150),
        )
