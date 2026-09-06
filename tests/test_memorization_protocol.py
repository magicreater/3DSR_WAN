from __future__ import annotations

import pytest

from rl3dsr.validation.memorization import (
    camera_usage_verdict,
    memorization_verdict,
    nvs_consistency_verdict,
    select_seen_checkpoint,
)


def _row(step, psnr, ssim, lpips, *, condition="correct", seed=2201, view=0):
    return {
        "evaluation_group": "seen",
        "checkpoint_step": step,
        "condition": condition,
        "inference_seed": seed,
        "view_index": view,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
    }


def test_seen_selection_ignores_non_correct_rows_and_applies_all_ties():
    rows = []
    for view in range(4):
        rows.extend(
            (
                _row(250, 30.000, 0.90, 0.10, view=view),
                _row(500, 29.995, 0.91, 0.09, view=view),
                _row(750, 29.995, 0.91, 0.09, view=view),
                _row(1000, 99.0, 1.0, 0.0, condition="shuffled", view=view),
            )
        )
    result = select_seen_checkpoint(rows)
    assert result["selected"]["step"] == 500


def test_seen_selection_rejects_missing_views_or_mixed_seeds():
    rows = [_row(250, 30.0, 0.9, 0.1, view=view) for view in range(3)]
    with pytest.raises(ValueError, match="four unique"):
        select_seen_checkpoint(rows)
    rows.append(_row(250, 30.0, 0.9, 0.1, seed=2202, view=3))
    with pytest.raises(ValueError, match="shared fixed"):
        select_seen_checkpoint(rows)


def test_memorization_verdict_checks_every_seed_and_view():
    rows = []
    for seed in (2201, 2202, 2203, 2204):
        for view in range(4):
            rows.extend(
                (
                    _row(1000, 32.5, 0.965, 0.020, seed=seed, view=view),
                    _row(1000, 33.0, 0.970, 0.015, condition="vae_ceiling", seed=seed, view=view),
                    _row(1000, 28.0, 0.910, 0.120, condition="bicubic", seed=seed, view=view),
                    _row(1000, 29.0, 0.930, 0.080, condition="stage1", seed=seed, view=view),
                )
            )
    verdict = memorization_verdict(rows)
    assert verdict["STRICT_MEMORIZATION"] == "STRICT_MEMORIZATION_PASS"
    assert verdict["USEFUL_OVERFIT"] == "USEFUL_OVERFIT_PASS"
    rows[0]["psnr"] = 27.9
    failed = memorization_verdict(rows)
    assert failed["STRICT_MEMORIZATION"] == "STRICT_MEMORIZATION_FAIL"
    assert failed["USEFUL_OVERFIT"] == "USEFUL_OVERFIT_FAIL"


def test_camera_usage_requires_loss_and_response():
    passed = camera_usage_verdict(
        [{
            "correct_flow_loss": 0.8,
            "shuffled_flow_loss": 1.0,
            "correct_disabled_delta": 2e-5,
            "correct_shuffled_delta": 3e-5,
        }]
    )
    assert passed["status"] == "CAMERA_USAGE_PASS"


def test_nvs_requires_every_scene_and_full_seed_to_beat_both_comparators():
    rows = []
    for scene in ("chair", "lego", "drums"):
        rows.extend(
            (
                {"scene": scene, "arm": "hr", "psnr": 35.0, "lpips": 0.02},
                {"scene": scene, "arm": "stage1", "psnr": 25.0, "lpips": 0.15},
                {"scene": scene, "arm": "simple_rre_seed_42", "psnr": 26.0, "lpips": 0.13},
                {"scene": scene, "arm": "full_rre_seed_42", "psnr": 27.0, "lpips": 0.11},
                {"scene": scene, "arm": "full_rre_seed_43", "psnr": 27.1, "lpips": 0.10},
            )
        )
    assert nvs_consistency_verdict(rows)["status"] == "NVS_CONSISTENCY_PASS"
    overshoot = [dict(row) for row in rows]
    overshoot[3]["psnr"] = 45.0
    assert nvs_consistency_verdict(overshoot)["status"] == "NVS_CONSISTENCY_FAIL"
    rows[-1]["psnr"] = 24.0
    assert nvs_consistency_verdict(rows)["status"] == "NVS_CONSISTENCY_FAIL"
