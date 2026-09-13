import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage3_1_local_band_sufficiency.py"
SPEC = importlib.util.spec_from_file_location("local_band_sufficiency", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def metric_row(condition, view, *, psnr, ssim, lpips, mae):
    return {
        "condition": condition,
        "inference_seed": 3302,
        "group_id": f"chair:{view:03d}",
        "view_index": view,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "mae": mae,
    }


def grid(shuffle_psnr=0.1):
    rows = []
    for view in (0, 33, 66, 99):
        rows.append(metric_row("correct", view, psnr=30, ssim=.9, lpips=.1, mae=.01))
        rows.append(metric_row("correct_repeat", view, psnr=30, ssim=.9, lpips=.1, mae=.01))
        rows.append(metric_row("target_drop", view, psnr=25, ssim=.8, lpips=.15, mae=.02))
        rows.append(metric_row("shuffle_fusion", view, psnr=30 - shuffle_psnr, ssim=.9 - .001, lpips=.101, mae=.011))
        rows.append(metric_row("shuffle_geometry", view, psnr=29.9, ssim=.899, lpips=.101, mae=.011))
        rows.append(metric_row("shuffle_all", view, psnr=29.8, ssim=.898, lpips=.102, mae=.012))
        rows.append(metric_row("shuffle_pair", view, psnr=30.1, ssim=.901, lpips=.099, mae=.009))
    return rows


def test_paired_deltas_direction_and_grid():
    result = MODULE.paired_deltas(grid())
    assert result["deltas"]["target_drop"]["psnr"] == pytest.approx(5)
    assert result["deltas"]["target_drop"]["lpips"] == pytest.approx(.05)
    assert result["deltas"]["shuffle_fusion"]["psnr"] == pytest.approx(.1)
    assert max(result["repeat_jitter"].values()) == 0


def test_paired_deltas_rejects_missing_condition():
    rows = grid()[:-1]
    with pytest.raises(ValueError, match="missing paired condition"):
        MODULE.paired_deltas(rows)


def test_camera_diagnostic_scope_flags(tmp_path):
    rows = []
    flags = {
        "correct_repeat": (False, False),
        "target_drop": (False, False),
        "shuffle_fusion": (True, False),
        "shuffle_geometry": (False, True),
        "shuffle_all": (True, True),
        "shuffle_pair": (True, True),
    }
    for view in (0, 33, 66, 99):
        for condition, (fusion, geometry) in flags.items():
            rows.append({"condition": condition, "fusion_camera_changed": fusion, "geometry_camera_changed": geometry})
    path = tmp_path / "diagnostics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = MODULE.validate_camera_diagnostics(tmp_path)
    assert result["camera_scope_pass"] is True


def test_config_defaults_keep_global_backward_compatibility():
    payload = {"arm": "A3", "target_lr_dropout": .5}
    result = MODULE.config_defaults(payload)
    assert result["epipolar_attention"] == "global_bias"
    assert result["epipolar_band"] == 1.5


def test_validate_run_manifest_enforces_frozen_contract(tmp_path):
    config = {
        "arm": "A3",
        "train_scenes": ["chair"],
        "target_lr_dropout": 0.5,
        "epipolar_attention": "local_band",
        "epipolar_band": 1.5,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    bridge = tmp_path / "bridge.pt"
    bridge.write_bytes(b"bridge")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "config": config,
        "provenance": {
            "training_seed": 42,
            "bridge_checkpoint_sha256": MODULE.sha256_file(bridge),
            "rre_checkpoint": None,
        },
        **MODULE.EXPECTED_PARAMETER_COUNTS,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = MODULE.validate_run_manifest(
        run_dir, config_path=config_path, seed=42, bridge_checkpoint=bridge
    )
    assert result["pass"] is True
    assert all(result["checks"].values())

    manifest["config"]["target_lr_dropout"] = 0.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = MODULE.validate_run_manifest(
        run_dir, config_path=config_path, seed=42, bridge_checkpoint=bridge
    )
    assert result["pass"] is False
    assert result["checks"]["config"] is False
