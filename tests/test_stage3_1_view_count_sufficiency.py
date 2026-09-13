"""CPU-only checks for the Stage 3.1 view-count protocol."""
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "stage3_1_view_count_sufficiency.py"
SPEC = importlib.util.spec_from_file_location("stage3_1_view_count_sufficiency", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _manifest(path: Path, views: int) -> None:
    groups = []
    for anchor in range(100):
        indices = [anchor] + [((anchor + offset) % 100) for offset in range(1, views)]
        groups.append({"id": f"chair:{anchor:03d}", "scene": "chair", "anchor": anchor, "indices": indices})
    path.write_text(json.dumps({"version": 1, "scope": "seen_train_sr", "groups": groups}), encoding="utf-8")


def test_manifest_groups_require_exact_chair_group_count(tmp_path):
    path = tmp_path / "manifest.json"
    _manifest(path, 4)
    assert len(MODULE._manifest_groups(path)) == 100
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["groups"] = payload["groups"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 100"):
        MODULE._manifest_groups(path)


def test_frozen_json_rejects_protocol_drift(tmp_path):
    path = tmp_path / "protocol.json"
    MODULE.write_frozen_json(path, {"version": 1})
    MODULE.write_frozen_json(path, {"version": 1})
    with pytest.raises(RuntimeError, match="frozen artifact drift"):
        MODULE.write_frozen_json(path, {"version": 2})


def test_gate_requires_both_auxiliary_and_fusion_camera_effects():
    def cell(remove_psnr, fusion_psnr, remove_ssim=0.1, fusion_ssim=0.01):
        delta_rows = {
            "remove": [{"psnr": remove_psnr, "ssim": remove_ssim, "lpips": 0.1, "mae": 0.1}] * 4,
            "target_drop": [{"psnr": 5.0, "ssim": 0.1, "lpips": 0.1, "mae": 0.1}] * 4,
            "shuffle_fusion": [{"psnr": fusion_psnr, "ssim": fusion_ssim, "lpips": 0.01, "mae": 0.01}] * 4,
        }
        return {
            "means": {"correct": {"psnr": 28.0, "ssim": 0.9, "lpips": 0.03, "mae": 0.01}},
            "deltas": {name: {metric: sum(row[metric] for row in rows) / len(rows) for metric in MODULE.METRICS} for name, rows in delta_rows.items()},
            "delta_rows": delta_rows,
        }

    v4 = cell(1.0, 0.01)
    v8 = cell(2.0, 0.02, remove_ssim=0.2, fusion_ssim=0.02)
    assert MODULE._gate({"v4": v4, "v8": v8}, {"pass": True})["pilot_pass"]
    v8["delta_rows"]["shuffle_fusion"] = [
        {"psnr": 0.01, "ssim": 0.02, "lpips": 0.01, "mae": 0.01}
    ] * 4
    assert not MODULE._gate({"v4": v4, "v8": v8}, {"pass": True})["pilot_pass"]


def test_gate_requires_integrity():
    cell = {
        "means": {"correct": {"psnr": 28.0, "ssim": 0.9, "lpips": 0.03, "mae": 0.01}},
        "deltas": {
            "target_drop": {"psnr": 5.0, "ssim": 0.1},
        },
        "delta_rows": {
            "remove": [{"psnr": 1.0, "ssim": 0.1, "lpips": 0.1, "mae": 0.1}] * 4,
            "shuffle_fusion": [{"psnr": 0.1, "ssim": 0.01, "lpips": 0.01, "mae": 0.01}] * 4,
        },
    }
    assert not MODULE._gate({"v4": cell, "v8": cell}, {"pass": False})["pilot_pass"]


def test_preflight_requires_one_cell():
    args = MODULE.build_parser().parse_args([
        "preflight",
        "--campaign-root", "campaign",
    ])
    assert args.cell is None
