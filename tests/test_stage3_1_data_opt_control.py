import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def control():
    path = Path(__file__).resolve().parents[1] / "scripts" / "stage3_1_data_opt_control.py"
    spec = importlib.util.spec_from_file_location("stage3_1_data_opt_control", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(control, condition, group_id, *, psnr, ssim, lpips, mae):
    return {
        "condition": condition,
        "inference_seed": control.INFERENCE_SEED,
        "group_id": group_id,
        "view_index": int(group_id.split(":", 1)[1]),
        "step": 1000,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "mae": mae,
    }


def test_metric_delta_uses_good_direction(control):
    correct = {"psnr": 30.0, "ssim": 0.9, "lpips": 0.1, "mae": 0.02}
    changed = {"psnr": 29.0, "ssim": 0.8, "lpips": 0.2, "mae": 0.03}
    assert control._metric_delta(correct, changed, "psnr") == 1.0
    assert control._metric_delta(correct, changed, "ssim") == pytest.approx(0.1)
    assert control._metric_delta(correct, changed, "lpips") == pytest.approx(0.1)
    assert control._metric_delta(correct, changed, "mae") == pytest.approx(0.01)


def test_validate_eval_rows_requires_exact_fixed_probe(control, tmp_path):
    output = tmp_path / "eval"
    output.mkdir()
    rows = []
    for group_id in control.PROBE_IDS:
        rows.extend(
            _row(control, condition, group_id, psnr=30.0, ssim=0.9, lpips=0.1, mae=0.02)
            for condition in control.CONDITIONS
        )
    output.joinpath("evaluation_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output.joinpath("baseline_rows.jsonl").write_text(
        "".join(json.dumps({"condition": "bicubic"}) + "\n" for _ in range(8)), encoding="utf-8"
    )
    loaded, indexed = control._validate_eval_rows("chair_1000", output, 1000)
    assert len(loaded) == 16
    assert len(indexed) == 16

    rows.pop()
    output.joinpath("evaluation_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="row counts"):
        control._validate_eval_rows("chair_1000", output, 1000)


def test_control_manifest_builder_keeps_probe_records(control, tmp_path):
    source = {
        "dataset_root": "/data/nerf_synthetic",
        "datasets": {"chair": {"views": 100, "hash": "chair"}},
        "groups": [
            {"id": group_id, "scene": "chair", "anchor": int(group_id[-3:]), "indices": [0, 1, 2, 3]}
            for group_id in control.PROBE_IDS
        ],
    }
    config = {"train_scenes": ["chair"], "image_size": 256, "scale": 4, "views": 4}
    manifest = control._make_control_manifest(source, config, tmp_path / "source.json")
    assert [row["id"] for row in manifest["groups"]] == list(control.PROBE_IDS)
    assert manifest["subsets"]["probe"] == list(control.PROBE_IDS)
    assert manifest["sampling_signature"]["train_scenes"] == ["chair"]


def test_inspect_configs_has_four_cells(control):
    result = control.inspect_configs(Path(__file__).resolve().parents[1])
    assert set(result["cells"]) == {"chair_1000", "chair_2000", "five_1000", "five_2000"}
    assert all(item["target_lr_dropout"] == 0.5 for item in result["cells"].values())
