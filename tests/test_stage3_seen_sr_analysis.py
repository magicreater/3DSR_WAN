"""Contracts for the fixed Stage 3 seen-view analysis."""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def analysis():
    path = Path(__file__).resolve().parents[1] / "scripts/stage3_seen_sr_analysis.py"
    spec = importlib.util.spec_from_file_location("stage3_seen_sr_analysis_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(scene, quality, *, condition="correct"):
    return {
        "scene": scene,
        "view_index": 0,
        "inference_seed": 3302,
        "condition": condition,
        "psnr": quality,
        "ssim": quality / 100,
        "lpips": 1 - quality / 100,
        "mae": 1 - quality / 100,
    }


def campaign(analysis):
    data = {"full": {}, "baseline": {}, "intervention": {}}
    full_quality = {"A0": 27.0, "A1": 28.0, "A2": 29.0, "A3": 30.0}
    for arm in analysis.ARMS:
        for seed in analysis.TRAIN_SEEDS:
            key = (arm, seed)
            data["full"][key] = [metrics(scene, full_quality[arm]) for scene in analysis.SCENES]
            data["baseline"][key] = [
                row
                for scene in analysis.SCENES
                for row in (metrics(scene, 25.0, condition="bicubic"),
                            metrics(scene, 35.0, condition="vae_ceiling"))
            ]
            condition_quality = {
                "correct": full_quality[arm],
                "correct_repeat": full_quality[arm],
                "remove": full_quality[arm] - (1.0 if arm in ("A2", "A3") else 0.0),
                "duplicate": full_quality[arm] - (1.0 if arm in ("A2", "A3") else 0.0),
                "shuffle_camera": full_quality[arm] - (0.5 if arm == "A3" else 0.0),
            }
            data["intervention"][key] = [
                metrics(scene, value, condition=condition)
                for scene in analysis.SCENES
                for condition, value in condition_quality.items()
            ]
    return data


def test_summary_passes_only_when_all_fixed_contributions_hold(analysis):
    data = campaign(analysis)
    assert analysis.build_summary(data)["verdicts"] == {
        "SR_FIT_PASS": True,
        "CROSS_VIEW_PASS": True,
        "GEOMETRY_PASS": True,
        "SEEN_SR_EFFECTIVE": True,
    }
    data["full"][("A3", 43)] = [metrics(scene, 29.0) for scene in analysis.SCENES]
    verdicts = analysis.build_summary(data)["verdicts"]
    assert verdicts["GEOMETRY_PASS"] is False
    assert verdicts["SEEN_SR_EFFECTIVE"] is False


def test_row_validation_rejects_duplicate_and_nonfinite_metrics(analysis):
    rows = [metrics(scene, 30.0) for scene in analysis.SCENES]
    analysis.validate_rows(rows, len(rows), {"correct"})
    with pytest.raises(ValueError, match="duplicate"):
        analysis.validate_rows(rows + [dict(rows[0])], len(rows) + 1, {"correct"})
    invalid = [dict(row) for row in rows]
    invalid[0]["lpips"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        analysis.validate_rows(invalid, len(invalid), {"correct"})


def test_contact_sheet_has_distinct_full_and_fixed_center_crop(analysis, tmp_path):
    source = Image.new("RGB", (128, 128), "red")
    ImageDraw.Draw(source).rectangle((16, 16, 111, 111), fill="green")
    view = "view_000"
    refs = tmp_path / "full" / "A3" / "seed42" / "images" / "chair" / view / "reference"
    refs.mkdir(parents=True)
    for name in ("hr", "lr_nearest", "bicubic", "vae_ceiling"):
        source.save(refs / f"{name}.png")
    for arm in analysis.ARMS:
        path = tmp_path / "full" / arm / "seed42" / "images" / "chair" / view / "seed_3302"
        path.mkdir(parents=True)
        source.save(path / "correct.png")
    path = tmp_path / "qualitative" / "A3" / "seed42" / "images" / "chair" / view / "seed_3302"
    path.mkdir(parents=True)
    source.save(path / "remove.png")
    source.save(path / "shuffle_camera.png")
    output = tmp_path / "analysis"
    output.mkdir()

    analysis.contact_sheets(
        tmp_path,
        output,
        {"chair": {"best": 0, "median": 0, "worst": 0}},
    )

    full = Image.open(output / "qualitative_chair_full.png")
    crop = Image.open(output / "qualitative_chair.png")
    assert full.size == (130 + 12 * 128, 32 + 3 * (128 + 24))
    assert crop.size == (130 + 12 * 96, 32 + 3 * (96 + 24))
    assert full.getpixel((130, 32)) == (255, 0, 0)
    assert crop.getpixel((130, 32)) == (0, 128, 0)
