"""Pure protocol logic for Stage 2 seen-view memorization and 3DGS checks."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean


METRICS = ("psnr", "ssim", "lpips")


def _averages(rows: list[dict], condition: str) -> dict[str, float]:
    selected = [row for row in rows if row["condition"] == condition]
    if not selected:
        raise ValueError(f"missing {condition} metric rows")
    return {metric: mean(float(row[metric]) for row in selected) for metric in METRICS}


def select_seen_checkpoint(rows: list[dict], *, psnr_tolerance_db: float = 0.01) -> dict:
    """Rank checkpoints only from correct-camera seen pure-noise rows."""

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("condition") == "correct" and row.get("evaluation_group") == "seen":
            grouped[int(row["checkpoint_step"])].append(row)
    if not grouped:
        raise ValueError("no correct-camera seen rows were supplied")
    selection_seeds = {int(row["inference_seed"]) for values in grouped.values() for row in values}
    if len(selection_seeds) != 1:
        raise ValueError("checkpoint selection requires one shared fixed inference seed")
    candidates = []
    for step, values in grouped.items():
        seeds = {int(row["inference_seed"]) for row in values}
        if len(seeds) != 1:
            raise ValueError("checkpoint selection requires exactly one fixed inference seed")
        views = [int(row["view_index"]) for row in values]
        if len(views) != 4 or len(set(views)) != 4:
            raise ValueError("each checkpoint requires exactly four unique seen views")
        metrics = {metric: mean(float(row[metric]) for row in values) for metric in METRICS}
        candidates.append({"step": step, "view_count": len(values), **metrics})
    best_psnr = max(row["psnr"] for row in candidates)
    eligible = [row for row in candidates if best_psnr - row["psnr"] <= psnr_tolerance_db]
    chosen = min(eligible, key=lambda row: (-row["ssim"], row["lpips"], row["step"]))
    return {"selected": chosen, "candidates": sorted(candidates, key=lambda row: row["step"])}


def camera_usage_verdict(control_rows: list[dict]) -> dict:
    if not control_rows:
        return {"status": "CAMERA_USAGE_UNAVAILABLE", "passed": False}
    correct = mean(float(row["correct_flow_loss"]) for row in control_rows)
    shuffled = mean(float(row["shuffled_flow_loss"]) for row in control_rows)
    disabled_delta = mean(float(row["correct_disabled_delta"]) for row in control_rows)
    shuffled_delta = mean(float(row["correct_shuffled_delta"]) for row in control_rows)
    passed = correct < shuffled and min(disabled_delta, shuffled_delta) > 1e-6
    return {
        "status": "CAMERA_USAGE_PASS" if passed else "CAMERA_USAGE_FAIL",
        "passed": passed,
        "mean_correct_flow_loss": correct,
        "mean_shuffled_flow_loss": shuffled,
        "mean_correct_disabled_delta": disabled_delta,
        "mean_correct_shuffled_delta": shuffled_delta,
    }


def memorization_verdict(rows: list[dict]) -> dict:
    """Apply strict-ceiling and useful-overfit gates to final four-seed rows."""

    averages = {condition: _averages(rows, condition) for condition in ("correct", "vae_ceiling", "bicubic", "stage1")}
    correct_rows = [row for row in rows if row["condition"] == "correct"]
    correct_pairs = {
        (int(row["inference_seed"]), int(row["view_index"])) for row in correct_rows
    }
    correct_seeds = {seed for seed, _view in correct_pairs}
    correct_views = {view for _seed, view in correct_pairs}
    if len(correct_seeds) != 4 or len(correct_views) != 4 or len(correct_pairs) != 16:
        raise ValueError("final memorization requires four inference seeds by four unique views")
    bicubic_by_view = {
        int(row["view_index"]): row for row in rows if row["condition"] == "bicubic"
    }
    stage1_by_seed_view = {
        (int(row["inference_seed"]), int(row["view_index"])): row
        for row in rows
        if row["condition"] == "stage1"
    }
    if not correct_rows or not bicubic_by_view or not stage1_by_seed_view:
        raise ValueError("final memorization rows are incomplete")

    strict_mean = (
        averages["correct"]["psnr"] >= averages["vae_ceiling"]["psnr"] - 1.0
        and averages["correct"]["ssim"] >= averages["vae_ceiling"]["ssim"] - 0.01
        and averages["correct"]["lpips"] <= averages["vae_ceiling"]["lpips"] + 0.01
    )
    strict_floor = all(
        float(row["psnr"]) >= float(bicubic_by_view[int(row["view_index"])]["psnr"])
        for row in correct_rows
    )
    strict = strict_mean and strict_floor

    useful_mean = (
        averages["correct"]["psnr"] > max(averages["bicubic"]["psnr"], averages["stage1"]["psnr"])
        and averages["correct"]["ssim"] > max(averages["bicubic"]["ssim"], averages["stage1"]["ssim"])
        and averages["correct"]["lpips"] < min(averages["bicubic"]["lpips"], averages["stage1"]["lpips"])
    )
    no_view_regression = True
    for row in correct_rows:
        view = int(row["view_index"])
        seed_view = (int(row["inference_seed"]), view)
        comparators = (bicubic_by_view[view], stage1_by_seed_view[seed_view])
        no_view_regression &= all(
            float(row["psnr"]) >= float(other["psnr"])
            and float(row["ssim"]) >= float(other["ssim"])
            and float(row["lpips"]) <= float(other["lpips"])
            for other in comparators
        )
    useful = useful_mean and no_view_regression
    return {
        "STRICT_MEMORIZATION": "STRICT_MEMORIZATION_PASS" if strict else "STRICT_MEMORIZATION_FAIL",
        "USEFUL_OVERFIT": "USEFUL_OVERFIT_PASS" if useful else "USEFUL_OVERFIT_FAIL",
        "strict_passed": strict,
        "useful_passed": useful,
        "strict_seed_view_psnr_floor_passed": strict_floor,
        "no_seed_view_regression": no_view_regression,
        "mean_metrics": averages,
    }


def nvs_consistency_verdict(rows: list[dict]) -> dict:
    """Require every full-RRE scene/seed to improve both direct comparators and HR gap."""

    by_scene_arm = {(row["scene"], row["arm"]): row for row in rows}
    scenes = sorted({row["scene"] for row in rows})
    checks = []
    for scene in scenes:
        hr = by_scene_arm[(scene, "hr")]
        for arm in ("full_rre_seed_42", "full_rre_seed_43"):
            current = by_scene_arm[(scene, arm)]
            stage1 = by_scene_arm[(scene, "stage1")]
            simple = by_scene_arm[(scene, "simple_rre_seed_42")]
            psnr_ok = float(current["psnr"]) >= max(float(stage1["psnr"]), float(simple["psnr"]))
            lpips_ok = float(current["lpips"]) <= min(float(stage1["lpips"]), float(simple["lpips"]))
            hr_psnr_gap = abs(float(hr["psnr"]) - float(current["psnr"]))
            comparator_psnr_gap = min(
                abs(float(hr["psnr"]) - float(stage1["psnr"])),
                abs(float(hr["psnr"]) - float(simple["psnr"])),
            )
            hr_lpips_gap = abs(float(current["lpips"]) - float(hr["lpips"]))
            comparator_lpips_gap = min(
                abs(float(stage1["lpips"]) - float(hr["lpips"])),
                abs(float(simple["lpips"]) - float(hr["lpips"])),
            )
            gap_ok = hr_psnr_gap <= comparator_psnr_gap and hr_lpips_gap <= comparator_lpips_gap
            checks.append({
                "scene": scene,
                "arm": arm,
                "psnr_no_regression": psnr_ok,
                "lpips_no_regression": lpips_ok,
                "hr_gap_reduced": gap_ok,
            })
    passed = bool(checks) and all(all(value for key, value in row.items() if key not in {"scene", "arm"}) for row in checks)
    return {
        "status": "NVS_CONSISTENCY_PASS" if passed else "NVS_CONSISTENCY_FAIL",
        "passed": passed,
        "checks": checks,
    }
