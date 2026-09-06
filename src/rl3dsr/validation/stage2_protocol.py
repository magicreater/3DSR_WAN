"""Deterministic split and checkpoint-selection rules for Stage 2."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


PROTOCOL_VERSION = 1
SCENES = ("chair", "lego", "drums")
ARMS = ("baseline_no_geometry", "rre_geometry", "plucker_geometry")
GEOMETRY_ARMS = ("rre_geometry", "plucker_geometry")
TRAIN_VIEW_INDICES = (0, 33, 66, 99)
FAR_VIEW_INDICES = (0, 50, 100, 150)
EXPECTED_NEAR_VIEW_INDICES = {
    "chair": (30, 7, 14, 41),
    "lego": (41, 36, 58, 43),
    "drums": (1, 72, 98, 18),
}
CANDIDATE_STEPS = (500, 1000, 1500, 2000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_digest(payload: dict) -> str:
    value = dict(payload)
    value.pop("protocol_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluation_groups(
    phase: str,
    *,
    seen: Iterable[int] | None = None,
    near: Iterable[int] | None = None,
    far: Iterable[int] | None = None,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    if phase == "selection":
        if seen is None or near is None:
            raise ValueError("selection requires seen and near view indices")
        if far is not None:
            raise ValueError("selection must not receive far view indices")
        return (
            ("seen", "train", tuple(int(value) for value in seen)),
            ("near-held-out", "train", tuple(int(value) for value in near)),
        )
    if phase == "final-test":
        if far is None:
            raise ValueError("final-test requires far view indices")
        if seen is not None or near is not None:
            raise ValueError("final-test must not receive seen or near view indices")
        return (("far-held-out", "test", tuple(int(value) for value in far)),)
    raise ValueError(f"unknown evaluation phase: {phase}")


def _camera_center(frame: dict) -> tuple[float, float, float]:
    matrix = frame.get("transform_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
    ):
        raise ValueError("transform_matrix must be 4x4")
    return tuple(float(matrix[row][3]) for row in range(3))


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _angular_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    first_norm = _norm(first)
    second_norm = _norm(second)
    if first_norm <= 0 or second_norm <= 0:
        raise ValueError("camera center cannot be the origin")
    cosine = sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)
    return math.acos(max(-1.0, min(1.0, cosine)))


def _translation_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def select_near_views(
    transforms_path: Path,
    train_indices: Iterable[int] = TRAIN_VIEW_INDICES,
) -> tuple[tuple[int, ...], list[dict]]:
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("transforms file must contain a frames list")
    train = tuple(int(index) for index in train_indices)
    if len(train) != 4 or len(set(train)) != 4:
        raise ValueError("Stage 2 requires four unique training views")
    if min(train) < 0 or max(train) >= len(frames):
        raise IndexError("training view index is outside transforms_train")
    centers = [_camera_center(frame) for frame in frames]
    excluded = set(train)
    selected = []
    details = []
    for train_index in train:
        candidates = []
        for candidate_index, center in enumerate(centers):
            if candidate_index in excluded:
                continue
            candidates.append((
                _angular_distance(centers[train_index], center),
                _translation_distance(centers[train_index], center),
                candidate_index,
            ))
        angle, translation, candidate_index = min(candidates)
        selected.append(candidate_index)
        details.append({
            "train_view_index": train_index,
            "near_view_index": candidate_index,
            "angular_distance_radians": angle,
            "translation_distance": translation,
        })
    if len(set(selected)) != len(selected):
        raise RuntimeError(
            "independent nearest-camera selection produced duplicate validation views"
        )
    return tuple(selected), details


def build_protocol(dataset_root: Path) -> dict:
    scenes = {}
    for scene in SCENES:
        scene_root = dataset_root / scene
        train_path = scene_root / "transforms_train.json"
        test_path = scene_root / "transforms_test.json"
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(f"missing NeRF Synthetic transforms for {scene}")
        near, distances = select_near_views(train_path)
        if near != EXPECTED_NEAR_VIEW_INDICES[scene]:
            raise RuntimeError(
                f"{scene} near views drifted: "
                f"expected {EXPECTED_NEAR_VIEW_INDICES[scene]}, got {near}"
            )
        if set(near) & set(TRAIN_VIEW_INDICES):
            raise RuntimeError(f"{scene} seen and near views overlap")
        scenes[scene] = {
            "scene_root": str(scene_root.resolve()),
            "train_transforms": str(train_path.resolve()),
            "train_transforms_sha256": sha256_file(train_path),
            "test_transforms": str(test_path.resolve()),
            "test_transforms_sha256": sha256_file(test_path),
            "seen_view_indices": list(TRAIN_VIEW_INDICES),
            "near_view_indices": list(near),
            "far_view_indices": list(FAR_VIEW_INDICES),
            "near_distances": distances,
        }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "date": "2026-09-03",
        "scenes": scenes,
        "candidate_steps": list(CANDIDATE_STEPS),
        "selection": {
            "evaluation_group": "near-held-out",
            "primary": "mean_correct_psnr",
            "psnr_tie_tolerance_db": 0.01,
            "tie_breakers": [
                "mean_correct_ssim_desc",
                "mean_correct_lpips_asc",
                "step_asc",
            ],
        },
        "historical_far_disclosure": (
            "The far cameras were evaluated in the 2026-09-02 campaign and "
            "are not pristine project-history holdouts."
        ),
    }
    result["protocol_sha256"] = protocol_digest(result)
    return result


def select_checkpoint(
    rows: list[dict],
    expected_views: Iterable[int],
    candidate_steps: Iterable[int] = CANDIDATE_STEPS,
) -> tuple[dict, list[dict]]:
    expected = set(int(index) for index in expected_views)
    summaries = []
    for step in (int(value) for value in candidate_steps):
        selected = [
            row
            for row in rows
            if row.get("evaluation_group") == "near-held-out"
            and row.get("condition") == "correct"
            and int(row.get("checkpoint_step", -1)) == step
        ]
        actual = {int(row["view_index"]) for row in selected}
        if actual != expected:
            raise RuntimeError(
                f"checkpoint {step} near views {sorted(actual)} "
                f"!= expected {sorted(expected)}"
            )

        def mean(key: str) -> float:
            values = [float(row[key]) for row in selected]
            if not values or any(not math.isfinite(value) for value in values):
                raise RuntimeError(f"checkpoint {step} has invalid {key}")
            return statistics.fmean(values)

        summaries.append({
            "step": step,
            "mean_correct_psnr": mean("psnr"),
            "mean_correct_ssim": mean("ssim"),
            "mean_correct_lpips": mean("lpips"),
            "mean_correct_mae": mean("mae"),
            "near_view_count": len(selected),
        })
    maximum_psnr = max(row["mean_correct_psnr"] for row in summaries)
    eligible = [
        row
        for row in summaries
        if maximum_psnr - row["mean_correct_psnr"] <= 0.01
    ]
    chosen = min(
        eligible,
        key=lambda row: (
            -row["mean_correct_ssim"],
            row["mean_correct_lpips"],
            row["step"],
        ),
    )
    audit = [
        {
            **row,
            "psnr_tie_eligible": row in eligible,
            "selected": row is chosen,
        }
        for row in summaries
    ]
    return dict(chosen), audit
