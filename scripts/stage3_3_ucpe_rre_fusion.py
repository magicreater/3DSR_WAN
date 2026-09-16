#!/usr/bin/env python3
"""Run the fail-closed Stage 3.3 UCPE/RRE fusion validation ladder.

The driver never launches Stage 4 or 4DSR.  Every cell is model-only
initialized from the frozen Phase-C checkpoint; interrupted cells resume only
from their own exact checkpoint and immutable manifest.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from statistics import mean

import torch

from rl3dsr.validation.stage3_protocol import Stage3Config, load_stage3_config


AUDIT_BASELINE = "883c55ba30837b11889322e6d757d0aaaf8f7ec7"
UCPE_SOURCE_COMMIT = "d992f1807803ba99331e807e8f018ed552886afd"
PHASE_C_CHECKPOINT_SHA256 = "49730b5f4fcb12009cc3d249a5814884e082b404ba482ec746a23715980cdd47"
PHASE_C_PROTOCOL_SHA256 = "dd7c5b5df2c8f344ebd0d1d816870c9ef46f3b0e0fccf60c166b0e2293c9eada"
SEEN_MANIFEST_SHA256 = "f99c3e21921fd255df34a48b6c6991da24e0d1f022f7dae096d1d647975e457b"
BRIDGE_SHA256 = "af964951c53f35e0bed55a1af5229546fb63a4732e84f96d97105fa9a06ff83d"
PHASE_C_CORRECT = {"psnr": 28.459492013270047, "ssim": 0.9413247108459473}
INFERENCE_SEED = 3302
PROBE_IDS = ("chair:000", "chair:033", "chair:066", "chair:099")
MODES = (
    "correct",
    "correct_repeat",
    "remove",
    "target_drop",
    "shuffle_fusion",
    "target_drop_shuffle_fusion",
    "mispaired_lr",
    "mispaired_camera",
    "joint_permute",
    "fusion_camera_dose_half",
    "fusion_camera_dose_full",
    "shuffle_geometry",
)
METRICS = ("psnr", "ssim")
NUMERICAL_TOLERANCE = 1e-7
FIVE_TRAIN_SCENES = ("chair", "lego", "drums", "hotdog", "mic")


@dataclass(frozen=True)
class Cell:
    name: str
    config_name: str
    seed: int
    reset_fusion: bool


FIXED_CELLS = {
    "h0": Cell("h0", "A3_rank_weight1_chair_200.json", 42, False),
    "a3_pilot": Cell("a3_pilot", "A3_rank_weight1_chair_1000.json", 42, False),
    "a4_pilot": Cell("a4_pilot", "A4_rre_epipolar_chair_1000.json", 42, True),
    "a5_pilot": Cell("a5_pilot", "A5_pairing_chair_1000.json", 42, True),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_frozen_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"immutable artifact drift: {path}")
        return
    path.write_text(text, encoding="utf-8")


def write_frozen_json(path: Path, payload: object) -> None:
    write_frozen_text(path, _json_text(payload))


def copy_or_verify(source: Path, destination: Path) -> None:
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != data:
            raise RuntimeError(f"immutable copy drift: {destination}")
        return
    destination.write_bytes(data)


def _git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")


def _config_path(args, cell: Cell) -> Path:
    return args.campaign_root / "config" / cell.config_name


def _train_dir(args, cell_name: str) -> Path:
    return args.campaign_root / "train" / cell_name


def _eval_dir(args, cell_name: str) -> Path:
    return args.campaign_root / "eval" / cell_name


def prepare(args) -> dict:
    root = args.repo_root.resolve()
    revision = _git_revision(root)
    if not _is_ancestor(root, AUDIT_BASELINE, revision):
        raise RuntimeError("Stage 3.3 branch does not descend from the frozen audit baseline")
    frozen = {
        "phase_c_checkpoint": (args.phase_c_checkpoint, PHASE_C_CHECKPOINT_SHA256),
        "phase_c_protocol": (args.phase_c_protocol, PHASE_C_PROTOCOL_SHA256),
        "seen_manifest": (args.seen_manifest, SEEN_MANIFEST_SHA256),
        "bridge_checkpoint": (args.bridge_checkpoint, BRIDGE_SHA256),
    }
    for label, (path, expected) in frozen.items():
        _assert_hash(path, expected, label)
    seen = read_json(args.seen_manifest)
    if seen.get("dataset_root") != str(args.dataset_root.resolve()):
        raise RuntimeError("frozen seen manifest dataset root drift")
    for scene, hashes in seen.get("datasets", {}).items():
        root_scene = args.dataset_root / scene
        _assert_hash(
            root_scene / "transforms_train.json",
            hashes["transforms_train_sha256"], f"{scene} transforms",
        )
        if sha256_tree(root_scene / "train") != hashes["train_images_sha256"]:
            raise RuntimeError(f"{scene} image data hash mismatch")
    for cell in FIXED_CELLS.values():
        source = root / "configs" / "stage3_3" / cell.config_name
        destination = _config_path(args, cell)
        copy_or_verify(source, destination)
        config = load_stage3_config(destination)
        if (
            config.camera_rank_weight != 1.0
            or config.allow_self_view_source
            or config.target_lr_dropout != 0.5
            or config.views != 4
            or config.train_scenes != ("chair",)
        ):
            raise RuntimeError(f"Stage 3.3 config drift: {cell.name}")
    official_weight = None
    if args.official_ucpe_weight is not None:
        if not args.official_ucpe_weight.is_file():
            raise FileNotFoundError(args.official_ucpe_weight)
        official_weight = {
            "status": "available_for_parity_only",
            "path": str(args.official_ucpe_weight.resolve()),
            "sha256": sha256_file(args.official_ucpe_weight),
        }
    else:
        official_weight = {
            "status": "unavailable_auth_redirect",
            "path": None,
            "sha256": None,
        }
    source_paths = (
        root / "scripts" / "stage3_3_ucpe_rre_fusion.py",
        root / "scripts" / "stage3_experiment.py",
        root / "scripts" / "convert_ucpe_checkpoint.py",
        root / "src" / "rl3dsr" / "models" / "wan" / "geometry_conditioning.py",
        root / "src" / "rl3dsr" / "models" / "wan" / "lr_fusion.py",
        root / "src" / "rl3dsr" / "models" / "wan" / "stage3.py",
        root / "src" / "rl3dsr" / "validation" / "stage3_protocol.py",
    )
    source_manifest = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in source_paths
    }
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).splitlines()
    protocol = {
        "schema_version": 1,
        "scope": "stage3_3_ucpe_rre_fusion",
        "audit_baseline": AUDIT_BASELINE,
        "git_revision": revision,
        "git_status": git_status,
        "source_sha256": source_manifest,
        "training_parent": {
            label: {"path": str(path.resolve()), "sha256": expected}
            for label, (path, expected) in frozen.items()
        },
        "probe_ids": list(PROBE_IDS),
        "inference_seed": INFERENCE_SEED,
        "modes": list(MODES),
        "phase_c_correct": PHASE_C_CORRECT,
        "gates": {
            "correct_psnr_floor_delta": -0.10,
            "correct_ssim_floor_delta": -0.001,
            "remove_psnr": 1.0,
            "target_drop_psnr": 3.0,
            "fusion_psnr": 0.05,
            "fusion_ssim": 0.0005,
            "directional_probes": 3,
            "wrong_correct_ratio": 0.05,
            "rank_active_max": 0.5,
            "dose_monotonic_probes": 3,
            "numerical_tolerance": NUMERICAL_TOLERANCE,
        },
        "ucpe": {
            "repository": "https://github.com/chengzhag/UCPE",
            "source_commit": UCPE_SOURCE_COMMIT,
            "paper": "https://arxiv.org/abs/2512.07237",
            "weight_source": (
                "https://monashuni-my.sharepoint.com/:f:/g/personal/"
                "cheng_zhang_monash_edu/IgCoTNrYOJRJRKtk5A6I1yiCAR9c64-BOrsId5GYsUxE9y4"
                "?e=hD26qU"
            ),
            "converter_version": 1,
            "official_weight": official_weight,
            "experiment_scope": "pinhole project-specific A4/A5; no pretrained UCPE injection",
        },
        "forbidden": ["10000-step training", "Stage 4", "4DSR"],
    }
    write_frozen_json(args.campaign_root / "protocol.json", protocol)
    return protocol


def _finite_metric_rows(rows: list[dict]) -> bool:
    return all(
        all(math.isfinite(float(row[name])) for name in METRICS)
        for row in rows
    )


def evaluation_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    expected = {(group, mode) for group in PROBE_IDS for mode in MODES}
    index: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row.get("group_id"), row.get("condition"))
        if key in index:
            raise ValueError(f"duplicate evaluation key: {key}")
        index[key] = row
    if set(index) != expected:
        missing = sorted(expected - set(index))
        extra = sorted(set(index) - expected)
        raise ValueError(f"evaluation key mismatch; missing={missing}, extra={extra}")
    if not _finite_metric_rows(rows):
        raise ValueError("evaluation metrics must be finite")
    return index


def _degradations(index: dict, condition: str, metric: str) -> list[float]:
    return [
        float(index[(group, "correct")][metric])
        - float(index[(group, condition)][metric])
        for group in PROBE_IDS
    ]


def _mean_condition(index: dict, condition: str, metric: str) -> float:
    return mean(float(index[(group, condition)][metric]) for group in PROBE_IDS)


def candidate_gate(
    evaluation_rows: list[dict],
    train_rows: list[dict],
    *,
    expected_steps: int,
    reference: dict[str, float] = PHASE_C_CORRECT,
    tolerance: float = NUMERICAL_TOLERANCE,
) -> dict:
    index = evaluation_index(evaluation_rows)
    if len(train_rows) != expected_steps or [row.get("step") for row in train_rows] != list(
        range(1, expected_steps + 1)
    ):
        raise ValueError("training rows are not complete and contiguous")
    last = train_rows[-100:]
    required_train = ("correct_flow_loss", "wrong_flow_loss", "camera_rank_active_fraction")
    if any(
        key not in row or not math.isfinite(float(row[key]))
        for row in last for key in required_train
    ):
        raise ValueError("last-100 training telemetry is incomplete or nonfinite")

    deltas = {
        condition: {
            metric: _degradations(index, condition, metric) for metric in METRICS
        }
        for condition in MODES if condition != "correct"
    }
    correct = {metric: _mean_condition(index, "correct", metric) for metric in METRICS}
    directional = lambda values: sum(value > 0 for value in values) >= 3
    checks = {
        "correct_psnr_nonregression": correct["psnr"] - reference["psnr"] >= -0.10,
        "correct_ssim_nonregression": correct["ssim"] - reference["ssim"] >= -0.001,
        "correct_repeat_equivalent": all(
            abs(value) <= tolerance
            for metric in METRICS for value in deltas["correct_repeat"][metric]
        ),
        "joint_permute_equivalent": all(
            abs(value) <= tolerance
            for metric in METRICS for value in deltas["joint_permute"][metric]
        ),
        "remove_psnr": mean(deltas["remove"]["psnr"]) >= 1.0,
        "remove_direction": directional(deltas["remove"]["psnr"]),
        "target_drop_psnr": mean(deltas["target_drop"]["psnr"]) >= 3.0,
        "target_drop_direction": directional(deltas["target_drop"]["psnr"]),
    }
    for condition in ("shuffle_fusion", "target_drop_shuffle_fusion"):
        checks.update({
            f"{condition}_psnr": mean(deltas[condition]["psnr"]) >= 0.05,
            f"{condition}_ssim": mean(deltas[condition]["ssim"]) >= 0.0005,
            f"{condition}_psnr_direction": directional(deltas[condition]["psnr"]),
            f"{condition}_ssim_direction": directional(deltas[condition]["ssim"]),
        })
    dose_counts = {}
    for metric in METRICS:
        monotonic = 0
        for group in PROBE_IDS:
            correct_value = float(index[(group, "correct")][metric])
            half = float(index[(group, "fusion_camera_dose_half")][metric])
            full = float(index[(group, "fusion_camera_dose_full")][metric])
            monotonic += int(correct_value + tolerance >= half >= full - tolerance)
        dose_counts[metric] = monotonic
        checks[f"camera_dose_{metric}_monotonic"] = monotonic >= 3
    correct_flow = mean(float(row["correct_flow_loss"]) for row in last)
    wrong_gap = mean(
        float(row["wrong_flow_loss"]) - float(row["correct_flow_loss"])
        for row in last
    )
    rank_active = mean(float(row["camera_rank_active_fraction"]) for row in last)
    checks["last100_wrong_correct_margin"] = wrong_gap >= 0.05 * correct_flow
    checks["last100_rank_hinge_activity"] = rank_active <= 0.5
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "correct": correct,
        "deltas": {
            condition: {
                metric: {"mean": mean(values), "values": values}
                for metric, values in metrics.items()
            }
            for condition, metrics in deltas.items()
        },
        "last100": {
            "correct_flow": correct_flow,
            "wrong_minus_correct": wrong_gap,
            "wrong_correct_ratio": wrong_gap / correct_flow if correct_flow else None,
            "rank_active_fraction": rank_active,
        },
        "camera_dose_monotonic_probes": dose_counts,
    }


def _gpu_record(gpu: int) -> dict:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    parsed = []
    for row in rows:
        index, uuid, memory, utilization = [item.strip() for item in row.split(",")]
        parsed.append({
            "index": int(index), "uuid": uuid,
            "memory_used_mib": int(memory), "utilization_percent": int(utilization),
        })
    selected = next((item for item in parsed if item["index"] == gpu), None)
    if selected is None:
        raise RuntimeError(f"GPU index {gpu} is unavailable")
    processes = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    selected["compute_processes"] = [
        row.strip() for row in processes
        if row.strip() and row.split(",", 1)[0].strip() == selected["uuid"]
    ]
    if (
        selected["memory_used_mib"] > 16
        or selected["utilization_percent"] != 0
        or selected["compute_processes"]
    ):
        raise RuntimeError(f"selected GPU is not exclusively idle: {selected}")
    return selected


def _runtime_args(args) -> list[str]:
    values = [
        "--dataset-root", str(args.dataset_root),
        "--model-dir", str(args.model_dir),
        "--lq-source", str(args.lq_source),
        "--lq-checkpoint", str(args.lq_checkpoint),
        "--bridge-checkpoint", str(args.bridge_checkpoint),
        "--device", "cuda",
    ]
    if args.rre_checkpoint is not None:
        values.extend(("--rre-checkpoint", str(args.rre_checkpoint)))
    return values


def _run_logged(command: list[str], *, args, name: str) -> None:
    gpu = _gpu_record(args.gpu)
    control = args.campaign_root / "control"
    if (control / f"{name}.json").exists():
        previous = read_json(control / f"{name}.json")
        if previous.get("exit_code") == 0:
            raise RuntimeError(f"completed command record already exists: {name}")
        attempt = 2
        while (control / f"{name}_attempt{attempt}.json").exists():
            attempt += 1
        name = f"{name}_attempt{attempt}"
    log = args.campaign_root / "logs" / f"{name}.log"
    record = control / f"{name}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONPATH": str(args.repo_root / "src"),
        "PYTHONUNBUFFERED": "1",
    })
    with log.open("x", encoding="utf-8") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        completed = subprocess.run(
            command, cwd=args.repo_root, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    payload = {"command": command, "gpu": gpu, "exit_code": completed.returncode}
    payload["log"] = str(log.resolve())
    payload["log_sha256"] = sha256_file(log)
    write_frozen_json(record, payload)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def _complete_training(output: Path, steps: int) -> bool:
    final = output / f"stage3_step_{steps}.pt"
    rows = output / "train_steps.jsonl"
    if not final.is_file() or not rows.is_file():
        return False
    values = read_jsonl(rows)
    return len(values) == steps and [row.get("step") for row in values] == list(range(1, steps + 1))


def train_cell(args, cell: Cell) -> Path:
    prepare(args)
    config_path = _config_path(args, cell)
    config = load_stage3_config(config_path)
    output = _train_dir(args, cell.name)
    final = output / f"stage3_step_{config.steps}.pt"
    if _complete_training(output, config.steps):
        return final
    resume = None
    if output.exists() and any(output.iterdir()):
        candidates = sorted(
            output.glob("stage3_step_*.pt"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        if not candidates:
            raise RuntimeError(f"incomplete cell is not resumable: {output}")
        resume = candidates[-1]
    command = [
        args.python, str(args.repo_root / "scripts" / "stage3_experiment.py"),
        "train", "--config", str(config_path), *_runtime_args(args),
        "--seed", str(cell.seed), "--output-dir", str(output),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    else:
        command.extend(("--init-checkpoint", str(args.phase_c_checkpoint)))
        if cell.reset_fusion:
            command.append("--init-reset-fusion")
    resume_label = (
        f"resume_{int(resume.stem.rsplit('_', 1)[1])}" if resume is not None
        else "initial"
    )
    write_frozen_json(
        args.campaign_root / "cells" / f"{cell.name}.json",
        {
            "name": cell.name,
            "seed": cell.seed,
            "config": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "steps": config.steps,
            "parent_checkpoint": str(args.phase_c_checkpoint.resolve()),
            "parent_checkpoint_sha256": PHASE_C_CHECKPOINT_SHA256,
            "reset_fusion": cell.reset_fusion,
        },
    )
    _run_logged(
        command, args=args,
        name=f"{cell.name}_train_{resume_label}",
    )
    if not _complete_training(output, config.steps):
        raise RuntimeError(f"training cell did not complete exactly {config.steps} steps")
    return final


def _sampling_signature(config: Stage3Config) -> dict:
    return {
        "train_scenes": list(config.train_scenes),
        "image_size": config.image_size,
        "scale": config.scale,
        "views": config.views,
    }


def seen_manifest_for_cell(args, cell: Cell) -> Path:
    config = load_stage3_config(_config_path(args, cell))
    source = read_json(args.seen_manifest)
    if source.get("sampling_signature") == _sampling_signature(config):
        return args.seen_manifest
    destination = args.campaign_root / "manifest" / f"{cell.name}.json"
    if destination.is_file():
        payload = read_json(destination)
        if payload.get("sampling_signature") != _sampling_signature(config):
            raise RuntimeError(f"seen manifest drift: {destination}")
        return destination
    command = [
        args.python, str(args.repo_root / "scripts" / "stage3_experiment.py"),
        "prepare-seen", "--config", str(_config_path(args, cell)),
        "--dataset-root", str(args.dataset_root), "--manifest", str(destination),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.repo_root / "src")
    completed = subprocess.run(command, cwd=args.repo_root, env=environment, check=False)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return destination


def _complete_evaluation(output: Path) -> bool:
    summary = output / "evaluation_summary.json"
    rows = output / "evaluation_rows.jsonl"
    if not summary.is_file() or not rows.is_file():
        return False
    payload = read_json(summary)
    values = read_jsonl(rows)
    try:
        evaluation_index(values)
    except ValueError:
        return False
    return (
        payload.get("rows") == len(PROBE_IDS) * len(MODES)
        and payload.get("conditions") == list(MODES)
        and payload.get("inference_seeds") == [INFERENCE_SEED]
        and payload.get("groups") == len(PROBE_IDS)
    )


def evaluate_cell(args, cell: Cell) -> Path:
    checkpoint = train_cell(args, cell)
    output = _eval_dir(args, cell.name)
    if _complete_evaluation(output):
        return output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to reuse incomplete evaluation: {output}")
    manifest = seen_manifest_for_cell(args, cell)
    command = [
        args.python, str(args.repo_root / "scripts" / "stage3_experiment.py"),
        "seen-eval", "--config", str(_config_path(args, cell)),
        *_runtime_args(args), "--checkpoint", str(checkpoint),
        "--seen-manifest", str(manifest), "--subset", "probe",
        "--group-ids", *PROBE_IDS, "--inference-seeds", str(INFERENCE_SEED),
        "--modes", *MODES, "--save-diagnostics", "--save-images",
        "--output-dir", str(output),
    ]
    _run_logged(command, args=args, name=f"{cell.name}_evaluate")
    if not _complete_evaluation(output):
        raise RuntimeError("evaluation did not produce the exact frozen key set")
    return output


def _integrity(args, cell: Cell) -> dict:
    config = load_stage3_config(_config_path(args, cell))
    train_dir = _train_dir(args, cell.name)
    eval_dir = _eval_dir(args, cell.name)
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        prepare(args)
        rows = read_jsonl(train_dir / "train_steps.jsonl")
        eval_rows = read_jsonl(eval_dir / "evaluation_rows.jsonl")
        summary = read_json(eval_dir / "evaluation_summary.json")
        manifest = read_json(train_dir / "run_manifest.json")
        checkpoint = train_dir / f"stage3_step_{config.steps}.pt"
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        seen_manifest = seen_manifest_for_cell(args, cell)
        control_records = [
            read_json(path) for path in (args.campaign_root / "control").glob(f"{cell.name}_*.json")
        ]
        checks.update({
            "training_complete": len(rows) == config.steps and [r.get("step") for r in rows] == list(range(1, config.steps + 1)),
            "evaluation_keys": set(evaluation_index(eval_rows)) == {(g, m) for g in PROBE_IDS for m in MODES},
            "evaluation_rows": len(eval_rows) == summary.get("rows") == len(PROBE_IDS) * len(MODES),
            "baseline_rows": len(read_jsonl(eval_dir / "baseline_rows.jsonl")) == 8,
            "diagnostic_rows": len(read_jsonl(eval_dir / "diagnostics.jsonl")) == len(PROBE_IDS) * (len(MODES) - 1),
            "images": len(list((eval_dir / "images").rglob("*.png"))) == 64,
            "manifest_config": manifest.get("config") == read_json(_config_path(args, cell)),
            "manifest_seed": manifest.get("provenance", {}).get("training_seed") == cell.seed,
            "manifest_parent": manifest.get("provenance", {}).get("parent_checkpoint_sha256") == PHASE_C_CHECKPOINT_SHA256,
            "checkpoint": checkpoint.is_file(),
            "checkpoint_payload": (
                checkpoint_payload.get("step") == config.steps
                and checkpoint_payload.get("config") == read_json(_config_path(args, cell))
            ),
            "paired_protocol": summary.get("inference_seeds") == [INFERENCE_SEED] and summary.get("conditions") == list(MODES),
            "evaluation_checkpoint": summary.get("checkpoint") == str(checkpoint.resolve()),
            "evaluation_seed_and_keys": all(
                row.get("inference_seed") == INFERENCE_SEED
                and row.get("checkpoint") == str(checkpoint.resolve())
                and row.get("train_seed") == cell.seed
                for row in eval_rows
            ),
            "seen_manifest_hash": summary.get("seen_manifest_sha256") == sha256_file(seen_manifest),
            "command_records": bool(control_records) and all(
                Path(record.get("log", "")).is_file()
                and record.get("log_sha256") == sha256_file(Path(record["log"]))
                for record in control_records
            ) and any(
                record.get("exit_code") == 0
                and "seen-eval" in record.get("command", [])
                for record in control_records
            ) and any(
                record.get("exit_code") == 0
                and "train" in record.get("command", [])
                for record in control_records
            ),
        })
        if config.pairing_supervision:
            pairing = manifest.get("pairing") or {}
            preflight = Path(pairing.get("preflight", ""))
            checks["pairing_preflight"] = (
                pairing.get("enabled") is True
                and preflight.is_file()
                and pairing.get("preflight_sha256") == sha256_file(preflight)
                and isinstance(pairing.get("weight"), (int, float))
            )
        details["checkpoint_sha256"] = sha256_file(checkpoint)
        details["training_rows_sha256"] = sha256_file(train_dir / "train_steps.jsonl")
        details["evaluation_rows_sha256"] = sha256_file(eval_dir / "evaluation_rows.jsonl")
    except Exception as exc:
        checks["artifact_read_complete"] = False
        details["error"] = f"{type(exc).__name__}: {exc}"
    return {"pass": bool(checks) and all(checks.values()), "checks": checks, "details": details}


def _write_diagnostic_plots(args, cell: Cell, gate: dict) -> None:
    analysis = args.campaign_root / "analysis" / cell.name
    curves = analysis / "training_curves.png"
    dose = analysis / "camera_dose.png"
    if not curves.is_file() or not dose.is_file():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = read_jsonl(_train_dir(args, cell.name) / "train_steps.jsonl")
        steps = [row["step"] for row in rows]
        if not curves.is_file():
            figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            for name in ("correct_flow_loss", "wrong_flow_loss", "camera_rank_loss", "pairing_loss"):
                if all(name in row for row in rows):
                    axes[0].plot(steps, [row[name] for row in rows], label=name)
            for name in (
                "fusion_gradient_norm", "camera_rank_gradient_norm",
                "pairing_fusion_gradient_norm", "final_fusion_gradient_norm",
            ):
                if all(name in row for row in rows):
                    axes[1].plot(steps, [row[name] for row in rows], label=name)
            axes[0].set_ylabel("loss")
            axes[1].set_ylabel("gradient norm")
            axes[1].set_xlabel("optimizer step")
            axes[0].legend(fontsize=7)
            axes[1].legend(fontsize=7)
            figure.tight_layout()
            figure.savefig(curves, dpi=160)
            plt.close(figure)
        if not dose.is_file():
            figure, axis = plt.subplots(figsize=(7, 4))
            x = [0.0, 0.5, 1.0]
            for metric in METRICS:
                values = [
                    0.0,
                    gate["deltas"]["fusion_camera_dose_half"][metric]["mean"],
                    gate["deltas"]["fusion_camera_dose_full"][metric]["mean"],
                ]
                axis.plot(x, values, marker="o", label=f"{metric} degradation")
            axis.set_xlabel("wrong-camera dose")
            axis.set_ylabel("correct minus dose metric")
            axis.legend()
            figure.tight_layout()
            figure.savefig(dose, dpi=160)
            plt.close(figure)


def _write_contact_sheet(args, cell: Cell) -> None:
    from PIL import Image, ImageDraw

    destination = args.campaign_root / "analysis" / cell.name / "probe_contact_sheet.png"
    if destination.is_file():
        return
    modes = (
        "correct", "shuffle_fusion", "target_drop_shuffle_fusion",
        "fusion_camera_dose_half", "fusion_camera_dose_full",
    )
    images = []
    for group in PROBE_IDS:
        scene, anchor = group.split(":")
        row = []
        for mode in modes:
            path = (
                _eval_dir(args, cell.name) / "images" / scene / f"view_{int(anchor):03d}"
                / f"seed_{INFERENCE_SEED}" / f"{mode}.png"
            )
            row.append(Image.open(path).convert("RGB"))
        images.append(row)
    width, height = images[0][0].size
    label_height = 24
    canvas = Image.new("RGB", (width * len(modes), (height + label_height) * len(images)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(images):
        for column, image in enumerate(row):
            x, y = column * width, row_index * (height + label_height)
            canvas.paste(image, (x, y + label_height))
            draw.text((x + 4, y + 4), f"{PROBE_IDS[row_index]} {modes[column]}", fill="black")
    canvas.save(destination)


def analyze_cell(args, cell: Cell) -> dict:
    output = evaluate_cell(args, cell)
    config = load_stage3_config(_config_path(args, cell))
    gate = candidate_gate(
        read_jsonl(output / "evaluation_rows.jsonl"),
        read_jsonl(_train_dir(args, cell.name) / "train_steps.jsonl"),
        expected_steps=config.steps,
    )
    integrity = _integrity(args, cell)
    passed = gate["pass"] and integrity["pass"]
    result = {
        "cell": cell.name,
        "arm": config.arm,
        "seed": cell.seed,
        "steps": config.steps,
        "pass": passed,
        "gate": gate,
        "integrity": integrity,
        "next": next_action(cell.name, passed),
        "CAMERA_FUSION_PASS": False,
        "STAGE4_READY": False,
    }
    analysis = args.campaign_root / "analysis" / cell.name
    write_frozen_json(analysis / "summary.json", result)
    _write_diagnostic_plots(args, cell, gate)
    _write_contact_sheet(args, cell)
    lines = [
        f"# Stage 3.3 — {cell.name}", "",
        f"Verdict: **{'PASS' if passed else 'HOLD'}**", "",
        "Stage 4 and 4DSR were not run.", "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in gate["checks"].items())
    write_frozen_text(analysis / "report.md", "\n".join(lines) + "\n")
    return result


def next_action(cell_name: str, passed: bool) -> str:
    if cell_name == "h0":
        return "RUN_A3_1000" if passed else "RUN_A4_1000"
    if cell_name == "a3_pilot":
        return "RUN_REPLICATION" if passed else "FINAL_HOLD"
    if cell_name == "a4_pilot":
        return "RUN_REPLICATION" if passed else "RUN_A5_1000"
    if cell_name == "a5_pilot":
        return "RUN_REPLICATION" if passed else "FINAL_HOLD"
    return "CONTINUE_REPLICATION" if passed else "FINAL_HOLD"


def derive_replication_config(base: Stage3Config, kind: str) -> Stage3Config:
    common = {"steps": 2000, "checkpoint_every": 500}
    if kind == "seed43":
        return replace(base, **common)
    if kind == "v8":
        return replace(base, views=8, **common)
    if kind == "five_scene":
        return replace(
            base,
            train_scenes=FIVE_TRAIN_SCENES,
            validation_scenes=("ficus",),
            test_scenes=("materials", "ship"),
            **common,
        )
    raise ValueError(f"unknown replication kind: {kind}")


def _replication_cell(args, winner: Cell, kind: str) -> Cell:
    base = load_stage3_config(_config_path(args, winner))
    config = derive_replication_config(base, kind)
    name = f"{winner.name}_{kind}_2000"
    config_name = f"{name}.json"
    write_frozen_json(args.campaign_root / "config" / config_name, config.to_dict())
    return Cell(name, config_name, 43 if kind == "seed43" else 42, winner.reset_fusion)


def smoke(args) -> dict:
    target = args.campaign_root / "preflight" / "real_wan_a4_smoke.json"
    if target.is_file():
        payload = read_json(target)
        if not payload.get("pass"):
            raise RuntimeError("recorded real-Wan smoke failed")
        return payload
    prepare(args)
    _gpu_record(args.gpu)
    import stage3_experiment as stage3
    config = load_stage3_config(_config_path(args, FIXED_CELLS["a4_pilot"]))
    runtime = stage3.load_runtime(
        config,
        model_dir=args.model_dir,
        lq_source=args.lq_source,
        lq_checkpoint=args.lq_checkpoint,
        bridge_checkpoint=args.bridge_checkpoint,
        stage3_checkpoint=args.phase_c_checkpoint,
        model_only_initialization=True,
        reset_initialization_fusion=True,
        device="cuda",
    )
    runtime.module.train()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    group = next(
        item for item in read_json(args.seen_manifest)["groups"]
        if item["id"] == PROBE_IDS[0]
    )
    _, hr, lr, camera = stage3._load_indices(
        args.dataset_root, group["scene"], "train", group["indices"], config, runtime.device
    )
    with torch.no_grad():
        clean = runtime.vae.encode_multiview(hr)
    prepared = runtime.module.prepare_multiview(
        lr, camera, tuple(clean.shape[2:]), (config.image_size, config.image_size)
    )
    noise = torch.randn_like(clean)
    noisy, timestep, target_flow = stage3.flow_matching_pair(
        clean, noise, torch.tensor([0.5], device=runtime.device)
    )
    prediction = runtime.module.predict(
        runtime.dit, noisy, timestep, None, prepared, camera, tuple(clean.shape[2:])
    )
    torch.nn.functional.mse_loss(prediction.float(), target_flow.float()).backward()

    def grad_status(module) -> dict:
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        finite = [g for g in gradients if g is not None and torch.isfinite(g).all()]
        return {
            "parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "gradient_tensors": len(finite),
            "gradient_norm": float(torch.stack([g.float().square().sum() for g in finite]).sum().sqrt()) if finite else 0.0,
        }

    gradients = {
        "bridge": grad_status(runtime.module.conditioner),
        "geometry": grad_status(runtime.module.geometry),
        "fusion": grad_status(runtime.module.fusion),
    }
    with torch.no_grad():
        sampled = stage3.sample_latents(
            runtime, lr, camera, tuple(clean.shape), 2, config.image_size,
            seed=330299, sampling_shift=config.sampling_shift, dtype=torch.bfloat16,
        )
        decoded = runtime.vae.decode_multiview(sampled)
    payload = {
        "pass": (
            all(item["gradient_tensors"] > 0 and item["gradient_norm"] > 0 for item in gradients.values())
            and not any(p.grad is not None for p in runtime.dit.model.parameters())
            and not any(p.grad is not None for p in runtime.vae.model.model.parameters())
            and tuple(decoded.shape) == tuple(hr.shape)
            and torch.isfinite(decoded).all().item()
        ),
        "config": config.to_dict(),
        "gradients": gradients,
        "wan_gradients": sum(p.grad is not None for p in runtime.dit.model.parameters()),
        "vae_gradients": sum(p.grad is not None for p in runtime.vae.model.model.parameters()),
        "sample_steps": 2,
        "decoded_shape": list(decoded.shape),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(runtime.device) / 2**20,
    }
    write_frozen_json(target, payload)
    del runtime, hr, lr, camera, clean, prepared, prediction, sampled, decoded
    gc.collect()
    torch.cuda.empty_cache()
    if not payload["pass"]:
        raise RuntimeError(f"real-Wan A4 smoke failed: {payload}")
    return payload


def a5_probe_coverage(args) -> dict:
    """Fail closed on every directed pair of each frozen probe before A5 trains."""
    target = args.campaign_root / "preflight" / "a5_four_probe_coverage.json"
    if target.is_file():
        payload = read_json(target)
        if not payload.get("pass"):
            raise ValueError("A5 four-probe coverage preflight failed")
        return payload
    prepare(args)
    _gpu_record(args.gpu)
    import stage3_experiment as stage3

    config = load_stage3_config(_config_path(args, FIXED_CELLS["a5_pilot"]))
    runtime = stage3.load_runtime(
        config,
        model_dir=args.model_dir,
        lq_source=args.lq_source,
        lq_checkpoint=args.lq_checkpoint,
        bridge_checkpoint=args.bridge_checkpoint,
        stage3_checkpoint=args.phase_c_checkpoint,
        model_only_initialization=True,
        reset_initialization_fusion=True,
        device="cuda",
    )
    runtime.module.eval()
    groups = {group["id"]: group for group in read_json(args.seen_manifest)["groups"]}
    results = []
    try:
        with torch.no_grad():
            for group_id in PROBE_IDS:
                group = groups[group_id]
                _, hr, lr, camera = stage3._load_indices(
                    args.dataset_root, group["scene"], "train", group["indices"],
                    config, runtime.device,
                )
                clean = runtime.vae.encode_multiview(hr)
                wrong = stage3.derange_auxiliary_fusion_camera(
                    camera, torch.Generator().manual_seed(330300 + group["anchor"])
                )
                try:
                    _, pairing = runtime.module.prepare_multiview(
                        lr, camera, tuple(clean.shape[2:]),
                        (config.image_size, config.image_size),
                        pairing_camera=wrong,
                        pairing_minimum_coverage=config.pairing_minimum_coverage,
                    )
                except ValueError as exc:
                    if "pairing coverage failed" not in str(exc):
                        raise
                    raise ValueError(
                        f"A5 four-probe coverage preflight failed: {group_id}: {exc}"
                    ) from exc
                rows = [
                    {
                        "target_view": int(row["target_view"]),
                        "source_view": int(row["source_view"]),
                        "coverage": float(row["coverage"]),
                        "pair_count": int(row["target_patches"].numel()),
                    }
                    for row in pairing["pairs"]
                ]
                expected = {
                    (target_view, source_view)
                    for target_view in range(config.views)
                    for source_view in range(config.views)
                    if target_view != source_view
                }
                identities = [(row["target_view"], row["source_view"]) for row in rows]
                if (
                    len(rows) != config.views * (config.views - 1)
                    or len(set(identities)) != len(rows)
                    or set(identities) != expected
                    or any(row["coverage"] < config.pairing_minimum_coverage for row in rows)
                ):
                    raise ValueError(f"A5 four-probe coverage preflight failed: {group_id}")
                results.append({"group_id": group_id, "pairs": rows})
                del hr, lr, camera, clean, pairing
    finally:
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
    payload = {
        "pass": len(results) == len(PROBE_IDS),
        "probe_ids": list(PROBE_IDS),
        "parent_checkpoint_sha256": PHASE_C_CHECKPOINT_SHA256,
        "minimum_coverage": config.pairing_minimum_coverage,
        "results": results,
    }
    write_frozen_json(target, payload)
    return payload


def cpu_tests(args) -> dict:
    target = args.campaign_root / "preflight" / "cpu_tests.json"
    if target.is_file():
        payload = read_json(target)
        if payload.get("exit_code") != 0:
            raise RuntimeError("recorded CPU test suite failed")
        return payload
    log = args.campaign_root / "logs" / "cpu_tests.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [args.python, "-m", "pytest", "-q"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.repo_root / "src")
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=args.repo_root, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    payload = {
        "command": command,
        "exit_code": completed.returncode,
        "log": str(log.resolve()),
        "log_sha256": sha256_file(log),
    }
    write_frozen_json(target, payload)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return payload


def recompute_readiness(args, winner: Cell, completed: list[str]) -> dict:
    # Stage 3.3's four-probe 2000-step cells are not the old five-scene,
    # two-seed, 4000-step A0-A3 population.  Never relabel their new gates as
    # the frozen definitions in stage3_seen_sr_analysis.py.
    names = [winner.name] + [name for name in completed if name.startswith(f"{winner.name}_")]
    verdicts = {
        "CROSS_VIEW_PASS": False,
        "GEOMETRY_PASS": False,
        "SEEN_SR_EFFECTIVE": False,
    }
    return {
        "scope": "frozen Stage 3 seen-view readiness definitions",
        "cells": names,
        "recomputed": False,
        "reason": (
            "The frozen stage3_seen_sr_analysis.py definitions require matched "
            "A0-A3 arms, two seeds and five-scene 4000-step full/intervention "
            "populations. The bounded Stage 3.3 ladder cannot supply those "
            "inputs; the four-probe gate cannot replace them. HOLD pending a "
            "separately approved matched frozen-protocol recomputation."
        ),
        "verdicts": verdicts,
    }


def _write_machine_verdict(args, *, status: str, winner: str | None = None,
                           completed: list[str] | None = None,
                           readiness: dict | None = None) -> dict:
    passed = status == "PASS"
    payload = {
        "verdict": status,
        "winner": winner,
        "completed_cells": completed or [],
        "CAMERA_FUSION_PASS": passed,
        "STAGE4_READY": passed,
        "stage4_ran": False,
        "four_dsr_ran": False,
        "readiness": readiness,
    }
    write_frozen_json(args.campaign_root / "analysis" / "machine_verdict.json", payload)
    return payload


def run(args) -> dict:
    prepare(args)
    cpu_tests(args)
    smoke(args)
    completed = []
    h0 = analyze_cell(args, FIXED_CELLS["h0"])
    completed.append("h0")
    if h0["pass"]:
        winner = FIXED_CELLS["a3_pilot"]
        pilot = analyze_cell(args, winner)
        completed.append(winner.name)
        if not pilot["pass"]:
            return _write_machine_verdict(args, status="HOLD", completed=completed)
    else:
        winner = FIXED_CELLS["a4_pilot"]
        pilot = analyze_cell(args, winner)
        completed.append(winner.name)
        if not pilot["pass"]:
            winner = FIXED_CELLS["a5_pilot"]
            try:
                a5_probe_coverage(args)
                pilot = analyze_cell(args, winner)
            except ValueError as exc:
                if "A5 four-probe coverage preflight failed" not in str(exc):
                    raise
                write_frozen_json(
                    args.campaign_root / "analysis" / "a5_preflight_hold.json",
                    {"verdict": "HOLD", "reason": str(exc)},
                )
                return _write_machine_verdict(args, status="HOLD", completed=completed)
            except subprocess.CalledProcessError as exc:
                log = args.campaign_root / "logs" / "a5_pilot_train_initial.log"
                text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
                if not any(marker in text for marker in (
                    "A5 calibration coverage", "A5 pairing coverage",
                    "pairing coverage failed for batch",
                )):
                    raise
                write_frozen_json(
                    args.campaign_root / "analysis" / "a5_preflight_hold.json",
                    {"verdict": "HOLD", "reason": "A5 pairing coverage preflight failed",
                     "log_sha256": sha256_file(log), "exit_code": exc.returncode},
                )
                return _write_machine_verdict(args, status="HOLD", completed=completed)
            completed.append(winner.name)
            if not pilot["pass"]:
                return _write_machine_verdict(args, status="HOLD", completed=completed)
    for kind in ("seed43", "v8", "five_scene"):
        cell = _replication_cell(args, winner, kind)
        result = analyze_cell(args, cell)
        completed.append(cell.name)
        if not result["pass"]:
            return _write_machine_verdict(
                args, status="HOLD", winner=winner.name, completed=completed
            )
    readiness = recompute_readiness(args, winner, completed)
    write_frozen_json(args.campaign_root / "analysis" / "readiness.json", readiness)
    ready = all(readiness["verdicts"].values())
    return _write_machine_verdict(
        args, status="PASS" if ready else "HOLD", winner=winner.name,
        completed=completed, readiness=readiness,
    )


def status(args) -> dict:
    verdict = args.campaign_root / "analysis" / "machine_verdict.json"
    if verdict.is_file():
        return read_json(verdict)
    cells = {}
    for name, cell in FIXED_CELLS.items():
        config_path = _config_path(args, cell)
        if config_path.is_file():
            config = load_stage3_config(config_path)
            cells[name] = {
                "training_complete": _complete_training(_train_dir(args, name), config.steps),
                "evaluation_complete": _complete_evaluation(_eval_dir(args, name)),
                "analysis_complete": (args.campaign_root / "analysis" / name / "summary.json").is_file(),
            }
    return {"verdict": "IN_PROGRESS", "cells": cells}


def _require_runtime(args) -> None:
    for name in (
        "dataset_root", "model_dir", "lq_source", "lq_checkpoint",
        "bridge_checkpoint", "phase_c_checkpoint", "phase_c_protocol", "seen_manifest",
    ):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} must exist: {path}")


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    original = Path("/data/linzizhuo/RL3DSR_WAN_REMO")
    phase_c = original / "artifacts" / "stage3_2_camera_causality_20260913" / "phase_c"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "cpu-tests", "smoke", "a5-preflight", "train", "evaluate", "analyze", "run", "status",
    ))
    parser.add_argument("--cell", choices=tuple(FIXED_CELLS))
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--campaign-root", type=Path,
        default=root / "artifacts" / "stage3_3_ucpe_rre_fusion_20260916",
    )
    parser.add_argument("--phase-c-checkpoint", type=Path, default=phase_c / "train" / "rank" / "seed42" / "stage3_step_1000.pt")
    parser.add_argument("--phase-c-protocol", type=Path, default=phase_c / "protocol.json")
    parser.add_argument("--seen-manifest", type=Path, default=original / "artifacts" / "stage3_2_camera_causality_20260913" / "manifest" / "v4_seen_groups.json")
    parser.add_argument("--dataset-root", type=Path, default=original / "datasets" / "nerf_synthetic")
    parser.add_argument("--model-dir", type=Path, default=original / "models" / "Wan2.1-T2V-1.3B")
    parser.add_argument("--lq-source", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py"))
    parser.add_argument("--lq-checkpoint", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"))
    parser.add_argument("--bridge-checkpoint", type=Path, default=original / "artifacts" / "stage1" / "c_campaign_20260902" / "c_main" / "best_dev.pt")
    parser.add_argument("--rre-checkpoint", type=Path)
    parser.add_argument("--official-ucpe-weight", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for name in (
        "repo_root", "campaign_root", "phase_c_checkpoint", "phase_c_protocol",
        "seen_manifest", "dataset_root", "model_dir", "lq_source", "lq_checkpoint",
        "bridge_checkpoint",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.command != "status":
        _require_runtime(args)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "cpu-tests":
        result = cpu_tests(args)
    elif args.command == "smoke":
        result = smoke(args)
    elif args.command == "a5-preflight":
        result = a5_probe_coverage(args)
    elif args.command in {"train", "evaluate", "analyze"}:
        if args.cell is None:
            raise ValueError(f"{args.command} requires --cell")
        cell = FIXED_CELLS[args.cell]
        result = {
            "train": train_cell,
            "evaluate": evaluate_cell,
            "analyze": analyze_cell,
        }[args.command](args, cell)
    elif args.command == "run":
        result = run(args)
    else:
        result = status(args)
    if isinstance(result, Path):
        result = {"path": str(result)}
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
