#!/usr/bin/env python3
"""Run and analyze the bounded Stage 3.1 V4-vs-V8 chair pilot.

The driver is intentionally conservative: it creates independent manifests,
refuses to overwrite non-empty outputs, checks exclusive GPU use before every
real-Wan launch, and only starts the seed-43 replication after the pilot gate
passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from statistics import mean
from typing import Iterable


SEED = 42
REPLICATION_SEED = 43
INFERENCE_SEED = 3302
PROBE_IDS = ("chair:000", "chair:033", "chair:066", "chair:099")
MODES = (
    "correct",
    "correct_repeat",
    "remove",
    "target_drop",
    "shuffle_fusion",
    "shuffle_geometry",
    "shuffle_pair",
)
METRICS = ("psnr", "ssim", "lpips", "mae")
GOOD_DIRECTION = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0, "mae": -1.0}
STEPS = 2000

CELL_CONFIGS = {
    "v4": ("A3_views4_local_band_chair_2000.json", 4, "v4_seen_groups.json"),
    "v8": ("A3_views8_local_band_chair_2000.json", 8, "v8_seen_groups.json"),
}

EXPECTED_PARAMETER_COUNTS = {
    "trainable_parameters": 47839104,
    "frozen_wan_parameters": 1418996800,
    "frozen_vae_parameters": 126892531,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_frozen_json(path: Path, payload: dict) -> None:
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError(f"frozen artifact drift: {path}")
        return
    write_json(path, payload)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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


def command_text(command: list[str]) -> str:
    return shlex.join([str(item) for item in command]) + "\n"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path(root: Path, cell: str) -> Path:
    return root / "configs" / "stage3_1" / CELL_CONFIGS[cell][0]


def manifest_path(campaign: Path, cell: str) -> Path:
    return campaign / "manifest" / CELL_CONFIGS[cell][2]


def train_path(campaign: Path, cell: str, seed: int) -> Path:
    return campaign / "train" / cell / f"seed{seed}"


def eval_path(campaign: Path, cell: str, seed: int) -> Path:
    return train_path(campaign, cell, seed) / "eval"


def _run(command: list[str], *, cwd: Path, log_path: Path | None = None, env: dict[str, str] | None = None) -> None:
    if log_path is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(command_text(command), encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as stream:
        subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def _write_command(campaign: Path, name: str, command: list[str]) -> None:
    path = campaign / "control" / f"{name}.command.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(command_text(command), encoding="utf-8")


def _gpu_is_idle(gpu: int) -> bool:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return not query.stdout.strip()


def _env(root: Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _run_runtime(command: list[str], *, root: Path, campaign: Path, name: str, gpu: int) -> None:
    if not _gpu_is_idle(gpu):
        raise RuntimeError(f"GPU {gpu} is not idle; refusing to start {name}")
    _write_command(campaign, name, command)
    log_path = campaign / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(command_text(command))
        subprocess.run(command, cwd=root, env=_env(root, gpu), stdout=stream, stderr=subprocess.STDOUT, check=True)


def _runtime_command(args, subcommand: str, config: Path, *extra: str) -> list[str]:
    return [
        args.python,
        str(args.repo_root / "scripts" / "stage3_experiment.py"),
        subcommand,
        "--config", str(config),
        "--dataset-root", str(args.dataset_root),
        "--model-dir", str(args.model_dir),
        "--lq-source", str(args.lq_source),
        "--lq-checkpoint", str(args.lq_checkpoint),
        "--bridge-checkpoint", str(args.bridge_checkpoint),
        *extra,
    ]


def _validate_config(root: Path, cell: str) -> dict:
    payload = read_json(config_path(root, cell))
    expected_views = CELL_CONFIGS[cell][1]
    checks = {
        "arm": payload.get("arm") == "A3",
        "train_scenes": payload.get("train_scenes") == ["chair"],
        "views": payload.get("views") == expected_views,
        "steps": payload.get("steps") == STEPS,
        "epipolar_attention": payload.get("epipolar_attention") == "local_band",
        "epipolar_band": payload.get("epipolar_band") == 1.5,
        "target_lr_dropout": payload.get("target_lr_dropout") == 0.5,
        "nearest_views": payload.get("nearest_views") == 12,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid {cell} config: {checks}")
    return payload


def _manifest_groups(path: Path) -> dict[str, dict]:
    payload = read_json(path)
    if payload.get("version") != 1 or payload.get("scope") != "seen_train_sr":
        raise ValueError(f"unsupported manifest: {path}")
    groups = {item["id"]: item for item in payload.get("groups", [])}
    if len(groups) != 100:
        raise ValueError(f"expected 100 chair groups in {path}, got {len(groups)}")
    return groups


def prepare(args) -> dict:
    root = args.repo_root.resolve()
    campaign = args.campaign_root.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    configs = {cell: _validate_config(root, cell) for cell in CELL_CONFIGS}
    for cell in CELL_CONFIGS:
        target = manifest_path(campaign, cell)
        if target.exists():
            existing = read_json(target)
            if existing.get("scope") != "seen_train_sr":
                raise RuntimeError(f"refusing to reuse invalid manifest: {target}")
            continue
        command = [
            args.python,
            str(root / "scripts" / "stage3_experiment.py"),
            "prepare-seen",
            "--config", str(config_path(root, cell)),
            "--dataset-root", str(args.dataset_root),
            "--manifest", str(target),
        ]
        _run(command, cwd=root, env=_env(root, args.gpu), log_path=campaign / "logs" / f"prepare_{cell}.log")
    v4 = _manifest_groups(manifest_path(campaign, "v4"))
    v8 = _manifest_groups(manifest_path(campaign, "v8"))
    if set(v4) != set(v8):
        raise ValueError("V4/V8 manifest anchors differ")
    nesting = []
    for group_id in sorted(v4):
        left = v4[group_id]
        right = v8[group_id]
        if left["anchor"] != right["anchor"] or left["scene"] != right["scene"]:
            raise ValueError(f"anchor metadata differs for {group_id}")
        left_indices = list(left["indices"])
        right_indices = list(right["indices"])
        if len(left_indices) != 4 or len(right_indices) != 8 or left_indices[0] != right_indices[0]:
            raise ValueError(f"nested view contract failed for {group_id}")
        if not set(left_indices).issubset(set(right_indices)):
            raise ValueError(f"V4 is not a subset of V8 for {group_id}")
        nesting.append({"id": group_id, "v4": left_indices, "v8": right_indices})
    protocol = {
        "schema_version": 1,
        "scope": "stage3_1_view_count_sufficiency",
        "training_seed": SEED,
        "replication_seed": REPLICATION_SEED,
        "inference_seed": INFERENCE_SEED,
        "probe_ids": list(PROBE_IDS),
        "steps": STEPS,
        "cells": {
            cell: {
                "config": str(config_path(root, cell)),
                "config_sha256": sha256_file(config_path(root, cell)),
                "views": CELL_CONFIGS[cell][1],
                "manifest": str(manifest_path(campaign, cell)),
                "manifest_sha256": sha256_file(manifest_path(campaign, cell)),
            }
            for cell in CELL_CONFIGS
        },
        "nested_groups": nesting,
    }
    write_frozen_json(campaign / "protocol.json", protocol)
    return protocol


def _train_complete(path: Path, steps: int) -> bool:
    log = path / "train_steps.jsonl"
    checkpoint = path / f"stage3_step_{steps:04d}.pt"
    manifest = path / "run_manifest.json"
    if not all(item.is_file() for item in (log, checkpoint, manifest)):
        return False
    rows = read_jsonl(log)
    return len(rows) == steps and [int(row.get("step", -1)) for row in rows] == list(range(1, steps + 1))


def train_cell(args, cell: str, seed: int) -> None:
    root = args.repo_root.resolve()
    campaign = args.campaign_root.resolve()
    config = config_path(root, cell)
    output = train_path(campaign, cell, seed)
    if _train_complete(output, STEPS):
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete training output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = _runtime_command(
        args,
        "train",
        config,
        "--output-dir", str(output),
        "--seed", str(seed),
    )
    _run_runtime(command, root=root, campaign=campaign, name=f"{cell}_seed{seed}_train", gpu=args.gpu)
    if not _train_complete(output, STEPS):
        raise RuntimeError(f"training completeness check failed: {output}")


def _eval_complete(path: Path) -> bool:
    summary_path = path / "evaluation_summary.json"
    rows_path = path / "evaluation_rows.jsonl"
    baseline_path = path / "baseline_rows.jsonl"
    diagnostics_path = path / "diagnostics.jsonl"
    if not all(item.is_file() for item in (summary_path, rows_path, baseline_path, diagnostics_path)):
        return False
    summary = read_json(summary_path)
    rows = read_jsonl(rows_path)
    baseline = read_jsonl(baseline_path)
    diagnostics = read_jsonl(diagnostics_path)
    return (
        summary.get("scope") == "seen_train_sr"
        and summary.get("subset") == "probe"
        and summary.get("rows") == 28
        and summary.get("baseline_rows") == 8
        and summary.get("groups") == 4
        and summary.get("inference_seeds") == [INFERENCE_SEED]
        and tuple(summary.get("conditions", ())) == MODES
        and len(rows) == 28
        and len(baseline) == 8
        and len(diagnostics) == 24
    )


def evaluate_cell(args, cell: str, seed: int) -> None:
    root = args.repo_root.resolve()
    campaign = args.campaign_root.resolve()
    config = config_path(root, cell)
    checkpoint = train_path(campaign, cell, seed) / f"stage3_step_{STEPS:04d}.pt"
    output = eval_path(campaign, cell, seed)
    if _eval_complete(output):
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete evaluation output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = _runtime_command(
        args,
        "seen-eval",
        config,
        "--checkpoint", str(checkpoint),
        "--seen-manifest", str(manifest_path(campaign, cell)),
        "--subset", "probe",
        "--inference-seeds", str(INFERENCE_SEED),
        "--modes", *MODES,
        "--save-diagnostics",
        "--output-dir", str(output),
    )
    _run_runtime(command, root=root, campaign=campaign, name=f"{cell}_seed{seed}_eval", gpu=args.gpu)
    if not _eval_complete(output):
        raise RuntimeError(f"evaluation completeness check failed: {output}")


def preflight_cell(args, cell: str) -> dict:
    """Run one real-Wan inference group without training or checkpoint writes."""
    import torch

    root = args.repo_root.resolve()
    campaign = args.campaign_root.resolve()
    config_payload = _validate_config(root, cell)
    manifest = read_json(manifest_path(campaign, cell))
    group = next(item for item in manifest["groups"] if item["id"] == PROBE_IDS[0])
    config_module = __import__("rl3dsr.validation.stage3_protocol", fromlist=["load_stage3_config"])
    config = config_module.load_stage3_config(config_path(root, cell))
    stage3 = __import__("stage3_experiment")
    if not _gpu_is_idle(args.gpu):
        raise RuntimeError(f"GPU {args.gpu} is not idle; refusing preflight {cell}")
    runtime = stage3.load_runtime(
        config,
        model_dir=args.model_dir,
        lq_source=args.lq_source,
        lq_checkpoint=args.lq_checkpoint,
        bridge_checkpoint=args.bridge_checkpoint,
        device="cuda",
    )
    runtime.module.eval()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    _, hr, lr, camera = stage3._load_indices(
        args.dataset_root,
        group["scene"],
        "train",
        group["indices"],
        config,
        runtime.device,
    )
    with torch.inference_mode():
        latent_shape = tuple(runtime.vae.encode_multiview(hr).shape)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        sampled, diagnostics = stage3.sample_latents(
            runtime,
            lr,
            camera,
            latent_shape,
            config.sampling_steps,
            config.image_size,
            seed=INFERENCE_SEED,
            sampling_shift=config.sampling_shift,
            dtype=torch.bfloat16,
            return_diagnostics=True,
        )
        decoded = runtime.vae.decode_multiview(sampled)
    result = {
        "cell": cell,
        "views": CELL_CONFIGS[cell][1],
        "group_id": group["id"],
        "latent_shape": list(latent_shape),
        "sample_shape": list(sampled.shape),
        "decoded_shape": list(decoded.shape),
        "sampling_steps": config.sampling_steps,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(runtime.device) / 2**20,
        "fusion_diagnostics": diagnostics.get("fusion_diagnostics", {}),
    }
    write_json(campaign / "preflight" / f"{cell}.json", result)
    return result


def _index_rows(path: Path) -> dict[tuple[str, str, int], dict]:
    rows = read_jsonl(path / "evaluation_rows.jsonl")
    if len(rows) != 28:
        raise ValueError(f"unexpected row count in {path}")
    indexed = {}
    for row in rows:
        key = (row.get("condition"), row.get("group_id"), int(row.get("view_index")))
        if key in indexed:
            raise ValueError(f"duplicate evaluation identity: {key}")
        indexed[key] = row
    for group_id in PROBE_IDS:
        view_index = int(group_id.split(":", 1)[1])
        for mode in MODES:
            if (mode, group_id, view_index) not in indexed:
                raise ValueError(f"missing paired row: {mode}/{group_id}")
    return indexed


def _mean_rows(indexed: dict[tuple[str, str, int], dict], condition: str) -> dict[str, float]:
    rows = [indexed[(condition, group_id, int(group_id.split(":", 1)[1]))] for group_id in PROBE_IDS]
    return {metric: mean(float(row[metric]) for row in rows) for metric in METRICS}


def _delta_rows(indexed: dict[tuple[str, str, int], dict], intervention: str) -> list[dict]:
    result = []
    for group_id in PROBE_IDS:
        view_index = int(group_id.split(":", 1)[1])
        correct = indexed[("correct", group_id, view_index)]
        changed = indexed[(intervention, group_id, view_index)]
        result.append({
            "group_id": group_id,
            **{
                metric: (float(correct[metric]) - float(changed[metric])) * GOOD_DIRECTION[metric]
                for metric in METRICS
            },
        })
    return result


def _summary_delta(rows: list[dict]) -> dict:
    return {
        metric: mean(float(row[metric]) for row in rows)
        for metric in METRICS
    } | {
        f"positive_count_{metric}": sum(float(row[metric]) > 0 for row in rows)
        for metric in METRICS
    }


def _fusion_means(path: Path) -> dict[str, float]:
    diagnostics = read_jsonl(path / "diagnostics.jsonl")
    values: dict[str, list[float]] = {}
    for row in diagnostics:
        block = ((row.get("fusion_correct") or {}).get("fusion") or {})
        for name, value in block.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(name, []).append(float(value))
    return {name: mean(items) for name, items in values.items() if items}


def _all_finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _checkpoint_integrity(path: Path, *, config: dict, seed: int, bridge_hash: str) -> dict:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        provenance = dict(payload.get("provenance") or {})
        adapters = dict(payload.get("adapters") or {})
        adapter_tensors = [
            tensor
            for state in adapters.values()
            if isinstance(state, dict)
            for tensor in state.values()
        ]
        checks = {
            "format": payload.get("format") == "rl3dsr-stage3" and payload.get("format_version") == 1,
            "step": payload.get("step") == STEPS,
            "config": payload.get("config") == config,
            "training_seed": provenance.get("training_seed") == seed,
            "bridge_hash": provenance.get("bridge_checkpoint_sha256") == bridge_hash,
            "rre_checkpoint_absent": provenance.get("rre_checkpoint") is None,
            "adapter_tensors_present": bool(adapter_tensors),
            "adapter_tensors_finite": bool(adapter_tensors) and all(torch.isfinite(tensor).all().item() for tensor in adapter_tensors),
            "training_state_step": (payload.get("training_state") or {}).get("step") == STEPS,
        }
        return {"pass": all(checks.values()), "checks": checks, "error": None}
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return {"pass": False, "checks": {}, "error": str(error)}


def _diagnostic_integrity(path: Path) -> dict:
    rows = read_jsonl(path / "diagnostics.jsonl")
    evaluation = _index_rows(path)
    expected_flags = {
        "correct_repeat": (False, False),
        "remove": (False, False),
        "target_drop": (False, False),
        "shuffle_fusion": (True, False),
        "shuffle_geometry": (False, True),
        "shuffle_pair": (True, True),
    }
    conditions = {condition: [row for row in rows if row.get("condition") == condition] for condition in expected_flags}
    camera_scope = all(
        len(selected) == len(PROBE_IDS)
        and all(
            (bool(row.get("fusion_camera_changed")), bool(row.get("geometry_camera_changed"))) == expected_flags[condition]
            for row in selected
        )
        for condition, selected in conditions.items()
    )
    paired_seed = all(
        row.get("inference_seed") == INFERENCE_SEED
        and row.get("sample_seed") == INFERENCE_SEED * 1_000_000 + int(row["group_id"].split(":", 1)[1])
        for row in rows
    )
    stats = []
    for row in rows:
        for name in ("fusion_correct", "fusion_changed"):
            block = ((row.get(name) or {}).get("fusion") or {})
            if block:
                stats.append(block)
    attention = bool(stats) and all(
        abs(float(block.get("attention_mass_total", math.nan)) - 1.0) <= 1e-5
        and abs(
            float(block.get("same_view_attention_mass", math.nan))
            + float(block.get("cross_view_attention_mass", math.nan))
            + float(block.get("null_attention_mass", math.nan))
            - float(block.get("attention_mass_total", math.nan))
        ) <= 1e-5
        and all(
            0.0 <= float(block.get(key, math.nan)) <= 1.0
            for key in ("retained_key_ratio", "active_auxiliary_source_ratio")
        )
        for block in stats
    )
    jitter = {
        metric: max(
            abs(
                float(evaluation[("correct", group_id, int(group_id.split(":", 1)[1]))][metric])
                - float(evaluation[("correct_repeat", group_id, int(group_id.split(":", 1)[1]))][metric])
            )
            for group_id in PROBE_IDS
        )
        for metric in METRICS
    }
    checks = {
        "row_count": len(rows) == 24,
        "camera_scope": camera_scope,
        "paired_sample_seed": paired_seed,
        "correct_repeat_exact": all(value == 0.0 for value in jitter.values()),
        "attention_mass_and_ratios": attention,
        "finite": _all_finite(rows),
    }
    return {"pass": all(checks.values()), "checks": checks, "repeat_jitter": jitter}


def _preflight_integrity(campaign: Path, cell: str) -> dict:
    path = campaign / "preflight" / f"{cell}.json"
    if not path.is_file():
        return {"pass": False, "checks": {"artifact": False}}
    payload = read_json(path)
    views = CELL_CONFIGS[cell][1]
    block = ((payload.get("fusion_diagnostics") or {}).get("fusion") or {})
    checks = {
        "artifact": True,
        "cell": payload.get("cell") == cell and payload.get("views") == views,
        "latent_shape": payload.get("latent_shape") == [1, 16, views, 32, 32],
        "sample_shape": payload.get("sample_shape") == [1, 16, views, 32, 32],
        "decoded_shape": payload.get("decoded_shape") == [1, 3, views, 256, 256],
        "sampling_steps": payload.get("sampling_steps") == 50,
        "peak_memory_finite": math.isfinite(float(payload.get("peak_gpu_memory_mib", math.nan))),
        "attention_mass": abs(float(block.get("attention_mass_total", math.nan)) - 1.0) <= 1e-5,
        "key_ratio": 0.0 <= float(block.get("retained_key_ratio", math.nan)) <= 1.0,
    }
    return {"pass": all(checks.values()), "checks": checks}


def _cell_integrity(root: Path, campaign: Path, protocol: dict, cell: str, seed: int) -> dict:
    config_file = config_path(root, cell)
    manifest_file = manifest_path(campaign, cell)
    train_dir = train_path(campaign, cell, seed)
    evaluation_dir = eval_path(campaign, cell, seed)
    config = read_json(config_file)
    manifest = read_json(manifest_file)
    run_manifest = read_json(train_dir / "run_manifest.json")
    evaluation_summary = read_json(evaluation_dir / "evaluation_summary.json")
    train_rows = read_jsonl(train_dir / "train_steps.jsonl")
    evaluation_rows = read_jsonl(evaluation_dir / "evaluation_rows.jsonl")
    baseline_rows = read_jsonl(evaluation_dir / "baseline_rows.jsonl")
    provenance = dict(run_manifest.get("provenance") or {})
    bridge_path = Path(provenance.get("bridge_checkpoint", ""))
    bridge_hash = sha256_file(bridge_path) if bridge_path.is_file() else ""
    protocol_cell = protocol["cells"][cell]
    manifest_hash = sha256_file(manifest_file)
    dataset_checks = {}
    dataset_root = Path(manifest.get("dataset_root", ""))
    for scene, expected in manifest.get("datasets", {}).items():
        scene_root = dataset_root / scene
        dataset_checks[scene] = {
            "views": expected.get("views") == len(list((scene_root / "train").glob("*.png"))),
            "transforms_train_sha256": expected.get("transforms_train_sha256") == sha256_file(scene_root / "transforms_train.json"),
            "train_images_sha256": expected.get("train_images_sha256") == sha256_tree(scene_root / "train"),
        }
    counts = {
        name: run_manifest.get(name) == expected
        for name, expected in EXPECTED_PARAMETER_COUNTS.items()
    }
    checkpoint = _checkpoint_integrity(
        train_dir / f"stage3_step_{STEPS:04d}.pt",
        config=config,
        seed=seed,
        bridge_hash=bridge_hash,
    )
    diagnostics = _diagnostic_integrity(evaluation_dir)
    log_text = "\n".join(
        (campaign / "logs" / f"{cell}_seed{seed}_{phase}.log").read_text(encoding="utf-8", errors="replace")
        for phase in ("train", "eval")
    ).lower()
    checks = {
        "config_sha256": sha256_file(config_file) == protocol_cell["config_sha256"],
        "manifest_sha256": manifest_hash == protocol_cell["manifest_sha256"],
        "train_log_continuous": _train_complete(train_dir, STEPS),
        "evaluation_complete": _eval_complete(evaluation_dir),
        "config_matches_run": run_manifest.get("config") == config,
        "training_seed": provenance.get("training_seed") == seed and evaluation_summary.get("training_seed") == seed,
        "bridge_hash": bool(bridge_hash) and provenance.get("bridge_checkpoint_sha256") == bridge_hash,
        "rre_checkpoint_absent": provenance.get("rre_checkpoint") is None,
        "parameter_counts": all(counts.values()),
        "checkpoint_load": checkpoint["pass"],
        "evaluation_manifest": evaluation_summary.get("seen_manifest_sha256") == manifest_hash,
        "dataset_manifest": evaluation_summary.get("dataset_manifest") == manifest.get("datasets") and all(all(values.values()) for values in dataset_checks.values()),
        "finite_artifacts": _all_finite(train_rows) and _all_finite(evaluation_rows) and _all_finite(baseline_rows),
        "diagnostics": diagnostics["pass"],
        "logs_clean": not any(token in log_text for token in ("out of memory", "traceback", "\nnan", " nan")),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "parameter_counts": counts,
        "dataset_checks": dataset_checks,
        "checkpoint": checkpoint,
        "diagnostics": diagnostics,
    }


def _integrity(root: Path, campaign: Path, protocol: dict, seed: int) -> dict:
    v4 = _manifest_groups(manifest_path(campaign, "v4"))
    v8 = _manifest_groups(manifest_path(campaign, "v8"))
    anchors_match = set(v4) == set(v8) and all(v4[key]["anchor"] == v8[key]["anchor"] for key in v4)
    nested = anchors_match and all(set(v4[key]["indices"]).issubset(v8[key]["indices"]) for key in v4)
    protocol_nested = protocol.get("nested_groups") == [
        {"id": key, "v4": v4[key]["indices"], "v8": v8[key]["indices"]}
        for key in sorted(v4)
    ]
    configs = {cell: read_json(config_path(root, cell)) for cell in CELL_CONFIGS}
    common_v4 = {key: value for key, value in configs["v4"].items() if key != "views"}
    common_v8 = {key: value for key, value in configs["v8"].items() if key != "views"}
    cells = {cell: _cell_integrity(root, campaign, protocol, cell, seed) for cell in CELL_CONFIGS}
    preflight = {cell: _preflight_integrity(campaign, cell) for cell in CELL_CONFIGS}
    checks = {
        "anchors_match": anchors_match,
        "nested_view_sets": nested,
        "protocol_nested_sets": protocol_nested,
        "config_common_fields": common_v4 == common_v8,
        "dataset_hashes_match": read_json(manifest_path(campaign, "v4")).get("datasets") == read_json(manifest_path(campaign, "v8")).get("datasets"),
        "preflight": all(item["pass"] for item in preflight.values()),
        "cells": all(item["pass"] for item in cells.values()),
    }
    return {"pass": all(checks.values()), "checks": checks, "preflight": preflight, "cells": cells}


def _gate(cell_payload: dict[str, dict], integrity: dict) -> dict:
    v4 = cell_payload["v4"]
    v8 = cell_payload["v8"]
    correct4, correct8 = v4["means"]["correct"], v8["means"]["correct"]
    correct_nonregression = {
        "psnr": correct8["psnr"] - correct4["psnr"] >= -0.25,
        "ssim": correct8["ssim"] - correct4["ssim"] >= -0.005,
        "lpips": correct8["lpips"] - correct4["lpips"] <= 0.01,
        "mae": correct8["mae"] - correct4["mae"] <= 0.002,
    }
    target_floor = {
        cell: payload["deltas"]["target_drop"]["psnr"] >= 4.0
        and payload["deltas"]["target_drop"]["ssim"] >= 0.05
        for cell, payload in cell_payload.items()
    }
    aux_diff_rows = [
        {
            metric: v8_row[metric] - v4_row[metric]
            for metric in METRICS
        }
        for v4_row, v8_row in zip(v4["delta_rows"]["remove"], v8["delta_rows"]["remove"])
    ]
    fusion_diff_rows = [
        {
            metric: v8_row[metric] - v4_row[metric]
            for metric in METRICS
        }
        for v4_row, v8_row in zip(v4["delta_rows"]["shuffle_fusion"], v8["delta_rows"]["shuffle_fusion"])
    ]
    aux = {
        metric: {
            "mean_difference": mean(row[metric] for row in aux_diff_rows),
            "positive_probe_count": sum(row[metric] > 0 for row in aux_diff_rows),
        }
        for metric in METRICS
    }
    fusion = {
        metric: {
            "mean_difference": mean(row[metric] for row in fusion_diff_rows),
            "positive_probe_count": sum(row[metric] > 0 for row in fusion_diff_rows),
        }
        for metric in METRICS
    }
    aux_pass = (
        aux["psnr"]["mean_difference"] > 0
        and aux["ssim"]["mean_difference"] > 0
        and aux["psnr"]["positive_probe_count"] >= 3
        and aux["ssim"]["positive_probe_count"] >= 3
        and aux["lpips"]["mean_difference"] >= 0
        and aux["mae"]["mean_difference"] >= 0
    )
    fusion_pass = (
        fusion["psnr"]["mean_difference"] > 0
        and fusion["ssim"]["mean_difference"] > 0
        and fusion["psnr"]["positive_probe_count"] >= 3
        and fusion["ssim"]["positive_probe_count"] >= 3
        and fusion["lpips"]["mean_difference"] >= 0
        and fusion["mae"]["mean_difference"] >= 0
    )
    checks = {
        "pilot_integrity": integrity["pass"],
        "correct_nonregression": all(correct_nonregression.values()),
        "target_floor_v4": target_floor["v4"],
        "target_floor_v8": target_floor["v8"],
        "auxiliary_view_gain": aux_pass,
        "fusion_camera_gain": fusion_pass,
    }
    return {
        "checks": checks,
        "correct_nonregression": correct_nonregression,
        "target_floor": target_floor,
        "auxiliary_view": aux,
        "fusion_camera": fusion,
        "pilot_pass": all(checks.values()),
    }


def analyze(args, *, seeds: Iterable[int] = (SEED,)) -> dict:
    campaign = args.campaign_root.resolve()
    root = args.repo_root.resolve()
    protocol = read_json(campaign / "protocol.json")
    cells_payload = {}
    replication_payload = {}
    integrity_payload = {}
    for seed in seeds:
        per_seed = {}
        for cell in CELL_CONFIGS:
            path = eval_path(campaign, cell, seed)
            indexed = _index_rows(path)
            delta_rows = {mode: _delta_rows(indexed, mode) for mode in ("remove", "target_drop", "shuffle_fusion", "shuffle_geometry", "shuffle_pair")}
            per_seed[cell] = {
                "seed": seed,
                "means": {mode: _mean_rows(indexed, mode) for mode in MODES},
                "deltas": {mode: _summary_delta(rows) for mode, rows in delta_rows.items()},
                "delta_rows": delta_rows,
                "fusion_diagnostics": _fusion_means(path),
            }
        integrity = _integrity(root, campaign, protocol, seed)
        integrity_payload[str(seed)] = integrity
        gate = _gate(per_seed, integrity)
        if seed == SEED:
            cells_payload = per_seed
            pilot_gate = gate
        else:
            replication_payload = {"cells": per_seed, "gate": gate}
    result = {
        "scope": "stage3_1_view_count_sufficiency",
        "claim_limit": "seen-train SR only; no held-out, NVS, 3DGS, depth, 4DSR or RL claim",
        "protocol": protocol,
        "integrity": integrity_payload,
        "pilot": {"cells": cells_payload, "gate": pilot_gate},
        "replication": replication_payload,
    }
    analysis_dir = campaign / "analysis"
    write_json(analysis_dir / "view_count_summary.json", result)
    rows = []
    for phase, payload in (("pilot", {"cells": cells_payload, "gate": pilot_gate}), ("replication", replication_payload)):
        for cell, cell_payload in payload.get("cells", {}).items():
            for condition, metrics in cell_payload["deltas"].items():
                rows.append({"phase": phase, "cell": cell, "seed": cell_payload["seed"], "condition": condition, **metrics})
    if rows:
        write_csv(analysis_dir / "view_count_deltas.csv", rows)
    lines = [
        "# Stage 3.1 V4 vs V8 view-count sufficiency",
        "",
        "Claim boundary: seen-training-view SR only; this does not establish held-out generalization, NVS, 3DGS, depth, 4DSR or RL readiness.",
        "",
        "## Pilot seed 42",
        "",
        f"Pilot integrity: **{'PASS' if integrity_payload[str(SEED)]['pass'] else 'FAIL'}**",
        "",
        "| cell | condition | ΔPSNR | ΔSSIM | ΔLPIPS | ΔMAE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cell, payload in cells_payload.items():
        for condition, metrics in payload["deltas"].items():
            lines.append(f"| {cell} | {condition} | {metrics['psnr']:.6f} | {metrics['ssim']:.6f} | {metrics['lpips']:.6f} | {metrics['mae']:.6f} |")
    lines.extend([
        "",
        f"Pilot gate: **{'PASS' if pilot_gate['pilot_pass'] else 'HOLD'}**",
        "",
        "- Auxiliary-view gain means V8 minus V4 deltas for `remove`.",
        "- Fusion-camera gain means V8 minus V4 deltas for `shuffle_fusion`.",
        "- A PASS requires both gains, target-drop floors, and correct-output non-regression.",
    ])
    aux_pass = pilot_gate["checks"]["auxiliary_view_gain"]
    fusion_pass = pilot_gate["checks"]["fusion_camera_gain"]
    if aux_pass and not fusion_pass:
        lines.append("- Interpretation: auxiliary images help, but correct camera pairing still does not drive fusion strongly enough.")
    elif aux_pass and fusion_pass:
        lines.append("- Interpretation: both auxiliary evidence and camera-paired fusion improve with V8.")
    else:
        lines.append("- Interpretation: V8 does not establish stronger auxiliary-view use under the preregistered gate.")
    if not pilot_gate["pilot_pass"]:
        lines.append("- Replication: not run; seed 43 is gated off by the pilot HOLD.")
    if replication_payload:
        lines.extend(["", "## Seed 43 replication", "", f"Replication gate: **{'PASS' if replication_payload['gate']['pilot_pass'] else 'HOLD'}**"])
    (analysis_dir / "view_count_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _require_runtime(args) -> None:
    for name in ("dataset_root", "model_dir", "lq_source", "lq_checkpoint", "bridge_checkpoint"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} must exist: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "train", "evaluate", "analyze", "run-pilot", "run-all"))
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/nerf_synthetic"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/Wan2.1-T2V-1.3B"))
    parser.add_argument("--lq-source", type=Path)
    parser.add_argument("--lq-checkpoint", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cell", choices=tuple(CELL_CONFIGS))
    parser.add_argument("--seed", type=int, choices=(SEED, REPLICATION_SEED))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.campaign_root = args.campaign_root.resolve()
    if args.command == "prepare":
        _require_runtime(args)
        print(json.dumps(prepare(args), indent=2, ensure_ascii=False))
        return
    if args.command == "analyze":
        result = analyze(args, seeds=(SEED, REPLICATION_SEED) if (args.campaign_root / "train" / "v4" / "seed43").exists() else (SEED,))
        print(json.dumps({"pilot_pass": result["pilot"]["gate"]["pilot_pass"], "replication": bool(result["replication"])}, indent=2))
        return
    _require_runtime(args)
    prepare(args)
    if args.command == "preflight":
        if args.cell is None:
            raise ValueError("--cell is required for preflight")
        result = preflight_cell(args, args.cell)
        print(json.dumps(result, indent=2))
        return
    if args.command == "train":
        if args.cell is None or args.seed is None:
            raise ValueError("--cell and --seed are required for train")
        train_cell(args, args.cell, args.seed)
        return
    if args.command == "evaluate":
        if args.cell is None or args.seed is None:
            raise ValueError("--cell and --seed are required for evaluate")
        evaluate_cell(args, args.cell, args.seed)
        return
    for cell in CELL_CONFIGS:
        train_cell(args, cell, SEED)
        evaluate_cell(args, cell, SEED)
    pilot = analyze(args, seeds=(SEED,))
    print(json.dumps({"pilot_pass": pilot["pilot"]["gate"]["pilot_pass"]}, indent=2))
    if args.command == "run-pilot" or not pilot["pilot"]["gate"]["pilot_pass"]:
        return
    for cell in CELL_CONFIGS:
        train_cell(args, cell, REPLICATION_SEED)
        evaluate_cell(args, cell, REPLICATION_SEED)
    result = analyze(args, seeds=(SEED, REPLICATION_SEED))
    print(json.dumps({
        "pilot_pass": result["pilot"]["gate"]["pilot_pass"],
        "replication_pass": result["replication"]["gate"]["pilot_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
