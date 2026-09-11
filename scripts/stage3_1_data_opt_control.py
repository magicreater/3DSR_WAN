#!/usr/bin/env python3
"""Run and analyze the Stage 3.1 data/optimization sufficiency control.

The driver intentionally leaves the Stage 3 model and dropout implementation
unchanged.  It owns only the four-cell experiment protocol, paired seen-view
evaluation, and a delta-only report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Iterable


SEED = 42
INFERENCE_SEED = 3302
PROBE_IDS = ("chair:000", "chair:033", "chair:066", "chair:099")
CONDITIONS = ("correct", "correct_repeat", "target_drop", "shuffle_camera")
INTERVENTIONS = ("target_drop", "shuffle_camera")
METRICS = ("psnr", "ssim", "lpips", "mae")
GOOD_DIRECTION = {
    "psnr": 1.0,
    "ssim": 1.0,
    "lpips": -1.0,
    "mae": -1.0,
}


CELL_CONFIGS = {
    "chair_1000": ("A3_target_drop_chair.json", ("chair",), 1000, "chair_probe.json"),
    "chair_2000": ("A3_target_drop_chair_2000.json", ("chair",), 2000, "chair_probe.json"),
    "five_1000": ("A3_target_drop_full_1000.json", ("chair", "lego", "drums", "hotdog", "mic"), 1000, "five_probe.json"),
    "five_2000": ("A3_target_drop_full.json", ("chair", "lego", "drums", "hotdog", "mic"), 2000, "five_probe.json"),
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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cell_config(root: Path, cell: str) -> Path:
    try:
        filename = CELL_CONFIGS[cell][0]
    except KeyError as exc:
        raise ValueError(f"unknown cell: {cell}") from exc
    return root / "configs" / "stage3_1" / filename


def load_cell_config(root: Path, cell: str) -> dict:
    path = cell_config(root, cell)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    expected_scenes, expected_steps, _manifest = CELL_CONFIGS[cell][1:]
    if payload.get("arm") != "A3":
        raise ValueError(f"{cell}: arm must remain A3")
    if tuple(payload.get("train_scenes", ())) != expected_scenes:
        raise ValueError(f"{cell}: unexpected train_scenes")
    if int(payload.get("steps", -1)) != expected_steps:
        raise ValueError(f"{cell}: unexpected steps")
    if float(payload.get("target_lr_dropout", -1.0)) != 0.5:
        raise ValueError(f"{cell}: target_lr_dropout must remain 0.5")
    if SEED not in payload.get("training_seeds", ()):
        raise ValueError(f"{cell}: training seed {SEED} is not registered")
    return payload


def inspect_configs(root: Path) -> dict:
    configs = {cell: load_cell_config(root, cell) for cell in CELL_CONFIGS}
    invariant_fields = {
        key
        for key in configs["chair_1000"]
        if key not in {"train_scenes", "validation_scenes", "test_scenes", "steps"}
    }
    for field in sorted(invariant_fields):
        values = {json.dumps(config.get(field), sort_keys=True) for config in configs.values()}
        if len(values) != 1:
            raise ValueError(f"config drift in invariant field {field}: {sorted(values)}")
    return {
        "cells": {
            cell: {
                "config": str(cell_config(root, cell).resolve()),
                "train_scenes": list(config["train_scenes"]),
                "steps": config["steps"],
                "target_lr_dropout": config["target_lr_dropout"],
            }
            for cell, config in configs.items()
        },
        "training_seed": SEED,
        "inference_seed": INFERENCE_SEED,
        "probe_ids": list(PROBE_IDS),
        "invariant_fields": sorted(invariant_fields),
    }


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(map(str, command)), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"command failed with {returncode}; see {log_path}")


def _command_file(path: Path, command: list[str]) -> None:
    text = " ".join(map(str, command)) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"command drift detected: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _env(repo: Path, gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(repo / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    return environment


def _gpu_idle(gpu: int) -> bool:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3 or int(fields[0]) != gpu:
            continue
        return int(float(fields[1])) <= 500 and int(float(fields[2])) <= 5
    raise RuntimeError(f"GPU {gpu} was not reported by nvidia-smi")


def _source_manifest_path(campaign: Path, manifest_name: str) -> Path:
    stem = "chair" if manifest_name == "chair_probe.json" else "five"
    return campaign / "manifest" / f"generated_{stem}_full.json"


def _control_manifest_path(campaign: Path, manifest_name: str) -> Path:
    return campaign / "manifest" / manifest_name


def _make_control_manifest(source: dict, config: dict, source_path: Path) -> dict:
    available = {row["id"]: row for row in source.get("groups", [])}
    if any(group_id not in available for group_id in PROBE_IDS):
        raise ValueError(f"source manifest lacks required probe groups: {source_path}")
    groups = [available[group_id] for group_id in PROBE_IDS]
    datasets = {"chair": source.get("datasets", {}).get("chair")}
    if datasets["chair"] is None:
        raise ValueError(f"source manifest lacks chair dataset metadata: {source_path}")
    return {
        "version": 1,
        "scope": "seen_train_sr",
        "sampling_signature": {
            "train_scenes": list(config["train_scenes"]),
            "image_size": config["image_size"],
            "scale": config["scale"],
            "views": config["views"],
        },
        "dataset_root": source["dataset_root"],
        "datasets": datasets,
        "groups": groups,
        "subsets": {
            "full": list(PROBE_IDS),
            "probe": list(PROBE_IDS),
            "intervention": list(PROBE_IDS),
        },
        "control_protocol": {
            "source_manifest": str(source_path.resolve()),
            "probe_ids": list(PROBE_IDS),
        },
    }


def prepare_manifests(args) -> dict:
    root = args.repo_root.resolve()
    campaign = args.campaign_root.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    inspect_configs(root)
    source_paths = {}
    for cell in CELL_CONFIGS:
        config = load_cell_config(root, cell)
        manifest_name = CELL_CONFIGS[cell][3]
        source = _source_manifest_path(campaign, manifest_name)
        source_paths[cell] = source
        if not source.is_file():
            command = [
                args.python,
                str(root / "scripts" / "stage3_experiment.py"),
                "prepare-seen",
                "--config",
                str(cell_config(root, cell)),
                "--dataset-root",
                str(args.dataset_root),
                "--manifest",
                str(source),
            ]
            _command_file(campaign / "control" / f"prepare_{cell}.command.txt", command)
            _run_command(command, cwd=root, env=_env(root, args.gpu), log_path=campaign / "logs" / f"prepare_{cell}.log")
        source_payload = read_json(source)
        control = _make_control_manifest(source_payload, config, source)
        target = _control_manifest_path(campaign, manifest_name)
        encoded = json.dumps(control, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if target.is_file():
            if target.read_text(encoding="utf-8") != encoded:
                raise RuntimeError(f"control manifest drift detected: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(encoded, encoding="utf-8")
    chair = read_json(_control_manifest_path(campaign, "chair_probe.json"))
    five = read_json(_control_manifest_path(campaign, "five_probe.json"))
    if chair["groups"] != five["groups"] or chair["subsets"] != five["subsets"]:
        raise RuntimeError("chair and five-scene control manifests do not share identical probe groups")
    manifest_hashes = {
        name: sha256_file(_control_manifest_path(campaign, name))
        for name in ("chair_probe.json", "five_probe.json")
    }
    payload = {
        "schema_version": 1,
        "scope": "stage3_1_data_optimization_control",
        "training_seed": SEED,
        "inference_seed": INFERENCE_SEED,
        "probe_ids": list(PROBE_IDS),
        "cells": inspect_configs(root)["cells"],
        "source_manifests": {cell: str(path.resolve()) for cell, path in source_paths.items()},
        "control_manifests": manifest_hashes,
    }
    atomic_json(campaign / "protocol_manifest.json", payload)
    return payload


def _train_path(campaign: Path, cell: str) -> Path:
    return campaign / "train" / cell / f"seed{SEED}"


def _checkpoint(path: Path, steps: int) -> Path:
    return path / f"stage3_step_{steps:04d}.pt"


def _train_rows(path: Path) -> list[dict]:
    rows = read_jsonl(path / "train_steps.jsonl")
    expected = list(range(1, len(rows) + 1))
    if [int(row.get("step", -1)) for row in rows] != expected:
        raise ValueError(f"non-contiguous training log: {path}")
    return rows


def _find_resume_checkpoint(path: Path, target_steps: int) -> Path | None:
    candidates = []
    for item in path.glob("stage3_step_*.pt"):
        try:
            step = int(item.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if 0 < step < target_steps:
            candidates.append((step, item))
    return max(candidates, default=(0, None))[1]


def _assert_training_complete(path: Path, steps: int, config: dict) -> None:
    checkpoint = _checkpoint(path, steps)
    if not checkpoint.is_file():
        raise RuntimeError(f"missing final checkpoint: {checkpoint}")
    rows = _train_rows(path)
    if len(rows) != steps:
        raise RuntimeError(f"training log length {len(rows)} != {steps}: {path}")
    manifest = read_json(path / "run_manifest.json")
    saved = dict(manifest.get("config") or {})
    saved.setdefault("target_lr_dropout", 0.0)
    expected = json.loads(json.dumps(config))
    if saved != expected or manifest.get("provenance", {}).get("training_seed") != SEED:
        raise RuntimeError(f"training manifest mismatch: {path}")
    if sum(int(row.get("target_lr_drop_count", 0)) for row in rows) <= 0:
        raise RuntimeError(f"target dropout was never sampled: {path}")


def train_cell(args, cell: str) -> None:
    root = args.repo_root.resolve()
    campaign = args.campaign_root.resolve()
    config = load_cell_config(root, cell)
    output = _train_path(campaign, cell)
    final = _checkpoint(output, int(config["steps"]))
    if final.is_file() and (output / "run_manifest.json").is_file():
        _assert_training_complete(output, int(config["steps"]), config)
        print(f"SKIP complete train {cell}", flush=True)
        return
    if output.exists() and not any(output.iterdir()):
        output.rmdir()
    resume = _find_resume_checkpoint(output, int(config["steps"])) if output.exists() else None
    if output.exists() and resume is None and any(output.iterdir()):
        raise RuntimeError(f"partial training output has no resumable checkpoint: {output}")
    command = [
        args.python,
        str(root / "scripts" / "stage3_experiment.py"),
        "train",
        "--config",
        str(cell_config(root, cell)),
        "--seed",
        str(SEED),
        "--dataset-root",
        str(args.dataset_root),
        "--model-dir",
        str(args.model_dir),
        "--lq-source",
        str(args.lq_source),
        "--lq-checkpoint",
        str(args.lq_checkpoint),
        "--bridge-checkpoint",
        str(args.bridge_checkpoint),
        "--output-dir",
        str(output),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    _command_file(campaign / "control" / f"{cell}_train.command.txt", command)
    if not _gpu_idle(args.gpu):
        raise RuntimeError(f"GPU {args.gpu} is not idle; refusing to start {cell}")
    _run_command(command, cwd=root, env=_env(root, args.gpu), log_path=campaign / "logs" / f"{cell}_train.log")
    _assert_training_complete(output, int(config["steps"]), config)


def _eval_complete(path: Path) -> bool:
    summary_path = path / "evaluation_summary.json"
    rows_path = path / "evaluation_rows.jsonl"
    baseline_path = path / "baseline_rows.jsonl"
    if not (summary_path.is_file() and rows_path.is_file() and baseline_path.is_file()):
        return False
    summary = read_json(summary_path)
    rows = read_jsonl(rows_path)
    baseline = read_jsonl(baseline_path)
    return (
        summary.get("rows") == 16
        and summary.get("groups") == 4
        and summary.get("inference_seeds") == [INFERENCE_SEED]
        and set(summary.get("conditions", ())) == set(CONDITIONS)
        and len(rows) == 16
        and len(baseline) == 8
    )


def evaluate_cell(args, cell: str) -> None:
    root = args.repo_root.resolve()
    campaign = args.campaign_root.resolve()
    config = load_cell_config(root, cell)
    train_dir = _train_path(campaign, cell)
    checkpoint = _checkpoint(train_dir, int(config["steps"]))
    _assert_training_complete(train_dir, int(config["steps"]), config)
    output = train_dir / "eval"
    if output.exists() and _eval_complete(output):
        print(f"SKIP complete eval {cell}", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete evaluation: {output}")
    manifest_name = CELL_CONFIGS[cell][3]
    command = [
        args.python,
        str(root / "scripts" / "stage3_experiment.py"),
        "seen-eval",
        "--config",
        str(cell_config(root, cell)),
        "--checkpoint",
        str(checkpoint),
        "--dataset-root",
        str(args.dataset_root),
        "--model-dir",
        str(args.model_dir),
        "--lq-source",
        str(args.lq_source),
        "--lq-checkpoint",
        str(args.lq_checkpoint),
        "--bridge-checkpoint",
        str(args.bridge_checkpoint),
        "--seen-manifest",
        str(campaign / "manifest" / manifest_name),
        "--subset",
        "probe",
        "--inference-seeds",
        str(INFERENCE_SEED),
        "--modes",
        *CONDITIONS,
        "--output-dir",
        str(output),
    ]
    _command_file(campaign / "control" / f"{cell}_eval.command.txt", command)
    if not _gpu_idle(args.gpu):
        raise RuntimeError(f"GPU {args.gpu} is not idle; refusing to start evaluation {cell}")
    _run_command(command, cwd=root, env=_env(root, args.gpu), log_path=campaign / "logs" / f"{cell}_eval.log")
    if not _eval_complete(output):
        raise RuntimeError(f"evaluation completeness check failed: {output}")


def _metric_delta(correct: dict, intervention: dict, metric: str) -> float:
    return (float(correct[metric]) - float(intervention[metric])) * GOOD_DIRECTION[metric]


def _validate_eval_rows(cell: str, path: Path, steps: int) -> tuple[list[dict], dict]:
    rows = read_jsonl(path / "evaluation_rows.jsonl")
    baseline = read_jsonl(path / "baseline_rows.jsonl")
    if len(rows) != 16 or len(baseline) != 8:
        raise ValueError(f"{cell}: unexpected evaluation row counts")
    keys = {(row.get("condition"), row.get("inference_seed"), row.get("group_id"), row.get("view_index")) for row in rows}
    expected = {
        (condition, INFERENCE_SEED, group_id, int(group_id.split(":", 1)[1]))
        for condition in CONDITIONS
        for group_id in PROBE_IDS
    }
    if keys != expected:
        raise ValueError(f"{cell}: evaluation identities do not match fixed probe")
    if any(int(row.get("step", -1)) != steps for row in rows):
        raise ValueError(f"{cell}: evaluation step mismatch")
    indexed = {(row["condition"], row["inference_seed"], row["group_id"], row["view_index"]): row for row in rows}
    for group_id in PROBE_IDS:
        view_index = int(group_id.split(":", 1)[1])
        for condition in CONDITIONS:
            if (condition, INFERENCE_SEED, group_id, view_index) not in indexed:
                raise ValueError(f"{cell}: missing {condition} row for {group_id}")
    return rows, indexed


def _summary_row(cell: str, condition: str, delta_rows: list[dict]) -> dict:
    result = {"cell": cell, "condition": condition, "probe_count": len(delta_rows)}
    for metric in METRICS:
        values = [float(row[f"delta_{metric}"]) for row in delta_rows]
        result[f"delta_{metric}"] = mean(values)
        result[f"min_delta_{metric}"] = min(values)
        result[f"max_delta_{metric}"] = max(values)
        result[f"positive_count_{metric}"] = sum(value > 0 for value in values)
    return result


def analyze(args) -> dict:
    root = args.repo_root.resolve()
    campaign = args.campaign_root.resolve()
    inspect_configs(root)
    protocol_path = campaign / "protocol_manifest.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = read_json(protocol_path)
    if protocol.get("training_seed") != SEED or protocol.get("inference_seed") != INFERENCE_SEED:
        raise ValueError("control protocol seed drift")
    if tuple(protocol.get("probe_ids", ())) != PROBE_IDS:
        raise ValueError("control protocol probe drift")
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    cell_payloads = {}
    for cell, (_filename, _scenes, steps, _manifest) in CELL_CONFIGS.items():
        train_dir = _train_path(campaign, cell)
        config = load_cell_config(root, cell)
        train_rows = _train_rows(train_dir)
        if len(train_rows) != steps:
            raise ValueError(f"{cell}: expected {steps} train rows")
        rows, indexed = _validate_eval_rows(cell, train_dir / "eval", steps)
        deltas_by_condition = {}
        for condition in INTERVENTIONS:
            deltas = []
            for group_id in PROBE_IDS:
                view_index = int(group_id.split(":", 1)[1])
                correct = indexed[("correct", INFERENCE_SEED, group_id, view_index)]
                intervention = indexed[(condition, INFERENCE_SEED, group_id, view_index)]
                row = {
                    "cell": cell,
                    "condition": condition,
                    "inference_seed": INFERENCE_SEED,
                    "group_id": group_id,
                    "view_index": view_index,
                }
                for metric in METRICS:
                    row[f"delta_{metric}"] = _metric_delta(correct, intervention, metric)
                deltas.append(row)
                detail_rows.append(row)
            deltas_by_condition[condition] = deltas
            summary_rows.append(_summary_row(cell, condition, deltas))
        cell_payloads[cell] = {
            "train_scenes": list(config["train_scenes"]),
            "steps": steps,
            "train_rows": len(train_rows),
            "target_lr_drop_count": sum(int(row.get("target_lr_drop_count", 0)) for row in train_rows),
            "evaluation_rows": len(rows),
            "baseline_rows": 8,
            "deltas": {
                condition: _summary_row(cell, condition, deltas_by_condition[condition])
                for condition in INTERVENTIONS
            },
        }
    summary_index = {(row["cell"], row["condition"]): row for row in summary_rows}
    contrast_rows = []
    for condition in INTERVENTIONS:
        for metric in METRICS:
            contrast_rows.extend(
                [
                    {
                        "contrast": "step_effect_chair",
                        "condition": condition,
                        "metric": metric,
                        "value": summary_index[("chair_2000", condition)][f"delta_{metric}"] - summary_index[("chair_1000", condition)][f"delta_{metric}"],
                    },
                    {
                        "contrast": "step_effect_five",
                        "condition": condition,
                        "metric": metric,
                        "value": summary_index[("five_2000", condition)][f"delta_{metric}"] - summary_index[("five_1000", condition)][f"delta_{metric}"],
                    },
                    {
                        "contrast": "scene_effect_1000",
                        "condition": condition,
                        "metric": metric,
                        "value": summary_index[("five_1000", condition)][f"delta_{metric}"] - summary_index[("chair_1000", condition)][f"delta_{metric}"],
                    },
                    {
                        "contrast": "scene_effect_2000",
                        "condition": condition,
                        "metric": metric,
                        "value": summary_index[("five_2000", condition)][f"delta_{metric}"] - summary_index[("chair_2000", condition)][f"delta_{metric}"],
                    },
                ]
            )
    analysis_dir = campaign / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "control_deltas.csv", detail_rows)
    write_csv(analysis_dir / "control_summary.csv", summary_rows)
    write_csv(analysis_dir / "factorial_contrasts.csv", contrast_rows)
    payload = {
        "schema_version": 1,
        "scope": "stage3_1_data_optimization_control",
        "claim_limit": "delta-only seen-train SR intervention control; no SR-fit, held-out, NVS, 3DGS or Stage 4 effectiveness claim",
        "training_seed": SEED,
        "inference_seed": INFERENCE_SEED,
        "probe_ids": list(PROBE_IDS),
        "delta_definition": {
            "psnr": "correct - intervention",
            "ssim": "correct - intervention",
            "lpips": "intervention - correct",
            "mae": "intervention - correct",
        },
        "cells": cell_payloads,
        "factorial_contrasts": contrast_rows,
        "generated_unix": time.time(),
    }
    atomic_json(analysis_dir / "control_deltas.json", payload)
    lines = [
        "# Stage 3.1 数据/优化充分性控制",
        "",
        "仅报告固定 chair probe、固定 inference seed 下，`target_drop` 和 `shuffle_camera` 相对 `correct` 的配对差值。",
        "",
        "## Paired deltas",
        "",
        "| cell | condition | ΔPSNR | ΔSSIM | ΔLPIPS | ΔMAE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['cell']} | {row['condition']} | {row['delta_psnr']:.6f} | {row['delta_ssim']:.6f} | {row['delta_lpips']:.6f} | {row['delta_mae']:.6f} |"
        )
    lines.extend(["", "## Factorial contrasts", "", "| contrast | condition | metric | value |", "| --- | --- | --- | ---: |"])
    for row in contrast_rows:
        lines.append(f"| {row['contrast']} | {row['condition']} | {row['metric']} | {row['value']:.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `shuffle_camera` 随数据量或 steps 系统性增大、且 `target_drop` 保持存在，才支持训练充分性解释。",
            "- 四个 cell 的 `shuffle_camera` 仍接近零而 `target_drop` 明显，则更支持相机/几何路径问题。",
            "- 仅单个 cell 改善时，结论保持条件性，不升级为方法学通过。",
        ]
    )
    (analysis_dir / "control_deltas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "prepare", "train", "evaluate", "analyze", "run-all"))
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=False, default=Path("datasets/nerf_synthetic"))
    parser.add_argument("--model-dir", type=Path, required=False, default=Path("models/Wan2.1-T2V-1.3B"))
    parser.add_argument("--lq-source", type=Path, required=False)
    parser.add_argument("--lq-checkpoint", type=Path, required=False)
    parser.add_argument("--bridge-checkpoint", type=Path, required=False)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cell", choices=tuple(CELL_CONFIGS))
    return parser


def _require_runtime_args(args) -> None:
    for name in ("dataset_root", "model_dir", "lq_source", "lq_checkpoint", "bridge_checkpoint"):
        path = getattr(args, name)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} must exist: {path}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.campaign_root = args.campaign_root.resolve()
    if args.command == "inspect":
        print(json.dumps(inspect_configs(args.repo_root), indent=2, ensure_ascii=False))
        return
    if args.command == "prepare":
        _require_runtime_args(args)
        print(json.dumps(prepare_manifests(args), indent=2, ensure_ascii=False))
        return
    if args.command == "analyze":
        print(json.dumps(analyze(args), indent=2, ensure_ascii=False))
        return
    _require_runtime_args(args)
    if args.command == "train":
        if args.cell is None:
            raise ValueError("--cell is required for train")
        prepare_manifests(args)
        train_cell(args, args.cell)
        return
    if args.command == "evaluate":
        if args.cell is None:
            raise ValueError("--cell is required for evaluate")
        prepare_manifests(args)
        evaluate_cell(args, args.cell)
        return
    prepare_manifests(args)
    for cell in CELL_CONFIGS:
        train_cell(args, cell)
        evaluate_cell(args, cell)
    print(json.dumps(analyze(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
