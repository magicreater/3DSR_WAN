#!/usr/bin/env python3
"""Run the bounded Stage 3.1 local-band sufficiency campaign.

This driver owns experiment orchestration and paired analysis only.  It does
not change the Wan model, dropout implementation, or evaluation definitions.
The default ``run`` command executes the chair/seed42 pilot and continues to
the replication cells only when the predeclared pilot gate passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from statistics import mean


INFERENCE_SEED = 3302
PROBE_IDS = ("chair:000", "chair:033", "chair:066", "chair:099")
MODES = (
    "correct",
    "correct_repeat",
    "target_drop",
    "shuffle_fusion",
    "shuffle_geometry",
    "shuffle_all",
    "shuffle_pair",
)
METRICS = ("psnr", "ssim", "lpips", "mae")
GOOD_DIRECTION = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0, "mae": -1.0}
LOCAL_FIELDS = ("epipolar_attention", "epipolar_band")
CHECKPOINT_RE = re.compile(r"stage3_step_(\d{4})\.pt$")

PILOT_THRESHOLDS = {
    "correct_psnr_floor": -0.25,
    "correct_ssim_floor": -0.005,
    "correct_lpips_ceiling": 0.01,
    "correct_mae_ceiling": 0.002,
    "target_drop_psnr_floor": 4.0,
    "target_drop_ssim_floor": 0.05,
    "shuffle_fusion_psnr_floor": 0.05,
    "shuffle_fusion_ssim_floor": 0.0005,
    "shuffle_fusion_psnr_improvement": 0.04,
    "shuffle_fusion_ssim_improvement": 0.0004,
}

# These counts are part of the frozen Stage 3.1 runtime contract.  A changed
# count means the evaluated checkpoint was produced by a different model.
EXPECTED_PARAMETER_COUNTS = {
    "trainable_parameters": 47839104,
    "frozen_wan_parameters": 1418996800,
    "frozen_vae_parameters": 126892531,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def command_text(command: list[str]) -> str:
    return shlex.join([str(item) for item in command]) + "\n"


def config_defaults(payload: dict) -> dict:
    value = dict(payload)
    value.setdefault("epipolar_attention", "global_bias")
    value.setdefault("epipolar_band", 1.5)
    value.setdefault("target_lr_dropout", 0.0)
    return value


def copy_or_verify(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"artifact snapshot drift: {destination}")
        return
    shutil.copy2(source, destination)


def validate_local_config(path: Path, *, train_scenes: tuple[str, ...], steps: int = 1000) -> dict:
    payload = config_defaults(read_json(path))
    if payload.get("arm") != "A3":
        raise ValueError("local-band config must use arm A3")
    if tuple(payload.get("train_scenes", ())) != train_scenes:
        raise ValueError(f"unexpected train_scenes in {path}")
    if payload.get("steps") != steps:
        raise ValueError(f"unexpected steps in {path}")
    if payload.get("target_lr_dropout") != 0.5:
        raise ValueError("local-band config must keep target_lr_dropout=0.5")
    if payload.get("epipolar_attention") != "local_band":
        raise ValueError("local-band config must set epipolar_attention=local_band")
    if payload.get("epipolar_band") != 1.5:
        raise ValueError("local-band config must set epipolar_band=1.5")
    if 42 not in payload.get("training_seeds", ()) or 43 not in payload.get("training_seeds", ()):
        raise ValueError("local-band config must list training seeds 42 and 43")
    return payload


def compare_common_config(local_payload: dict, global_payload: dict) -> None:
    local = config_defaults(local_payload)
    baseline = config_defaults(global_payload)
    local["epipolar_attention"] = "global_bias"
    if local != baseline:
        differing = sorted(key for key in set(local) | set(baseline) if local.get(key) != baseline.get(key))
        raise ValueError(f"local/global config drift outside local-band fields: {differing}")


def _indexed(rows: list[dict]) -> dict[tuple, dict]:
    result: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("condition"), row.get("inference_seed"), row.get("group_id"), row.get("view_index"))
        if key[0] not in MODES or any(value is None for value in key[1:]):
            raise ValueError(f"invalid evaluation identity: {key}")
        if key in result:
            raise ValueError(f"duplicate evaluation identity: {key}")
        for metric in METRICS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"missing metric {metric}: {key}")
        result[key] = row
    return result


def paired_deltas(rows: list[dict]) -> dict:
    values = _indexed(rows)
    identities = sorted({key[1:] for key in values if key[0] == "correct"})
    if len(identities) != len(PROBE_IDS):
        raise ValueError(f"expected {len(PROBE_IDS)} correct identities, got {len(identities)}")
    if {identity[2] for identity in identities} != {0, 33, 66, 99}:
        raise ValueError("expected four fixed probe view indices")
    for identity in identities:
        for condition in MODES:
            if (condition, *identity) not in values:
                raise ValueError(f"missing paired condition: {(condition, *identity)}")
    means = {}
    for condition in MODES:
        means[condition] = {
            metric: mean(float(values[(condition, *identity)][metric]) for identity in identities)
            for metric in METRICS
        }
    deltas = {}
    per_probe = {}
    for condition in MODES[2:]:
        rows_for_condition = []
        for identity in identities:
            reference = values[("correct", *identity)]
            changed = values[(condition, *identity)]
            item = {
                metric: (float(reference[metric]) - float(changed[metric])) * GOOD_DIRECTION[metric]
                for metric in METRICS
            }
            rows_for_condition.append({"identity": identity, **item})
        per_probe[condition] = rows_for_condition
        deltas[condition] = {
            metric: mean(item[metric] for item in rows_for_condition)
            for metric in METRICS
        }
    jitter = {
        metric: max(
            abs(float(values[("correct", *identity)][metric]) - float(values[("correct_repeat", *identity)][metric]))
            for identity in identities
        )
        for metric in METRICS
    }
    return {"means": means, "deltas": deltas, "per_probe": per_probe, "repeat_jitter": jitter, "identities": identities}


def validate_camera_diagnostics(eval_dir: Path) -> dict:
    diagnostics = read_jsonl(eval_dir / "diagnostics.jsonl")
    expected_flags = {
        "correct_repeat": (False, False),
        "target_drop": (False, False),
        "shuffle_fusion": (True, False),
        "shuffle_geometry": (False, True),
        "shuffle_all": (True, True),
        "shuffle_pair": (True, True),
    }
    if len(diagnostics) != len(PROBE_IDS) * (len(MODES) - 1):
        raise ValueError("diagnostic row count mismatch")
    for condition, flags in expected_flags.items():
        selected = [row for row in diagnostics if row.get("condition") == condition]
        if len(selected) != len(PROBE_IDS):
            raise ValueError(f"diagnostic count mismatch for {condition}")
        if any((bool(row.get("fusion_camera_changed")), bool(row.get("geometry_camera_changed"))) != flags for row in selected):
            raise ValueError(f"camera hash scope mismatch for {condition}")
    return {"rows": len(diagnostics), "camera_scope_pass": True}


def eval_complete(eval_dir: Path) -> bool:
    required = ("evaluation_summary.json", "evaluation_rows.jsonl", "baseline_rows.jsonl", "diagnostics.jsonl")
    if not all((eval_dir / name).is_file() for name in required):
        return False
    summary = read_json(eval_dir / "evaluation_summary.json")
    rows = read_jsonl(eval_dir / "evaluation_rows.jsonl")
    baseline = read_jsonl(eval_dir / "baseline_rows.jsonl")
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
    )


def train_complete(train_dir: Path, steps: int = 1000) -> bool:
    log = train_dir / "train_steps.jsonl"
    checkpoint = train_dir / f"stage3_step_{steps:04d}.pt"
    manifest = train_dir / "run_manifest.json"
    if not all(path.is_file() for path in (log, checkpoint, manifest)):
        return False
    rows = read_jsonl(log)
    return len(rows) == steps and [row.get("step") for row in rows] == list(range(1, steps + 1))


def validate_run_manifest(train_dir: Path, *, config_path: Path, seed: int, bridge_checkpoint: Path) -> dict:
    manifest = read_json(train_dir / "run_manifest.json")
    expected_config = config_defaults(read_json(config_path))
    observed_config = config_defaults(dict(manifest.get("config") or {}))
    provenance = dict(manifest.get("provenance") or {})
    checks = {
        "config": observed_config == expected_config,
        "training_seed": provenance.get("training_seed") == seed,
        "bridge_hash": provenance.get("bridge_checkpoint_sha256") == sha256_file(bridge_checkpoint),
        "rre_checkpoint_absent": provenance.get("rre_checkpoint") is None,
    }
    checks.update({
        name: manifest.get(name) == expected
        for name, expected in EXPECTED_PARAMETER_COUNTS.items()
    })
    return {"pass": all(checks.values()), "checks": checks}


def integrity(train_dir: Path, eval_dir: Path, *, config_path: Path, seed: int,
              bridge_checkpoint: Path) -> dict:
    train_ok = train_complete(train_dir)
    eval_ok = eval_complete(eval_dir)
    manifest_ok = False
    manifest_checks = None
    manifest_error = None
    if train_ok:
        try:
            manifest_result = validate_run_manifest(
                train_dir,
                config_path=config_path,
                seed=seed,
                bridge_checkpoint=bridge_checkpoint,
            )
            manifest_ok = bool(manifest_result["pass"])
            manifest_checks = manifest_result["checks"]
        except (OSError, TypeError, ValueError) as exc:
            manifest_error = str(exc)
    camera_ok = False
    diagnostics_error = None
    if eval_ok:
        try:
            validate_camera_diagnostics(eval_dir)
            camera_ok = True
        except (OSError, ValueError) as exc:
            diagnostics_error = str(exc)
    return {
        "train": train_ok,
        "manifest": manifest_ok,
        "manifest_checks": manifest_checks,
        "manifest_error": manifest_error,
        "evaluation": eval_ok,
        "camera_scope": camera_ok,
        "diagnostics_error": diagnostics_error,
        "pass": train_ok and manifest_ok and eval_ok and camera_ok,
    }


def _gpu_is_idle(gpu: int) -> None:
    query = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT)
    selected = None
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and int(parts[0]) == gpu:
            selected = parts
            break
    if selected is None:
        raise RuntimeError(f"GPU {gpu} is not visible")
    memory_mib = float(selected[1])
    utilization = float(selected[2])
    if memory_mib > 64 or utilization > 1:
        raise RuntimeError(f"GPU {gpu} is not idle: memory={memory_mib} MiB utilization={utilization}%")


def _execute(command: list[str], *, cwd: Path, gpu: int, log_path: Path, command_path: Path, append: bool = False) -> None:
    encoded = command_text(command)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    if command_path.is_file() and command_path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"command drift detected: {command_path}")
    if not command_path.is_file():
        command_path.write_text(encoded, encoding="utf-8")
    _gpu_is_idle(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(cwd / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    mode = "a" if append else "w"
    print("+", command_text(command).rstrip(), flush=True)
    with log_path.open(mode, encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"command failed with exit code {returncode}; see {log_path}")


def _runtime_args(args) -> list[str]:
    return [
        "--dataset-root", str(args.dataset_root),
        "--model-dir", str(args.model_dir),
        "--lq-source", str(args.lq_source),
        "--lq-checkpoint", str(args.lq_checkpoint),
        "--bridge-checkpoint", str(args.bridge_checkpoint),
    ]


def train_command(args, config: Path, output: Path, seed: int, resume: Path | None = None) -> list[str]:
    command = [args.python, str(args.repo_root / "scripts" / "stage3_experiment.py"), "train", "--config", str(config), *_runtime_args(args), "--output-dir", str(output), "--seed", str(seed)]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def eval_command(args, config: Path, checkpoint: Path, output: Path, manifest: Path) -> list[str]:
    return [
        args.python, str(args.repo_root / "scripts" / "stage3_experiment.py"), "seen-eval",
        "--config", str(config), "--checkpoint", str(checkpoint), *_runtime_args(args),
        "--seen-manifest", str(manifest), "--subset", "probe", "--inference-seeds", str(INFERENCE_SEED),
        "--modes", *MODES, "--save-diagnostics", "--output-dir", str(output),
    ]


def _resume_checkpoint(train_dir: Path, steps: int = 1000) -> Path | None:
    if not train_dir.exists() or not any(train_dir.iterdir()):
        return None
    candidates = []
    for path in train_dir.glob("stage3_step_*.pt"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            value = int(match.group(1))
            if 0 < value < steps:
                candidates.append((value, path))
    if not candidates:
        raise RuntimeError(f"incomplete training directory has no resumable checkpoint: {train_dir}")
    value, checkpoint = max(candidates)
    rows = read_jsonl(train_dir / "train_steps.jsonl")
    if [row.get("step") for row in rows] != list(range(1, value + 1)):
        raise RuntimeError(f"training log does not match resumable checkpoint: {train_dir}")
    return checkpoint


def run_train(args, *, cell: str, seed: int, config: Path) -> Path:
    output = args.campaign_root / "train" / cell / f"seed{seed}"
    if train_complete(output):
        print(f"SKIP complete training {cell}/seed{seed}", flush=True)
        return output
    resume = _resume_checkpoint(output)
    command = train_command(args, config, output, seed, resume)
    suffix = "resume" if resume is not None else "initial"
    _execute(command, cwd=args.repo_root, gpu=args.gpu,
             log_path=args.campaign_root / "logs" / f"{cell}_seed{seed}_train_{suffix}.log",
             command_path=args.campaign_root / "control" / f"{cell}_seed{seed}_train_{suffix}.command.txt",
             append=resume is not None)
    if not train_complete(output):
        raise RuntimeError(f"training output is incomplete: {output}")
    return output


def run_eval(args, *, cell: str, seed: int, config: Path, manifest: Path, train_dir: Path) -> Path:
    output = args.campaign_root / "eval" / cell / f"seed{seed}"
    if eval_complete(output):
        print(f"SKIP complete evaluation {cell}/seed{seed}", flush=True)
        return output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to reuse incomplete evaluation output: {output}")
    checkpoint = train_dir / "stage3_step_1000.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    command = eval_command(args, config, checkpoint, output, manifest)
    _execute(command, cwd=args.repo_root, gpu=args.gpu,
             log_path=args.campaign_root / "logs" / f"{cell}_seed{seed}_eval.log",
             command_path=args.campaign_root / "control" / f"{cell}_seed{seed}_eval.command.txt")
    if not eval_complete(output):
        raise RuntimeError(f"evaluation output is incomplete: {output}")
    validate_camera_diagnostics(output)
    return output


def cell_result(cell: str, seed: int, train_dir: Path, eval_dir: Path,
                baseline_eval: Path | None = None, *, config_path: Path,
                bridge_checkpoint: Path) -> dict:
    status = integrity(
        train_dir,
        eval_dir,
        config_path=config_path,
        seed=seed,
        bridge_checkpoint=bridge_checkpoint,
    )
    local = paired_deltas(read_jsonl(eval_dir / "evaluation_rows.jsonl")) if status["evaluation"] else None
    baseline = None
    if baseline_eval is not None:
        if not eval_complete(baseline_eval):
            raise RuntimeError(f"global-bias baseline is incomplete: {baseline_eval}")
        baseline = paired_deltas(read_jsonl(baseline_eval / "evaluation_rows.jsonl"))
    comparison = None
    if local is not None and baseline is not None:
        comparison = {
            metric: local["means"]["correct"][metric] - baseline["means"]["correct"][metric]
            for metric in METRICS
        }
    return {"cell": cell, "seed": seed, "train_dir": str(train_dir), "eval_dir": str(eval_dir),
            "integrity": status, "local": local, "global": baseline, "correct_local_minus_global": comparison}


def pilot_gate(result: dict) -> dict:
    checks = {}
    status = result["integrity"]
    checks["integrity"] = bool(status["pass"])
    local = result.get("local")
    baseline = result.get("global")
    if local is None or baseline is None:
        return {"pass": False, "checks": {**checks, "analysis_available": False}, "thresholds": PILOT_THRESHOLDS}
    checks["analysis_available"] = True
    comparison = result["correct_local_minus_global"]
    checks["correct_psnr_nonregression"] = comparison["psnr"] >= PILOT_THRESHOLDS["correct_psnr_floor"]
    checks["correct_ssim_nonregression"] = comparison["ssim"] >= PILOT_THRESHOLDS["correct_ssim_floor"]
    checks["correct_lpips_nonregression"] = comparison["lpips"] <= PILOT_THRESHOLDS["correct_lpips_ceiling"]
    checks["correct_mae_nonregression"] = comparison["mae"] <= PILOT_THRESHOLDS["correct_mae_ceiling"]
    target = local["deltas"]["target_drop"]
    fusion = local["deltas"]["shuffle_fusion"]
    global_fusion = baseline["deltas"]["shuffle_fusion"]
    checks["target_drop_psnr"] = target["psnr"] >= PILOT_THRESHOLDS["target_drop_psnr_floor"]
    checks["target_drop_ssim"] = target["ssim"] >= PILOT_THRESHOLDS["target_drop_ssim_floor"]
    checks["shuffle_fusion_psnr"] = fusion["psnr"] >= PILOT_THRESHOLDS["shuffle_fusion_psnr_floor"]
    checks["shuffle_fusion_ssim"] = fusion["ssim"] >= PILOT_THRESHOLDS["shuffle_fusion_ssim_floor"]
    checks["shuffle_fusion_psnr_improvement"] = fusion["psnr"] - global_fusion["psnr"] >= PILOT_THRESHOLDS["shuffle_fusion_psnr_improvement"]
    checks["shuffle_fusion_ssim_improvement"] = fusion["ssim"] - global_fusion["ssim"] >= PILOT_THRESHOLDS["shuffle_fusion_ssim_improvement"]
    checks["shuffle_fusion_lpips_direction"] = fusion["lpips"] >= 0
    checks["shuffle_fusion_mae_direction"] = fusion["mae"] >= 0
    positive_probes = [
        item for item in local["per_probe"]["shuffle_fusion"]
        if item["psnr"] > 0 and item["ssim"] > 0
    ]
    checks["shuffle_fusion_probe_consistency"] = len(positive_probes) >= 3
    checks["repeat_jitter"] = all(value <= 1e-7 for value in local["repeat_jitter"].values())
    return {"pass": all(checks.values()), "checks": checks, "thresholds": PILOT_THRESHOLDS}


def write_report(args, results: list[dict], gate: dict) -> None:
    analysis = args.campaign_root / "analysis"
    csv_rows = []
    for result in results:
        cell = result["cell"]
        local = result.get("local")
        baseline = result.get("global")
        if local is None:
            continue
        for condition, values in (("correct", local["means"]["correct"]), *local["deltas"].items()):
            baseline_values = None
            if baseline is not None:
                baseline_values = baseline["means"]["correct"] if condition == "correct" else baseline["deltas"].get(condition)
            for metric in METRICS:
                csv_rows.append({
                    "cell": cell,
                    "seed": result["seed"],
                    "condition": condition,
                    "metric": metric,
                    "local_band_mean": values[metric],
                    "global_bias_mean": None if baseline_values is None else baseline_values[metric],
                    "local_minus_global": None if baseline_values is None else values[metric] - baseline_values[metric],
                })
    write_csv(analysis / "local_band_vs_global.csv", csv_rows)
    summary = {
        "scope": "stage3_1_local_band_sufficiency",
        "inference_seed": INFERENCE_SEED,
        "probe_ids": list(PROBE_IDS),
        "modes": list(MODES),
        "results": results,
        "pilot_gate": gate,
    }
    write_json(analysis / "local_band_summary.json", summary)
    lines = [
        "# local_band 3DSR sufficiency",
        "",
        "This is a Stage 3.1 seen-view gate before Stage 4 large-scale supervised 3DSR; 4DSR remains a separate Stage 5 route.",
        "It does not claim temporal consistency, held-out generalization, NVS or 3DGS.",
        "",
    ]
    for result in results:
        lines.extend([f"## `{result['cell']}/seed{result['seed']}`", ""])
        lines.append(f"- integrity: `{json.dumps(result['integrity'], sort_keys=True)}`")
        local = result.get("local")
        if local is None:
            lines.append("- analysis: unavailable")
            continue
        lines.extend(["", "| condition | PSNR | SSIM | LPIPS | MAE |", "|---|---:|---:|---:|---:|"])
        for condition, values in local["deltas"].items():
            lines.append(f"| `{condition}` | {values['psnr']:.6f} | {values['ssim']:.6f} | {values['lpips']:.6f} | {values['mae']:.6f} |")
        lines.append(f"| `correct_repeat jitter max` | {local['repeat_jitter']['psnr']:.6g} | {local['repeat_jitter']['ssim']:.6g} | {local['repeat_jitter']['lpips']:.6g} | {local['repeat_jitter']['mae']:.6g} |")
        if result.get("correct_local_minus_global") is not None:
            values = result["correct_local_minus_global"]
            lines.append("")
            lines.append(f"- correct local-band minus global-bias: `{json.dumps(values, sort_keys=True)}`")
    lines.extend(["", "## Pilot gate", "", f"- verdict: **{'EXPAND' if gate['pass'] else 'HOLD'}**", ""])
    for name, value in gate["checks"].items():
        lines.append(f"- `{name}`: `{value}`")
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "local_band_sufficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(args) -> tuple[Path, Path, Path, Path]:
    args.campaign_root.mkdir(parents=True, exist_ok=True)
    local_chair = args.repo_root / "configs" / "stage3_1" / "A3_local_band_chair.json"
    global_chair = args.repo_root / "configs" / "stage3_1" / "A3_target_drop_chair.json"
    local_payload = validate_local_config(local_chair, train_scenes=("chair",))
    compare_common_config(local_payload, read_json(global_chair))
    if not args.chair_manifest.is_file() or not args.global_chair_eval.is_dir():
        raise FileNotFoundError("frozen chair manifest or global-bias evaluation is missing")
    copy_or_verify(local_chair, args.campaign_root / "configs" / local_chair.name)
    protocol = {
        "scope": "stage3_1_local_band_sufficiency",
        "training_seed_pilot": 42,
        "inference_seed": INFERENCE_SEED,
        "probe_ids": list(PROBE_IDS),
        "modes": list(MODES),
        "local_config": str(local_chair),
        "local_config_sha256": sha256_file(local_chair),
        "chair_manifest": str(args.chair_manifest),
        "chair_manifest_sha256": sha256_file(args.chair_manifest),
        "global_chair_eval": str(args.global_chair_eval),
    }
    protocol_path = args.campaign_root / "protocol.json"
    if protocol_path.is_file() and read_json(protocol_path) != protocol:
        raise RuntimeError(f"protocol drift detected: {protocol_path}")
    if not protocol_path.is_file():
        write_json(protocol_path, protocol)
    return local_chair, args.chair_manifest, args.global_chair_eval, args.campaign_root / "train" / "chair" / "seed42"


def run_pilot(args) -> tuple[dict, dict]:
    local_chair, chair_manifest, global_eval, train_dir = prepare(args)
    train_dir = run_train(args, cell="chair", seed=42, config=local_chair)
    eval_dir = run_eval(args, cell="chair", seed=42, config=local_chair, manifest=chair_manifest, train_dir=train_dir)
    result = cell_result(
        "chair", 42, train_dir, eval_dir, global_eval,
        config_path=local_chair, bridge_checkpoint=args.bridge_checkpoint,
    )
    gate = pilot_gate(result)
    write_report(args, [result], gate)
    return result, gate


def make_full_config(args) -> Path:
    template = args.repo_root / "configs" / "stage3_1" / "A3_target_drop_full_1000.json"
    payload = read_json(template)
    payload["epipolar_attention"] = "local_band"
    payload["epipolar_band"] = 1.5
    output = args.campaign_root / "configs" / "A3_local_band_full.json"
    if output.is_file() and read_json(output) != payload:
        raise RuntimeError(f"full local-band config drift: {output}")
    if not output.is_file():
        write_json(output, payload)
    validate_local_config(output, train_scenes=("chair", "lego", "drums", "hotdog", "mic"))
    compare_common_config(payload, read_json(template))
    return output


def run_expansion(args, pilot_result: dict) -> None:
    if not pilot_gate(pilot_result)["pass"]:
        raise RuntimeError("refusing expansion because pilot gate is not PASS")
    local_chair = args.repo_root / "configs" / "stage3_1" / "A3_local_band_chair.json"
    full_config = make_full_config(args)
    chair_manifest = args.chair_manifest
    five_manifest = args.five_manifest
    results = [pilot_result]
    chair43_train = run_train(args, cell="chair", seed=43, config=local_chair)
    chair43_eval = run_eval(args, cell="chair", seed=43, config=local_chair, manifest=chair_manifest, train_dir=chair43_train)
    results.append(cell_result(
        "chair", 43, chair43_train, chair43_eval, None,
        config_path=local_chair, bridge_checkpoint=args.bridge_checkpoint,
    ))
    five42_train = run_train(args, cell="five", seed=42, config=full_config)
    five42_eval = run_eval(args, cell="five", seed=42, config=full_config, manifest=five_manifest, train_dir=five42_train)
    results.append(cell_result(
        "five", 42, five42_train, five42_eval, args.global_five_eval,
        config_path=full_config, bridge_checkpoint=args.bridge_checkpoint,
    ))
    five43_train = run_train(args, cell="five", seed=43, config=full_config)
    five43_eval = run_eval(args, cell="five", seed=43, config=full_config, manifest=five_manifest, train_dir=five43_train)
    results.append(cell_result(
        "five", 43, five43_train, five43_eval, None,
        config_path=full_config, bridge_checkpoint=args.bridge_checkpoint,
    ))
    final_checks = {result["cell"] + "_seed" + str(result["seed"]): result["integrity"]["pass"] for result in results}
    final_gate = {"pass": all(final_checks.values()), "checks": final_checks, "thresholds": PILOT_THRESHOLDS}
    write_report(args, results, final_gate)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "pilot", "expand", "analyze"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lq-source", type=Path, required=True)
    parser.add_argument("--lq-checkpoint", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--chair-manifest", type=Path, required=True)
    parser.add_argument("--five-manifest", type=Path, required=True)
    parser.add_argument("--global-chair-eval", type=Path, required=True)
    parser.add_argument("--global-five-eval", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.campaign_root = args.campaign_root.resolve()
    for name in ("dataset_root", "model_dir", "lq_source", "lq_checkpoint", "bridge_checkpoint", "chair_manifest", "five_manifest", "global_chair_eval", "global_five_eval"):
        value = getattr(args, name)
        if not value.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {value}")
    if args.command in {"run", "pilot"}:
        result, gate = run_pilot(args)
        print(json.dumps({"pilot": result, "gate": gate}, indent=2, ensure_ascii=False), flush=True)
        if args.command == "run" and gate["pass"]:
            run_expansion(args, result)
        return
    if args.command == "expand":
        summary = read_json(args.campaign_root / "analysis" / "local_band_summary.json")
        pilot = next((item for item in summary.get("results", []) if item.get("cell") == "chair" and item.get("seed") == 42), None)
        if pilot is None:
            raise RuntimeError("pilot result is missing; run pilot first")
        run_expansion(args, pilot)
        return
    if args.command == "analyze":
        summary = read_json(args.campaign_root / "analysis" / "local_band_summary.json")
        print(json.dumps(summary.get("pilot_gate", {}), indent=2), flush=True)
        return


if __name__ == "__main__":
    main()
