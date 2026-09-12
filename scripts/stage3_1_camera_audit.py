"""Run the no-training Stage 3.1 camera-scope audit.

This driver only evaluates frozen Stage 3.1 checkpoints. It deliberately
keeps the four original cells, manifest, probe set, inference seed and model
assets unchanged while separating LR-fusion and Wan-geometry camera scopes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
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
CELL_CONFIGS = {
    "chair_1000": ("A3_target_drop_chair.json", 1000, "chair_probe.json"),
    "chair_2000": ("A3_target_drop_chair_2000.json", 2000, "chair_probe.json"),
    "five_1000": ("A3_target_drop_full_1000.json", 1000, "five_probe.json"),
    "five_2000": ("A3_target_drop_full.json", 2000, "five_probe.json"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cell_dir(campaign: Path, cell: str) -> Path:
    return campaign / "eval" / cell


def command(args, cell: str) -> list[str]:
    config, steps, manifest = CELL_CONFIGS[cell]
    checkpoint = args.previous_campaign / "train" / cell / "seed42" / f"stage3_step_{steps:04d}.pt"
    return [
        args.python,
        str(args.repo_root / "scripts" / "stage3_experiment.py"),
        "seen-eval",
        "--config",
        str(args.repo_root / "configs" / "stage3_1" / config),
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
        str(args.previous_campaign / "manifest" / manifest),
        "--subset",
        "probe",
        "--inference-seeds",
        str(INFERENCE_SEED),
        "--modes",
        *MODES,
        "--save-diagnostics",
        "--output-dir",
        str(cell_dir(args.campaign_root, cell)),
    ]


def expected_complete(path: Path) -> bool:
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
        summary.get("rows") == len(PROBE_IDS) * len(MODES)
        and summary.get("groups") == len(PROBE_IDS)
        and summary.get("inference_seeds") == [INFERENCE_SEED]
        and tuple(summary.get("conditions", ())) == MODES
        and len(rows) == len(PROBE_IDS) * len(MODES)
        and len(baseline) == len(PROBE_IDS) * 2
        and len(diagnostics) == len(PROBE_IDS) * (len(MODES) - 1)
    )


def run_cell(args, cell: str) -> None:
    output = cell_dir(args.campaign_root, cell)
    if expected_complete(output):
        print(f"SKIP complete camera audit {cell}", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to reuse incomplete audit output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command_line = command(args, cell)
    command_path = args.campaign_root / "control" / f"{cell}.command.txt"
    command_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = " ".join(command_line) + "\n"
    if command_path.is_file() and command_path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"command drift detected: {command_path}")
    command_path.write_text(encoded, encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = str(args.repo_root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    log_path = args.campaign_root / "logs" / f"{cell}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command_line), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command_line,
            cwd=args.repo_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if process.wait():
            raise RuntimeError(f"camera audit failed for {cell}; see {log_path}")
    if not expected_complete(output):
        raise RuntimeError(f"camera audit output is incomplete: {output}")


def _indexed(rows: list[dict]) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = (row.get("condition"), row.get("inference_seed"), row.get("group_id"), row.get("view_index"))
        if key in result:
            raise ValueError(f"duplicate evaluation identity: {key}")
        result[key] = row
    return result


def analyze_cell(path: Path, cell: str) -> tuple[list[dict], dict]:
    rows = read_jsonl(path / "evaluation_rows.jsonl")
    diagnostics = read_jsonl(path / "diagnostics.jsonl")
    values = _indexed(rows)
    identities = sorted({key[1:] for key in values if key[0] == "correct"})
    jitter = {
        metric: max(
            abs(float(values[("correct", *identity)][metric]) - float(values[("correct_repeat", *identity)][metric]))
            for identity in identities
        )
        for metric in METRICS
    }
    delta_rows = []
    for condition in MODES:
        if condition in {"correct", "correct_repeat"}:
            continue
        paired = []
        for identity in identities:
            correct = values[("correct", *identity)]
            changed = values[(condition, *identity)]
            paired.append({
                metric: (float(correct[metric]) - float(changed[metric])) * GOOD_DIRECTION[metric]
                for metric in METRICS
            })
        delta_rows.append({"cell": cell, "condition": condition, "probe_count": len(paired), **{
            f"delta_{metric}": mean(item[metric] for item in paired) for metric in METRICS
        }, **{f"repeat_jitter_{metric}": jitter[metric] for metric in METRICS}})
    diagnostic_summary = {}
    for condition in MODES:
        if condition == "correct":
            continue
        selected = [row for row in diagnostics if row.get("condition") == condition]
        if not selected:
            continue
        diagnostic_summary[condition] = {
            "rows": len(selected),
            "prepared_feature_delta_mean": mean(float(row["prepared_feature_delta"]) for row in selected),
            "prepared_feature_relative_delta_mean": mean(float(row["feature_relative_delta"]) for row in selected),
            "velocity_delta_relative_mean": mean(float(row["per_step_velocity_delta_relative_mean"]) for row in selected),
            "fusion_camera_changed_rows": sum(bool(row["fusion_camera_changed"]) for row in selected),
            "geometry_camera_changed_rows": sum(bool(row["geometry_camera_changed"]) for row in selected),
        }
    return delta_rows, {"cell": cell, "repeat_jitter": jitter, "diagnostics": diagnostic_summary}


def analyze(args) -> None:
    all_rows = []
    details = {"scope": "stage3_1_camera_scope_audit", "inference_seed": INFERENCE_SEED, "modes": list(MODES), "cells": {}}
    for cell in CELL_CONFIGS:
        rows, summary = analyze_cell(cell_dir(args.campaign_root, cell), cell)
        all_rows.extend(rows)
        details["cells"][cell] = summary
    write_json(args.campaign_root / "analysis" / "camera_scope_deltas.json", details)
    write_csv(args.campaign_root / "analysis" / "camera_scope_deltas.csv", all_rows)
    lines = [
        "# Stage 3.1 Camera Scope Audit",
        "",
        "This report reuses frozen checkpoints and only changes intervention scope.",
        "It does not claim held-out generalization, NVS, 3DGS or formal Stage 4 effectiveness.",
        "",
    ]
    for cell, summary in details["cells"].items():
        lines.extend([f"## `{cell}`", ""])
        lines.append("| condition | PSNR delta | SSIM delta | LPIPS delta | MAE delta |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in [item for item in all_rows if item["cell"] == cell]:
            lines.append("| `{condition}` | {delta_psnr:.6f} | {delta_ssim:.6f} | {delta_lpips:.6f} | {delta_mae:.6f} |".format(**row))
        lines.append("")
        lines.append(f"- repeat jitter: `{json.dumps(summary['repeat_jitter'], sort_keys=True)}`")
        lines.append(f"- diagnostics: `{json.dumps(summary['diagnostics'], sort_keys=True)}`")
        lines.append("")
    args.campaign_root.joinpath("analysis").mkdir(parents=True, exist_ok=True)
    (args.campaign_root / "analysis" / "camera_scope_deltas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--previous-campaign", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--lq-source", type=Path)
    parser.add_argument("--lq-checkpoint", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path)
    parser.add_argument("--python", default="python")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cell", choices=tuple(CELL_CONFIGS))
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.campaign_root = args.campaign_root.resolve()
    args.previous_campaign = args.previous_campaign.resolve()
    if args.command == "analyze":
        analyze(args)
        return
    required = (args.dataset_root, args.model_dir, args.lq_source, args.lq_checkpoint, args.bridge_checkpoint)
    if any(value is None for value in required):
        parser.error("run requires dataset/model/LQ/bridge paths")
    cells = (args.cell,) if args.cell else tuple(CELL_CONFIGS)
    for cell in cells:
        run_cell(args, cell)


if __name__ == "__main__":
    main()
