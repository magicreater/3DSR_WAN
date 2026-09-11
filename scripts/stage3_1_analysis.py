#!/usr/bin/env python3
"""Analyze the bounded Stage 3.1 target-view dropout campaign."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = ("psnr", "ssim", "lpips", "mae")
GOOD_SIGN = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0, "mae": -1.0}
CONDITIONS = {"correct", "correct_repeat", "target_drop", "shuffle_camera"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_metrics(row: dict) -> None:
    for metric in METRICS:
        value = row.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"missing or nonfinite {metric}: {row}")


def row_key(row: dict, include_condition: bool = False) -> tuple:
    key = (row.get("inference_seed"), row.get("group_id"), row.get("view_index"))
    return (row.get("condition"), *key) if include_condition else key


def indexed(rows: list[dict], include_condition: bool = False) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        validate_metrics(row)
        key = row_key(row, include_condition)
        if key in result:
            raise ValueError(f"duplicate evaluation identity: {key}")
        result[key] = row
    return result


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty rows")
    return {metric: statistics.mean(float(row[metric]) for row in rows) for metric in METRICS}


def condition_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    return {condition: mean_metrics(values) for condition, values in sorted(grouped.items())}


def paired_gain(rows: list[dict], comparator: str) -> dict[str, float]:
    values = indexed(rows, include_condition=True)
    identities = sorted({key[1:] for key in values if key[0] == "correct"})
    paired = []
    for identity in identities:
        correct = values[("correct", *identity)]
        reference = values[(comparator, *identity)]
        paired.append({
            metric: (float(correct[metric]) - float(reference[metric])) * GOOD_SIGN[metric]
            for metric in METRICS
        })
    return {metric: statistics.mean(row[metric] for row in paired) for metric in METRICS}


def repeat_jitter(rows: list[dict]) -> dict[str, float]:
    values = indexed(rows, include_condition=True)
    identities = sorted({key[1:] for key in values if key[0] == "correct"})
    return {
        metric: max(
            abs(float(values[("correct", *identity)][metric]) - float(values[("correct_repeat", *identity)][metric]))
            for identity in identities
        )
        for metric in METRICS
    }


def directional_pass(gain: dict[str, float], noise: dict[str, float]) -> bool:
    return (
        gain["psnr"] > noise["psnr"]
        and gain["ssim"] >= -noise["ssim"]
        and gain["lpips"] >= -noise["lpips"]
        and gain["mae"] >= -noise["mae"]
    )


def load_run(train_path: Path) -> dict:
    run_dir = train_path.parent
    evaluation_dir = run_dir / "eval"
    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(evaluation_dir / "evaluation_rows.jsonl")
    baseline_rows = read_jsonl(evaluation_dir / "baseline_rows.jsonl")
    conditions = {str(row.get("condition")) for row in eval_rows}
    unknown = conditions - CONDITIONS
    if unknown:
        raise ValueError(f"unknown Stage 3.1 conditions in {evaluation_dir}: {sorted(unknown)}")
    result = {
        "label": str(run_dir),
        "train": {
            "logged_steps": len(train_rows),
            "last_step": train_rows[-1].get("step") if train_rows else None,
            "loss_first": train_rows[0].get("loss") if train_rows else None,
            "loss_last": train_rows[-1].get("loss") if train_rows else None,
            "loss_mean": statistics.mean(float(row["loss"]) for row in train_rows) if train_rows else None,
            "fusion_gradient_mean": statistics.mean(
                float(row["fusion_gradient_norm"])
                for row in train_rows
                if isinstance(row.get("fusion_gradient_norm"), (int, float))
            ) if any(isinstance(row.get("fusion_gradient_norm"), (int, float)) for row in train_rows) else None,
            "target_lr_drop_count": sum(int(row.get("target_lr_drop_count", 0)) for row in train_rows),
            "target_lr_drop_microsteps": sum(len(row.get("target_lr_dropped", [])) for row in train_rows),
        },
        "evaluation": {
            "rows": len(eval_rows),
            "baseline_rows": len(baseline_rows),
            "conditions": sorted(conditions),
            "means": condition_means(eval_rows),
        },
    }
    if {"correct", "correct_repeat"} <= conditions:
        result["evaluation"]["repeat_jitter_max"] = repeat_jitter(eval_rows)
    noise = result["evaluation"].get("repeat_jitter_max", {metric: 0.0 for metric in METRICS})
    for comparator in ("target_drop", "shuffle_camera"):
        if comparator in conditions:
            result["evaluation"][comparator + "_gain"] = paired_gain(eval_rows, comparator)
            result["evaluation"][comparator + "_pass"] = directional_pass(
                result["evaluation"][comparator + "_gain"], noise
            )
    baseline = {(row.get("group_id"), row.get("view_index")): row for row in baseline_rows if row.get("condition") == "bicubic"}
    correct = [row for row in eval_rows if row.get("condition") == "correct"]
    if baseline and correct:
        gains = []
        for row in correct:
            reference = baseline[(row.get("group_id"), row.get("view_index"))]
            gains.append({metric: (float(row[metric]) - float(reference[metric])) * GOOD_SIGN[metric] for metric in METRICS})
        result["evaluation"]["sr_fit_gain"] = {
            metric: statistics.mean(row[metric] for row in gains) for metric in METRICS
        }
        result["evaluation"]["sr_fit_pass"] = all(
            value > 0 for value in result["evaluation"]["sr_fit_gain"].values()
        )
    else:
        result["evaluation"]["sr_fit_pass"] = False
    diagnostics = evaluation_dir / "diagnostics.jsonl"
    if diagnostics.is_file():
        result["diagnostics"] = read_jsonl(diagnostics)
    return result


def discover_runs(root: Path) -> list[dict]:
    train_paths = sorted(root.rglob("train_steps.jsonl"))
    if not train_paths:
        raise FileNotFoundError(f"no Stage 3.1 training logs under {root}")
    return [load_run(path) for path in train_paths]


def pilot_pass(runs: list[dict]) -> bool | None:
    pilot = [run for run in runs if "/pilot/" in run["label"].replace("\\", "/")]
    dropout = next((run for run in pilot if "target_drop" in run["label"]), None)
    base = next((run for run in pilot if "base" in run["label"]), None)
    if dropout is None or base is None:
        return None
    drop_eval = dropout["evaluation"]
    base_eval = base["evaluation"]
    return bool(
        drop_eval.get("sr_fit_pass")
        and drop_eval.get("target_drop_pass")
        and drop_eval.get("shuffle_camera_pass")
        and not (base_eval.get("target_drop_pass") and base_eval.get("shuffle_camera_pass"))
    )


def phase2_pass(runs: list[dict]) -> bool | None:
    phase2 = [run for run in runs if "/phase2/" in run["label"].replace("\\", "/")]
    if not phase2:
        return None
    required = ("sr_fit_pass", "target_drop_pass", "shuffle_camera_pass")
    return all(all(run["evaluation"].get(key, False) for key in required) for run in phase2)


def plot(output: Path, runs: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for run in runs:
        path = Path(run["label"]) / "train_steps.jsonl"
        rows = read_jsonl(path)
        axes[0].plot([row["step"] for row in rows], [row["loss"] for row in rows], label=run["label"])
        axes[1].plot(
            [row["step"] for row in rows],
            [row.get("fusion_gradient_norm", float("nan")) for row in rows],
            label=run["label"],
        )
    axes[0].set_title("Training loss")
    axes[1].set_title("Fusion gradient norm")
    for axis in axes:
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)

    rows = []
    for run in runs:
        evaluation = run["evaluation"]
        for condition in ("target_drop", "shuffle_camera"):
            gain = evaluation.get(condition + "_gain")
            if gain:
                rows.append((run["label"], condition, gain["psnr"]))
    if rows:
        fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
        axis.bar([f"{label}\n{condition}" for label, condition, _ in rows], [value for _, _, value in rows])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("good-direction PSNR delta (dB)")
        axis.set_title("Stage 3.1 causal intervention deltas")
        fig.savefig(output / "causal_deltas.png", dpi=160)
        plt.close(fig)


def report(output: Path, runs: list[dict], pilot: bool | None, phase2: bool | None) -> None:
    overall = phase2 if phase2 is not None else pilot
    lines = [
        "# Stage 3.1 目标视图 Dropout 修复验证",
        "",
        f"- `PILOT_PASS`: **{pilot}**",
        f"- `PHASE2_PASS`: **{phase2}**",
        f"- `STAGE3_1_PASS`: **{overall}**",
        "",
        "判定范围仅包含 seen-train SR 的目标视图依赖与相机干预因果性，不包含 held-out generalization、NVS、3DGS 或正式大规模训练有效性。",
        "",
        "## Runs",
        "",
    ]
    for run in runs:
        evaluation = run["evaluation"]
        lines.extend([
            f"### `{run['label']}`",
            "",
            f"- steps: `{run['train']['last_step']}`; loss: `{run['train']['loss_first']}` -> `{run['train']['loss_last']}`",
            f"- conditions: `{', '.join(evaluation['conditions'])}`",
            f"- SR fit: `{evaluation.get('sr_fit_pass')}`",
            f"- target-drop pass: `{evaluation.get('target_drop_pass')}`",
            f"- shuffle-camera pass: `{evaluation.get('shuffle_camera_pass')}`",
            "",
        ])
    (output / "stage3_1_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.campaign_root.resolve())
    normalized = []
    for run in runs:
        run["label"] = str(Path(run["label"]).resolve())
        normalized.append(run)
    runs = normalized
    pilot = pilot_pass(runs)
    phase2 = phase2_pass(runs)
    verdict = {
        "scope": "seen_train_sr_stage3_1",
        "pilot_pass": pilot,
        "phase2_pass": phase2,
        "stage3_1_pass": phase2 if phase2 is not None else pilot,
        "claim_limit": "No held-out generalization, NVS, 3DGS or formal large-scale effectiveness claim.",
        "runs": runs,
    }
    write_json(output / "stage3_1_verdict.json", verdict)
    summary_rows = []
    for run in runs:
        evaluation = run["evaluation"]
        summary_rows.append({
            "label": run["label"],
            "last_step": run["train"]["last_step"],
            "loss_last": run["train"]["loss_last"],
            "fusion_gradient_mean": run["train"]["fusion_gradient_mean"],
            "target_lr_drop_count": run["train"]["target_lr_drop_count"],
            "target_drop_psnr_gain": evaluation.get("target_drop_gain", {}).get("psnr"),
            "shuffle_camera_psnr_gain": evaluation.get("shuffle_camera_gain", {}).get("psnr"),
            "sr_fit_pass": evaluation.get("sr_fit_pass"),
            "target_drop_pass": evaluation.get("target_drop_pass"),
            "shuffle_camera_pass": evaluation.get("shuffle_camera_pass"),
        })
    write_csv(output / "summary_rows.csv", summary_rows)
    plot(output, runs)
    report(output, runs, pilot, phase2)


if __name__ == "__main__":
    main()
