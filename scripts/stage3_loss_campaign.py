#!/usr/bin/env python3
"""Run the bounded Stage 3 loss comparison on two otherwise idle RTX 4090s."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import stage3_experiment as st
from rl3dsr.validation.stage3_protocol import load_stage3_config

import stage3_loss_experiment as loss_experiment


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage3_loss_conditional_20260926"
PYTHON = Path(sys.executable)
GPU_IDS = (1, 3)
GPU_UUIDS = {
    1: "GPU-e2eaf00a-e9b0-b681-d6f0-c9597f5c418d",
    3: "GPU-52673cef-d39f-fbe8-bdc0-bfda70852e4b",
}
SCREEN = ("flow", "recover", "rgb", "both")
FULL = ("full_core", "full_ordinary", "full_challenge")
CONTROLS = ("geometry_off", "fusion_off")
SCREEN_MODES = ("correct",)
FULL_MODES = (
    "correct", "correct_repeat", "remove", "target_drop",
    "shuffle_fusion", "target_drop_shuffle_fusion", "mispaired_lr",
    "mispaired_camera", "shuffle_geometry",
)


def sha256(path: Path) -> str:
    return loss_experiment.sha256(path)


def frozen(path: Path, payload: object) -> None:
    loss_experiment.frozen_json(path, payload)


def gpu_free(index: int) -> None:
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = [line.strip().split(", ") for line in query.splitlines()]
    matching = [row for row in rows if int(row[0]) == index]
    if len(matching) != 1 or matching[0][1] != GPU_UUIDS[index]:
        raise RuntimeError(f"selected GPU {index} identity changed")
    if int(matching[0][2]) > 100:
        raise RuntimeError(f"selected GPU {index} is occupied")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ], text=True)
    if GPU_UUIDS[index] in apps:
        raise RuntimeError(f"selected GPU {index} has a compute process")


def runtime_args() -> list[str]:
    return [
        "--dataset-root", str(loss_experiment.RUNTIME_ROOT / "datasets/nerf_synthetic"),
        "--model-dir", str(loss_experiment.RUNTIME_ROOT / "models/Wan2.1-T2V-1.3B"),
        "--lq-source", str(loss_experiment.LQ_SOURCE),
        "--lq-checkpoint", str(loss_experiment.LQ_CHECKPOINT),
        "--bridge-checkpoint", str(loss_experiment.BRIDGE),
    ]


def command(label: str, argv: list[str], gpu: int) -> None:
    gpu_free(gpu)
    log = OUT / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT / "src")}
    with log.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(argv, cwd=ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"{label} failed ({completed.returncode}); inspect {log}")


def config_path(phase: str) -> Path:
    return ROOT / "configs" / f"stage3_loss_{'screen' if phase == 'screen' else 'full'}.json"


def manifest_path(phase: str) -> Path:
    return OUT / "manifest" / f"{phase}.json"


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for phase in ("screen", "full"):
        config = load_stage3_config(config_path(phase))
        path = manifest_path(phase)
        if not path.exists():
            st._prepare_seen_manifest(SimpleNamespace(
                dataset_root=loss_experiment.RUNTIME_ROOT / "datasets/nerf_synthetic",
                manifest=path,
            ), config)
    source = [
        ROOT / "scripts/stage3_loss_experiment.py",
        ROOT / "scripts/stage3_experiment.py",
        ROOT / "src/rl3dsr/models/wan/stage3.py",
        ROOT / "src/rl3dsr/models/wan/lr_fusion.py",
        ROOT / "src/rl3dsr/models/wan/flow.py",
        ROOT / "src/rl3dsr/models/wan/sampling.py",
    ]
    model_dir = loss_experiment.RUNTIME_ROOT / "models/Wan2.1-T2V-1.3B"
    model_files = sorted(path for path in model_dir.rglob("*") if path.is_file())
    data = loss_experiment.RUNTIME_ROOT / "datasets/nerf_synthetic/lego"
    data_files = sorted(data.glob("transforms_*.json"))
    payload = {
        "protocol": "stage3_loss_conditional_v1",
        "stage4": "HOLD",
        "gpu_uuids": GPU_UUIDS,
        "python": str(PYTHON),
        "initialization": {"checkpoint": str(loss_experiment.PARENT),
                           "sha256": sha256(loss_experiment.PARENT),
                           "reset_fusion": True},
        "calibration_sha256": sha256(OUT / "calibration.json"),
        "config_sha256": {phase: sha256(config_path(phase)) for phase in ("screen", "full")},
        "manifest_sha256": {phase: sha256(manifest_path(phase)) for phase in ("screen", "full")},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source},
        "assets_sha256": {str(path): sha256(path) for path in [
            loss_experiment.BRIDGE, loss_experiment.LQ_CHECKPOINT,
            *model_files, *data_files,
        ]},
        "screen_arms": SCREEN,
        "screen_inference_seed": 3302,
        "full_inference_seeds": [3302, 3303],
        "full_interventions": FULL_MODES,
    }
    frozen(OUT / "protocol.json", payload)
    print(json.dumps({"prepared": True, "protocol_sha256": sha256(OUT / "protocol.json")}), flush=True)


def assert_protocol() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    if protocol["initialization"]["sha256"] != sha256(loss_experiment.PARENT):
        raise RuntimeError("parent checkpoint changed")
    if protocol["calibration_sha256"] != sha256(OUT / "calibration.json"):
        raise RuntimeError("calibration changed")
    for phase in ("screen", "full"):
        if (protocol["config_sha256"][phase] != sha256(config_path(phase))
                or protocol["manifest_sha256"][phase] != sha256(manifest_path(phase))):
            raise RuntimeError(f"{phase} config or manifest changed")
    for relative, expected in protocol["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"source changed: {relative}")


def cell(name: str, phase: str, mode: str, pair: str, gpu: int) -> None:
    assert_protocol()
    config = load_stage3_config(config_path(phase))
    final = OUT / "train" / name / f"stage3_step_{config.steps:04d}.pt"
    if not final.exists():
        existing = sorted(final.parent.glob("stage3_step_*.pt"))
        tag = "initial" if not existing else f"resume_{existing[-1].stem}"
        command(f"{name}_train_{tag}", [
            str(PYTHON), str(ROOT / "scripts/stage3_loss_experiment.py"), "train",
            "--config", str(config_path(phase)),
            "--calibration", str(OUT / "calibration.json"),
            "--output", str(final.parent), "--loss-mode", mode,
            "--pair-mode", pair, "--seed", "42",
        ], gpu)
    if not final.is_file():
        raise RuntimeError(f"{name} final checkpoint absent")
    log = final.parent / "train_steps.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    if len(rows) != config.steps or [row["step"] for row in rows] != list(range(1, config.steps + 1)):
        raise RuntimeError(f"{name} training record incomplete")
    evaluation = OUT / "eval" / name
    if not (evaluation / "evaluation_summary.json").is_file():
        if evaluation.exists() and any(evaluation.iterdir()):
            raise RuntimeError(f"{name} evaluation directory incomplete")
        modes = SCREEN_MODES if phase == "screen" else FULL_MODES
        seeds = (3302,) if phase == "screen" else (3302, 3303)
        manifest = json.loads(manifest_path(phase).read_text())
        probe_ids = [row for row in manifest["subsets"]["probe"] if row.startswith("lego:")]
        command(f"{name}_eval", [
            str(PYTHON), str(ROOT / "scripts/stage3_experiment.py"), "seen-eval",
            "--config", str(config_path(phase)), *runtime_args(),
            "--checkpoint", str(final), "--seen-manifest", str(manifest_path(phase)),
            "--subset", "probe", "--group-ids", *probe_ids,
            "--inference-seeds", *map(str, seeds), "--modes", *modes,
            "--save-images", "--output-dir", str(evaluation),
        ], gpu)
    print(json.dumps({"complete": name, "gpu": gpu}), flush=True)


def run_parallel(cells: list[tuple[str, str, str, str]]) -> None:
    # A worker owns one GPU for the complete train+evaluation cell.
    queues = {GPU_IDS[0]: cells[::2], GPU_IDS[1]: cells[1::2]}

    def worker(gpu: int):
        for name, phase, mode, pair in queues[gpu]:
            cell(name, phase, mode, pair, gpu)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(worker, gpu): gpu for gpu in GPU_IDS if queues[gpu]}
        for future in as_completed(futures):
            future.result()


def metrics(name: str) -> dict[str, float]:
    path = OUT / "eval" / name / "evaluation_rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    chosen = [row for row in rows if row["condition"] == "correct"]
    if len(chosen) not in (4, 8):
        raise RuntimeError(f"{name}: expected 4 or 8 correct rows")
    return {key: sum(row[key] for row in chosen) / len(chosen)
            for key in ("psnr", "ssim", "lpips")}


def curves(names: tuple[str, ...], label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name in names:
        path = OUT / "train" / name / "train_steps.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        expected = 1000 if label == "screen" else 4000
        if len(rows) != expected:
            raise RuntimeError(f"{name}: incomplete curve source")
        steps = [int(row["step"]) for row in rows]
        axes[0].plot(steps, [float(row["mean_flow"]) for row in rows],
                     alpha=0.6, linewidth=0.8, label=name)
        axes[1].plot(steps, [float(row["gradient_norm"]) for row in rows],
                     alpha=0.6, linewidth=0.8, label=name)
    axes[0].set_ylabel("flow MSE")
    axes[1].set_ylabel("gradient norm")
    axes[1].set_xlabel("optimizer step")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    target = OUT / "analysis" / f"curves_{label}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"curve output already exists: {target}")
    figure.savefig(target, dpi=150)
    plt.close(figure)


def matched_training(names: tuple[str, ...], expected_steps: int) -> None:
    references = None
    digests = set()
    for name in names:
        root = OUT / "train" / name
        manifest = json.loads((root / "run_manifest.json").read_text())
        digests.add(manifest["initial_trainable_sha256"])
        rows = [json.loads(line) for line in (root / "train_steps.jsonl").read_text().splitlines()]
        if len(rows) != expected_steps or [row["step"] for row in rows] != list(range(1, expected_steps + 1)):
            raise RuntimeError(f"{name}: incomplete train steps")
        examples = [[(micro["scene"], micro["views"], micro["sigma"],
                      micro["target_lr_dropped"]) for micro in row["micro"]]
                    for row in rows]
        if references is None:
            references = examples
        elif examples != references:
            raise RuntimeError(f"{name}: training samples or random sequence unmatched")
    if len(digests) != 1:
        raise RuntimeError(f"initial trainable parameters differ: {digests}")


def review_screen() -> str:
    assert_protocol()
    matched_training(tuple(f"screen_{name}" for name in SCREEN), 1000)
    values = {mode: metrics(f"screen_{mode}") for mode in SCREEN}
    selected = max(SCREEN, key=lambda name: (
        values[name]["psnr"], values[name]["ssim"], -values[name]["lpips"],
        -SCREEN.index(name),
    ))
    frozen(OUT / "screen_review.json", {
        "selected_loss": selected, "metrics": values,
        "criterion": "mean correct 50-step PSNR, SSIM, then LPIPS",
    })
    if not (OUT / "analysis/curves_screen.png").exists():
        curves(tuple(f"screen_{name}" for name in SCREEN), "screen")
    print(json.dumps({"selected_loss": selected, "metrics": values}), flush=True)
    return selected


def review_full() -> str:
    """Select on correct 50-step quality; report interventions separately."""
    assert_protocol()
    matched_training(FULL, 4000)
    baseline_path = Path(
        "/data/linzizhuo/RL3DSR_WAN_REMO-stage3-final/"
        "artifacts/stage3_final_20260924/eval/a5_v4_lego_s42/evaluation_rows.jsonl"
    )
    baseline = [json.loads(line) for line in baseline_path.read_text().splitlines()]
    baseline = {row["group_id"]: row for row in baseline
                if row["condition"] == "correct" and row["inference_seed"] == 3302}
    if set(baseline) != {"lego:000", "lego:033", "lego:066", "lego:099"}:
        raise RuntimeError("matched A5 probe rows are incomplete")
    baseline_mean = {key: sum(row[key] for row in baseline.values()) / 4
                     for key in ("psnr", "ssim", "lpips")}
    results = {}
    for name in FULL:
        path = OUT / "eval" / name / "evaluation_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        indexed = {(row["group_id"], row["inference_seed"], row["condition"]): row
                   for row in rows}
        correct = [row for row in rows if row["condition"] == "correct"]
        if (len(correct) != 8 or len(indexed) != len(rows)
                or {row["group_id"] for row in correct} != set(baseline)
                or {row["inference_seed"] for row in correct} != {3302, 3303}):
            raise RuntimeError(f"{name}: invalid full evaluation rows")
        for row in correct:
            if row["inference_seed"] == 3302 and row["view_indices"] != baseline[row["group_id"]]["view_indices"]:
                raise RuntimeError(f"{name}: unmatched A5 camera context")
        means = {key: sum(row[key] for row in correct) / 8
                 for key in ("psnr", "ssim", "lpips")}
        correct_3302 = [row for row in correct if row["inference_seed"] == 3302]
        matched = {key: sum(row[key] for row in correct_3302) / 4
                   for key in ("psnr", "ssim", "lpips")}
        contrasts = {}
        for mode in FULL_MODES:
            if mode in ("correct", "correct_repeat"):
                continue
            deltas = []
            for row in correct:
                other = indexed.get((row["group_id"], row["inference_seed"], mode))
                if other is None:
                    raise RuntimeError(f"{name}: missing {mode} intervention")
                deltas.append({"group_id": row["group_id"], "seed": row["inference_seed"],
                               "psnr": row["psnr"] - other["psnr"],
                               "ssim": row["ssim"] - other["ssim"],
                               "lpips": other["lpips"] - row["lpips"]})
            contrasts[mode] = {
                "mean": {key: sum(delta[key] for delta in deltas) / 8
                         for key in ("psnr", "ssim", "lpips")},
                "positive_psnr_by_seed": {str(seed): sum(
                    delta["psnr"] > 0 for delta in deltas if delta["seed"] == seed
                ) for seed in (3302, 3303)},
            }
        results[name] = {
            "correct_mean_two_seeds": means,
            "correct_mean_seed_3302": matched,
            "matched_a5_delta": {key: matched[key] - baseline_mean[key]
                                 for key in ("psnr", "ssim", "lpips")},
            "a5_quality_gate": (matched["psnr"] - baseline_mean["psnr"] >= -0.10
                                and matched["ssim"] - baseline_mean["ssim"] >= -0.001),
            "correct_minus_intervention": contrasts,
        }
    selected = max(FULL, key=lambda name: (
        results[name]["correct_mean_two_seeds"]["psnr"],
        results[name]["correct_mean_two_seeds"]["ssim"],
        -results[name]["correct_mean_two_seeds"]["lpips"],
        -FULL.index(name),
    ))
    frozen(OUT / "full_review.json", {
        "best_candidate": selected,
        "selection": "two-seed correct 50-step PSNR, SSIM, then LPIPS",
        "baseline_a5_seed_3302": baseline_mean,
        "baseline_a5_rows_sha256": sha256(baseline_path),
        "quality_gate": "matched A5 seed3302 PSNR delta >= -0.10dB and SSIM delta >= -0.001",
        "intervention_positive_gate": "mean PSNR > 0 and >= 3/4 probes per seed",
        "arms": results,
        "stage4": "HOLD",
    })
    if not (OUT / "analysis/curves_full.png").exists():
        curves(FULL, "full")
    print(json.dumps({"selected": selected, "quality_gate": results[selected]["a5_quality_gate"]}), flush=True)
    return selected


def control_cell(control: str, gpu: int) -> None:
    if control not in CONTROLS:
        raise ValueError(control)
    checkpoint = OUT / "train" / f"control_{control}" / "stage3_step_4000.pt"
    script = ROOT / "scripts/stage3_loss_control.py"
    if not checkpoint.is_file():
        prior = sorted(checkpoint.parent.glob("stage3_step_*.pt"))
        tag = "initial" if not prior else f"resume_{prior[-1].stem}"
        command(f"control_{control}_train_{tag}", [
            str(PYTHON), str(script), "train", "--control", control,
        ], gpu)
    if not checkpoint.is_file():
        raise RuntimeError(f"control checkpoint missing: {control}")
    output = OUT / "eval" / f"control_{control}"
    if not (output / "summary.json").is_file():
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"partial control evaluation: {output}")
        command(f"control_{control}_eval", [
            str(PYTHON), str(script), "eval", "--control", control,
        ], gpu)
    print(json.dumps({"complete": f"control_{control}", "gpu": gpu}), flush=True)


def review_controls() -> None:
    review = json.loads((OUT / "full_review.json").read_text())
    selected = review["best_candidate"]
    correct = review["arms"][selected]["correct_mean_two_seeds"]
    controls = {}
    for control in CONTROLS:
        summary = json.loads((OUT / "eval" / f"control_{control}" / "summary.json").read_text())
        means = summary["means"]
        controls[control] = {
            "mean": means,
            "full_minus_retrained_off": {key: correct[key] - means[key]
                                          for key in ("psnr", "ssim", "lpips")},
        }
    frozen(OUT / "control_review.json", {
        "selected_full_arm": selected, "correct_mean_two_seeds": correct,
        "matched_training_steps": 4000, "controls": controls,
        "interpretation": "module-off retraining tests contribution after adaptation; one-checkpoint knockout tests immediate reliance",
    })
    if not (OUT / "analysis/curves_controls.png").exists():
        curves(tuple(f"control_{name}" for name in CONTROLS), "controls")


def diagnostic_cell(name: str, gpu: int) -> None:
    output = OUT / "diagnostics" / name
    if (output / "summary.json").is_file():
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"partial diagnostic output: {output}")
    checkpoint = OUT / "train" / name / "stage3_step_4000.pt"
    command(f"{name}_knockout", [
        str(PYTHON), str(ROOT / "scripts/stage3_loss_diagnostics.py"),
        "--config", str(config_path("full")),
        "--checkpoint", str(checkpoint),
        "--manifest", str(manifest_path("full")),
        "--output", str(output), "--seeds", "3302", "3303",
    ], gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "screen", "review-screen", "full", "review-full", "diagnostics", "controls", "review-controls"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    assert_protocol()
    if args.command == "screen":
        run_parallel([(f"screen_{name}", "screen", name, "none") for name in SCREEN])
    elif args.command == "review-screen":
        review_screen()
    elif args.command == "review-full":
        review_full()
    elif args.command == "diagnostics":
        selected = json.loads((OUT / "full_review.json").read_text())["best_candidate"]
        names = (selected,) if selected == "full_core" else (selected, "full_core")
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = [pool.submit(diagnostic_cell, name, gpu)
                       for name, gpu in zip(names, GPU_IDS)]
            for future in as_completed(futures):
                future.result()
    elif args.command == "controls":
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(control_cell, control, gpu)
                       for control, gpu in zip(CONTROLS, GPU_IDS)]
            for future in as_completed(futures):
                future.result()
    elif args.command == "review-controls":
        review_controls()
    else:
        review = json.loads((OUT / "screen_review.json").read_text())
        mode = review["selected_loss"]
        run_parallel([
            ("full_core", "full", mode, "none"),
            ("full_ordinary", "full", mode, "ordinary"),
            ("full_challenge", "full", mode, "challenge"),
        ])


if __name__ == "__main__":
    main()
