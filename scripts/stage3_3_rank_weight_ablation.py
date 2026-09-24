#!/usr/bin/env python3
"""Run the staged A6S camera/LR ranking-weight ablation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import stage3_experiment as stage3
import stage3_3_symmetric_rank as a6s
import stage3_3_target_rank as a6
import stage3_3_ucpe_rre_fusion as prior
import stage4_readiness_review as review
from rl3dsr.validation.stage3_protocol import evenly_spaced_indices, load_stage3_config


ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = a6s.BASE_REVISION
DEFAULT_ROOT = ROOT / "artifacts/stage3_3_a6sw_20260922"
VARIANTS = {
    "w1_lego_seed42": {"rank_weight": 0.625, "camera_fraction": 0.8},
    "w2_lego_seed42": {"rank_weight": 0.875, "camera_fraction": 6 / 7},
}
CELL_ORDER = tuple(VARIANTS)
EXPECTED_TRACKED_CHANGES = {
    "scripts/stage3_experiment.py",
    "src/rl3dsr/models/wan/stage3.py",
    "src/rl3dsr/validation/stage3_protocol.py",
    "tests/test_stage3_conditioning.py",
    "tests/test_stage3_protocol.py",
}
CAMERA_CONDITIONS = (
    "shuffle_fusion",
    "mispaired_camera",
    "target_drop_shuffle_fusion",
)


def _source_bundle_sha256(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cell_paths(args, name):
    return (
        args.campaign_root / "config" / f"{name}.json",
        args.campaign_root / "manifest" / f"{name}.json",
        args.campaign_root / "train" / name,
        args.campaign_root / "eval" / name,
    )


def prepare(args):
    template = load_stage3_config(a6.CONFIG_TEMPLATE)
    if (
        template.arm != "A6"
        or template.pairing_supervision
        or template.steps != 1000
        or template.views != 4
        or template.target_lr_dropout != 0.5
        or template.camera_rank_weight != 1.0
        or template.camera_rank_margin_ratio != 0.05
    ):
        raise RuntimeError("frozen A6 template drift")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True
    ).strip()
    if revision != BASE_REVISION:
        raise RuntimeError("A6SW base revision mismatch")
    tracked = {
        line[3:].replace("\\", "/")
        for line in subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=args.repo_root,
            text=True,
        ).splitlines()
    }
    if tracked - EXPECTED_TRACKED_CHANGES:
        raise RuntimeError(f"unexpected tracked changes: {sorted(tracked)}")
    parent_hash = prior.sha256_file(args.phase_c_checkpoint)
    if parent_hash != prior.PHASE_C_CHECKPOINT_SHA256:
        raise RuntimeError("Phase C parent checkpoint hash mismatch")

    cells = {}
    for name, weights in VARIANTS.items():
        config_path, manifest_path, _, _ = cell_paths(args, name)
        config = replace(
            template,
            train_scenes=("lego",),
            validation_scenes=("chair", "drums"),
            test_scenes=("ficus", "hotdog", "materials"),
            training_seeds=(42,),
            camera_rank_weight=weights["rank_weight"],
            symmetric_correspondence_rank=True,
            symmetric_camera_fraction=weights["camera_fraction"],
        )
        prior.write_frozen_json(config_path, config.to_dict())
        if not manifest_path.exists():
            stage3._prepare_seen_manifest(
                SimpleNamespace(dataset_root=args.dataset_root, manifest=manifest_path), config
            )
        manifest = prior.read_json(manifest_path)
        probes = tuple(manifest["subsets"]["probe"])
        view_count = manifest["datasets"]["lego"]["views"]
        expected = tuple(
            f"lego:{index:03d}" for index in evenly_spaced_indices(view_count, 4)
        )
        if (
            manifest["dataset_root"] != str(args.dataset_root.resolve())
            or manifest["sampling_signature"] != stage3._sampling_signature(config)
            or probes != expected
        ):
            raise RuntimeError(f"{name}: manifest/config drift")
        cells[name] = {
            "scene": "lego",
            "seed": 42,
            "config": str(config_path.resolve()),
            "config_sha256": prior.sha256_file(config_path),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": prior.sha256_file(manifest_path),
            "probe_ids": probes,
            "rank_weight": weights["rank_weight"],
            "camera_fraction": weights["camera_fraction"],
            "camera_effective_weight": (
                weights["rank_weight"] * weights["camera_fraction"]
            ),
            "lr_effective_weight": (
                weights["rank_weight"] * (1 - weights["camera_fraction"])
            ),
        }

    source_paths = (
        ROOT / "scripts/stage3_3_rank_weight_ablation.py",
        ROOT / "scripts/stage3_3_symmetric_rank.py",
        ROOT / "scripts/stage3_experiment.py",
        ROOT / "scripts/stage4_readiness_review.py",
        ROOT / "src/rl3dsr/models/wan/stage3.py",
        ROOT / "src/rl3dsr/validation/stage3_protocol.py",
        ROOT / "tests/test_stage3_3_rank_weight_ablation.py",
        ROOT / "tests/test_stage3_3_symmetric_rank.py",
        ROOT / "tests/test_stage3_conditioning.py",
        ROOT / "tests/test_stage3_protocol.py",
    )
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=args.repo_root
    )
    protocol = {
        "schema_version": 1,
        "scope": "stage3_3_a6sw_rank_weight_ablation",
        "base_revision": revision,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_bundle_sha256": _source_bundle_sha256(source_paths),
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): prior.sha256_file(path)
            for path in source_paths
        },
        "phase_c_checkpoint_sha256": parent_hash,
        "matched_a5_lego_sha256": prior.sha256_file(args.a5_lego_summary),
        "a6_lego_summary_sha256": prior.sha256_file(args.a6_lego_summary),
        "a6s_lego_summary_sha256": prior.sha256_file(args.a6s_lego_summary),
        "loss": {
            "flow_scope": "all_views",
            "rank_scope": "target_view_0",
            "margin_ratio": 0.05,
            "shared_auxiliary_permutation": True,
        },
        "inference_seed": prior.INFERENCE_SEED,
        "modes": prior.MODES,
        "cell_order": CELL_ORDER,
        "cells": cells,
        "hard_gate": {
            "camera_conditions": CAMERA_CONDITIONS,
            "psnr": review.PSNR_CORRESPONDENCE,
            "ssim": review.SSIM_CORRESPONDENCE,
            "directional_probes": review.DIRECTIONAL_PROBES,
            "a6_minus_a5_psnr": review.PSNR_NONREGRESSION,
            "a6_minus_a5_ssim": review.SSIM_NONREGRESSION,
        },
        "diagnostic_only": [
            "camera_dose_psnr_monotonic",
            "camera_dose_ssim_monotonic",
            "last100_wrong_correct_margin",
            "last100_rank_hinge_activity",
        ],
        "stage4_ready": False,
        "forbidden": [
            "A7", "A8", "attention alignment", "PCGrad", "GradNorm",
            "Stage 4 training", "4DSR",
        ],
    }
    prior.write_frozen_json(args.campaign_root / "protocol.json", protocol)
    return protocol


def preflight(args, name):
    protocol = prepare(args)
    return a6s._preflight_prepared(
        args, name, protocol, cell_paths(args, name)
    )


def run_cell(args, name):
    protocol = prepare(args)
    item = protocol["cells"][name]
    config_path, manifest_path, train_dir, eval_dir = cell_paths(args, name)
    control = SimpleNamespace(
        repo_root=args.repo_root, campaign_root=args.campaign_root, gpu=args.gpu
    )
    preflight_path = args.campaign_root / "preflight" / f"{name}.json"
    if not preflight_path.exists():
        prior._run_logged(
            [args.python, str(Path(__file__)), "preflight", "--cell", name,
             "--campaign-root", str(args.campaign_root), "--gpu", str(args.gpu)],
            args=control,
            name=f"{name}_preflight",
        )
    preflight_result = prior.read_json(preflight_path)
    if (
        not preflight_result.get("pass")
        or preflight_result.get("config_sha256") != item["config_sha256"]
        or preflight_result.get("manifest_sha256") != item["manifest_sha256"]
    ):
        raise RuntimeError(f"{name}: preflight invalid")

    final = train_dir / "stage3_step_1000.pt"
    if not prior._complete_training(train_dir, 1000):
        command = [
            args.python, str(args.repo_root / "scripts/stage3_experiment.py"), "train",
            "--config", str(config_path), *a6._runtime_args(args),
            "--seed", "42", "--output-dir", str(train_dir),
        ]
        checkpoints = sorted(train_dir.glob("stage3_step_*.pt"))
        if checkpoints:
            command.extend(("--resume", str(checkpoints[-1])))
        elif train_dir.exists() and any(train_dir.iterdir()):
            raise RuntimeError(f"{name}: incomplete run without a checkpoint")
        else:
            command.extend((
                "--init-checkpoint", str(args.phase_c_checkpoint),
                "--init-reset-fusion",
            ))
        prior._run_logged(command, args=control, name=f"{name}_train")
    if not final.is_file() or not prior._complete_training(train_dir, 1000):
        raise RuntimeError(f"{name}: training incomplete")
    if not (eval_dir / "evaluation_summary.json").is_file():
        if eval_dir.exists() and any(eval_dir.iterdir()):
            raise RuntimeError(f"{name}: incomplete evaluation")
        prior._run_logged(
            [args.python, str(args.repo_root / "scripts/stage3_experiment.py"), "seen-eval",
             "--config", str(config_path), *a6._runtime_args(args),
             "--checkpoint", str(final), "--seen-manifest", str(manifest_path),
             "--subset", "probe", "--group-ids", *item["probe_ids"],
             "--inference-seeds", str(prior.INFERENCE_SEED),
             "--modes", *prior.MODES, "--save-diagnostics", "--save-images",
             "--output-dir", str(eval_dir)],
            args=control,
            name=f"{name}_evaluate",
        )
    return analyze(args, name)


def _paired_gate(gate, condition):
    values = gate["deltas"][condition]
    positive = {
        metric: sum(value > 0 for value in values[metric]["values"])
        for metric in ("psnr", "ssim")
    }
    checks = {
        "psnr_mean": values["psnr"]["mean"] >= review.PSNR_CORRESPONDENCE,
        "ssim_mean": values["ssim"]["mean"] >= review.SSIM_CORRESPONDENCE,
        "psnr_direction": positive["psnr"] >= review.DIRECTIONAL_PROBES,
        "ssim_direction": positive["ssim"] >= review.DIRECTIONAL_PROBES,
    }
    return {
        "pass": all(checks.values()),
        "mean": {metric: values[metric]["mean"] for metric in ("psnr", "ssim")},
        "values": {metric: values[metric]["values"] for metric in ("psnr", "ssim")},
        "positive_probes": positive,
        "checks": checks,
    }


def analyze(args, name):
    protocol = prepare(args)
    item = protocol["cells"][name]
    config_path, _, train_dir, eval_dir = cell_paths(args, name)
    config = load_stage3_config(config_path)
    checkpoint = train_dir / "stage3_step_1000.pt"
    rows = prior.read_jsonl(eval_dir / "evaluation_rows.jsonl")
    train = prior.read_jsonl(train_dir / "train_steps.jsonl")
    baseline = prior.read_jsonl(eval_dir / "baseline_rows.jsonl")
    diagnostics = prior.read_jsonl(eval_dir / "diagnostics.jsonl")
    evaluation = prior.read_json(eval_dir / "evaluation_summary.json")
    manifest = prior.read_json(train_dir / "run_manifest.json")
    probes = tuple(item["probe_ids"])
    index = prior.evaluation_index(rows, probes)
    base = {row["group_id"]: row for row in baseline if row["condition"] == "bicubic"}
    telemetry = train[-100:]
    gradient_keys = tuple(
        f"{prefix}_rank_{group}_gradient_norm"
        for prefix in ("camera", "lr")
        for group in ("qk", "value", "output", "bridge")
    )
    weights = {
        "correspondence_rank_weight": config.camera_rank_weight,
        "camera_rank_effective_weight": (
            config.camera_rank_weight * config.symmetric_camera_fraction
        ),
        "lr_rank_effective_weight": (
            config.camera_rank_weight * (1 - config.symmetric_camera_fraction)
        ),
    }
    integrity = {
        "training_complete": (
            len(train) == 1000
            and [row.get("step") for row in train] == list(range(1, 1001))
        ),
        "evaluation_rows": len(rows) == 48 == evaluation.get("rows"),
        "baseline_rows": len(baseline) == 8 and len(base) == 4,
        "diagnostic_rows": len(diagnostics) == 44,
        "images": len(list((eval_dir / "images").rglob("*.png"))) == 64,
        "manifest_config": manifest.get("config") == json.loads(json.dumps(config.to_dict())),
        "manifest_parent": (
            manifest.get("provenance", {}).get("parent_checkpoint_sha256")
            == protocol["phase_c_checkpoint_sha256"]
        ),
        "manifest_seed": manifest.get("provenance", {}).get("training_seed") == 42,
        "symmetric_contract": all(
            row.get("camera_rank_scope") == "target_view_0"
            and row.get("lr_rank_scope") == "target_view_0"
            and len(row.get("per_view_lr_wrong_flow_loss", [])) == 4
            and math.isfinite(float(row.get("correspondence_rank_loss", float("nan"))))
            for row in train
        ),
        "weight_telemetry": all(
            all(abs(float(row.get(key, float("nan"))) - value) <= 1e-12
                for key, value in weights.items())
            for row in train
        ),
        "rank_gradient_groups": all(
            all(math.isfinite(float(row.get(key, float("nan")))) for row in telemetry)
            and sum(float(row[key]) for row in telemetry) > 0
            for key in gradient_keys
        ),
        "paired_noise": all(
            len({row["sample_seed"] for row in diagnostics if row["group_id"] == group}) == 1
            for group in probes
        ),
        "hashes": evaluation.get("seen_manifest_sha256") == item["manifest_sha256"],
    }
    cell = prior.Cell(name, f"{name}.json", 42, True)
    equivariance = prior.fusion_equivariance(
        SimpleNamespace(campaign_root=args.campaign_root, v2=True), cell
    )
    gate = prior.candidate_gate(
        rows, train, expected_steps=1000, reference=None, v2=True,
        fusion_equivariant=equivariance["pass"], probe_ids=probes,
    )
    a5 = review._load_cell(args.a5_lego_summary, arm="A5", scene="lego", seed=42)
    if tuple(a5["probe_ids"]) != probes:
        raise RuntimeError("matched A5/A6SW probes differ")
    quality = review.nonregression(review._gate(a5["summary"])["correct"], gate["correct"])
    lr = _paired_gate(gate, "mispaired_lr")
    camera = {condition: _paired_gate(gate, condition) for condition in CAMERA_CONDITIONS}
    dependency_checks = {
        key: gate["checks"][key]
        for key in (
            "correct_repeat_equivalent", "joint_permute_equivalent",
            "remove_psnr", "remove_direction",
            "target_drop_psnr", "target_drop_direction",
        )
    }
    core = {
        "integrity": all(integrity.values()),
        "fusion_equivariance": equivariance["pass"],
        "dependency_and_invariants": all(dependency_checks.values()),
        "matched_a5_nonregression": quality["pass"],
        "lr_correspondence": lr["pass"],
        "camera_correspondence": all(value["pass"] for value in camera.values()),
    }
    passed = all(core.values())
    if not core["integrity"]:
        failure, next_step = "AUDIT_INVALID", "FIX_AUDIT_INPUTS"
    elif not core["dependency_and_invariants"] or not core["fusion_equivariance"]:
        failure, next_step = "CORE_INVALID", "FIX_CORE_CONTRACT"
    elif not core["matched_a5_nonregression"]:
        failure, next_step = "QUALITY_FAIL", "REVIEW_QUALITY_GRADIENT_CONFLICT"
    elif not core["lr_correspondence"]:
        failure, next_step = "LR_FAIL", "REVIEW_LR_RANK_WEIGHT"
    elif not core["camera_correspondence"]:
        failure, next_step = "CAMERA_FAIL", (
            "RUN_W2" if name == CELL_ORDER[0] else "REVIEW_ATTENTION_CORRESPONDENCE"
        )
    else:
        failure, next_step = None, "REPLICATE_WEIGHT_CANDIDATE"
    bicubic_gain = sum(
        index[(group, "correct")]["psnr"] - base[group]["psnr"] for group in probes
    ) / len(probes)
    result = {
        "status": "WEIGHT_CANDIDATE" if passed else "HOLD",
        "pass": passed,
        "failure_category": failure,
        "next": next_step,
        "weights": weights,
        "bicubic_psnr_gain": bicubic_gain,
        "hard_gate": {"pass": passed, "checks": core},
        "dependency_checks": dependency_checks,
        "matched_a5_nonregression": quality,
        "lr_correspondence": lr,
        "camera_correspondence": camera,
        "diagnostics": {
            "camera_dose_monotonic_probes": gate["camera_dose_monotonic_probes"],
            "last100": gate["last100"],
            "checks": {
                key: gate["checks"][key]
                for key in protocol["diagnostic_only"]
            },
        },
        "integrity": {"pass": all(integrity.values()), "checks": integrity},
        "fusion_equivariance": equivariance,
        "checkpoint_sha256": prior.sha256_file(checkpoint),
        "WEIGHT_CANDIDATE_READY": passed,
        "STAGE4_READY": False,
    }
    prior.write_frozen_json(
        args.campaign_root / "analysis" / name / "summary.json", result
    )
    return result


def _write_final(args, results):
    candidate = next((name for name, value in results.items() if value and value["pass"]), None)
    last = next((value for value in reversed(tuple(results.values())) if value is not None), None)
    payload = {
        "schema_version": 1,
        "verdict": "WEIGHT_CANDIDATE" if candidate else "HOLD",
        "WEIGHT_CANDIDATE_READY": candidate is not None,
        "STAGE4_READY": False,
        "selected_candidate": candidate,
        "next": "REPLICATE_WEIGHT_CANDIDATE" if candidate else (last or {}).get("next", "HOLD"),
        "cells": {
            name: None if value is None else {
                "status": value["status"],
                "failure_category": value["failure_category"],
                "pass": value["pass"],
                "weights": value["weights"],
                "hard_gate": value["hard_gate"],
            }
            for name, value in results.items()
        },
        "protocol_sha256": prior.sha256_file(args.campaign_root / "protocol.json"),
    }
    prior.write_frozen_json(args.campaign_root / "machine_verdict.json", payload)
    output = args.campaign_root / "stage4_review"
    prior.write_frozen_json(output / "stage4_review.json", payload)
    rows = [
        "# A6S 权重消融评审", "",
        f"**{payload['verdict']}；STAGE4_READY=false。**", "",
        "| 单元 | Camera权重 | LR权重 | 硬门槛 | 失败类别 |",
        "|---|---:|---:|:---:|---|",
    ]
    for name, value in results.items():
        if value is None:
            rows.append(f"| {name} | - | - | 未运行 | 阶段止损 |")
        else:
            weights = value["weights"]
            rows.append(
                f"| {name} | {weights['camera_rank_effective_weight']:.3f} | "
                f"{weights['lr_rank_effective_weight']:.3f} | "
                f"{'通过' if value['pass'] else '失败'} | {value['failure_category'] or '-'} |"
            )
    rows.extend(["", "## 下一步", "", payload["next"]])
    prior.write_frozen_text(output / "stage4_review.md", "\n".join(rows) + "\n")
    return payload


def run(args):
    results = {name: None for name in CELL_ORDER}
    w1 = run_cell(args, CELL_ORDER[0])
    results[CELL_ORDER[0]] = w1
    if not w1["pass"] and w1["failure_category"] == "CAMERA_FAIL":
        results[CELL_ORDER[1]] = run_cell(args, CELL_ORDER[1])
    return _write_final(args, results)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run-cell", "analyze", "run"))
    parser.add_argument("--cell", choices=CELL_ORDER)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase-c-checkpoint", type=Path, default=a6.ORIGINAL / "artifacts/stage3_2_camera_causality_20260913/phase_c/train/rank/seed42/stage3_step_1000.pt")
    parser.add_argument("--dataset-root", type=Path, default=a6.ORIGINAL / "datasets/nerf_synthetic")
    parser.add_argument("--model-dir", type=Path, default=a6.ORIGINAL / "models/Wan2.1-T2V-1.3B")
    parser.add_argument("--lq-source", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py"))
    parser.add_argument("--lq-checkpoint", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"))
    parser.add_argument("--bridge-checkpoint", type=Path, default=a6.ORIGINAL / "artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt")
    parser.add_argument("--a5-lego-summary", type=Path, default=review.DEFAULT_A5_LEGO)
    parser.add_argument("--a6-lego-summary", type=Path, default=ROOT / "artifacts/stage3_3_a6_20260921/analysis/a6_lego_seed42/summary.json")
    parser.add_argument("--a6s-lego-summary", type=Path, default=ROOT / "artifacts/stage3_3_a6s_20260922/analysis/a6_lego_seed42/summary.json")
    parser.add_argument("--python", default=sys.executable)
    parser.set_defaults(rre_checkpoint=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.repo_root = ROOT
    args.campaign_root = args.campaign_root.resolve()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "run":
        result = run(args)
    else:
        if args.cell is None:
            raise SystemExit(f"{args.command} requires --cell")
        result = {"preflight": preflight, "run-cell": run_cell, "analyze": analyze}[
            args.command
        ](args, args.cell)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
