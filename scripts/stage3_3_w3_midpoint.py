#!/usr/bin/env python3
"""Pre-registered W3 target-flow test, then conditional replication."""
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
import stage3_3_rank_weight_ablation as previous
import stage3_3_symmetric_rank as symmetric
import stage3_3_target_rank as a6
import stage3_3_ucpe_rre_fusion as prior
import stage4_readiness_review as review
from rl3dsr.validation.stage3_protocol import evenly_spaced_indices, load_stage3_config


ROOT = previous.ROOT
OLD = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-target-rank")
DEFAULT_ROOT = ROOT / "artifacts/stage3_3_targetflow_20260923"
W1W2_ROOT = OLD / "artifacts/stage3_3_a6sw_20260922"
CELLS = {
    "f1_lego_seed42": a6.CELLS["a6_lego_seed42"],
    "f1_lego_seed43": a6.CELLS["a6_lego_seed43"],
    "f1_chair_seed42": a6.CELLS["a6_chair_seed42"],
}
ORDER = tuple(CELLS)
TOTAL_WEIGHT = 0.75
CAMERA_FRACTION = 5 / 6
TARGET_FLOW_FRACTION = 0.5


def paths(args, name):
    return previous.cell_paths(args, name)


def prepare(args):
    template = load_stage3_config(a6.CONFIG_TEMPLATE)
    if (template.arm != "A6" or template.steps != 1000 or template.views != 4
            or template.target_lr_dropout != 0.5 or template.camera_rank_margin_ratio != 0.05
            or template.camera_rank_weight != 1.0 or template.pairing_supervision):
        raise RuntimeError("frozen A6 template drift")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != previous.BASE_REVISION:
        raise RuntimeError("base revision drift")
    tracked = {line[3:].replace("\\", "/") for line in subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).splitlines()}
    if tracked - previous.EXPECTED_TRACKED_CHANGES:
        raise RuntimeError(f"unexpected tracked changes: {sorted(tracked)}")
    parent_hash = prior.sha256_file(args.phase_c_checkpoint)
    if parent_hash != prior.PHASE_C_CHECKPOINT_SHA256:
        raise RuntimeError("Phase C checkpoint drift")
    old = prior.read_json(args.w1w2_root / "protocol.json")
    for name in previous.CELL_ORDER:
        item = old["cells"][name]
        old_config, old_manifest, old_train, _ = previous.cell_paths(
            SimpleNamespace(campaign_root=args.w1w2_root), name
        )
        old_summary = prior.read_json(args.w1w2_root / "analysis" / name / "summary.json")
        if (prior.sha256_file(old_config) != item["config_sha256"]
                or prior.sha256_file(old_manifest) != item["manifest_sha256"]
                or prior.sha256_file(old_train / "stage3_step_1000.pt") != old_summary["checkpoint_sha256"]):
            raise RuntimeError("W1/W2 frozen input drift")
    audit = prior.read_json(args.gradient_audit)
    if (not audit.get("read_only")
            or audit.get("w1w2_protocol_sha256") != prior.sha256_file(args.w1w2_root / "protocol.json")):
        raise RuntimeError("paired gradient audit missing or mismatched")
    midpoint_audit = prior.read_json(args.gradient_audit_midpoint)
    if (not midpoint_audit.get("read_only") or midpoint_audit.get("checkpoint_step") != 500
            or midpoint_audit.get("w1w2_protocol_sha256") != audit["w1w2_protocol_sha256"]):
        raise RuntimeError("step500 gradient audit missing or mismatched")
    image_audit = prior.read_json(args.campaign_root / "audit" / "image_audit.json")
    gradient_audit = prior.read_json(args.campaign_root / "audit" / "gradient_audit.json")
    if (not image_audit.get("read_only_inputs") or not gradient_audit.get("read_only_inputs")
            or tuple(image_audit.get("probes", ())) != ("lego:000", "lego:033", "lego:066", "lego:099")
            or any(len(gradient_audit.get("stages", {}).get(stage, ())) != 24
                   for stage in ("initialization", "step500", "step1000"))):
        raise RuntimeError("target-flow attribution audit incomplete")
    cells = {}
    for name, (scene, seed, validation, test) in CELLS.items():
        config_path, manifest_path, _, _ = paths(args, name)
        config = replace(
            template, train_scenes=(scene,), validation_scenes=validation,
            test_scenes=test, training_seeds=(seed,), camera_rank_weight=TOTAL_WEIGHT,
            symmetric_correspondence_rank=True, symmetric_camera_fraction=CAMERA_FRACTION,
            target_view_flow_fraction=TARGET_FLOW_FRACTION,
        )
        prior.write_frozen_json(config_path, config.to_dict())
        if not manifest_path.exists():
            stage3._prepare_seen_manifest(
                SimpleNamespace(dataset_root=args.dataset_root, manifest=manifest_path), config
            )
        manifest = prior.read_json(manifest_path)
        expected = tuple(f"{scene}:{i:03d}" for i in evenly_spaced_indices(
            manifest["datasets"][scene]["views"], 4
        ))
        if (manifest["dataset_root"] != str(args.dataset_root.resolve())
                or manifest["sampling_signature"] != stage3._sampling_signature(config)
                or tuple(manifest["subsets"]["probe"]) != expected):
            raise RuntimeError(f"{name}: manifest drift")
        cells[name] = {
            "scene": scene, "seed": seed, "config_sha256": prior.sha256_file(config_path),
            "manifest_sha256": prior.sha256_file(manifest_path), "probe_ids": expected,
        }
    source_paths = [
        ROOT / path for path in (
            "scripts/stage3_3_w3_midpoint.py", "scripts/stage3_3_weight_gradient_audit.py",
            "scripts/stage3_3_targetflow_audit.py",
            "scripts/stage3_3_rank_weight_ablation.py", "scripts/stage3_3_symmetric_rank.py",
            "scripts/stage3_experiment.py", "scripts/stage4_readiness_review.py",
            "src/rl3dsr/models/wan/stage3.py", "src/rl3dsr/validation/stage3_protocol.py",
            "tests/test_stage3_3_w3_midpoint.py",
        )
    ]
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=ROOT)
    protocol = {
        "schema_version": 1, "scope": "stage3_3_a6s_targetflow", "base_revision": revision,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): prior.sha256_file(p)
                          for p in source_paths},
        "source_bundle_sha256": previous._source_bundle_sha256(source_paths),
        "w1w2_protocol_sha256": prior.sha256_file(args.w1w2_root / "protocol.json"),
        "gradient_audit_sha256": prior.sha256_file(args.gradient_audit),
        "gradient_audit_midpoint_sha256": prior.sha256_file(args.gradient_audit_midpoint),
        "targetflow_image_audit_sha256": prior.sha256_file(args.campaign_root / "audit" / "image_audit.json"),
        "targetflow_gradient_audit_sha256": prior.sha256_file(args.campaign_root / "audit" / "gradient_audit.json"),
        "phase_c_checkpoint_sha256": parent_hash,
        "matched_a5_sha256": {
            "lego_seed42": prior.sha256_file(args.a5_lego_summary),
            "chair_seed42": prior.sha256_file(args.a5_chair_summary),
        },
        "loss": {"flow_scope": "target_0.5_aux_each_1/6", "rank_scope": "target_view_0",
                 "margin_ratio": 0.05, "shared_auxiliary_permutation": True,
                 "total_weight": TOTAL_WEIGHT, "camera_fraction": CAMERA_FRACTION,
                 "camera_effective_weight": 0.625, "lr_effective_weight": 0.125},
        "inference_seed": prior.INFERENCE_SEED, "modes": prior.MODES,
        "cell_order": ORDER, "cells": cells,
        "hard_gate": {"psnr": 0.03, "ssim": 0.0003, "positive_probes": 3,
                      "matched_a5_psnr": -0.10, "matched_a5_ssim": -0.001,
                      "conditions": ("mispaired_lr", *previous.CAMERA_CONDITIONS)},
        "diagnostic_only": ("lpips", "per_probe_ssim", "camera_dose", "hinge"),
        "stage4_hold_until_all_three_pass": True,
    }
    prior.write_frozen_json(args.campaign_root / "protocol.json", protocol)
    return protocol


def preflight(args, name):
    return symmetric._preflight_prepared(args, name, prepare(args), paths(args, name))


def analyze(args, name):
    protocol = prepare(args)
    item = protocol["cells"][name]
    config_path, _, train_dir, eval_dir = paths(args, name)
    config = load_stage3_config(config_path)
    train = prior.read_jsonl(train_dir / "train_steps.jsonl")
    rows = prior.read_jsonl(eval_dir / "evaluation_rows.jsonl")
    baseline = prior.read_jsonl(eval_dir / "baseline_rows.jsonl")
    diagnostics = prior.read_jsonl(eval_dir / "diagnostics.jsonl")
    evaluation = prior.read_json(eval_dir / "evaluation_summary.json")
    manifest = prior.read_json(train_dir / "run_manifest.json")
    probes = tuple(item["probe_ids"])
    telemetry = train[-100:]
    weights = {"correspondence_rank_weight": TOTAL_WEIGHT,
               "camera_rank_effective_weight": 0.625, "lr_rank_effective_weight": 0.125}
    gradient_keys = tuple(f"{prefix}_rank_{group}_gradient_norm"
                          for prefix in ("camera", "lr") for group in ("qk", "value", "output", "bridge"))
    integrity = {
        "training_complete": len(train) == 1000 and [r.get("step") for r in train] == list(range(1, 1001)),
        "evaluation_rows": len(rows) == 48 == evaluation.get("rows"),
        "baseline_rows": len(baseline) == 8,
        "diagnostic_rows": len(diagnostics) == 44,
        "images": len(list((eval_dir / "images").rglob("*.png"))) == 64,
        "config_hash": prior.sha256_file(config_path) == item["config_sha256"],
        "manifest_hash": prior.sha256_file(args.campaign_root / "manifest" / f"{name}.json") == item["manifest_sha256"],
        "manifest_config": manifest.get("config") == json.loads(json.dumps(config.to_dict())),
        "manifest_parent": manifest.get("provenance", {}).get("parent_checkpoint_sha256") == protocol["phase_c_checkpoint_sha256"],
        "manifest_seed": manifest.get("provenance", {}).get("training_seed") == item["seed"],
        "paired_noise": all(len({r["sample_seed"] for r in diagnostics if r["group_id"] == probe}) == 1 for probe in probes),
        "evaluation_manifest": evaluation.get("seen_manifest_sha256") == item["manifest_sha256"],
        "weight_telemetry": all(all(abs(float(r.get(k, float("nan"))) - v) <= 1e-12 for k, v in weights.items()) for r in train),
        "rank_scope": all(r.get("camera_rank_scope") == "target_view_0" and r.get("lr_rank_scope") == "target_view_0"
                          and len(r.get("per_view_lr_wrong_flow_loss", [])) == 4 for r in train),
        "gradient_groups": all(all(math.isfinite(float(r.get(k, float("nan")))) for r in telemetry)
                               and sum(float(r[k]) for r in telemetry) > 0 for k in gradient_keys),
        "checkpoint": (train_dir / "stage3_step_1000.pt").is_file(),
    }
    equivariance = prior.fusion_equivariance(
        SimpleNamespace(campaign_root=args.campaign_root, v2=True),
        prior.Cell(name, f"{name}.json", item["seed"], True),
    )
    gate = prior.candidate_gate(rows, train, expected_steps=1000, reference=None, v2=True,
                                fusion_equivariant=equivariance["pass"], probe_ids=probes)
    interventions = {condition: previous._paired_gate(gate, condition)
                     for condition in protocol["hard_gate"]["conditions"]}
    a5_path = args.a5_lego_summary if name == ORDER[0] else args.a5_chair_summary if name == ORDER[2] else None
    quality = None
    if a5_path is not None:
        a5 = review._load_cell(a5_path, arm="A5", scene=item["scene"], seed=item["seed"])
        if tuple(a5["probe_ids"]) != probes:
            raise RuntimeError("matched A5 probe mismatch")
        quality = review.nonregression(review._gate(a5["summary"])["correct"], gate["correct"])
    baseline_index = {r["group_id"]: r for r in baseline if r["condition"] == "bicubic"}
    index = prior.evaluation_index(rows, probes)
    bicubic_gain = sum(index[(probe, "correct")]["psnr"] - baseline_index[probe]["psnr"]
                       for probe in probes) / len(probes)
    dependency = {key: gate["checks"][key] for key in (
        "correct_repeat_equivalent", "joint_permute_equivalent", "remove_psnr",
        "remove_direction", "target_drop_psnr", "target_drop_direction",
    )}
    checks = {"integrity": all(integrity.values()), "fusion_equivariance": equivariance["pass"],
              "dependency": all(dependency.values()), "bicubic_gain": bicubic_gain > 0,
              "quality": quality is None or quality["pass"],
              **{condition: value["pass"] for condition, value in interventions.items()}}
    passed = all(checks.values())
    if not checks["integrity"]:
        failure, next_step = "AUDIT_INVALID", "FIX_AUDIT_INPUTS"
    elif not checks["fusion_equivariance"] or not checks["dependency"]:
        failure, next_step = "CORE_INVALID", "FIX_CORE_CONTRACT"
    elif not checks["quality"] or not checks["bicubic_gain"]:
        failure, next_step = "QUALITY_FAIL", "REVIEW_FLOW_RANK_GRADIENTS"
    elif not checks["mispaired_lr"]:
        failure, next_step = "LR_FAIL", "REVIEW_LR_CONSTRAINT"
    elif not all(checks[c] for c in previous.CAMERA_CONDITIONS):
        failure, next_step = "CAMERA_FAIL", "REVIEW_CORRESPONDENCE_STRUCTURE"
    else:
        failure, next_step = None, "RUN_NEXT_REPLICATION" if name != ORDER[-1] else "REVIEW_STAGE4"
    result = {
        "status": "PASS" if passed else "HOLD", "pass": passed,
        "failure_category": failure, "next": next_step,
        "hard_gate": {"pass": passed, "checks": checks},
        "integrity": {"pass": checks["integrity"], "checks": integrity},
        "fusion_equivariance": equivariance, "dependency_checks": dependency,
        "matched_a5_nonregression": quality, "bicubic_psnr_gain": bicubic_gain,
        "interventions": interventions,
        "diagnostics": {"correct": gate["correct"], "camera_dose": gate["camera_dose_monotonic_probes"],
                        "last100": gate["last100"], "lpips": "not_recorded_by_frozen_evaluator",
                        "per_probe_ssim": {c: gate["deltas"][c]["ssim"]["values"] for c in interventions}},
        "checkpoint_sha256": prior.sha256_file(train_dir / "stage3_step_1000.pt"),
        "STAGE4_READY": False,
    }
    prior.write_frozen_json(args.campaign_root / "analysis" / name / "summary.json", result)
    return result


def run_cell(args, name):
    protocol = prepare(args)
    item = protocol["cells"][name]
    config_path, manifest_path, train_dir, eval_dir = paths(args, name)
    control = SimpleNamespace(repo_root=ROOT, campaign_root=args.campaign_root, gpu=args.gpu)
    preflight_path = args.campaign_root / "preflight" / f"{name}.json"
    if not preflight_path.is_file():
        prior._run_logged([args.python, str(Path(__file__)), "preflight", "--cell", name,
                           "--campaign-root", str(args.campaign_root), "--gpu", str(args.gpu)],
                          args=control, name=f"{name}_preflight")
    preflight_result = prior.read_json(preflight_path)
    if (not preflight_result.get("pass") or preflight_result.get("config_sha256") != item["config_sha256"]
            or preflight_result.get("manifest_sha256") != item["manifest_sha256"]):
        raise RuntimeError(f"{name}: preflight invalid")
    checkpoint = train_dir / "stage3_step_1000.pt"
    if not prior._complete_training(train_dir, 1000):
        command = [args.python, str(ROOT / "scripts/stage3_experiment.py"), "train",
                   "--config", str(config_path), *a6._runtime_args(args), "--seed", str(item["seed"]),
                   "--output-dir", str(train_dir)]
        existing = sorted(train_dir.glob("stage3_step_*.pt"))
        if existing:
            command.extend(("--resume", str(existing[-1])))
        elif train_dir.exists() and any(train_dir.iterdir()):
            raise RuntimeError(f"{name}: incomplete train without checkpoint")
        else:
            command.extend(("--init-checkpoint", str(args.phase_c_checkpoint), "--init-reset-fusion"))
        prior._run_logged(command, args=control, name=f"{name}_train")
    if not checkpoint.is_file() or not prior._complete_training(train_dir, 1000):
        raise RuntimeError(f"{name}: training incomplete")
    if not (eval_dir / "evaluation_summary.json").is_file():
        if eval_dir.exists() and any(eval_dir.iterdir()):
            raise RuntimeError(f"{name}: incomplete evaluation")
        prior._run_logged([args.python, str(ROOT / "scripts/stage3_experiment.py"), "seen-eval",
                           "--config", str(config_path), *a6._runtime_args(args), "--checkpoint", str(checkpoint),
                           "--seen-manifest", str(manifest_path), "--subset", "probe", "--group-ids",
                           *item["probe_ids"], "--inference-seeds", str(prior.INFERENCE_SEED),
                           "--modes", *prior.MODES, "--save-diagnostics", "--save-images",
                           "--output-dir", str(eval_dir)], args=control, name=f"{name}_evaluate")
    return analyze(args, name)


def verdict(args, results):
    candidate = len(results) == len(ORDER) and all(results[name]["pass"] for name in ORDER)
    failed = next((value for value in results.values() if not value["pass"]), None)
    payload = {"schema_version": 1, "STAGE4_READY": False,
               "verdict": "REVIEW_STAGE4" if candidate else "HOLD",
               "next": "REVIEW_STAGE4_ONLY" if candidate else (failed or {}).get("next", "CONTINUE_REPLICATION"),
               "protocol_sha256": prior.sha256_file(args.campaign_root / "protocol.json"),
               "gradient_audit_sha256": prior.sha256_file(args.gradient_audit),
               "cells": {name: None if name not in results else {
                   "pass": results[name]["pass"], "hard_gate": results[name]["hard_gate"],
                   "checkpoint_sha256": results[name]["checkpoint_sha256"],
                   "summary_sha256": prior.sha256_file(args.campaign_root / "analysis" / name / "summary.json"),
               } for name in ORDER}}
    prior.write_frozen_json(args.campaign_root / "machine_verdict.json", payload)
    prior.write_frozen_json(args.campaign_root / "stage4_review" / "stage4_review.json", payload)
    lines = ["# 目标视图 flow 加权：Stage 4 训练前评审", "", f"结论：{payload['verdict']}；Stage 4 HOLD。", "",
             "| Cell | 质量 | LR | Camera 三项 | 结论 |", "|---|---|---|---|---|"]
    for name in ORDER:
        result = results.get(name)
        if result is None:
            lines.append(f"| {name} | — | — | — | 未运行（阶段止损） |")
        else:
            c = result["hard_gate"]["checks"]
            lines.append(f"| {name} | {c['quality']} | {c['mispaired_lr']} | "
                         f"{all(c[key] for key in previous.CAMERA_CONDITIONS)} | {result['failure_category'] or 'PASS'} |")
    lines.extend(["", f"下一步：{payload['next']}", ""])
    prior.write_frozen_text(args.campaign_root / "stage4_review" / "stage4_review.md", "\n".join(lines))
    return payload


def run(args):
    results = {}
    for name in ORDER:
        result = run_cell(args, name)
        results[name] = result
        if not result["pass"]:
            break
    return verdict(args, results)


def build_parser():
    base = previous.build_parser().parse_args(["prepare"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run-cell", "analyze", "run"))
    parser.add_argument("--cell", choices=ORDER)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--w1w2-root", type=Path, default=W1W2_ROOT)
    parser.add_argument("--gradient-audit", type=Path, default=W1W2_ROOT / "gradient_audit" / "w1_w2.json")
    parser.add_argument("--gradient-audit-midpoint", type=Path, default=W1W2_ROOT / "gradient_audit" / "w1_w2_step500.json")
    parser.add_argument("--a5-chair-summary", type=Path, default=review.DEFAULT_A5_CHAIR)
    for name in ("gpu", "phase_c_checkpoint", "dataset_root", "model_dir", "lq_source",
                 "lq_checkpoint", "bridge_checkpoint", "a5_lego_summary", "python"):
        parser.add_argument("--" + name.replace("_", "-"), type=int if name == "gpu" else str if name == "python" else Path,
                            default=getattr(base, name))
    parser.set_defaults(rre_checkpoint=None)
    return parser


def main():
    args = build_parser().parse_args()
    args.repo_root = ROOT
    args.campaign_root = args.campaign_root.resolve()
    args.w1w2_root = args.w1w2_root.resolve()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "run":
        result = run(args)
    else:
        if args.cell is None:
            raise SystemExit("--cell required")
        result = {"preflight": preflight, "run-cell": run_cell, "analyze": analyze}[args.command](args, args.cell)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
