#!/usr/bin/env python3
"""Run the fail-closed A6 target-view camera-ranking campaign."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

import stage3_experiment as stage3
import stage3_3_ucpe_rre_fusion as prior
from rl3dsr.models.wan import flow_matching_pair
from rl3dsr.validation.stage3_protocol import evenly_spaced_indices, load_stage3_config


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path("/data/linzizhuo/RL3DSR_WAN_REMO")
CONFIG_TEMPLATE = ROOT / "configs/stage3_3/A6_target_rank_lego_1000.json"
CELLS = {
    "a6_lego_seed42": ("lego", 42, ("chair",), ("drums",)),
    "a6_lego_seed43": ("lego", 43, ("chair",), ("drums",)),
    "a6_chair_seed42": ("chair", 42, ("lego",), ("drums",)),
}


def cell_paths(args, name):
    return (
        args.campaign_root / "config" / f"{name}.json",
        args.campaign_root / "manifest" / f"{name}.json",
        args.campaign_root / "train" / name,
        args.campaign_root / "eval" / name,
    )


def prepare(args):
    template = load_stage3_config(CONFIG_TEMPLATE)
    if (
        template.arm != "A6"
        or template.pairing_supervision
        or template.steps != 1000
        or template.views != 4
        or template.target_lr_dropout != 0.5
        or template.camera_rank_weight != 1.0
        or template.camera_rank_margin_ratio != 0.05
    ):
        raise RuntimeError("A6 target-rank template drift")
    parent_hash = prior.sha256_file(args.phase_c_checkpoint)
    if parent_hash != prior.PHASE_C_CHECKPOINT_SHA256:
        raise RuntimeError("Phase C parent checkpoint hash mismatch")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=args.repo_root,
        text=True,
    ).strip():
        raise RuntimeError("tracked source must be clean before freezing A6")

    cells = {}
    for name, (scene, seed, validation, test) in CELLS.items():
        config_path, manifest_path, _, _ = cell_paths(args, name)
        config = replace(
            template,
            train_scenes=(scene,),
            validation_scenes=validation,
            test_scenes=test,
            training_seeds=(seed,),
        )
        prior.write_frozen_json(config_path, config.to_dict())
        if not manifest_path.exists():
            stage3._prepare_seen_manifest(
                SimpleNamespace(dataset_root=args.dataset_root, manifest=manifest_path), config
            )
        manifest = prior.read_json(manifest_path)
        probes = tuple(manifest["subsets"]["probe"])
        view_count = manifest["datasets"][scene]["views"]
        expected = tuple(
            f"{scene}:{index:03d}" for index in evenly_spaced_indices(view_count, 4)
        )
        if (
            manifest["dataset_root"] != str(args.dataset_root.resolve())
            or manifest["sampling_signature"] != stage3._sampling_signature(config)
            or probes != expected
        ):
            raise RuntimeError(f"{name}: manifest/config drift")
        cells[name] = {
            "scene": scene,
            "seed": seed,
            "config": str(config_path.resolve()),
            "config_sha256": prior.sha256_file(config_path),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": prior.sha256_file(manifest_path),
            "probe_ids": probes,
        }

    source_paths = (
        ROOT / "scripts/stage3_3_target_rank.py",
        ROOT / "scripts/stage3_experiment.py",
        ROOT / "src/rl3dsr/validation/stage3_protocol.py",
        ROOT / "src/rl3dsr/models/wan/lr_fusion.py",
        ROOT / "src/rl3dsr/models/wan/stage3.py",
    )
    protocol = {
        "schema_version": 1,
        "scope": "stage3_3_a6_target_rank",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): prior.sha256_file(path)
            for path in source_paths
        },
        "phase_c_checkpoint_sha256": parent_hash,
        "inference_seed": prior.INFERENCE_SEED,
        "modes": prior.MODES,
        "cells": cells,
        "gate": {
            "shuffle_fusion_psnr": 0.03,
            "shuffle_fusion_ssim": 0.0003,
            "directional_probes": 3,
            "dose_monotonic_probes": 3,
        },
        "forbidden": ["Counter", "8 views", "10000 steps", "Stage 4", "4DSR"],
        "stage4_ready": False,
    }
    prior.write_frozen_json(args.campaign_root / "protocol.json", protocol)
    return protocol


def _runtime_args(args):
    return [
        "--dataset-root", str(args.dataset_root),
        "--model-dir", str(args.model_dir),
        "--lq-source", str(args.lq_source),
        "--lq-checkpoint", str(args.lq_checkpoint),
        "--bridge-checkpoint", str(args.bridge_checkpoint),
        "--device", "cuda",
    ]


def _rank_probe(runtime, config, lr, camera, clean, wrong_camera, *, seed):
    prepared = runtime.module.prepare_multiview(
        lr, camera, tuple(clean.shape[2:]), (config.image_size, config.image_size)
    )
    wrong_prepared = runtime.module.prepare_multiview(
        lr, wrong_camera, tuple(clean.shape[2:]), (config.image_size, config.image_size)
    )
    generator = torch.Generator(device=runtime.device).manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator, device=runtime.device, dtype=clean.dtype)
    sigma = torch.tensor([0.5], device=runtime.device)
    noisy, timestep, target = flow_matching_pair(clean, noise, sigma)
    correct = runtime.module.predict(
        runtime.dit, noisy, timestep, None, prepared, camera, tuple(clean.shape[2:])
    )
    wrong = runtime.module.predict(
        runtime.dit, noisy, timestep, None, wrong_prepared, camera, tuple(clean.shape[2:])
    )
    flow, e_correct, e_wrong, rank = stage3._camera_pair_training_losses(
        correct,
        wrong,
        target,
        margin_ratio=config.camera_rank_margin_ratio,
        target_view_only=True,
    )
    return flow, e_correct, e_wrong, rank, correct, wrong, target


def preflight(args, name):
    protocol = prepare(args)
    item = protocol["cells"][name]
    config_path, manifest_path, _, _ = cell_paths(args, name)
    config = load_stage3_config(config_path)
    manifest = prior.read_json(manifest_path)
    group = next(row for row in manifest["groups"] if row["id"] == item["probe_ids"][0])
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
    _, hr, lr, camera = stage3._load_indices(
        args.dataset_root, item["scene"], "train", group["indices"], config, runtime.device
    )
    lr = lr.clone()
    lr[:, :, 0] = 0
    with torch.no_grad():
        clean = runtime.vae.encode_multiview(hr)
    wrong_camera = stage3.derange_auxiliary_fusion_camera(
        camera, torch.Generator().manual_seed(6000)
    )
    trainable = [parameter for parameter in runtime.module.parameters() if parameter.requires_grad]
    initial = _rank_probe(runtime, config, lr, camera, clean, wrong_camera, seed=3302)
    initial_gradients = torch.autograd.grad(
        initial[3].mean(), trainable, retain_graph=False, allow_unused=True
    )
    initial_groups = stage3._camera_rank_gradient_groups(
        runtime.module, trainable, initial_gradients
    )
    output_gradient = next(
        gradient
        for parameter, gradient in zip(trainable, initial_gradients)
        if parameter is runtime.module.fusion.output.weight
    )
    if output_gradient is None or not torch.isfinite(output_gradient).all() or not output_gradient.any():
        raise RuntimeError("A6 zero-init output path has no ranking gradient")
    with torch.no_grad():
        scale = output_gradient.float().norm().clamp_min(1e-12)
        runtime.module.fusion.output.weight.add_(-1e-3 * output_gradient / scale)

    active = _rank_probe(runtime, config, lr, camera, clean, wrong_camera, seed=3302)
    contrast = active[1].mean() - active[2].mean()
    active_gradients = torch.autograd.grad(
        contrast, trainable, retain_graph=False, allow_unused=True
    )
    active_groups = stage3._camera_rank_gradient_groups(runtime.module, trainable, active_gradients)
    if any(not math.isfinite(value) or value <= 0 for value in active_groups.values()):
        raise RuntimeError(f"A6 camera contrast does not reach every required group: {active_groups}")
    result = {
        "pass": True,
        "group_id": group["id"],
        "view_indices": group["indices"],
        "target_lr_dropped": True,
        "latent_shape": list(clean.shape),
        "rank_shape": list(initial[3].shape),
        "per_view_shape": list(stage3.per_view_flow_losses(active[4], active[6]).shape),
        "initial_gradient_groups": initial_groups,
        "active_contrast_gradient_groups": active_groups,
        "config_sha256": item["config_sha256"],
        "manifest_sha256": item["manifest_sha256"],
    }
    prior.write_frozen_json(args.campaign_root / "preflight" / f"{name}.json", result)
    return result


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
            [
                args.python,
                str(Path(__file__)),
                "preflight",
                "--cell", name,
                "--campaign-root", str(args.campaign_root),
                "--gpu", str(args.gpu),
            ],
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
            args.python,
            str(args.repo_root / "scripts/stage3_experiment.py"),
            "train",
            "--config", str(config_path),
            *_runtime_args(args),
            "--seed", str(item["seed"]),
            "--output-dir", str(train_dir),
        ]
        checkpoints = sorted(train_dir.glob("stage3_step_*.pt"))
        if checkpoints:
            command.extend(("--resume", str(checkpoints[-1])))
        elif train_dir.exists() and any(train_dir.iterdir()):
            raise RuntimeError(f"{name}: incomplete run without a checkpoint; preserving artifacts")
        else:
            command.extend(("--init-checkpoint", str(args.phase_c_checkpoint), "--init-reset-fusion"))
        prior._run_logged(command, args=control, name=f"{name}_train")
    if not final.is_file() or not prior._complete_training(train_dir, 1000):
        raise RuntimeError(f"{name}: training incomplete")

    if not (eval_dir / "evaluation_summary.json").is_file():
        if eval_dir.exists() and any(eval_dir.iterdir()):
            raise RuntimeError(f"{name}: incomplete evaluation; preserving artifacts")
        prior._run_logged(
            [
                args.python,
                str(args.repo_root / "scripts/stage3_experiment.py"),
                "seen-eval",
                "--config", str(config_path),
                *_runtime_args(args),
                "--checkpoint", str(final),
                "--seen-manifest", str(manifest_path),
                "--subset", "probe",
                "--group-ids", *item["probe_ids"],
                "--inference-seeds", str(prior.INFERENCE_SEED),
                "--modes", *prior.MODES,
                "--save-diagnostics",
                "--save-images",
                "--output-dir", str(eval_dir),
            ],
            args=control,
            name=f"{name}_evaluate",
        )
    return analyze(args, name)


def analyze(args, name):
    protocol = prepare(args)
    item = protocol["cells"][name]
    config_path, manifest_path, train_dir, eval_dir = cell_paths(args, name)
    checkpoint = train_dir / "stage3_step_1000.pt"
    rows = prior.read_jsonl(eval_dir / "evaluation_rows.jsonl")
    train = prior.read_jsonl(train_dir / "train_steps.jsonl")
    baseline = prior.read_jsonl(eval_dir / "baseline_rows.jsonl")
    diagnostics = prior.read_jsonl(eval_dir / "diagnostics.jsonl")
    summary = prior.read_json(eval_dir / "evaluation_summary.json")
    manifest = prior.read_json(train_dir / "run_manifest.json")
    probes = tuple(item["probe_ids"])
    index = prior.evaluation_index(rows, probes)
    base = {row["group_id"]: row for row in baseline if row["condition"] == "bicubic"}
    gradient_keys = (
        "camera_rank_qk_gradient_norm",
        "camera_rank_value_gradient_norm",
        "camera_rank_output_gradient_norm",
        "camera_rank_bridge_gradient_norm",
    )
    telemetry = train[-100:]
    integrity = {
        "training_complete": len(train) == 1000
        and [row.get("step") for row in train] == list(range(1, 1001)),
        "evaluation_rows": len(rows) == 48 == summary.get("rows"),
        "baseline_rows": len(baseline) == 8 and len(base) == 4,
        "diagnostic_rows": len(diagnostics) == 44,
        "images": len(list((eval_dir / "images").rglob("*.png"))) == 64,
        "manifest_config": manifest.get("config")
        == json.loads(json.dumps(load_stage3_config(config_path).to_dict())),
        "manifest_parent": manifest.get("provenance", {}).get("parent_checkpoint_sha256")
        == protocol["phase_c_checkpoint_sha256"],
        "manifest_seed": manifest.get("provenance", {}).get("training_seed") == item["seed"],
        "no_pairing_loss": "pairing" not in manifest
        and all("pairing_loss" not in row for row in train),
        "target_rank_scope": all(
            row.get("camera_rank_scope") == "target_view_0" for row in train
        ),
        "per_view_telemetry": all(
            len(row.get("per_view_correct_flow_loss", [])) == 4
            and len(row.get("per_view_wrong_flow_loss", [])) == 4
            and all(
                math.isfinite(float(value))
                for key in ("per_view_correct_flow_loss", "per_view_wrong_flow_loss")
                for value in row[key]
            )
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
        "camera_scope": all(
            row["fusion_camera_changed"] and not row["geometry_camera_changed"]
            for row in diagnostics
            if row["condition"] == "shuffle_fusion"
        ),
        "hashes": summary.get("seen_manifest_sha256") == item["manifest_sha256"],
    }
    cell = prior.Cell(name, f"{name}.json", item["seed"], True)
    equivariance = prior.fusion_equivariance(
        SimpleNamespace(campaign_root=args.campaign_root, v2=True), cell
    )
    gate = prior.candidate_gate(
        rows,
        train,
        expected_steps=1000,
        reference=None,
        v2=True,
        fusion_equivariant=equivariance["pass"],
        probe_ids=probes,
    )
    bicubic_gain = sum(
        index[(group, "correct")]["psnr"] - base[group]["psnr"] for group in probes
    ) / len(probes)
    passed = bicubic_gain > 0 and gate["pass"] and all(integrity.values())
    result = {
        "status": "PILOT_PASS" if passed else "DOMAIN_FAIL" if bicubic_gain <= 0 else "HOLD",
        "pass": passed,
        "bicubic_psnr_gain": bicubic_gain,
        "gate": gate,
        "integrity": {"pass": all(integrity.values()), "checks": integrity},
        "fusion_equivariance": equivariance,
        "checkpoint_sha256": prior.sha256_file(checkpoint),
        "next": "RUN_LEGO_SEED43" if name == "a6_lego_seed42" and passed
        else "RUN_CHAIR_SEED42" if name == "a6_lego_seed43" and passed
        else "REVIEW_STAGE4" if name == "a6_chair_seed42" and passed
        else "A7_REQUIRED",
        "CAMERA_FUSION_PASS": passed,
        "STAGE4_READY": False,
    }
    prior.write_frozen_json(args.campaign_root / "analysis" / name / "summary.json", result)
    return result


def run(args):
    first = run_cell(args, "a6_lego_seed42")
    second = run_cell(args, "a6_lego_seed43") if first["pass"] else None
    third = run_cell(args, "a6_chair_seed42") if second and second["pass"] else None
    result = {
        "a6_lego_seed42": first["status"],
        "a6_lego_seed43": None if second is None else second["status"],
        "a6_chair_seed42": None if third is None else third["status"],
        "next": first["next"] if second is None else second["next"] if third is None else third["next"],
        "STAGE4_READY": False,
    }
    prior.write_frozen_json(args.campaign_root / "machine_verdict.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run-cell", "analyze", "run"))
    parser.add_argument("--cell", choices=tuple(CELLS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--campaign-root", type=Path, default=ROOT / "artifacts/stage3_3_a6_20260921")
    parser.add_argument("--phase-c-checkpoint", type=Path, default=ORIGINAL / "artifacts/stage3_2_camera_causality_20260913/phase_c/train/rank/seed42/stage3_step_1000.pt")
    parser.add_argument("--dataset-root", type=Path, default=ORIGINAL / "datasets/nerf_synthetic")
    parser.add_argument("--model-dir", type=Path, default=ORIGINAL / "models/Wan2.1-T2V-1.3B")
    parser.add_argument("--lq-source", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py"))
    parser.add_argument("--lq-checkpoint", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"))
    parser.add_argument("--bridge-checkpoint", type=Path, default=ORIGINAL / "artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt")
    parser.add_argument("--python", default=sys.executable)
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
        result = {
            "preflight": preflight,
            "run-cell": run_cell,
            "analyze": analyze,
        }[args.command](args, args.cell)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
