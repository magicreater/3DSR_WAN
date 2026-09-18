#!/usr/bin/env python3
"""Bounded A5 data control: Synthetic lego versus Mip-NeRF 360 counter."""
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
from rl3dsr.models.wan.lr_fusion import pairing_info_nce_loss
from rl3dsr.validation.stage3_protocol import evenly_spaced_indices, load_stage3_config


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path("/data/linzizhuo/RL3DSR_WAN_REMO")
SOURCE = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-verified/artifacts/stage3_3_v2_20260917")
CELLS = {
    "lego_seed42": ("lego", "nerf_synthetic", 1, 42, ("chair",), ("drums",)),
    "counter_seed42": ("counter", "mipnerf360", 4, 42, ("kitchen",), ("garden",)),
    "counter_seed43": ("counter", "mipnerf360", 4, 43, ("kitchen",), ("garden",)),
}


def paths(args, name):
    scene, kind, _, _, _, _ = CELLS[name]
    dataset = args.synthetic_root if kind == "nerf_synthetic" else args.mip_root
    return (
        dataset,
        args.campaign_root / "config" / f"{name}.json",
        args.campaign_root / "manifest" / f"{name}.json",
        args.campaign_root / "train" / name,
        args.campaign_root / "eval" / name,
    )


def prepare(args):
    source = load_stage3_config(args.source_config)
    if source.arm != "A5" or source.steps != 1000:
        raise ValueError("source must be frozen A5-v2 1000-step config")
    parent_hash = prior.sha256_file(args.phase_c_checkpoint)
    if parent_hash != prior.PHASE_C_CHECKPOINT_SHA256:
        raise RuntimeError("Phase C parent checkpoint hash mismatch")
    cells = {}
    for name, (scene, kind, factor, seed, validation, test) in CELLS.items():
        dataset, config_path, manifest_path, _, _ = paths(args, name)
        config = replace(
            source, train_scenes=(scene,), validation_scenes=validation,
            test_scenes=test, training_seeds=(seed,),
            dataset_kind=kind, image_factor=factor,
        )
        prior.write_frozen_json(config_path, config.to_dict())
        if not manifest_path.exists():
            stage3._prepare_seen_manifest(
                SimpleNamespace(dataset_root=dataset, manifest=manifest_path), config
            )
        manifest = prior.read_json(manifest_path)
        if (manifest["dataset_root"] != str(dataset.resolve())
                or manifest["sampling_signature"] != stage3._sampling_signature(config)
                or manifest.get("dataset_kind", "nerf_synthetic") != kind
                or manifest.get("image_factor", 1) != factor):
            raise RuntimeError(f"{name} manifest does not match frozen config/data")
        probe_ids = tuple(manifest["subsets"]["probe"])
        count = manifest["datasets"][scene]["views"]
        if len(probe_ids) != 4 or probe_ids != tuple(
            f"{scene}:{index:03d}" for index in evenly_spaced_indices(count, 4)
        ):
            raise RuntimeError(f"{name} probes are not the frozen four anchors")
        cells[name] = {
            "scene": scene, "dataset_kind": kind, "image_factor": factor,
            "seed": seed, "dataset_root": str(dataset.resolve()),
            "config": str(config_path.resolve()), "config_sha256": prior.sha256_file(config_path),
            "seen_manifest": str(manifest_path.resolve()),
            "seen_manifest_sha256": prior.sha256_file(manifest_path),
            "probe_ids": probe_ids,
        }
    protocol = {
        "scope": "stage3_3_data_control", "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True
        ).strip(),
        "source_config_sha256": prior.sha256_file(args.source_config),
        "phase_c_checkpoint_sha256": parent_hash,
        "inference_seed": prior.INFERENCE_SEED,
        "modes": prior.MODES, "cells": cells,
        "gate": {"shuffle_fusion_psnr": 0.03, "shuffle_fusion_ssim": 0.0003,
                 "directional_probes": 3, "dose_monotonic_probes": 3},
        "stage4_ready": False,
    }
    prior.write_frozen_json(args.campaign_root / "protocol.json", protocol)
    return protocol


def preflight(args, name):
    protocol = prepare(args)
    prior._gpu_record(args.gpu)
    item = protocol["cells"][name]
    dataset, config_path, manifest_path, _, _ = paths(args, name)
    config = load_stage3_config(config_path)
    manifest = prior.read_json(manifest_path)
    groups = {row["id"]: row for row in manifest["groups"]}
    runtime = stage3.load_runtime(
        config, model_dir=args.model_dir, lq_source=args.lq_source,
        lq_checkpoint=args.lq_checkpoint, bridge_checkpoint=args.bridge_checkpoint,
        stage3_checkpoint=args.phase_c_checkpoint, model_only_initialization=True,
        reset_initialization_fusion=True, device="cuda",
    )
    runtime.module.train()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    results = []
    for offset, group_id in enumerate(item["probe_ids"]):
        group = groups[group_id]
        _, hr, lr, camera = stage3._load_indices(
            dataset, item["scene"], "train", group["indices"], config, runtime.device
        )
        camera.validate(batch=1)
        with torch.no_grad():
            clean = runtime.vae.encode_multiview(hr)
        wrong = stage3.derange_auxiliary_fusion_camera(
            camera, torch.Generator().manual_seed(6000 + offset)
        )
        prepared, state = runtime.module.prepare_multiview(
            lr, camera, tuple(clean.shape[2:]), (config.image_size, config.image_size),
            pairing_camera=wrong, pairing_minimum_coverage=config.pairing_minimum_coverage,
        )
        if prepared.shape[0] != 1 or len(state["pairs"]) != config.views * (config.views - 1):
            raise RuntimeError(f"{group_id}: wrong prepared shape or pair count")
        coverage = min(float(row["coverage"]) for row in state["pairs"])
        loss = pairing_info_nce_loss(state, temperature=config.pairing_temperature)
        gradients = torch.autograd.grad(loss, tuple(runtime.module.fusion.parameters()), allow_unused=True)
        gradient_norm = sum(float(g.detach().float().square().sum()) for g in gradients if g is not None) ** 0.5
        if not math.isfinite(gradient_norm) or gradient_norm <= 0 or coverage < 0.05:
            raise RuntimeError(f"{group_id}: pairing coverage or gradient preflight failed")
        results.append({
            "group_id": group_id, "view_indices": group["indices"],
            "minimum_pair_coverage": coverage, "pairing_gradient_norm": gradient_norm,
            "latent_shape": list(clean.shape), "prepared_shape": list(prepared.shape),
        })
    result = {
        "pass": True, "config_sha256": item["config_sha256"],
        "seen_manifest_sha256": item["seen_manifest_sha256"], "probes": results,
    }
    prior.write_frozen_json(args.campaign_root / "preflight" / f"{name}.json", result)
    return result


def runtime_args(args, dataset):
    return [
        "--dataset-root", str(dataset), "--model-dir", str(args.model_dir),
        "--lq-source", str(args.lq_source), "--lq-checkpoint", str(args.lq_checkpoint),
        "--bridge-checkpoint", str(args.bridge_checkpoint), "--device", "cuda",
    ]


def run_cell(args, name):
    protocol = prepare(args)
    item = protocol["cells"][name]
    dataset, config_path, manifest_path, train_dir, eval_dir = paths(args, name)
    control = SimpleNamespace(repo_root=args.repo_root, campaign_root=args.campaign_root, gpu=args.gpu)
    preflight_path = args.campaign_root / "preflight" / f"{name}.json"
    if not preflight_path.exists():
        prior._run_logged(
            [sys.executable, str(Path(__file__)), "preflight", "--cell", name,
             "--campaign-root", str(args.campaign_root), "--gpu", str(args.gpu)],
            args=control, name=f"{name}_preflight",
        )
    pre = prior.read_json(preflight_path)
    if not pre.get("pass") or pre.get("config_sha256") != item["config_sha256"]:
        raise RuntimeError(f"{name}: preflight invalid")
    final = train_dir / "stage3_step_1000.pt"
    complete = final.is_file() and prior._complete_training(train_dir, 1000)
    if not complete:
        command = [
            sys.executable, str(args.repo_root / "scripts/stage3_experiment.py"),
            "train", "--config", str(config_path), *runtime_args(args, dataset),
            "--seed", str(item["seed"]), "--output-dir", str(train_dir),
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
        raise RuntimeError(f"{name}: 1000-step training incomplete")
    summary = eval_dir / "evaluation_summary.json"
    if not summary.is_file():
        if eval_dir.exists() and any(eval_dir.iterdir()):
            raise RuntimeError(f"{name}: incomplete evaluation; preserving artifacts")
        prior._run_logged([
            sys.executable, str(args.repo_root / "scripts/stage3_experiment.py"),
            "seen-eval", "--config", str(config_path), *runtime_args(args, dataset),
            "--checkpoint", str(final), "--seen-manifest", str(manifest_path),
            "--subset", "probe", "--group-ids", *item["probe_ids"],
            "--inference-seeds", str(prior.INFERENCE_SEED),
            "--modes", *prior.MODES, "--save-diagnostics", "--save-images",
            "--output-dir", str(eval_dir),
        ], args=control, name=f"{name}_evaluate")
    return analyze(args, name)


def analyze(args, name):
    protocol = prepare(args)
    item = protocol["cells"][name]
    _, config_path, manifest_path, train_dir, eval_dir = paths(args, name)
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
    if (len(base) != 4 or len(rows) != 48 or len(baseline) != 8
            or len(diagnostics) != 44 or summary["rows"] != 48
            or summary["inference_seeds"] != [prior.INFERENCE_SEED]
            or summary["seen_manifest_sha256"] != item["seen_manifest_sha256"]
            or manifest["config"] != json.loads(json.dumps(load_stage3_config(config_path).to_dict()))
            or manifest["provenance"]["parent_checkpoint_sha256"] != protocol["phase_c_checkpoint_sha256"]
            or not prior._complete_training(train_dir, 1000)):
        raise RuntimeError(f"{name}: artifact integrity failed")
    if any(
        len({row["sample_seed"] for row in diagnostics if row["group_id"] == group}) != 1
        for group in probes
    ):
        raise RuntimeError(f"{name}: paired noise mismatch")
    shuffled = [row for row in diagnostics if row["condition"] == "shuffle_fusion"]
    if len(shuffled) != 4 or not all(
        row["fusion_camera_changed"] and not row["geometry_camera_changed"] for row in shuffled
    ):
        raise RuntimeError(f"{name}: camera intervention scope mismatch")
    eq_file = args.campaign_root / "analysis" / name / "fusion_equivariance.json"
    if not eq_file.exists():
        prior.fusion_equivariance(
            SimpleNamespace(campaign_root=args.campaign_root, v2=False),
            prior.Cell(name, f"{name}.json", item["seed"], True),
        )
    eq = prior.read_json(eq_file)
    gate = prior.candidate_gate(
        rows, train, expected_steps=1000, reference=None, v2=True,
        fusion_equivariant=eq["pass"], probe_ids=probes,
    )
    bicubic_gain = sum(index[(group, "correct")]["psnr"] - base[group]["psnr"] for group in probes) / 4
    status = "DOMAIN_FAIL" if bicubic_gain <= 0 else "PILOT_PASS" if gate["pass"] else "HOLD"
    result = {
        "status": status, "gate": gate, "bicubic_psnr_gain": bicubic_gain,
        "fusion_equivariance": eq, "checkpoint_sha256": prior.sha256_file(checkpoint),
        "config_sha256": item["config_sha256"],
        "seen_manifest_sha256": item["seen_manifest_sha256"],
        "target_feature_delta_mean": sum(row["view_relative_delta"][0] for row in shuffled) / 4,
        "aux_feature_delta_mean": sum(v for row in shuffled for v in row["view_relative_delta"][1:]) / 12,
        "velocity_delta_mean": sum(row["per_step_velocity_delta_relative_mean"] for row in shuffled) / 4,
    }
    prior.write_frozen_json(args.campaign_root / "analysis" / name / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run-cell", "analyze", "run"))
    parser.add_argument("--cell", choices=tuple(CELLS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--campaign-root", type=Path, default=ROOT / "artifacts/stage3_3_data_control_20260918")
    parser.add_argument("--source-config", type=Path, default=SOURCE / "config/A5_pairing_chair_1000_v2.json")
    parser.add_argument("--phase-c-checkpoint", type=Path, default=ORIGINAL / "artifacts/stage3_2_camera_causality_20260913/phase_c/train/rank/seed42/stage3_step_1000.pt")
    parser.add_argument("--synthetic-root", type=Path, default=ORIGINAL / "datasets/nerf_synthetic")
    parser.add_argument("--mip-root", type=Path, default=ORIGINAL / "datasets/360_v2")
    parser.add_argument("--model-dir", type=Path, default=ORIGINAL / "models/Wan2.1-T2V-1.3B")
    parser.add_argument("--lq-source", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py"))
    parser.add_argument("--lq-checkpoint", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"))
    parser.add_argument("--bridge-checkpoint", type=Path, default=ORIGINAL / "artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt")
    args = parser.parse_args()
    args.repo_root = ROOT
    args.campaign_root = args.campaign_root.resolve()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "run":
        lego = run_cell(args, "lego_seed42")
        counter = run_cell(args, "counter_seed42")
        repeat = (
            run_cell(args, "counter_seed43")
            if counter["status"] == "PILOT_PASS" and lego["status"] != "PILOT_PASS"
            else None
        )
        result = {"lego": lego["status"], "counter": counter["status"],
                  "counter_seed43": None if repeat is None else repeat["status"]}
    else:
        if args.cell is None:
            parser.error(f"{args.command} requires --cell")
        result = {
            "preflight": preflight, "run-cell": run_cell, "analyze": analyze,
        }[args.command](args, args.cell)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
