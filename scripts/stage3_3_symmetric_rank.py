#!/usr/bin/env python3
"""Run the fail-closed A6S symmetric camera/LR ranking campaign."""
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

import numpy as np
import torch

import stage3_experiment as stage3
import stage3_3_target_rank as a6
import stage3_3_ucpe_rre_fusion as prior
import stage4_readiness_review as review
from rl3dsr.validation.stage3_protocol import evenly_spaced_indices, load_stage3_config


ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "820265309749eff2c7c4d77a12e2d8b8c348ee04"
DEFAULT_ROOT = ROOT / "artifacts/stage3_3_a6s_20260922"
CELL_ORDER = tuple(a6.CELLS)
EXPECTED_TRACKED_CHANGES = {
    "scripts/stage3_experiment.py",
    "src/rl3dsr/models/wan/stage3.py",
    "src/rl3dsr/validation/stage3_protocol.py",
    "tests/test_stage3_protocol.py",
}
LITERATURE_SEARCH = {
    "date": "2026-09-22",
    "arxiv_query": 'all:"multi-view diffusion" AND (all:correspondence OR all:epipolar)',
    "arxiv_results": 21,
    "screened_sources": [
        "arXiv:2503.14463",
        "arXiv:2412.18565",
        "arXiv:2307.01097",
        "arXiv:2512.03045",
        "arXiv:2412.06614",
        "ICML:Radford2021CLIP",
        "ICCV:Zhang2017StackGAN",
        "CVPR:Wallace2024DiffusionDPO",
    ],
    "limitations": [
        "Crossref title search was broad/noisy",
        "OpenAlex and Semantic Scholar APIs were rate-limited",
    ],
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_bundle_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cell_paths(args, name):
    return a6.cell_paths(args, name)


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
        raise RuntimeError("A6S base revision mismatch")
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
    for name, (scene, seed, validation, test) in a6.CELLS.items():
        config_path, manifest_path, _, _ = cell_paths(args, name)
        config = replace(
            template,
            train_scenes=(scene,),
            validation_scenes=validation,
            test_scenes=test,
            training_seeds=(seed,),
            camera_rank_weight=0.5,
            symmetric_correspondence_rank=True,
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
        ROOT / "scripts/stage3_3_symmetric_rank.py",
        ROOT / "scripts/stage3_experiment.py",
        ROOT / "scripts/stage4_readiness_review.py",
        ROOT / "src/rl3dsr/models/wan/stage3.py",
        ROOT / "src/rl3dsr/validation/stage3_protocol.py",
        ROOT / "tests/test_stage3_3_symmetric_rank.py",
        ROOT / "tests/test_stage3_protocol.py",
    )
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=args.repo_root
    )
    protocol = {
        "schema_version": 1,
        "scope": "stage3_3_a6s_symmetric_rank",
        "base_revision": revision,
        "git_diff_sha256": _sha256_bytes(diff),
        "source_bundle_sha256": _source_bundle_sha256(source_paths),
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): prior.sha256_file(path)
            for path in source_paths
        },
        "phase_c_checkpoint_sha256": parent_hash,
        "matched_a5_sha256": {
            "chair_seed42": prior.sha256_file(args.a5_chair_summary),
            "lego_seed42": prior.sha256_file(args.a5_lego_summary),
        },
        "loss": {
            "flow_scope": "all_views",
            "rank_scope": "target_view_0",
            "rank_weight": 0.5,
            "camera_fraction": 0.5,
            "lr_fraction": 0.5,
            "margin_ratio": 0.05,
            "shared_auxiliary_permutation": True,
        },
        "literature_search": LITERATURE_SEARCH,
        "inference_seed": prior.INFERENCE_SEED,
        "modes": prior.MODES,
        "cell_order": CELL_ORDER,
        "cells": cells,
        "gate": {
            "shuffle_fusion_psnr": 0.03,
            "shuffle_fusion_ssim": 0.0003,
            "mispaired_lr_psnr": review.PSNR_CORRESPONDENCE,
            "mispaired_lr_ssim": review.SSIM_CORRESPONDENCE,
            "directional_probes": review.DIRECTIONAL_PROBES,
            "a6_minus_a5_psnr": review.PSNR_NONREGRESSION,
            "a6_minus_a5_ssim": review.SSIM_NONREGRESSION,
        },
        "forbidden": ["A7", "A8", "8 views", "10000 steps", "Stage 4 training", "4DSR"],
        "stage4_ready": False,
    }
    prior.write_frozen_json(args.campaign_root / "protocol.json", protocol)
    return protocol


def _rank_probe(runtime, config, lr, camera, clean, permutation, *, seed):
    wrong_camera = stage3._camera_with_auxiliary_permutation(camera, permutation)
    wrong_lr = stage3.permute_view_tensor(lr, permutation)
    prepared = runtime.module.prepare_multiview(
        lr, camera, tuple(clean.shape[2:]), (config.image_size, config.image_size)
    )
    camera_prepared = runtime.module.prepare_multiview(
        lr, wrong_camera, tuple(clean.shape[2:]), (config.image_size, config.image_size)
    )
    lr_prepared = runtime.module.prepare_multiview(
        wrong_lr, camera, tuple(clean.shape[2:]), (config.image_size, config.image_size)
    )
    generator = torch.Generator(device=runtime.device).manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator, device=runtime.device, dtype=clean.dtype)
    sigma = torch.tensor([0.5], device=runtime.device)
    noisy, timestep, target = stage3.flow_matching_pair(clean, noise, sigma)
    predictions = [
        runtime.module.predict(
            runtime.dit, noisy, timestep, None, value, camera, tuple(clean.shape[2:])
        )
        for value in (prepared, camera_prepared, lr_prepared)
    ]
    flow, e_correct, e_camera, camera_rank = stage3._camera_pair_training_losses(
        predictions[0], predictions[1], target,
        margin_ratio=config.camera_rank_margin_ratio, target_view_only=True,
    )
    _, _, e_lr, lr_rank = stage3._camera_pair_training_losses(
        predictions[0], predictions[2], target,
        margin_ratio=config.camera_rank_margin_ratio, target_view_only=True,
    )
    return {
        "flow": flow,
        "e_correct": e_correct,
        "e_camera": e_camera,
        "e_lr": e_lr,
        "camera_rank": camera_rank,
        "lr_rank": lr_rank,
        "combined_rank": stage3._symmetric_correspondence_rank(
            camera_rank, lr_rank, config.symmetric_camera_fraction
        ),
        "predictions": predictions,
        "target": target,
    }


def _gradient_groups(runtime, trainable, gradients, prefix):
    return {
        name.replace("camera_rank_", f"{prefix}_rank_"): value
        for name, value in stage3._camera_rank_gradient_groups(
            runtime.module, trainable, gradients
        ).items()
    }


def _preflight_prepared(args, name, protocol, paths):
    item = protocol["cells"][name]
    config_path, manifest_path, _, _ = paths
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
    permutation = stage3._deranged_auxiliary_indices(
        config.views, torch.Generator().manual_seed(6000)
    )
    trainable = [parameter for parameter in runtime.module.parameters() if parameter.requires_grad]
    initial = _rank_probe(runtime, config, lr, camera, clean, permutation, seed=3302)
    camera_gradients = torch.autograd.grad(
        initial["camera_rank"].mean(), trainable, retain_graph=True, allow_unused=True
    )
    lr_gradients = torch.autograd.grad(
        initial["lr_rank"].mean(), trainable, retain_graph=False, allow_unused=True
    )
    output_index = next(
        index for index, parameter in enumerate(trainable)
        if parameter is runtime.module.fusion.output.weight
    )
    output_gradient = sum(
        value for value in (camera_gradients[output_index], lr_gradients[output_index])
        if value is not None
    )
    if not torch.isfinite(output_gradient).all() or not output_gradient.any():
        raise RuntimeError("A6S zero-init output path has no symmetric ranking gradient")
    with torch.no_grad():
        scale = output_gradient.float().norm().clamp_min(1e-12)
        runtime.module.fusion.output.weight.add_(-1e-3 * output_gradient / scale)

    active = _rank_probe(runtime, config, lr, camera, clean, permutation, seed=3302)
    camera_active = torch.autograd.grad(
        (active["e_correct"] - active["e_camera"]).mean(),
        trainable, retain_graph=True, allow_unused=True,
    )
    lr_active = torch.autograd.grad(
        (active["e_correct"] - active["e_lr"]).mean(),
        trainable, retain_graph=False, allow_unused=True,
    )
    camera_groups = _gradient_groups(runtime, trainable, camera_active, "camera")
    lr_groups = _gradient_groups(runtime, trainable, lr_active, "lr")
    if any(not math.isfinite(value) or value <= 0 for value in (*camera_groups.values(), *lr_groups.values())):
        raise RuntimeError("A6S contrast does not reach every required gradient group")
    wrong_lr = stage3.permute_view_tensor(lr, permutation)
    if not torch.equal(lr[:, :, 0], wrong_lr[:, :, 0]) or torch.equal(lr[:, :, 1:], wrong_lr[:, :, 1:]):
        raise RuntimeError("A6S LR permutation did not preserve target and change auxiliaries")
    result = {
        "pass": True,
        "group_id": group["id"],
        "view_indices": group["indices"],
        "target_lr_dropped": True,
        "permutation": permutation.tolist(),
        "latent_shape": list(clean.shape),
        "rank_shape": list(initial["combined_rank"].shape),
        "initial_camera_gradient_groups": _gradient_groups(
            runtime, trainable, camera_gradients, "camera"
        ),
        "initial_lr_gradient_groups": _gradient_groups(
            runtime, trainable, lr_gradients, "lr"
        ),
        "active_camera_gradient_groups": camera_groups,
        "active_lr_gradient_groups": lr_groups,
        "correspondence_rank_weight": config.camera_rank_weight,
        "camera_rank_effective_weight": (
            config.camera_rank_weight * config.symmetric_camera_fraction
        ),
        "lr_rank_effective_weight": (
            config.camera_rank_weight * (1 - config.symmetric_camera_fraction)
        ),
        "frozen_wan": not any(p.requires_grad for p in runtime.dit.model.parameters()),
        "frozen_vae": not any(p.requires_grad for p in runtime.vae.model.model.parameters()),
        "config_sha256": item["config_sha256"],
        "manifest_sha256": item["manifest_sha256"],
    }
    prior.write_frozen_json(args.campaign_root / "preflight" / f"{name}.json", result)
    return result


def preflight(args, name):
    return _preflight_prepared(
        args, name, prepare(args), cell_paths(args, name)
    )


def run_cell(args, name):
    protocol = prepare(args)
    item = protocol["cells"][name]
    config_path, manifest_path, train_dir, eval_dir = cell_paths(args, name)
    control = SimpleNamespace(repo_root=args.repo_root, campaign_root=args.campaign_root, gpu=args.gpu)
    preflight_path = args.campaign_root / "preflight" / f"{name}.json"
    if not preflight_path.exists():
        prior._run_logged(
            [args.python, str(Path(__file__)), "preflight", "--cell", name,
             "--campaign-root", str(args.campaign_root), "--gpu", str(args.gpu)],
            args=control, name=f"{name}_preflight",
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
        raise RuntimeError(f"{name}: training incomplete")
    if not (eval_dir / "evaluation_summary.json").is_file():
        if eval_dir.exists() and any(eval_dir.iterdir()):
            raise RuntimeError(f"{name}: incomplete evaluation; preserving artifacts")
        prior._run_logged(
            [args.python, str(args.repo_root / "scripts/stage3_experiment.py"), "seen-eval",
             "--config", str(config_path), *a6._runtime_args(args),
             "--checkpoint", str(final), "--seen-manifest", str(manifest_path),
             "--subset", "probe", "--group-ids", *item["probe_ids"],
             "--inference-seeds", str(prior.INFERENCE_SEED), "--modes", *prior.MODES,
             "--save-diagnostics", "--save-images", "--output-dir", str(eval_dir)],
            args=control, name=f"{name}_evaluate",
        )
    return analyze(args, name)


def _matched_nonregression(args, item, gate):
    path = None
    key = None
    if item["scene"] == "lego" and item["seed"] == 42:
        path, key = args.a5_lego_summary, "lego_seed42"
    elif item["scene"] == "chair" and item["seed"] == 42:
        path, key = args.a5_chair_summary, "chair_seed42"
    if path is None:
        return None
    a5 = review._load_cell(path, arm="A5", scene=item["scene"], seed=item["seed"])
    if tuple(a5["probe_ids"]) != tuple(item["probe_ids"]):
        raise RuntimeError(f"{key}: matched A5/A6S probes differ")
    return review.nonregression(review._gate(a5["summary"])["correct"], gate["correct"])


def analyze(args, name):
    protocol = prepare(args)
    item = protocol["cells"][name]
    config_path, _, train_dir, eval_dir = cell_paths(args, name)
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
    telemetry = train[-100:]
    gradient_keys = tuple(
        f"{prefix}_rank_{group}_gradient_norm"
        for prefix in ("camera", "lr") for group in ("qk", "value", "output", "bridge")
    )
    integrity = {
        "training_complete": len(train) == 1000 and [row.get("step") for row in train] == list(range(1, 1001)),
        "evaluation_rows": len(rows) == 48 == summary.get("rows"),
        "baseline_rows": len(baseline) == 8 and len(base) == 4,
        "diagnostic_rows": len(diagnostics) == 44,
        "images": len(list((eval_dir / "images").rglob("*.png"))) == 64,
        "manifest_config": manifest.get("config") == json.loads(json.dumps(load_stage3_config(config_path).to_dict())),
        "manifest_parent": manifest.get("provenance", {}).get("parent_checkpoint_sha256") == protocol["phase_c_checkpoint_sha256"],
        "manifest_seed": manifest.get("provenance", {}).get("training_seed") == item["seed"],
        "symmetric_contract": all(
            row.get("camera_rank_scope") == "target_view_0"
            and row.get("lr_rank_scope") == "target_view_0"
            and len(row.get("per_view_lr_wrong_flow_loss", [])) == 4
            and math.isfinite(float(row.get("correspondence_rank_loss", float("nan"))))
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
        "hashes": summary.get("seen_manifest_sha256") == item["manifest_sha256"],
    }
    cell = prior.Cell(name, f"{name}.json", item["seed"], True)
    equivariance = prior.fusion_equivariance(
        SimpleNamespace(campaign_root=args.campaign_root, v2=True), cell
    )
    gate = prior.candidate_gate(
        rows, train, expected_steps=1000, reference=None, v2=True,
        fusion_equivariant=equivariance["pass"], probe_ids=probes,
    )
    correspondence = review.correspondence(gate)
    nonregression = _matched_nonregression(args, item, gate)
    bicubic_gain = sum(
        index[(group, "correct")]["psnr"] - base[group]["psnr"] for group in probes
    ) / len(probes)
    passed = (
        bicubic_gain > 0 and gate["pass"] and correspondence["pass"]
        and all(integrity.values()) and (nonregression is None or nonregression["pass"])
    )
    if passed:
        next_step = "RUN_LEGO_SEED43" if name == CELL_ORDER[0] else "RUN_CHAIR_SEED42" if name == CELL_ORDER[1] else "REVIEW_STAGE4"
    elif not correspondence["pass"]:
        next_step = "REVIEW_ATTENTION_CORRESPONDENCE"
    elif nonregression is not None and not nonregression["pass"]:
        next_step = "REVIEW_QUALITY_GRADIENT_CONFLICT"
    else:
        next_step = "HOLD"
    result = {
        "status": "PILOT_PASS" if passed else "DOMAIN_FAIL" if bicubic_gain <= 0 else "HOLD",
        "pass": passed,
        "bicubic_psnr_gain": bicubic_gain,
        "gate": gate,
        "symmetric_correspondence": correspondence,
        "matched_a5_nonregression": nonregression,
        "integrity": {"pass": all(integrity.values()), "checks": integrity},
        "fusion_equivariance": equivariance,
        "checkpoint_sha256": prior.sha256_file(checkpoint),
        "next": next_step,
        "CAMERA_FUSION_PASS": passed,
        "STAGE4_READY": False,
    }
    prior.write_frozen_json(args.campaign_root / "analysis" / name / "summary.json", result)
    return result


def _write_final_review(args, results):
    cells = {}
    matched = {}
    for name, result in results.items():
        if result is None:
            continue
        cells[name] = {
            "existing_a6_valid": result["gate"]["pass"] and result["integrity"]["pass"],
            "correspondence": result["symmetric_correspondence"],
        }
        if result["matched_a5_nonregression"] is not None:
            key = f"{a6.CELLS[name][0]}_seed{a6.CELLS[name][1]}"
            matched[key] = result["matched_a5_nonregression"]
    complete = len(cells) == len(CELL_ORDER)
    checks = {
        "all_cells_complete": complete,
        "a6_existing_gates": complete and all(value["existing_a6_valid"] for value in cells.values()),
        "matched_a5_nonregression": len(matched) == 2 and all(value["pass"] for value in matched.values()),
        "symmetric_correspondence": complete and all(value["correspondence"]["pass"] for value in cells.values()),
    }
    ready = all(checks.values())
    failed = next((value for value in results.values() if value is not None and not value["pass"]), None)
    payload = {
        "schema_version": 1,
        "verdict": "STAGE4_READY" if ready else "HOLD",
        "STAGE4_READY": ready,
        "next": "RUN_STAGE4" if ready else (failed or {}).get("next", "HOLD"),
        "thresholds": {
            "a6_minus_a5_psnr": review.PSNR_NONREGRESSION,
            "a6_minus_a5_ssim": review.SSIM_NONREGRESSION,
            "mispaired_lr_psnr": review.PSNR_CORRESPONDENCE,
            "mispaired_lr_ssim": review.SSIM_CORRESPONDENCE,
            "directional_probes": review.DIRECTIONAL_PROBES,
        },
        "checks": checks,
        "matched_nonregression": matched,
        "a6_cells": cells,
        "inputs": {str(args.campaign_root / "protocol.json"): prior.sha256_file(args.campaign_root / "protocol.json")},
    }
    output = args.campaign_root / "stage4_review"
    prior.write_frozen_json(output / "stage4_review.json", payload)
    output.mkdir(parents=True, exist_ok=True)
    markdown = review.render_markdown(payload).replace(
        "A6 已能对错误 pose 作出响应", "A6S 同时训练错误 pose 与错误辅助 LR"
    )
    path = output / "stage4_review.md"
    if path.exists() and path.read_text(encoding="utf-8") != markdown:
        raise RuntimeError("refusing to overwrite frozen Stage 4 review")
    path.write_text(markdown, encoding="utf-8")
    return payload


def run(args):
    results = {name: None for name in CELL_ORDER}
    for name in CELL_ORDER:
        result = run_cell(args, name)
        results[name] = result
        if not result["pass"]:
            break
    final_review = _write_final_review(args, results)
    verdict = {
        **{name: None if result is None else result["status"] for name, result in results.items()},
        "next": final_review["next"],
        "STAGE4_READY": final_review["STAGE4_READY"],
    }
    prior.write_frozen_json(args.campaign_root / "machine_verdict.json", verdict)
    return verdict


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
    parser.add_argument("--a5-chair-summary", type=Path, default=review.DEFAULT_A5_CHAIR)
    parser.add_argument("--a5-lego-summary", type=Path, default=review.DEFAULT_A5_LEGO)
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
        result = {"preflight": preflight, "run-cell": run_cell, "analyze": analyze}[
            args.command
        ](args, args.cell)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
