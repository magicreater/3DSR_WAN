#!/usr/bin/env python3
"""Freeze and run the final Stage 3 matched 4/8-view campaign."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

import stage3_experiment as stage3
from rl3dsr.validation.stage3_protocol import load_stage3_config


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/linzizhuo/RL3DSR_WAN_REMO")
OUT = ROOT / "artifacts/stage3_final_20260924"
PARENT = DATA / "artifacts/stage3_2_camera_causality_20260913/phase_c/train/rank/seed42/stage3_step_1000.pt"
BRIDGE = DATA / "artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt"
LQ_SOURCE = Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py")
LQ_CHECKPOINT = Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt")
GPUS = ((1, "GPU-e2eaf00a-e9b0-b681-d6f0-c9597f5c418d"),
        (3, "GPU-52673cef-d39f-fbe8-bdc0-bfda70852e4b"))
MODES = ("correct", "correct_repeat", "remove", "target_drop", "shuffle_fusion",
         "target_drop_shuffle_fusion", "mispaired_lr", "mispaired_camera",
         "aux_permute", "joint_permute", "fusion_camera_dose_half",
         "fusion_camera_dose_full", "shuffle_geometry")
SOURCE = ("scripts/stage3_experiment.py", "scripts/stage3_final_campaign.py",
          "scripts/stage3_final_preflight.py", "scripts/stage3_final_review.py",
          "scripts/stage3_final_readiness.py",
          "src/rl3dsr/models/wan/lr_fusion.py", "src/rl3dsr/models/wan/stage3.py",
          "src/rl3dsr/models/wan/dit.py", "src/rl3dsr/models/wan/lq_conditioning.py",
          "src/rl3dsr/validation/stage3_protocol.py", "docs/PROJECT_SPEC.md")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise RuntimeError(f"frozen artifact changed: {path}")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(data)
    return sha(path)


def cell_name(kind, views, scene="lego", seed=42):
    return f"{kind}_v{views}_{scene}_s{seed}"


def paths(name):
    return (OUT / "config" / f"{name}.json", OUT / "manifest" / f"{name}.json",
            OUT / "train" / name, OUT / "eval" / name)


def config_for(kind, views, scene, seed):
    if kind == "new":
        template = load_stage3_config(Path(
            "/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-target-rank/artifacts/"
            "stage3_3_a6sw3_20260923/config/w3_lego_seed42.json"
        ))
        return replace(template, views=views, steps=4000, train_scenes=(scene,),
                       validation_scenes=(("chair",) if scene == "lego" else ("lego",)),
                       test_scenes=("drums",), training_seeds=(seed,),
                       dynamic_fusion=True, shared_multiview_rope=True)
    if kind == "a5":
        template = load_stage3_config(Path(
            "/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-verified/artifacts/"
            "stage3_3_v2_20260917/config/A5_pairing_chair_1000_v2.json"
        ))
        return replace(template, views=views, steps=4000, train_scenes=(scene,),
                       validation_scenes=(("chair",) if scene == "lego" else ("lego",)),
                       test_scenes=("drums",), training_seeds=(seed,))
    raise ValueError(kind)


def prepare(names, *, protocol_path=None):
    if sha(PARENT) != "49730b5f4fcb12009cc3d249a5814884e082b404ba482ec746a23715980cdd47":
        raise RuntimeError("parent checkpoint drift")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, text=True).strip():
        raise RuntimeError("commit tracked source before freezing campaign")
    cells = {}
    for name in names:
        kind, v, scene, s = name.split("_")
        views, seed = int(v[1:]), int(s[1:])
        config = config_for(kind, views, scene, seed)
        config_path, manifest_path, _, _ = paths(name)
        config_hash = frozen(config_path, config.to_dict())
        if not manifest_path.exists():
            stage3._prepare_seen_manifest(
                SimpleNamespace(dataset_root=DATA / "datasets/nerf_synthetic", manifest=manifest_path), config)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["sampling_signature"] != stage3._sampling_signature(config):
            raise RuntimeError(f"sampling signature drift: {name}")
        cells[name] = {"kind": kind, "views": views, "scene": scene, "seed": seed,
                       "config_sha256": config_hash, "manifest_sha256": sha(manifest_path),
                       "probes": manifest["subsets"]["probe"], "dataset": manifest["datasets"]}
    # All arms use the same four anchors, and V4 auxiliaries prefix V8.
    by_scene_seed = {}
    for name, cell in cells.items():
        key = (cell["scene"], cell["seed"])
        by_scene_seed.setdefault(key, []).append((name, cell))
    for group in by_scene_seed.values():
        for name, cell in group:
            manifest = json.loads(paths(name)[1].read_text())
            for other_name, other in group:
                other_manifest = json.loads(paths(other_name)[1].read_text())
                assert cell["probes"] == other["probes"]
                if cell["views"] == 4 and other["views"] == 8:
                    indexed = {row["id"]: row["indices"] for row in other_manifest["groups"]}
                    if any(row["indices"] != indexed[row["id"]][:4] for row in manifest["groups"]):
                        raise RuntimeError("4/8 view manifest is not nested")
    schedule_hashes = {}
    for (scene, seed), group in by_scene_seed.items():
        widest = max(cell["views"] for _, cell in group)
        config = config_for("new", widest, scene, seed)
        _, sequence = stage3._scene_data(DATA / "datasets/nerf_synthetic" / scene,
                                         config, stage3.Split.TRAIN)
        camera = stage3._camera(sequence.observations, config.image_size, torch.device("cpu"))
        generator = torch.Generator().manual_seed(seed + 2000)
        sampled = []
        for step in range(1, 4001):
            for micro in range(config.gradient_accumulation):
                anchor = int(torch.randint(len(sequence.observations), (), generator=generator))
                indices = stage3.sample_view_indices(
                    camera, anchor, generator, view_count=widest,
                    nearest=min(config.nearest_views, len(sequence.observations) - 1))
                sampled.append({"step": step, "micro": micro, "wide": indices,
                                "v4": indices[:4]})
        schedule_hashes[f"{scene}_s{seed}"] = frozen(
            OUT / "schedule" / f"{scene}_s{seed}.json", sampled)
    payload = {"schema": 1, "source_commit": revision,
               "source_sha256": {p: sha(ROOT / p) for p in SOURCE},
               "parent_checkpoint_sha256": sha(PARENT), "bridge_checkpoint_sha256": sha(BRIDGE),
               "lq_checkpoint_sha256": sha(LQ_CHECKPOINT),
               "wan_model_dir": str(DATA / "models/Wan2.1-T2V-1.3B"),
               "dataset_root": str(DATA / "datasets/nerf_synthetic"),
               "inference_seed": 3302, "modes": MODES, "cells": cells,
               "training_schedule_sha256": schedule_hashes,
               "eligible_gpus": GPUS, "STAGE4_READY": False}
    protocol = protocol_path or OUT / "protocol.json"
    if protocol.exists():
        previous = json.loads(protocol.read_text())
        if previous["source_commit"] != revision or previous["source_sha256"] != payload["source_sha256"]:
            raise RuntimeError("campaign source drift")
        if any(previous["cells"].get(name) != cell for name, cell in cells.items()):
            raise RuntimeError("campaign cell drift")
        if set(previous["cells"]) != set(cells):
            raise RuntimeError("campaign cell set drift")
    else:
        frozen(protocol, payload)
    return payload


def idle_gpu():
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True).splitlines()
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    for index, uuid in GPUS:
        selected = [row for row in rows if uuid in row]
        if (len(selected) == 1
                and int(selected[0].split(",")[-1].strip().split()[0]) < 100
                and uuid not in processes):
            return index, uuid
    return None


def run_command(command, label):
    log = OUT / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise RuntimeError(f"existing log needs audit before retry: {log}")
    selected = idle_gpu()
    while selected is None:
        print(f"both eligible GPUs occupied; waiting before {label}", flush=True)
        time.sleep(60)
        selected = idle_gpu()
    frozen(OUT / "logs" / f"{label}_gpu.json", {"index": selected[0], "uuid": selected[1]})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(selected[0]), "PYTHONPATH": str(ROOT / "src")}
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"{label} failed ({process.returncode}); inspect {log}")


def runtime_args():
    return ["--dataset-root", str(DATA / "datasets/nerf_synthetic"),
            "--model-dir", str(DATA / "models/Wan2.1-T2V-1.3B"),
            "--lq-source", str(LQ_SOURCE), "--lq-checkpoint", str(LQ_CHECKPOINT),
            "--bridge-checkpoint", str(BRIDGE)]


def run_cell(name, protocol):
    cell = protocol["cells"][name]
    config_path, manifest_path, train_dir, eval_dir = paths(name)
    config = load_stage3_config(config_path)
    if sha(config_path) != cell["config_sha256"] or sha(manifest_path) != cell["manifest_sha256"]:
        raise RuntimeError(f"frozen cell drift: {name}")
    final = train_dir / "stage3_step_4000.pt"
    if not final.exists():
        checkpoints = sorted(train_dir.glob("stage3_step_*.pt"))
        if train_dir.exists() and any(train_dir.iterdir()) and not checkpoints:
            raise RuntimeError(f"incomplete run without checkpoint: {name}")
        command = [sys.executable, str(ROOT / "scripts/stage3_experiment.py"), "train",
                   "--config", str(config_path), *runtime_args(), "--seed", str(cell["seed"]),
                   "--output-dir", str(train_dir)]
        if checkpoints:
            command.extend(("--resume", str(checkpoints[-1])))
        else:
            command.extend(("--init-checkpoint", str(PARENT), "--init-reset-fusion"))
        run_command(command, name + (f"_train_resume{checkpoints[-1].stem.rsplit('_', 1)[1]}"
                                     if checkpoints else "_train_initial"))
    rows = [json.loads(line) for line in (train_dir / "train_steps.jsonl").read_text().splitlines()]
    if len(rows) != 4000 or [r.get("step") for r in rows] != list(range(1, 4001)):
        raise RuntimeError(f"training not complete: {name}")
    if not (eval_dir / "evaluation_summary.json").exists():
        if eval_dir.exists() and any(eval_dir.iterdir()):
            raise RuntimeError(f"incomplete evaluation: {name}")
        command = [sys.executable, str(ROOT / "scripts/stage3_experiment.py"), "seen-eval",
                   "--config", str(config_path), *runtime_args(), "--checkpoint", str(final),
                   "--seen-manifest", str(manifest_path), "--subset", "probe", "--group-ids",
                   *cell["probes"], "--inference-seeds", "3302", "--modes", *MODES,
                   "--save-diagnostics", "--save-images", "--output-dir", str(eval_dir)]
        run_command(command, name + "_eval")
    print(f"COMPLETE {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "replicate"))
    args = parser.parse_args()
    initial = tuple(cell_name(kind, views) for views in (4, 8) for kind in ("a5", "new"))
    if args.command == "prepare":
        print(json.dumps(prepare(initial), indent=2), flush=True)
        return
    if args.command == "replicate":
        review = json.loads((OUT / "analysis/four_arm_review.json").read_text())
        views = review.get("selected_views")
        if review.get("status") != "STAGE3_REPLICATION_REQUIRED" or views not in (4, 8):
            raise RuntimeError("four-arm gate has no selected configuration")
        names = tuple(cell_name(kind, views, scene, seed)
                      for scene, seed in (("lego", 43), ("chair", 42))
                      for kind in ("a5", "new"))
        protocol_path = OUT / "replication_protocol.json"
        replication = prepare(names, protocol_path=protocol_path)
        for name in names:
            run_cell(name, replication)
        subprocess.run([sys.executable, str(ROOT / "scripts/stage3_final_readiness.py")],
                       cwd=ROOT, check=True)
        return
    protocol = json.loads((OUT / "protocol.json").read_text())
    if set(protocol["cells"]) != set(initial):
        raise RuntimeError("run protocol must contain all four initial cells")
    for name in initial:
        run_cell(name, protocol)
    subprocess.run([sys.executable, str(ROOT / "scripts/stage3_final_review.py")],
                   cwd=ROOT, check=True)
    review = json.loads((OUT / "analysis/four_arm_review.json").read_text())
    if review.get("status") == "STAGE3_REPLICATION_REQUIRED":
        subprocess.run([sys.executable, str(Path(__file__)), "replicate"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
