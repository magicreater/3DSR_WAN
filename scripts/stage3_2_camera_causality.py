#!/usr/bin/env python3
"""Audit camera-pair intervention strength before any further Stage 3 training."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from statistics import mean

import torch

from rl3dsr.data import NeRFSyntheticAdapter, Split
from rl3dsr.models.wan.geometry_conditioning import CameraBatch
from rl3dsr.models.wan.lr_fusion import (
    epipolar_local_key_mask,
    patch_fundamental_matrices,
)
from rl3dsr.validation.stage3_protocol import intervene_lr, load_stage3_config


INFERENCE_SEED = 3302
PROBE_IDS = ("chair:000", "chair:033", "chair:066", "chair:099")
MODES = (
    "correct",
    "correct_repeat",
    "shuffle_fusion",
    "far_shuffle_fusion",
    "no_self_correct",
    "no_self_shuffle_fusion",
    "no_self_far_shuffle_fusion",
)
METRICS = ("psnr", "ssim", "lpips", "mae")
GOOD_DIRECTION = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0, "mae": -1.0}
CELLS = {
    "v4": ("A3_views4_local_band_chair_2000.json", "v4_seen_groups.json"),
    "v8": ("A3_views8_local_band_chair_2000.json", "v8_seen_groups.json"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_text(payload), encoding="utf-8")
    temporary.replace(path)


def write_frozen_json(path: Path, payload: object) -> None:
    text = _json_text(payload)
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"frozen artifact drift: {path}")
        return
    write_json(path, payload)


def write_frozen_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows)
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"frozen artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def copy_or_verify(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"frozen snapshot drift: {destination}")
        return
    shutil.copy2(source, destination)


def _git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _origin_master_revision(root: Path) -> str:
    output = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/master"], cwd=root, text=True
    ).strip()
    if not output:
        raise RuntimeError("origin/master does not resolve")
    return output.split()[0]


def _camera(observations, resolution: int, device: torch.device) -> CameraBatch:
    intrinsics, transforms = [], []
    for observation in observations:
        k = torch.from_numpy(observation.K.copy()).float()
        k[0] *= resolution / observation.width
        k[1] *= resolution / observation.height
        intrinsics.append(k)
        transforms.append(torch.from_numpy(observation.T_world_from_camera.copy()).float())
    return CameraBatch(
        torch.stack(intrinsics).unsqueeze(0).to(device),
        torch.stack(transforms).unsqueeze(0).to(device),
        (resolution, resolution),
        "multiview",
    )


def _subset_camera(camera: CameraBatch, indices: list[int]) -> CameraBatch:
    return CameraBatch(
        camera.K[:, indices],
        camera.T_world_from_camera[:, indices],
        camera.image_size,
        camera.sequence_kind,
        reference_index=0,
    )


def select_far_donors(camera: CameraBatch, group_indices: list[int]) -> dict:
    """Greedily choose unique cameras farthest from each auxiliary optical axis."""
    camera.validate(batch=1)
    if camera.sequence_kind != "multiview" or len(group_indices) < 2:
        raise ValueError("far donors require a multiview group with auxiliaries")
    count = camera.K.shape[1]
    if len(set(group_indices)) != len(group_indices) or any(not 0 <= value < count for value in group_indices):
        raise ValueError("group indices must be unique and in range")
    available = set(range(count)) - set(group_indices)
    if len(available) < len(group_indices) - 1:
        raise ValueError("not enough cameras outside the source group")
    axes = camera.T_world_from_camera[0, :, :3, 2].detach().float().cpu()
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    selected, angles = [], []
    for source in group_indices[1:]:
        ranked = []
        for candidate in available:
            dot = float((axes[source] * axes[candidate]).sum().clamp(-1, 1))
            angle = math.degrees(math.acos(dot))
            ranked.append((-angle, candidate, angle))
        _, donor, angle = min(ranked)
        selected.append(donor)
        angles.append(angle)
        available.remove(donor)
    return {"indices": selected, "angles_deg": angles}


def mask_comparison(correct: torch.Tensor, wrong: torch.Tensor) -> dict[str, float]:
    if correct.shape != wrong.shape or correct.dtype != torch.bool or wrong.dtype != torch.bool:
        raise ValueError("candidate masks must be matching booleans")
    intersection = int((correct & wrong).sum())
    union = int((correct | wrong).sum())
    jaccard = 1.0 if union == 0 else intersection / union
    return {"mask_jaccard": jaccard, "mask_churn": 1.0 - jaccard}


def _mask_stats(camera: CameraBatch, patch_grid: tuple[int, int], band: float) -> tuple[torch.Tensor, dict]:
    allowed, usable = epipolar_local_key_mask(camera, patch_grid, band=band)
    matrices, valid = patch_fundamental_matrices(camera, patch_grid)
    batch, queries, keys = allowed.shape
    views = camera.K.shape[1]
    patches = keys // views
    query_views = torch.arange(queries, device=allowed.device) // patches
    query_patches = torch.arange(queries, device=allowed.device) % patches
    source_views = torch.arange(views, device=allowed.device)
    cross_pairs = query_views[:, None] != source_views[None]
    cross_keys = cross_pairs.repeat_interleave(patches, dim=1)
    cross_allowed = allowed & cross_keys[None]
    by_source = allowed.reshape(batch, queries, views, patches)
    usable_cross = usable & cross_pairs[None]
    nonempty = by_source.any(dim=-1)
    gh, gw = patch_grid
    query_pixels = torch.stack((
        query_patches.remainder(gw).float() + 0.5,
        query_patches.div(gw, rounding_mode="floor").float() + 0.5,
        torch.ones_like(query_patches, dtype=torch.float32),
    ), dim=-1)
    lines = torch.einsum("bqvij,qj->bqvi", matrices[:, query_views], query_pixels)
    corners = lines.new_tensor(((0, 0, 1), (gw, 0, 1), (0, gh, 1), (gw, gh, 1)))
    corner_values = torch.einsum("bqvi,ci->bqvc", lines, corners)
    intersects_extent = (
        corner_values.amin(dim=-1) <= 0
    ) & (corner_values.amax(dim=-1) >= 0)
    in_bounds_cross = usable_cross & intersects_extent
    in_bounds = (
        float(nonempty[in_bounds_cross].float().mean())
        if bool(in_bounds_cross.any())
        else 0.0
    )
    pair_cross = ~torch.eye(views, dtype=torch.bool, device=valid.device)
    stats = {
        "valid_pair_ratio": float(valid[:, pair_cross].float().mean()),
        "usable_query_pair_ratio": float(
            usable_cross.float().sum() / (batch * cross_pairs.sum()).clamp_min(1)
        ),
        "in_bounds_query_pair_ratio": float(
            in_bounds_cross.float().sum() / (batch * cross_pairs.sum()).clamp_min(1)
        ),
        "in_bounds_query_nonempty_ratio": in_bounds,
        "zero_key_query_pair_count": int((in_bounds_cross & ~nonempty).sum()),
        "cross_key_retention": float(cross_allowed.float().sum() / (batch * cross_keys.sum()).clamp_min(1)),
        "cross_keys_per_query": float(cross_allowed.sum() / max(batch * queries, 1)),
    }
    return cross_allowed, stats


def _angle_deltas(camera: CameraBatch, changed: CameraBatch) -> list[float]:
    left = camera.T_world_from_camera[0, 1:, :3, 2].detach().float().cpu()
    right = changed.T_world_from_camera[0, 1:, :3, 2].detach().float().cpu()
    left = left / left.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return [math.degrees(math.acos(float(value.clamp(-1, 1)))) for value in (left * right).sum(-1)]


def output_gate(deltas: list[dict]) -> dict:
    if len(deltas) != len(PROBE_IDS):
        raise ValueError("output gate requires four paired probes")
    averages = {metric: mean(float(row[metric]) for row in deltas) for metric in METRICS}
    positive = {
        metric: sum(float(row[metric]) > 0 for row in deltas)
        for metric in ("psnr", "ssim")
    }
    checks = {
        "psnr_floor": averages["psnr"] >= 0.05,
        "ssim_floor": averages["ssim"] >= 0.0005,
        "lpips_nonreverse": averages["lpips"] >= 0,
        "mae_nonreverse": averages["mae"] >= 0,
        "psnr_probe_direction": positive["psnr"] >= 3,
        "ssim_probe_direction": positive["ssim"] >= 3,
    }
    return {"pass": all(checks.values()), "checks": checks, "means": averages, "positive_probes": positive}


def phase_a_decision(*, ordinary_churn: float, far_churn: float, min_far_angle: float,
                     in_bounds_nonempty: float, far_output_gate: dict,
                     integrity_pass: bool = True) -> dict:
    if not integrity_pass or min_far_angle < 60 or in_bounds_nonempty < 1.0 or far_churn < 0.25:
        return {"verdict": "AUDIT_INVALID", "proceed_phase_b": False}
    if far_output_gate.get("pass"):
        label = "INTERVENTION_TOO_WEAK" if ordinary_churn < 0.25 else "FAR_PROTOCOL_CAMERA_RESPONSE"
        return {"verdict": label, "proceed_phase_b": False}
    return {"verdict": "PROCEED_PHASE_B", "proceed_phase_b": True}


def _cell_files(root: Path, source_campaign: Path, cell: str) -> tuple[Path, Path, Path]:
    config_name, manifest_name = CELLS[cell]
    config = root / "configs" / "stage3_1" / config_name
    manifest = source_campaign / "manifest" / manifest_name
    checkpoint = source_campaign / "train" / cell / "seed42" / "stage3_step_2000.pt"
    return config, manifest, checkpoint


def _all_camera(dataset_root: Path, scene: str, resolution: int) -> CameraBatch:
    sequence = NeRFSyntheticAdapter(dataset_root / scene).index(Split.TRAIN)
    return _camera(sequence.observations, resolution, torch.device("cpu"))


def prepare(args) -> dict:
    root = args.repo_root.resolve()
    campaign = args.campaign_root.resolve()
    source = args.source_campaign.resolve()
    revision = _git_revision(root)
    origin_revision = _origin_master_revision(root)
    if revision != origin_revision:
        raise RuntimeError(
            f"Phase A must bind a pushed commit: HEAD={revision}, origin/master={origin_revision}"
        )
    campaign.mkdir(parents=True, exist_ok=True)
    protocol_cells = {}
    for cell in CELLS:
        source_config, source_manifest, checkpoint = _cell_files(root, source, cell)
        for path in (source_config, source_manifest, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        config_target = campaign / "configs" / source_config.name
        manifest_target = campaign / "manifest" / source_manifest.name
        copy_or_verify(source_config, config_target)
        copy_or_verify(source_manifest, manifest_target)
        config = load_stage3_config(config_target)
        seen = read_json(manifest_target)
        groups = seen.get("groups", [])
        if len(groups) != 100 or config.train_scenes != ("chair",):
            raise ValueError(f"{cell} must contain exactly 100 chair anchors")
        all_camera = _all_camera(args.dataset_root, "chair", config.image_size)
        donor_groups = []
        for group in groups:
            selection = select_far_donors(all_camera, group["indices"])
            donor_groups.append({
                "id": group["id"],
                "scene": group["scene"],
                "anchor": group["anchor"],
                "source_indices": group["indices"],
                "fusion_camera_indices": [group["anchor"], *selection["indices"]],
                "donor_angles_deg": selection["angles_deg"],
            })
        donors = {
            "version": 1,
            "scope": "stage3_2_far_camera_donors",
            "seen_manifest_sha256": sha256_file(manifest_target),
            "sampling_signature": {
                "train_scenes": list(config.train_scenes),
                "image_size": config.image_size,
                "scale": config.scale,
                "views": config.views,
            },
            "selection": "unique greedy maximum optical-axis angle outside source group",
            "groups": donor_groups,
        }
        donor_path = campaign / "manifest" / f"{cell}_far_camera_donors.json"
        write_frozen_json(donor_path, donors)
        protocol_cells[cell] = {
            "config": str(config_target),
            "config_sha256": sha256_file(config_target),
            "seen_manifest": str(manifest_target),
            "seen_manifest_sha256": sha256_file(manifest_target),
            "camera_donor_manifest": str(donor_path),
            "camera_donor_manifest_sha256": sha256_file(donor_path),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    protocol = {
        "schema_version": 1,
        "scope": "stage3_2_camera_causality_phase_a",
        "git_revision": revision,
        "origin_master_revision": origin_revision,
        "training_seed": 42,
        "inference_seed": INFERENCE_SEED,
        "probe_ids": list(PROBE_IDS),
        "modes": list(MODES),
        "epipolar_band": 1.5,
        "mask_churn_floor": 0.25,
        "far_camera_angle_floor_deg": 60.0,
        "stage4_status": "HOLD",
        "stage4_scope": "large-scale supervised 3DSR; not started",
        "stage5_4dsr": "preserved; outside this campaign",
        "cells": protocol_cells,
    }
    write_frozen_json(campaign / "protocol.json", protocol)
    return protocol


def audit_geometry(args) -> dict:
    campaign = args.campaign_root.resolve()
    protocol = read_json(campaign / "protocol.json")
    rows = []
    for cell in CELLS:
        config = load_stage3_config(Path(protocol["cells"][cell]["config"]))
        seen = read_json(Path(protocol["cells"][cell]["seen_manifest"]))
        donors = {item["id"]: item for item in read_json(
            Path(protocol["cells"][cell]["camera_donor_manifest"])
        )["groups"]}
        all_camera = _all_camera(args.dataset_root, "chair", config.image_size)
        patch_grid = (config.image_size // 16, config.image_size // 16)
        for group in seen["groups"]:
            correct_camera = _subset_camera(all_camera, group["indices"])
            correct_mask, correct_stats = _mask_stats(correct_camera, patch_grid, config.epipolar_band)
            dummy = torch.zeros(1, 3, config.views, 1, 1)
            sample_seed = INFERENCE_SEED * 1_000_000 + group["anchor"]
            _, ordinary_camera, _ = intervene_lr(
                dummy,
                correct_camera,
                "shuffle_camera",
                generator=torch.Generator().manual_seed(sample_seed * 10 + MODES.index("shuffle_fusion")),
            )
            far_item = donors[group["id"]]
            far_camera = _subset_camera(all_camera, far_item["fusion_camera_indices"])
            for name, changed_camera, angles in (
                ("shuffle_fusion", ordinary_camera, _angle_deltas(correct_camera, ordinary_camera)),
                ("far_shuffle_fusion", far_camera, far_item["donor_angles_deg"]),
            ):
                changed_mask, changed_stats = _mask_stats(changed_camera, patch_grid, config.epipolar_band)
                comparison = mask_comparison(correct_mask, changed_mask)
                rows.append({
                    "cell": cell,
                    "group_id": group["id"],
                    "anchor": group["anchor"],
                    "intervention": name,
                    "minimum_camera_angle_deg": min(angles),
                    **{f"correct_{key}": value for key, value in correct_stats.items()},
                    **{f"changed_{key}": value for key, value in changed_stats.items()},
                    **comparison,
                })
    if len(rows) != 400:
        raise RuntimeError(f"expected 400 geometry rows, got {len(rows)}")
    path = campaign / "phase_a" / "geometry_rows.jsonl"
    write_frozen_jsonl(path, rows)
    write_csv(path.with_suffix(".csv"), rows)
    by_cell = {}
    for cell in CELLS:
        by_cell[cell] = {}
        for intervention in ("shuffle_fusion", "far_shuffle_fusion"):
            selected = [row for row in rows if row["cell"] == cell and row["intervention"] == intervention]
            by_cell[cell][intervention] = {
                "rows": len(selected),
                "mean_mask_jaccard": mean(row["mask_jaccard"] for row in selected),
                "mean_mask_churn": mean(row["mask_churn"] for row in selected),
                "minimum_camera_angle_deg": min(row["minimum_camera_angle_deg"] for row in selected),
                "minimum_in_bounds_query_nonempty_ratio": min(
                    min(row["correct_in_bounds_query_nonempty_ratio"], row["changed_in_bounds_query_nonempty_ratio"])
                    for row in selected
                ),
                "maximum_zero_key_query_pair_count": max(
                    max(row["correct_zero_key_query_pair_count"], row["changed_zero_key_query_pair_count"])
                    for row in selected
                ),
                "mean_changed_cross_key_retention": mean(row["changed_cross_key_retention"] for row in selected),
                "mean_changed_cross_keys_per_query": mean(row["changed_cross_keys_per_query"] for row in selected),
                "mean_changed_valid_pair_ratio": mean(row["changed_valid_pair_ratio"] for row in selected),
            }
    summary = {"scope": "cross_view_local_band_geometry", "rows": len(rows), "cells": by_cell}
    write_json(campaign / "phase_a" / "geometry_summary.json", summary)
    return summary


def _gpu_is_idle(gpu: int) -> None:
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError(f"GPU work is already running; refusing to share GPU {gpu}")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    selected = next((line for line in rows if int(line.split(",")[0]) == gpu), None)
    if selected is None:
        raise RuntimeError(f"GPU {gpu} is not visible")
    _, memory, utilization = [value.strip() for value in selected.split(",")]
    if float(memory) > 64 or float(utilization) > 1:
        raise RuntimeError(f"GPU {gpu} is not idle: memory={memory} MiB, utilization={utilization}%")


def _runtime_args(args) -> list[str]:
    return [
        "--dataset-root", str(args.dataset_root),
        "--model-dir", str(args.model_dir),
        "--lq-source", str(args.lq_source),
        "--lq-checkpoint", str(args.lq_checkpoint),
        "--bridge-checkpoint", str(args.bridge_checkpoint),
    ]


def preflight(args) -> dict:
    import stage3_experiment as stage3

    _gpu_is_idle(args.gpu)
    protocol = read_json(args.campaign_root / "protocol.json")
    cell = protocol["cells"]["v4"]
    config = load_stage3_config(Path(cell["config"]))
    seen = read_json(Path(cell["seen_manifest"]))
    group = next(item for item in seen["groups"] if item["id"] == PROBE_IDS[0])
    donor = next(item for item in read_json(Path(cell["camera_donor_manifest"]))["groups"] if item["id"] == group["id"])
    runtime = stage3.load_runtime(
        config,
        model_dir=args.model_dir,
        lq_source=args.lq_source,
        lq_checkpoint=args.lq_checkpoint,
        bridge_checkpoint=args.bridge_checkpoint,
        stage3_checkpoint=Path(cell["checkpoint"]),
        device="cuda",
    )
    runtime.module.eval()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    _, hr, lr, camera = stage3._load_indices(
        args.dataset_root, group["scene"], "train", group["indices"], config, runtime.device
    )
    far_camera = stage3._camera_for_indices(
        args.dataset_root, group["scene"], donor["fusion_camera_indices"], config, runtime.device
    )
    with torch.inference_mode():
        clean = runtime.vae.encode_multiview(hr)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        sampled, diagnostics = stage3.sample_latents(
            runtime,
            lr,
            camera,
            tuple(clean.shape),
            1,
            config.image_size,
            fusion_camera=far_camera,
            geometry_camera=camera,
            allow_self_view_source=False,
            seed=INFERENCE_SEED * 1_000_000,
            sampling_shift=config.sampling_shift,
            return_diagnostics=True,
        )
        decoded = runtime.vae.decode_multiview(sampled)
    fusion = diagnostics["fusion_diagnostics"]["fusion"]
    result = {
        "git_revision": protocol["git_revision"],
        "group_id": group["id"],
        "checkpoint_sha256": sha256_file(Path(cell["checkpoint"])),
        "latent_shape": list(clean.shape),
        "sample_shape": list(sampled.shape),
        "decoded_shape": list(decoded.shape),
        "same_view_attention_mass": fusion["same_view_attention_mass"],
        "attention_mass_total": fusion["attention_mass_total"],
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(runtime.device) / 2**20,
        "pass": (
            list(sampled.shape) == list(clean.shape)
            and list(decoded.shape) == list(hr.shape)
            and fusion["same_view_attention_mass"] <= 1e-7
            and abs(fusion["attention_mass_total"] - 1.0) <= 1e-5
        ),
    }
    write_json(args.campaign_root / "preflight" / "v4_real_wan.json", result)
    del runtime, sampled, decoded, clean, hr, lr, camera, far_camera
    gc.collect()
    torch.cuda.empty_cache()
    if not result["pass"]:
        raise RuntimeError(f"real-Wan preflight failed: {result}")
    return result


def evaluate(args) -> Path:
    _gpu_is_idle(args.gpu)
    protocol = read_json(args.campaign_root / "protocol.json")
    cell = protocol["cells"]["v4"]
    output = args.campaign_root / "phase_a" / "eval_v4"
    if output.exists() and any(output.iterdir()):
        summary = output / "evaluation_summary.json"
        if summary.is_file() and read_json(summary).get("rows") == 28:
            return output
        raise RuntimeError(f"refusing to reuse incomplete evaluation directory: {output}")
    command = [
        args.python,
        str(args.repo_root / "scripts" / "stage3_experiment.py"),
        "seen-eval",
        "--config", cell["config"],
        *_runtime_args(args),
        "--checkpoint", cell["checkpoint"],
        "--seen-manifest", cell["seen_manifest"],
        "--camera-donor-manifest", cell["camera_donor_manifest"],
        "--subset", "probe",
        "--inference-seeds", str(INFERENCE_SEED),
        "--modes", *MODES,
        "--save-diagnostics",
        "--save-images",
        "--output-dir", str(output),
    ]
    control = args.campaign_root / "control" / "phase_a_v4_eval.command.txt"
    control.parent.mkdir(parents=True, exist_ok=True)
    control.write_text(shlex.join(command) + "\n", encoding="utf-8")
    log = args.campaign_root / "logs" / "phase_a_v4_eval.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PYTHONPATH"] = str(args.repo_root / "src")
    with log.open("w", encoding="utf-8") as stream:
        stream.write(shlex.join(command) + "\n")
        subprocess.run(command, cwd=args.repo_root, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    summary = read_json(output / "evaluation_summary.json")
    if summary.get("rows") != 28 or summary.get("baseline_rows") != 8:
        raise RuntimeError("Phase A evaluation row count mismatch")
    return output


def _paired_deltas(rows: list[dict], reference: str, changed: str) -> list[dict]:
    indexed = {(row["condition"], row["group_id"]): row for row in rows}
    result = []
    for group_id in PROBE_IDS:
        correct = indexed[(reference, group_id)]
        wrong = indexed[(changed, group_id)]
        result.append({
            "group_id": group_id,
            **{
                metric: (float(correct[metric]) - float(wrong[metric])) * GOOD_DIRECTION[metric]
                for metric in METRICS
            },
        })
    return result


def phase_a_integrity(campaign: Path, repo_root: Path) -> dict:
    """Validate the complete frozen Phase A artifact set before interpreting metrics."""
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        protocol_path = campaign / "protocol.json"
        protocol = read_json(protocol_path)
        checks["protocol_git_is_current_head"] = protocol["git_revision"] == _git_revision(repo_root)
        checks["protocol_git_is_pushed_origin_master"] = (
            protocol["git_revision"] == protocol.get("origin_master_revision")
        )
        for cell, item in protocol["cells"].items():
            for key in ("config", "seen_manifest", "camera_donor_manifest", "checkpoint"):
                path = Path(item[key])
                checks[f"{cell}_{key}_exists"] = path.is_file()
                checks[f"{cell}_{key}_hash"] = (
                    path.is_file() and sha256_file(path) == item[f"{key}_sha256"]
                )

        geometry_rows = read_jsonl(campaign / "phase_a" / "geometry_rows.jsonl")
        geometry_summary = read_json(campaign / "phase_a" / "geometry_summary.json")
        checks["geometry_rows_400"] = len(geometry_rows) == geometry_summary.get("rows") == 400
        checks["geometry_values_finite"] = all(
            math.isfinite(float(value))
            for row in geometry_rows
            for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )

        output = campaign / "phase_a" / "eval_v4"
        evaluation_rows = read_jsonl(output / "evaluation_rows.jsonl")
        baseline_rows = read_jsonl(output / "baseline_rows.jsonl")
        diagnostic_rows = read_jsonl(output / "diagnostics.jsonl")
        evaluation_summary = read_json(output / "evaluation_summary.json")
        checks["evaluation_rows_28"] = len(evaluation_rows) == evaluation_summary.get("rows") == 28
        checks["baseline_rows_8"] = len(baseline_rows) == evaluation_summary.get("baseline_rows") == 8
        checks["diagnostic_rows_20"] = len(diagnostic_rows) == evaluation_summary.get("diagnostic_rows") == 20
        checks["evaluation_protocol"] = (
            evaluation_summary.get("conditions") == list(MODES)
            and evaluation_summary.get("inference_seeds") == [INFERENCE_SEED]
            and evaluation_summary.get("groups") == 4
        )
        v4 = protocol["cells"]["v4"]
        checks["evaluation_seen_manifest_hash"] = (
            evaluation_summary.get("seen_manifest_sha256") == v4["seen_manifest_sha256"]
        )
        checks["evaluation_donor_manifest_hash"] = (
            evaluation_summary.get("camera_donor_manifest_sha256")
            == v4["camera_donor_manifest_sha256"]
        )
        checks["evaluation_metrics_finite"] = all(
            math.isfinite(float(row[metric]))
            for row in (*evaluation_rows, *baseline_rows)
            for metric in METRICS
        )
        checks["evaluation_images_44"] = len(list((output / "images").rglob("*.png"))) == 44

        preflight_result = read_json(campaign / "preflight" / "v4_real_wan.json")
        checks["real_wan_preflight"] = bool(preflight_result.get("pass"))
        checks["preflight_checkpoint_hash"] = (
            preflight_result.get("checkpoint_sha256") == v4["checkpoint_sha256"]
        )
        log_text = (campaign / "logs" / "phase_a_v4_eval.log").read_text(
            encoding="utf-8", errors="replace"
        )
        checks["evaluation_log_clean"] = not any(
            marker in log_text.lower() for marker in ("traceback", "out of memory", "cuda error")
        ) and re.search(r"\bnan\b", log_text, flags=re.IGNORECASE) is None
        details.update({
            "protocol_sha256": sha256_file(protocol_path),
            "evaluation_rows": len(evaluation_rows),
            "baseline_rows": len(baseline_rows),
            "diagnostic_rows": len(diagnostic_rows),
            "geometry_rows": len(geometry_rows),
            "image_files": len(list((output / "images").rglob("*.png"))),
        })
    except Exception as error:
        details["error"] = f"{type(error).__name__}: {error}"
        checks["artifact_read_complete"] = False
    return {"pass": bool(checks) and all(checks.values()), "checks": checks, "details": details}


def analyze(args) -> dict:
    campaign = args.campaign_root.resolve()
    integrity = phase_a_integrity(campaign, args.repo_root.resolve())
    geometry = read_json(campaign / "phase_a" / "geometry_summary.json")
    eval_rows = read_jsonl(campaign / "phase_a" / "eval_v4" / "evaluation_rows.jsonl")
    if len(eval_rows) != 28:
        raise ValueError("Phase A requires exactly 28 evaluation rows")
    comparisons = {
        "shuffle_fusion": _paired_deltas(eval_rows, "correct", "shuffle_fusion"),
        "far_shuffle_fusion": _paired_deltas(eval_rows, "correct", "far_shuffle_fusion"),
        "no_self_shuffle_fusion": _paired_deltas(eval_rows, "no_self_correct", "no_self_shuffle_fusion"),
        "no_self_far_shuffle_fusion": _paired_deltas(eval_rows, "no_self_correct", "no_self_far_shuffle_fusion"),
    }
    gates = {name: output_gate(rows) for name, rows in comparisons.items()}
    v4 = geometry["cells"]["v4"]
    ordinary_churn = v4["shuffle_fusion"]["mean_mask_churn"]
    far_churn = v4["far_shuffle_fusion"]["mean_mask_churn"]
    min_far_angle = min(
        geometry["cells"][cell]["far_shuffle_fusion"]["minimum_camera_angle_deg"]
        for cell in CELLS
    )
    in_bounds_nonempty = min(
        geometry["cells"][cell][mode]["minimum_in_bounds_query_nonempty_ratio"]
        for cell in CELLS for mode in ("shuffle_fusion", "far_shuffle_fusion")
    )
    decision = phase_a_decision(
        ordinary_churn=ordinary_churn,
        far_churn=far_churn,
        min_far_angle=min_far_angle,
        in_bounds_nonempty=in_bounds_nonempty,
        far_output_gate=gates["far_shuffle_fusion"],
        integrity_pass=integrity["pass"],
    )
    result = {
        "scope": "stage3_2_camera_causality_phase_a",
        "claim_limit": "seen-train 3DSR camera-pair causality only",
        "stage4_status": "HOLD",
        "geometry": geometry,
        "deltas": comparisons,
        "output_gates": gates,
        "integrity": integrity,
        "decision": decision,
    }
    analysis = campaign / "analysis"
    write_json(analysis / "phase_a_summary.json", result)
    write_json(analysis / "machine_verdict.json", {
        "CAMERA_FUSION_PASS": False,
        "STAGE4_READY": False,
        "phase_a_verdict": decision["verdict"],
        "proceed_phase_b": decision["proceed_phase_b"],
        "integrity_pass": integrity["pass"],
    })
    delta_rows = []
    for name, rows in comparisons.items():
        delta_rows.extend({"comparison": name, **row} for row in rows)
    write_csv(analysis / "phase_a_deltas.csv", delta_rows)
    lines = [
        "# Stage 3.2 camera-pair causality — Phase A",
        "",
        "Stage 4 large-scale supervised 3DSR remains **HOLD**. Stage 5 4DSR is preserved but outside this campaign.",
        "",
        f"Verdict: **{decision['verdict']}**",
        "",
        f"- Ordinary shuffle mean mask churn: `{ordinary_churn:.6f}`",
        f"- Far shuffle mean mask churn: `{far_churn:.6f}`",
        f"- Minimum far donor angle: `{min_far_angle:.3f}°`",
        f"- Minimum in-bounds nonempty ratio: `{in_bounds_nonempty:.6f}`",
        f"- Far output gate: `{'PASS' if gates['far_shuffle_fusion']['pass'] else 'FAIL'}`",
        f"- No-self far output gate: `{'PASS' if gates['no_self_far_shuffle_fusion']['pass'] else 'FAIL'}`",
        f"- Artifact integrity: `{'PASS' if integrity['pass'] else 'FAIL'}`",
    ]
    (analysis / "phase_a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _require_runtime(args) -> None:
    for name in ("dataset_root", "model_dir", "lq_source", "lq_checkpoint", "bridge_checkpoint"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} must exist: {path}")


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "preflight", "evaluate", "analyze", "run"))
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--campaign-root", type=Path, default=root / "artifacts" / "stage3_2_camera_causality_20260913")
    parser.add_argument("--source-campaign", type=Path, default=root / "artifacts" / "stage3_1_views_20260912")
    parser.add_argument("--dataset-root", type=Path, default=root / "datasets" / "nerf_synthetic")
    parser.add_argument("--model-dir", type=Path, default=root / "models" / "Wan2.1-T2V-1.3B")
    parser.add_argument("--lq-source", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py"))
    parser.add_argument("--lq-checkpoint", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"))
    parser.add_argument("--bridge-checkpoint", type=Path, default=root / "artifacts" / "stage1" / "c_campaign_20260902" / "c_main" / "best_dev.pt")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.campaign_root = args.campaign_root.resolve()
    args.source_campaign = args.source_campaign.resolve()
    args.dataset_root = args.dataset_root.resolve()
    if args.command in {"prepare", "audit", "preflight", "evaluate", "run"}:
        _require_runtime(args)
    if args.command in {"prepare", "run"}:
        prepare(args)
    if args.command in {"audit", "run"}:
        audit_geometry(args)
    if args.command in {"preflight", "run"}:
        preflight(args)
    if args.command in {"evaluate", "run"}:
        evaluate(args)
    if args.command in {"analyze", "run"}:
        result = analyze(args)
        print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
