#!/usr/bin/env python3
"""Frozen W3 plus one calibrated target-image SSIM candidate."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import stage3_experiment as stage3
import stage3_3_target_rank as a6
import stage3_3_ucpe_rre_fusion as prior
from rl3dsr.validation.stage3_protocol import evenly_spaced_indices, load_stage3_config


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/linzizhuo/RL3DSR_WAN_REMO")
W3 = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-target-rank/artifacts/stage3_3_a6sw3_20260923")
CAMPAIGN = ROOT / "artifacts/stage3_3_structure_20260923"
PARENT = DATA / "artifacts/stage3_2_camera_causality_20260913/phase_c/train/rank/seed42/stage3_step_1000.pt"
CELLS = {
    "s1_lego_seed42": a6.CELLS["a6_lego_seed42"],
    "s1_lego_seed43": a6.CELLS["a6_lego_seed43"],
    "s1_chair_seed42": a6.CELLS["a6_chair_seed42"],
}


def sha256(path):
    return prior.sha256_file(path)


def prepare():
    calibration_path = CAMPAIGN / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    w3_config_path = W3 / "config/w3_lego_seed42.json"
    if (calibration["parent_sha256"] != sha256(PARENT)
            or calibration["w3_config_sha256"] != sha256(w3_config_path)
            or len(calibration["batches"]) != 8
            or abs(calibration["weighted_gradient_ratio_median"] - 0.1) > 1e-9):
        raise RuntimeError("structure calibration drift")
    w3 = load_stage3_config(w3_config_path)
    if w3.target_view_flow_fraction is not None or w3.correct_image_ssim_weight is not None:
        raise RuntimeError("W3 flow must be uniform and without structure loss")
    source_paths = [ROOT / p for p in (
        "scripts/stage3_experiment.py", "scripts/stage3_3_structure.py",
        "scripts/stage3_3_structure_calibrate.py", "src/rl3dsr/validation/stage3_protocol.py",
        "src/rl3dsr/models/wan/stage3.py",
    )]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    cells = {}
    for name, (scene, seed, validation, test) in CELLS.items():
        config = replace(w3, train_scenes=(scene,), validation_scenes=validation,
                         test_scenes=test, training_seeds=(seed,),
                         correct_image_ssim_weight=calibration["weight"])
        config_path = CAMPAIGN / "config" / f"{name}.json"
        manifest_path = CAMPAIGN / "manifest" / f"{name}.json"
        prior.write_frozen_json(config_path, config.to_dict())
        if not manifest_path.exists():
            stage3._prepare_seen_manifest(
                SimpleNamespace(dataset_root=DATA / "datasets/nerf_synthetic",
                                manifest=manifest_path), config
            )
        manifest = json.loads(manifest_path.read_text())
        probes = tuple(f"{scene}:{i:03d}" for i in evenly_spaced_indices(
            manifest["datasets"][scene]["views"], 4
        ))
        if (tuple(manifest["subsets"]["probe"]) != probes
                or manifest["sampling_signature"] != stage3._sampling_signature(config)):
            raise RuntimeError(f"{name}: probe manifest drift")
        cells[name] = {"scene": scene, "seed": seed, "probe_ids": probes,
                       "config_sha256": sha256(config_path),
                       "manifest_sha256": sha256(manifest_path)}
    if sha256(CAMPAIGN / "manifest/s1_lego_seed42.json") != sha256(
            W3 / "manifest/w3_lego_seed42.json"):
        raise RuntimeError("Lego probes differ from W3")
    protocol = {
        "schema_version": 1, "scope": "stage3_3_w3_target_ssim",
        "source_sha256": source_hashes,
        "parent_checkpoint_sha256": sha256(PARENT),
        "w3_config_sha256": sha256(w3_config_path),
        "calibration_sha256": sha256(calibration_path),
        "structure_weight": calibration["weight"],
        "flow_scope": "four_view_uniform", "rank_weight": w3.camera_rank_weight,
        "symmetric_camera_fraction": w3.symmetric_camera_fraction,
        "inference_seed": 3302,
        "modes": prior.MODES,
        "cells": cells,
        "hard_gate": {"quality_psnr_delta_min": -0.10,
                      "quality_ssim_delta_min": -0.001,
                      "intervention_psnr_mean_min": 0.03,
                      "intervention_ssim_mean_min": 0.0003,
                      "positive_probes_min": 3},
        "stage4_ready": False,
    }
    prior.write_frozen_json(CAMPAIGN / "protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
