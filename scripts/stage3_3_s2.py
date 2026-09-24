#!/usr/bin/env python3
"""Freeze the single S2 decoded-camera ranking pilot and its replicas."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import stage3_experiment as stage3
import stage3_3_structure as s1
import stage3_3_ucpe_rre_fusion as prior
from rl3dsr.validation.stage3_protocol import evenly_spaced_indices, load_stage3_config


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "artifacts/stage3_3_s2_decoded_pair_20260924"
CELLS = {name.replace("s1_", "s2_"): value for name, value in s1.CELLS.items()}
PARENT = s1.PARENT


def prepare():
    audit = json.loads((ROOT / "artifacts/stage3_3_a6_direction_20260924/review.json").read_text())
    if audit["status"] != "S2_PILOT_SUPPORTED" or audit["STAGE4_READY"]:
        raise RuntimeError("decoded-direction audit does not support S2")
    calibration_path = s1.CAMPAIGN / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    w3_path = s1.W3 / "config/w3_lego_seed42.json"
    if (calibration["parent_sha256"] != prior.sha256_file(PARENT)
            or calibration["w3_config_sha256"] != prior.sha256_file(w3_path)
            or calibration["weight"] != 0.048734944021346524):
        raise RuntimeError("S1 calibration or common parent changed")
    w3 = load_stage3_config(w3_path)
    if w3.correct_image_ssim_weight is not None or w3.paired_image_ssim_rank:
        raise RuntimeError("W3 configuration changed")
    sources = (
        "scripts/stage3_experiment.py", "scripts/stage3_3_s2.py",
        "src/rl3dsr/validation/stage3_protocol.py", "src/rl3dsr/models/wan/stage3.py",
    )
    cells = {}
    for name, (scene, seed, validation, test) in CELLS.items():
        config = replace(w3, train_scenes=(scene,), validation_scenes=validation,
                         test_scenes=test, training_seeds=(seed,),
                         correct_image_ssim_weight=calibration["weight"],
                         paired_image_ssim_rank=True)
        config_path = CAMPAIGN / "config" / f"{name}.json"
        manifest_path = CAMPAIGN / "manifest" / f"{name}.json"
        prior.write_frozen_json(config_path, config.to_dict())
        if not manifest_path.exists():
            stage3._prepare_seen_manifest(
                SimpleNamespace(dataset_root=s1.DATA / "datasets/nerf_synthetic",
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
                       "config_sha256": prior.sha256_file(config_path),
                       "manifest_sha256": prior.sha256_file(manifest_path)}
    if prior.sha256_file(CAMPAIGN / "manifest/s2_lego_seed42.json") != prior.sha256_file(
            s1.W3 / "manifest/w3_lego_seed42.json"):
        raise RuntimeError("Lego probes differ from W3")
    protocol = {
        "scope": "stage3_3_s2_single_decoded_camera_pair",
        "audit_sha256": prior.sha256_file(ROOT / "artifacts/stage3_3_a6_direction_20260924/review.json"),
        "source_sha256": {path: prior.sha256_file(ROOT / path) for path in sources},
        "parent_checkpoint_sha256": prior.sha256_file(PARENT),
        "w3_config_sha256": prior.sha256_file(w3_path),
        "calibration_sha256": prior.sha256_file(calibration_path),
        "structure_weight": calibration["weight"],
        "paired_margin_ssim": 0.0003,
        "inference_seed": 3302,
        "modes": prior.MODES,
        "cells": cells,
        "hard_gate": {"quality_psnr_delta_min": -0.10,
                      "quality_ssim_delta_min": -0.001,
                      "intervention_psnr_mean_min": 0.03,
                      "intervention_ssim_mean_min": 0.0003,
                      "positive_probes_min": 3},
        "STAGE4_READY": False,
    }
    prior.write_frozen_json(CAMPAIGN / "protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
