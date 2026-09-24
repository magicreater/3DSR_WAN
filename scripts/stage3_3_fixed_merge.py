#!/usr/bin/env python3
"""One preregistered 1:1 W3/S1 inference-only adapter checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import stage3_experiment as stage3
from rl3dsr.models.wan.stage3 import validate_stage3_checkpoint_payload
from rl3dsr.validation.stage3_protocol import load_stage3_config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate_adapters(left, right):
    if left.keys() != right.keys() or set(left) != {"bridge", "geometry", "fusion"}:
        raise ValueError("adapter modules differ")
    merged = {}
    for module in ("bridge", "geometry", "fusion"):
        a, b = left[module], right[module]
        if a.keys() != b.keys():
            raise ValueError(f"{module} keys differ")
        merged[module] = {}
        for key in a:
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != y.dtype or x.dtype != torch.float32:
                raise ValueError(f"{module}.{key} shape or dtype differs")
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"{module}.{key} is nonfinite")
            merged[module][key] = x.mul(0.5).add(y, alpha=0.5)
    return merged


def merge(w3_path, s1_path, w3_manifest, s1_manifest, w3_config_path, output):
    w3_path, s1_path, output = map(Path, (w3_path, s1_path, output))
    hashes = {"W3": sha256(w3_path), "S1": sha256(s1_path)}
    if sha256(w3_manifest) != sha256(s1_manifest):
        raise ValueError("W3/S1 frozen probe manifests differ")
    w3 = torch.load(w3_path, map_location="cpu", weights_only=True)
    s1 = torch.load(s1_path, map_location="cpu", weights_only=True)
    if any(p.get("format") != "rl3dsr-stage3" or p.get("step") != 1000 for p in (w3, s1)):
        raise ValueError("sources must be Stage 3 step1000 checkpoints")
    if w3["bridge_architecture"] != s1["bridge_architecture"] or w3["provenance"] != s1["provenance"]:
        raise ValueError("W3/S1 initialization or provenance differs")
    w3_config, s1_config = dict(w3["config"]), dict(s1["config"])
    if w3_config.pop("correct_image_ssim_weight", None) is not None:
        raise ValueError("W3 must not use decoded SSIM")
    if not isinstance(s1_config.pop("correct_image_ssim_weight", None), float):
        raise ValueError("S1 must use decoded SSIM")
    if w3_config != s1_config:
        raise ValueError("source configs differ beyond decoded SSIM")
    merged = interpolate_adapters(w3["adapters"], s1["adapters"])
    payload = {"format": w3["format"], "format_version": w3["format_version"],
               "step": 1000, "config": w3["config"],
               "bridge_architecture": w3["bridge_architecture"], "adapters": merged,
               "training_state": None,
               "provenance": {**w3["provenance"], "inference_only_merge": True,
                              "merge_coefficient_s1": 0.5,
                              "merge_sources_sha256": hashes,
                              "merge_manifest_sha256": sha256(w3_manifest)}}
    config = load_stage3_config(w3_config_path)
    if payload["config"] != json.loads(json.dumps(config.to_dict())):
        raise ValueError("W3 config file differs from checkpoint")
    validate_stage3_checkpoint_payload(payload, stage3._checkpoint_validation_module(config),
                                       expected_config=config.to_dict())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(payload, stream)
    loaded = torch.load(output, map_location="cpu", weights_only=True)
    validate_stage3_checkpoint_payload(loaded, stage3._checkpoint_validation_module(config),
                                       expected_config=config.to_dict())
    if loaded["training_state"] is not None or loaded["provenance"]["merge_sources_sha256"] != hashes:
        raise RuntimeError("merged checkpoint roundtrip failed")
    if hashes != {"W3": sha256(w3_path), "S1": sha256(s1_path)}:
        raise RuntimeError("source checkpoint changed during merge")
    return {"checkpoint": str(output), "checkpoint_sha256": sha256(output),
            "source_sha256": hashes, "manifest_sha256": sha256(w3_manifest),
            "coefficient_s1": 0.5, "training_state": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w3", type=Path, required=True)
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--w3-manifest", type=Path, required=True)
    parser.add_argument("--s1-manifest", type=Path, required=True)
    parser.add_argument("--w3-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.w3, args.s1, args.w3_manifest, args.s1_manifest,
                           args.w3_config, args.output), indent=2))
