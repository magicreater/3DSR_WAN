#!/usr/bin/env python3
"""Read-only, paired W1/W2 gradient audit on the frozen Lego probes."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import torch

import stage3_experiment as stage3
import stage3_3_rank_weight_ablation as weights
import stage3_3_symmetric_rank as symmetric
import stage3_3_ucpe_rre_fusion as prior
from rl3dsr.validation.stage3_protocol import load_stage3_config


LOSSES = ("flow", "camera_rank", "lr_rank")
PAIRS = tuple(itertools.combinations(LOSSES, 2))


def gradient_statistics(gradients, hidden):
    """Compare unweighted objectives, preserving zero-gradient cases."""
    groups = {}
    for loss, (qkv, output, *bridge) in gradients.items():
        groups[loss] = {
            "qk": (None if qkv is None else qkv[: 2 * hidden],),
            "value": (None if qkv is None else qkv[2 * hidden :],),
            "output": (output,),
            "bridge": tuple(bridge),
        }
    result = {}
    for group in ("qk", "value", "output", "bridge", "shared"):
        vectors = {
            loss: tuple(itertools.chain.from_iterable(groups[loss].values()))
            if group == "shared" else groups[loss][group]
            for loss in LOSSES
        }
        norms = {
            loss: math.sqrt(sum(float(t.float().square().sum()) for t in tensors if t is not None))
            for loss, tensors in vectors.items()
        }
        cosines = {}
        for left, right in PAIRS:
            dot = sum(
                float(a.float().flatten().dot(b.float().flatten()))
                for a, b in zip(vectors[left], vectors[right])
                if a is not None and b is not None
            )
            denominator = norms[left] * norms[right]
            cosines[f"{left}_vs_{right}"] = None if denominator == 0 else dot / denominator
        if any(not math.isfinite(value) for value in (*norms.values(), *(v for v in cosines.values() if v is not None))):
            raise RuntimeError("nonfinite gradient statistic")
        result[group] = {
            "norms": norms,
            "zero_gradient": {loss: value == 0 for loss, value in norms.items()},
            "cosines": cosines,
        }
    return result


def summarize(rows):
    pairs = {f"{a}_vs_{b}": [] for a, b in PAIRS}
    zeros = {loss: 0 for loss in LOSSES}
    for row in rows:
        shared = row["gradients"]["shared"]
        for loss in LOSSES:
            zeros[loss] += int(shared["zero_gradient"][loss])
        for pair, value in shared["cosines"].items():
            if value is not None:
                pairs[pair].append(value)
    return {
        "zero_gradient_count": zeros,
        "cosine": {
            pair: {"valid": len(values), "negative": sum(v < 0 for v in values),
                   "negative_fraction": None if not values else sum(v < 0 for v in values) / len(values),
                   "mean": None if not values else sum(values) / len(values)}
            for pair, values in pairs.items()
        },
    }


def audit(args):
    protocol = weights.prepare(args)
    rows = {}
    checkpoint_hashes = {}
    for name in weights.CELL_ORDER:
        item = protocol["cells"][name]
        config_path, manifest_path, train_dir, _ = weights.cell_paths(args, name)
        checkpoint = train_dir / f"stage3_step_{args.checkpoint_step:04d}.pt"
        summary = prior.read_json(args.campaign_root / "analysis" / name / "summary.json")
        checkpoint_hash = prior.sha256_file(checkpoint)
        if ((args.checkpoint_step == 1000 and checkpoint_hash != summary["checkpoint_sha256"])
                or prior.sha256_file(config_path) != item["config_sha256"]
                or prior.sha256_file(manifest_path) != item["manifest_sha256"]):
            raise RuntimeError(f"{name}: frozen input hash mismatch")
        checkpoint_hashes[name] = checkpoint_hash
        config = load_stage3_config(config_path)
        manifest = prior.read_json(manifest_path)
        runtime = stage3.load_runtime(
            config, model_dir=args.model_dir, lq_source=args.lq_source,
            lq_checkpoint=args.lq_checkpoint, bridge_checkpoint=args.bridge_checkpoint,
            stage3_checkpoint=checkpoint, device="cuda",
        )
        runtime.module.train()
        runtime.dit.model.eval().requires_grad_(False)
        runtime.vae.model.model.eval().requires_grad_(False)
        if any(p.requires_grad for p in itertools.chain(runtime.dit.model.parameters(), runtime.vae.model.model.parameters())):
            raise RuntimeError("Wan/VAE are not frozen")
        parameters = [runtime.module.fusion.qkv.weight, runtime.module.fusion.output.weight,
                      *runtime.module.conditioner.bridge.parameters()]
        rows[name] = []
        for probe_id in item["probe_ids"]:
            group = next(group for group in manifest["groups"] if group["id"] == probe_id)
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
            probe = symmetric._rank_probe(runtime, config, lr, camera, clean, permutation, seed=3302)
            objectives = {key: probe[key].mean() for key in LOSSES}
            gradients = {
                loss: torch.autograd.grad(value, parameters, retain_graph=loss != LOSSES[-1], allow_unused=True)
                for loss, value in objectives.items()
            }
            statistics = gradient_statistics(gradients, runtime.module.fusion.hidden_dim)
            rows[name].append({
                "probe_id": probe_id, "view_indices": group["indices"],
                "noise_seed": 3302, "sigma": 0.5, "permutation": permutation.tolist(),
                "target_lr_dropped": True,
                "errors": {key: float(probe[key].mean().detach()) for key in ("flow", "e_correct", "e_camera", "e_lr", "camera_rank", "lr_rank")},
                "gradients": statistics,
            })
        del runtime
        torch.cuda.empty_cache()
    for first, second in zip(rows[weights.CELL_ORDER[0]], rows[weights.CELL_ORDER[1]]):
        if any(first[key] != second[key] for key in ("probe_id", "view_indices", "noise_seed", "sigma", "permutation", "target_lr_dropped")):
            raise RuntimeError("W1/W2 probe pairing mismatch")
    payload = {
        "schema_version": 1, "read_only": True, "checkpoint_step": args.checkpoint_step,
        "w1w2_protocol_sha256": prior.sha256_file(args.campaign_root / "protocol.json"),
        "checkpoints_sha256": checkpoint_hashes,
        "objectives": "unweighted correct all-view flow; target-only camera/LR hinge",
        "cells": {name: {"probes": value, "summary": summarize(value)} for name, value in rows.items()},
    }
    prior.write_frozen_json(args.output, payload)
    return payload


def main():
    parser = weights.build_parser()
    parser.add_argument("--output", type=Path, default=weights.DEFAULT_ROOT / "gradient_audit" / "w1_w2.json")
    parser.add_argument("--checkpoint-step", type=int, choices=(500, 1000), default=1000)
    args = parser.parse_args()
    if args.command != "prepare":
        raise SystemExit("use the prepare positional command for this read-only audit")
    args.repo_root = weights.ROOT
    args.campaign_root = args.campaign_root.resolve()
    print(json.dumps(audit(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
