#!/usr/bin/env python3
"""Apply the frozen A6 quality and intervention gates to S2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import stage3_3_rank_weight_ablation as previous
import stage3_3_s2 as campaign
import stage3_3_ucpe_rre_fusion as prior
import stage4_readiness_review as review
from rl3dsr.validation.stage3_protocol import load_stage3_config


def analyze(name):
    protocol = json.loads((campaign.CAMPAIGN / "protocol.json").read_text())
    item = protocol["cells"][name]
    config_path = campaign.CAMPAIGN / "config" / f"{name}.json"
    manifest_path = campaign.CAMPAIGN / "manifest" / f"{name}.json"
    train_dir = campaign.CAMPAIGN / "train" / name
    eval_dir = campaign.CAMPAIGN / "eval" / name
    checkpoint = train_dir / "stage3_step_1000.pt"
    config = load_stage3_config(config_path)
    train = prior.read_jsonl(train_dir / "train_steps.jsonl")
    rows = prior.read_jsonl(eval_dir / "evaluation_rows.jsonl")
    baseline = prior.read_jsonl(eval_dir / "baseline_rows.jsonl")
    diagnostics = prior.read_jsonl(eval_dir / "diagnostics.jsonl")
    evaluation = prior.read_json(eval_dir / "evaluation_summary.json")
    manifest = prior.read_json(train_dir / "run_manifest.json")
    probes = tuple(item["probe_ids"])
    integrity = {
        "source_hashes": all(prior.sha256_file(campaign.ROOT / path) == digest
                             for path, digest in protocol["source_sha256"].items()),
        "parent_hash": prior.sha256_file(campaign.PARENT) == protocol["parent_checkpoint_sha256"],
        "config_hash": prior.sha256_file(config_path) == item["config_sha256"],
        "manifest_hash": prior.sha256_file(manifest_path) == item["manifest_sha256"],
        "training_complete": len(train) == 1000 and [r.get("step") for r in train] == list(range(1, 1001)),
        "structure_weight": all(r.get("correct_image_ssim_weight") == protocol["structure_weight"]
                                and 0 <= r.get("correct_image_ssim_loss", -1) <= 2 for r in train),
        "paired_structure": all(0 <= r.get("paired_image_ssim_rank_loss", -1) <= 2
                                for r in train),
        "train_config": manifest.get("config") == json.loads(json.dumps(config.to_dict())),
        "train_parent": manifest.get("provenance", {}).get("parent_checkpoint_sha256") == protocol["parent_checkpoint_sha256"],
        "train_seed": manifest.get("provenance", {}).get("training_seed") == item["seed"],
        "checkpoint": checkpoint.is_file(),
        "evaluation_rows": len(rows) == 48 == evaluation.get("rows"),
        "evaluation_keys": set(prior.evaluation_index(rows, probes)) == {
            (probe, mode) for probe in probes for mode in protocol["modes"]},
        "baseline_rows": len(baseline) == 8,
        "diagnostic_rows": len(diagnostics) == 44,
        "images": len(list((eval_dir / "images").rglob("*.png"))) == 64,
        "evaluation_manifest": evaluation.get("seen_manifest_sha256") == item["manifest_sha256"],
        "evaluation_seed": evaluation.get("inference_seeds") == [3302],
    }
    if not all(integrity.values()):
        result = {"status": "HOLD", "pass": False, "failure_category": "AUDIT_INVALID",
                  "integrity": integrity,
                  "hard_gate": {"quality": False, "mispaired_lr": False,
                                **{key: False for key in previous.CAMERA_CONDITIONS}},
                  "checkpoint_sha256": prior.sha256_file(checkpoint) if checkpoint.is_file() else None,
                  "STAGE4_READY": False}
        prior.write_frozen_json(campaign.CAMPAIGN / "analysis" / name / "summary.json", result)
        return result
    equivariance = prior.fusion_equivariance(
        SimpleNamespace(campaign_root=campaign.CAMPAIGN, v2=True),
        prior.Cell(name, f"{name}.json", item["seed"], True),
    )
    gate = prior.candidate_gate(rows, train, expected_steps=1000, reference=None,
                                v2=True, fusion_equivariant=equivariance["pass"], probe_ids=probes)
    conditions = ("mispaired_lr", *previous.CAMERA_CONDITIONS)
    interventions = {condition: previous._paired_gate(gate, condition)
                     for condition in conditions}
    a5_path = (review.DEFAULT_A5_LEGO if name == "s2_lego_seed42"
               else review.DEFAULT_A5_CHAIR if name == "s2_chair_seed42" else None)
    quality = None
    if a5_path is not None:
        a5 = review._load_cell(a5_path, arm="A5", scene=item["scene"], seed=item["seed"])
        if tuple(a5["probe_ids"]) != probes:
            raise RuntimeError("matched A5 probes differ")
        quality = review.nonregression(review._gate(a5["summary"])["correct"], gate["correct"])
    baseline_index = {r["group_id"]: r for r in baseline if r["condition"] == "bicubic"}
    index = prior.evaluation_index(rows, probes)
    bicubic_gain = sum(index[(probe, "correct")]["psnr"] - baseline_index[probe]["psnr"]
                       for probe in probes) / len(probes)
    dependency = {key: gate["checks"][key] for key in (
        "correct_repeat_equivalent", "joint_permute_equivalent", "remove_psnr",
        "remove_direction", "target_drop_psnr", "target_drop_direction")}
    checks = {"integrity": True, "fusion_equivariance": equivariance["pass"],
              "dependency": all(dependency.values()), "bicubic_gain": bicubic_gain > 0,
              "quality": quality is None or quality["pass"],
              **{condition: value["pass"] for condition, value in interventions.items()}}
    passed = all(checks.values())
    if not checks["fusion_equivariance"] or not checks["dependency"]:
        failure = "CORE_INVALID"
    elif not checks["quality"] or not checks["bicubic_gain"]:
        failure = "QUALITY_FAIL"
    elif not checks["mispaired_lr"]:
        failure = "LR_FAIL"
    elif not all(checks[c] for c in previous.CAMERA_CONDITIONS):
        failure = "CAMERA_FAIL"
    else:
        failure = None
    result = {"status": "PASS" if passed else "HOLD", "pass": passed,
              "failure_category": failure, "hard_gate": checks,
              "integrity": integrity, "fusion_equivariance": equivariance,
              "dependency": dependency, "matched_a5_nonregression": quality,
              "bicubic_psnr_gain": bicubic_gain, "interventions": interventions,
              "correct": gate["correct"], "per_probe": {condition: {
                  metric: gate["deltas"][condition][metric]["values"] for metric in ("psnr", "ssim")
              } for condition in conditions},
              "checkpoint_sha256": prior.sha256_file(checkpoint),
              "review_source_sha256": prior.sha256_file(Path(__file__)),
              "STAGE4_READY": False}
    prior.write_frozen_json(campaign.CAMPAIGN / "analysis" / name / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=campaign.CELLS)
    print(json.dumps(analyze(parser.parse_args().cell), indent=2, allow_nan=False))
