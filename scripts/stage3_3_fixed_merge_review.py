#!/usr/bin/env python3
"""Apply the frozen S1 Stage 4 gate to the inference-only 1:1 merge."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import stage3_3_rank_weight_ablation as previous
import stage3_3_ucpe_rre_fusion as prior
import stage4_readiness_review as review


ROOT = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-structure")
MERGE = ROOT / "artifacts/stage3_3_merge_20260923"
W3 = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-target-rank/artifacts/stage3_3_a6sw3_20260923")
S1 = ROOT / "artifacts/stage3_3_structure_20260923"
NAME = "m1_lego_seed42"
PROBES = ("lego:000", "lego:033", "lego:066", "lego:099")


def analyze():
    manifest = prior.read_json(MERGE / "merge_manifest.json")
    checkpoint = MERGE / "checkpoints/m1_lego_seed42_step1000.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    eval_dir = MERGE / "eval" / NAME
    rows = prior.read_jsonl(eval_dir / "evaluation_rows.jsonl")
    baseline = prior.read_jsonl(eval_dir / "baseline_rows.jsonl")
    diagnostics = prior.read_jsonl(eval_dir / "diagnostics.jsonl")
    evaluation = prior.read_json(eval_dir / "evaluation_summary.json")
    w3_checkpoint = W3 / "train/w3_lego_seed42/stage3_step_1000.pt"
    s1_checkpoint = S1 / "train/s1_lego_seed42/stage3_step_1000.pt"
    w3_manifest = W3 / "manifest/w3_lego_seed42.json"
    s1_manifest = S1 / "manifest/s1_lego_seed42.json"
    equivariance = prior.fusion_equivariance(
        SimpleNamespace(campaign_root=MERGE, v2=True),
        prior.Cell(NAME, f"{NAME}.json", 42, True))
    integrity = {
        "source_hashes": manifest["source_sha256"] == {
            "W3": prior.sha256_file(w3_checkpoint), "S1": prior.sha256_file(s1_checkpoint)},
        "source_manifests": prior.sha256_file(w3_manifest) == prior.sha256_file(s1_manifest)
                            == manifest["manifest_sha256"],
        "checkpoint_hash": prior.sha256_file(checkpoint) == manifest["checkpoint_sha256"],
        "checkpoint_payload": payload["step"] == 1000 and payload["training_state"] is None
                              and payload["provenance"].get("merge_coefficient_s1") == 0.5
                              and payload["provenance"].get("merge_sources_sha256") == manifest["source_sha256"],
        "evaluation_rows": len(rows) == evaluation.get("rows") == 48,
        "evaluation_keys": set(prior.evaluation_index(rows, PROBES)) == {
            (probe, mode) for probe in PROBES for mode in prior.MODES},
        "baseline_rows": len(baseline) == 8,
        "diagnostic_rows": len(diagnostics) == 44,
        "images": len(list((eval_dir / "images").rglob("*.png"))) == 64,
        "evaluation_manifest": evaluation.get("seen_manifest_sha256") == manifest["manifest_sha256"],
        "evaluation_seed": evaluation.get("inference_seeds") == [3302],
        "evaluation_checkpoint": evaluation.get("checkpoint") == str(checkpoint.resolve()),
    }
    if not all(integrity.values()):
        result = {"pass": False, "status": "HOLD", "failure_category": "AUDIT_INVALID",
                  "integrity": integrity, "STAGE4_READY": False}
        prior.write_frozen_json(MERGE / "analysis" / NAME / "summary.json", result)
        return result
    # candidate_gate computes the frozen intervention deltas. Its training fields
    # come from W3 only; they are not used as M1 training claims or pass checks.
    train = prior.read_jsonl(W3 / "train/w3_lego_seed42/train_steps.jsonl")
    gate = prior.candidate_gate(rows, train, expected_steps=1000, reference=None,
                                v2=True, fusion_equivariant=equivariance["pass"],
                                probe_ids=PROBES)
    conditions = ("mispaired_lr", *previous.CAMERA_CONDITIONS)
    interventions = {condition: previous._paired_gate(gate, condition)
                     for condition in conditions}
    a5 = review._load_cell(review.DEFAULT_A5_LEGO, arm="A5", scene="lego", seed=42)
    if tuple(a5["probe_ids"]) != PROBES:
        raise RuntimeError("matched A5 probes differ")
    quality = review.nonregression(review._gate(a5["summary"])["correct"], gate["correct"])
    index = prior.evaluation_index(rows, PROBES)
    bicubic = {r["group_id"]: r for r in baseline if r["condition"] == "bicubic"}
    bicubic_gain = sum(index[(probe, "correct")]["psnr"] - bicubic[probe]["psnr"]
                       for probe in PROBES) / len(PROBES)
    dependency = {key: gate["checks"][key] for key in (
        "correct_repeat_equivalent", "joint_permute_equivalent", "remove_psnr",
        "remove_direction", "target_drop_psnr", "target_drop_direction")}
    checks = {"integrity": True, "fusion_equivariance": equivariance["pass"],
              "dependency": all(dependency.values()), "bicubic_gain": bicubic_gain > 0,
              "quality": quality["pass"],
              **{condition: value["pass"] for condition, value in interventions.items()}}
    passed = all(checks.values())
    failure = (None if passed else "CORE_INVALID" if not checks["fusion_equivariance"] or not checks["dependency"]
               else "QUALITY_FAIL" if not checks["quality"] or not checks["bicubic_gain"]
               else "LR_FAIL" if not checks["mispaired_lr"] else "CAMERA_FAIL")
    result = {"status": "PASS" if passed else "HOLD", "pass": passed,
              "failure_category": failure, "hard_gate": checks, "integrity": integrity,
              "fusion_equivariance": equivariance, "dependency": dependency,
              "matched_a5_nonregression": quality, "bicubic_psnr_gain": bicubic_gain,
              "interventions": interventions, "correct": gate["correct"],
              "per_probe": {condition: {metric: gate["deltas"][condition][metric]["values"]
                                        for metric in ("psnr", "ssim")}
                            for condition in conditions},
              "checkpoint_sha256": manifest["checkpoint_sha256"],
              "source_sha256": manifest["source_sha256"],
              "training_telemetry_basis": "W3 endpoint only; M1 is inference-only",
              "STAGE4_READY": False}
    prior.write_frozen_json(MERGE / "analysis" / NAME / "summary.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, allow_nan=False))
