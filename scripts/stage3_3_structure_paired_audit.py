#!/usr/bin/env python3
"""Read-only W3/F1 pairing and fixed-noise attribution summary."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import stage3_3_structure as campaign
import stage3_3_ucpe_rre_fusion as prior


F1 = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-targetflow/artifacts/stage3_3_targetflow_20260923")
PROBES = ("lego:000", "lego:033", "lego:066", "lego:099")
CONDITIONS = ("correct", "target_drop", "mispaired_lr", "mispaired_camera",
              "shuffle_fusion", "target_drop_shuffle_fusion")


def load_rows(path):
    rows = prior.read_jsonl(path)
    assert len(rows) == 48
    return {(r["group_id"], r["condition"]): r for r in rows}


def main():
    w3_config = json.loads((campaign.W3 / "config/w3_lego_seed42.json").read_text())
    f1_config = json.loads((F1 / "config/f1_lego_seed42.json").read_text())
    assert f1_config.pop("target_view_flow_fraction") == 0.5
    assert w3_config == f1_config
    assert prior.sha256_file(campaign.W3 / "manifest/w3_lego_seed42.json") == prior.sha256_file(
        F1 / "manifest/f1_lego_seed42.json")
    gradients = {
        "W3": json.loads((F1 / "audit/gradient_audit.json").read_text()),
        "F1": json.loads((campaign.CAMPAIGN / "audit/gradient_audit_f1.json").read_text()),
    }
    grad_summary = {}
    for stage in ("initialization", "step500", "step1000"):
        a, b = (gradients[label]["stages"][stage] for label in ("W3", "F1"))
        assert len(a) == len(b) == 24
        assert all((x["probe_id"], x["view_indices"], x["sigma"], x["target_lr_dropped"],
                    x["noise_seed"], x["permutation"]) ==
                   (y["probe_id"], y["view_indices"], y["sigma"], y["target_lr_dropped"],
                    y["noise_seed"], y["permutation"]) for x, y in zip(a, b))
        grad_summary[stage] = {}
        for label, rows in (("W3", a), ("F1", b)):
            grad_summary[stage][label] = {
                "camera_hinge_active": sum(r["camera_hinge"] > 0 for r in rows),
                "groups": {group: {
                    "mean_norm": {objective: mean(r["gradients"][group]["norms"][objective]
                                                  for r in rows)
                                  for objective in ("flow", "camera_rank", "lr_rank")},
                    "mean_flow_camera_cosine": mean(values) if (values := [
                        r["gradients"][group]["cosines"]["flow_vs_camera_rank"]
                        for r in rows if r["gradients"][group]["cosines"]["flow_vs_camera_rank"] is not None
                    ]) else None,
                    "negative_flow_camera_count": sum(
                        r["gradients"][group]["cosines"]["flow_vs_camera_rank"] < 0
                        for r in rows if r["gradients"][group]["cosines"]["flow_vs_camera_rank"] is not None),
                } for group in ("qk", "value", "output", "bridge", "shared")},
            }
    evaluations = {}
    for step in (500, 1000):
        folders = {
            "W3": (campaign.CAMPAIGN / "eval_step500_w3" if step == 500
                   else campaign.W3 / "eval/w3_lego_seed42"),
            "F1": (campaign.CAMPAIGN / "eval_step500_f1" if step == 500
                   else F1 / "eval/f1_lego_seed42"),
        }
        rows = {label: load_rows(folder / "evaluation_rows.jsonl")
                for label, folder in folders.items()}
        for probe in PROBES:
            view = f"view_{int(probe.split(':')[1]):03d}"
            for label in ("W3", "F1"):
                for ref in ("hr.png", "lr_nearest.png", "bicubic.png", "vae_ceiling.png"):
                    path = folders[label] / "images/lego" / view / "reference" / ref
                    assert path.is_file()
            for ref in ("hr.png", "lr_nearest.png", "bicubic.png", "vae_ceiling.png"):
                a, b = (folders[label] / "images/lego" / view / "reference" / ref
                        for label in ("W3", "F1"))
                assert prior.sha256_file(a) == prior.sha256_file(b)
            for condition in CONDITIONS:
                a, b = (rows[label][probe, condition] for label in ("W3", "F1"))
                assert (a["view_indices"], a["inference_seed"], a["view_index"]) == (
                    b["view_indices"], b["inference_seed"], b["view_index"])
        evaluations[str(step)] = {label: {
            "correct": {metric: mean(rows[label][probe, "correct"][metric] for probe in PROBES)
                        for metric in ("psnr", "ssim")},
            "deltas": {condition: {metric: mean(
                rows[label][probe, ("target_drop" if condition == "target_drop_shuffle_fusion"
                                    else "correct")][metric]
                - rows[label][probe, condition][metric] for probe in PROBES)
                for metric in ("psnr", "ssim")}
                for condition in CONDITIONS if condition != "correct"},
        } for label in ("W3", "F1")}
    result = {"read_only_inputs": True, "paired_config_except_flow": True,
              "paired_manifest": True, "paired_references_and_seeds": True,
              "checkpoint_sha256": {label: gradients[label]["checkpoint_sha256"]
                                    for label in ("W3", "F1")},
              "fixed_noise_gradients": grad_summary, "decoded_evaluation": evaluations,
              "inference": "F1 quality degrades by step500 and camera effect does not emerge by step1000; no persistent negative gradient cosine or simple camera-gradient collapse."}
    path = campaign.CAMPAIGN / "audit/paired_audit.json"
    prior.write_frozen_json(path, result)
    print(json.dumps({"path": str(path), "step500": evaluations["500"],
                      "step1000": evaluations["1000"]}, indent=2))


if __name__ == "__main__":
    main()
