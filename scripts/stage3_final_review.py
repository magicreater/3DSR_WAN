#!/usr/bin/env python3
"""Independent, fixed-endpoint gate for the final Stage 3 4/8-view comparison."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

from stage3_final_campaign import OUT, MODES, PARENT, ROOT, SOURCE, cell_name, paths, sha


INTERVENTIONS = ("mispaired_lr", "mispaired_camera", "shuffle_fusion",
                 "target_drop_shuffle_fusion")


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def read_cell(name, protocol):
    item = protocol["cells"][name]
    config_path, manifest_path, train_dir, eval_dir = paths(name)
    train = rows(train_dir / "train_steps.jsonl")
    evaluated = rows(eval_dir / "evaluation_rows.jsonl")
    baseline = rows(eval_dir / "baseline_rows.jsonl")
    summary = json.loads((eval_dir / "evaluation_summary.json").read_text())
    run_manifest = json.loads((train_dir / "run_manifest.json").read_text())
    config = json.loads(config_path.read_text())
    probes = tuple(item["probes"])
    index = {(r["group_id"], r["condition"]): r for r in evaluated}
    check = {
        "config_hash": sha(config_path) == item["config_sha256"],
        "manifest_hash": sha(manifest_path) == item["manifest_sha256"],
        "train_4000_contiguous": len(train) == 4000 and [r.get("step") for r in train] == list(range(1, 4001)),
        "train_finite": finite(train),
        "evaluation_complete": len(evaluated) == len(probes) * len(MODES)
        and len(index) == len(evaluated) and set(index) == {(p, m) for p in probes for m in MODES},
        "evaluation_finite": finite(evaluated),
        "baseline_complete": len(baseline) == len(probes) * 2,
        "summary_scope": summary.get("conditions") == list(MODES)
        and summary.get("inference_seeds") == [3302] and summary.get("step") == 4000
        and summary.get("seen_manifest_sha256") == item["manifest_sha256"],
        "run_config": run_manifest.get("config") == config,
        "run_parent": run_manifest.get("provenance", {}).get("parent_checkpoint_sha256") == protocol["parent_checkpoint_sha256"],
        "run_seed": run_manifest.get("provenance", {}).get("training_seed") == item["seed"],
        "checkpoint": (train_dir / "stage3_step_4000.pt").is_file(),
        "images": len(list((eval_dir / "images").rglob("*.png"))) == len(probes) * (4 + len(MODES)),
        "diagnostics": len(rows(eval_dir / "diagnostics.jsonl")) == len(probes) * (len(MODES) - 1),
    }
    if item["kind"] == "a5":
        check["a5_calibration"] = (run_manifest.get("pairing", {}).get("enabled") is True
                                   and (train_dir / "pairing_preflight.json").is_file())
    else:
        check["new_recipe"] = (config.get("dynamic_fusion") is True
                               and config.get("shared_multiview_rope") is True
                               and config.get("correct_image_ssim_weight") is None
                               and config.get("paired_image_ssim_rank") is None)
    if not all(check.values()):
        raise RuntimeError(f"invalid cell {name}: {[k for k,v in check.items() if not v]}")
    baseline_index = {(r["group_id"], r["condition"]): r for r in baseline}
    correct = {m: mean(float(index[(p, "correct")][m]) for p in probes) for m in ("psnr", "ssim")}
    bicubic = mean(float(index[(p, "correct")]["psnr"] - baseline_index[(p, "bicubic")]["psnr"])
                   for p in probes)
    return {"name": name, "checks": check, "probes": probes, "index": index,
            "correct": correct, "bicubic_gain": bicubic,
            "checkpoint_sha256": sha(train_dir / "stage3_step_4000.pt")}


def condition_gate(cell, condition):
    probes, index = cell["probes"], cell["index"]
    base = "target_drop" if condition == "target_drop_shuffle_fusion" else "correct"
    deltas = {metric: [float(index[(p, base)][metric] - index[(p, condition)][metric])
                       for p in probes] for metric in ("psnr", "ssim")}
    checks = {"psnr_mean": mean(deltas["psnr"]) >= .03,
              "ssim_mean": mean(deltas["ssim"]) >= .0003,
              "psnr_direction": sum(v > 0 for v in deltas["psnr"]) >= 3,
              "ssim_direction": sum(v > 0 for v in deltas["ssim"]) >= 3}
    return {"pass": all(checks.values()), "checks": checks,
            "mean": {m: mean(v) for m, v in deltas.items()}, "per_probe": deltas}


def candidate_gate(candidate, control):
    if candidate["probes"] != control["probes"]:
        raise RuntimeError("candidate and matched A5 probe IDs differ")
    quality = {"psnr": candidate["correct"]["psnr"] - control["correct"]["psnr"],
               "ssim": candidate["correct"]["ssim"] - control["correct"]["ssim"]}
    interventions = {c: condition_gate(candidate, c) for c in INTERVENTIONS}
    probes, index = candidate["probes"], candidate["index"]
    auxiliary = {m: [float(index[(p, "correct")][m] - index[(p, "aux_permute")][m])
                     for p in probes] for m in ("psnr", "ssim")}
    target_move = {m: [float(index[(p, "correct")][m] - index[(p, "joint_permute")][m])
                       for p in probes] for m in ("psnr", "ssim")}
    checks = {"quality_psnr": quality["psnr"] >= -.10,
              "quality_ssim": quality["ssim"] >= -.001,
              "bicubic": candidate["bicubic_gain"] > 0,
              "auxiliary_permute": max(auxiliary["psnr"]) <= .10
              and max(auxiliary["ssim"]) <= .001,
              **{c: result["pass"] for c, result in interventions.items()}}
    return {"pass": all(checks.values()), "checks": checks,
            "quality_vs_same_view_a5": quality, "correct": candidate["correct"],
            "bicubic_psnr_gain": candidate["bicubic_gain"],
            "interventions": interventions, "auxiliary_permute_loss": auxiliary,
            "target_slot_move_diagnostic": target_move}


def select(v4, v8, cells):
    if not v4["pass"] and not v8["pass"]:
        return None, {"reason": "both_view_counts_fail"}
    if v8["pass"] and not v4["pass"]:
        return 8, {"reason": "only_v8_passes"}
    if v4["pass"] and not v8["pass"]:
        return 4, {"reason": "only_v4_passes"}
    four = cells[cell_name("new", 4)]
    eight = cells[cell_name("new", 8)]
    psnr_per_probe = [float(eight["index"][(p, "correct")]["psnr"]
                            - four["index"][(p, "correct")]["psnr"]) for p in four["probes"]]
    checks = {"psnr_gain": mean(psnr_per_probe) >= .10,
              "ssim_nonregression": eight["correct"]["ssim"] >= four["correct"]["ssim"],
              "psnr_direction": sum(x > 0 for x in psnr_per_probe) >= 3,
              "camera_not_weaker": all(
                  v8["interventions"][c]["mean"][m] >= v4["interventions"][c]["mean"][m]
                  for c in ("mispaired_camera", "shuffle_fusion", "target_drop_shuffle_fusion")
                  for m in ("psnr", "ssim"))}
    return (8 if all(checks.values()) else 4), {"checks": checks, "psnr_per_probe": psnr_per_probe}


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    if sha(PARENT) != protocol["parent_checkpoint_sha256"]:
        raise RuntimeError("parent checkpoint changed")
    if any(sha(ROOT / p) != digest for p, digest in protocol["source_sha256"].items()):
        raise RuntimeError("frozen source changed")
    cells = {name: read_cell(name, protocol) for name in protocol["cells"]}
    four = candidate_gate(cells[cell_name("new", 4)], cells[cell_name("a5", 4)])
    eight = candidate_gate(cells[cell_name("new", 8)], cells[cell_name("a5", 8)])
    selected, view_choice = select(four, eight, cells)
    result = {"status": "HOLD" if selected is None else "STAGE3_REPLICATION_REQUIRED",
              "STAGE4_READY": False, "selected_views": selected,
              "view_choice": view_choice,
              "v4": four, "v8": eight,
              "cells": {name: {k: v for k, v in cell.items() if k != "index"}
                        for name, cell in cells.items()},
              "review_source_sha256": sha(Path(__file__))}
    destination = OUT / "analysis" / "four_arm_review.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "selected_views": selected,
                      "v4_pass": four["pass"], "v8_pass": eight["pass"]}, indent=2))


if __name__ == "__main__":
    main()
