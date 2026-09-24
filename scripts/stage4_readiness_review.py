#!/usr/bin/env python3
"""Review frozen A5/A6 artifacts before large-scale Stage 4 3DSR training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A6_ROOT = ROOT / "artifacts/stage3_3_a6_20260921"
DEFAULT_A5_CHAIR = Path(
    "/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-verified/artifacts/"
    "stage3_3_v2_20260917/analysis/a5_pilot/summary.json"
)
DEFAULT_A5_LEGO = Path(
    "/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-data-control/artifacts/"
    "stage3_3_data_control_20260918/analysis/lego_seed42/summary.json"
)
EXPECTED_A6_REVISION = "820265309749eff2c7c4d77a12e2d8b8c348ee04"
PSNR_NONREGRESSION = -0.10
SSIM_NONREGRESSION = -0.001
PSNR_CORRESPONDENCE = 0.03
SSIM_CORRESPONDENCE = 0.0003
DIRECTIONAL_PROBES = 3
NUMERICAL_TOLERANCE = 1e-12
A6_CELLS = {
    "a6_chair_seed42": ("chair", 42),
    "a6_lego_seed42": ("lego", 42),
    "a6_lego_seed43": ("lego", 43),
}
MATCHED = {
    "chair_seed42": ("chair", 42, "a6_chair_seed42"),
    "lego_seed42": ("lego", 42, "a6_lego_seed42"),
}


class AuditError(ValueError):
    pass


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise AuditError(f"missing input: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate(summary: dict) -> dict:
    gate = summary.get("gate", summary)
    if not isinstance(gate, dict) or not isinstance(gate.get("correct"), dict):
        raise AuditError("summary is missing gate.correct")
    return gate


def _campaign_and_cell(summary_path: Path) -> tuple[Path, str]:
    if summary_path.name != "summary.json" or summary_path.parent.parent.name != "analysis":
        raise AuditError(f"unexpected summary path layout: {summary_path}")
    return summary_path.parents[2], summary_path.parent.name


def _probe_ids(rows_path: Path) -> list[str]:
    if not rows_path.is_file():
        raise AuditError(f"missing input: {rows_path}")
    probes = []
    try:
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("condition") == "correct" and row.get("group_id") not in probes:
                probes.append(row["group_id"])
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise AuditError(f"invalid evaluation rows: {rows_path}: {exc}") from exc
    if len(probes) != 4:
        raise AuditError(f"expected four correct probes in {rows_path}, got {probes}")
    return probes


def _load_cell(summary_path: Path, *, arm: str, scene: str, seed: int) -> dict:
    summary_path = summary_path.resolve()
    campaign, cell = _campaign_and_cell(summary_path)
    evaluation_path = campaign / "eval" / cell / "evaluation_summary.json"
    rows_path = campaign / "eval" / cell / "evaluation_rows.jsonl"
    manifest_path = campaign / "train" / cell / "run_manifest.json"
    summary = _read_json(summary_path)
    evaluation = _read_json(evaluation_path)
    manifest = _read_json(manifest_path)
    config = manifest.get("config")
    provenance = manifest.get("provenance")
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise AuditError(f"invalid run manifest: {manifest_path}")

    expected_config = {
        "steps": 1000,
        "views": 4,
        "target_lr_dropout": 0.5,
        "image_size": 256,
        "scale": 4,
        "sampling_steps": 50,
        "sampling_shift": 5.0,
        "allow_self_view_source": False,
        "epipolar_attention": "local_band",
        "epipolar_band": 1.5,
    }
    errors = []
    if evaluation.get("arm") != arm or config.get("arm") != arm:
        errors.append(f"arm must be {arm}")
    if evaluation.get("step") != 1000 or evaluation.get("groups") != 4:
        errors.append("evaluation must contain four probes at step 1000")
    if evaluation.get("inference_seeds") != [3302]:
        errors.append("inference seed must be [3302]")
    if evaluation.get("training_seed") != seed or provenance.get("training_seed") != seed:
        errors.append(f"training seed must be {seed}")
    if config.get("train_scenes") != [scene] or seed not in config.get("training_seeds", []):
        errors.append(f"config must train {scene} with seed {seed}")
    if config.get("dataset_kind", "nerf_synthetic") != "nerf_synthetic":
        errors.append("dataset kind must be nerf_synthetic")
    if config.get("image_factor", 1) != 1:
        errors.append("image factor must be 1")
    errors.extend(
        f"config {key} must be {value!r}"
        for key, value in expected_config.items()
        if config.get(key) != value
    )
    dataset = evaluation.get("dataset_manifest")
    if not isinstance(dataset, dict) or list(dataset) != [scene]:
        errors.append(f"dataset manifest must contain only {scene}")
    if errors:
        raise AuditError(f"{cell}: " + "; ".join(errors))

    paths = (summary_path, evaluation_path, rows_path, manifest_path)
    return {
        "cell": cell,
        "scene": scene,
        "seed": seed,
        "summary": summary,
        "evaluation": evaluation,
        "config": config,
        "probe_ids": _probe_ids(rows_path),
        "dataset_manifest": dataset,
        "inputs": {str(path): _sha256(path) for path in paths},
    }


def nonregression(a5_correct: dict, a6_correct: dict) -> dict:
    deltas = {metric: float(a6_correct[metric]) - float(a5_correct[metric])
              for metric in ("psnr", "ssim")}
    checks = {
        "psnr": deltas["psnr"] >= PSNR_NONREGRESSION - NUMERICAL_TOLERANCE,
        "ssim": deltas["ssim"] >= SSIM_NONREGRESSION - NUMERICAL_TOLERANCE,
    }
    return {"pass": all(checks.values()), "a5": a5_correct, "a6": a6_correct,
            "delta": deltas, "checks": checks}


def correspondence(gate: dict) -> dict:
    try:
        values = gate["deltas"]["mispaired_lr"]
        psnr = [float(value) for value in values["psnr"]["values"]]
        ssim = [float(value) for value in values["ssim"]["values"]]
        means = {"psnr": float(values["psnr"]["mean"]),
                 "ssim": float(values["ssim"]["mean"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError("summary is missing finite mispaired_lr deltas") from exc
    if len(psnr) != 4 or len(ssim) != 4:
        raise AuditError("mispaired_lr must contain four paired probe deltas")
    positive = {"psnr": sum(value > 0 for value in psnr),
                "ssim": sum(value > 0 for value in ssim)}
    checks = {
        "psnr_mean": means["psnr"] >= PSNR_CORRESPONDENCE,
        "ssim_mean": means["ssim"] >= SSIM_CORRESPONDENCE,
        "psnr_direction": positive["psnr"] >= DIRECTIONAL_PROBES,
        "ssim_direction": positive["ssim"] >= DIRECTIONAL_PROBES,
        "existing_a6_gate": gate.get("pass") is True,
    }
    return {"pass": all(checks.values()), "mean": means, "values": {"psnr": psnr,
            "ssim": ssim}, "positive_probes": positive, "checks": checks}


def _matched(a5: dict, a6: dict) -> dict:
    if a5["scene"] != a6["scene"] or a5["seed"] != a6["seed"]:
        raise AuditError("matched A5/A6 scene or seed differs")
    if a5["probe_ids"] != a6["probe_ids"]:
        raise AuditError(f"{a6['scene']}: matched probe IDs differ")
    if a5["dataset_manifest"] != a6["dataset_manifest"]:
        raise AuditError(f"{a6['scene']}: matched dataset hashes differ")
    return nonregression(_gate(a5["summary"])["correct"], _gate(a6["summary"])["correct"])


def _valid_a6(cell: dict) -> bool:
    summary = cell["summary"]
    return (
        summary.get("pass") is True
        and summary.get("status") == "PILOT_PASS"
        and summary.get("CAMERA_FUSION_PASS") is True
        and summary.get("integrity", {}).get("pass") is True
        and _gate(summary).get("pass") is True
    )


def build_review(a6_root: Path, a5_chair_summary: Path, a5_lego_summary: Path) -> dict:
    inputs = {}
    try:
        a6_root = a6_root.resolve()
        protocol_path = a6_root / "protocol.json"
        verdict_path = a6_root / "machine_verdict.json"
        protocol = _read_json(protocol_path)
        machine_verdict = _read_json(verdict_path)
        inputs.update({str(path): _sha256(path) for path in (protocol_path, verdict_path)})
        if protocol.get("source_revision") != EXPECTED_A6_REVISION:
            raise AuditError("A6 protocol source revision mismatch")
        if protocol.get("stage4_ready") is not False or machine_verdict.get("STAGE4_READY") is not False:
            raise AuditError("frozen A6 inputs must not already authorize Stage 4")

        a6 = {
            name: _load_cell(a6_root / "analysis" / name / "summary.json",
                             arm="A6", scene=scene, seed=seed)
            for name, (scene, seed) in A6_CELLS.items()
        }
        a5 = {
            "chair_seed42": _load_cell(a5_chair_summary, arm="A5", scene="chair", seed=42),
            "lego_seed42": _load_cell(a5_lego_summary, arm="A5", scene="lego", seed=42),
        }
        for cell in (*a6.values(), *a5.values()):
            inputs.update(cell["inputs"])

        matched = {
            name: _matched(a5[name], a6[a6_name])
            for name, (_, _, a6_name) in MATCHED.items()
        }
        cells = {
            name: {
                "existing_a6_valid": _valid_a6(cell),
                "correspondence": correspondence(_gate(cell["summary"])),
            }
            for name, cell in a6.items()
        }
        checks = {
            "a6_existing_gates": all(value["existing_a6_valid"] for value in cells.values()),
            "matched_a5_nonregression": all(value["pass"] for value in matched.values()),
            "symmetric_correspondence": all(
                value["correspondence"]["pass"] for value in cells.values()
            ),
        }
        passed = all(checks.values())
        return {
            "schema_version": 1,
            "verdict": "STAGE4_READY" if passed else "HOLD",
            "STAGE4_READY": passed,
            "next": "RUN_STAGE4" if passed else "ADJUST_SYMMETRIC_CORRESPONDENCE_LOSS",
            "thresholds": {
                "a6_minus_a5_psnr": PSNR_NONREGRESSION,
                "a6_minus_a5_ssim": SSIM_NONREGRESSION,
                "mispaired_lr_psnr": PSNR_CORRESPONDENCE,
                "mispaired_lr_ssim": SSIM_CORRESPONDENCE,
                "directional_probes": DIRECTIONAL_PROBES,
            },
            "checks": checks,
            "matched_nonregression": matched,
            "a6_cells": cells,
            "inputs": inputs,
        }
    except (AuditError, OSError, KeyError, TypeError, ValueError) as exc:
        return {
            "schema_version": 1,
            "verdict": "AUDIT_INVALID",
            "STAGE4_READY": False,
            "next": "FIX_AUDIT_INPUTS",
            "errors": [str(exc)],
            "inputs": inputs,
        }


def render_markdown(review: dict) -> str:
    lines = ["# Stage 4 大规模 3DSR 训练前评审", ""]
    if review["verdict"] == "AUDIT_INVALID":
        lines.extend(["## 结论", "", "**AUDIT_INVALID：输入或来源校验失败，禁止进入 Stage 4。**", ""])
        lines.extend(f"- {error}" for error in review.get("errors", []))
        return "\n".join(lines) + "\n"

    lines.extend([
        "## 结论", "",
        f"**{review['verdict']}：STAGE4_READY={str(review['STAGE4_READY']).lower()}。**", "",
        "A6 已能对错误 pose 作出响应，但只有 matched 质量不退化和双向图像—相机 correspondence 同时成立，才允许进入大规模训练。", "",
        "## Matched A5/A6 非退化", "",
        "| 单元 | ΔPSNR (dB) | ΔSSIM | 通过 |",
        "|---|---:|---:|:---:|",
    ])
    for name, value in review["matched_nonregression"].items():
        lines.append(f"| {name} | {value['delta']['psnr']:.6f} | {value['delta']['ssim']:.6f} | {'是' if value['pass'] else '否'} |")
    lines.extend(["", "## 对称 correspondence", "",
                  "| A6 单元 | mispaired_lr ΔPSNR | 正向 probes | mispaired_lr ΔSSIM | 正向 probes | 通过 |",
                  "|---|---:|---:|---:|---:|:---:|"])
    for name, value in review["a6_cells"].items():
        item = value["correspondence"]
        lines.append(
            f"| {name} | {item['mean']['psnr']:.6f} | {item['positive_probes']['psnr']}/4 | "
            f"{item['mean']['ssim']:.6f} | {item['positive_probes']['ssim']}/4 | "
            f"{'是' if item['pass'] and value['existing_a6_valid'] else '否'} |"
        )
    if not review["STAGE4_READY"]:
        lines.extend(["", "## 下一步", "",
                      "保持 pose 与 A6 结构，先调整损失以同时约束错误 pose 和错误辅助图像配对；不启动 A7/A8，不删除 pose，不运行 Stage 4 大规模训练。"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a6-root", type=Path, default=DEFAULT_A6_ROOT)
    parser.add_argument("--a5-chair-summary", type=Path, default=DEFAULT_A5_CHAIR)
    parser.add_argument("--a5-lego-summary", type=Path, default=DEFAULT_A5_LEGO)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = (args.output_dir or args.a6_root / "stage4_review").resolve()
    review = build_review(args.a6_root, args.a5_chair_summary, args.a5_lego_summary)
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage4_review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "stage4_review.md").write_text(render_markdown(review), encoding="utf-8")
    print(json.dumps({"verdict": review["verdict"], "STAGE4_READY": review["STAGE4_READY"],
                      "output": str(output)}, indent=2))
    return 2 if review["verdict"] == "AUDIT_INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
