#!/usr/bin/env python3
"""Read-only attribution audit for the frozen A5/W1/W2/W3 Lego probes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

import stage3_experiment as stage3
import stage3_3_weight_gradient_audit as old_audit
import stage3_3_symmetric_rank as symmetric
from rl3dsr.validation.stage3_protocol import load_stage3_config


ROOT = Path(__file__).resolve().parents[1]
OLD = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-target-rank")
A5 = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-data-control/artifacts/stage3_3_data_control_20260918")
CAMPAIGN = ROOT / "artifacts/stage3_3_targetflow_20260923"
F1_CAMPAIGN = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-targetflow/artifacts/stage3_3_targetflow_20260923")
PROBES = ("lego:000", "lego:033", "lego:066", "lego:099")
SOURCES = {
    "A5": (A5, "lego_seed42"),
    "W1": (OLD / "artifacts/stage3_3_a6sw_20260922", "w1_lego_seed42"),
    "W2": (OLD / "artifacts/stage3_3_a6sw_20260922", "w2_lego_seed42"),
    "W3": (OLD / "artifacts/stage3_3_a6sw3_20260923", "w3_lego_seed42"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def paths(label, probe):
    campaign, cell = SOURCES[label]
    stem = campaign / "eval" / cell
    view = f"view_{int(probe.split(':')[1]):03d}"
    image = stem / "images" / "lego" / view
    return stem, image / "seed_3302", image / "reference"


def image_array(path):
    image = Image.open(path).convert("RGB")
    if image.size != (256, 256):
        raise RuntimeError(f"wrong image size: {path}: {image.size}")
    return np.asarray(image, dtype=np.float32) / 255.0


def image_audit(out):
    reference = {}
    result = {label: {} for label in SOURCES}
    row_index = {}
    for label, (campaign, cell) in SOURCES.items():
        stem = campaign / "eval" / cell
        evaluation = rows(stem / "evaluation_rows.jsonl")
        selected = [r for r in evaluation if r["condition"] in ("correct", "target_drop")]
        row_index[label] = {(r["group_id"], r["condition"]): r for r in selected}
        if len({r["group_id"] for r in selected}) != 4 or len(row_index[label]) != (4 if label == "A5" and not any(r["condition"] == "target_drop" for r in selected) else 8):
            raise RuntimeError(f"{label}: incomplete paired evaluation rows")
        if any(r["inference_seed"] != 3302 or r["train_seed"] != 42 for r in selected):
            raise RuntimeError(f"{label}: seed drift")
        for probe in PROBES:
            _, seed_dir, ref_dir = paths(label, probe)
            hr = image_array(ref_dir / "hr.png")
            if probe in reference and not np.array_equal(reference[probe], hr):
                raise RuntimeError(f"{label} {probe}: HR reference mismatch")
            reference[probe] = hr
            correct = image_array(seed_dir / "correct.png")
            drop_path = seed_dir / "target_drop.png"
            dropped = image_array(drop_path) if drop_path.exists() else None
            item = row_index[label][(probe, "correct")]
            if item["view_index"] != int(probe.split(":")[1]):
                raise RuntimeError(f"{label} {probe}: view index mismatch")
            if label != "A5" and item["view_indices"] != row_index["A5"][(probe, "correct")]["view_indices"]:
                raise RuntimeError(f"{label} {probe}: auxiliary view mismatch")
            if dropped is not None and (probe, "target_drop") not in row_index[label]:
                raise RuntimeError(f"{label} {probe}: target drop image has no metric row")
            result[label][probe] = {
                "psnr": item["psnr"], "ssim": item["ssim"],
                "lpips": item.get("lpips"), "view_indices": item["view_indices"],
                "hr_sha256": sha256(ref_dir / "hr.png"),
                "correct_sha256": sha256(seed_dir / "correct.png"),
            }
            if dropped is not None:
                result[label][probe]["target_drop_ssim"] = row_index[label][(probe, "target_drop")]["ssim"]
    out.mkdir(parents=True, exist_ok=True)
    for probe, hr in reference.items():
        images = {label: image_array(paths(label, probe)[1] / "correct.png") for label in SOURCES}
        errors = {label: np.abs(image - hr).mean(axis=2) for label, image in images.items()}
        scale = max(float(np.quantile(error, 0.99)) for error in errors.values())
        gray = np.asarray(Image.fromarray((hr * 255).astype(np.uint8)).convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        edge = gray > np.quantile(gray, 0.75)
        texture = np.asarray(Image.fromarray((hr * 255).astype(np.uint8)).convert("L").filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
        texture = np.abs(np.asarray(Image.fromarray((hr * 255).astype(np.uint8)).convert("L"), dtype=np.float32) - texture)
        textured = (texture > np.quantile(texture, 0.75)) & ~edge
        smooth = ~(edge | textured)
        for label, error in errors.items():
            result[label][probe]["mae_regions"] = {
                "edge": float(error[edge].mean()) if edge.any() else None,
                "texture": float(error[textured].mean()) if textured.any() else None,
                "other": float(error[smooth].mean()) if smooth.any() else None,
            }
        canvas = Image.new("RGB", (5 * 256, 3 * 286), "white")
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(("HR", *SOURCES)):
            array = hr if label == "HR" else images[label]
            canvas.paste(Image.fromarray((array * 255).astype(np.uint8)), (column * 256, 24))
            draw.text((column * 256 + 4, 4), label, fill="black")
            if label == "HR":
                continue
            error = np.uint8(np.clip(errors[label] / scale, 0, 1) * 255)
            canvas.paste(Image.fromarray(error).convert("RGB"), (column * 256, 310))
            drop_file = paths(label, probe)[1] / "target_drop.png"
            if drop_file.exists():
                drop = image_array(drop_file)
                canvas.paste(Image.fromarray((drop * 255).astype(np.uint8)), (column * 256, 596))
        draw.text((4, 290), f"absolute RGB error, common 99th-percentile scale {scale:.4f}", fill="black")
        draw.text((4, 576), "target_drop outputs; occlusion requires visual review", fill="black")
        canvas.save(out / f"{probe.replace(':', '_')}_comparison.png")
    means = {label: {metric: float(np.mean([result[label][probe][metric] for probe in PROBES])) for metric in ("psnr", "ssim")} for label in SOURCES}
    if abs(means["W3"]["psnr"] - 25.27201617365744) > 1e-8 or abs(means["W3"]["ssim"] - 0.8675757944583893) > 1e-8:
        raise RuntimeError("W3 summary metrics do not reproduce")
    payload = {"read_only_inputs": True, "probes": PROBES, "means": means, "per_probe": result,
               "error_scale": "per-probe common p99 across A5/W1/W2/W3; quantized PNG diagnostic only"}
    (out / "image_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def gradient_audit(out, args):
    campaign, cell = (SOURCES["W3"] if args.label == "W3"
                      else (F1_CAMPAIGN, "f1_lego_seed42"))
    config = load_stage3_config(campaign / "config" / f"{cell}.json")
    manifest_path = campaign / "manifest" / f"{cell}.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads((campaign / "protocol.json").read_text())
    if sha256(manifest_path) != protocol["cells"][cell]["manifest_sha256"]:
        raise RuntimeError("W3 manifest drift")
    if sha256(campaign / "config" / f"{cell}.json") != protocol["cells"][cell]["config_sha256"]:
        raise RuntimeError("W3 config drift")
    group_index = {row["id"]: row for row in manifest["groups"]}
    checkpoints = {
        "initialization": args.phase_c,
        "step500": campaign / "train" / cell / "stage3_step_0500.pt",
        "step1000": campaign / "train" / cell / "stage3_step_1000.pt",
    }
    if sha256(checkpoints["initialization"]) != protocol["phase_c_checkpoint_sha256"]:
        raise RuntimeError("Phase C parent drift")
    if sha256(checkpoints["step1000"]) != json.loads((campaign / "analysis" / cell / "summary.json").read_text())["checkpoint_sha256"]:
        raise RuntimeError("W3 checkpoint drift")
    output = {}
    for stage, checkpoint in checkpoints.items():
        initial = stage == "initialization"
        runtime = stage3.load_runtime(
            config, model_dir=args.model_dir, lq_source=args.lq_source,
            lq_checkpoint=args.lq_checkpoint, bridge_checkpoint=args.bridge_checkpoint,
            stage3_checkpoint=checkpoint, model_only_initialization=initial,
            reset_initialization_fusion=initial, device="cuda",
        )
        runtime.module.train()
        runtime.dit.model.eval().requires_grad_(False)
        runtime.vae.model.model.eval().requires_grad_(False)
        if any(p.requires_grad for p in runtime.dit.model.parameters()) or any(p.requires_grad for p in runtime.vae.model.model.parameters()):
            raise RuntimeError("Wan/VAE unexpectedly trainable")
        parameters = [runtime.module.fusion.qkv.weight, runtime.module.fusion.output.weight,
                      *runtime.module.conditioner.bridge.parameters()]
        output[stage] = []
        for probe_id in PROBES:
            group = group_index[probe_id]
            _, hr, raw_lr, camera = stage3._load_indices(
                args.dataset_root, "lego", "train", group["indices"], config, runtime.device
            )
            with torch.no_grad():
                clean = runtime.vae.encode_multiview(hr)
            permutation = stage3._deranged_auxiliary_indices(config.views, torch.Generator().manual_seed(6000))
            wrong_camera = stage3._camera_with_auxiliary_permutation(camera, permutation)
            for dropped in (False, True):
                lr = raw_lr.clone()
                if dropped:
                    lr[:, :, 0] = 0
                for sigma_value in (0.2, 0.5, 0.8):
                    prepared = [runtime.module.prepare_multiview(value, cam, tuple(clean.shape[2:]),
                                (config.image_size, config.image_size)) for value, cam in (
                                    (lr, camera), (lr, wrong_camera),
                                    (stage3.permute_view_tensor(lr, permutation), camera))]
                    noise = torch.randn(clean.shape, generator=torch.Generator(device=runtime.device).manual_seed(3302),
                                        device=runtime.device, dtype=clean.dtype)
                    noisy, timestep, target = stage3.flow_matching_pair(clean, noise, torch.tensor([sigma_value], device=runtime.device))
                    predictions = [runtime.module.predict(runtime.dit, noisy, timestep, None, item, camera,
                                    tuple(clean.shape[2:])) for item in prepared]
                    per_view = stage3.per_view_flow_losses(predictions[0], target)
                    _, _, _, camera_rank = stage3._camera_pair_training_losses(
                        predictions[0], predictions[1], target,
                        margin_ratio=config.camera_rank_margin_ratio, target_view_only=True)
                    _, _, _, lr_rank = stage3._camera_pair_training_losses(
                        predictions[0], predictions[2], target,
                        margin_ratio=config.camera_rank_margin_ratio, target_view_only=True)
                    fraction = config.target_view_flow_fraction
                    flow = (per_view.mean() if fraction is None else
                            (fraction * per_view[:, 0]
                             + (1 - fraction) * per_view[:, 1:].mean(dim=1)).mean())
                    objectives = {"flow": flow, "camera_rank": camera_rank.mean(), "lr_rank": lr_rank.mean()}
                    gradients = {key: torch.autograd.grad(value, parameters, retain_graph=key != "lr_rank",
                                 allow_unused=True) for key, value in objectives.items()}
                    output[stage].append({"probe_id": probe_id, "view_indices": group["indices"],
                        "sigma": sigma_value, "target_lr_dropped": dropped, "noise_seed": 3302,
                        "permutation": permutation.tolist(),
                        "per_view_flow": per_view.detach().mean(dim=0).float().cpu().tolist(),
                        "camera_hinge": float(camera_rank.detach().mean()), "lr_hinge": float(lr_rank.detach().mean()),
                        "gradients": old_audit.gradient_statistics(gradients, runtime.module.fusion.hidden_dim)})
        del runtime
        torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    payload = {"read_only_inputs": True, "label": args.label,
               "target_view_flow_fraction": config.target_view_flow_fraction,
               "checkpoint_sha256": {key: sha256(path) for key, path in checkpoints.items()},
               "grid": "4 probes x 2 target LR states x 3 sigma values x 3 checkpoints", "stages": output}
    (out / f"gradient_audit_{args.label.lower()}.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return {"rows": {key: len(value) for key, value in output.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("images", "gradients"))
    parser.add_argument("--output", type=Path, default=CAMPAIGN / "audit")
    parser.add_argument("--label", choices=("W3", "F1"), default="W3")
    parser.add_argument("--phase-c", type=Path, default=Path("/data/linzizhuo/RL3DSR_WAN_REMO/artifacts/stage3_2_camera_causality_20260913/phase_c/train/rank/seed42/stage3_step_1000.pt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/linzizhuo/RL3DSR_WAN_REMO/datasets/nerf_synthetic"))
    parser.add_argument("--model-dir", type=Path, default=Path("/data/linzizhuo/RL3DSR_WAN_REMO/models/Wan2.1-T2V-1.3B"))
    parser.add_argument("--lq-source", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py"))
    parser.add_argument("--lq-checkpoint", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"))
    parser.add_argument("--bridge-checkpoint", type=Path, default=Path("/data/linzizhuo/RL3DSR_WAN_REMO/artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt"))
    args = parser.parse_args()
    result = image_audit(args.output) if args.mode == "images" else gradient_audit(args.output, args)
    print(json.dumps(result if args.mode == "gradients" else result["means"], indent=2))


if __name__ == "__main__":
    main()
