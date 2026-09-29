"""Paired, path-specific camera interventions for the frozen A5 ficus endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch
from PIL import Image

import stage3_experiment as stage3
from rl3dsr.validation.decoded_space import frame_metrics
from rl3dsr.validation.stage3_protocol import load_stage3_config


MODES = ("correct", "shuffle_fusion", "shuffle_geometry", "shuffle_all")


def _trace_means(trace: list[dict]) -> dict[str, dict[str, float]]:
    result = {}
    for block in sorted({key for step in trace for key in step}, key=int):
        result[block] = {}
        for key in sorted({name for step in trace for name in step.get(block, {})}):
            values = [step[block][key] for step in trace if key in step.get(block, {})]
            result[block][key] = mean(values)
    return result


def _save_target(path: Path, decoded: torch.Tensor) -> None:
    rgb = ((decoded[0, :, 0].float().clamp(-1, 1) + 1) * 127.5)
    image = rgb.byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image, "RGB").save(path)


def run(args: argparse.Namespace) -> None:
    config = load_stage3_config(args.config)
    if config.views != 4 or config.validation_scenes != ("ficus",):
        raise ValueError("camera path audit requires the locked four-view ficus protocol")
    output = stage3.prepare_output(args.output_dir)
    runtime = stage3.load_runtime(
        config, model_dir=args.model_dir, lq_source=args.lq_source,
        lq_checkpoint=args.lq_checkpoint, bridge_checkpoint=args.bridge_checkpoint,
        stage3_checkpoint=args.checkpoint, device=args.device,
    )
    runtime.module.eval()
    metric = stage3._PerFrameLPIPS(runtime.device)
    view_generator = torch.Generator().manual_seed(config.validation_inference_seed)
    rows = []
    with torch.inference_mode():
        for group in range(config.validation_groups_per_scene):
            indices, hr, lr, camera = stage3._load_group(
                args.dataset_root, "ficus", "validate", config, view_generator,
                runtime.device,
            )
            shape = tuple(runtime.vae.encode_multiview(hr).shape)
            camera_hash = stage3._camera_digest(camera)
            wrong_hashes = []
            noise_hashes = []
            correct_target = None
            for mode in MODES:
                # Reset the generator for each mode so the *same* auxiliary
                # permutation is routed through fusion, RRE, or both.
                intervention_seed = config.validation_inference_seed * 1000000 + group * 100
                generator = torch.Generator().manual_seed(intervention_seed)
                changed_lr, fusion_camera, geometry_camera, mask = stage3._intervention(
                    lr, camera, mode, generator,
                )
                assert torch.equal(changed_lr, lr)
                assert torch.equal(fusion_camera.K[:, :1], camera.K[:, :1])
                assert torch.equal(geometry_camera.K[:, :1], camera.K[:, :1])
                fusion_hash = stage3._camera_digest(fusion_camera)
                geometry_hash = stage3._camera_digest(geometry_camera)
                if mode != "correct":
                    wrong_hashes.append(geometry_hash if mode == "shuffle_geometry" else fusion_hash)
                if mode in {"shuffle_fusion", "shuffle_all"}:
                    assert fusion_hash != camera_hash
                if mode in {"shuffle_geometry", "shuffle_all"}:
                    assert geometry_hash != camera_hash
                latent, diagnostics = stage3.sample_latents(
                    runtime, changed_lr, camera, shape, config.sampling_steps,
                    config.image_size, fusion_camera=fusion_camera,
                    geometry_camera=geometry_camera, source_mask=mask,
                    seed=config.validation_inference_seed * 10000 + group,
                    sampling_shift=config.sampling_shift, return_diagnostics=True,
                )
                noise_hashes.append(hashlib.sha256(
                    diagnostics["initial_noise"].numpy().tobytes()
                ).hexdigest())
                decoded = runtime.vae.decode_multiview(latent)
                target_metrics = frame_metrics(decoded, hr, perceptual_metric=metric)[0]
                if mode == "correct":
                    correct_target = decoded[0, :, 0].detach().float().cpu()
                row = {
                    "group": group, "view_indices": indices, "mode": mode,
                    "psnr": float(target_metrics["psnr"]),
                    "ssim": float(target_metrics["ssim"]),
                    "lpips": float(target_metrics["lpips"]),
                    "pixel_mae_vs_correct": float((
                        decoded[0, :, 0].detach().float().cpu() - correct_target
                    ).abs().mean()),
                    "fusion_camera_sha256": fusion_hash,
                    "geometry_camera_sha256": geometry_hash,
                    "initial_noise_sha256": noise_hashes[-1],
                    "fusion_diagnostics": diagnostics["fusion_diagnostics"],
                    "geometry_trace_mean": _trace_means(diagnostics["geometry_trace"]),
                    "injection_trace_mean": _trace_means(diagnostics["injection_trace"]),
                }
                rows.append(row)
                _save_target(output / f"group{group}_{mode}.png", decoded)
            assert len(set(wrong_hashes)) == 1, "camera paths received different permutations"
            assert len(set(noise_hashes)) == 1, "camera paths received different initial noise"
    with (output / "camera_path_rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    summary = {mode: {
        key: mean(row[key] for row in rows if row["mode"] == mode)
        for key in ("psnr", "ssim", "lpips", "pixel_mae_vs_correct")
    } for mode in MODES}
    (output / "summary.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": stage3._sha256(args.checkpoint),
        "scene": "ficus", "seed": config.validation_inference_seed,
        "sampling_steps": config.sampling_steps,
        "groups": config.validation_groups_per_scene,
        "means": summary,
        "correct_minus_wrong": {mode: {
            key: summary["correct"][key] - summary[mode][key]
            for key in ("psnr", "ssim")
        } for mode in MODES[1:]},
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    stage3._add_runtime_paths(parser)
    run(parser.parse_args())
