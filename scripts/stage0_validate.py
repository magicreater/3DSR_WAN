#!/usr/bin/env python3
"""Run the complete Stage 0 data/VAE/DiT validation on one GPU."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from rl3dsr.data import MipNeRF360Adapter, NeRFSyntheticAdapter
from rl3dsr.data.tensor_prep import load_observations
from rl3dsr.data.temporal_fixture import make_synthetic_video
from rl3dsr.models.wan import WanCheckpoint, WanDiT, WanVAE
from rl3dsr.validation.temporal_impulse import run_temporal_impulse, sample_to_tensor


def _check_tensor(value: torch.Tensor) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device), "finite": bool(torch.isfinite(value).all())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nerf-scene", type=Path, required=True)
    parser.add_argument("--mip-scene", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real Wan Stage 0 validation")
    checkpoint = WanCheckpoint.from_dir(args.model_dir)
    nerf = NeRFSyntheticAdapter(args.nerf_scene)
    mip = MipNeRF360Adapter(args.mip_scene, image_factor=4)
    nerf_sequence = nerf.index("test")
    mip_sequence = mip.index("test")
    mip_rgb = mip.load_rgb(mip_sequence.observations[0])
    vae = WanVAE.from_checkpoint(checkpoint.model_dir, device=device, dtype=torch.bfloat16)
    dit = WanDiT.from_checkpoint(checkpoint.model_dir, device=device, dtype=torch.bfloat16)
    report: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "wan_upstream_commit": "9737cba9c1c3c4d04b33fcad41c111989865d315",
        "checkpoint": str(checkpoint.model_dir),
        "camera_convention": "T_world_from_camera, canonical OpenCV +X right +Y down +Z forward",
        "datasets": {"nerf_synthetic": str(args.nerf_scene), "mipnerf360": str(args.mip_scene)},
        "real_rgb_samples": {
            "nerf_first": list(nerf.load_rgb(nerf_sequence.observations[0]).shape),
            "mip_first": list(mip_rgb.shape),
        },
        "3d": {},
        "4d": {},
    }

    views = load_observations(nerf, nerf_sequence, args.resolution)
    report["3d"]["source_observations"] = len(nerf_sequence.observations)
    report["3d"]["mip_test_observations"] = len(mip_sequence.observations)
    for count in (1, 4, 8, 16):
        rgb = views[:, :, :count]
        with torch.inference_mode():
            latents = vae.encode_multiview(rgb)
            decoded = vae.decode_multiview(latents)
            dit_output = dit(latents, torch.tensor([500.0], device=device))
        report["3d"][f"V={count}"] = {
            "input": _check_tensor(rgb.to(device)),
            "latents": _check_tensor(latents),
            "decoded": _check_tensor(decoded),
            "dit": _check_tensor(dit_output),
            "decoded_range": [float(decoded.min()), float(decoded.max())],
            "decoded_std": float(decoded.std()),
        }

    baseline = views[:, :, :4]
    modified = baseline.clone()
    modified[:, :, 2] = (modified[:, :, 2] + 0.2).clamp(-1, 1)
    with torch.inference_mode():
        z0 = vae.encode_multiview(baseline)
        z1 = vae.encode_multiview(modified)
    delta = (z1 - z0).abs()
    report["3d"]["cross_view_isolation"] = {
        "changed_view": 2,
        "changed_view_max_abs_delta": float(delta[:, :, 2].max()),
        "unchanged_views_max_abs_delta": float(delta[:, :, [0, 1, 3]].max()),
        "passed": bool(delta[:, :, 2].max() > 1e-6 and delta[:, :, [0, 1, 3]].max() < 1e-5),
    }

    for frame_count in (1, 5, 9, 17):
        sample = make_synthetic_video(frame_count, height=args.resolution, width=args.resolution, seed=0)
        rgb = sample_to_tensor(sample, device=device)
        with torch.inference_mode():
            latents = vae.encode_video(rgb)
            decoded = vae.decode_video(latents)
            dit_output = dit(latents, torch.tensor([500.0], device=device))
        report["4d"][f"T={frame_count}"] = {
            "input": _check_tensor(rgb),
            "latents": _check_tensor(latents),
            "decoded": _check_tensor(decoded),
            "dit": _check_tensor(dit_output),
            "measured_T_prime": int(latents.shape[2]),
            "decoded_range": [float(decoded.min()), float(decoded.max())],
            "decoded_std": float(decoded.std()),
        }
    report["temporal_impulse"] = run_temporal_impulse(vae)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 0 Validation Report",
        "",
        "## Status",
        "",
        "This report is generated from real Wan2.1 VAE and DiT execution. No training, optimizer, backward pass, adapter, pose encoder, or geometry input is used.",
        "",
        "## Environment and Sources",
        "",
        f"- Wan upstream commit: `{report['wan_upstream_commit']}`",
        f"- Checkpoint: `{checkpoint.model_dir}`",
        f"- Runtime: `{report['python']}`, PyTorch `{report['torch']}`, CUDA `{report['cuda']}`, device `{report['gpu']}`",
        f"- Datasets: NeRF Synthetic `{args.nerf_scene}`; Mip-NeRF 360 `{args.mip_scene}`",
        "- Camera: `T_world_from_camera`, OpenCV camera axes (+X right, +Y down, +Z forward); source world units preserved.",
        "- DiT context: deterministic neutral tensor `[B,512,4096]`; T5 is not loaded.",
        "",
        "## Measured Results",
        "",
        "```json",
        json.dumps(report, indent=2),
        "```",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/stage0_validate.py --model-dir /data/linzizhuo/RL3DSR_WAN_REMO/models/Wan2.1-T2V-1.3B --nerf-scene /data/linzizhuo/RL3DSR_WAN_REMO/datasets/nerf_synthetic/chair --mip-scene /data/linzizhuo/RL3DSR_WAN_REMO/datasets/360_v2/counter --resolution 64 --report docs/stage0_validation.md",
        "```",
        "",
        "## Limitations",
        "",
        "- The 4D input is a deterministic synthetic video fixture because the repository has no real pose-annotated 4D video dataset.",
        "- The 3D VAE path isolates views; the joint DiT receives the regrouped view axis as its joint sequence axis.",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
