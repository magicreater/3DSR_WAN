#!/usr/bin/env python3
"""Freeze one SSIM weight from eight deterministic W3-parent training batches."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import stage3_experiment as stage3
from rl3dsr.models.wan import FlowSamplingConfig, flow_matching_pair
from rl3dsr.models.wan.sampling import SigmaCycle
from rl3dsr.validation.stage3_protocol import load_stage3_config


ROOT = Path("/data/linzizhuo/RL3DSR_WAN_REMO")
W3 = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-target-rank/artifacts/stage3_3_a6sw3_20260923")
PARENT = ROOT / "artifacts/stage3_2_camera_causality_20260913/phase_c/train/rank/seed42/stage3_step_1000.pt"
OUT = Path(__file__).resolve().parents[1] / "artifacts/stage3_3_structure_20260923/calibration.json"


def gradient_norm(gradients):
    return math.sqrt(sum(float(g.float().square().sum()) for g in gradients if g is not None))


def main():
    stage3._seed_all(42)
    config = load_stage3_config(W3 / "config/w3_lego_seed42.json")
    runtime = stage3.load_runtime(
        config,
        model_dir=ROOT / "models/Wan2.1-T2V-1.3B",
        lq_source=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py"),
        lq_checkpoint=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"),
        bridge_checkpoint=ROOT / "artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt",
        stage3_checkpoint=PARENT, model_only_initialization=True,
        reset_initialization_fusion=True, device="cuda",
    )
    runtime.module.train()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    parameters = [p for p in runtime.module.parameters() if p.requires_grad]
    sigma_cycle = SigmaCycle(FlowSamplingConfig(config.sampling_steps, config.sampling_shift),
                             seed=1042, strategy="balanced")
    rows = []
    for index in range(8):
        seed = 6042 + index
        view_generator = torch.Generator().manual_seed(seed)
        noise_generator = torch.Generator(device=runtime.device).manual_seed(seed + 2000)
        dropout_generator = torch.Generator().manual_seed(seed + 4000)
        indices, hr, lr, camera = stage3._load_group(
            ROOT / "datasets/nerf_synthetic", "lego", "train", config,
            view_generator, runtime.device,
        )
        dropped = float(torch.rand((), generator=dropout_generator)) < config.target_lr_dropout
        if dropped:
            lr = lr.clone()
            lr[:, :, 0] = 0
        with torch.no_grad():
            clean = runtime.vae.encode_multiview(hr)
        sigma = sigma_cycle.next().reshape(1).to(runtime.device)
        noise = torch.randn(clean.shape, generator=noise_generator,
                            device=runtime.device, dtype=clean.dtype)
        noisy, timestep, target = flow_matching_pair(clean, noise, sigma)
        prepared = runtime.module.prepare_multiview(
            lr, camera, tuple(clean.shape[2:]), (config.image_size, config.image_size)
        )
        prediction = runtime.module.predict(
            runtime.dit, noisy, timestep, None, prepared, camera, tuple(clean.shape[2:])
        )
        flow = stage3.flow_matching_loss(prediction, target)
        structure = stage3.decoded_target_ssim_loss(runtime.vae, noisy, prediction, sigma, hr)
        flow_grad = torch.autograd.grad(flow, parameters, retain_graph=True, allow_unused=True)
        structure_grad = torch.autograd.grad(structure, parameters, allow_unused=True)
        f_norm, s_norm = gradient_norm(flow_grad), gradient_norm(structure_grad)
        if not all(math.isfinite(x) and x > 0 for x in (f_norm, s_norm, float(structure))):
            raise RuntimeError(f"invalid calibration gradient at batch {index}")
        rows.append({"batch": index, "seed": seed, "view_indices": indices,
                     "sigma": float(sigma), "target_lr_dropped": dropped,
                     "flow_loss": float(flow), "structure_loss": float(structure),
                     "flow_gradient_norm": f_norm, "structure_gradient_norm": s_norm,
                     "unweighted_ratio": s_norm / f_norm})
        print(f"calibration {index + 1}/8: flow={f_norm:.6g} structure={s_norm:.6g}", flush=True)
    weight = float(0.1 / np.median([r["unweighted_ratio"] for r in rows]))
    ratios = [weight * r["unweighted_ratio"] for r in rows]
    result = {"parent_checkpoint": str(PARENT), "parent_sha256": stage3._sha256(PARENT),
              "w3_config_sha256": stage3._sha256(W3 / "config/w3_lego_seed42.json"),
              "batches": rows, "target_gradient_ratio": 0.1,
              "weight": weight, "weighted_gradient_ratio_median": float(np.median(ratios)),
              "weighted_gradient_ratios": ratios,
              "peak_memory_mib": torch.cuda.max_memory_allocated(runtime.device) / 2**20}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "batches"}, indent=2))


if __name__ == "__main__":
    main()
