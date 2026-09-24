#!/usr/bin/env python3
"""One-batch Wan VAE backward feasibility check for the A6 structure arm."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
from torchmetrics.functional.image import structural_similarity_index_measure

import stage3_experiment as stage3
from rl3dsr.validation.stage3_protocol import load_stage3_config


ROOT = Path("/data/linzizhuo/RL3DSR_WAN_REMO")
W3 = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-target-rank/artifacts/stage3_3_a6sw3_20260923")
OUT = Path(__file__).resolve().parents[1] / "artifacts/stage3_3_structure_20260923/preflight_single.json"


def main():
    config = load_stage3_config(W3 / "config/w3_lego_seed42.json")
    runtime = stage3.load_runtime(
        config,
        model_dir=ROOT / "models/Wan2.1-T2V-1.3B",
        lq_source=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py"),
        lq_checkpoint=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"),
        bridge_checkpoint=ROOT / "artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt",
        stage3_checkpoint=W3 / "train/w3_lego_seed42/stage3_step_1000.pt",
        device="cuda",
    )
    runtime.module.train()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    manifest = json.loads((W3 / "manifest/w3_lego_seed42.json").read_text())
    group = next(row for row in manifest["groups"] if row["id"] == "lego:000")
    _, hr, lr, camera = stage3._load_indices(
        ROOT / "datasets/nerf_synthetic", "lego", "train", group["indices"], config, runtime.device
    )
    with torch.no_grad():
        clean = runtime.vae.encode_multiview(hr)
    noise = torch.randn(clean.shape, generator=torch.Generator(device=runtime.device).manual_seed(3302),
                        device=runtime.device, dtype=clean.dtype)
    sigma = torch.tensor([0.5], device=runtime.device)
    noisy, timestep, _ = stage3.flow_matching_pair(clean, noise, sigma)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    start = time.perf_counter()
    prepared = runtime.module.prepare_multiview(
        lr, camera, tuple(clean.shape[2:]), (config.image_size, config.image_size)
    )
    prediction = runtime.module.predict(
        runtime.dit, noisy, timestep, None, prepared, camera, tuple(clean.shape[2:])
    )
    x0_hat = noisy[:, :, :1] - sigma.reshape(1, 1, 1, 1, 1) * prediction[:, :, :1]
    decoded = runtime.vae.decode_multiview(x0_hat)[:, :, 0]
    reference = hr[:, :, 0]
    score = structural_similarity_index_measure(
        (decoded.float() + 1).mul(0.5), (reference.float() + 1).mul(0.5), data_range=1.0
    )
    loss = 1 - score
    parameters = [runtime.module.fusion.qkv.weight, runtime.module.fusion.output.weight,
                  *runtime.module.conditioner.bridge.parameters()]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    norm = math.sqrt(sum(float(gradient.float().square().sum()) for gradient in gradients
                         if gradient is not None))
    torch.cuda.synchronize(runtime.device)
    elapsed = time.perf_counter() - start
    result = {
        "checkpoint": str(W3 / "train/w3_lego_seed42/stage3_step_1000.pt"),
        "probe": group["id"], "indices": group["indices"], "noise_seed": 3302,
        "sigma": 0.5, "score": float(score.detach()), "gradient_norm": norm,
        "decoded_min": float(decoded.detach().min()), "decoded_max": float(decoded.detach().max()),
        "saturated_fraction": float(((decoded.detach() <= -1) | (decoded.detach() >= 1)).float().mean()),
        "peak_memory_mib": torch.cuda.max_memory_allocated(runtime.device) / 2**20,
        "forward_backward_seconds": elapsed,
        "wan_frozen": not any(parameter.requires_grad for parameter in runtime.dit.model.parameters()),
        "vae_frozen": not any(parameter.requires_grad for parameter in runtime.vae.model.model.parameters()),
    }
    result["pass"] = (all(math.isfinite(float(result[key])) for key in
                          ("score", "gradient_norm", "decoded_min", "decoded_max", "peak_memory_mib"))
                      and norm > 0 and result["peak_memory_mib"] < 20 * 1024
                      and result["wan_frozen"] and result["vae_frozen"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
