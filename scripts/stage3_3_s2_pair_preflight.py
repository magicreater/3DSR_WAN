#!/usr/bin/env python3
"""Check that the decoded camera hinge can update the trainable adapter."""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import stage3_experiment as stage3
import stage3_3_s2 as s2
from rl3dsr.validation.stage3_protocol import load_stage3_config


def main():
    config = load_stage3_config(s2.s1.CAMPAIGN / "config/s1_lego_seed42.json")
    checkpoint = s2.s1.CAMPAIGN / "train/s1_lego_seed42/stage3_step_1000.pt"
    runtime = stage3.load_runtime(
        config, model_dir=s2.s1.DATA / "models/Wan2.1-T2V-1.3B",
        lq_source=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py"),
        lq_checkpoint=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"),
        bridge_checkpoint=s2.s1.DATA / "artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt",
        stage3_checkpoint=checkpoint, device="cuda",
    )
    runtime.module.train()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    manifest = json.loads((s2.CAMPAIGN / "manifest/s2_lego_seed42.json").read_text())
    group = next(row for row in manifest["groups"] if row["id"] == "lego:099")
    _, hr, lr, camera = stage3._load_indices(
        s2.s1.DATA / "datasets/nerf_synthetic", "lego", "train",
        group["indices"], config, runtime.device,
    )
    with torch.no_grad():
        clean = runtime.vae.encode_multiview(hr)
    noise = torch.randn(clean.shape, device=runtime.device, dtype=clean.dtype,
                        generator=torch.Generator(device=runtime.device).manual_seed(3302))
    sigma = torch.tensor([0.5], device=runtime.device)
    noisy, timestep, _ = stage3.flow_matching_pair(clean, noise, sigma)
    wrong_camera = stage3.derange_auxiliary_fusion_camera(
        camera, torch.Generator().manual_seed(3302)
    )
    def predict(paired_camera):
        prepared = runtime.module.prepare_multiview(
            lr, paired_camera, tuple(clean.shape[2:]), (config.image_size, config.image_size)
        )
        return runtime.module.predict(
            runtime.dit, noisy, timestep, None, prepared, camera, tuple(clean.shape[2:])
        )
    torch.cuda.reset_peak_memory_stats(runtime.device)
    correct = stage3.decoded_target_ssim_loss(runtime.vae, noisy, predict(camera), sigma, hr)
    wrong = stage3.decoded_target_ssim_loss(runtime.vae, noisy, predict(wrong_camera), sigma, hr)
    paired = F.relu(0.0003 + correct - wrong)
    parameters = [runtime.module.fusion.qkv.weight, runtime.module.fusion.output.weight,
                  *runtime.module.conditioner.bridge.parameters()]
    gradients = torch.autograd.grad(paired, parameters, allow_unused=True)
    norm = math.sqrt(sum(float(g.float().square().sum()) for g in gradients if g is not None))
    result = {"probe": group["id"], "checkpoint_sha256": s2.prior.sha256_file(checkpoint),
              "correct_ssim": 1-float(correct.detach()), "wrong_ssim": 1-float(wrong.detach()),
              "paired_loss": float(paired.detach()), "paired_gradient_norm": norm,
              "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(runtime.device)/2**20,
              "wan_frozen": not any(p.requires_grad for p in runtime.dit.model.parameters()),
              "vae_frozen": not any(p.requires_grad for p in runtime.vae.model.model.parameters())}
    result["pass"] = (all(math.isfinite(result[k]) for k in
                          ("correct_ssim", "wrong_ssim", "paired_loss", "paired_gradient_norm"))
                      and result["paired_loss"] > 0 and result["paired_gradient_norm"] > 0
                      and result["wan_frozen"] and result["vae_frozen"])
    output = s2.CAMPAIGN / "preflight_reset/paired_gradient_099.json"
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
