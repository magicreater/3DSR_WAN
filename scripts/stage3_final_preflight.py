#!/usr/bin/env python3
"""Real-model, one-batch Stage 3 final candidate preflight."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torchmetrics.functional.image import structural_similarity_index_measure

import stage3_experiment as stage3
from rl3dsr.validation.stage3_protocol import load_stage3_config, nearest_view_indices


DATA = Path("/data/linzizhuo/RL3DSR_WAN_REMO")
PARENT = DATA / "artifacts/stage3_2_camera_causality_20260913/phase_c/train/rank/seed42/stage3_step_1000.pt"
BRIDGE = DATA / "artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt"
LQ_SOURCE = Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py")
LQ_CHECKPOINT = Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt")


def scores(decoded, target):
    a = (decoded[:, :, 0].float() + 1).mul(.5).clamp(0, 1)
    b = (target[:, :, 0].float() + 1).mul(.5).clamp(0, 1)
    mse = (a - b).square().mean()
    return {"psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
            "ssim": float(structural_similarity_index_measure(a, b, data_range=1.0))}


def run(config_path: Path, output: Path):
    config = load_stage3_config(config_path)
    assert config.arm == "A6" and config.dynamic_fusion and config.shared_multiview_rope
    stage3._seed_all(42)
    runtime = stage3.load_runtime(
        config, model_dir=DATA / "models/Wan2.1-T2V-1.3B",
        lq_source=LQ_SOURCE, lq_checkpoint=LQ_CHECKPOINT,
        bridge_checkpoint=BRIDGE, stage3_checkpoint=PARENT,
        model_only_initialization=True, reset_initialization_fusion=True,
        device="cuda",
    )
    runtime.module.train()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    adapter, sequence = stage3._scene_data(DATA / "datasets/nerf_synthetic/lego", config, stage3.Split.TRAIN)
    all_camera = stage3._camera(sequence.observations, config.image_size, torch.device("cpu"))
    indices = nearest_view_indices(all_camera, 0, config.views)
    _, hr, lr, camera = stage3._load_indices(DATA / "datasets/nerf_synthetic", "lego", "train",
                                              indices, config, runtime.device)
    with torch.no_grad():
        clean = runtime.vae.encode_multiview(hr)
    dropped = lr.clone()
    dropped[:, :, 0] = 0
    sigma = torch.tensor([.5], device=runtime.device)
    noise = torch.randn(clean.shape, generator=torch.Generator(device=runtime.device).manual_seed(3302),
                        device=runtime.device, dtype=clean.dtype)
    noisy, timestep, target = stage3.flow_matching_pair(clean, noise, sigma)
    wrong_indices = torch.tensor([0, *range(2, config.views), 1])
    wrong_camera = stage3._camera_with_auxiliary_permutation(camera, wrong_indices)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    prepared = runtime.module.prepare_multiview(dropped, camera, tuple(clean.shape[2:]),
                                                 (config.image_size, config.image_size))
    prediction = runtime.module.predict(runtime.dit, noisy, timestep, None, prepared,
                                        camera, tuple(clean.shape[2:]))
    wrong_prepared = runtime.module.prepare_multiview(dropped, wrong_camera, tuple(clean.shape[2:]),
                                                       (config.image_size, config.image_size))
    prediction_wrong = runtime.module.predict(runtime.dit, noisy, timestep, None, wrong_prepared,
                                              camera, tuple(clean.shape[2:]))
    wrong_lr = stage3.permute_view_tensor(dropped, wrong_indices)
    wrong_lr_prepared = runtime.module.prepare_multiview(wrong_lr, camera, tuple(clean.shape[2:]),
                                                          (config.image_size, config.image_size))
    prediction_lr_wrong = runtime.module.predict(runtime.dit, noisy, timestep, None, wrong_lr_prepared,
                                                 camera, tuple(clean.shape[2:]))
    flow, _, _, camera_rank = stage3._camera_pair_training_losses(
        prediction, prediction_wrong, target, margin_ratio=config.camera_rank_margin_ratio,
        target_view_only=True)
    _, _, _, lr_rank = stage3._camera_pair_training_losses(
        prediction, prediction_lr_wrong, target, margin_ratio=config.camera_rank_margin_ratio,
        target_view_only=True)
    loss = flow + config.camera_rank_weight * (
        config.symmetric_camera_fraction * camera_rank.mean()
        + (1 - config.symmetric_camera_fraction) * lr_rank.mean())
    loss.backward()
    trainable = tuple(p for p in runtime.module.parameters() if p.requires_grad)
    finite_gradients = all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    frozen = (all(p.grad is None and not p.requires_grad for p in runtime.dit.model.parameters())
              and all(p.grad is None and not p.requires_grad for p in runtime.vae.model.model.parameters()))
    peak_train_mib = torch.cuda.max_memory_allocated(runtime.device) / 2**20
    runtime.module.zero_grad(set_to_none=True)
    del loss, prediction, prediction_wrong, prediction_lr_wrong, prepared, wrong_prepared, wrong_lr_prepared
    torch.cuda.empty_cache()

    runtime.module.eval()
    original_freqs = runtime.dit.model.freqs
    with torch.inference_mode():
        sampled = stage3.sample_latents(runtime, dropped, camera, tuple(clean.shape),
                                        config.sampling_steps, config.image_size, seed=3302,
                                        sampling_shift=config.sampling_shift, dtype=torch.bfloat16)
        correct = scores(runtime.vae.decode_multiview(sampled), hr)
        joint = stage3.apply_joint_view_permutation(wrong_indices, lr=dropped,
                                                    fusion_camera=camera, geometry_camera=camera)
        permuted = stage3.sample_latents(runtime, joint["lr"], camera, tuple(clean.shape),
                                         config.sampling_steps, config.image_size,
                                         fusion_camera=joint["fusion_camera"],
                                         geometry_camera=joint["geometry_camera"],
                                         view_permutation=wrong_indices, seed=3302,
                                         sampling_shift=config.sampling_shift, dtype=torch.bfloat16)
        permuted = stage3.permute_view_tensor(permuted, joint["inverse_permutation"])
        reordered = scores(runtime.vae.decode_multiview(permuted), hr)
        phase_restored = runtime.dit.model.freqs is original_freqs
        # The default Wan call can move official CPU frequencies to CUDA.
        runtime.dit(noisy, timestep)
        native_phase_unchanged = torch.equal(runtime.dit.model.freqs.cpu(), original_freqs.cpu())
        empty_features = runtime.module.prepare_multiview(
            dropped, camera, tuple(clean.shape[2:]), (config.image_size, config.image_size),
            source_mask=torch.zeros(1, config.views, dtype=torch.bool, device=runtime.device))
        with torch.no_grad():
            runtime.module.fusion.output.weight.fill_(.01)
        pooled = torch.nn.functional.avg_pool2d(
            noisy.permute(0, 2, 1, 3, 4).reshape(config.views, 16, *noisy.shape[-2:]).float(), 2
        ).permute(0, 2, 3, 1).reshape(1, config.views, -1, 16)
        empty = runtime.module.fusion(empty_features.features, camera, empty_features.patch_grid,
                                      source_mask=empty_features.source_mask,
                                      latent_query=pooled, timestep=timestep)
        null_identity = torch.allclose(empty, empty_features.features, atol=1e-5)
    peak_total_mib = torch.cuda.max_memory_allocated(runtime.device) / 2**20
    result = {
        "views": config.views, "indices": indices, "target_lr_dropped": bool((dropped[:, :, 0] == 0).all()),
        "loss": float(flow.detach()), "finite_gradients": bool(finite_gradients),
        "frozen_wan_vae": bool(frozen), "null_identity": bool(null_identity),
        "phase_restored": bool(phase_restored),
        "native_phase_unchanged": bool(native_phase_unchanged),
        "peak_train_mib": peak_train_mib,
        "peak_total_mib": peak_total_mib, "correct": correct, "aux_permuted": reordered,
        "aux_permute_psnr_loss": correct["psnr"] - reordered["psnr"],
        "aux_permute_ssim_loss": correct["ssim"] - reordered["ssim"],
    }
    result["pass"] = (all(math.isfinite(v) for v in (result["loss"], peak_train_mib, peak_total_mib,
                                                    *correct.values(), *reordered.values()))
                      and all(result[k] for k in ("target_lr_dropped", "finite_gradients",
                                                    "frozen_wan_vae", "null_identity", "phase_restored",
                                                    "native_phase_unchanged"))
                      and peak_train_mib < 23500 and peak_total_mib < 23500)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["pass"]:
        raise RuntimeError("Stage 3 final preflight failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
