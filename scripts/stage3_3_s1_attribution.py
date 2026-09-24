#!/usr/bin/env python3
"""Fixed-noise W3/S1 audit of latent ranking and decoded camera SSIM."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

import stage3_experiment as stage3
from rl3dsr.validation.stage3_protocol import load_stage3_config


ROOT = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-structure")
W3 = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-target-rank/artifacts/stage3_3_a6sw3_20260923")
S1 = ROOT / "artifacts/stage3_3_structure_20260923"
CAMPAIGNS = {"W3": (W3, "w3_lego_seed42"), "S1": (S1, "s1_lego_seed42")}
PROBES = ("lego:000", "lego:033", "lego:066", "lego:099")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def grouped_gradients(module, structure, camera_rank):
    params = [module.fusion.qkv.weight, module.fusion.output.weight,
              *module.conditioner.bridge.parameters()]
    left = torch.autograd.grad(structure, params, retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(camera_rank, params, retain_graph=True, allow_unused=True)
    hidden = module.fusion.hidden_dim
    result = {}
    for group, indices in (("qk", (0,)), ("value", (0,)), ("output", (1,)),
                           ("bridge", tuple(range(2, len(params)))),
                           ("shared", tuple(range(len(params))))):
        pairs = []
        for index in indices:
            a, b = left[index], right[index]
            if index == 0 and group != "shared":
                part = slice(None, 2 * hidden) if group == "qk" else slice(2 * hidden, None)
                a = None if a is None else a[part]
                b = None if b is None else b[part]
            pairs.append((a, b))
        norm_a = math.sqrt(sum(float(a.float().square().sum()) for a, _ in pairs if a is not None))
        norm_b = math.sqrt(sum(float(b.float().square().sum()) for _, b in pairs if b is not None))
        dot = sum(float(a.float().flatten().dot(b.float().flatten())) for a, b in pairs
                  if a is not None and b is not None)
        cosine = None if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)
        if any(not math.isfinite(v) for v in (norm_a, norm_b, *(() if cosine is None else (cosine,)))):
            raise RuntimeError("nonfinite gradient statistic")
        result[group] = {"structure_norm": norm_a, "camera_rank_norm": norm_b,
                         "structure_camera_cosine": cosine}
    return result


def audit(label, device_index):
    campaign, cell = CAMPAIGNS[label]
    config_path = campaign / "config" / f"{cell}.json"
    manifest_path = campaign / "manifest" / f"{cell}.json"
    protocol = json.loads((campaign / "protocol.json").read_text())
    item = protocol["cells"][cell]
    if sha256(config_path) != item["config_sha256"] or sha256(manifest_path) != item["manifest_sha256"]:
        raise RuntimeError("frozen config or manifest drift")
    config = load_stage3_config(config_path)
    manifest = json.loads(manifest_path.read_text())
    groups = {row["id"]: row for row in manifest["groups"]}
    assert tuple(manifest["subsets"]["probe"]) == PROBES
    output = {"label": label, "config_sha256": sha256(config_path),
              "manifest_sha256": sha256(manifest_path), "stages": {}}
    for step in (500, 1000):
        checkpoint = campaign / "train" / cell / f"stage3_step_{step:04d}.pt"
        runtime = stage3.load_runtime(
            config, model_dir=Path("/data/linzizhuo/RL3DSR_WAN_REMO/models/Wan2.1-T2V-1.3B"),
            lq_source=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py"),
            lq_checkpoint=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"),
            bridge_checkpoint=Path("/data/linzizhuo/RL3DSR_WAN_REMO/artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt"),
            stage3_checkpoint=checkpoint, device="cuda",
        )
        runtime.module.train()
        runtime.dit.model.eval().requires_grad_(False)
        runtime.vae.model.model.eval().requires_grad_(False)
        if any(p.requires_grad for p in itertools.chain(runtime.dit.model.parameters(),
                                                       runtime.vae.model.model.parameters())):
            raise RuntimeError("Wan/VAE unexpectedly trainable")
        rows = []
        for probe_id in PROBES:
            group = groups[probe_id]
            _, hr, raw_lr, camera = stage3._load_indices(
                Path("/data/linzizhuo/RL3DSR_WAN_REMO/datasets/nerf_synthetic"),
                "lego", "train", group["indices"], config, runtime.device)
            with torch.no_grad():
                clean = runtime.vae.encode_multiview(hr)
            permutation = stage3._deranged_auxiliary_indices(
                config.views, torch.Generator().manual_seed(6000))
            wrong_camera = stage3._camera_with_auxiliary_permutation(camera, permutation)
            for dropped in (False, True):
                lr = raw_lr.clone()
                if dropped:
                    lr[:, :, 0] = 0
                prepared = runtime.module.prepare_multiview(
                    lr, camera, tuple(clean.shape[2:]), (config.image_size, config.image_size))
                wrong_prepared = runtime.module.prepare_multiview(
                    lr, wrong_camera, tuple(clean.shape[2:]), (config.image_size, config.image_size))
                for sigma_value in (0.2, 0.5, 0.8):
                    noise = torch.randn(clean.shape, generator=torch.Generator(device=runtime.device).manual_seed(3302),
                                        device=runtime.device, dtype=clean.dtype)
                    sigma = torch.tensor([sigma_value], device=runtime.device)
                    noisy, timestep, target = stage3.flow_matching_pair(clean, noise, sigma)
                    correct = runtime.module.predict(runtime.dit, noisy, timestep, None,
                                                     prepared, camera, tuple(clean.shape[2:]))
                    wrong = runtime.module.predict(runtime.dit, noisy, timestep, None,
                                                   wrong_prepared, camera, tuple(clean.shape[2:]))
                    e_correct, e_wrong, rank = stage3.camera_pair_ranking_loss(
                        correct[:, :, :1], wrong[:, :, :1], target[:, :, :1],
                        margin_ratio=config.camera_rank_margin_ratio)
                    structure = stage3.decoded_target_ssim_loss(runtime.vae, noisy, correct, sigma, hr)
                    with torch.no_grad():
                        wrong_ssim = 1 - stage3.decoded_target_ssim_loss(
                            runtime.vae, noisy, wrong, sigma, hr)
                    gradients = grouped_gradients(runtime.module, structure, rank.mean())
                    rows.append({"probe_id": probe_id, "view_indices": group["indices"],
                                 "target_lr_dropped": dropped, "sigma": sigma_value,
                                 "noise_seed": 3302, "permutation": permutation.tolist(),
                                 "latent_correct_mse": float(e_correct.mean().detach()),
                                 "latent_wrong_mse": float(e_wrong.mean().detach()),
                                 "latent_rank_hinge": float(rank.mean().detach()),
                                 "decoded_correct_ssim": float(1 - structure.detach()),
                                 "decoded_wrong_ssim": float(wrong_ssim),
                                 "decoded_camera_margin": float(1 - structure.detach() - wrong_ssim),
                                 "gradients": gradients})
                    del correct, wrong, structure, gradients
        output["stages"][str(step)] = {"checkpoint_sha256": sha256(checkpoint), "rows": rows}
        del runtime
        torch.cuda.empty_cache()
    path = S1 / "audit" / f"s1_attribution_{label.lower()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=CAMPAIGNS)
    parser.add_argument("--device-index", type=int, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES selects the exclusive physical GPU.
    print(audit(args.label, args.device_index), flush=True)
