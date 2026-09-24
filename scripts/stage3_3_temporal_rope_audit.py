#!/usr/bin/env python3
"""In-memory counterfactual: neutralize Wan frame RoPE for one-step attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import stage3_experiment as stage3
from stage3_3_architecture_audit import DATA, PERMUTATIONS, PROBES, decoded_metrics, digest, load_group, setup


def run(output: Path):
    config, groups, hashes, checkpoint = setup('W3')
    runtime = stage3.load_runtime(
        config, model_dir=DATA / 'models/Wan2.1-T2V-1.3B',
        lq_source=Path('/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py'),
        lq_checkpoint=Path('/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt'),
        bridge_checkpoint=DATA / 'artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt',
        stage3_checkpoint=checkpoint, device='cuda')
    runtime.module.eval()
    perceptual = stage3._PerFrameLPIPS(runtime.device)
    model = runtime.dit.model
    original_freqs = model.freqs
    head_dim = model.dim // model.num_heads
    temporal_columns = (head_dim - 4 * (head_dim // 6)) // 2
    if temporal_columns <= 0 or temporal_columns >= original_freqs.shape[1]:
        raise RuntimeError('unexpected temporal RoPE layout')
    neutral = original_freqs.clone()
    neutral[:, :temporal_columns] = neutral[:1, :temporal_columns]
    rows = []
    sigma = torch.tensor([0.5], device=runtime.device)
    context = torch.zeros(1, 512, 4096, device=runtime.device, dtype=torch.bfloat16)
    try:
        with torch.inference_mode():
            for probe in PROBES:
                _, hr, _, camera = load_group(config, groups[probe], runtime.device)
                clean = runtime.vae.encode_multiview(hr)
                noise = torch.randn(clean.shape,
                    generator=torch.Generator(device=runtime.device).manual_seed(3302),
                    device=runtime.device, dtype=clean.dtype)
                noisy, timestep, _ = stage3.flow_matching_pair(clean, noise, sigma)
                for mode, frequencies in (('official', original_freqs), ('frame_rope_neutral', neutral)):
                    model.freqs = frequencies
                    base = runtime.dit(noisy, timestep, context)
                    _, base_metrics = decoded_metrics(runtime, perceptual, noisy, base, sigma, hr, clean)
                    repeat = runtime.dit(noisy, timestep, context)
                    repeat_max = float((base.float() - repeat.float()).abs().max())
                    for name, indices in PERMUTATIONS.items():
                        permutation = torch.tensor(indices)
                        changed = runtime.dit(stage3.permute_view_tensor(noisy, permutation),
                                              timestep, context)
                        changed = stage3.permute_view_tensor(
                            changed, stage3.inverse_view_permutation(permutation))
                        _, changed_metrics = decoded_metrics(
                            runtime, perceptual, noisy, changed, sigma, hr, clean)
                        delta = base.float() - changed.float()
                        rows.append({
                            'probe': probe, 'mode': mode, 'permutation': name,
                            'indices': indices, 'reference': base_metrics,
                            'permuted': changed_metrics,
                            'reference_minus_permuted_ssim': base_metrics['ssim'] - changed_metrics['ssim'],
                            'target_velocity_relative_delta': float(
                                delta[:, :, :1].norm() / base[:, :, :1].float().norm().clamp_min(1e-12)),
                            'velocity_max_abs_delta': float(delta.abs().max()),
                            'repeat_max_abs_delta': repeat_max,
                        })
                print(f'{probe} complete', flush=True)
    finally:
        model.freqs = original_freqs
    assert digest(checkpoint) == hashes['checkpoint']
    assert len(rows) == len(PROBES) * len(PERMUTATIONS) * 2
    payload = {'rows': rows, 'temporal_rope_columns': temporal_columns,
               'scope': 'one-step Wan-only diagnostic; not a proposed model or scored candidate',
               'checkpoint_sha256': hashes['checkpoint'], 'config_sha256': hashes['config'],
               'manifest_sha256': hashes['manifest'], 'source_sha256': digest(__file__),
               'sigma': 0.5, 'noise_seed': 3302}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
