#!/usr/bin/env python3
"""Paired 50-step target-view trajectory diagnostics for frozen A6 checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import stage3_experiment as stage3
from stage3_3_a6_path_audit import CELLS, DATA, PROBES, digest, load_group, setup, target_ssim
from rl3dsr.models.wan.sampling import FlowSamplingConfig, _scheduler


def ratio(reference, changed):
    a, b = reference.float(), changed.float()
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


def run(label, output):
    config, groups, hashes, checkpoint = setup(label)
    runtime = stage3.load_runtime(config, model_dir=DATA / 'models/Wan2.1-T2V-1.3B',
        lq_source=Path('/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py'),
        lq_checkpoint=Path('/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt'),
        bridge_checkpoint=DATA / 'artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt',
        stage3_checkpoint=checkpoint, device='cuda')
    runtime.module.eval()
    scheduler = _scheduler(FlowSamplingConfig(config.sampling_steps, config.sampling_shift), runtime.device)
    sigmas = [float(scheduler.sigmas[i]) for i in range(config.sampling_steps)]
    rows = []
    with torch.no_grad():
        for probe in PROBES:
            group = groups[probe]
            _, hr, raw_lr, camera = load_group(config, group, runtime.device)
            clean = runtime.vae.encode_multiview(hr)
            for dropped in (False, True):
                lr = raw_lr.clone()
                if dropped:
                    lr[:, :, 0] = 0
                traces = {}
                for mode, mode_index in (('correct', 0), ('mispaired_camera', 7), ('shuffle_fusion', 4)):
                    sample_seed = 3302 * 1_000_000 + group['anchor']
                    generator = torch.Generator().manual_seed(sample_seed * 10 + mode_index)
                    changed_lr, fusion_camera, geometry_camera, mask = stage3._intervention(lr, camera, mode, generator)
                    assert torch.equal(changed_lr, lr) and torch.equal(geometry_camera.K, camera.K)
                    sampled, trace = stage3.sample_latents(
                        runtime, changed_lr, camera, tuple(clean.shape), config.sampling_steps,
                        config.image_size, fusion_camera=fusion_camera,
                        geometry_camera=geometry_camera, source_mask=mask,
                        seed=sample_seed, sampling_shift=config.sampling_shift,
                        dtype=torch.bfloat16, return_diagnostics=True)
                    assert len(trace['velocity_trace']) == config.sampling_steps
                    traces[mode] = {'initial_noise': trace['initial_noise'],
                                    'velocities': [x[:, :, :1] for x in trace['velocity_trace']],
                                    'final_ssim': target_ssim(runtime.vae, sampled, hr),
                                    'final_target_mse': float((sampled[:, :, :1] - clean[:, :, :1]).float().square().mean())}
                correct = traces['correct']
                comparisons = {}
                for mode in ('mispaired_camera', 'shuffle_fusion'):
                    changed = traces[mode]
                    assert torch.equal(correct['initial_noise'], changed['initial_noise'])
                    differences = [ratio(a, b) for a, b in zip(correct['velocities'], changed['velocities'])]
                    assert all(math.isfinite(x) for x in differences)
                    comparisons[mode] = {'per_step_target_velocity_relative_delta': differences,
                         'mean_target_velocity_relative_delta': sum(differences) / len(differences),
                         'first_step_target_velocity_relative_delta': differences[0],
                         'final_correct_minus_wrong_ssim': correct['final_ssim'] - changed['final_ssim'],
                         'final_wrong_minus_correct_latent_mse': changed['final_target_mse'] - correct['final_target_mse']}
                rows.append({'probe': probe, 'indices': group['indices'], 'dropped': dropped,
                             'correct_final_ssim': correct['final_ssim'], 'comparisons': comparisons})
                print(f'{label} {probe} dropped={dropped} complete', flush=True)
    payload = {'label': label, 'input_sha256': hashes, 'inference_seed': 3302,
               'sample_steps': config.sampling_steps, 'sigmas': sigmas,
               'trace_comparison': 'paired initial noise; trajectories differ after first scheduler update',
               'rows': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=CELLS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.label, args.output)
