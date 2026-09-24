#!/usr/bin/env python3
"""Audit S2 camera responses at the same latent state without training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import stage3_experiment as stage3
from stage3_3_a6_path_audit import CELLS, DATA, PROBES, digest, load_group, setup
from rl3dsr.models.wan.sampling import _scheduler
from rl3dsr.validation.decoded_space import frame_metrics, velocity_to_clean


ROOT = Path('/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-structure')
S2 = ROOT / 'artifacts/stage3_3_s2_decoded_pair_20260924'
CELLS['S2'] = (S2, 's2_lego_seed42', S2 / 'train/s2_lego_seed42/stage3_step_1000.pt')
PRIOR = S2 / 'analysis/s2_lego_seed42/summary.json'
MODES = ('correct', 'shuffle_fusion', 'mispaired_camera')
SAMPLE_STEPS = (0, 10, 25, 40, 48)
SIGMAS = (0.2, 0.5, 0.8)


def direction(correct: torch.Tensor, wrong: torch.Tensor, hr: torch.Tensor) -> dict:
    """Measure whether the decoded camera change points toward the HR image."""
    c = correct.float().clamp(-1, 1)
    w = wrong.float().clamp(-1, 1)
    h = hr.float().clamp(-1, 1)
    delta = (c - w).flatten()
    residual = (h - w).flatten()
    dot = float(torch.dot(delta, residual))
    denominator = float(delta.norm() * residual.norm())
    return {
        'cosine_to_hr': None if denominator == 0 else dot / denominator,
        'pixel_mse_gain': float((w - h).square().mean() - (c - h).square().mean()),
    }


def save_error_map(path: Path, correct: torch.Tensor, wrong: torch.Tensor, hr: torch.Tensor) -> None:
    """Use one fixed display scale for both absolute error maps."""
    def rgb(x):
        x = x[0, :, 0].detach().float().clamp(-1, 1).add(1).mul(0.5)
        return x.permute(1, 2, 0).cpu().numpy()

    reference, a, b = rgb(hr), rgb(correct), rgb(wrong)
    scale = 4.0
    strip = np.concatenate((a, b, reference,
                            np.clip(np.abs(a - reference) * scale, 0, 1),
                            np.clip(np.abs(b - reference) * scale, 0, 1)), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.rint(strip * 255))).save(path)


def metrics(vae, metric, latent: torch.Tensor, hr: torch.Tensor):
    decoded = vae.decode_multiview(latent[:, :, :1])
    return decoded, frame_metrics(decoded, hr[:, :, :1], perceptual_metric=metric)[0]


def paired(runtime, metric, sample, timestep, sigma, prepared, camera, clean, hr, context,
           *, map_path: Path | None = None):
    results = {}
    decoded = {}
    for mode in MODES:
        velocity = runtime.module.predict(runtime.dit, sample, timestep, context,
                                          prepared[mode], camera, tuple(clean.shape[2:]))
        x0 = velocity_to_clean(sample, velocity, sigma)
        image, image_metrics = metrics(runtime.vae, metric, x0, hr)
        results[mode] = {
            'latent_x0_mse': float((x0[:, :, :1].float() - clean[:, :, :1].float()).square().mean()),
            'metrics': {key: image_metrics[key] for key in ('mae', 'psnr', 'ssim', 'lpips')},
        }
        decoded[mode] = image
    correct = results['correct']
    for mode in MODES[1:]:
        wrong = results[mode]
        wrong['correct_minus_wrong_psnr'] = correct['metrics']['psnr'] - wrong['metrics']['psnr']
        wrong['correct_minus_wrong_ssim'] = correct['metrics']['ssim'] - wrong['metrics']['ssim']
        wrong['wrong_minus_correct_lpips'] = wrong['metrics']['lpips'] - correct['metrics']['lpips']
        wrong['wrong_minus_correct_latent_x0_mse'] = wrong['latent_x0_mse'] - correct['latent_x0_mse']
        wrong['direction'] = direction(decoded['correct'], decoded[mode], hr[:, :, :1])
    if map_path is not None:
        save_error_map(map_path, decoded['correct'], decoded['shuffle_fusion'], hr[:, :, :1])
    return results


def sampler_with_states(saved: dict):
    """Run the same UniPC calls as sample_conditioned_flow, retaining five input states."""
    def sampler(initial_noise, condition_features, context, *, predict_velocity, config):
        scheduler = _scheduler(config, initial_noise.device)
        sample = initial_noise.clone()
        for index, timestep in enumerate(scheduler.timesteps):
            if index in SAMPLE_STEPS:
                saved[index] = (sample.detach().clone(), float(timestep), float(scheduler.sigmas[index]))
            model_timestep = torch.full((sample.shape[0],), float(timestep),
                                         device=sample.device, dtype=torch.float32)
            velocity = predict_velocity(sample, model_timestep, context, condition_features)
            if velocity.shape != sample.shape or not torch.isfinite(velocity).all():
                raise RuntimeError('invalid sampler velocity')
            sample = scheduler.step(velocity, timestep, sample, return_dict=False)[0]
        return sample
    return sampler


def run(label: str, output: Path, probes: tuple[str, ...]):
    config, groups, hashes, checkpoint = setup(label)
    prior = json.loads(PRIOR.read_text())
    protocol = json.loads((S2 / 'protocol.json').read_text())
    cell = protocol['cells']['s2_lego_seed42']
    assert prior['status'] == 'HOLD' and prior['checkpoint_sha256'] == hashes['checkpoint']
    assert hashes['config'] == cell['config_sha256']
    assert hashes['manifest'] == cell['manifest_sha256']
    runtime = stage3.load_runtime(config, model_dir=DATA / 'models/Wan2.1-T2V-1.3B',
        lq_source=Path('/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py'),
        lq_checkpoint=Path('/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt'),
        bridge_checkpoint=DATA / 'artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt',
        stage3_checkpoint=checkpoint, device='cuda')
    runtime.module.eval()
    assert not any(p.requires_grad for p in runtime.vae.model.model.parameters())
    metric = stage3._PerFrameLPIPS(runtime.device)
    rows = []
    with torch.inference_mode():
        for probe in probes:
            group = groups[probe]
            _, hr, raw_lr, camera = load_group(config, group, runtime.device)
            clean = runtime.vae.encode_multiview(hr)
            _, oracle_metrics = metrics(runtime.vae, metric, clean, hr)
            for dropped in (False, True):
                lr = raw_lr.clone()
                if dropped:
                    lr[:, :, 0] = 0
                prepared = {}
                sample_seed = 3302 * 1_000_000 + group['anchor']
                for mode, mode_index in (('correct', 0), ('shuffle_fusion', 4), ('mispaired_camera', 7)):
                    generator = torch.Generator().manual_seed(sample_seed * 10 + mode_index)
                    changed_lr, fusion_camera, geometry_camera, mask = stage3._intervention(
                        lr, camera, mode, generator)
                    assert torch.equal(changed_lr, lr)
                    assert torch.equal(geometry_camera.K, camera.K)
                    assert torch.equal(geometry_camera.T_world_from_camera, camera.T_world_from_camera)
                    prepared[mode] = runtime.module.prepare_multiview(
                        changed_lr, fusion_camera, tuple(clean.shape[2:]),
                        (config.image_size, config.image_size), source_mask=mask)
                for sigma_value in SIGMAS:
                    sigma = torch.tensor([sigma_value], device=runtime.device)
                    noise = torch.randn(clean.shape,
                                        generator=torch.Generator(device=runtime.device).manual_seed(3302),
                                        device=runtime.device, dtype=clean.dtype)
                    noisy, timestep, _ = stage3.flow_matching_pair(clean, noise, sigma)
                    map_path = None
                    if sigma_value == 0.5:
                        map_path = output.parent / 'error_maps' / f'{label.lower()}_{probe.split(":")[1]}_{"drop" if dropped else "keep"}_sigma05.png'
                    result = paired(runtime, metric, noisy, timestep, sigma, prepared,
                                    camera, clean, hr, None, map_path=map_path)
                    rows.append({'probe': probe, 'indices': group['indices'], 'dropped': dropped,
                                 'state': 'forward_noise', 'sigma': sigma_value,
                                 'sampler_step': None, 'oracle_metrics': oracle_metrics,
                                 'conditions': result})
                states = {}
                final = stage3.sample_latents(runtime, lr, camera, tuple(clean.shape),
                    config.sampling_steps, config.image_size, seed=sample_seed,
                    sampling_shift=config.sampling_shift, dtype=torch.bfloat16,
                    sampler=sampler_with_states(states))
                assert tuple(states) == SAMPLE_STEPS
                _, final_metrics = metrics(runtime.vae, metric, final, hr)
                context = torch.zeros(1, 512, 4096, device=runtime.device, dtype=torch.bfloat16)
                for index, (sample, timestep_value, sigma_value) in states.items():
                    timestep = torch.tensor([timestep_value], device=runtime.device)
                    sigma = torch.tensor([sigma_value], device=runtime.device)
                    map_path = None
                    if index == 25:
                        map_path = output.parent / 'error_maps' / f'{label.lower()}_{probe.split(":")[1]}_{"drop" if dropped else "keep"}_sample25.png'
                    result = paired(runtime, metric, sample, timestep, sigma, prepared,
                                    camera, clean, hr, context, map_path=map_path)
                    rows.append({'probe': probe, 'indices': group['indices'], 'dropped': dropped,
                                 'state': 'correct_trajectory', 'sigma': sigma_value,
                                 'sampler_step': index, 'oracle_metrics': oracle_metrics,
                                 'correct_final_metrics': final_metrics,
                                 'conditions': result})
            print(f'{label} {probe} complete', flush=True)
    payload = {'label': label, 'input_sha256': hashes, 'prior_review_sha256': digest(PRIOR),
               'source_sha256': digest(__file__), 'inference_seed': 3302,
               'sampling_steps': config.sampling_steps, 'sample_step_indices': SAMPLE_STEPS,
               'probes': probes,
               'sigmas': SIGMAS, 'metric': 'frame_metrics (frozen evaluation implementation)',
               'rows': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(output, flush=True)


def self_check():
    hr = torch.ones(1, 3, 1, 2, 2)
    wrong = torch.zeros_like(hr)
    correct = hr * 0.5
    value = direction(correct, wrong, hr)
    assert math.isclose(value['cosine_to_hr'], 1.0, abs_tol=1e-6)
    assert value['pixel_mse_gain'] > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=(*CELLS, 'self-check'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probes', nargs='+', choices=PROBES, default=PROBES)
    args = parser.parse_args()
    if args.label == 'self-check':
        self_check()
    elif args.output is None:
        parser.error('--output is required for model audits')
    else:
        run(args.label, args.output, tuple(args.probes))
