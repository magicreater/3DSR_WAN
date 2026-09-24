#!/usr/bin/env python3
"""Read-only, paired audit of view-order dependence in the A6 inference path."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import stage3_experiment as stage3
from stage3_3_a6_path_audit import CELLS, DATA, PROBES, digest, load_group, setup
from stage3_3_s2_direction_audit import save_error_map
from rl3dsr.validation.decoded_space import frame_metrics, velocity_to_clean


ROOT = Path('/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-structure')
S2 = ROOT / 'artifacts/stage3_3_s2_decoded_pair_20260924'
CELLS['S2'] = (S2, 's2_lego_seed42', S2 / 'train/s2_lego_seed42/stage3_step_1000.pt')
PERMUTATIONS = {
    'swap_aux': (0, 2, 1, 3),
    'cycle_aux': (0, 2, 3, 1),
    'move_target': (1, 0, 2, 3),
}
PATHS = ('wan', 'geometry', 'bridge', 'full')


def tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(value.shape)).encode())
    h.update(str(value.dtype).encode())
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def predict(runtime, path, noisy, timestep, lr, camera, latent_shape, image_size, context):
    if path == 'wan':
        return runtime.dit(noisy, timestep, context)
    if path == 'geometry':
        prepared = None
    elif path == 'bridge':
        prepared = runtime.module.conditioner.multiview_features(
            lr, conditioning_size=(image_size, image_size), latent_shape=latent_shape)
    elif path == 'full':
        prepared = runtime.module.prepare_multiview(
            lr, camera, latent_shape, (image_size, image_size))
    else:
        raise ValueError(path)
    return runtime.module.predict(
        runtime.dit, noisy, timestep, context, prepared, camera, latent_shape)


def decoded_metrics(runtime, perceptual, noisy, velocity, sigma, hr, clean):
    x0 = velocity_to_clean(noisy, velocity, sigma)
    image = runtime.vae.decode_multiview(x0[:, :, :1])
    metrics = frame_metrics(image, hr[:, :, :1], perceptual_metric=perceptual)[0]
    return image, {
        **{key: float(metrics[key]) for key in ('psnr', 'ssim', 'lpips')},
        'latent_x0_mse': float((x0[:, :, :1].float() - clean[:, :, :1].float()).square().mean()),
    }


def self_check():
    x = torch.arange(4).reshape(1, 1, 4, 1, 1)
    for values in PERMUTATIONS.values():
        permutation = torch.tensor(values)
        assert sorted(values) == [0, 1, 2, 3]
        restored = stage3.permute_view_tensor(
            stage3.permute_view_tensor(x, permutation),
            stage3.inverse_view_permutation(permutation))
        assert torch.equal(restored, x)
    assert set(PATHS) == {'wan', 'geometry', 'bridge', 'full'}


def run(label: str, output: Path, probes: tuple[str, ...]):
    self_check()
    config, groups, hashes, checkpoint = setup(label)
    if config.views != 4:
        raise ValueError('frozen A6 audit requires four views')
    runtime = stage3.load_runtime(
        config, model_dir=DATA / 'models/Wan2.1-T2V-1.3B',
        lq_source=Path('/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py'),
        lq_checkpoint=Path('/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt'),
        bridge_checkpoint=DATA / 'artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt',
        stage3_checkpoint=checkpoint, device='cuda')
    runtime.module.eval()
    assert not any(p.requires_grad for p in runtime.dit.model.parameters())
    assert not any(p.requires_grad for p in runtime.vae.model.model.parameters())
    perceptual = stage3._PerFrameLPIPS(runtime.device)
    rows = []
    inputs = {}
    context = torch.zeros(1, 512, 4096, device=runtime.device, dtype=torch.bfloat16)
    sigma = torch.tensor([0.5], device=runtime.device)
    with torch.inference_mode():
        for probe in probes:
            group = groups[probe]
            _, hr, raw_lr, camera = load_group(config, group, runtime.device)
            clean = runtime.vae.encode_multiview(hr)
            noise = torch.randn(clean.shape,
                generator=torch.Generator(device=runtime.device).manual_seed(3302),
                device=runtime.device, dtype=clean.dtype)
            noisy, timestep, _ = stage3.flow_matching_pair(clean, noise, sigma)
            inputs[probe] = {
                'view_indices': group['indices'], 'hr_sha256': tensor_digest(hr),
                'lr_sha256': tensor_digest(raw_lr), 'K_sha256': tensor_digest(camera.K),
                'pose_sha256': tensor_digest(camera.T_world_from_camera),
                'clean_sha256': tensor_digest(clean), 'noise_sha256': tensor_digest(noise),
            }
            for dropped in (False, True):
                lr = raw_lr.clone()
                if dropped:
                    lr[:, :, 0] = 0
                for path in PATHS:
                    base = predict(runtime, path, noisy, timestep, lr, camera,
                                   tuple(clean.shape[2:]), config.image_size, context)
                    if not torch.isfinite(base).all():
                        raise RuntimeError('nonfinite reference velocity')
                    base_image, base_metrics = decoded_metrics(
                        runtime, perceptual, noisy, base, sigma, hr, clean)
                    if path == 'full':
                        repeated = predict(runtime, path, noisy, timestep, lr, camera,
                                           tuple(clean.shape[2:]), config.image_size, context)
                        repeat_max = float((base.float() - repeated.float()).abs().max())
                    else:
                        repeat_max = None
                    for name, indices in PERMUTATIONS.items():
                        permutation = torch.tensor(indices)
                        inverse = stage3.inverse_view_permutation(permutation)
                        changed = predict(runtime, path,
                            stage3.permute_view_tensor(noisy, permutation), timestep,
                            stage3.permute_view_tensor(lr, permutation),
                            stage3._camera_index_select(camera, permutation),
                            tuple(clean.shape[2:]), config.image_size, context)
                        changed = stage3.permute_view_tensor(changed, inverse)
                        if not torch.isfinite(changed).all():
                            raise RuntimeError('nonfinite permuted velocity')
                        changed_image, changed_metrics = decoded_metrics(
                            runtime, perceptual, noisy, changed, sigma, hr, clean)
                        delta = (base.float() - changed.float())
                        row = {
                            'label': label, 'probe': probe, 'dropped': dropped,
                            'path': path, 'permutation': name, 'indices': indices,
                            'sigma': 0.5, 'seed': 3302,
                            'reference': base_metrics, 'permuted': changed_metrics,
                            'reference_minus_permuted_psnr': base_metrics['psnr'] - changed_metrics['psnr'],
                            'reference_minus_permuted_ssim': base_metrics['ssim'] - changed_metrics['ssim'],
                            'permuted_minus_reference_lpips': changed_metrics['lpips'] - base_metrics['lpips'],
                            'target_velocity_relative_delta': float(
                                delta[:, :, :1].norm() / base[:, :, :1].float().norm().clamp_min(1e-12)),
                            'all_velocity_relative_delta': float(
                                delta.norm() / base.float().norm().clamp_min(1e-12)),
                            'velocity_max_abs_delta': float(delta.abs().max()),
                            'full_path_repeat_max_abs_delta': repeat_max,
                        }
                        rows.append(row)
                        if label == 'S2' and path == 'full':
                            save_error_map(output.parent / 'error_maps' /
                                f'{probe.replace(":", "_")}_{"drop" if dropped else "keep"}_{name}.png',
                                base_image, changed_image, hr[:, :, :1])
            print(f'{label} {probe} done', flush=True)
    if digest(checkpoint) != hashes['checkpoint']:
        raise RuntimeError('checkpoint changed during audit')
    expected = len(probes) * 2 * len(PATHS) * len(PERMUTATIONS)
    assert len(rows) == expected
    payload = {
        'label': label, 'rows': rows, 'inputs': inputs, 'source_sha256': digest(__file__),
        'checkpoint_sha256': hashes['checkpoint'], 'config_sha256': hashes['config'],
        'manifest_sha256': hashes['manifest'], 'full_model_revision': 'read-only audit',
        'path_order': PATHS, 'permutations': PERMUTATIONS,
        'metric': 'frame_metrics (frozen evaluation implementation)',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=(*CELLS, 'self-check'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probes', nargs='+', choices=PROBES, default=PROBES)
    args = parser.parse_args()
    if args.label == 'self-check':
        self_check()
    elif args.output is None:
        parser.error('--output is required')
    else:
        run(args.label, args.output, tuple(args.probes))
