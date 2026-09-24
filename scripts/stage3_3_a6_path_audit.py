#!/usr/bin/env python3
"""Paired A6 camera-path audit; never changes a checkpoint or evaluation gate."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

import stage3_experiment as stage3
from rl3dsr.models.wan.lr_fusion import epipolar_local_key_mask, patch_fundamental_matrices
from rl3dsr.validation.stage3_protocol import load_stage3_config


ROOT = Path('/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-structure')
DATA = Path('/data/linzizhuo/RL3DSR_WAN_REMO')
W3 = Path('/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-target-rank/artifacts/stage3_3_a6sw3_20260923')
S1 = ROOT / 'artifacts/stage3_3_structure_20260923'
M1 = ROOT / 'artifacts/stage3_3_merge_20260923'
CELLS = {
    'W3': (W3, 'w3_lego_seed42', W3 / 'train/w3_lego_seed42/stage3_step_1000.pt'),
    'S1': (S1, 's1_lego_seed42', S1 / 'train/s1_lego_seed42/stage3_step_1000.pt'),
    'M1': (M1, 'm1_lego_seed42', M1 / 'checkpoints/m1_lego_seed42_step1000.pt'),
}
PROBES = ('lego:000', 'lego:033', 'lego:066', 'lego:099')
MODES = ('correct', 'mispaired_camera', 'shuffle_fusion')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_group(config, group, device):
    return stage3._load_indices(DATA / 'datasets/nerf_synthetic', 'lego', 'train',
                                group['indices'], config, device)


def setup(label):
    campaign, cell, checkpoint = CELLS[label]
    config_path = campaign / 'config' / f'{cell}.json'
    config = load_stage3_config(config_path)
    manifest_path = campaign / 'manifest' / f'{cell}.json'
    if label == 'M1':
        manifest_path = W3 / 'manifest/w3_lego_seed42.json'
    manifest = json.loads(manifest_path.read_text())
    assert tuple(manifest['subsets']['probe']) == PROBES
    groups = {r['id']: r for r in manifest['groups']}
    assert all(p in groups for p in PROBES)
    return config, groups, {'config': digest(config_path), 'manifest': digest(manifest_path),
                            'checkpoint': digest(checkpoint)}, checkpoint


def gray(image):
    rgb = ((image.detach().float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5)
    return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def matches(hr, target, source):
    sift = cv2.SIFT_create()
    ka, da = sift.detectAndCompute(gray(hr[0, :, target]), None)
    kb, db = sift.detectAndCompute(gray(hr[0, :, source]), None)
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ab = matcher.knnMatch(da, db, k=2)
    ba = matcher.knnMatch(db, da, k=2)
    forward = {a.queryIdx: a.trainIdx for a, b in ab if a.distance < .7 * b.distance}
    reverse = {a.queryIdx: a.trainIdx for a, b in ba if a.distance < .7 * b.distance}
    pairs = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    if len(pairs) < 8:
        return np.empty((0, 2)), np.empty((0, 2)), len(pairs)
    a = np.float32([ka[i].pt for i, _ in pairs])
    b = np.float32([kb[j].pt for _, j in pairs])
    _, inliers = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.0, .99)
    if inliers is None:
        return np.empty((0, 2)), np.empty((0, 2)), len(pairs)
    keep = inliers.ravel().astype(bool)
    return a[keep], b[keep], len(pairs)


def geometry_audit(output):
    config, groups, hashes, _ = setup('W3')
    rows = []
    for probe in PROBES:
        _, hr, _, camera = load_group(config, groups[probe], torch.device('cpu'))
        grid = (config.image_size // 16, config.image_size // 16)
        f, valid = patch_fundamental_matrices(camera, grid)
        allowed, usable = epipolar_local_key_mask(camera, grid, band=config.epipolar_band)
        for source in range(1, config.views):
            a, b, raw = matches(hr, 0, source)
            if len(a):
                cell_a, cell_b = (a + .5) / 16, (b + .5) / 16
                homo_a = np.column_stack([cell_a, np.ones(len(a))])
                homo_b = np.column_stack([cell_b, np.ones(len(b))])
                mat = f[0, 0, source].numpy()
                lines = homo_a @ mat.T
                distance = abs(np.sum(homo_b * lines, axis=1)) / np.linalg.norm(lines[:, :2], axis=1)
                qa = np.clip(cell_a.astype(int), 0, grid[0] - 1)
                qb = np.clip(cell_b.astype(int), 0, grid[0] - 1)
                qi = qa[:, 1] * grid[1] + qa[:, 0]
                ki = source * grid[0] * grid[1] + qb[:, 1] * grid[1] + qb[:, 0]
                band_hit = allowed[0, qi, ki].numpy()
            else:
                distance, band_hit, qi, ki = np.array([]), np.array([]), [], []
            rows.append({'probe': probe, 'source': source, 'raw_mutual': raw,
                         'ransac_inliers': len(a), 'reliable': len(a) >= 20,
                         'epipolar_distance_median_cells': None if not len(a) else float(np.median(distance)),
                         'within_band_fraction': None if not len(a) else float(np.mean(band_hit)),
                         'camera_pair_valid': bool(valid[0, 0, source]),
                         'usable_query_fraction': float(usable[0, :grid[0] * grid[1], source].float().mean()),
                         'query_patches': list(map(int, qi)), 'source_patches': list(map(int, ki))})
    payload = {'scope': 'high-confidence image matches are proxy correspondences, not ground truth',
               'input_sha256': hashes, 'rows': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({'output': str(output), 'rows': [{k: v for k, v in r.items()
                            if k not in ('query_patches', 'source_patches')} for r in rows]}, indent=2))


@contextmanager
def capture_attention(fusion, tokens):
    saved = []
    original = torch.softmax

    def wrapped(value, dim, *args, **kwargs):
        result = original(value, dim, *args, **kwargs)
        if value.ndim == 4 and value.shape[-1] == tokens + 1 and dim == -1:
            saved.append(result.detach().float().cpu())
        return result

    with patch.object(torch, 'softmax', side_effect=wrapped):
        yield saved


def attention_summary(weights, base, allowed, geometry_rows, *, dropped):
    # Only verified SIFT/RANSAC matches are used. No label is inferred for occlusions.
    patches = base.shape[2]
    normalized = torch.nn.functional.normalize(base.detach().float().cpu(), dim=-1)
    values = []
    total = 0
    for pair in geometry_rows:
        if not pair['reliable']:
            continue
        source = pair['source']
        for qi, ki in zip(pair['query_patches'], pair['source_patches']):
            total += 1
            candidate = allowed[0, qi, source * patches:(source + 1) * patches].cpu()
            if not candidate[ki - source * patches] or int(candidate.sum()) < 2:
                continue
            att = weights[0, 0, qi, source * patches:(source + 1) * patches]
            selected = att[ki - source * patches]
            source_mass = att.sum()
            probabilities = att[candidate] / source_mass.clamp_min(1e-12)
            entropy = -float((probabilities * probabilities.clamp_min(1e-12).log()).sum())
            cosine = normalized[0, 0, qi] @ normalized[0, source].T
            cos_rank = float((cosine[candidate] <= cosine[ki - source * patches]).float().mean())
            values.append({'attention_top1': int(att.argmax()) == ki - source * patches,
                           'attention_match_mass': float(selected),
                           'attention_source_mass': float(source_mass),
                           'attention_uniform_ratio': float(selected / (source_mass / candidate.sum()).clamp_min(1e-12)),
                           'attention_entropy_fraction': entropy / math.log(int(candidate.sum())),
                           'frozen_cosine_percentile': cos_rank,
                           'candidate_count': int(candidate.sum())})
    return {'dropped': dropped, 'verified_matches_total': total,
            'verified_matches_in_band': len(values),
            'top1_fraction': None if not values else float(np.mean([v['attention_top1'] for v in values])),
            'match_vs_uniform_median': None if not values else float(np.median([v['attention_uniform_ratio'] for v in values])),
            'entropy_fraction_median': None if not values else float(np.median([v['attention_entropy_fraction'] for v in values])),
            'frozen_cosine_percentile_median': None if not values else float(np.median([v['frozen_cosine_percentile'] for v in values])),
            'candidate_count_median': None if not values else float(np.median([v['candidate_count'] for v in values]))}


def relative(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


def target_ssim(vae, latent, hr):
    decoded = vae.decode_multiview(latent[:, :, :1])[:, :, 0]
    return float(structural_similarity_index_measure((decoded.float() + 1) / 2,
                     (hr[:, :, 0].float() + 1) / 2, data_range=1.0))


def model_audit(label, geometry_path, output):
    config, groups, hashes, checkpoint = setup(label)
    geometry = json.loads(geometry_path.read_text())
    assert geometry['input_sha256']['manifest'] == digest(W3 / 'manifest/w3_lego_seed42.json')
    runtime = stage3.load_runtime(config, model_dir=DATA / 'models/Wan2.1-T2V-1.3B',
        lq_source=Path('/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py'),
        lq_checkpoint=Path('/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt'),
        bridge_checkpoint=DATA / 'artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt',
        stage3_checkpoint=checkpoint, device='cuda')
    runtime.module.eval()
    fusion = runtime.module.fusion
    rows = []
    with torch.no_grad():
        for probe in PROBES:
            group = groups[probe]
            _, hr, raw_lr, camera = load_group(config, group, runtime.device)
            clean = runtime.vae.encode_multiview(hr)
            grid = (clean.shape[-2] // 2, clean.shape[-1] // 2)
            assert grid == (config.image_size // 16, config.image_size // 16)
            relevant = [r for r in geometry['rows'] if r['probe'] == probe]
            for dropped in (False, True):
                lr = raw_lr.clone()
                if dropped:
                    lr[:, :, 0] = 0
                base = runtime.module.conditioner.multiview_features(
                    lr, conditioning_size=(config.image_size, config.image_size),
                    latent_shape=tuple(clean.shape[2:])).reshape(1, config.views, -1, 1536)
                outputs = {}
                for mode in MODES:
                    sample_seed = 3302 * 1_000_000 + group['anchor']
                    mode_index = {'correct': 0, 'shuffle_fusion': 4, 'mispaired_camera': 7}[mode]
                    generator = torch.Generator().manual_seed(sample_seed * 10 + mode_index)
                    changed_lr, fusion_camera, geometry_camera, mask = stage3._intervention(
                        lr, camera, mode, generator)
                    assert torch.equal(changed_lr, lr) and torch.equal(geometry_camera.K, camera.K)
                    with capture_attention(fusion, config.views * grid[0] * grid[1]) as saved:
                        prepared = fusion(base, fusion_camera, grid, source_mask=mask).reshape(1, -1, 1536)
                    assert len(saved) and sum(t.shape[2] for t in saved) == config.views * grid[0] * grid[1]
                    weights = torch.cat(saved, dim=2)
                    allowed, _ = epipolar_local_key_mask(fusion_camera, grid, band=config.epipolar_band)
                    outputs[mode] = {'prepared': prepared,
                                     'attention': attention_summary(weights, base, allowed, relevant, dropped=dropped)}
                    # Fixed-noise single-step paired target SSIM at all three audited sigmas.
                    one_step = []
                    for sigma_value in (.2, .5, .8):
                        noise = torch.randn(clean.shape, generator=torch.Generator(device=runtime.device).manual_seed(3302),
                                            device=runtime.device, dtype=clean.dtype)
                        sigma = torch.tensor([sigma_value], device=runtime.device)
                        noisy, timestep, target = stage3.flow_matching_pair(clean, noise, sigma)
                        velocity = runtime.module.predict(runtime.dit, noisy, timestep, None,
                                     prepared, camera, tuple(clean.shape[2:]))
                        one_step.append({'sigma': sigma_value, 'target_velocity_norm': float(velocity[:, :, :1].float().norm()),
                                         'decoded_ssim': target_ssim(runtime.vae, noisy - sigma.reshape(1, 1, 1, 1, 1) * velocity, hr),
                                         'latent_target_mse': float((velocity[:, :, :1] - target[:, :, :1]).float().square().mean())})
                    outputs[mode]['one_step'] = one_step
                correct = outputs['correct']['prepared']
                base_flat = base.reshape_as(correct)
                bridge_correct = runtime.module.conditioner.bridge_residuals(correct, torch.tensor([500.], device=runtime.device))
                for mode in MODES:
                    item = outputs[mode]
                    prepared = item.pop('prepared')
                    bridge = runtime.module.conditioner.bridge_residuals(prepared, torch.tensor([500.], device=runtime.device))
                    item['base_to_fusion_target_relative'] = relative(base_flat[:, :grid[0] * grid[1]], prepared[:, :grid[0] * grid[1]])
                    item['prepared_target_delta_relative'] = relative(correct[:, :grid[0] * grid[1]], prepared[:, :grid[0] * grid[1]])
                    item['bridge_target_delta_relative'] = {str(k): relative(bridge_correct[k][:, :grid[0] * grid[1]],
                                        bridge[k][:, :grid[0] * grid[1]]) for k in bridge}
                rows.append({'probe': probe, 'indices': group['indices'], 'dropped': dropped,
                             'conditions': outputs})
            print(f'{label} {probe} complete', flush=True)
    payload = {'label': label, 'input_sha256': hashes,
               'geometry_audit_sha256': digest(geometry_path), 'noise_seed': 3302,
               'rows': rows, 'scope': 'single-step and attention path; final multistep metrics are in frozen evaluations'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=('geometry', *CELLS))
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.label == 'geometry':
        geometry_audit(args.output)
    else:
        if args.geometry is None:
            parser.error('--geometry is required for model audits')
        model_audit(args.label, args.geometry, args.output)


if __name__ == '__main__':
    main()
