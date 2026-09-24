#!/usr/bin/env python3
"""Summarize the frozen W3/S1/M1 A6 path audit and apply its stop rule."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median

from stage3_3_a6_path_audit import digest


ROOT = Path('/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-structure')
AUDIT = ROOT / 'artifacts/stage3_3_a6_path_20260924'
LABELS = ('w3', 's1', 'm1')


def read(path):
    return json.loads(path.read_text())


def summarize_model(attention, trace, dropped):
    rows = [r for r in attention['rows'] if r['dropped'] == dropped]
    trajectory = [r for r in trace['rows'] if r['dropped'] == dropped]
    assert len(rows) == len(trajectory) == 4
    assert [r['probe'] for r in rows] == [r['probe'] for r in trajectory]
    result = {}
    for mode in ('correct', 'mispaired_camera', 'shuffle_fusion'):
        a = [r['conditions'][mode] for r in rows]
        stats = [x['attention'] for x in a]
        result[mode] = {
            'verified_matches_total': sum(x['verified_matches_total'] for x in stats),
            'verified_matches_in_band': sum(x['verified_matches_in_band'] for x in stats),
            'attention_top1_fraction_probe_mean': mean(x['top1_fraction'] for x in stats),
            'attention_entropy_fraction_probe_mean': mean(x['entropy_fraction_median'] for x in stats),
            'attention_match_to_uniform_probe_mean': mean(x['match_vs_uniform_median'] for x in stats),
            'frozen_feature_match_percentile_probe_mean': mean(x['frozen_cosine_percentile_median'] for x in stats),
            'prepared_target_relative_delta_probe_mean': mean(x['prepared_target_delta_relative'] for x in a),
            'bridge0_target_relative_delta_probe_mean': mean(x['bridge_target_delta_relative']['0'] for x in a),
        }
        if mode != 'correct':
            t = [r['comparisons'][mode] for r in trajectory]
            result[mode]['target_velocity_relative_delta_first_probe_mean'] = mean(
                x['first_step_target_velocity_relative_delta'] for x in t)
            result[mode]['target_velocity_relative_delta_50step_probe_mean'] = mean(
                x['mean_target_velocity_relative_delta'] for x in t)
            result[mode]['final_correct_minus_wrong_ssim_proxy_probe_mean'] = mean(
                x['final_correct_minus_wrong_ssim'] for x in t)
            result[mode]['per_step_target_velocity_relative_delta_probe_mean'] = [
                mean(x['per_step_target_velocity_relative_delta'][i] for x in t) for i in range(50)]
            result[mode]['single_step_decoded_ssim_margin_probe_mean'] = [
                {'sigma': a[0]['one_step'][i]['sigma'], 'margin': mean(
                    r['conditions']['correct']['one_step'][i]['decoded_ssim']
                    - r['conditions'][mode]['one_step'][i]['decoded_ssim'] for r in rows)}
                for i in range(3)
            ]
    return result


def main():
    geom_path = AUDIT / 'geometry.json'
    geom = read(geom_path)
    reliable = [r for r in geom['rows'] if r['reliable']]
    assert len(geom['rows']) == 12 and len(reliable) >= 10
    assert all(r['camera_pair_valid'] and r['usable_query_fraction'] == 1 for r in geom['rows'])
    assert all(r['within_band_fraction'] >= .95 for r in reliable)
    assert all(r['epipolar_distance_median_cells'] < .1 for r in reliable)
    inputs = {'geometry.json': digest(geom_path)}
    models = {}
    for label in LABELS:
        path = AUDIT / f'{label}_entropy.json'
        trace_path = AUDIT / f'{label}_trace.json'
        attention, trajectory = read(path), read(trace_path)
        assert attention['label'].lower() == trajectory['label'].lower() == label
        assert attention['input_sha256'] == trajectory['input_sha256']
        assert attention['geometry_audit_sha256'] == inputs['geometry.json']
        assert trajectory['sample_steps'] == 50 and trajectory['inference_seed'] == 3302
        assert len(attention['rows']) == len(trajectory['rows']) == 8
        inputs[path.name], inputs[trace_path.name] = digest(path), digest(trace_path)
        models[label] = {'input_sha256': attention['input_sha256'],
                         'retained': summarize_model(attention, trajectory, False),
                         'dropped': summarize_model(attention, trajectory, True)}
    assert len({models[x]['input_sha256']['manifest'] for x in LABELS}) == 1
    w, s, m = (models[x]['retained'] for x in LABELS)
    attention_regression = all(x['correct']['attention_top1_fraction_probe_mean']
                               < w['correct']['attention_top1_fraction_probe_mean']
                               for x in (s, m))
    s1_bridge_attenuated = (s['shuffle_fusion']['bridge0_target_relative_delta_probe_mean']
                            < .5 * w['shuffle_fusion']['bridge0_target_relative_delta_probe_mean'])
    s1_velocity_attenuated = (s['shuffle_fusion']['target_velocity_relative_delta_50step_probe_mean']
                              < .5 * w['shuffle_fusion']['target_velocity_relative_delta_50step_probe_mean'])
    sampled_only = all(all(x['margin'] > 0 for x in model['shuffle_fusion']['single_step_decoded_ssim_margin_probe_mean'])
                       for model in (s, m))
    assert not attention_regression and not s1_bridge_attenuated and not s1_velocity_attenuated
    assert not sampled_only
    verdict = {
        'status': 'HOLD', 'STAGE4_READY': False, 'train_candidate': False,
        'decision': 'NO_SUPPORTED_STRUCTURAL_CANDIDATE',
        'reason': 'Real-pose epipolar alignment passes the reliable proxy matches. S1/M1 attention match rate is not worse than W3; S1 target bridge and trajectory velocity responses are not attenuated; S1/M1 lose decoded camera advantage already in one step. The audit does not isolate a repairable implementation or single architecture bottleneck.',
        'gate': {'geometry_alignment_detected_error': False,
                 'selective_attention_regression': attention_regression,
                 's1_target_bridge_attenuation': s1_bridge_attenuated,
                 's1_target_velocity_attenuation': s1_velocity_attenuated,
                 'sampling_only_failure': sampled_only},
        'matching': {'reliable_pairs': len(reliable), 'all_pairs': len(geom['rows']),
                     'median_reliable_pair_epipolar_distance_cells': median(r['epipolar_distance_median_cells'] for r in reliable),
                     'inconclusive_pairs': [f"{r['probe']} source{r['source']}" for r in geom['rows'] if not r['reliable']]},
        'artifact_sha256': inputs,
        'audit_source_sha256': {'path_audit': digest(ROOT / 'scripts/stage3_3_a6_path_audit.py'),
                                'sampler_trace': digest(ROOT / 'scripts/stage3_3_a6_sampler_trace.py'),
                                'review': digest(ROOT / 'scripts/stage3_3_a6_path_review.py')},
        'models': models,
        'limitations': ['SIFT and RANSAC matches are a proxy, not depth-verified ground truth.',
                        'Attention top-1 considers only matched target-source patches and is not a full geometric-consistency measure.',
                        'After the first sampler update, velocity differences include divergent latent trajectories.',
                        'Final SSIM in this audit uses TorchMetrics; frozen evaluation rows remain the gate authority.'],
    }
    output = AUDIT / 'review.json'
    with output.open('x') as stream:
        json.dump(verdict, stream, indent=2, allow_nan=False)
    print(json.dumps({k: verdict[k] for k in ('status', 'decision', 'gate', 'matching')}, indent=2))


if __name__ == '__main__':
    main()
