"""Regenerate descriptive final C summaries from authoritative metric rows (CPU only)."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from stage1_c_analysis import aggregate
from rl3dsr.validation.decoded_space import strict_3d_quality_verdict


def summarize(campaign):
    final = json.loads((campaign / 'final/3d_strict_metrics.json').read_text())
    rows = final['metric_rows']
    assert final['noise_seeds'] == [2201, 2202, 2203, 2204]
    means, paired = aggregate(rows)
    verdict = strict_3d_quality_verdict(rows)
    assert verdict == final['verdict'], 'recomputed strict verdict differs'
    metrics = ('psnr', 'ssim', 'lpips')
    summaries = {}
    for group in ('seed', 'position'):
        summaries[group] = []
        for value in sorted({r[group] for r in rows}):
            selected = [r for r in rows if r[group] == value and r['condition'] == 'correct']
            record = {group: value, 'n_views_or_seeds': len(selected)}
            for m in metrics:
                record[m] = statistics.mean(r[m] for r in selected)
            for control in ('bicubic', 'shuffled', 'disabled'):
                matches = [r for r in paired if r[group] == value and r['comparator'] == control]
                record[control + '_all_metric_wins'] = sum(r['all_metric_win'] for r in matches)
                for m in metrics:
                    baseline = statistics.mean(r[m] for r in rows if r[group] == value and r['condition'] == control)
                    record[control + '_' + m + '_margin'] = record[m] - baseline
                base_lpips = statistics.mean(r['lpips'] for r in rows if r[group] == value and r['condition'] == control)
                record[control + '_lpips_relative_reduction'] = 1 - record['lpips'] / base_lpips
            summaries[group].append(record)
    correct = [r for r in rows if r['condition'] == 'correct']
    variation = {m: {'sample_sd_across_four_seed_means': statistics.stdev(r[m] for r in summaries['seed']),
                     'range_across_seed_means': max(r[m] for r in summaries['seed']) - min(r[m] for r in summaries['seed']),
                     'worst_seed_view': (max if m == 'lpips' else min)(correct, key=lambda r: r[m])}
                 for m in metrics}
    out = campaign / 'analysis'
    out.mkdir(exist_ok=True)
    payload = {'strict_verdict': verdict, 'means': means, 'summaries': summaries, 'variation': variation,
               'margin_convention': 'signed correct minus reference for all three metrics; LPIPS negative is better',
               'scope': 'One chair scene, four training views, one training seed; noise replicates are not independent scenes.'}
    (out / 'final_summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    for name, records in [('final_seed_summary', summaries['seed']), ('final_view_summary', summaries['position']), ('final_raw_metrics', rows)]:
        with (out / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps({'recomputed_verdict_identical': True, 'rows': len(rows), 'pairs': len(correct),
                      'status': 'PASS' if verdict['passed'] else 'FAIL', 'variation': {m: variation[m]['sample_sd_across_four_seed_means'] for m in metrics}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    summarize(parser.parse_args().campaign)
