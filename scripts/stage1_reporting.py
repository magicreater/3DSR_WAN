"""Small reproducibility and plotting helpers for Stage 1 experiments."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def snapshot_source(output_dir):
    root = Path(__file__).resolve().parents[1]
    paths = []
    for directory in ('src', 'scripts', 'tests', 'docs', 'third_party'):
        paths.extend(p for p in (root / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    paths.extend(root / name for name in ('AGENTS.md', 'pyproject.toml', '.gitignore', 'environment-stage0.yml'))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(paths)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    target = Path(output_dir) / ('source_' + digest[:12])
    if not target.with_suffix('.tar.gz').exists():
        temporary = target.with_suffix('.tar.gz.tmp')
        with tarfile.open(temporary, 'w:gz') as archive:
            for p in paths:
                archive.add(p, arcname=str(p.relative_to(root)))
        os.replace(temporary, target.with_suffix('.tar.gz'))
        atomic_json(target.with_suffix('.sha256.json'), hashes)
        target.with_suffix('.diff').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root))
        target.with_suffix('.status.txt').write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=root))
    return {'sha256': digest, 'archive': str(target.with_suffix('.tar.gz').resolve())}


def plot_training_curves(output_dir, step_rows, eval_rows):
    if not step_rows and not eval_rows:
        return
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(16, 15), constrained_layout=True)
    panels = [('loss', 'Flow loss'), ('sigma', 'Training sigma'),
              ('learning_rate', 'Learning rate'), ('gradient_norm', 'Gradient norm'),
              ('peak_gpu_memory_mib', 'GPU memory (MiB)'), ('step_seconds', 'Step time (s)')]
    steps = [r['step'] for r in step_rows]
    for axis, (key, label) in zip(axes.flat, panels):
        values = [r.get(key, np.nan) for r in step_rows]
        axis.plot(steps, values, linewidth=.7, alpha=.45, label='raw')
        if key == 'loss' and len(values) >= 50:
            axis.plot(steps[49:], np.convolve(values, np.ones(50)/50, 'valid'), label='50-step mean')
            axis.legend()
        axis.set_title(label)
        if key == 'learning_rate':
            axis.set_yscale('log')
    eval_steps = [r['step'] for r in eval_rows]
    best = max(eval_rows, key=lambda r: (bool(r.get('dev_candidate_pass')), bool(r.get('decoded_quality_pass')),
               r.get('correct_psnr_margin', -1e9), r.get('correct_ssim_margin', -1e9),
               r.get('correct_lpips_reduction', -1e9))) if eval_rows else None
    for j, metric in enumerate(('psnr', 'ssim', 'lpips')):
        axis = axes[2, j]
        for condition in ('correct', 'bicubic', 'vae_ceiling', 'shuffled', 'disabled'):
            values = [r.get(condition + '_' + metric, np.nan) for r in eval_rows]
            if any(np.isfinite(values)):
                axis.plot(eval_steps, values, marker='.', label=condition)
        if eval_rows and ('bicubic_' + metric) in eval_rows[0]:
            baseline = eval_rows[0]['bicubic_' + metric]
            threshold = baseline + (.25 if metric == 'psnr' else .005) if metric != 'lpips' else baseline*.95
            axis.axhline(threshold, color='black', linestyle='--', label='bicubic gate')
        if best:
            axis.axvline(best['step'], color='grey', linestyle=':', label='best dev')
        axis.set_title(metric.upper() + (' (dB)' if metric == 'psnr' else ''))
        if axis.lines:
            axis.legend(fontsize=7)
        gap_axis = axes[3, j]
        suffix = 'lpips_reduction' if metric == 'lpips' else metric + '_gap'
        for control in ('shuffled', 'disabled'):
            values = [r.get('correct_' + control + '_' + suffix, np.nan) for r in eval_rows]
            if any(np.isfinite(values)):
                gap_axis.plot(eval_steps, values, marker='.', label='vs ' + control)
        gap_axis.set_title(metric.upper() + (' relative reduction' if metric == 'lpips' else ' paired gap'))
        if gap_axis.lines:
            gap_axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel('Training step')
        axis.grid(alpha=.2)
    temporary = Path(output_dir) / 'training_curves.tmp.png'
    fig.savefig(temporary, dpi=130)
    plt.close(fig)
    os.replace(temporary, Path(output_dir) / 'training_curves.png')

    diagnostics = sorted({key for r in step_rows for key in r if key.startswith('block_')})
    if diagnostics:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
        for key in diagnostics:
            axis = axes[0] if 'gate_' in key else axes[1]
            axis.plot(steps, [r.get(key, np.nan) for r in step_rows], label=key, linewidth=.8)
        for axis, title in zip(axes, ('Time gates', 'Residual / Wan input token RMS')):
            axis.set_title(title)
            axis.set_xlabel('Training step')
            axis.grid(alpha=.2)
            axis.legend(fontsize=6, ncol=3)
        temporary = Path(output_dir) / 'bridge_diagnostics.tmp.png'
        fig.savefig(temporary, dpi=130)
        plt.close(fig)
        os.replace(temporary, Path(output_dir) / 'bridge_diagnostics.png')


def write_checkpoint_index(output_dir, config_hash, eval_rows, best_step):
    output_dir = Path(output_dir)
    previous_path = output_dir / 'checkpoint_index.json'
    previous = json.loads(previous_path.read_text())['checkpoints'] if previous_path.exists() else []
    known = {row['step']: row for row in previous}
    quality = {int(r['step']): r for r in eval_rows}
    records = []
    for path in sorted(output_dir.glob('[34]d_step_[0-9][0-9][0-9][0-9].pt')):
        step = int(path.stem.rsplit('_', 1)[-1])
        if step in known:
            row = known[step]
        else:
            import torch
            metadata = torch.load(path, map_location='cpu', weights_only=True)['experiment']
            row = {'step': step, 'adapter': str(path.resolve()),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'training_state': str(path.with_name(path.stem + '_training_state.pt').resolve()),
                'config_sha256': metadata.get('config_sha256', config_hash),
                'parent_checkpoint_sha256': metadata.get('parent_checkpoint_sha256'),
                'source_identity': metadata.get('source_identity')}
        row.update(quality=quality.get(step), is_best=(step == best_step))
        records.append(row)
    atomic_json(previous_path, {'schema_version': 2, 'checkpoints': records})
