#!/usr/bin/env python3
"""Bounded, serial Stage 1 C experiment campaign (all GPU process time counted)."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from stage1_reporting import atomic_json

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'artifacts/stage1/c_campaign_20260902'
PYTHON = '/home/linzizhuo/miniconda3/envs/rl3dsr-stcdit/bin/python'
COMMON = ['--model-dir', str(ROOT/'models/Wan2.1-T2V-1.3B'),
          '--scene', str(ROOT/'datasets/nerf_synthetic/chair'),
          '--lq-source', '/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py',
          '--lq-checkpoint', '/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt']


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def quality_key(row):
    return (bool(row.get('dev_candidate_pass')), bool(row.get('decoded_quality_pass')),
            row['correct_psnr_margin'], row['correct_ssim_margin'], row['correct_lpips_reduction'])


def run_gpu(command, label, *, search=True):
    ledger_path = CAMPAIGN/'campaign.json'
    ledger = json.loads(ledger_path.read_text())
    limit = ledger['training_budget_gpu_seconds'] if search else ledger['budget_gpu_seconds']
    remaining = limit-ledger['gpu_seconds_used']
    if remaining < 120:
        raise RuntimeError('GPU budget exhausted; final validation reserve retained')
    (CAMPAIGN/'logs').mkdir(exist_ok=True)
    log_path = CAMPAIGN/'logs'/f'{label}.log'
    if log_path.exists():
        raise RuntimeError(f'refusing to overwrite process log: {log_path}')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONPATH': str(ROOT/'src')+':'+str(ROOT/'scripts'),
           'RL3DSR_WAN_MODEL_DIR': str(ROOT/'models/Wan2.1-T2V-1.3B'),
           'RL3DSR_LQ_SOURCE': COMMON[5], 'RL3DSR_LQ_CHECKPOINT': COMMON[7],
           'RL3DSR_ADAPTER_3D': str(ROOT/'artifacts/stage1/optimized_sigma_balanced/best_dev.pt'),
           'RL3DSR_ADAPTER_4D': str(ROOT/'artifacts/stage1/repro/4d_adapter.pt'),
           'RL3DSR_GPU_DEADLINE': str(time.time()+remaining-60),
           'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4'}
    started = time.monotonic()
    event = {'label': label, 'command': command, 'started_unix': time.time(), 'search': search, 'log': str(log_path)}
    ledger['active'] = event
    atomic_json(ledger_path, ledger)
    print(json.dumps({'event':'launch', **event}), flush=True)
    returncode = -1
    try:
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError('GPU campaign time limit reached')
    finally:
        event.update(elapsed_seconds=time.monotonic()-started, returncode=returncode)
        ledger['gpu_seconds_used'] += event['elapsed_seconds']
        ledger.setdefault('processes', []).append(event)
        ledger['active'] = None
        atomic_json(ledger_path, ledger)
        print(json.dumps({'event':'process_complete', **event}), flush=True)
    if returncode:
        raise RuntimeError(f'{label} failed with {returncode}; see {log_path}')


def train(name, blocks, timed, sigma, steps, *, resume=False, stop=True):
    directory = CAMPAIGN/name
    command = [PYTHON, str(ROOT/'scripts/stage1_experiment.py'), *COMMON, '--kind', '3d',
               '--output-dir', str(directory), '--hr-resolution', '256', '--views', '4', '--sample-count', '1',
               '--max-steps', str(steps), '--eval-interval', '250', '--checkpoint-interval', '250',
               '--max-hours', '6', '--quality-overfit', '--bridge-blocks', *map(str, blocks),
               '--sampling-strategy', sigma]
    if timed:
        command.append('--bridge-time-conditioning')
    if stop:
        command.append('--stop-on-dev-pass')
    if resume:
        last = read_rows(directory/'train_steps.jsonl')[-1]['step']
        command += ['--init-adapter', str(directory/f'3d_step_{last:04d}.pt'),
                    '--resume-state', str(directory/f'3d_step_{last:04d}_training_state.pt')]
    run_gpu(command, f'{name}_{steps:04d}')
    rows=read_rows(directory/'checkpoint_metrics.jsonl')
    return max(rows,key=quality_key)


def extend(spec):
    name,blocks,timed,sigma=spec
    for steps in (4000,5000,6000,7000,8000):
        best=train(name,blocks,timed,sigma,steps,resume=True)
        if best.get('dev_candidate_pass'):
            return best
        rows=read_rows(CAMPAIGN/name/'checkpoint_metrics.jsonl')
        recent=[r for r in rows if steps-1000 < r['step'] <= steps]
        earlier=[r for r in rows if r['step'] <= steps-1000]
        improvements=(max(r['correct_psnr'] for r in recent)-max(r['correct_psnr'] for r in earlier),
                      max(r['correct_ssim'] for r in recent)-max(r['correct_ssim'] for r in earlier),
                      1-min(r['correct_lpips'] for r in recent)/min(r['correct_lpips'] for r in earlier))
        if all(v < threshold for v,threshold in zip(improvements,(.05,.001,.01))):
            atomic_json(CAMPAIGN/name/'plateau_stop.json', {'step':steps,'improvements':improvements,'thresholds':[.05,.001,.01]})
            break
    return best


def finish_candidate(spec, best):
    candidate = {'run': spec[0], 'checkpoint': best['checkpoint'], 'step':best['step'],
                 'development_metrics':best, 'frozen_unix':time.time()}
    import hashlib
    candidate['sha256']=hashlib.sha256(Path(best['checkpoint']).read_bytes()).hexdigest()
    atomic_json(CAMPAIGN/'frozen_candidate.json', candidate)
    command=[PYTHON,str(ROOT/'scripts/stage1_decoded_eval.py'),*COMMON,'--kinds','3d',
             '--adapter-3d',best['checkpoint'],'--output-dir',str(CAMPAIGN/'final'),
             '--strict-3d-overfit','--noise-seeds','2201','2202','2203','2204']
    run_gpu(command,'final_four_seeds',search=False)
    result=json.loads((CAMPAIGN/'final/3d_strict_metrics.json').read_text())
    atomic_json(CAMPAIGN/'search_result.json',{'status':'FINAL_METRICS_PASS_VISUAL_PENDING' if result['verdict']['passed'] else 'FINAL_FAIL',**candidate})


def campaign():
    specs=[('c_main',(0,1,2,3),True,'balanced'),
           ('c_multilayer',(0,1,2,3),False,'balanced'),
           ('c_time',(0,),True,'balanced')]
    results=[]
    for spec in specs:
        best=train(*spec,2000)
        results.append((spec,best))
        if best.get('dev_candidate_pass'):
            finish_candidate(spec,best)
            return
    spec,_=max(results,key=lambda pair:quality_key(pair[1]))
    best=extend(spec)
    results.append((spec,best))
    if best.get('dev_candidate_pass'):
        finish_candidate(spec,best)
        return
    sigma_spec=('c_sigma',(0,1,2,3),True,'permutation')
    best=train(*sigma_spec,2000)
    results.append((sigma_spec,best))
    if not best.get('dev_candidate_pass'):
        best=extend(sigma_spec)
        results.append((sigma_spec,best))
    if best.get('dev_candidate_pass'):
        finish_candidate(sigma_spec,best)
        return
    main_2k=max([r for r in read_rows(CAMPAIGN/'c_main/checkpoint_metrics.jsonl') if r['step']<=2000],key=quality_key)
    sigma_2k=max([r for r in read_rows(CAMPAIGN/'c_sigma/checkpoint_metrics.jsonl') if r['step']<=2000],key=quality_key)
    depth_sigma='balanced' if quality_key(main_2k)>=quality_key(sigma_2k) else 'permutation'
    depth_spec=('c_depth',tuple(range(8)),True,depth_sigma)
    best=train(*depth_spec,2000)
    results.append((depth_spec,best))
    if not best.get('dev_candidate_pass'):
        best=extend(depth_spec)
        results.append((depth_spec,best))
    if best.get('dev_candidate_pass'):
        finish_candidate(depth_spec,best)
        return
    spec,best=max(results,key=lambda pair:quality_key(pair[1]))
    atomic_json(CAMPAIGN/'search_result.json',{'status':'DEVELOPMENT_FAIL','run':spec[0],'best':best})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight','smoke','resume','campaign','regression'])
    args=parser.parse_args()
    if not Path(COMMON[5]).is_file():
        raise RuntimeError('configured FlashVSR reference source does not exist: '+COMMON[5])
    if args.mode in ('preflight','regression'):
        run_gpu([PYTHON,'-m','pytest','-q'],args.mode+'_full',search=args.mode=='preflight')
    elif args.mode=='smoke':
        train('smoke',(0,1,2,3),True,'balanced',4,stop=False)
    elif args.mode=='resume':
        train('smoke',(0,1,2,3),True,'balanced',8,resume=True,stop=False)
    else:
        campaign()
