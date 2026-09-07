# Stage 3 Seen-view SR Protocol

Date: 2026-09-07

## Claim boundary

This campaign tests whether the shared Stage 3 model can fit and improve SR on
the official training views while causally using auxiliary LR views and their
camera correspondence. It does not evaluate held-out views or scenes, novel
view synthesis, 3DGS, depth reprojection, 3D consistency, 4DSR, or RL.

## Fixed experiment

- Scenes: `chair`, `lego`, `drums`, `hotdog`, `mic`.
- Arms: A0 no fusion, A1 same-view attention, A2 visual cross-view attention,
  A3 epipolar-biased cross-view attention.
- Training seeds: 42 and 43; 4,000 optimizer steps; checkpoint every 500 steps.
- Each target view uses the three nearest optical-axis neighbours from the same
  official train split. The immutable seen-view manifest contains all 500
  target groups plus fixed 20-group probe and 80-group intervention subsets.
- Wan, VAE, and the FlashVSR LR projector remain frozen. The Stage 1 bridge,
  Full RRE, and applicable fusion module are trainable. Full RRE starts from the
  same seeded zero-output initialization in every arm; no per-scene Stage 2 RRE
  checkpoint is loaded.
- Main evaluation uses the step-4000 checkpoint without training-set candidate
  selection. Pure-noise 50-step decoding uses inference seeds 3302-3304.
- Baselines are Bicubic and the Wan VAE HR round-trip ceiling.

## Commands

Run from the repository root with `PYTHONPATH=src` and the verified
`rl3dsr-stcdit` interpreter. The asset paths below are fixed for this server.

```bash
PY=/home/linzizhuo/miniconda3/envs/rl3dsr-stcdit/bin/python
DATA=datasets/nerf_synthetic
MODEL=models/Wan2.1-T2V-1.3B
LQ_SOURCE=/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py
LQ_CHECKPOINT=/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt
BRIDGE=artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt
ROOT=artifacts/stage3_seen_sr_20260907

PYTHONPATH=src $PY scripts/stage3_experiment.py prepare-seen \
  --config configs/stage3/A0.json --dataset-root $DATA \
  --manifest $ROOT/manifest/seen_groups.json

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 $PY scripts/stage3_experiment.py train \
  --config configs/stage3/A0.json --seed 42 --dataset-root $DATA \
  --model-dir $MODEL --lq-source $LQ_SOURCE --lq-checkpoint $LQ_CHECKPOINT \
  --bridge-checkpoint $BRIDGE --output-dir $ROOT/train/A0/seed42
```

Do not pass `--rre-checkpoint`. Repeat training for A0-A3 and seeds 42/43 on
exclusive idle GPUs. Use `--resume` only once after an infrastructure failure
and only with the matching output directory and immutable checkpoint.

For every retained checkpoint, run `seen-eval --subset probe
--inference-seeds 3301 --modes correct`. For step 4000, run:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 $PY scripts/stage3_experiment.py seen-eval \
  --config configs/stage3/A0.json --checkpoint $ROOT/train/A0/seed42/stage3_step_4000.pt \
  --dataset-root $DATA --model-dir $MODEL --lq-source $LQ_SOURCE \
  --lq-checkpoint $LQ_CHECKPOINT --bridge-checkpoint $BRIDGE \
  --seen-manifest $ROOT/manifest/seen_groups.json --subset full \
  --inference-seeds 3302 3303 3304 --modes correct --save-images \
  --output-dir $ROOT/full/A0/seed42
```

The intervention run uses subset `intervention` and modes `correct
correct_repeat remove duplicate shuffle_camera`. Raw JSONL/CSV, checkpoints,
images, manifests, plots, and reports remain under the campaign root.

## Verdicts

- `SR_FIT_PASS`: A3 improves scene-equal PSNR over Bicubic for both training
  seeds, with non-decreasing SSIM and non-worsening LPIPS/MAE.
- `CROSS_VIEW_PASS`: A2 improves over A1 and correct A2 auxiliaries outperform
  removed and duplicated auxiliaries; A0/A1 remain within repeat jitter.
- `GEOMETRY_PASS`: A3 improves over A2, correct A3 fusion cameras outperform
  shuffled cameras, and A2 remains invariant to fusion-camera shuffling.

There is no fixed 0.1 or 0.2 dB effect-size threshold. PSNR gains must exceed
the measured same-input repeat jitter and all directions must hold for both
training seeds. Failure is reported without threshold changes or extra tuning.
