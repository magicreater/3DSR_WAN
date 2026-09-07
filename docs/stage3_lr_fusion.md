# Stage 3: Pose-aware Cross-view LR Fusion

## Scope

Stage 3 adds a static-3D LR fusion branch between the existing frozen LR
encoder and LR bridge. It keeps the complete RRE camera path active. Temporal
4DSR inputs bypass the fusion branch.

The four matched arms are:

| Arm | LR fusion | Claim it isolates |
| --- | --- | --- |
| A0 | off | Existing LR bridge plus full RRE baseline |
| A1 | same-view attention | Added attention capacity without cross-view evidence |
| A2 | visual cross-view attention | Benefit from other LR views |
| A3 | epipolar-biased cross-view attention | Additional benefit from camera geometry |

Stage 3 must prove two separate statements: auxiliary LR views improve the
target view, and correct camera geometry makes that evidence more useful.
Lower training loss or sharper images alone do not prove either statement.

## Implemented interfaces

- `LRViewFusion` provides `off`, `same_view`, `visual`, and `epipolar` modes,
  query chunking, source masks, a null evidence candidate, single-view fallback,
  and temporal bypass.
- `patch_fundamental_matrices` converts canonical pixel intrinsics and OpenCV
  camera-to-world transforms into patch-coordinate epipolar constraints.
- `Stage3Conditioning.prepare_multiview` computes frozen LR features and fusion
  once before a denoising trajectory. `predict` reuses the result while the full
  RRE path continues to receive the original camera.
- Stage 3 checkpoints contain the bridge, full RRE, fusion, optimizer, and RNG
  state. Wan, VAE, and the LR projector remain external frozen assets.
- `stage3_experiment.py` provides `inspect`, `train`, `validate`, `intervene`,
  `freeze`, and one-time `test` commands.

## Data protocol

- Shared training scenes: `chair`, `lego`, `drums`, `hotdog`, `mic`.
- Development validation covers held-out views from those scenes plus the
  held-out scene `ficus`.
- Final test scenes are `materials` and `ship` and are inaccessible to
  checkpoint selection.
- Train seeds are 42 and 43. Validation uses inference seed 3301. Final testing
  uses 3302-3305.
- Every arm uses the same scene routes, view sampler, 4,000 optimizer steps,
  gradient accumulation, checkpoint interval, and inference schedule.

## Command sequence

The commands below are templates. Replace the asset paths before starting any
experiment.

```bash
PY=/home/linzizhuo/miniconda3/envs/rl3dsr-stcdit/bin/python
ROOT=/data/linzizhuo/RL3DSR_WAN_REMO
CONFIG=$ROOT/configs/stage3/A3.json

$PY $ROOT/scripts/stage3_experiment.py inspect --config $CONFIG

$PY $ROOT/scripts/stage3_experiment.py train \
  --config $CONFIG --seed 42 \
  --dataset-root /path/to/nerf_synthetic \
  --model-dir /path/to/Wan2.1-T2V-1.3B \
  --lq-source /path/to/projector.py \
  --lq-checkpoint /path/to/projector.pt \
  --bridge-checkpoint /path/to/stage1_bridge.pt \
  --rre-checkpoint /path/to/full_rre.pt \
  --output-dir /path/to/stage3/A3/seed42

$PY $ROOT/scripts/stage3_experiment.py validate \
  --config $CONFIG --checkpoint /path/to/stage3_step_0500.pt \
  --dataset-root /path/to/nerf_synthetic \
  --model-dir /path/to/Wan2.1-T2V-1.3B \
  --lq-source /path/to/projector.py \
  --lq-checkpoint /path/to/projector.pt \
  --bridge-checkpoint /path/to/stage1_bridge.pt \
  --rre-checkpoint /path/to/full_rre.pt \
  --output-dir /path/to/validation/step0500

$PY $ROOT/scripts/stage3_experiment.py intervene \
  --config $CONFIG --checkpoint /path/to/frozen_stage3.pt \
  --dataset-root /path/to/nerf_synthetic \
  --model-dir /path/to/Wan2.1-T2V-1.3B \
  --lq-source /path/to/projector.py \
  --lq-checkpoint /path/to/projector.pt \
  --bridge-checkpoint /path/to/stage1_bridge.pt \
  --rre-checkpoint /path/to/full_rre.pt \
  --output-dir /path/to/interventions

$PY $ROOT/scripts/stage3_experiment.py freeze \
  --config $CONFIG \
  --rows /path/to/validation/*/evaluation_rows.jsonl \
  --manifest /path/to/A3_seed42_frozen.json

$PY $ROOT/scripts/stage3_experiment.py test \
  --config $CONFIG --manifest /path/to/A3_seed42_frozen.json \
  --dataset-root /path/to/nerf_synthetic \
  --model-dir /path/to/Wan2.1-T2V-1.3B \
  --lq-source /path/to/projector.py \
  --lq-checkpoint /path/to/projector.pt \
  --bridge-checkpoint /path/to/stage1_bridge.pt \
  --rre-checkpoint /path/to/full_rre.pt \
  --output-dir /path/to/final-test
```

Run validation for every immutable candidate checkpoint, then freeze exactly
one checkpoint per arm and training seed. The final-test claim is written
before test data are read and cannot be repeated by changing the output path.

## Decision gates

1. Structural gate: synthetic geometry, shapes, disabled behavior, gradients,
   checkpoint round-trip, resume alignment, and output protection pass. A later
   real-GPU forward must also pass before training starts.
2. Evidence gate: with target LR fixed, correct auxiliaries outperform removed
   or duplicated auxiliaries; correct fusion-camera correspondence outperforms
   shuffled correspondence; local auxiliary edits cause a measurable target
   response. These comparisons must exceed repeated-run BF16 jitter.
3. Shared-training gate: A2 must improve over A1 on scene-equal held-out
   validation, and A3 must improve over A2 without a material worst-scene
   regression. Both train seeds must show the same direction.

Provisional thresholds for advancing the fusion contribution into Stage 4 are:

- A3 versus A0: at least +0.20 dB scene-equal PSNR and -0.005 LPIPS, with SSIM
  non-decreasing;
- A3 versus A2: at least +0.10 dB PSNR or -0.003 LPIPS;
- no validation scene worse than A0 by more than 0.20 dB PSNR or +0.005 LPIPS;
- removal or camera shuffling removes at least half of A3's gain over A0;
- the above directions hold for both train seeds.

After Stage 4, RL is justified only if the large-scale frozen final test keeps
the A3-over-A0 image-quality gain, correct-camera interventions remain causal,
and downstream multi-view reconstruction or 3DGS NVS quality is non-degrading.
RL should target a remaining measured limitation rather than compensate for an
unproven conditioning path.

## Current status

As of September 6, 2026, no Stage 3 training, real-Wan forward, 50-step decode,
or 3DGS evaluation has been run by this implementation task. Passing lightweight
tests establishes implementation readiness only.
