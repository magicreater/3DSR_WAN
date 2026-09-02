# Stage 2 Geometry Adapter Results

Date: 2026-09-02

## Scope

This is the first bounded Stage 2 experiment on NeRF Synthetic `chair`, using
training views `[0, 33, 66, 99]`, HR `256x256`, 4x antialiased bicubic LR,
the frozen Stage 1 `c_main/best_dev.pt` bridge, and a frozen Wan VAE/DiT.
Only the new geometry adapter was optimized for 500 steps with seed 42.

The experiment is a mechanism and consistency check, not a generalization
claim. Stage 1's prior quality limitation remains in force.

## Architecture

`GeometryConditioner` converts canonical `K` and
`T_world_from_camera` into token-aligned ray features. RRE uses a relative
ray frame, normalized relative camera origin, ray direction, normalized image
coordinates, intrinsics, and a sequence-kind indicator. The Plucker control
uses relative ray direction, moment, origin, image coordinates, intrinsics,
and sequence-kind indicator.

Both representations use a 192-dimensional encoder, four-head attention over
the view axis, a two-layer MLP, timestep FiLM, and four zero-initialized
`192 -> 1536` block projections for Wan blocks 0-3.

## Frozen and Trainable Parameters

| Component | Status |
|---|---|
| Wan VAE | frozen |
| Wan DiT, patch embedding, blocks, output head | frozen |
| text context | frozen deterministic zero context |
| FlashVSR LR projector | frozen |
| Stage 1 LR bridge | frozen, loaded from `c_main/best_dev.pt` |
| geometry encoder and attention | trainable |
| geometry block projections | trainable, zero initialized |

Trainable counts:

| Adapter | Parameters |
|---|---:|
| RRE | 1,536,430 |
| Plucker | 1,535,266 |

The Wan DiT, VAE, and Stage 1 bridge had no gradient tensors in the smoke and
500-step runs. Both final checkpoints reload with exact parameter equality.

## Final Fixed-Sigma Results

The baseline is the same frozen Stage 1 parent checkpoint with geometry
disabled. Values below are the final evaluation at step 500, sigma `0.5`,
with identical latent noise and text context.

| Run | Flow loss | PSNR | SSIM | LPIPS | Correct-vs-baseline latent delta |
|---|---:|---:|---:|---:|---:|
| Stage 1 baseline | 0.0086323 | 28.8913 | 0.941875 | 0.0401143 | 0 |
| RRE | 0.0080722 | 28.8736 | 0.942953 | 0.0388098 | 0.01703 |
| Plucker | 0.0080265 | 28.9335 | 0.942944 | 0.0393181 | 0.01681 |

Relative to baseline:

- RRE: flow loss `-6.49%`, SSIM `+0.00108`, LPIPS `-3.25%`, PSNR `-0.0177 dB`.
- Plucker: flow loss `-7.01%`, SSIM `+0.00107`, LPIPS `-1.98%`, PSNR `+0.0423 dB`.

Correct-camera flow loss was lower than shuffled-camera loss for both adapters:

- RRE: `0.0080722` vs `0.0085411`.
- Plucker: `0.0080265` vs `0.0085841`.

This supports camera usage after training. It does not establish full
multi-view geometric correctness or cross-scene generalization.

## Qualitative Review

The fixed-noise contact sheets are retained at:

- `artifacts/stage2/rre_500/qualitative_contact.png`
- `artifacts/stage2/plucker_500/qualitative_contact.png`

The RRE sheet shows geometry-correct outputs visually close to the Stage 1
baseline at this short budget, with no obvious camera-induced view collapse.
The shuffled condition remains similar, so the current visual evidence is
weaker than the flow-loss evidence. A longer run and multi-scene evaluation
are required before claiming improved 3D consistency.

## Reproduction

Use the server checkout and the retained commands:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
python scripts/stage2_experiment.py \
  --model-dir models/Wan2.1-T2V-1.3B \
  --scene datasets/nerf_synthetic/chair \
  --lq-source /home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py \
  --lq-checkpoint /home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt \
  --parent-checkpoint artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt \
  --output-dir artifacts/stage2/rre_500 \
  --representation rre --steps 500 --eval-interval 50 --checkpoint-interval 100
```

Change `--representation rre` to `plucker` and use a separate output
directory for the matched control.

Each run retains `run_manifest.json`, `camera_cache.pt`, JSONL/CSV training
and evaluation data, step checkpoints, training state, final checkpoint,
reload result, curves, diagnostics, and qualitative review metadata.

## Status and Next Step

Stage 2 implementation and short-run validation: **PASS**.

Camera usage sanity: **PASS for fixed-sigma flow intervention**.

Geometry-aware SR quality: **INCONCLUSIVE**. Plucker has a small PSNR gain;
RRE has the better LPIPS reduction; neither result is sufficient to claim a
robust geometry-consistency improvement from one scene and one seed.

Next experiment should be a pre-registered 1000-2000 step extension on the
same scene followed by the same configuration on at least two additional
NeRF Synthetic scenes. Do not unfreeze Wan or the Stage 1 bridge until those
comparisons are complete.
