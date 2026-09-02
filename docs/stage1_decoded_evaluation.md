# Stage 1 Decoded-Space Evaluation

Date: 2026-09-01

## Verdict

**Overall: CAUSAL BUT WEAK VISUAL EFFECT**

The decoded results strengthen the Stage 1 causal conclusion, but they do not establish a usable end-to-end SR sampler.

- **Static 3D: VISIBLE SUPPORT.** Correct LR improves every view over shuffled LR at every tested one-step noise level, and the images visibly retain the correct chair silhouette and texture.
- **Dynamic 4D: CAUSAL BUT WEAK / NOISE-DEPENDENT.** Correct LR improves mean decoded metrics for `sigma <= 0.8`, especially against the disabled adapter, but frame-level wins against shuffled LR are not uniform. At `sigma=0.95`, correct LR is worse on average.
- **Eight-step trajectory: not generation-quality.** It preserves measurable conditioning dependence, but produces severe color/stripe/texture artifacts. It is supplementary causal evidence only.

The original Stage 1 causal PASS is not invalidated. This evaluation narrows its interpretation: the trained adapter controls frozen Wan predictions and decoded content, but the current 50-step micro-overfit checkpoint is not a practical SR inference checkpoint.

## Protocol

No training was run and neither adapter checkpoint was modified.

- Wan2.1 T2V 1.3B VAE and DiT: frozen BF16.
- FlashVSR LQ projector: frozen BF16.
- Adapter checkpoints: existing `3d_adapter.pt` and `4d_adapter.pt`.
- Text: fixed all-zero `[1,512,4096]` BF16 context.
- Degradation: deterministic 4x bicubic antialiased downsample, HR `128x128`, LR `32x32`.
- Conditions: correct, shuffled, neutral, adapter disabled.
- One-step estimate: `x0_hat = x_sigma - sigma * v_prediction` at `sigma = 0.2, 0.5, 0.8, 0.95`.
- Supplementary trajectory: 8 FlowMatch Euler steps, Wan schedule shift `5`, shared initial noise, explicit final `sigma=0`.
- Metrics: RGB MAE, PSNR, windowed SSIM, and 4D adjacent-frame `temporal_delta_mae` after display-range clipping.
- LPIPS was not run because no LPIPS package or external perceptual weights were added.

## Static 3D

Dataset: NeRF Synthetic `chair`, train views `[0,14,28,42]` and `[56,70,84,99]`, evaluated as two `V=4` tensors. VAE processing remained independently per view, followed by regrouping to `[B,16,V,h,w]` for Wan DiT.

### One-step decoded metrics

| Sigma | Condition | MAE down | PSNR up | SSIM up | Correct all-metric wins vs shuffled |
|---:|---|---:|---:|---:|---:|
| 0.20 | correct | 0.02075 | 24.681 | 0.8891 | 8/8 |
| 0.20 | shuffled | 0.02609 | 23.000 | 0.8600 | |
| 0.50 | correct | 0.03019 | 22.159 | 0.8289 | 8/8 |
| 0.50 | shuffled | 0.04247 | 19.874 | 0.7651 | |
| 0.80 | correct | 0.04937 | 19.063 | 0.7716 | 8/8 |
| 0.80 | shuffled | 0.06213 | 17.385 | 0.7076 | |
| 0.95 | correct | 0.08427 | 15.772 | 0.7441 | 8/8 |
| 0.95 | shuffled | 0.08745 | 15.453 | 0.7166 | |

Across `sigma=0.2/0.5/0.8`, correct LR gains `+1.881 dB` mean PSNR over shuffled LR and wins all three metrics on `24/24` views. Across all four sigmas it wins all three metrics on `32/32` views versus shuffled LR and `31/32` versus the disabled adapter.

The contact sheets show the same effect. Correct LR recovers the corresponding chair orientation, silhouette, and green upholstery. Shuffled LR introduces the other view group's shape; neutral and disabled conditions largely collapse into texture-like blobs. Correct one-step results still remain below the bicubic baseline (`25.573 dB`) and VAE ceiling (`31.877 dB`), so the result demonstrates control rather than final SR quality.

![3D one-step sigma 0.5](../artifacts/stage1/decoded/3d/3d_one_step_sigma_0p50_contact.png)

![3D one-step sigma 0.8](../artifacts/stage1/decoded/3d/3d_one_step_sigma_0p80_contact.png)

### Eight-step trajectory

Correct LR reaches `13.302 dB / 0.6863 SSIM`, versus shuffled `12.687 dB / 0.6404` and disabled `8.715 dB / 0.3823`. Correct wins all three metrics on `8/8` views versus both controls. However, the images have strong yellow/brown texture and color artifacts and are not usable SR outputs.

![3D eight-step trajectory](../artifacts/stage1/decoded/3d/3d_trajectory_8_contact.png)

## Dynamic 4D

Fixture: four deterministic coherent-motion clips, seeds `0-3`, each `T=9`. Native Wan temporal VAE produced `[1,16,3,16,16]` and decoded back to all 9 frames. Frames were never flattened into independent images.

### One-step decoded metrics

| Sigma | Condition | MAE down | PSNR up | SSIM up | Correct all-metric wins vs shuffled |
|---:|---|---:|---:|---:|---:|
| 0.20 | correct | 0.01164 | 34.775 | 0.9763 | 23/36 |
| 0.20 | shuffled | 0.01235 | 34.360 | 0.9744 | |
| 0.50 | correct | 0.02379 | 29.462 | 0.9553 | 16/36 |
| 0.50 | shuffled | 0.02442 | 28.974 | 0.9518 | |
| 0.80 | correct | 0.05157 | 23.196 | 0.8864 | 27/36 |
| 0.80 | shuffled | 0.06351 | 21.788 | 0.8543 | |
| 0.95 | correct | 0.14005 | 14.969 | 0.6595 | 3/36 |
| 0.95 | shuffled | 0.13466 | 15.494 | 0.6728 | |

For `sigma=0.2/0.5/0.8`, correct LR gains `+0.770 dB` mean PSNR over shuffled LR and wins all three metrics on `66/108` frames. It wins `107/108` frames against the disabled adapter. The strongest shuffled-LR separation occurs at `sigma=0.8`: `+1.407 dB`, `+0.0322 SSIM`, and `27/36` all-metric wins.

At `sigma=0.95`, the relationship reverses: correct LR is `-0.525 dB` below shuffled LR and wins all three metrics on only `3/36` frames. The failure gallery shows over-conditioned, enlarged, wrong-color objects instead of the target cyan disc. These cases are included rather than filtered.

![4D one-step sigma 0.5](../artifacts/stage1/decoded/4d/stage1_motion_seed_0_one_step_sigma_0p50_contact.png)

![4D one-step sigma 0.8](../artifacts/stage1/decoded/4d/stage1_motion_seed_0_one_step_sigma_0p80_contact.png)

![4D sigma 0.95 failures](../artifacts/stage1/decoded/4d/4d_sigma_0p95_failure_gallery_1.png)

### Temporal behavior

| Mode | Correct | Shuffled | Disabled |
|---|---:|---:|---:|
| sigma 0.20 | 0.00900 | 0.00927 | 0.01393 |
| sigma 0.50 | 0.01230 | 0.01310 | 0.02201 |
| sigma 0.80 | 0.02125 | 0.02222 | 0.02718 |
| sigma 0.95 | 0.02725 | 0.02675 | 0.02547 |
| 8-step trajectory | 0.03598 | 0.03897 | 0.03801 |

Correct LR improves adjacent-frame change fidelity through `sigma=0.8` and on the short trajectory, but not at `sigma=0.95`.

### Eight-step trajectory

Correct LR is better than shuffled LR on mean metrics (`13.471` vs `12.657 dB`, `0.5438` vs `0.5067` SSIM), with `18/36` all-metric frame wins. Against disabled, correct has higher SSIM but slightly worse PSNR and MAE. The decoded video has severe vertical stripe and texture artifacts, so this trajectory does not support a generation-quality claim.

![4D eight-step trajectory](../artifacts/stage1/decoded/4d/stage1_motion_seed_0_trajectory_8_contact.png)

## Numerical and engineering validation

- All decoded tensors were finite; nonfinite count was zero.
- All decoded values were already within `[-1,1]`; out-of-range ratio was zero.
- Peak allocated GPU memory: 3D `4697.7 MiB`, 4D `4668.3 MiB`.
- Full decoded evaluation runtime: `51.65 s` on one RTX 4090.
- Adapter SHA256:
  - 3D: `20596ec45b7202574050c8e3a0191be3a8f0b3cb9d480af146b4bec51c2be6e7`
  - 4D: `ae04e8f1812cae6212415c55794245bce2571099b76983e1d5d0f75bba552c26`
- Full repository suite with real Wan, LQ, and both adapters: **126 passed**.
- Real decoded integration test verified trained checkpoint loading, changed prediction, independent-view 3D decode, native-temporal 4D decode, finite results, and frozen backbone/projector.
- Stage 0 real regressions included 3D cross-view isolation and native 4D `T=1/5/9/17` VAE/DiT forwards.

Raw metrics and all generated media are in `artifacts/stage1/decoded/decoded_metrics.json`, `artifacts/stage1/decoded/3d/`, and `artifacts/stage1/decoded/4d/`.

## Limitations

- Both adapters are 50-step micro-overfit checkpoints on tiny deterministic datasets.
- One-step decoded metrics measure conditional recovery from a noisy target latent; they are not a direct deterministic SR benchmark against bicubic.
- The eight-step path is deliberately short and uses empty text. Its failure does not erase conditioning causality, but it shows the current checkpoint/sampler combination is not ready for inference-quality SR.
- The synthetic 4D fixture has simple motion and appearance; no claim is made about real-world video generalization.
- No perceptual learned metric was added because doing so would require a new dependency and weights.


## Later Stage 1 C closeout

This historical decoded evaluation remains unchanged. The later Experiment C closeout is recorded in artifacts/stage1/c_campaign_20260902/report_zh.md; its final four-seed strict verdict is FAIL.
