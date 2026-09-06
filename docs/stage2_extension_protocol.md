# Stage 2 Near-Selected Extension Protocol

Date: 2026-09-03

## Fixed comparison

- Scenes: `chair`, `lego`, `drums`.
- Arms: `baseline_no_geometry`, `rre_geometry`, `plucker_geometry`.
- Seen training views: `transforms_train` indices `[0, 33, 66, 99]`.
- Near-held-out validation views are the nearest unused `transforms_train`
  cameras by normalized camera-center angle:
  - chair: `[30, 7, 14, 41]`
  - lego: `[41, 36, 58, 43]`
  - drums: `[1, 72, 98, 18]`
- Far-held-out final views: `transforms_test` indices
  `[0, 50, 100, 150]`.
- Geometry adapters train for 2000 steps with retained checkpoints at
  500, 1000, 1500, and 2000.
- Wan VAE, Wan DiT, FlashVSR projector, and Stage 1 bridge remain frozen.

The protocol manifest records the transforms hashes and exact camera
distances. Any split or dataset drift stops the campaign.

## Selection and final test

The `search` phase can load only seen and near-held-out views. Each geometry
arm is selected independently for each scene using near-held-out decoded
metrics:

1. highest mean correct PSNR;
2. within 0.01 dB of the maximum, highest SSIM;
3. then lowest LPIPS;
4. then the earlier checkpoint.

`frozen_candidates.json` stores the selected checkpoint paths, hashes, steps,
protocol hash, parent-checkpoint hash, and near metrics. The `test` phase
validates those hashes and evaluates only the frozen candidates on far views.
Completed scene/arm results are atomically committed and cannot be overwritten.

The far cameras were already evaluated by the 2026-09-02 campaign. They are
therefore run once under this protocol but are not pristine project-history
holdouts.

## Evidence and verdicts

- Training logs retain per-step loss, gradient norm, sigma, memory, and timing.
- Seen and near metrics are reported at every retained checkpoint.
- Far metrics are reported only for the frozen checkpoints; no far checkpoint
  trajectory is generated.
- PSNR, SSIM, LPIPS, MAE, pose interventions, and fixed-noise qualitative
  contact sheets are retained before aggregation.
- GT depth remains evaluation-only. Without a validated metric depth scale,
  reprojection and cross-view reconstruction are marked unavailable.

Camera usage, image quality, and geometry consistency remain independent
verdicts. One seed is descriptive evidence, not a statistical-significance
claim.
