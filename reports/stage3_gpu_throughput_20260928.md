# Stage 3 A5 training throughput, 2026-09-28

## Changes

- Cache each scene's index and CPU camera batch instead of rebuilding them for every sampled group.
- Transfer the epipolar valid-pair mask from GPU to CPU once and reuse the computed pair list.
- Reuse successful camera validation while tensor versions and camera metadata are unchanged.
- Offer `paired_wan_forward=true` for A5 camera ranking: concatenate the correct and wrong conditions into one Wan batch. It defaults to `false`; the running four-arm campaign was not changed.

These target the Python/data overhead and repeated GPU work described in the [PyTorch performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide) and [data-loading guide](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html). The paired forward is a project-specific optimization, not a result claimed by those guides.

## Matched diagnostic

One idle RTX 4090 (GPU 2), four-view A5 with LoRA, seed 42, identical parent checkpoint, dataset, sample/view/noise/dropout sequence. Each run used 20 training steps; means and medians exclude the first three warm-up steps. The baseline was commit `0de4fdc`; the final implementation was commit `5f7893f`. Outputs are in `artifacts/benchmark_gpu2_20_steps` in the A5 LoRA worktree and `artifacts/benchmark_gpu2_paired_forward` in this worktree on `server-4090`.

| Training path | Mean step | Median step | Peak allocated GPU memory |
|---|---:|---:|---:|
| Baseline | 6.587 s | 6.566 s | 11,061 MiB |
| Scene/pair caching | 6.231 s | 6.205 s | 11,061 MiB |
| Plus camera validation caching | 6.069 s | 6.064 s | 11,061 MiB |
| Plus paired Wan forward | **4.149 s** | **4.126 s** | **11,001 MiB** |

The final path reduced mean step time by **37.0%** (1.59 times as many steps per unit time). The maximum absolute per-step loss difference was 0.000405; the BF16 batched path is not bitwise identical. A real Wan one-step eight-view LoRA smoke also completed, with a 16,059 MiB peak allocated memory and nonzero LoRA gradient. GPU 2 utilization sampled during the final run was mostly 29–47%; this short sample is not a stable utilization estimate.

## Validation and use

The full remote test suite passed: **467 passed, 9 skipped**. The final 20-step run wrote `stage3_step_0020.pt`; the existing live four-arm campaign remains on its original checkout and configuration. The speed result does not establish equal final PSNR/SSIM/LPIPS or camera-use quality. Keep the flag off for those running arms and compare it in a fresh matched validation run before adopting it for a long training campaign. Stage 4 remains on hold under the existing readiness gates.
