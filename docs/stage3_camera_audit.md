# Stage 3.1 Camera Scope Audit

The camera audit is a frozen-checkpoint diagnostic. It does not retrain Wan,
change the existing Stage 3.1 campaign, or claim held-out generalization,
NVS, 3DGS, or formal Stage 4 effectiveness.

## Intervention scopes

`correct` and `correct_repeat` are identical controls. `target_drop` removes
only the target LR image. `shuffle_fusion` changes the camera consumed by
`LRViewFusion` while keeping the geometry camera correct. `shuffle_geometry`
changes only the camera passed to the Wan/Full-RRE path. `shuffle_all` changes
both camera paths. `shuffle_pair` applies the same auxiliary permutation to LR
images and cameras, preserving the target at index 0. The historical
`shuffle_camera` name remains a fusion-only compatibility alias and should not
be used for new conclusions.

## Recorded diagnostics

With `--save-diagnostics`, `seen-eval` records the prepared feature delta,
camera tensor hashes, per-step velocity traces, per-block geometry residual
summaries, and Wan injection summaries. All intervention modes use the same
inference seed and therefore the same initial noise for paired comparisons.

## Structural option

The default `epipolar_attention=global_bias` preserves existing behavior. A
new configuration may set `epipolar_attention=local_band` and
`epipolar_band` (in patch-coordinate units) to restrict valid source keys to a
local epipolar band. Degenerate camera pairs fall back to global keys and the
null candidate remains available, so the base pipeline has no external 3D
dependency.
