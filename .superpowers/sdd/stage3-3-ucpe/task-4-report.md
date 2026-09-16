# Task 4 report: gated A5 direct pairing supervision

## Inherited-state audit

- Started from commit `3d51bf3` with uncommitted partial Task 4 work in exactly four owned production files plus untracked `tests/test_stage3_pairing.py`.
- Preserved that work and Tasks 1-3. No server, Stage 3.2, Task 5, GitHub, or experiment state was touched.
- Read `AGENTS.md`, `docs/PROJECT_SPEC.md`, the full Stage 3.3 plan, and the enriched Task 4 brief before editing. Used CodeGraph to trace Stage 3 config, fusion, conditioning, checkpoint, and training paths.
- The inherited implementation already contained test-first coverage for A5 activation, mutual-nearest pairing, fail-closed coverage, InfoNCE, detached feature discovery, calibration, persistence, resume state, and telemetry. The first local command used the worktree `.venv` and was blocked during collection because that environment lacks torch. After the controller supplied the established local PyTorch environment, all focused tests ran successfully.

## Completed implementation

- Added explicit `A5` configuration while preserving A4 and A0-A3 inference behavior. The fixed protocol is surfaced in config and manifest: temperature `0.07`, minimum per-pair coverage `0.05`, 8 calibration batches, target gradient ratio `0.25`, and clip `[0.01, 10]`.
- Added detached cosine mutual-nearest correspondence discovery inside the correct local epipolar band for every valid ordered batch/target/source pair. Any required pair below coverage fails closed.
- Added an opt-in differentiable A5 pairing state. LR features are detached before Q/K projection, so correspondence discovery and pairing loss cannot reach the frozen LR encoder, while Q/K projection weights receive gradients. Nonzero-distortion UCM inputs are rejected because the candidate band is pinhole-only.
- Added InfoNCE over correct-camera transformed Q/K positives with wrong-camera transformed/band-admitted hard negatives.
- Added deterministic eight-batch calibration and immutable preflight artifact/manifest binding. Preflight uses private view, camera-pairing, noise, sigma, and dropout generators; hashes actual HR/LR/latent/noise batch tensors and camera provenance; and restores Python, NumPy, CPU/CUDA torch RNG, module modes, and mutable buffers. It asserts no model parameter mutation and never steps or mutates the optimizer.
- Added fixed-weight resume semantics and model-only/scratch recalibration. Checkpoints preserve pairing RNG state and calibrated weight while legacy/default-off paths retain zero pairing weight.
- Added separate JSONL/CSV telemetry for main flow, output rank, unweighted/weighted pairing loss, count/coverage, fixed weight, per-component fusion gradient norms, and final combined fusion gradient norm.
- Corrected A5-specific configuration diagnostics and documented the opt-in tuple return contract on `prepare_multiview`.

## Verification

- `C:\Users\22973\miniconda3\envs\pytorch\python.exe -m pytest -q tests/test_stage3_pairing.py` -> `12 passed` before the added UCM guard test, then included in broader runs.
- Focused/adjacent: `tests/test_stage3_pairing.py tests/test_lr_fusion.py tests/test_stage3_conditioning.py tests/test_stage3_protocol.py tests/test_stage3_runner.py` -> `86 passed in 2.97s`.
- All Stage 3-focused tests plus LR fusion -> `120 passed in 3.18s`.
- `python -m compileall -q src scripts tests` -> PASS.
- `git diff --check` -> PASS (only Git's informational LF-to-CRLF warnings).
- The local all-suite attempt reached 17 tests and then the known Windows MKL abort in `tests/test_colmap_camera.py`; no Python assertion failure was reported. Controller should run the authoritative full suite in the server environment.

## Files

- `src/rl3dsr/models/wan/lr_fusion.py`
- `src/rl3dsr/models/wan/stage3.py`
- `src/rl3dsr/validation/stage3_protocol.py`
- `scripts/stage3_experiment.py`
- `tests/test_stage3_pairing.py`

## Review fix round 1

Addressed every scoped review finding on top of `179d148`:

- Split A5 flow conditioning from pairing evidence under target LR dropout. Flow fusion receives the dropped LR, while mutual-nearest discovery and differentiable Q/K use the original frozen LR features. Added a test proving matches and Q/K state are identical with and without flow target dropout.
- Changed calibration from mean-of-per-batch norms to the global L2 norm of the componentwise mean of all eight gradient vectors. Per-batch norms remain diagnostics and the exact reduction is persisted.
- Changed step telemetry to accumulate the already loss-reduced microbatch gradient vectors componentwise before taking flow/rank/pairing norms. These are now directly comparable with the final combined fusion gradient norm. Rank fusion gradients are sliced from the existing all-trainable rank traversal, eliminating the duplicate rank autograd pass.
- Intersected correspondence candidates with per-query/source `usable`; unusable target rows cannot match and still remain in the coverage denominator.
- Added fail-before-restore A5 resume validation for exact protocol equality, clipped finite weight, preflight existence/hash, artifact protocol/weight, and checkpoint weight.
- Expanded preflight isolation across conditioning, VAE, and DiT wrappers/modules: modes, buffers, projector caches, fusion/bridge/geometry diagnostics, DiT injection diagnostics, Python/NumPy/torch CPU/CUDA RNG, and parameter-version assertions are restored or fail closed.

TDD evidence for this round:

- New focused tests first produced `6 failed, 11 passed`: unusable-row matching, original-LR dropout isolation, vector-reduced calibration, resume validation, cache/diagnostic isolation, and accumulated-gradient telemetry all failed against `179d148`.
- After the minimal fixes, `tests/test_stage3_pairing.py` -> `17 passed`.
- Focused/adjacent Task 4 suite -> `90 passed in 2.99s`.
- All Stage 3-focused tests plus LR fusion -> `124 passed in 3.17s`.
- `compileall` and `git diff --check` are rerun before the fix commit.

## Review fix round 2

Addressed the remaining resume-integrity and RNG-order findings on top of `fda8c86`:

- `validate_pairing_resume` now accepts only the corrected immutable preflight schema. It verifies the exact vector-reduction declaration, exactly eight deterministic seed records, complete batch/per-pair records, positive finite reduced and per-batch norms, per-pair coverage and aggregate consistency, unique view indices/pair identities, 64-character lowercase SHA256 values, raw gradient-ratio weight, exact clipping, and equality across artifact, manifest, and checkpoint.
- Truncated legacy mean-of-norms preflight artifacts are explicitly rejected rather than treated as resumable A5 runs.
- `_seed_all` now executes only after checkpoint, log, manifest, config, and A5 preflight integrity validation succeeds. A failed resume therefore leaves Python, NumPy, torch CPU, and—by the same control-flow boundary—CUDA RNG untouched.

TDD evidence for this round:

- The new resume-schema and RNG tests first produced `9 failed, 1 passed` against `fda8c86`.
- Corrected pairing tests -> `26 passed`.
- Focused/adjacent Task 4 suite -> `99 passed in 2.98s`.
- All Stage 3-focused tests plus LR fusion -> `133 passed in 3.20s`.
- `compileall` and `git diff --check` passed before commit.

## Review fix round 3

Addressed the checkpoint-wide resume validation and complete coverage-provenance findings on top of `68e3c8c`:

- Added a mutation-free checkpoint payload validator shared by the normal Stage 3 loader and the resume preflight path. Before runtime construction or seeding, resume now validates serialized config, bridge architecture, the exact adapter set, every adapter key/shape, and all finite values.
- Added full format-v2 training-state prevalidation for step, optimizer/sigma state containers, Python/NumPy/CPU/CUDA RNG records, dropout RNG, pairing RNG, and the A5 fixed weight. The runtime restore path reuses the same validation helper without requiring an A5 weight for rank-only pairing RNG use.
- Built the validation-only Stage 3 adapter schema directly on the PyTorch `meta` device. This avoids both real parameter allocation and global CPU RNG consumption while inspecting a failed checkpoint.
- Extended the immutable calibration artifact with `view_count`, `target_patch_count`, and the complete valid directed target/source identity list for each batch. Each coverage record now carries its patch denominator.
- Resume validation now requires exact batch index, equality between coverage and `pair_count / target_patch_count`, equality between recorded coverage identities and the complete declared valid-pair set, consistent aggregate pair counts, and exact configured view count. Truncated identity or coverage lists, wrong batch indexes, and mismatched counts fail closed.

TDD evidence for this round:

- The first targeted RED run produced `10 failed, 2 passed`, exposing the missing pairing-state denominators/identities and the old preflight schema.
- New checkpoint-integrity tests cover missing pairing/dropout/CUDA RNG state, legacy training-state format, checkpoint config mismatch, adapter architecture mismatch, nonfinite adapter values, and verify unchanged Python, NumPy, torch CPU, and mocked CUDA RNG on every rejected resume.
- `tests/test_stage3_pairing.py` -> `38 passed`.
- All Stage 3-focused tests plus LR fusion, FullRRE, and UCPE camera contracts -> `174 passed in 3.84s`.
- `python -m compileall -q src scripts tests` and `git diff --check` -> PASS (only informational LF-to-CRLF warnings).

## Review fix round 4

Addressed the remaining restore-equivalence and independent pair-oracle blockers on top of `629270f`:

- Resume training-state validation now trial-parses Python state with a private `random.Random`, NumPy state with a private `RandomState`, CPU generator states with private `torch.Generator` instances, and CUDA noise/global states with private CUDA generators when available. These checks do not write any global RNG.
- CUDA state count and pinned state size are validated before restore. CPU-only test generators remain supported by the shared runtime restore helper through an explicit noise-device contract, while production resume prevalidation requires the CUDA format.
- Optimizer state is trial-loaded into a safe AdamW over the validation schema parameters. Empty or structurally incompatible states are rejected before model runtime construction.
- SigmaCycle state is checked for the fixed balanced sampling configuration, exact cycle length, permutation order, position/cycle integrity, and then trial-loaded with the real `SigmaCycle.load_state_dict` method on a scheduler-free private instance. This keeps prevalidation aligned with the actual restore path without importing the diffusion runtime.
- Preflight validation independently constructs the complete `V*(V-1)` directed non-self pair set for batch zero. It no longer accepts a jointly truncated `valid_pair_identities` plus coverage list as self-consistent evidence.
- Calibration independently derives and persists that same Cartesian pair oracle and fails closed if actual coverage identities are incomplete; it does not trust pairing-state metadata as the oracle.
- Coverage equality is now exact (`coverage == pair_count / target_patch_count`), so even a `1e-11` perturbation is rejected.

TDD evidence for this round:

- After correcting a test-only dependency setup, the targeted RED run produced six expected failures: jointly truncated identities/coverage, sub-tolerance coverage perturbation, one-byte torch RNG state, invalid NumPy bit generator, empty optimizer state, and empty SigmaCycle state.
- A separate CUDA-count RED test failed through the later log check until prevalidation enforced the device-state count.
- A calibration-oracle RED test proved the old path copied a deliberately truncated state identity list instead of deriving the full Cartesian set.
- `tests/test_stage3_pairing.py` -> `45 passed`.
- All Stage 3-focused tests plus LR fusion, FullRRE, and UCPE camera contracts -> `181 passed in 4.04s` before the final verification rerun.
- A direct meta-schema optimizer/SigmaCycle trial reported `meta True` and unchanged CPU RNG.

## Review fix round 5 (final)

Preserved the uncommitted round-4 work and completed the final schema-hardening
findings on top of `1ac65a1`:

- AdamW format-v2 resume prevalidation now requires exactly one canonical
  parameter group, the exact ordered parameter-id list and state-id set, exact
  group keys, exact hyperparameter types/values (including finite floats), and
  equality to the optimizer built from the declared Stage 3 configuration.
- Every AdamW parameter state now has the exact non-AMSGrad key set, a plain
  CPU `float32` scalar step equal to the checkpoint step, and plain finite CPU
  moment tensors with the exact parameter shape and dtype. String/mismatched
  learning rates, incomplete state, unexpected AMSGrad state, tensor
  subclasses, integer/wrong steps, nonfinite values, and wrong shapes fail
  before runtime construction or RNG restoration.
- SigmaCycle prevalidation now enforces exact scalar and tensor types, positive
  finite configuration values equal to the declared sampling config, the fixed
  balanced strategy, plain CPU generator/order tensors, an exact permutation,
  and strict position/cycle ranges. Bool, float, integer-type, or tensor-subclass
  masquerades are rejected before restore.
- Calibration coverage is independently validated both while constructing each
  preflight batch and immediately before opening the immutable artifact. Every
  batch must contain exactly `V*(V-1)` unique directed non-self rows, exactly the
  Cartesian identity set, exact `coverage == pair_count / target_patch_count`,
  the configured minimum coverage, and exact aggregate count/minimum fields.
  Invalid calibration leaves no artifact behind.

Final-round TDD evidence:

- The inherited focused file first passed (`55 passed`), establishing the
  starting state without discarding the partial changes.
- New exact-schema and pre-write coverage tests then produced the expected
  `12 failed, 55 deselected`; a second edge-case RED run produced `2 failed,
  67 deselected` for optimizer-step equality and malformed identity handling.
- After the minimal fixes, the pairing tests passed (`69 passed` as part of the
  final focused run), and the complete requested Stage 3/fusion/FullRRE/UCPE
  focused set passed: `211 passed in 4.51s`.
- `python -m compileall -q src scripts tests` and `git diff --check` passed;
  Git emitted only informational LF-to-CRLF warnings.

## Controller closeout

The final independent review identified two narrow restore defects after round 5:

- format-v2 training state is now mandatory only for A5 pairing supervision;
  legacy A0-A4 format-v1 checkpoints retain the documented optional dropout and
  CUDA RNG behavior;
- SigmaCycle `cycle` and `position` must exactly match
  `checkpoint_step * gradient_accumulation`, preventing a well-typed but
  trajectory-changing scheduler state from resuming.

Regression tests cover a real legacy A4 prevalidation path and an inconsistent
SigmaCycle state. Server verification passed `105` pairing/runner tests before
the final focused and repository-wide reruns.
