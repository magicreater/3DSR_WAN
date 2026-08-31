# AGENTS.md

## Project

This repository implements **RL3dSR**, a unified super-resolution framework built on a pretrained Wan video diffusion backbone.

The project targets:

- **3DSR**: multi-view low-resolution images with known camera poses.
- **4DSR**: single-view low-resolution video with a camera pose for each frame.

The detailed architecture and roadmap are defined in `PROJECT_SPEC.md`.

---

## Source of Truth

Before making non-trivial changes:

1. Read this file.
2. Read `PROJECT_SPEC.md`.
3. Inspect the relevant existing code and tests.
4. Prefer the current repository implementation over assumptions about upstream projects.

Priority when instructions conflict:

`explicit task > PROJECT_SPEC.md > AGENTS.md > existing conventions`

Do not silently change architectural decisions defined in `PROJECT_SPEC.md`.

---

## Working Principles

### Plan before modifying

For any task touching multiple modules, model interfaces, training behavior, or data flow:

- inspect the affected code first;
- identify the smallest coherent implementation;
- state the intended changes before editing;
- keep unrelated refactoring out of the task.

Prefer incremental changes that leave the repository runnable after each step.

### Preserve pretrained behavior

Wan is the pretrained foundation of this project.

When adding LR conditioning, pose conditioning, or 3D/4D functionality:

- preserve pretrained weights whenever possible;
- prefer adapters, projections, residual branches, zero-initialized modules, or other minimally invasive extensions;
- avoid rewriting backbone behavior unless there is strong evidence that it is necessary;
- ensure new conditioning can initially behave approximately as an identity/no-op.

Do not sacrifice Wan's video temporal modeling merely to simplify the 3DSR path.

### Keep 3D and 4D semantics distinct

Do not treat multi-view images as ordinary temporal video frames.

- In **3DSR**, the sequence dimension represents different camera views.
- In **4DSR**, the sequence dimension represents real temporal frames.

Shared modules are encouraged, but semantic differences must remain explicit in data loading, masking, conditioning, VAE usage, and evaluation.

### Camera pose is first-class conditioning

Pose-aware conditioning must support:

- per-view poses for 3DSR;
- per-frame poses for 4DSR.

Do not design pose conditioning around only one of these cases.

Pose representations and coordinate conventions must be documented and tested.

### No point-cloud dependency in the base pipeline

The base supervised model must not require:

- point clouds,
- VGGT features,
- SLAM geometry,
- reconstructed meshes,
- depth maps derived from external reconstruction systems.

Such geometry may be introduced later during RL/post-training experiments without redesigning the base inference interface.

---

## Implementation Rules

Keep modules focused and interfaces explicit.

Prefer:

- reusable conditioning modules;
- configuration-driven behavior;
- explicit tensor-shape documentation at important interfaces;
- assertions for ambiguous dimensions;
- deterministic evaluation paths;
- small targeted tests.

Avoid:

- hard-coded dataset assumptions;
- hidden reshaping of view/time dimensions;
- duplicated 3D and 4D implementations when a clean shared abstraction exists;
- large monolithic model wrappers;
- premature optimization;
- speculative features not required by the current stage.

When introducing a new module, clearly define:

- inputs and tensor shapes;
- outputs;
- coordinate/frame conventions if applicable;
- which parameters are pretrained, frozen, or trainable;
- expected behavior when the new conditioning is disabled.

---

## Validation

Every meaningful change must include the cheapest validation capable of catching structural errors.

At minimum check, as applicable:

- imports and configuration loading;
- tensor shapes;
- forward pass;
- gradient flow for intended trainable modules;
- frozen parameters remain frozen;
- checkpoint save/load compatibility;
- conditioning-off behavior;
- multi-view batching;
- temporal batching;
- camera-pose alignment;
- VAE encode/decode shape consistency.

Before claiming a task is complete, run the relevant tests or smoke checks and report exactly what was verified.

Do not claim correctness solely from code inspection.

---

## Experiments

Keep experimental decisions out of core model code whenever possible.

Experiment-specific behavior should live in configuration.

For each training experiment, preserve enough information to reproduce:

- configuration;
- checkpoint;
- dataset/split;
- random seed when relevant;
- major metrics.

Do not modify evaluation definitions merely to improve reported results.

---

## Documentation Discipline

Update `PROJECT_SPEC.md` only when a project-level architectural decision changes.

Do not turn it into a development log.

Temporary findings, failed experiments, implementation plans, and detailed task notes should live elsewhere.

When code behavior and documentation disagree, resolve the inconsistency as part of the task.

---

## Scope Control

Implement only what is needed for the current milestone.

The following are intentionally deferred unless explicitly requested:

- point-cloud conditioning;
- VGGT/SLAM integration;
- geometry-based RL rewards;
- large backbone redesigns;
- unrelated generative tasks.

Favor a correct, testable minimal implementation over a broad speculative one.