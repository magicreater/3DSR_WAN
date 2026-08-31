# PROJECT_SPEC.md

## 1. Goal

**RL3dSR** aims to adapt a pretrained Wan video diffusion model into a unified super-resolution framework that supports both:

### 3DSR

Input:

- multiple low-resolution images of the same static scene;
- camera intrinsics/extrinsics for every view.

Output:

- super-resolved multi-view images that preserve:
  - image fidelity;
  - cross-view appearance consistency;
  - geometric consistency.

### 4DSR

Input:

- a low-resolution single-view video;
- a camera pose for every frame.

Output:

- a super-resolved video that preserves:
  - spatial detail;
  - temporal consistency;
  - scene consistency under camera motion.

The base model should share as much architecture as practical between these tasks without incorrectly treating multi-view geometry as temporal motion.

---

## 2. Foundation Model

Use the pretrained **Wan** model as the primary backbone.

The adaptation strategy should be minimally invasive:

- reuse pretrained backbone and VAE weights;
- add task-specific conditioning through lightweight modules where possible;
- preserve compatibility with pretrained behavior;
- avoid training a new generative backbone from scratch.

The architecture should remain compatible with later RL/post-training.

---

## 3. Input Representation

Conceptually, both tasks provide an LR observation sequence:

```text
LR observations
+ camera information
→ Wan-based SR model
→ HR observations
```

However, the sequence semantics differ.

### 3DSR

```text
[x1, x2, ..., xV]
```

represents **V different camera views of one static scene**.

Each view has corresponding camera information:

```text
P1, P2, ..., PV
```

There is no physical temporal ordering requirement.

### 4DSR

```text
[x1, x2, ..., xT]
```

represents **T chronological frames from one video**.

Each frame has a corresponding camera pose:

```text
P1, P2, ..., PT
```

Camera motion therefore provides useful geometric information and should be incorporated through the same general pose-conditioning system.

### Canonical camera/data contract

Dataset adapters must convert source metadata into one shared contract before
degradation, batching, or pose conditioning:

- `K` is a per-observation `3 x 3` pinhole intrinsic matrix in pixel
  coordinates, with image `u` increasing right and `v` increasing down;
- `T_world_from_camera` is a per-observation `4 x 4` camera-to-world transform;
- the canonical camera frame follows OpenCV axes: `+X` right, `+Y` down, and
  `+Z` forward;
- source world coordinates, units, scale, and up direction are preserved by the
  data adapter unless a later, explicit normalization stage says otherwise;
- every sequence carries `sequence_kind`, which is `multiview` for 3DSR and
  `temporal` for 4DSR.

The detailed field, matrix, and adapter rules are defined in
[`camera_data_contract.md`](camera_data_contract.md). Raw dataset pose matrices
must never cross this boundary without an explicit source-to-canonical
conversion.

---

## 4. LR Conditioning

The model must explicitly condition generation on the low-resolution observations.

LR conditioning should be designed as a reusable component rather than embedded throughout the backbone.

Preferred properties:

- retains information from the original LR input;
- aligns naturally with Wan latent/token representations;
- supports both multi-view batches and temporal video batches;
- introduces minimal disturbance to pretrained features;
- can be ablated independently.

The exact injection mechanism may evolve based on experiments, but the external conditioning interface should remain stable.

---

## 5. Pose-Aware Conditioning

Camera pose is a core input of RL3dSR.

A shared pose-conditioning interface should support:

```text
3DSR: pose per view
4DSR: pose per frame
```

The adapter may encode information derived from:

- camera extrinsics;
- intrinsics when available;
- relative camera transformations;
- ray/plücker-style geometric representations if experimentally justified.

The implementation should avoid committing the entire architecture to one representation prematurely.

Pose features should enter the network through lightweight conditioning/adaptation modules rather than requiring a redesign of the Wan backbone.

Important requirements:

- consistent coordinate conventions;
- explicit pose normalization;
- correct association between each observation and its pose;
- support for variable view/frame counts where practical;
- easy pose-conditioning ablation.

---

## 6. Wan VAE Semantics

Wan's VAE contains temporal modeling/compression designed for video.

This temporal mechanism must **not automatically be applied to multi-view images as if different views were consecutive video frames**.

### 4DSR

For real video:

- preserve Wan's native temporal VAE behavior whenever practical;
- maintain the pretrained temporal inductive bias.

### 3DSR

Different camera views represent spatial viewpoint variation rather than physical time.

The 3DSR path should therefore avoid relying on temporal compression assumptions that imply temporal continuity between views.

Implementation choices may include adapting how multi-view observations are encoded or grouped, but must preserve compatibility with the shared downstream backbone whenever possible.

Any modification to this behavior must be validated against both 3DSR performance and future 4DSR compatibility.

---

## 7. Geometry Policy

The initial supervised pipeline does **not** use explicit reconstructed geometry.

The base input is limited to:

```text
LR observations + camera information
```

Do not require:

- point clouds;
- depth from VGGT;
- SLAM point maps;
- meshes;
- external 3D reconstruction.

This keeps training and inference simple and dataset-independent.

Explicit geometry is reserved for later experiments, primarily as an optional source of supervision or reward during RL/post-training.

---

## 8. Model Design Principle

Prefer a shared architecture of the form:

```text
LR input
   │
   ├── LR conditioning ─────────┐
   │                            │
camera pose                     │
   │                            │
   └── pose adapter ────────────┤
                                ▼
                       pretrained Wan
                                │
                                ▼
                           HR output
```

The exact internal injection points are experimental.

The important architectural invariant is:

> 3DSR and 4DSR should share the core SR model and conditioning abstractions while preserving their different view/time semantics.

---

## 9. Training Strategy

Development should proceed incrementally.

### Stage 1 — Baseline infrastructure

Establish:

- Wan loading;
- dataset interfaces;
- LR degradation pipeline;
- camera metadata pipeline;
- evaluation pipeline;
- reliable inference/smoke tests.

No major backbone training should be necessary here.

### Stage 2 — Conditioning integration

Implement and validate:

- LR conditioning;
- pose-aware conditioning;
- 3DSR-compatible sequence handling;
- architecture paths required to preserve future video support.

Initially verify structure using forward passes, shapes, parameter counts, and controlled ablations before expensive training.

### Stage 3 — Supervised 3DSR training

Train the adapted model on multi-view SR.

Evaluate:

- per-view reconstruction quality;
- perceptual quality;
- cross-view consistency;
- geometry-related consistency where measurable.

This stage establishes the primary 3DSR model.

### Stage 4 — 4DSR extension

Enable supervised training/evaluation on single-view video.

Reuse:

- LR conditioning;
- pose conditioning;
- shared Wan backbone.

Preserve the native temporal modeling required by video.

Evaluate spatial SR quality together with temporal consistency.

### Stage 5 — RL / post-training

After a strong supervised baseline exists, investigate RL or other post-training methods.

Potential signals may include:

- perceptual quality;
- multi-view consistency;
- temporal consistency;
- geometric consistency;
- optional reconstructed geometry from VGGT/SLAM or related systems.

Explicit 3D geometry should be introduced here only if it provides measurable benefit.

---

## 10. Evaluation

### 3DSR

At minimum evaluate image-level SR metrics such as:

- PSNR;
- SSIM;
- LPIPS.

Also include metrics or downstream evaluations capable of detecting multi-view inconsistency when feasible.

A model that independently sharpens each view but damages cross-view consistency is not considered successful.

### 4DSR

Evaluate:

- spatial reconstruction/perceptual quality;
- temporal consistency;
- flicker;
- stability under camera motion.

A 4DSR extension must not obtain sharper frames by significantly degrading temporal coherence.

---

## 11. Ablations

The implementation should make the following comparisons straightforward:

```text
Wan baseline
+ LR conditioning
+ LR conditioning + pose conditioning
```

Later:

```text
supervised model
vs.
RL/post-trained model
```

Geometry-based conditioning or rewards should be evaluated separately rather than becoming an implicit dependency of the baseline.

---

## 12. Success Criteria

The project is successful when one Wan-based framework can demonstrate:

1. meaningful SR improvement over LR input;
2. strong 3DSR performance on multi-view data;
3. measurable benefit from pose-aware conditioning;
4. preservation of cross-view consistency;
5. extension to single-view-video 4DSR without redesigning the entire model;
6. preservation of temporal coherence in 4DSR;
7. an architecture that can later accept RL/post-training without changing the base inference contract.

The research priority is not simply maximizing single-image SR metrics, but exploiting camera-aware structure while retaining Wan's useful pretrained generative and temporal priors.
