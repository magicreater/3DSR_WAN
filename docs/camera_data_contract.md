# Canonical Camera and Data Contract

This document defines the CPU-side metadata boundary shared by dataset
adapters, degradation, batching, and pose conditioning. Dataset-specific camera
conventions must be converted at this boundary.

## Sequence semantics

A sequence has one explicit semantic kind:

- `multiview`: different views of one static scene, used by 3DSR;
- `temporal`: chronologically ordered video frames, reserved for 4DSR.

`NeRFSyntheticAdapter` always emits `multiview`. Its observation order is the
order of `frames[]` in the selected transforms JSON. That order does not imply
physical time.

Each observation contains:

- a stable `observation_id` and zero-based `frame_index`;
- an absolute RGB path and explicit pixel `width` and `height`;
- `K`, a read-only `float32[3,3]` intrinsic matrix;
- `T_world_from_camera`, a read-only `float32[4,4]` extrinsic matrix.

Image pixels, LR observations, depth, normals, masks, and reconstructed geometry
are not metadata-contract fields.

## Intrinsic convention

Pixel coordinates use `u` increasing right and `v` increasing down. For NeRF
Synthetic, the horizontal field of view in `camera_angle_x` and the actual PNG
header dimensions define

```text
f = 0.5 * width / tan(0.5 * camera_angle_x)

K = [ f  0  width/2  ]
    [ 0  f  height/2 ]
    [ 0  0     1     ]
```

The source dataset is modeled as a centered, square-pixel pinhole camera with
no skew or distortion. No resize or crop is performed by this adapter; a later
operation that changes image geometry must update `K` explicitly.

## Extrinsic convention

`T_world_from_camera` is camera-to-world:

```text
p_world_h = T_world_from_camera @ p_camera_h
```

The canonical local camera axes are OpenCV axes:

- `+X`: image right;
- `+Y`: image down;
- `+Z`: camera forward.

They form a right-handed coordinate system. The world coordinate system remains
the source dataset's right-handed world frame. Adapters do not center camera
positions, normalize scale, change units, or force a common world up-axis.

NeRF Synthetic `transform_matrix` values are Blender/OpenGL camera-to-world
transforms, whose local `+Y` points up and `+Z` points backward. Convert them by
changing only the camera basis:

```text
T_world_from_camera = T_source_c2w @ diag(1, -1, -1, 1)
```

This preserves the source world frame and camera origin. The COLMAP adapter
inverts COLMAP's world-to-camera transform and retains its OpenCV camera axes;
it does not reuse the NeRF Synthetic axis flip.

All canonical transforms must be finite rigid homogeneous transforms. The last
row is `[0,0,0,1]`; the rotation is orthonormal with determinant `+1`, checked
with absolute tolerance `1e-4`.

## NeRF Synthetic adapter boundary

For one scene and one of `train`, `val`, or `test`, only
`transforms_<split>.json` defines the observation set. The adapter:

1. preserves `frames[]` order and pose-image association;
2. resolves each `frames[].file_path` inside the scene root, adding `.png` only
   when the JSON path has no suffix;
3. reads only referenced PNG headers during indexing and accepts RGB/RGBA modes;
4. never scans image directories, so unreferenced depth and normal PNGs are not
   samples;
5. decodes pixels only through `load_rgb`, compositing RGBA deterministically
   over white by default or black when configured;
6. returns contiguous `uint8[H,W,3]` pixels without resizing, normalization,
   random background, or degradation.

Absolute paths, path traversal, non-PNG references, duplicate observations,
missing files, grayscale references, malformed matrices, and invalid camera
angles are dataset format errors. Errors include scene, split, frame, and path
context where applicable.

## Mip-NeRF 360 adapter boundary

`MipNeRF360Adapter(scene_root, image_factor=1, holdout_every=8)` implements the
same `index(split)` / `load_rgb(observation)` interface. It always returns
`SequenceKind.MULTIVIEW`, with the existing immutable metadata fields.

### Registration, identity, and splits

Only `sparse/0/cameras.bin` and `sparse/0/images.bin` define the camera/observation
set. The adapter supports `PINHOLE` cameras (`fx, fy, cx, cy`) and associates each
image with its camera ID; camera IDs need not be contiguous or start at one.
Other camera models fail explicitly instead of dropping their distortion terms.

Images are sorted by their exact COLMAP names using case-sensitive lexical order,
not by image ID or binary record order. `observation_id` is the COLMAP image name,
independent of resolution. `frame_index` is its zero-based position in this full
sorted list **before** split selection, so a split may contain non-contiguous
indices. With default `holdout_every=8`, positions `0, 8, 16, ...` are test and all
others are train. Holdout must be an integer >= 2; changing it changes the
evaluation view set. `val` and empty selected splits raise `DatasetFormatError`;
val is never silently aliased to test.

This is a within-scene view split, not a cross-scene SR evaluation protocol.
Scene-level training/validation policy remains the responsibility of a later
dataset composition layer.

All registrations are validated before subsetting, including duplicate IDs,
names, and resolved paths across train/test. Extra files in image directories
are ignored; file lists are never globbed, zipped, or used to infer camera order.
Paths must resolve inside their specified image directory. Original and selected
images must be RGB JPEGs; missing paths and incorrect formats/modes fail instead
of being skipped or automatically converted.

### COLMAP pose conversion

COLMAP stores a scalar-first quaternion `(qw, qx, qy, qz)` and translation `t`,
representing `p_camera = R @ p_world + t`. `convert_colmap_w2c(qvec, tvec)` returns
the float64 matrix

```text
T_world_from_camera = [ R.T   -R.T @ t ]
                      [ 0 0 0     1   ]
```

Quaternions must be finite and unit length within absolute tolerance `1e-4`;
only near-unit numerical deviations are normalized. Zero or materially
non-unit quaternions are errors. The metadata constructor performs the final
read-only float32 conversion. No Y/Z flip, world rotation, recentering, PCA, or
scene scaling is applied.

The reader uses the little-endian COLMAP binary format. Variable points2D
sections are length-checked against the file size and skipped without decoding
or retaining their geometric payload. Truncated records, invalid counts,
unterminated/overlong names (over 4095 UTF-8 bytes), and trailing bytes are errors.
Neither `points3D.bin` nor `poses_bounds.npy` is opened.

### Resolution and intrinsics

`image_factor` selects an existing directory: `1 -> images`, `2 -> images_2`,
`4 -> images_4`, `8 -> images_8`. Other factors are unsupported. Missing
directories do not trigger generation or fallback. Files are matched by exact
COLMAP image name across these directories.

Only image headers are read during indexing. The original JPEG's dimensions
must match those in its COLMAP camera record. For the selected JPEG, use its
actual header dimensions rather than assuming an exact integer downsampling:

```text
sx = target_width / source_width
sy = target_height / source_height
K_target = diag(sx, sy, 1) @ K_source
```

`scale_intrinsics(K, *, source_width, source_height, target_width, target_height)`
returns a new float64 matrix and never mutates its input. This scales each row
of pixel intrinsics independently, including the principal point. It expresses
dimension scaling under this contract's pixel coordinates, without crop offsets
or an additional half-pixel translation. Odd-size rounding can make `sx != sy`.

`load_rgb()` checks that the observation belongs to the selected image directory
and still has the indexed dimensions, then returns contiguous `uint8[H,W,3]`.
It does not resize, normalize, or apply EXIF orientation. Resolution selection
is not an HR-to-LR degradation pipeline and does not produce paired SR samples.

### Example

```python
from rl3dsr import MipNeRF360Adapter, Split

adapter = MipNeRF360Adapter("/path/to/360_v2/counter", image_factor=4)
sequence = adapter.index(Split.TEST)  # Metadata and JPEG headers only.
observation = sequence.observations[0]
rgb = adapter.load_rgb(observation)  # Explicit pixel decoding.
```

The module requires only the existing NumPy/Pillow dependencies and the Python
standard library, without PyCOLMAP, Torch, or GPU initialization.

## References

- The original NeRF Blender loader reads only JSON frame paths, treats
  `transform_matrix` as camera-to-world, and derives focal length from
  `camera_angle_x`: <https://github.com/bmild/nerf/blob/master/load_blender.py>
- Nerfstudio documents the OpenGL/OpenCV Y/Z axis difference and performs an
  explicit COLMAP conversion:
  <https://github.com/nerfstudio-project/nerfstudio/blob/50e0e3c70c775e89333256213363badbf074f29d/docs/quickstart/data_conventions.md>
  and
  <https://github.com/nerfstudio-project/nerfstudio/blob/50e0e3c70c775e89333256213363badbf074f29d/nerfstudio/data/dataparsers/colmap_dataparser.py>.
- MultiNeRF uses filename sorting and view-level holdout in its LLFF loader;
  its config defaults to alphabetical ordering and holdout 8:
  [dataset loader](https://github.com/google-research/multinerf/blob/main/internal/datasets.py),
  [config defaults](https://github.com/google-research/multinerf/blob/main/internal/configs.py).
  RL3dSR does not adopt MultiNeRF's OpenGL axis flip or PCA normalization.
- The COLMAP binary layout and quaternion ordering were checked against
  [Nerfstudio's COLMAP reader](https://github.com/nerfstudio-project/nerfstudio/blob/50e0e3c70c775e89333256213363badbf074f29d/nerfstudio/data/utils/colmap_parsing_utils.py).
  RL3dSR implements only the metadata subset and does not vendor its geometry
  reader or import its framework dependencies.
