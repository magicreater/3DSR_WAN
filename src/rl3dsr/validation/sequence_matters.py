"""Deterministic, non-overwriting Blender staging for SequenceMatters."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_white(source: Path, target: Path, size: tuple[int, int]) -> None:
    with Image.open(source) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        value = Image.alpha_composite(background, rgba).convert("RGB")
        if value.size != size:
            value = value.resize(size, Image.Resampling.LANCZOS)
        value.save(target)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _transform_sha256(frames: list[dict]) -> str:
    payload = [frame.get("transform_matrix") for frame in frames]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_deterministic_ply(path: Path, *, seed: int = 2201, count: int = 100_000) -> str:
    """Write the fixed random Blender initialization used by every arm of a scene."""

    if path.exists():
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    vertices = np.empty(
        count,
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    positions = generator.uniform(-1.3, 1.3, size=(count, 3)).astype(np.float32)
    colors = generator.integers(0, 256, size=(count, 3), dtype=np.uint8)
    vertices["x"], vertices["y"], vertices["z"] = positions.T
    vertices["nx"] = vertices["ny"] = vertices["nz"] = 0
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
    temporary.replace(path)
    return sha256_file(path)


def stage_blender_dataset(
    target: Path,
    *,
    source_scene: Path,
    train_images: dict[int, Path],
    train_lr_images: dict[int, Path],
    train_indices: tuple[int, ...],
    shared_ply: Path,
    image_size: int = 256,
    lr_size: int = 64,
) -> dict:
    """Create one isolated 3DGS arm and refuse all overwrite attempts."""

    if target.exists():
        raise FileExistsError(f"refusing to overwrite staged dataset: {target}")
    if set(train_images) != set(train_indices) or set(train_lr_images) != set(train_indices):
        raise ValueError("training image maps must exactly match train_indices")
    train_payload = json.loads((source_scene / "transforms_train.json").read_text(encoding="utf-8"))
    test_payload = json.loads((source_scene / "transforms_test.json").read_text(encoding="utf-8"))
    frames = train_payload.get("frames", [])
    if len(frames) <= max(train_indices):
        raise ValueError("training transform indices are outside the source scene")

    train_dir = target / "train"
    lr_dir = target / "train_lr"
    test_dir = target / "test"
    train_dir.mkdir(parents=True)
    lr_dir.mkdir()
    test_dir.mkdir()
    staged_train = []
    train_names = []
    for index in train_indices:
        frame = dict(frames[index])
        stem = Path(str(frame["file_path"])).name
        name = stem + ".png"
        train_names.append(name)
        shutil.copy2(train_images[index], train_dir / name)
        shutil.copy2(train_lr_images[index], lr_dir / name)
        frame["file_path"] = f"./train/{stem}"
        staged_train.append(frame)

    staged_test = []
    test_names = []
    for frame_value in test_payload.get("frames", []):
        frame = dict(frame_value)
        stem = Path(str(frame["file_path"])).name
        name = stem + ".png"
        source = source_scene / "test" / name
        if not source.is_file():
            raise FileNotFoundError(f"missing source test image: {source}")
        _rgb_white(source, test_dir / name, (image_size, image_size))
        frame["file_path"] = f"./test/{stem}"
        staged_test.append(frame)
        test_names.append(name)

    _write_json(
        target / "transforms_train.json",
        {**{key: value for key, value in train_payload.items() if key != "frames"}, "frames": staged_train},
    )
    _write_json(
        target / "transforms_test.json",
        {**{key: value for key, value in test_payload.items() if key != "frames"}, "frames": staged_test},
    )
    shutil.copy2(shared_ply, target / "points3d.ply")
    validation = validate_staged_blender(
        target,
        train_names=tuple(train_names),
        test_names=tuple(test_names),
        image_size=image_size,
        lr_size=lr_size,
        expected_ply_sha256=sha256_file(shared_ply),
    )
    staged_train_payload = json.loads(
        (target / "transforms_train.json").read_text(encoding="utf-8")
    )
    staged_test_payload = json.loads(
        (target / "transforms_test.json").read_text(encoding="utf-8")
    )
    train_transform_sha256 = _transform_sha256(staged_train)
    test_transform_sha256 = _transform_sha256(staged_test)
    if _transform_sha256(staged_train_payload["frames"]) != train_transform_sha256:
        raise RuntimeError("staged training camera matrices changed")
    if _transform_sha256(staged_test_payload["frames"]) != test_transform_sha256:
        raise RuntimeError("staged test camera matrices changed")
    manifest = {
        "schema_version": 1,
        "source_scene": str(source_scene.resolve()),
        "train_indices": list(train_indices),
        "train_names": train_names,
        "test_count": len(test_names),
        "points3d_sha256": validation["points3d_sha256"],
        "transforms_train_sha256": sha256_file(target / "transforms_train.json"),
        "transforms_test_sha256": sha256_file(target / "transforms_test.json"),
        "train_transform_matrices_sha256": train_transform_sha256,
        "test_transform_matrices_sha256": test_transform_sha256,
        "input_image_sha256": {
            str(index): sha256_file(train_images[index]) for index in train_indices
        },
        "input_lr_sha256": {
            str(index): sha256_file(train_lr_images[index]) for index in train_indices
        },
        "staged_image_sha256": {
            str(index): sha256_file(target / "train" / train_names[position])
            for position, index in enumerate(train_indices)
        },
        "staged_lr_sha256": {
            str(index): sha256_file(target / "train_lr" / train_names[position])
            for position, index in enumerate(train_indices)
        },
        "validation": validation,
    }
    if manifest["input_image_sha256"] != manifest["staged_image_sha256"]:
        raise RuntimeError("staged SR image hashes differ from frozen inputs")
    if manifest["input_lr_sha256"] != manifest["staged_lr_sha256"]:
        raise RuntimeError("staged LR image hashes differ from frozen inputs")
    _write_json(target / "staging_manifest.json", manifest)
    return manifest


def validate_staged_blender(
    root: Path,
    *,
    train_names: tuple[str, ...],
    test_names: tuple[str, ...],
    image_size: int,
    lr_size: int,
    expected_ply_sha256: str,
) -> dict:
    actual_train = tuple(sorted(path.name for path in (root / "train").glob("*.png")))
    actual_lr = tuple(sorted(path.name for path in (root / "train_lr").glob("*.png")))
    actual_test = tuple(sorted(path.name for path in (root / "test").glob("*.png")))
    if actual_train != tuple(sorted(train_names)) or actual_lr != tuple(sorted(train_names)):
        raise RuntimeError("staged train/train_lr names do not match the four-view protocol")
    if actual_test != tuple(sorted(test_names)):
        raise RuntimeError("staged test names do not match transforms_test")
    for name in actual_train:
        with Image.open(root / "train" / name) as image:
            if image.size != (image_size, image_size):
                raise RuntimeError(f"invalid staged SR size: {name} {image.size}")
        with Image.open(root / "train_lr" / name) as image:
            if image.size != (lr_size, lr_size):
                raise RuntimeError(f"invalid staged LR size: {name} {image.size}")
    for name in actual_test:
        with Image.open(root / "test" / name) as image:
            if image.size != (image_size, image_size):
                raise RuntimeError(f"invalid staged test size: {name} {image.size}")
    train_payload = json.loads((root / "transforms_train.json").read_text(encoding="utf-8"))
    test_payload = json.loads((root / "transforms_test.json").read_text(encoding="utf-8"))
    if len(train_payload.get("frames", ())) != len(train_names):
        raise RuntimeError("transforms_train does not contain exactly four frames")
    if len(test_payload.get("frames", ())) != len(test_names):
        raise RuntimeError("transforms_test frame count changed")
    ply_sha256 = sha256_file(root / "points3d.ply")
    if ply_sha256 != expected_ply_sha256:
        raise RuntimeError("shared points3d.ply hash mismatch")
    return {
        "train_count": len(actual_train),
        "test_count": len(actual_test),
        "image_size": image_size,
        "lr_size": lr_size,
        "points3d_sha256": ply_sha256,
    }
