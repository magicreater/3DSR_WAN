from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from rl3dsr.validation.sequence_matters import (
    sha256_file,
    stage_blender_dataset,
    write_deterministic_ply,
)


def _image(path, size, color, mode="RGB"):
    Image.new(mode, size, color).save(path)


def test_sequence_matters_staging_preserves_subset_and_shared_inputs(tmp_path):
    source = tmp_path / "source"
    (source / "test").mkdir(parents=True)
    frames = []
    for index in range(5):
        transform = np.eye(4).tolist()
        transform[0][3] = index
        frames.append({"file_path": f"./train/r_{index}", "transform_matrix": transform})
    (source / "transforms_train.json").write_text(
        json.dumps({"camera_angle_x": 0.7, "frames": frames}), encoding="utf-8"
    )
    test_frames = []
    for index in range(3):
        test_frames.append({"file_path": f"./test/r_{index}", "transform_matrix": np.eye(4).tolist()})
        _image(source / "test" / f"r_{index}.png", (32, 32), (10, 20, 30, 128), "RGBA")
    (source / "transforms_test.json").write_text(
        json.dumps({"camera_angle_x": 0.7, "frames": test_frames}), encoding="utf-8"
    )

    indices = (0, 1, 3, 4)
    sr, lr = {}, {}
    for index in indices:
        sr[index] = tmp_path / f"sr_{index}.png"
        lr[index] = tmp_path / f"lr_{index}.png"
        _image(sr[index], (256, 256), (index, 2, 3))
        _image(lr[index], (64, 64), (index, 2, 3))
    shared = tmp_path / "shared.ply"
    shared_sha = write_deterministic_ply(shared, count=16)
    target = tmp_path / "staged"
    manifest = stage_blender_dataset(
        target,
        source_scene=source,
        train_images=sr,
        train_lr_images=lr,
        train_indices=indices,
        shared_ply=shared,
    )
    staged = json.loads((target / "transforms_train.json").read_text(encoding="utf-8"))
    assert [frame["transform_matrix"] for frame in staged["frames"]] == [frames[index]["transform_matrix"] for index in indices]
    assert all(frame["file_path"].startswith("./train/") for frame in staged["frames"])
    assert manifest["test_count"] == 3
    assert manifest["points3d_sha256"] == shared_sha == sha256_file(target / "points3d.ply")
    assert manifest["input_image_sha256"] == manifest["staged_image_sha256"]
    assert manifest["input_lr_sha256"] == manifest["staged_lr_sha256"]
    assert manifest["train_transform_matrices_sha256"]
    assert manifest["test_transform_matrices_sha256"]
    assert len(tuple((target / "train").glob("*.png"))) == 4
    assert len(tuple((target / "train_lr").glob("*.png"))) == 4
    assert len(tuple((target / "test").glob("*.png"))) == 3
    with Image.open(target / "test" / "r_0.png") as image:
        assert image.size == (256, 256)
        assert image.mode == "RGB"
    with pytest.raises(FileExistsError):
        stage_blender_dataset(
            target,
            source_scene=source,
            train_images=sr,
            train_lr_images=lr,
            train_indices=indices,
            shared_ply=shared,
        )
