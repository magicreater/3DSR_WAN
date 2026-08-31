from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rl3dsr import (
    AlphaBackground,
    DatasetFormatError,
    NeRFSyntheticAdapter,
    SequenceKind,
    Split,
)


IDENTITY_C2W = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

ROTATED_TRANSLATED_C2W = [
    [0.0, -1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 2.0],
    [0.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 0.0, 1.0],
]


@pytest.fixture
def scene_root(tmp_path: Path) -> Path:
    scene = tmp_path / "lego"
    image_dir = scene / "test"
    image_dir.mkdir(parents=True)

    Image.new("RGB", (4, 2), (10, 20, 30)).save(image_dir / "r_0.png")
    Image.new("RGBA", (4, 2), (255, 0, 0, 128)).save(image_dir / "r_1.png")

    # These files deliberately look like dataset assets but are not JSON frames.
    Image.new("I;16", (4, 2), 1000).save(image_dir / "r_0_depth_0001.png")
    Image.new("RGB", (4, 2), (127, 127, 255)).save(
        image_dir / "r_0_normal_0001.png"
    )

    _write_manifest(
        scene,
        {
            "camera_angle_x": math.pi / 2.0,
            "frames": [
                {"file_path": "./test/r_0", "transform_matrix": IDENTITY_C2W},
                {
                    "file_path": "./test/r_1.png",
                    "transform_matrix": ROTATED_TRANSLATED_C2W,
                },
            ],
        },
    )
    return scene


def test_indexes_only_manifest_rgb_with_canonical_camera_contract(
    scene_root: Path,
) -> None:
    sequence = NeRFSyntheticAdapter(scene_root).index(Split.TEST)

    assert sequence.scene_id == "lego"
    assert sequence.split is Split.TEST
    assert sequence.sequence_kind is SequenceKind.MULTIVIEW
    assert [item.observation_id for item in sequence.observations] == [
        "test/r_0",
        "test/r_1.png",
    ]
    assert [item.frame_index for item in sequence.observations] == [0, 1]
    assert len(sequence.observations) == 2
    assert all(item.rgb_path.is_absolute() for item in sequence.observations)
    assert all("depth" not in item.rgb_path.name for item in sequence.observations)
    assert all("normal" not in item.rgb_path.name for item in sequence.observations)

    first, second = sequence.observations
    expected_k = np.array(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(first.K, expected_k, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(
        first.T_world_from_camera,
        np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        second.T_world_from_camera[:3, 3],
        [1.0, 2.0, 3.0],
        atol=1e-6,
        rtol=0.0,
    )
    assert first.K.dtype == np.float32
    assert first.T_world_from_camera.dtype == np.float32
    assert not first.K.flags.writeable
    assert not first.T_world_from_camera.flags.writeable
    with pytest.raises(ValueError):
        first.K[0, 0] = 10.0


def test_load_rgb_is_contiguous_and_composites_rgba_deterministically(
    scene_root: Path,
) -> None:
    white_adapter = NeRFSyntheticAdapter(scene_root)
    white_sequence = white_adapter.index("test")

    rgb = white_adapter.load_rgb(white_sequence.observations[0])
    rgba_over_white = white_adapter.load_rgb(white_sequence.observations[1])

    assert rgb.shape == (2, 4, 3)
    assert rgb.dtype == np.uint8
    assert rgb.flags.c_contiguous
    np.testing.assert_array_equal(rgb[0, 0], [10, 20, 30])
    np.testing.assert_array_equal(rgba_over_white[0, 0], [255, 127, 127])

    black_adapter = NeRFSyntheticAdapter(
        scene_root,
        alpha_background=AlphaBackground.BLACK,
    )
    black_sequence = black_adapter.index("test")
    rgba_over_black = black_adapter.load_rgb(black_sequence.observations[1])
    np.testing.assert_array_equal(rgba_over_black[0, 0], [128, 0, 0])


@pytest.mark.parametrize(
    ("mutate_manifest", "expected"),
    [
        (lambda manifest: manifest.pop("camera_angle_x"), "missing camera_angle_x"),
        (lambda manifest: manifest.update(camera_angle_x=0.0), "camera_angle_x"),
        (lambda manifest: manifest.pop("frames"), "missing frames"),
        (
            lambda manifest: manifest["frames"][0].update(
                transform_matrix=[[1.0, 0.0], [0.0, 1.0]]
            ),
            "shape (4, 4)",
        ),
        (
            lambda manifest: manifest["frames"][0].update(
                transform_matrix=[
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "orthonormal",
        ),
    ],
)
def test_rejects_invalid_manifest_and_camera_metadata(
    scene_root: Path,
    mutate_manifest,
    expected: str,
) -> None:
    manifest = _read_manifest(scene_root)
    mutate_manifest(manifest)
    _write_manifest(scene_root, manifest)

    with pytest.raises(DatasetFormatError) as error:
        NeRFSyntheticAdapter(scene_root).index("test")

    message = str(error.value)
    assert "scene='lego'" in message
    assert "split='test'" in message
    assert expected in message


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("../outside", "stay within the scene root"),
        ("./test/missing", "does not exist"),
        ("./test/r_0.jpg", "must be a PNG"),
        ("C:\\outside\\r_0", "stay within the scene root"),
    ],
)
def test_rejects_unsafe_or_missing_rgb_references(
    scene_root: Path,
    file_path: str,
    expected: str,
) -> None:
    manifest = _read_manifest(scene_root)
    manifest["frames"][0]["file_path"] = file_path
    _write_manifest(scene_root, manifest)

    with pytest.raises(DatasetFormatError) as error:
        NeRFSyntheticAdapter(scene_root).index("test")

    message = str(error.value)
    assert "scene='lego'" in message
    assert "split='test'" in message
    assert "frame=0" in message
    assert expected in message


def test_rejects_referenced_grayscale_image(scene_root: Path) -> None:
    manifest = _read_manifest(scene_root)
    manifest["frames"][0]["file_path"] = "./test/r_0_depth_0001"
    _write_manifest(scene_root, manifest)

    with pytest.raises(DatasetFormatError, match="RGB or RGBA") as error:
        NeRFSyntheticAdapter(scene_root).index("test")

    assert "frame=0" in str(error.value)
    assert "r_0_depth_0001.png" in str(error.value)


def test_rejects_duplicate_frame_paths(scene_root: Path) -> None:
    manifest = _read_manifest(scene_root)
    manifest["frames"][1]["file_path"] = "./test/r_0.png"
    _write_manifest(scene_root, manifest)

    with pytest.raises(DatasetFormatError, match="duplicate referenced RGB path"):
        NeRFSyntheticAdapter(scene_root).index("test")


def test_rejects_corrupt_manifest_with_context(scene_root: Path) -> None:
    (scene_root / "transforms_test.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="failed to read manifest") as error:
        NeRFSyntheticAdapter(scene_root).index("test")

    assert "scene='lego'" in str(error.value)
    assert "split='test'" in str(error.value)


def test_rejects_unsupported_split(scene_root: Path) -> None:
    with pytest.raises(DatasetFormatError, match="unsupported split"):
        NeRFSyntheticAdapter(scene_root).index("validation")


def _read_manifest(scene_root: Path) -> dict:
    return json.loads((scene_root / "transforms_test.json").read_text(encoding="utf-8"))


def _write_manifest(scene_root: Path, manifest: dict) -> None:
    (scene_root / "transforms_test.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

