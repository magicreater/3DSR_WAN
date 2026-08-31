import builtins
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin
import pytest

from rl3dsr import DatasetFormatError, MipNeRF360Adapter, SequenceKind, Split


def _all_observations(adapter):
    observations = adapter.index("train").observations + adapter.index("test").observations
    return sorted(observations, key=lambda item: item.frame_index)


def test_registration_sort_split_identity_and_camera_association(mip_scene):
    adapter = MipNeRF360Adapter(mip_scene.root)
    train, test = adapter.index(Split.TRAIN), adapter.index(Split.TEST)
    assert train.scene_id == "mip_scene"
    assert train.sequence_kind is test.sequence_kind is SequenceKind.MULTIVIEW
    assert train.split is Split.TRAIN and test.split is Split.TEST
    assert [item.frame_index for item in test.observations] == [0, 8]
    assert [item.observation_id for item in test.observations] == ["img_00.JPG", "img_08.JPG"]
    assert {x.observation_id for x in train.observations}.isdisjoint(
        x.observation_id for x in test.observations
    )
    observations = _all_observations(adapter)
    assert [item.observation_id for item in observations] == [f"img_{i:02d}.JPG" for i in range(10)]
    cameras = {camera["id"]: camera for camera in mip_scene.cameras}
    raw_images = {image["name"]: image for image in mip_scene.images}
    for item in observations:
        raw = raw_images[item.observation_id]
        camera = cameras[raw["camera_id"]]
        fx, fy, cx, cy = camera["params"]
        np.testing.assert_array_equal(item.K, [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        assert (item.width, item.height) == (camera["width"], camera["height"])
        assert item.rgb_path == mip_scene.root / "images" / item.observation_id
        # Known inverse of the fixture's Z rotation, independently of IDs.
        np.testing.assert_allclose(item.T_world_from_camera[:3, 3], [-2, item.frame_index, -3], atol=1e-6)
        assert item.K.dtype == item.T_world_from_camera.dtype == np.float32
        assert not item.K.flags.writeable and not item.T_world_from_camera.flags.writeable


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_resolution_levels_keep_identity_pose_and_use_actual_dimensions(mip_scene, factor):
    original = _all_observations(MipNeRF360Adapter(mip_scene.root))
    selected = _all_observations(MipNeRF360Adapter(mip_scene.root, image_factor=factor))
    directory = "images" if factor == 1 else f"images_{factor}"
    for source, target in zip(original, selected):
        assert source.observation_id == target.observation_id
        assert source.frame_index == target.frame_index
        assert target.rgb_path.parent.name == directory
        np.testing.assert_array_equal(target.T_world_from_camera, source.T_world_from_camera)
        expected = source.K.astype(np.float64).copy()
        expected[0] *= target.width / source.width
        expected[1] *= target.height / source.height
        np.testing.assert_allclose(target.K, expected, rtol=1e-6)
    if factor == 2:
        assert not np.allclose(selected[0].K[:2], original[0].K[:2] / 2)


def test_binary_permutation_does_not_change_split_or_alignment(mip_scene):
    before = _all_observations(MipNeRF360Adapter(mip_scene.root))
    mip_scene.images.reverse()
    mip_scene.write_images()
    after = _all_observations(MipNeRF360Adapter(mip_scene.root))
    for old, new in zip(before, after):
        assert (old.observation_id, old.frame_index) == (new.observation_id, new.frame_index)
        np.testing.assert_array_equal(old.K, new.K)
        np.testing.assert_array_equal(old.T_world_from_camera, new.T_world_from_camera)


def test_holdout_configuration_is_explicit(mip_scene):
    sequence = MipNeRF360Adapter(mip_scene.root, holdout_every=3).index("test")
    assert [item.frame_index for item in sequence.observations] == [0, 3, 6, 9]


def test_index_does_not_decode_pixels_scan_directories_or_access_geometry(mip_scene, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("index attempted pixel decoding or directory scanning")

    original_open = builtins.open
    original_path_open = Path.open

    def check_path(path):
        if isinstance(path, (str, Path)):
            assert Path(path).name not in {"points3D.bin", "poses_bounds.npy"}

    def checked_open(path, *args, **kwargs):
        check_path(path)
        return original_open(path, *args, **kwargs)

    def checked_path_open(path, *args, **kwargs):
        check_path(path)
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", checked_open)
    monkeypatch.setattr(Path, "open", checked_path_open)
    monkeypatch.setattr(Image.Image, "load", forbidden)
    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "load", forbidden)
    for name in ("glob", "rglob", "iterdir"):
        monkeypatch.setattr(Path, name, forbidden)
    assert len(MipNeRF360Adapter(mip_scene.root, image_factor=4).index("train").observations) == 8


@pytest.mark.parametrize("parameter,value", [
    ("image_factor", 0), ("image_factor", 3), ("image_factor", True),
    ("image_factor", 2.0), ("holdout_every", 1), ("holdout_every", True),
    ("holdout_every", 2.5),
])
def test_invalid_configuration_is_rejected(mip_scene, parameter, value):
    with pytest.raises(DatasetFormatError, match=parameter):
        MipNeRF360Adapter(mip_scene.root, **{parameter: value})


@pytest.mark.parametrize("split", [Split.VAL, "validation", "all"])
def test_unsupported_split_never_aliases_test(mip_scene, split):
    with pytest.raises(DatasetFormatError, match="split"):
        MipNeRF360Adapter(mip_scene.root).index(split)


def test_unknown_camera_reference_in_any_split_is_rejected(mip_scene):
    mip_scene.images[0]["camera_id"] = 999
    mip_scene.write_images()
    with pytest.raises(DatasetFormatError, match="camera") as error:
        MipNeRF360Adapter(mip_scene.root).index("test")
    assert "999" in str(error.value) and "img_09.JPG" in str(error.value)


def test_invalid_quaternion_has_image_and_scene_context(mip_scene):
    mip_scene.images[0]["qvec"] = (2, 0, 0, 0)
    mip_scene.write_images()
    with pytest.raises(DatasetFormatError, match="unit norm") as error:
        MipNeRF360Adapter(mip_scene.root).index("train")
    assert "scene='mip_scene'" in str(error.value)
    assert "img_09.JPG" in str(error.value)


@pytest.mark.parametrize("name", ["../outside.JPG", "/outside.JPG", "C:\\outside.JPG", "sub/../../outside.JPG"])
def test_registered_paths_cannot_escape_image_directory(mip_scene, name):
    mip_scene.images[0]["name"] = name
    mip_scene.write_images()
    with pytest.raises(DatasetFormatError, match="path"):
        MipNeRF360Adapter(mip_scene.root).index("train")


def test_different_names_resolving_to_same_rgb_are_rejected(mip_scene):
    target = next(image for image in mip_scene.images if image["name"] == "img_00.JPG")
    mip_scene.images[0]["name"] = "./img_00.JPG"
    mip_scene.images[0]["camera_id"] = target["camera_id"]
    mip_scene.write_images()
    with pytest.raises(DatasetFormatError, match="duplicate.*path"):
        MipNeRF360Adapter(mip_scene.root).index("train")


def test_symlink_cannot_escape_selected_directory(mip_scene, tmp_path):
    path = mip_scene.root / "images_4/img_00.JPG"
    outside = tmp_path / "outside.JPG"
    Image.new("RGB", (4, 2)).save(outside)
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(DatasetFormatError, match="outside"):
        MipNeRF360Adapter(mip_scene.root, image_factor=4).index("test")


def test_missing_resolution_does_not_fall_back(mip_scene):
    (mip_scene.root / "images_4").rename(mip_scene.root / "removed_images_4")
    with pytest.raises(DatasetFormatError, match="images_4"):
        MipNeRF360Adapter(mip_scene.root, image_factor=4).index("test")


def test_missing_registered_image_is_not_silently_skipped(mip_scene):
    (mip_scene.root / "images_4/img_00.JPG").unlink()
    with pytest.raises(DatasetFormatError, match="img_00.JPG"):
        MipNeRF360Adapter(mip_scene.root, image_factor=4).index("test")


def test_original_dimensions_must_match_colmap(mip_scene):
    Image.new("RGB", (2, 2)).save(mip_scene.root / "images/img_00.JPG")
    with pytest.raises(DatasetFormatError, match="dimensions"):
        MipNeRF360Adapter(mip_scene.root, image_factor=4).index("test")


@pytest.mark.parametrize("mode,format", [("L", "JPEG"), ("RGB", "PNG")])
def test_referenced_non_rgb_jpeg_is_rejected(mip_scene, mode, format):
    Image.new(mode, (4, 2)).save(mip_scene.root / "images_4/img_00.JPG", format=format)
    with pytest.raises(DatasetFormatError, match="RGB JPEG"):
        MipNeRF360Adapter(mip_scene.root, image_factor=4).index("test")


def test_decoder_returns_unmodified_pixels_and_detects_dimension_drift(mip_scene):
    adapter = MipNeRF360Adapter(mip_scene.root, image_factor=4)
    observation = adapter.index("test").observations[0]
    pixels = adapter.load_rgb(observation)
    with Image.open(observation.rgb_path) as image:
        np.testing.assert_array_equal(pixels, np.asarray(image))
    assert pixels.dtype == np.uint8 and pixels.flags.c_contiguous
    assert pixels.shape == (observation.height, observation.width, 3)
    Image.new("RGB", (2, 7)).save(observation.rgb_path)
    with pytest.raises(DatasetFormatError, match="dimensions"):
        adapter.load_rgb(observation)


def test_decoder_rejects_observation_from_different_resolution(mip_scene):
    observation = MipNeRF360Adapter(mip_scene.root).index("test").observations[0]
    with pytest.raises(DatasetFormatError, match="outside"):
        MipNeRF360Adapter(mip_scene.root, image_factor=4).load_rgb(observation)


def test_empty_selected_split_is_an_error(mip_scene):
    mip_scene.images = mip_scene.images[:1]
    mip_scene.write_images()
    with pytest.raises(DatasetFormatError, match="empty"):
        MipNeRF360Adapter(mip_scene.root).index("train")
