import struct

import pytest

from rl3dsr.data import DatasetFormatError
from rl3dsr.data.colmap import read_cameras_binary, read_images_binary


def test_binary_reader_handles_ids_and_skips_variable_tracks(mip_scene):
    cameras = read_cameras_binary(mip_scene.camera_file)
    images = read_images_binary(mip_scene.image_file)
    assert set(cameras) == {7, 42}
    assert cameras[7].params == (10.0, 12.0, 5.5, 3.5)
    assert len(images) == 10
    for raw in mip_scene.images:
        image = images[raw["id"]]
        assert image.name == raw["name"]
        assert image.camera_id == raw["camera_id"]
        assert image.qvec == raw["qvec"]
        assert image.tvec == raw["tvec"]
        assert not hasattr(image, "points2d")


@pytest.mark.parametrize("which,cut", [("camera", 4), ("camera", 40), ("image", 4), ("image", 74), ("image", -1)])
def test_truncated_binary_is_bounded_and_contextual(mip_scene, which, cut):
    path = mip_scene.camera_file if which == "camera" else mip_scene.image_file
    path.write_bytes(path.read_bytes()[:cut])
    reader = read_cameras_binary if which == "camera" else read_images_binary
    with pytest.raises(DatasetFormatError) as error:
        reader(path)
    assert str(path) in str(error.value)
    assert "offset" in str(error.value)


def test_unterminated_image_name_is_rejected(mip_scene):
    mip_scene.image_file.write_bytes(
        struct.pack("<Q", 1) + struct.pack("<i7di", 1, 1, 0, 0, 0, 0, 0, 0, 7)
        + b"no_null_terminator"
    )
    with pytest.raises(DatasetFormatError, match="name"):
        read_images_binary(mip_scene.image_file)


def test_huge_points2d_count_is_rejected_before_seek(mip_scene):
    mip_scene.image_file.write_bytes(
        struct.pack("<Q", 1) + struct.pack("<i7di", 1, 1, 0, 0, 0, 0, 0, 0, 7)
        + b"image.JPG\0" + struct.pack("<Q", 2**63)
    )
    with pytest.raises(DatasetFormatError, match="points2D"):
        read_images_binary(mip_scene.image_file)


@pytest.mark.parametrize("which", ["camera", "image"])
def test_duplicate_binary_ids_are_rejected(mip_scene, which):
    if which == "camera":
        mip_scene.cameras[1]["id"] = mip_scene.cameras[0]["id"]
        mip_scene.write_cameras()
        reader, path = read_cameras_binary, mip_scene.camera_file
    else:
        mip_scene.images[1]["id"] = mip_scene.images[0]["id"]
        mip_scene.write_images()
        reader, path = read_images_binary, mip_scene.image_file
    with pytest.raises(DatasetFormatError, match="duplicate"):
        reader(path)


def test_duplicate_registered_names_are_rejected(mip_scene):
    mip_scene.images[1]["name"] = mip_scene.images[0]["name"]
    mip_scene.write_images()
    with pytest.raises(DatasetFormatError, match="duplicate"):
        read_images_binary(mip_scene.image_file)


def test_unsupported_model_is_not_silently_treated_as_pinhole(mip_scene):
    mip_scene.cameras[0]["model"] = 4
    mip_scene.write_cameras()
    with pytest.raises(DatasetFormatError, match="PINHOLE"):
        read_cameras_binary(mip_scene.camera_file)


def test_real_simple_pinhole_record_reports_unsupported_model(mip_scene):
    # SIMPLE_PINHOLE has only three parameters, shorter than PINHOLE.
    mip_scene.camera_file.write_bytes(
        struct.pack("<QiiQQ3d", 1, 7, 0, 11, 7, 10.0, 5.5, 3.5)
    )
    with pytest.raises(DatasetFormatError, match="only PINHOLE"):
        read_cameras_binary(mip_scene.camera_file)


@pytest.mark.parametrize("which", ["camera", "image"])
def test_huge_record_count_is_bounded_by_file_size(mip_scene, which):
    reader, path = ((read_cameras_binary, mip_scene.camera_file) if which == "camera"
                    else (read_images_binary, mip_scene.image_file))
    path.write_bytes(struct.pack("<Q", 2**63) + b"\0" * 128)
    with pytest.raises(DatasetFormatError, match="count"):
        reader(path)


@pytest.mark.parametrize("which", ["camera", "image"])
def test_trailing_bytes_are_rejected(mip_scene, which):
    reader, path = ((read_cameras_binary, mip_scene.camera_file) if which == "camera"
                    else (read_images_binary, mip_scene.image_file))
    path.write_bytes(path.read_bytes() + b"unexpected")
    with pytest.raises(DatasetFormatError, match="trailing"):
        reader(path)
