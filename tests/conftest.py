"""Tiny synthetic COLMAP model writers; never use the real datasets."""

from dataclasses import dataclass
from pathlib import Path
import struct

from PIL import Image
import pytest


@dataclass
class MipFixture:
    root: Path
    cameras: list[dict]
    images: list[dict]

    @property
    def camera_file(self):
        return self.root / "sparse/0/cameras.bin"

    @property
    def image_file(self):
        return self.root / "sparse/0/images.bin"

    def write_cameras(self):
        with self.camera_file.open("wb") as stream:
            stream.write(struct.pack("<Q", len(self.cameras)))
            for camera in self.cameras:
                stream.write(struct.pack(
                    "<iiQQ", camera["id"], camera["model"], camera["width"], camera["height"]
                ))
                stream.write(struct.pack("<4d", *camera["params"]))

    def write_images(self):
        with self.image_file.open("wb") as stream:
            stream.write(struct.pack("<Q", len(self.images)))
            for image in self.images:
                stream.write(struct.pack(
                    "<i7di", image["id"], *image["qvec"], *image["tvec"], image["camera_id"]
                ))
                stream.write(image["name"].encode("utf-8") + b"\0")
                stream.write(struct.pack("<Q", len(image["points2d"])))
                for point in image["points2d"]:
                    stream.write(struct.pack("<ddq", *point))


@pytest.fixture
def mip_scene(tmp_path):
    root = tmp_path / "mip_scene"
    (root / "sparse/0").mkdir(parents=True)
    cameras = [
        dict(id=42, model=1, width=13, height=9, params=(20.0, 24.0, 6.5, 4.5)),
        dict(id=7, model=1, width=11, height=7, params=(10.0, 12.0, 5.5, 3.5)),
    ]
    images = []
    # Storage order and IDs intentionally do not match filename order.
    for i in [9, 1, 6, 0, 8, 2, 4, 7, 3, 5]:
        camera = cameras[i % 2]
        name = f"img_{i:02d}.JPG"
        images.append(dict(
            id=100 - 3 * i, camera_id=camera["id"], name=name,
            qvec=(2**-0.5, 0.0, 0.0, 2**-0.5),
            tvec=(float(i), 2.0, 3.0),
            points2d=[(0.25, 1.5, -1)] * (i % 3),
        ))
        for factor in [1, 2, 4, 8]:
            directory = root / ("images" if factor == 1 else f"images_{factor}")
            directory.mkdir(exist_ok=True)
            width = max(1, (camera["width"] + factor - 1) // factor)
            height = max(1, camera["height"] // factor)
            Image.new("RGB", (width, height), (20 * i, 40, 80)).save(directory / name)
            Image.new("RGB", (2, 2)).save(directory / "not_registered.JPG")
    result = MipFixture(root, cameras, images)
    result.write_cameras()
    result.write_images()
    (root / "sparse/0/points3D.bin").write_bytes(b"must not be read")
    (root / "poses_bounds.npy").write_bytes(b"must not be read")
    return result
