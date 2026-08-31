"""Bounded readers for COLMAP PINHOLE camera/image metadata, not geometry.

Binary layout follows COLMAP's little-endian reconstruction format. Only
``cameras.bin`` and ``images.bin`` are needed; variable points2D tracks are
length-checked and skipped, without allocating or decoding them.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import struct
from typing import BinaryIO

from rl3dsr.data.contracts import DatasetFormatError


@dataclass(frozen=True, slots=True)
class ColmapCamera:
    camera_id: int
    width: int
    height: int
    params: tuple[float, float, float, float]  # fx, fy, cx, cy; PINHOLE only


@dataclass(frozen=True, slots=True)
class ColmapImage:
    image_id: int
    camera_id: int
    name: str
    qvec: tuple[float, float, float, float]  # qw, qx, qy, qz
    tvec: tuple[float, float, float]


class _BinaryReader:
    def __init__(self, stream: BinaryIO, path: Path):
        self.stream = stream
        self.path = path
        self.size = os.fstat(stream.fileno()).st_size

    @property
    def remaining(self) -> int:
        return self.size - self.stream.tell()

    def fail(self, message: str):
        raise DatasetFormatError(
            f"path={self.path} offset={self.stream.tell()}: {message}"
        )

    def unpack(self, fmt: str) -> tuple:
        size = struct.calcsize("<" + fmt)
        if size > self.remaining:
            self.fail(f"truncated binary: need {size} bytes, have {self.remaining}")
        raw = self.stream.read(size)
        if len(raw) != size:
            self.fail("truncated binary during read")
        return struct.unpack("<" + fmt, raw)

    def count(self, label: str, minimum_record_size: int) -> int:
        count, = self.unpack("Q")
        if count == 0:
            self.fail(f"no {label} records")
        if count > self.remaining // minimum_record_size:
            self.fail(f"truncated binary: declared {label} count exceeds remaining bytes")
        return count

    def image_name(self) -> str:
        raw = bytearray()
        # Filesystem paths cannot reasonably exceed this size. Bound malformed
        # unterminated names independently of the reconstruction file size.
        for _ in range(min(4096, self.remaining)):
            byte = self.stream.read(1)
            if byte == b"\0":
                try:
                    name = raw.decode("utf-8")
                except UnicodeDecodeError:
                    self.fail("image name must be valid UTF-8")
                if not name:
                    self.fail("image name must not be empty")
                return name
            if not byte:
                break
            raw.extend(byte)
        self.fail("unterminated image name or name exceeds 4095 UTF-8 bytes")

    def skip_points2d(self):
        count, = self.unpack("Q")
        byte_count = count * 24  # float64 x, float64 y, int64 point3D_id
        if byte_count > self.remaining:
            self.fail("truncated points2D section: declared length exceeds file size")
        self.stream.seek(byte_count, os.SEEK_CUR)

    def finish(self):
        if self.remaining:
            self.fail(f"unexpected trailing bytes: {self.remaining}")


@contextmanager
def _open_reader(path: Path):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            yield _BinaryReader(stream, path)
    except OSError as exc:
        raise DatasetFormatError(f"path={path}: cannot read COLMAP binary: {exc}") from exc


def read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    """Read PINHOLE cameras, keyed by camera ID (not record position)."""
    cameras = {}
    with _open_reader(path) as reader:
        # Only assume the fixed header here. Other camera models may have fewer
        # parameters and must report unsupported model, not false truncation.
        for _ in range(reader.count("camera", 24)):
            camera_id, model_id, width, height = reader.unpack("iiQQ")
            if camera_id < 0 or camera_id in cameras:
                reader.fail(f"invalid or duplicate camera ID {camera_id}")
            if model_id != 1:
                reader.fail(f"camera_id={camera_id}: only PINHOLE (model 1) is supported, got {model_id}")
            params = reader.unpack("4d")
            if width == 0 or height == 0:
                reader.fail(f"camera_id={camera_id}: camera dimensions must be positive")
            if not all(math.isfinite(p) for p in params) or min(params[:2]) <= 0:
                reader.fail(f"camera_id={camera_id}: intrinsics must be finite with positive focal lengths")
            cameras[camera_id] = ColmapCamera(camera_id, width, height, params)
        reader.finish()
    return cameras


def read_images_binary(path: Path) -> dict[int, ColmapImage]:
    """Read registered image metadata, discarding all points2D track payloads."""
    images = {}
    names = set()
    with _open_reader(path) as reader:
        for _ in range(reader.count("image", 73)):
            image_id, *pose, camera_id = reader.unpack("i7di")
            if image_id < 0 or image_id in images:
                reader.fail(f"invalid or duplicate image ID {image_id}")
            name = reader.image_name()
            if name in names:
                reader.fail(f"image_id={image_id}: duplicate image name {name!r}")
            if not all(math.isfinite(value) for value in pose):
                reader.fail(f"image_id={image_id} name={name!r}: pose must be finite")
            reader.skip_points2d()
            images[image_id] = ColmapImage(
                image_id, camera_id, name, tuple(pose[:4]), tuple(pose[4:])
            )
            names.add(name)
        reader.finish()
    return images
