"""Metadata-first Mip-NeRF 360 adapter using registered COLMAP images only."""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from rl3dsr.camera import convert_colmap_w2c, scale_intrinsics
from rl3dsr.data.colmap import read_cameras_binary, read_images_binary
from rl3dsr.data.contracts import (
    DatasetFormatError, ObservationMetadata, SequenceKind, SequenceMetadata, Split,
)


@dataclass(frozen=True, slots=True)
class MipNeRF360Adapter:
    """Read one scene with filename-sorted, view-level holdout splits.

    ``image_factor`` selects existing images only; it is not a degradation or
    resize operation. The camera contract remains OpenCV camera-to-world.
    """

    scene_root: Path
    image_factor: int = 1
    holdout_every: int = 8

    def __post_init__(self):
        try:
            root = Path(self.scene_root).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise ValueError("expected a scene directory")
        except (OSError, RuntimeError, ValueError) as exc:
            raise DatasetFormatError(f"scene_root={self.scene_root}: {exc}") from exc
        if (
            isinstance(self.image_factor, bool)
            or not isinstance(self.image_factor, Integral)
            or self.image_factor not in (1, 2, 4, 8)
        ):
            raise DatasetFormatError(f"scene={root.name!r}: image_factor must be one of 1, 2, 4, 8")
        if (
            isinstance(self.holdout_every, bool)
            or not isinstance(self.holdout_every, Integral)
            or self.holdout_every < 2
        ):
            raise DatasetFormatError(f"scene={root.name!r}: holdout_every must be an integer >= 2")
        object.__setattr__(self, "scene_root", root)
        object.__setattr__(self, "image_factor", int(self.image_factor))
        object.__setattr__(self, "holdout_every", int(self.holdout_every))

    @property
    def _image_directory_name(self) -> str:
        return "images" if self.image_factor == 1 else f"images_{self.image_factor}"

    def index(self, split: Split | str) -> SequenceMetadata:
        """Validate the registered scene and return its selected multiview split.

        All registrations are validated before subsetting, including duplicate
        paths across splits. Indexing reads image headers, not image pixels.
        """
        context = f"scene={self.scene_root.name!r} split={split!r}"
        try:
            split = Split(split)
        except (ValueError, TypeError) as exc:
            raise DatasetFormatError(f"{context}: unsupported split") from exc
        if split is Split.VAL:
            raise DatasetFormatError(f"{context}: val split is not supported; it is not an alias of test")
        context = f"scene={self.scene_root.name!r} split={split.value!r}"

        try:
            source_directory = self._image_directory("images")
            target_directory = self._image_directory(self._image_directory_name)
            cameras = read_cameras_binary(self.scene_root / "sparse/0/cameras.bin")
            images = read_images_binary(self.scene_root / "sparse/0/images.bin")
        except (ValueError, OSError, RuntimeError) as exc:
            raise DatasetFormatError(f"{context}: {exc}") from exc

        observations = []
        source_paths, target_paths = set(), set()
        for frame_index, image in enumerate(sorted(images.values(), key=lambda item: item.name)):
            image_context = (
                f"{context} frame={frame_index} image_id={image.image_id} "
                f"camera_id={image.camera_id} name={image.name!r}"
            )
            try:
                if image.camera_id not in cameras:
                    raise DatasetFormatError("unknown camera ID reference")
                camera = cameras[image.camera_id]
                source_path = _resolve_image_path(source_directory, image.name)
                target_path = _resolve_image_path(target_directory, image.name)
                if source_path in source_paths or target_path in target_paths:
                    raise DatasetFormatError(f"duplicate resolved RGB path: {target_path}")
                source_paths.add(source_path)
                target_paths.add(target_path)

                source_size = _jpeg_size(source_path)
                if source_size != (camera.width, camera.height):
                    raise DatasetFormatError(
                        f"path={source_path}: original dimensions {source_size} do not match "
                        f"COLMAP dimensions {(camera.width, camera.height)}"
                    )
                width, height = source_size if target_path == source_path else _jpeg_size(target_path)
                fx, fy, cx, cy = camera.params
                intrinsics = scale_intrinsics(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    source_width=camera.width, source_height=camera.height,
                    target_width=width, target_height=height,
                )
                transform = convert_colmap_w2c(image.qvec, image.tvec)
                observation = ObservationMetadata(
                    observation_id=image.name,
                    frame_index=frame_index,
                    rgb_path=target_path,
                    width=width,
                    height=height,
                    K=intrinsics,
                    T_world_from_camera=transform,
                )
            except (ValueError, OSError, RuntimeError) as exc:
                raise DatasetFormatError(f"{image_context}: {exc}") from exc
            is_test = frame_index % self.holdout_every == 0
            if is_test == (split is Split.TEST):
                observations.append(observation)

        if not observations:
            raise DatasetFormatError(f"{context}: empty selected split with holdout_every={self.holdout_every}")
        return SequenceMetadata(
            scene_id=self.scene_root.name,
            split=split,
            sequence_kind=SequenceKind.MULTIVIEW,
            observations=tuple(observations),
        )

    def load_rgb(self, observation: ObservationMetadata) -> NDArray[np.uint8]:
        """Decode one RGB JPEG as contiguous uint8[H,W,3], without EXIF rotation."""
        context = f"scene={self.scene_root.name!r}"
        if not isinstance(observation, ObservationMetadata):
            raise DatasetFormatError(f"{context}: load_rgb requires ObservationMetadata")
        context += f" frame={observation.frame_index} path={observation.rgb_path}"
        try:
            directory = self._image_directory(self._image_directory_name)
            path = observation.rgb_path.resolve(strict=True)
            if not path.is_relative_to(directory):
                raise DatasetFormatError("observation is outside the selected image directory")
            with Image.open(path) as image:
                _validate_jpeg(image, path)
                if image.size != (observation.width, observation.height):
                    raise DatasetFormatError(
                        f"image dimensions changed after indexing: expected "
                        f"{(observation.width, observation.height)}, got {image.size}"
                    )
                pixels = np.array(image, dtype=np.uint8, copy=True, order="C")
            if pixels.shape != (observation.height, observation.width, 3):
                raise DatasetFormatError(f"unexpected decoded RGB shape: {pixels.shape}")
        except (ValueError, OSError, RuntimeError) as exc:
            raise DatasetFormatError(f"{context}: {exc}") from exc
        return pixels

    def _image_directory(self, name: str) -> Path:
        path = (self.scene_root / name).resolve(strict=True)
        if not path.is_relative_to(self.scene_root):
            raise DatasetFormatError(f"path={path}: image directory resolves outside the scene root")
        if not path.is_dir():
            raise DatasetFormatError(f"path={path}: expected image directory")
        return path


def _resolve_image_path(directory: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        not name or relative.is_absolute() or PureWindowsPath(name).drive
        or ".." in relative.parts or "\\" in name
    ):
        raise DatasetFormatError(f"invalid registered image path: {name!r}")
    if relative.suffix.lower() not in {".jpg", ".jpeg"}:
        raise DatasetFormatError(f"registered image path must name a JPEG: {name!r}")
    path = directory.joinpath(*relative.parts).resolve(strict=True)
    if not path.is_relative_to(directory):
        raise DatasetFormatError(f"path={path}: image resolves outside the selected image directory")
    if not path.is_file():
        raise DatasetFormatError(f"path={path}: registered RGB is not a file")
    return path


def _validate_jpeg(image: Image.Image, path: Path):
    if image.format != "JPEG" or image.mode != "RGB":
        raise DatasetFormatError(
            f"path={path}: expected RGB JPEG, got format={image.format!r} mode={image.mode!r}"
        )
    if min(image.size) <= 0:
        raise DatasetFormatError(f"path={path}: image dimensions must be positive")


def _jpeg_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        _validate_jpeg(image, path)
        return image.size
