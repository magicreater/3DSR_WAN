"""Metadata-first adapter for the original NeRF Synthetic dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from rl3dsr.camera import build_pinhole_intrinsics, convert_nerf_synthetic_c2w
from rl3dsr.data.contracts import (
    AlphaBackground,
    DatasetFormatError,
    ObservationMetadata,
    SequenceKind,
    SequenceMetadata,
    Split,
)


@dataclass(frozen=True, slots=True)
class NeRFSyntheticAdapter:
    """Index one NeRF Synthetic scene without scanning image directories."""

    scene_root: Path
    alpha_background: AlphaBackground = AlphaBackground.WHITE

    def __post_init__(self) -> None:
        try:
            scene_root = Path(self.scene_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise DatasetFormatError(
                f"scene_root={self.scene_root!s}: scene directory does not exist"
            ) from exc
        if not scene_root.is_dir():
            raise DatasetFormatError(
                f"scene_root={scene_root}: expected a scene directory"
            )
        try:
            alpha_background = AlphaBackground(self.alpha_background)
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(
                f"scene={scene_root.name!r}: unsupported alpha background "
                f"{self.alpha_background!r}"
            ) from exc

        object.__setattr__(self, "scene_root", scene_root)
        object.__setattr__(self, "alpha_background", alpha_background)

    def index(self, split: Split | str) -> SequenceMetadata:
        """Return one ordered multiview sequence for ``split``.

        Only ``frames[].file_path`` entries in the selected manifest define the
        observation set. Image pixels are not decoded during indexing.
        """

        try:
            canonical_split = Split(split)
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(
                self._context(split=split) + f": unsupported split {split!r}"
            ) from exc

        manifest_path = self.scene_root / f"transforms_{canonical_split.value}.json"
        manifest = self._read_manifest(manifest_path, canonical_split)

        if "camera_angle_x" not in manifest:
            raise DatasetFormatError(
                self._context(split=canonical_split, path=manifest_path)
                + ": missing camera_angle_x"
            )
        camera_angle_x = manifest["camera_angle_x"]
        if "frames" not in manifest:
            raise DatasetFormatError(
                self._context(split=canonical_split, path=manifest_path)
                + ": missing frames"
            )
        frames = manifest["frames"]
        if not isinstance(frames, list) or not frames:
            raise DatasetFormatError(
                self._context(split=canonical_split, path=manifest_path)
                + ": frames must be a non-empty list"
            )

        observations: list[ObservationMetadata] = []
        seen_observation_ids: set[str] = set()
        seen_rgb_paths: set[Path] = set()
        for frame_index, frame in enumerate(frames):
            frame_context = self._context(
                split=canonical_split,
                frame_index=frame_index,
            )
            if not isinstance(frame, dict):
                raise DatasetFormatError(frame_context + ": frame must be an object")
            raw_file_path = frame.get("file_path")
            observation_id, rgb_path = self._resolve_rgb_path(
                raw_file_path,
                split=canonical_split,
                frame_index=frame_index,
            )
            if observation_id in seen_observation_ids:
                raise DatasetFormatError(
                    self._context(
                        split=canonical_split,
                        frame_index=frame_index,
                        path=rgb_path,
                    )
                    + f": duplicate observation_id {observation_id!r}"
                )
            if rgb_path in seen_rgb_paths:
                raise DatasetFormatError(
                    self._context(
                        split=canonical_split,
                        frame_index=frame_index,
                        path=rgb_path,
                    )
                    + ": duplicate referenced RGB path"
                )

            width, height = self._read_png_header(
                rgb_path,
                split=canonical_split,
                frame_index=frame_index,
            )
            try:
                intrinsics = build_pinhole_intrinsics(
                    width=width,
                    height=height,
                    camera_angle_x=camera_angle_x,
                )
            except ValueError as exc:
                raise DatasetFormatError(frame_context + f": {exc}") from exc
            if "transform_matrix" not in frame:
                raise DatasetFormatError(frame_context + ": missing transform_matrix")
            try:
                transform = convert_nerf_synthetic_c2w(frame["transform_matrix"])
            except ValueError as exc:
                raise DatasetFormatError(frame_context + f": {exc}") from exc

            observations.append(
                ObservationMetadata(
                    observation_id=observation_id,
                    frame_index=frame_index,
                    rgb_path=rgb_path,
                    width=width,
                    height=height,
                    K=intrinsics,
                    T_world_from_camera=transform,
                )
            )
            seen_observation_ids.add(observation_id)
            seen_rgb_paths.add(rgb_path)

        return SequenceMetadata(
            scene_id=self.scene_root.name,
            split=canonical_split,
            sequence_kind=SequenceKind.MULTIVIEW,
            observations=tuple(observations),
        )

    def load_rgb(self, observation: ObservationMetadata) -> NDArray[np.uint8]:
        """Decode one indexed observation as contiguous ``uint8[H, W, 3]``."""

        if not isinstance(observation, ObservationMetadata):
            raise DatasetFormatError(
                f"scene={self.scene_root.name!r}: load_rgb requires ObservationMetadata"
            )
        try:
            rgb_path = observation.rgb_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise DatasetFormatError(
                self._context(
                    frame_index=observation.frame_index,
                    path=observation.rgb_path,
                )
                + ": referenced RGB file no longer exists"
            ) from exc
        if not rgb_path.is_relative_to(self.scene_root):
            raise DatasetFormatError(
                self._context(frame_index=observation.frame_index, path=rgb_path)
                + ": observation is outside this adapter's scene root"
            )

        try:
            with Image.open(rgb_path) as image:
                self._validate_png_image(
                    image,
                    rgb_path,
                    split=None,
                    frame_index=observation.frame_index,
                )
                if image.size != (observation.width, observation.height):
                    raise DatasetFormatError(
                        self._context(
                            frame_index=observation.frame_index,
                            path=rgb_path,
                        )
                        + ": image dimensions changed after indexing; "
                        f"expected {(observation.width, observation.height)}, "
                        f"got {image.size}"
                    )

                if image.mode == "RGBA":
                    color = (
                        (255, 255, 255, 255)
                        if self.alpha_background is AlphaBackground.WHITE
                        else (0, 0, 0, 255)
                    )
                    background = Image.new("RGBA", image.size, color)
                    rgb_image = Image.alpha_composite(background, image).convert("RGB")
                else:
                    rgb_image = image
                pixels = np.array(rgb_image, dtype=np.uint8, copy=True, order="C")
        except DatasetFormatError:
            raise
        except (OSError, UnidentifiedImageError) as exc:
            raise DatasetFormatError(
                self._context(frame_index=observation.frame_index, path=rgb_path)
                + f": failed to decode PNG: {exc}"
            ) from exc

        expected_shape = (observation.height, observation.width, 3)
        if pixels.shape != expected_shape:
            raise DatasetFormatError(
                self._context(frame_index=observation.frame_index, path=rgb_path)
                + f": decoded RGB must have shape {expected_shape}, got {pixels.shape}"
            )
        return np.ascontiguousarray(pixels)

    def _read_manifest(self, path: Path, split: Split) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except FileNotFoundError as exc:
            raise DatasetFormatError(
                self._context(split=split, path=path) + ": manifest does not exist"
            ) from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DatasetFormatError(
                self._context(split=split, path=path)
                + f": failed to read manifest: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise DatasetFormatError(
                self._context(split=split, path=path)
                + ": manifest root must be an object"
            )
        return manifest

    def _resolve_rgb_path(
        self,
        raw_file_path: Any,
        *,
        split: Split,
        frame_index: int,
    ) -> tuple[str, Path]:
        context = self._context(split=split, frame_index=frame_index)
        if not isinstance(raw_file_path, str) or not raw_file_path:
            raise DatasetFormatError(context + ": file_path must be a non-empty string")

        normalized = raw_file_path[2:] if raw_file_path.startswith("./") else raw_file_path
        relative = PurePosixPath(normalized)
        if (
            not normalized
            or relative.is_absolute()
            or PureWindowsPath(normalized).is_absolute()
            or ".." in relative.parts
        ):
            raise DatasetFormatError(
                context + f": file_path must stay within the scene root, got {raw_file_path!r}"
            )
        if "\\" in normalized:
            raise DatasetFormatError(
                context + f": file_path must use POSIX separators, got {raw_file_path!r}"
            )

        suffix = relative.suffix.lower()
        if suffix and suffix != ".png":
            raise DatasetFormatError(
                context + f": referenced RGB must be a PNG, got {raw_file_path!r}"
            )
        rgb_relative = relative if suffix else PurePosixPath(relative.as_posix() + ".png")
        candidate = self.scene_root.joinpath(*rgb_relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise DatasetFormatError(
                self._context(split=split, frame_index=frame_index, path=candidate)
                + ": referenced RGB file does not exist"
            ) from exc
        if not resolved.is_relative_to(self.scene_root):
            raise DatasetFormatError(
                self._context(split=split, frame_index=frame_index, path=resolved)
                + ": referenced RGB resolves outside the scene root"
            )
        if not resolved.is_file():
            raise DatasetFormatError(
                self._context(split=split, frame_index=frame_index, path=resolved)
                + ": referenced RGB is not a file"
            )
        return relative.as_posix(), resolved

    def _read_png_header(
        self,
        path: Path,
        *,
        split: Split,
        frame_index: int,
    ) -> tuple[int, int]:
        try:
            with Image.open(path) as image:
                self._validate_png_image(
                    image,
                    path,
                    split=split,
                    frame_index=frame_index,
                )
                return image.size
        except DatasetFormatError:
            raise
        except (OSError, UnidentifiedImageError) as exc:
            raise DatasetFormatError(
                self._context(split=split, frame_index=frame_index, path=path)
                + f": failed to read PNG header: {exc}"
            ) from exc

    def _validate_png_image(
        self,
        image: Image.Image,
        path: Path,
        *,
        split: Split | None,
        frame_index: int,
    ) -> None:
        context = self._context(split=split, frame_index=frame_index, path=path)
        if image.format != "PNG":
            raise DatasetFormatError(context + f": expected PNG format, got {image.format!r}")
        if image.mode not in {"RGB", "RGBA"}:
            raise DatasetFormatError(
                context + f": referenced RGB must use RGB or RGBA mode, got {image.mode!r}"
            )
        width, height = image.size
        if width <= 0 or height <= 0:
            raise DatasetFormatError(context + ": image dimensions must be positive")

    def _context(
        self,
        *,
        split: Split | str | None = None,
        frame_index: int | None = None,
        path: Path | None = None,
    ) -> str:
        parts = [f"scene={self.scene_root.name!r}"]
        if split is not None:
            split_value = split.value if isinstance(split, Split) else split
            parts.append(f"split={split_value!r}")
        if frame_index is not None:
            parts.append(f"frame={frame_index}")
        if path is not None:
            parts.append(f"path={path}")
        return " ".join(parts)

