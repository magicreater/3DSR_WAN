#!/usr/bin/env python3
"""Create explicit NeRF Synthetic test-depth manifests without changing data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, required=True)
    args = parser.parse_args()
    if args.depth_scale <= 0:
        raise ValueError("depth-scale must be positive")
    test_root = args.scene / "test"
    rgb_paths = sorted(test_root.glob("r_*.png"), key=lambda path: int(path.stem.split("_")[1]))
    frames: dict[str, str] = {}
    for rgb in rgb_paths:
        parts = rgb.stem.split("_")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        index = int(parts[1])
        matches = sorted(test_root.glob(f"r_{index}_depth_*.png"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected exactly one depth for test frame {index}, found {len(matches)}")
        depth = matches[0]
        with Image.open(rgb) as rgb_image, Image.open(depth) as depth_image:
            if rgb_image.size != depth_image.size:
                raise RuntimeError(f"RGB/depth size mismatch at frame {index}")
        frames[str(index)] = str(depth.resolve())
    payload = {
        "format_version": 1,
        "scene_id": args.scene.name,
        "depth_scale": args.depth_scale,
        "depth_channel": 0,
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
