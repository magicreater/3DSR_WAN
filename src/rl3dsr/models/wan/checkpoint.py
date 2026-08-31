"""Checkpoint paths and construction for the vendored official Wan source."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WanCheckpoint:
    """Validated external Wan2.1-T2V-1.3B checkpoint paths."""

    model_dir: Path
    config_path: Path
    dit_path: Path
    vae_path: Path

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "WanCheckpoint":
        root = Path(model_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Wan model directory does not exist: {root}")
        paths = {
            "config_path": root / "config.json",
            "dit_path": root / "diffusion_pytorch_model.safetensors",
            "vae_path": root / "Wan2.1_VAE.pth",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing Wan checkpoint files: " + ", ".join(missing))
        try:
            config = json.loads(paths["config_path"].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Wan config: {paths['config_path']}") from exc
        if not isinstance(config, dict) or config.get("in_dim") != 16 or config.get("out_dim") != 16:
            raise ValueError("checkpoint config must describe a 16-channel Wan T2V model")
        return cls(root, **paths)

    def read_config(self) -> dict[str, Any]:
        return json.loads(self.config_path.read_text(encoding="utf-8"))
