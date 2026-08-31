#!/usr/bin/env python3
"""Run the Stage 0 Wan VAE temporal impulse experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rl3dsr.models.wan import WanVAE
from rl3dsr.validation.temporal_impulse import run_temporal_impulse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    vae = WanVAE.from_checkpoint(args.model_dir, device=args.device, dtype=torch.bfloat16)
    result = run_temporal_impulse(vae)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
