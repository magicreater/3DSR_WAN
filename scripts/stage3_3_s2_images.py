#!/usr/bin/env python3
"""Make paired HR/output/error sheets for the frozen Lego probe set."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import stage3_3_s2 as campaign
import stage3_3_structure_paired_audit as audit
import stage3_3_ucpe_rre_fusion as prior


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255


def main():
    sources = {
        "W3": campaign.s1.W3 / "eval/w3_lego_seed42",
        "S1": campaign.s1.CAMPAIGN / "eval/s1_lego_seed42",
        "S2": campaign.CAMPAIGN / "eval/s2_lego_seed42",
    }
    out = campaign.CAMPAIGN / "analysis/s2_lego_seed42/error_maps"
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for probe in audit.PROBES:
        view = f"view_{int(probe.split(':')[1]):03d}"
        reference = rgb(sources["S2"] / "images/lego" / view / "reference/hr.png")
        outputs = {name: rgb(root / "images/lego" / view / "seed_3302/correct.png")
                   for name, root in sources.items()}
        errors = {name: np.abs(image - reference).mean(axis=2)
                  for name, image in outputs.items()}
        scale = max(float(np.quantile(error, .99)) for error in errors.values())
        sheet = Image.new("RGB", (256 * 4, 256 * 2 + 48), "white")
        draw = ImageDraw.Draw(sheet)
        for column, name in enumerate(("HR", *sources)):
            image = reference if name == "HR" else outputs[name]
            sheet.paste(Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)),
                        (column * 256, 24))
            draw.text((column * 256 + 4, 4), name, fill="black")
            if name == "HR":
                continue
            error = np.uint8(np.clip(errors[name] / scale, 0, 1) * 255)
            sheet.paste(Image.fromarray(error).convert("RGB"), (column * 256, 280))
        draw.text((4, 538), f"absolute RGB error; common p99 scale {scale:.4f}", fill="black")
        target = out / f"{probe.replace(':', '_')}.png"
        sheet.save(target)
        rows[probe] = {"error_scale": scale, "mae": {
            name: float(error.mean()) for name, error in errors.items()},
                       "path": str(target)}
    prior.write_frozen_json(out.parent / "error_maps.json", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
