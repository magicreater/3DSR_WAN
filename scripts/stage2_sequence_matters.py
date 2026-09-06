#!/usr/bin/env python3
"""Stage, train, render, and analyze the fixed SequenceMatters 3DGS arms."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rl3dsr.validation.memorization import nvs_consistency_verdict
from rl3dsr.validation.sequence_matters import (
    sha256_file,
    stage_blender_dataset,
    validate_staged_blender,
    write_deterministic_ply,
)


SCENES = ("chair", "lego", "drums")
ARMS = (
    "hr",
    "bicubic",
    "stage1",
    "simple_rre_seed_42",
    "full_rre_seed_42",
    "full_rre_seed_43",
)
VIEW_INDICES = (0, 33, 66, 99)
CONTACT_VIEWS = (0, 50, 100, 150)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str], *, cwd: Path, gpu: int, log: Path) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"command failed with {returncode}; see {log}")


def _source_image_root(args, scene: str, arm: str) -> Path:
    if arm in {"hr", "bicubic", "stage1"}:
        return args.campaign_root / scene / "stage1" / "final" / "images" / "seed_2201" / arm
    return args.campaign_root / scene / arm / "final" / "images" / "seed_2201" / "correct"


def _images(root: Path) -> dict[int, Path]:
    result = {index: root / f"view_{index:03d}.png" for index in VIEW_INDICES}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen SR inputs: {missing}")
    return result


def _config(args, scene: str) -> Path:
    path = args.sequence_root / "configs" / f"{scene}.yml"
    staged_scene = args.sequence_root / "staged" / scene
    content = (
        f"hr_source_dir: {staged_scene}\n"
        f"lr_source_dir: {staged_scene}\n"
        f"vsr_save_dir: {staged_scene}\n"
        "video_save_path: null\nvsr_model: psrt\nspynet_path: null\nvsr_model_path: null\n"
        "downscale_factor: 4\nsubpixel: bicubic\nwhite_background: true\nals: false\n"
        "num_images_in_sequence: 4\nsimilarity: feature\nthres_values: [45]\nlambda_tex: 0.60\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"SequenceMatters config drift: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path


def _stage(args) -> None:
    if not (args.campaign_root / "final_evaluation_complete.json").is_file():
        raise RuntimeError("3DGS staging requires the frozen final Wan evaluation")
    summary = []
    for scene in SCENES:
        config = _config(args, scene)
        shared_ply = args.sequence_root / "shared_points" / f"{scene}.ply"
        ply_sha = write_deterministic_ply(shared_ply, seed=2201)
        lr_images = _images(
            args.campaign_root / scene / "stage1" / "final" / "images" / "seed_2201" / "lr"
        )
        source_scene = args.dataset_root / scene
        test_payload = json.loads((source_scene / "transforms_test.json").read_text(encoding="utf-8"))
        if len(test_payload.get("frames", ())) != 200:
            raise RuntimeError(f"scene {scene} must contain exactly 200 novel-view cameras")
        test_names = tuple(Path(frame["file_path"]).name + ".png" for frame in test_payload["frames"])
        train_payload = json.loads((source_scene / "transforms_train.json").read_text(encoding="utf-8"))
        train_names = tuple(Path(train_payload["frames"][index]["file_path"]).name + ".png" for index in VIEW_INDICES)
        scene_manifests = []
        for arm in ARMS:
            target = args.sequence_root / "staged" / scene / arm
            source_images = _images(_source_image_root(args, scene, arm))
            expected_image_sha256 = {
                str(index): sha256_file(source_images[index]) for index in VIEW_INDICES
            }
            expected_lr_sha256 = {
                str(index): sha256_file(lr_images[index]) for index in VIEW_INDICES
            }
            config_sha256 = sha256_file(config)
            if target.exists():
                if not (target / "staging_manifest.json").is_file():
                    raise RuntimeError(f"refusing incomplete staged dataset: {target}")
                validation = validate_staged_blender(
                    target,
                    train_names=train_names,
                    test_names=test_names,
                    image_size=256,
                    lr_size=64,
                    expected_ply_sha256=ply_sha,
                )
                manifest = json.loads((target / "staging_manifest.json").read_text(encoding="utf-8"))
                expected = {
                    "arm": arm,
                    "sequence_config": str(config.resolve()),
                    "sequence_config_sha256": config_sha256,
                    "input_image_sha256": expected_image_sha256,
                    "input_lr_sha256": expected_lr_sha256,
                    "points3d_sha256": ply_sha,
                }
                if any(manifest.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"staged input or config drift detected: {target}")
                staged_image_sha256 = {
                    str(index): sha256_file(target / "train" / train_names[position])
                    for position, index in enumerate(VIEW_INDICES)
                }
                staged_lr_sha256 = {
                    str(index): sha256_file(target / "train_lr" / train_names[position])
                    for position, index in enumerate(VIEW_INDICES)
                }
                if staged_image_sha256 != expected_image_sha256 or staged_lr_sha256 != expected_lr_sha256:
                    raise RuntimeError(f"staged image content drift detected: {target}")
            else:
                manifest = stage_blender_dataset(
                    target,
                    source_scene=source_scene,
                    train_images=source_images,
                    train_lr_images=lr_images,
                    train_indices=VIEW_INDICES,
                    shared_ply=shared_ply,
                )
                validation = manifest["validation"]
                manifest.update(
                    arm=arm,
                    sequence_config=str(config.resolve()),
                    sequence_config_sha256=config_sha256,
                )
                _atomic_json(target / "staging_manifest.json", manifest)
            scene_manifests.append(manifest)
            summary.append({"scene": scene, "arm": arm, **validation, "config_sha256": config_sha256})
        for key in (
            "input_lr_sha256",
            "staged_lr_sha256",
            "points3d_sha256",
            "sequence_config_sha256",
            "train_transform_matrices_sha256",
            "test_transform_matrices_sha256",
        ):
            values = {json.dumps(manifest[key], sort_keys=True) for manifest in scene_manifests}
            if len(values) != 1:
                raise RuntimeError(f"SequenceMatters arms do not share {key} for {scene}")
    complete = args.sequence_root / "staging_complete.json"
    if complete.is_file():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        if saved.get("arms") != summary:
            raise RuntimeError("SequenceMatters staging summary drift detected")
    else:
        _atomic_json(complete, {"arms": summary, "completed_unix": time.time()})


def _paths(args, scene: str, arm: str) -> tuple[Path, Path, Path]:
    return (
        args.sequence_root / "staged" / scene / arm,
        args.sequence_root / "models" / scene / arm,
        _config(args, scene),
    )


def _train_one(args, scene: str, arm: str) -> None:
    staged, model, config = _paths(args, scene, arm)
    complete = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if complete.is_file():
        return
    if model.exists() and any(model.iterdir()):
        raise RuntimeError(f"refusing to overwrite partial 3DGS result: {model}")
    command = [
        str(args.seqmat_python),
        "train.py",
        "-s", str(staged),
        "-m", str(model),
        "--config", str(config),
        "--eval",
        "--skip_vsr",
        "--precomputed_vsr_dir", str(staged / "train"),
        "--iterations", "30000",
        "--test_iterations", "7000", "30000",
        "--save_iterations", "7000", "30000",
        "--port", str(args.port),
    ]
    _run(
        command,
        cwd=args.sequence_matters,
        gpu=args.gpu,
        log=args.sequence_root / "logs" / f"train_{scene}_{arm}.log",
    )
    if not complete.is_file():
        raise RuntimeError(f"SequenceMatters did not create iteration 30000: {model}")
    protocol = json.loads((args.campaign_root / "protocol_manifest.json").read_text(encoding="utf-8"))
    _atomic_json(
        model / "rl3dsr_run_manifest.json",
        {
            "scene": scene,
            "arm": arm,
            "physical_gpu": args.gpu,
            "sequence_seed": 0,
            "iterations": 30000,
            "lambda_tex": 0.60,
            "subpixel": "bicubic",
            "white_background": True,
            "staging_manifest_sha256": sha256_file(staged / "staging_manifest.json"),
            "config_sha256": sha256_file(config),
            "sequence_matters": protocol["sequence_matters"],
            "completed_unix": time.time(),
        },
    )


def _render_one(args, scene: str, arm: str) -> None:
    _staged, model, config = _paths(args, scene, arm)
    test_renders = model / "test" / "ours_30000" / "renders"
    train_renders = model / "train" / "ours_30000" / "renders"
    if len(tuple(test_renders.glob("*.png"))) == 200 and len(tuple(train_renders.glob("*.png"))) == 4:
        return
    if test_renders.exists() or train_renders.exists():
        raise RuntimeError(f"refusing to overwrite partial renders: {model}")
    _run(
        [
            str(args.seqmat_python),
            "render_samename.py",
            "-m", str(model),
            "--config", str(config),
            "--iteration", "30000",
        ],
        cwd=args.sequence_matters,
        gpu=args.gpu,
        log=args.sequence_root / "logs" / f"render_{scene}_{arm}.log",
    )
    if len(tuple(test_renders.glob("*.png"))) != 200 or len(tuple(train_renders.glob("*.png"))) != 4:
        raise RuntimeError(f"render count mismatch: {model}")


def _render_metrics(args, scene: str, arm: str, split: str, criterion) -> tuple[list[dict], dict]:
    import torch
    import torchvision.transforms.functional as tf

    sys.path.insert(0, str(args.sequence_matters))
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    _staged, model, _config_path = _paths(args, scene, arm)
    root = model / split / "ours_30000"
    rows = []
    for render_path in sorted((root / "renders").glob("*.png")):
        gt_path = root / "gt" / render_path.name
        render = tf.to_tensor(Image.open(render_path).convert("RGB")).unsqueeze(0).cuda()
        target = tf.to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).cuda()
        rows.append({
            "scene": scene,
            "arm": arm,
            "split": split,
            "view": render_path.stem,
            "ssim": float(ssim(render, target)),
            "psnr": float(psnr(render, target)),
            "lpips": float(criterion(render, target)),
        })
    if not rows:
        raise RuntimeError(f"no {split} renders found: {model}")
    summary = {
        "scene": scene,
        "arm": arm,
        "split": split,
        "view_count": len(rows),
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in ("psnr", "ssim", "lpips")},
    }
    return rows, summary


def _metrics_one(args, scene: str, arm: str) -> None:
    _staged, model, _config_path = _paths(args, scene, arm)
    target = model / "rl3dsr_metrics.json"
    if target.is_file():
        return
    if not (model / "test" / "ours_30000" / "renders").is_dir():
        raise RuntimeError("render before computing SequenceMatters metrics")
    sys.path.insert(0, str(args.sequence_matters))
    from lpipsPyTorch.modules.lpips import LPIPS

    criterion = LPIPS("vgg").cuda().eval()
    rows, summaries = [], []
    for split in ("train", "test"):
        split_rows, summary = _render_metrics(args, scene, arm, split, criterion)
        rows.extend(split_rows)
        summaries.append(summary)
    _write_csv(model / "rl3dsr_per_view_metrics.csv", rows)
    _atomic_json(target, {"summaries": summaries})


def _panel(image: Image.Image, label: str, size: tuple[int, int] = (256, 280)) -> Image.Image:
    result = Image.new("RGB", size, "white")
    result.paste(image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS), (0, 24))
    ImageDraw.Draw(result).text((6, 5), label, fill="black")
    return result


def _contact_and_video(args, scene: str) -> None:
    import imageio.v2 as imageio

    output = args.sequence_root / "analysis" / scene
    output.mkdir(parents=True, exist_ok=True)
    reference_model = args.sequence_root / "models" / scene / "hr" / "test" / "ours_30000"
    labels = ("HR target", "HR-3DGS", "Bicubic-3DGS", "Stage1-3DGS", "Simple RRE-3DGS", "Full RRE s42", "Full RRE s43")
    arm_order = (None, *ARMS)
    sheets = []
    for view in CONTACT_VIEWS:
        name = f"r_{view}.png"
        panels = []
        target = Image.open(reference_model / "gt" / name).convert("RGB")
        panels.append(_panel(target, labels[0]))
        for arm, label in zip(ARMS, labels[1:]):
            image = Image.open(args.sequence_root / "models" / scene / arm / "test" / "ours_30000" / "renders" / name)
            panels.append(_panel(image, label))
        sheet = Image.new("RGB", (256 * len(panels), 280), "white")
        for index, panel in enumerate(panels):
            sheet.paste(panel, (256 * index, 0))
        sheet.save(output / f"contact_view_{view:03d}.png")
        sheets.append(sheet)

        for arm in ARMS:
            render = np.asarray(Image.open(args.sequence_root / "models" / scene / arm / "test" / "ours_30000" / "renders" / name).convert("RGB"), dtype=np.float32)
            truth = np.asarray(target, dtype=np.float32)
            error = np.abs(render - truth).mean(axis=-1).clip(0, 64) / 64
            heat = np.stack((error * 255, np.square(error) * 180, np.zeros_like(error)), axis=-1).astype(np.uint8)
            Image.fromarray(heat).save(output / f"error_{arm}_view_{view:03d}.png")

    orbit = output / "orbit_comparison.mp4"
    with imageio.get_writer(orbit, fps=30, codec="libx264", quality=8) as writer:
        for view in range(200):
            name = f"r_{view}.png"
            target = Image.open(reference_model / "gt" / name).convert("RGB")
            panels = [_panel(target, labels[0])]
            for arm, label in zip(ARMS, labels[1:]):
                image = Image.open(args.sequence_root / "models" / scene / arm / "test" / "ours_30000" / "renders" / name)
                panels.append(_panel(image, label))
            frame = Image.new("RGB", (1024, 560), "white")
            for index, panel in enumerate(panels):
                frame.paste(panel, ((index % 4) * 256, (index // 4) * 280))
            writer.append_data(np.asarray(frame))


def _extract_curves(args, scene: str) -> None:
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    output = args.sequence_root / "analysis" / scene
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ARMS:
        model = args.sequence_root / "models" / scene / arm
        accumulator = EventAccumulator(str(model)).Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                rows.append({"scene": scene, "arm": arm, "tag": tag, "step": event.step, "value": event.value, "wall_time": event.wall_time})
    _write_csv(output / "tensorboard_scalars.csv", rows)
    tags = ("train_loss_patches/total_loss", "train_loss_patches/l1_loss", "iter_time", "total_points", "test/loss_viewpoint - psnr", "train/loss_viewpoint - psnr")
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    colors = {arm: color for arm, color in zip(ARMS, ("#1f2430", "#a3befa", "#f0986e", "#a3d576", "#f390ca", "#736422"))}
    for axis, tag in zip(axes.flat, tags):
        for arm in ARMS:
            values = [row for row in rows if row["arm"] == arm and row["tag"] == tag]
            if values:
                axis.plot([row["step"] for row in values], [row["value"] for row in values], label=arm, color=colors[arm], linewidth=1.2)
        axis.set_title(tag)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7, ncol=2)
    figure.savefig(output / "training_curves.png", dpi=180)
    plt.close(figure)


def _analyze(args) -> None:
    summaries, per_view = [], []
    for scene in SCENES:
        for arm in ARMS:
            model = args.sequence_root / "models" / scene / arm
            metrics = json.loads((model / "rl3dsr_metrics.json").read_text(encoding="utf-8"))
            summaries.extend(metrics["summaries"])
            with (model / "rl3dsr_per_view_metrics.csv").open(newline="", encoding="utf-8") as handle:
                per_view.extend(csv.DictReader(handle))
        _contact_and_video(args, scene)
        _extract_curves(args, scene)
    test_rows = [row for row in summaries if row["split"] == "test"]
    verdict = nvs_consistency_verdict(test_rows)
    output = args.sequence_root / "analysis"
    _write_csv(output / "summary_metrics.csv", summaries)
    _write_csv(output / "per_view_metrics.csv", per_view)
    _atomic_json(output / "nvs_consistency_verdict.json", verdict)
    _atomic_json(args.sequence_root / "analysis_complete.json", {"verdict": verdict, "completed_unix": time.time()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "train-one", "render-one", "metrics-one", "run-one", "analyze"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--campaign-root", type=Path, default=Path("artifacts/stage2_full_rre_memorization_20260904"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/nerf_synthetic"))
    parser.add_argument("--sequence-matters", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/SequenceMatters"))
    parser.add_argument("--seqmat-python", type=Path, default=Path("/home/linzizhuo/miniconda3/envs/seqmat/bin/python"))
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=6009)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    for name in ("campaign_root", "dataset_root", "sequence_matters", "seqmat_python"):
        path = getattr(args, name)
        if not path.is_absolute():
            path = args.repo / path
        setattr(args, name, path.resolve())
    args.sequence_root = args.campaign_root / "sequence_matters"
    if args.mode == "stage":
        _stage(args)
        return
    if args.mode == "analyze":
        _analyze(args)
        return
    if args.scene is None or args.arm is None:
        raise ValueError("train/render/metrics modes require --scene and --arm")
    if args.mode in {"train-one", "run-one"}:
        _train_one(args, args.scene, args.arm)
    if args.mode in {"render-one", "run-one"}:
        _render_one(args, args.scene, args.arm)
    if args.mode in {"metrics-one", "run-one"}:
        _metrics_one(args, args.scene, args.arm)


if __name__ == "__main__":
    main()
