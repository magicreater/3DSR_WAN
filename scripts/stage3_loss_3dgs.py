#!/usr/bin/env python3
"""Nested 4/8/16-input SequenceMatters diagnostic for Stage 3 SR."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

import stage3_experiment as st
import stage3_loss_campaign as campaign
import stage3_loss_experiment as experiment
from rl3dsr.data import Split
from rl3dsr.validation.sequence_matters import (
    sha256_file, stage_blender_dataset, validate_staged_blender,
    write_deterministic_ply,
)
from rl3dsr.validation.stage3_protocol import load_stage3_config, nearest_view_indices


OUT = campaign.OUT / "3dgs"
DATASET = experiment.RUNTIME_ROOT / "datasets/nerf_synthetic/lego"
PRIOR = Path("/data/linzizhuo/RL3DSR_WAN_REMO-stage3-final")
A5_CONFIG = PRIOR / "artifacts/stage3_final_20260924/config/a5_v4_lego_s42.json"
A5_CHECKPOINT = PRIOR / "artifacts/stage3_final_20260924/train/a5_v4_lego_s42/stage3_step_4000.pt"
SEQUENCE = Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/SequenceMatters")
SEQUENCE_PYTHON = Path("/home/linzizhuo/miniconda3/envs/seqmat/bin/python")
ARMS = ("candidate", "a5", "bicubic", "hr")
COUNTS = (4, 8, 16)


def frozen(path: Path, payload: object):
    experiment.frozen_json(path, payload)


def nested_indices(centers: torch.Tensor, total: int = 16) -> list[int]:
    """Farthest-point order by angular distance, seeded at frame zero."""
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < total:
        raise ValueError("need at least 16 three-dimensional camera centers")
    directions = F.normalize(centers.float(), dim=-1)
    similarity = (directions @ directions.T).clamp(-1, 1)
    angle = torch.acos(similarity)
    selected = [0]
    while len(selected) < total:
        remaining = [index for index in range(len(centers)) if index not in selected]
        selected.append(max(remaining, key=lambda index: (
            float(angle[index, selected].min()), -index,
        )))
    return selected


def prepare():
    if OUT.exists():
        raise RuntimeError("3DGS preparation output already exists")
    OUT.mkdir(parents=True)
    config = load_stage3_config(campaign.config_path("full"))
    _, sequence = st._scene_data(DATASET, config, Split.TRAIN)
    camera = st._camera(sequence.observations, config.image_size, torch.device("cpu"))
    centers = camera.T_world_from_camera[0, :, :3, 3]
    indices = nested_indices(centers)
    if len(set(indices)) != 16 or indices[0] != 0:
        raise RuntimeError("nested 3DGS indices invalid")
    contexts = {str(index): nearest_view_indices(camera, index, 4) for index in indices}
    if any(len(set(group)) != 4 or group[0] != int(index)
           for index, group in contexts.items()):
        raise RuntimeError("SR contexts invalid")
    frozen(OUT / "view_protocol.json", {
        "scene": "lego", "sr_context_views": 4,
        "nested_indices": indices,
        "sets": {str(count): indices[:count] for count in COUNTS},
        "legacy_four_comparator": [0, 33, 66, 99],
        "sr_contexts": contexts,
        "source_train_transforms_sha256": sha256_file(DATASET / "transforms_train.json"),
        "source_test_transforms_sha256": sha256_file(DATASET / "transforms_test.json"),
        "candidate_selection": "full review correct SR quality first",
        "sampler": "native 50-step, seed 3302*1e6+target_index, CFG=1",
    })


def selected_checkpoint() -> Path:
    review = json.loads((campaign.OUT / "full_review.json").read_text())
    arm = review["best_candidate"]
    if arm not in ("full_core", "full_ordinary", "full_challenge"):
        raise RuntimeError("full review did not select a valid candidate")
    return campaign.OUT / "train" / arm / "stage3_step_4000.pt"


def save_new(path: Path, value: torch.Tensor):
    if path.exists():
        raise RuntimeError(f"source image already exists: {path}")
    st._save_frame(path, value)


def generate():
    protocol = json.loads((OUT / "view_protocol.json").read_text())
    indices = protocol["nested_indices"]
    config = load_stage3_config(campaign.config_path("full"))
    candidate = selected_checkpoint()
    if not candidate.is_file() or not A5_CHECKPOINT.is_file():
        raise FileNotFoundError("3DGS source checkpoint absent")
    # All source images use a frozen 4-view context. The later 3DGS count does
    # not change a target's SR input, noise, checkpoint, or predicted pixels.
    for index in indices:
        context = protocol["sr_contexts"][str(index)]
        _, hr, lr, _ = st._load_indices(
            experiment.RUNTIME_ROOT / "datasets/nerf_synthetic", "lego",
            "train", context, config, torch.device("cpu"),
        )
        save_new(OUT / "source/hr" / f"view_{index:03d}.png", hr[:, :, :1])
        save_new(OUT / "source/lr" / f"view_{index:03d}.png", lr[:, :, :1])
        bicubic = F.interpolate(lr[:, :, 0], size=(256, 256),
                                mode="bicubic", align_corners=False).unsqueeze(2).clamp(-1, 1)
        save_new(OUT / "source/bicubic" / f"view_{index:03d}.png", bicubic)
    for arm, config_path, checkpoint in (
        ("a5", A5_CONFIG, A5_CHECKPOINT),
        ("candidate", campaign.config_path("full"), candidate),
    ):
        arm_config = load_stage3_config(config_path)
        st._seed_all(3302)
        model = experiment.runtime(arm_config, checkpoint)
        model.module.eval()
        model.dit.model.eval().requires_grad_(False)
        model.vae.model.model.eval().requires_grad_(False)
        for index in indices:
            context = protocol["sr_contexts"][str(index)]
            _, hr, lr, camera = st._load_indices(
                experiment.RUNTIME_ROOT / "datasets/nerf_synthetic", "lego",
                "train", context, arm_config, model.device,
            )
            with torch.inference_mode():
                clean = model.vae.encode_multiview(hr)
                latent = st.sample_latents(
                    model, lr, camera, tuple(clean.shape), arm_config.sampling_steps,
                    arm_config.image_size, seed=3302 * 1_000_000 + index,
                    sampling_shift=arm_config.sampling_shift, dtype=torch.bfloat16,
                )
                decoded = model.vae.decode_multiview(latent)[:, :, :1]
            save_new(OUT / "source" / arm / f"view_{index:03d}.png", decoded)
        del model
        torch.cuda.empty_cache()
    image_hashes = {
        arm: {str(index): sha256_file(OUT / "source" / arm / f"view_{index:03d}.png")
              for index in indices}
        for arm in (*ARMS, "lr")
    }
    frozen(OUT / "source_manifest.json", {
        "view_protocol_sha256": sha256_file(OUT / "view_protocol.json"),
        "candidate_checkpoint_sha256": sha256_file(candidate),
        "a5_checkpoint_sha256": sha256_file(A5_CHECKPOINT),
        "candidate_config_sha256": sha256_file(campaign.config_path("full")),
        "a5_config_sha256": sha256_file(A5_CONFIG),
        "images": image_hashes,
    })


def yaml_config(count: int) -> Path:
    path = OUT / f"count_{count}" / "configs" / "lego.yml"
    staged = OUT / f"count_{count}" / "staged" / "lego"
    content = (
        f"hr_source_dir: {staged}\n"
        f"lr_source_dir: {staged}\n"
        f"vsr_save_dir: {staged}\n"
        "video_save_path: null\nvsr_model: psrt\nspynet_path: null\n"
        "vsr_model_path: null\ndownscale_factor: 4\nsubpixel: bicubic\n"
        "white_background: true\nals: false\n"
        f"num_images_in_sequence: {count}\nsimilarity: feature\n"
        "thres_values: [45]\nlambda_tex: 0.60\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError("SequenceMatters YAML drift")
    if not path.exists():
        path.write_text(content)
    return path


def stage():
    protocol = json.loads((OUT / "view_protocol.json").read_text())
    source = json.loads((OUT / "source_manifest.json").read_text())
    if source["view_protocol_sha256"] != sha256_file(OUT / "view_protocol.json"):
        raise RuntimeError("3DGS view protocol drift")
    shared_ply = OUT / "shared_points/lego.ply"
    ply_hash = write_deterministic_ply(shared_ply, seed=2201)
    for count in COUNTS:
        yaml = yaml_config(count)
        indices = tuple(protocol["sets"][str(count)])
        summaries = []
        for arm in ARMS:
            target = OUT / f"count_{count}" / "staged" / "lego" / arm
            train_images = {index: OUT / "source" / arm / f"view_{index:03d}.png"
                            for index in indices}
            lr_images = {index: OUT / "source/lr" / f"view_{index:03d}.png"
                         for index in indices}
            for index in indices:
                if (sha256_file(train_images[index]) != source["images"][arm][str(index)]
                        or sha256_file(lr_images[index]) != source["images"]["lr"][str(index)]):
                    raise RuntimeError("3DGS source image drift")
            if target.exists():
                saved = json.loads((target / "staging_manifest.json").read_text())
                names = tuple(saved["train_names"])
                validate_staged_blender(
                    target, train_names=names,
                    test_names=tuple(sorted(p.name for p in (target / "test").glob("*.png"))),
                    image_size=256, lr_size=64, expected_ply_sha256=ply_hash,
                )
                if saved["train_indices"] != list(indices):
                    raise RuntimeError("staged 3DGS indices changed")
            else:
                saved = stage_blender_dataset(
                    target, source_scene=DATASET,
                    train_images=train_images, train_lr_images=lr_images,
                    train_indices=indices, shared_ply=shared_ply,
                )
            summaries.append({"arm": arm, "manifest_sha256": sha256_file(target / "staging_manifest.json"),
                              "train_count": saved["validation"]["train_count"]})
        frozen(OUT / f"count_{count}" / "staging_summary.json", {
            "count": count, "indices": list(indices), "yaml_sha256": sha256_file(yaml),
            "shared_ply_sha256": ply_hash, "arms": summaries,
        })


def run_command(argv: list[str], *, gpu: int, log: Path):
    campaign.gpu_free(gpu)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
           "PYTHONPATH": str(campaign.ROOT / "src")}
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.run(argv, cwd=SEQUENCE, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"SequenceMatters failed ({process.returncode}): {log}")


def render_metrics(model: Path, count: int, arm: str, gpu: int):
    # Match the existing SequenceMatters validation implementation and metric
    # models; all arms use the same held-out 200 test cameras.
    import torchvision.transforms.functional as tf
    sys.path.insert(0, str(SEQUENCE))
    from lpipsPyTorch.modules.lpips import LPIPS
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    criterion = LPIPS("vgg").cuda().eval()
    rows = []
    for split, expected in (("train", count), ("test", 200)):
        root = model / split / "ours_30000"
        renders = sorted((root / "renders").glob("*.png"))
        if len(renders) != expected:
            raise RuntimeError(f"{arm}/{count} {split} render count {len(renders)} != {expected}")
        for path in renders:
            with torch.inference_mode():
                prediction = tf.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).cuda()
                target = tf.to_tensor(Image.open(root / "gt" / path.name).convert("RGB")).unsqueeze(0).cuda()
                rows.append({"count": count, "arm": arm, "split": split,
                             "view": path.stem, "psnr": float(psnr(prediction, target)),
                             "ssim": float(ssim(prediction, target)),
                             "lpips": float(criterion(prediction, target))})
    with (model / "rl3dsr_per_view_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frozen(model / "rl3dsr_metrics.json", {
        split: {key: float(np.mean([row[key] for row in rows if row["split"] == split]))
                for key in ("psnr", "ssim", "lpips")}
        for split in ("train", "test")
    })


def one(count: int, arm: str, gpu: int):
    base = OUT / f"count_{count}"
    staged = base / "staged" / "lego" / arm
    model = base / "models" / "lego" / arm
    yaml = base / "configs" / "lego.yml"
    complete = model / "point_cloud/iteration_30000/point_cloud.ply"
    if not complete.exists():
        if model.exists() and any(model.iterdir()):
            raise RuntimeError(f"partial 3DGS model requires audit: {model}")
        run_command([
            str(SEQUENCE_PYTHON), "train.py", "-s", str(staged), "-m", str(model),
            "--config", str(yaml), "--eval", "--skip_vsr",
            "--precomputed_vsr_dir", str(staged / "train"),
            "--iterations", "30000", "--test_iterations", "7000", "30000",
            "--save_iterations", "7000", "30000", "--port", str(6009 + gpu),
        ], gpu=gpu, log=base / "logs" / f"train_{arm}.log")
    if not complete.is_file():
        raise RuntimeError(f"3DGS model incomplete: {arm}/{count}")
    test = model / "test/ours_30000/renders"
    train = model / "train/ours_30000/renders"
    if not test.exists() and not train.exists():
        run_command([
            str(SEQUENCE_PYTHON), "render_samename.py", "-m", str(model),
            "--config", str(yaml), "--iteration", "30000",
        ], gpu=gpu, log=base / "logs" / f"render_{arm}.log")
    if len(tuple(test.glob("*.png"))) != 200 or len(tuple(train.glob("*.png"))) != count:
        raise RuntimeError(f"3DGS render incomplete: {arm}/{count}")
    if not (model / "rl3dsr_metrics.json").is_file():
        run_command([
            str(sys.executable), str(Path(__file__).resolve()), "metrics-one",
            "--count", str(count), "--arm", arm,
        ], gpu=gpu, log=base / "logs" / f"metrics_{arm}.log")
    print(json.dumps({"complete": f"{count}/{arm}", "gpu": gpu}), flush=True)


def run():
    jobs = [(count, arm) for count in COUNTS for arm in ARMS]
    queues = {campaign.GPU_IDS[0]: jobs[::2], campaign.GPU_IDS[1]: jobs[1::2]}
    def worker(gpu: int):
        for count, arm in queues[gpu]:
            one(count, arm, gpu)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in campaign.GPU_IDS]
        for future in as_completed(futures):
            future.result()


def review():
    rows = []
    for count in COUNTS:
        for arm in ARMS:
            path = OUT / f"count_{count}" / "models/lego" / arm / "rl3dsr_metrics.json"
            payload = json.loads(path.read_text())
            rows.append({"count": count, "arm": arm, **payload["test"]})
    by = {(row["count"], row["arm"]): row for row in rows}
    comparisons = {str(count): {
        f"candidate_minus_{arm}": {
            metric: by[(count, "candidate")][metric] - by[(count, arm)][metric]
            for metric in ("psnr", "ssim", "lpips")
        } for arm in ("a5", "bicubic", "hr")
    } for count in COUNTS}
    trends = {arm: {
        f"{earlier}_to_{later}": {
            metric: by[(later, arm)][metric] - by[(earlier, arm)][metric]
            for metric in ("psnr", "ssim", "lpips")
        } for earlier, later in ((4, 8), (8, 16))
    } for arm in ARMS}
    frozen(OUT / "review.json", {"rows": rows, "comparisons": comparisons,
                                 "view_count_trends": trends,
                                 "scope": "Lego 200 held-out NVS cameras; downstream diagnostic"})
    with (OUT / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for count in COUNTS:
        for view in (0, 50, 100, 150):
            name = f"r_{view}.png"
            reference = np.asarray(Image.open(
                OUT / f"count_{count}" / "models/lego/hr/test/ours_30000/gt" / name
            ).convert("RGB"), dtype=np.float32)
            sheet = Image.new("RGB", (256 * (len(ARMS) + 1), 280), "white")
            for column, arm in enumerate(("target", *ARMS)):
                if arm == "target":
                    pixels = reference.astype(np.uint8)
                else:
                    pixels = np.asarray(Image.open(
                        OUT / f"count_{count}" / "models/lego" / arm
                        / "test/ours_30000/renders" / name
                    ).convert("RGB"))
                    error = np.abs(pixels.astype(np.float32) - reference).mean(-1)
                    heat = np.stack((np.clip(error * 4, 0, 255),
                                     np.clip(error * 2, 0, 255),
                                     np.zeros_like(error)), -1).astype(np.uint8)
                    error_path = OUT / f"count_{count}" / "analysis" / f"error_{arm}_{name}"
                    error_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(heat).save(error_path)
                panel = Image.new("RGB", (256, 280), "white")
                panel.paste(Image.fromarray(pixels), (0, 24))
                ImageDraw.Draw(panel).text((6, 5), arm, fill="black")
                sheet.paste(panel, (256 * column, 0))
            path = OUT / f"count_{count}" / "analysis" / f"contact_{name}"
            sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "generate", "stage", "run", "review", "metrics-one"))
    parser.add_argument("--count", type=int, choices=COUNTS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.command == "metrics-one":
        if args.count is None or args.arm is None:
            raise ValueError("metrics-one needs --count and --arm")
        render_metrics(OUT / f"count_{args.count}" / "models/lego" / args.arm,
                       args.count, args.arm, 0)
    else:
        {"prepare": prepare, "generate": generate, "stage": stage,
         "run": run, "review": review}[args.command]()


if __name__ == "__main__":
    main()
