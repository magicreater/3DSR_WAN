#!/usr/bin/env python3
"""Run the full-RRE four-view memorization campaign without held-out selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from rl3dsr.validation.memorization import memorization_verdict, select_seen_checkpoint


SCENES = ("chair", "lego", "drums")
TRAINING_SEEDS = (42, 43)
INFERENCE_SEEDS = (2201, 2202, 2203, 2204)
VIEW_INDICES = (0, 33, 66, 99)
FULL_STEPS = tuple(range(250, 2001, 250))
SIMPLE_STEPS = (500, 1000, 1500, 2000)
UCPE_REVISION = "d992f1807803ba99331e807e8f018ed552886afd"
PROPE_REVISION = "4c11297761225d25258e5ec21c61e9b19ab61e38"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(command: list[str], *, cwd: Path, gpu: int, log: Path) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(cwd / "src") + os.pathsep + environment.get("PYTHONPATH", "")
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


def _required(args) -> None:
    for name in (
        "model_dir",
        "dataset_root",
        "lq_source",
        "lq_checkpoint",
        "parent_checkpoint",
        "simple_rre_root",
    ):
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {path}")


def _protocol(args) -> dict:
    scenes = {}
    for scene in SCENES:
        root = args.dataset_root / scene
        train = root / "transforms_train.json"
        test = root / "transforms_test.json"
        payload = json.loads(train.read_text(encoding="utf-8"))
        if len(payload.get("frames", ())) <= max(VIEW_INDICES):
            raise RuntimeError(f"scene {scene} does not contain the fixed training views")
        scenes[scene] = {
            "root": str(root.resolve()),
            "transforms_train_sha256": _sha256(train),
            "transforms_test_sha256": _sha256(test),
            "train_view_indices": list(VIEW_INDICES),
        }
    sequence_files = {}
    for name in ("train.py", "render_samename.py", "vsr/precomputed_vsr.py"):
        path = args.sequence_matters / name
        sequence_files[name] = _sha256(path)
    protocol = {
        "schema_version": 1,
        "objective": "four-view HR memorization; no Wan held-out generalization claim",
        "artifact_root": str(args.output_root.resolve()),
        "training_seeds": list(TRAINING_SEEDS),
        "development_inference_seed": INFERENCE_SEEDS[0],
        "final_inference_seeds": list(INFERENCE_SEEDS),
        "candidate_steps": list(FULL_STEPS),
        "selection": "seen 50-step pure-noise PSNR, SSIM, LPIPS, earlier step",
        "scenes": scenes,
        "full_rre": {
            "method": "relray_absmap",
            "adaptation": "parallel",
            "branch_count": 30,
            "compression": 8,
            "hidden_dim": 192,
            "attention_heads": 1,
            "world_up": [0.0, 0.0, 1.0],
            "ucpe_revision": UCPE_REVISION,
            "prope_revision": PROPE_REVISION,
        },
        "sequence_matters": {
            "path": str(args.sequence_matters.resolve()),
            "git_revision": subprocess.check_output(
                ["git", "-C", str(args.sequence_matters), "rev-parse", "HEAD"], text=True
            ).strip(),
            "files": sequence_files,
        },
    }
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    protocol["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    path = args.output_root / "protocol_manifest.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != protocol:
            raise RuntimeError("memorization protocol or source data drift detected")
    else:
        _atomic_json(path, protocol)
    return protocol


def _resume(train_dir: Path) -> Path | None:
    checkpoints = []
    for path in train_dir.glob("geometry_step_*.pt"):
        if path.name.endswith("_training_state.pt"):
            continue
        if path.with_name(path.stem + "_training_state.pt").is_file():
            checkpoints.append(path)
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[1]), default=None)


def _train(args, scene: str, seed: int, *, steps: int = 2000, smoke: bool = False) -> Path:
    root = args.output_root / ("smoke" if smoke else scene) / f"full_rre_seed_{seed}" / "train"
    result = root / "result.json"
    final = root / "geometry_final.pt"
    if result.is_file() and final.is_file():
        return final
    root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/stage2_experiment.py",
        "--model-dir", str(args.model_dir),
        "--scene", str(args.dataset_root / scene),
        "--lq-source", str(args.lq_source),
        "--lq-checkpoint", str(args.lq_checkpoint),
        "--parent-checkpoint", str(args.parent_checkpoint),
        "--output-dir", str(root),
        "--representation", "rre_full",
        "--seed", str(seed),
        "--steps", str(steps),
        "--eval-interval", str(steps if smoke else 50),
        "--checkpoint-interval", str(steps if smoke else 250),
        "--learning-rate", "0.0001",
        "--view-indices", *map(str, VIEW_INDICES),
        "--device", "cuda",
    ]
    if smoke:
        command.append("--no-decoded-eval")
    resume = _resume(root)
    if resume is not None:
        command.extend(("--resume", str(resume)))
    elif any(root.iterdir()):
        raise RuntimeError(f"non-empty training directory is not resumable: {root}")
    _run(
        command,
        cwd=args.repo,
        gpu=args.gpu,
        log=args.output_root / "logs" / f"train_{'smoke_' if smoke else ''}{scene}_seed_{seed}.log",
    )
    if not final.is_file() or not result.is_file():
        raise RuntimeError(f"training did not complete: {root}")
    return final


def _evaluation_complete(
    path: Path,
    step: int,
    seeds: tuple[int, ...],
    representation: str,
    checkpoint_sha256: str | None,
) -> bool:
    summary = path / "summary.json"
    if not summary.is_file():
        return False
    data = json.loads(summary.read_text(encoding="utf-8"))
    return (
        int(data.get("checkpoint_step", -1)) == step
        and data.get("inference_seeds") == list(seeds)
        and data.get("representation") == representation
        and data.get("checkpoint_sha256") == checkpoint_sha256
    )


def _evaluate(
    args,
    *,
    scene: str,
    target: Path,
    representation: str,
    step: int,
    seeds: tuple[int, ...],
    checkpoint: Path | None,
    sampling_steps: int = 50,
    save_images: bool = False,
    pure_noise_controls: bool = False,
) -> None:
    checkpoint_sha256 = _sha256(checkpoint) if checkpoint is not None else None
    if _evaluation_complete(target, step, seeds, representation, checkpoint_sha256):
        return
    if target.exists():
        raise RuntimeError(f"refusing to overwrite incomplete evaluation: {target}")
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    command = [
        sys.executable,
        "scripts/stage2_memorization_eval.py",
        "--model-dir", str(args.model_dir),
        "--scene", str(args.dataset_root / scene),
        "--lq-source", str(args.lq_source),
        "--lq-checkpoint", str(args.lq_checkpoint),
        "--parent-checkpoint", str(args.parent_checkpoint),
        "--representation", representation,
        "--checkpoint-step", str(step),
        "--view-indices", *map(str, VIEW_INDICES),
        "--inference-seeds", *map(str, seeds),
        "--sampling-steps", str(sampling_steps),
        "--output-dir", str(temporary),
        "--device", "cuda",
    ]
    if checkpoint is not None:
        command.extend(("--geometry-checkpoint", str(checkpoint)))
    if save_images:
        command.append("--save-images")
    if pure_noise_controls:
        command.append("--pure-noise-camera-controls")
    _run(
        command,
        cwd=args.repo,
        gpu=args.gpu,
        log=args.output_root / "logs" / f"eval_{scene}_{representation}_{step:04d}_{'_'.join(map(str, seeds))}.log",
    )
    if not _evaluation_complete(temporary, step, seeds, representation, checkpoint_sha256):
        raise RuntimeError(f"evaluation did not finish: {temporary}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(target)


def _search(args) -> None:
    if (args.output_root / "frozen_candidates.json").exists():
        raise RuntimeError("candidate selection is already frozen")
    _protocol(args)
    candidates = []
    for scene in SCENES:
        for seed in TRAINING_SEEDS:
            _train(args, scene, seed)
            combined = []
            for step in FULL_STEPS:
                checkpoint = args.output_root / scene / f"full_rre_seed_{seed}" / "train" / f"geometry_step_{step:04d}.pt"
                target = args.output_root / scene / f"full_rre_seed_{seed}" / "development" / f"step_{step:04d}"
                _evaluate(
                    args,
                    scene=scene,
                    target=target,
                    representation="rre_full",
                    step=step,
                    seeds=(INFERENCE_SEEDS[0],),
                    checkpoint=checkpoint,
                )
                combined.extend(_jsonl(target / "metric_rows.jsonl"))
            selected = select_seen_checkpoint(combined)["selected"]
            checkpoint = args.output_root / scene / f"full_rre_seed_{seed}" / "train" / f"geometry_step_{selected['step']:04d}.pt"
            candidates.append({
                "scene": scene,
                "arm": f"full_rre_seed_{seed}",
                "training_seed": seed,
                "representation": "rre_full",
                "checkpoint_step": selected["step"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "development_metrics": selected,
            })

        combined = []
        for step in SIMPLE_STEPS:
            checkpoint = args.simple_rre_root / scene / "rre_geometry" / "train" / f"geometry_step_{step:04d}.pt"
            target = args.output_root / scene / "simple_rre_seed_42" / "development" / f"step_{step:04d}"
            _evaluate(
                args,
                scene=scene,
                target=target,
                representation="rre",
                step=step,
                seeds=(INFERENCE_SEEDS[0],),
                checkpoint=checkpoint,
            )
            combined.extend(_jsonl(target / "metric_rows.jsonl"))
        selected = select_seen_checkpoint(combined)["selected"]
        checkpoint = args.simple_rre_root / scene / "rre_geometry" / "train" / f"geometry_step_{selected['step']:04d}.pt"
        candidates.append({
            "scene": scene,
            "arm": "simple_rre_seed_42",
            "training_seed": 42,
            "representation": "rre",
            "checkpoint_step": selected["step"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "development_metrics": selected,
        })
    manifest = {
        "schema_version": 1,
        "protocol_sha256": json.loads((args.output_root / "protocol_manifest.json").read_text())["protocol_sha256"],
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "parent_checkpoint_sha256": _sha256(args.parent_checkpoint),
        "selection_uses": "seen views only; fixed inference seed 2201",
        "candidates": candidates,
        "frozen_unix": time.time(),
    }
    _atomic_json(args.output_root / "frozen_candidates.json", manifest)


def _load_frozen(args) -> dict:
    path = args.output_root / "frozen_candidates.json"
    if not path.is_file():
        raise RuntimeError("final evaluation requires frozen_candidates.json")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    protocol = _protocol(args)
    if frozen.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("frozen candidate protocol mismatch")
    if frozen.get("parent_checkpoint_sha256") != _sha256(args.parent_checkpoint):
        raise RuntimeError("parent checkpoint changed after selection")
    expected = {(scene, arm) for scene in SCENES for arm in ("simple_rre_seed_42", "full_rre_seed_42", "full_rre_seed_43")}
    actual = {(row["scene"], row["arm"]) for row in frozen.get("candidates", [])}
    if actual != expected:
        raise RuntimeError("frozen candidate set is incomplete")
    for row in frozen["candidates"]:
        path = Path(row["checkpoint"])
        if not path.is_file() or _sha256(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"frozen checkpoint changed: {path}")
    return frozen


def _final(args) -> None:
    complete = args.output_root / "final_evaluation_complete.json"
    if complete.exists():
        raise RuntimeError("final evaluation is already complete")
    frozen = _load_frozen(args)
    candidates = {(row["scene"], row["arm"]): row for row in frozen["candidates"]}
    verdicts = []
    for scene in SCENES:
        baseline = args.output_root / scene / "stage1" / "final"
        _evaluate(
            args,
            scene=scene,
            target=baseline,
            representation="stage1",
            step=0,
            seeds=INFERENCE_SEEDS,
            checkpoint=None,
            save_images=True,
        )
        baseline_rows = _jsonl(baseline / "metric_rows.jsonl")
        for arm in ("simple_rre_seed_42", "full_rre_seed_42", "full_rre_seed_43"):
            candidate = candidates[(scene, arm)]
            target = args.output_root / scene / arm / "final"
            _evaluate(
                args,
                scene=scene,
                target=target,
                representation=candidate["representation"],
                step=int(candidate["checkpoint_step"]),
                seeds=INFERENCE_SEEDS,
                checkpoint=Path(candidate["checkpoint"]),
                save_images=True,
                pure_noise_controls=True,
            )
            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            row = {
                "scene": scene,
                "arm": arm,
                "checkpoint_step": candidate["checkpoint_step"],
                "camera_usage": summary["camera_usage"],
            }
            if arm.startswith("full_rre"):
                row.update(memorization_verdict(_jsonl(target / "metric_rows.jsonl") + baseline_rows))
            verdicts.append(row)
    payload = {
        "schema_version": 1,
        "inference_seeds": list(INFERENCE_SEEDS),
        "verdicts": verdicts,
        "completed_unix": time.time(),
    }
    _atomic_json(args.output_root / "memorization_verdicts.json", payload)
    _atomic_json(
        complete,
        {
            "frozen_candidates_sha256": _sha256(args.output_root / "frozen_candidates.json"),
            "memorization_verdicts_sha256": _sha256(args.output_root / "memorization_verdicts.json"),
            "completed_unix": time.time(),
        },
    )


def _smoke(args) -> None:
    _protocol(args)
    checkpoint = _train(args, "chair", 42, steps=4, smoke=True)
    _evaluate(
        args,
        scene="chair",
        target=args.output_root / "smoke" / "full_rre_seed_42" / "eval",
        representation="rre_full",
        step=4,
        seeds=(INFERENCE_SEEDS[0],),
        checkpoint=checkpoint,
        sampling_steps=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "smoke", "search", "final", "all"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-dir", type=Path, default=Path("models/Wan2.1-T2V-1.3B"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/nerf_synthetic"))
    parser.add_argument("--lq-source", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/diffsynth/models/wan_video_dit.py"))
    parser.add_argument("--lq-checkpoint", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt"))
    parser.add_argument("--parent-checkpoint", type=Path, default=Path("artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt"))
    parser.add_argument("--simple-rre-root", type=Path, default=Path("artifacts/stage2_near_select_20260903"))
    parser.add_argument("--sequence-matters", type=Path, default=Path("/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/SequenceMatters"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/stage2_full_rre_memorization_20260904"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    for name in ("model_dir", "dataset_root", "lq_source", "lq_checkpoint", "parent_checkpoint", "simple_rre_root", "sequence_matters", "output_root"):
        path = getattr(args, name)
        if not path.is_absolute():
            path = args.repo / path
        setattr(args, name, path.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    _required(args)
    if args.mode == "preflight":
        print(json.dumps(_protocol(args), indent=2))
    elif args.mode == "smoke":
        _smoke(args)
    elif args.mode == "search":
        _search(args)
    elif args.mode == "final":
        _final(args)
    else:
        _search(args)
        _final(args)


if __name__ == "__main__":
    main()
