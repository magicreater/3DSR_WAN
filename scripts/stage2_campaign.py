#!/usr/bin/env python3
"""Run the Stage 2 seen/near selection and frozen far-test protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from rl3dsr.validation.stage2_protocol import (
    ARMS,
    CANDIDATE_STEPS,
    GEOMETRY_ARMS,
    SCENES,
    build_protocol,
    protocol_digest,
    select_checkpoint,
    sha256_file,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _run(command: list[str], cwd: Path, *, gpu: int, log_path: Path) -> None:
    print("+", " ".join(str(value) for value in command), flush=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(cwd / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
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
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"command failed with {returncode}; see {log_path}")


def _required_paths(args) -> None:
    for name in (
        "model_dir",
        "dataset_root",
        "lq_source",
        "lq_checkpoint",
        "parent_checkpoint",
    ):
        path = getattr(args, name)
        if path is None or not path.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} is required and must exist")


def _ensure_protocol(args) -> dict:
    current = build_protocol(args.dataset_root)
    path = args.output_root / "protocol_manifest.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != current or saved.get("protocol_sha256") != protocol_digest(saved):
            raise RuntimeError("dataset or protocol drift detected")
        return saved
    args.output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, current)
    return current


def _base_command(args, scene_root: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/stage2_eval.py",
        "--model-dir",
        str(args.model_dir),
        "--scene",
        str(scene_root),
        "--lq-source",
        str(args.lq_source),
        "--lq-checkpoint",
        str(args.lq_checkpoint),
        "--parent-checkpoint",
        str(args.parent_checkpoint),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda",
    ]


def _evaluation_complete(path: Path, phase: str, checkpoint_step: int) -> bool:
    summary = path / "eval_summary.json"
    if not summary.is_file():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return payload.get("phase") == phase and int(payload.get("checkpoint_step", -1)) == checkpoint_step


def _evaluate(
    args,
    *,
    target: Path,
    scene_root: Path,
    phase: str,
    checkpoint_step: int,
    protocol_scene: dict,
    arm: str,
    checkpoint: Path | None,
) -> None:
    if _evaluation_complete(target, phase, checkpoint_step):
        return
    if target.exists():
        raise RuntimeError(f"refusing to overwrite incomplete evaluation: {target}")
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    command = _base_command(args, scene_root, temporary)
    command.extend(["--phase", phase, "--checkpoint-step", str(checkpoint_step)])
    if phase == "selection":
        command.extend([
            "--seen-view-indices",
            *map(str, protocol_scene["seen_view_indices"]),
            "--near-view-indices",
            *map(str, protocol_scene["near_view_indices"]),
        ])
    else:
        command.extend([
            "--far-view-indices",
            *map(str, protocol_scene["far_view_indices"]),
        ])
        if args.depth_manifest_root is not None:
            depth_manifest = args.depth_manifest_root / f"{scene_root.name}.json"
            if depth_manifest.is_file():
                command.extend(["--depth-manifest", str(depth_manifest)])
    if checkpoint is not None:
        representation = "rre" if arm == "rre_geometry" else "plucker"
        command.extend([
            "--geometry-checkpoint",
            str(checkpoint),
            "--representation",
            representation,
        ])
    log_path = args.output_root / "logs" / (
        f"{phase}_{scene_root.name}_{arm}_{checkpoint_step:04d}.log"
    )
    _run(command, args.repo, gpu=args.gpu, log_path=log_path)
    if not _evaluation_complete(temporary, phase, checkpoint_step):
        raise RuntimeError(f"evaluation did not produce a valid summary: {temporary}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(target)


def _latest_resume(train_dir: Path) -> Path | None:
    candidates = []
    for checkpoint in train_dir.glob("geometry_step_*.pt"):
        if checkpoint.name.endswith("_training_state.pt"):
            continue
        state = checkpoint.with_name(checkpoint.stem + "_training_state.pt")
        if state.is_file():
            candidates.append(checkpoint)
    return max(
        candidates,
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        default=None,
    )


def _train(args, scene: str, arm: str, protocol_scene: dict) -> Path:
    train_dir = args.output_root / scene / arm / "train"
    final = train_dir / "geometry_final.pt"
    result = train_dir / "result.json"
    if final.is_file() and result.is_file():
        return final
    train_dir.mkdir(parents=True, exist_ok=True)
    representation = "rre" if arm == "rre_geometry" else "plucker"
    command = [
        sys.executable,
        "scripts/stage2_experiment.py",
        "--model-dir",
        str(args.model_dir),
        "--scene",
        protocol_scene["scene_root"],
        "--lq-source",
        str(args.lq_source),
        "--lq-checkpoint",
        str(args.lq_checkpoint),
        "--parent-checkpoint",
        str(args.parent_checkpoint),
        "--output-dir",
        str(train_dir),
        "--representation",
        representation,
        "--view-indices",
        *map(str, protocol_scene["seen_view_indices"]),
        "--steps",
        str(args.steps),
        "--eval-interval",
        "50",
        "--checkpoint-interval",
        "500",
        "--device",
        "cuda",
    ]
    resume = _latest_resume(train_dir)
    if resume is not None:
        command.extend(["--resume", str(resume)])
    elif any(train_dir.iterdir()):
        raise RuntimeError(
            f"training directory is nonempty without a resumable checkpoint: {train_dir}"
        )
    log_path = args.output_root / "logs" / f"train_{scene}_{arm}.log"
    _run(command, args.repo, gpu=args.gpu, log_path=log_path)
    expected_steps = CANDIDATE_STEPS if args.steps == 2000 else (args.steps,)
    for step in expected_steps:
        checkpoint = train_dir / f"geometry_step_{step:04d}.pt"
        state = train_dir / f"geometry_step_{step:04d}_training_state.pt"
        if not checkpoint.is_file() or not state.is_file():
            raise RuntimeError(f"missing retained checkpoint/state at step {step}: {train_dir}")
    if not final.is_file() or not result.is_file():
        raise RuntimeError(f"training did not complete: {train_dir}")
    return final


def _analyze(args) -> None:
    command = [
        sys.executable,
        "scripts/stage2_analysis.py",
        "--campaign-root",
        str(args.output_root),
    ]
    _run(
        command,
        args.repo,
        gpu=args.gpu,
        log_path=args.output_root / "logs" / "analysis.log",
    )


def _search(args) -> None:
    if args.steps != 2000:
        raise ValueError("the registered search protocol requires exactly 2000 steps")
    if (args.output_root / "final_test").exists():
        raise RuntimeError("search must not run after far-test output exists")
    if (args.output_root / "frozen_candidates.json").exists():
        raise RuntimeError("candidate selection is already frozen")
    protocol = _ensure_protocol(args)
    parent_hash = sha256_file(args.parent_checkpoint)
    candidates = []
    audit_rows = []
    for scene in SCENES:
        protocol_scene = protocol["scenes"][scene]
        scene_root = Path(protocol_scene["scene_root"])
        _evaluate(
            args,
            target=(
                args.output_root
                / scene
                / "baseline_no_geometry"
                / "selection"
                / "baseline"
            ),
            scene_root=scene_root,
            phase="selection",
            checkpoint_step=0,
            protocol_scene=protocol_scene,
            arm="baseline_no_geometry",
            checkpoint=None,
        )
        for arm in GEOMETRY_ARMS:
            _train(args, scene, arm, protocol_scene)
            selection_rows = []
            for step in CANDIDATE_STEPS:
                checkpoint = (
                    args.output_root
                    / scene
                    / arm
                    / "train"
                    / f"geometry_step_{step:04d}.pt"
                )
                target = (
                    args.output_root
                    / scene
                    / arm
                    / "selection"
                    / f"step_{step:04d}"
                )
                _evaluate(
                    args,
                    target=target,
                    scene_root=scene_root,
                    phase="selection",
                    checkpoint_step=step,
                    protocol_scene=protocol_scene,
                    arm=arm,
                    checkpoint=checkpoint,
                )
                selection_rows.extend(_read_csv(target / "evaluation_rows.csv"))
            chosen, audit = select_checkpoint(
                selection_rows,
                protocol_scene["near_view_indices"],
            )
            checkpoint = (
                args.output_root
                / scene
                / arm
                / "train"
                / f"geometry_step_{chosen['step']:04d}.pt"
            )
            candidates.append({
                "scene": scene,
                "arm": arm,
                "representation": "rre" if arm == "rre_geometry" else "plucker",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "step": chosen["step"],
                "near_metrics": chosen,
            })
            audit_rows.extend({"scene": scene, "arm": arm, **row} for row in audit)
    frozen = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["protocol_sha256"],
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "parent_checkpoint_sha256": parent_hash,
        "selection_metric": protocol["selection"],
        "candidates": candidates,
        "frozen_unix": time.time(),
    }
    _write_csv(args.output_root / "checkpoint_selection_audit.csv", audit_rows)
    _atomic_json(args.output_root / "frozen_candidates.json", frozen)
    _analyze(args)


def _load_frozen(args, protocol: dict) -> dict:
    path = args.output_root / "frozen_candidates.json"
    if not path.is_file():
        raise RuntimeError("far test requires frozen_candidates.json")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("frozen candidate protocol hash mismatch")
    if frozen.get("parent_checkpoint_sha256") != sha256_file(args.parent_checkpoint):
        raise RuntimeError("parent checkpoint changed after selection")
    expected = {(scene, arm) for scene in SCENES for arm in GEOMETRY_ARMS}
    actual = {
        (row.get("scene"), row.get("arm"))
        for row in frozen.get("candidates", [])
    }
    if actual != expected:
        raise RuntimeError("frozen candidate set is incomplete")
    for row in frozen["candidates"]:
        checkpoint = Path(row["checkpoint"])
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != row["checkpoint_sha256"]
        ):
            raise RuntimeError(f"frozen checkpoint changed: {checkpoint}")
    return frozen


def _test(args) -> None:
    complete_path = args.output_root / "final_test_complete.json"
    if complete_path.exists():
        raise RuntimeError("final far test is already complete and cannot be rerun")
    protocol = _ensure_protocol(args)
    frozen = _load_frozen(args, protocol)
    by_key = {(row["scene"], row["arm"]): row for row in frozen["candidates"]}
    ledger_path = args.output_root / "final_test_ledger.json"
    ledger = (
        json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger_path.is_file()
        else {
            "protocol_sha256": protocol["protocol_sha256"],
            "frozen_candidates_sha256": sha256_file(
                args.output_root / "frozen_candidates.json"
            ),
            "completed": {},
        }
    )
    for scene in SCENES:
        protocol_scene = protocol["scenes"][scene]
        scene_root = Path(protocol_scene["scene_root"])
        for arm in ARMS:
            key = f"{scene}/{arm}"
            target = args.output_root / "final_test" / scene / arm
            if key in ledger["completed"]:
                if not _evaluation_complete(
                    target,
                    "final-test",
                    int(ledger["completed"][key]["checkpoint_step"]),
                ):
                    raise RuntimeError(f"ledger/output mismatch for {key}")
                continue
            if target.exists():
                raise RuntimeError(f"unregistered final-test output exists: {target}")
            candidate = by_key.get((scene, arm))
            checkpoint = None if candidate is None else Path(candidate["checkpoint"])
            checkpoint_step = 0 if candidate is None else int(candidate["step"])
            _evaluate(
                args,
                target=target,
                scene_root=scene_root,
                phase="final-test",
                checkpoint_step=checkpoint_step,
                protocol_scene=protocol_scene,
                arm=arm,
                checkpoint=checkpoint,
            )
            ledger["completed"][key] = {
                "checkpoint_step": checkpoint_step,
                "checkpoint_sha256": (
                    None if checkpoint is None else sha256_file(checkpoint)
                ),
                "completed_unix": time.time(),
            }
            _atomic_json(ledger_path, ledger)
    _atomic_json(
        complete_path,
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "frozen_candidates_sha256": sha256_file(
                args.output_root / "frozen_candidates.json"
            ),
            "completed_arms": sorted(ledger["completed"]),
            "completed_unix": time.time(),
            "historical_far_disclosure": protocol["historical_far_disclosure"],
        },
    )
    _analyze(args)


def _smoke(args) -> None:
    protocol = _ensure_protocol(args)
    scene = "chair"
    arm = "rre_geometry"
    protocol_scene = protocol["scenes"][scene]
    train_dir = args.output_root / scene / arm / "train"
    if train_dir.exists() and any(train_dir.iterdir()):
        raise RuntimeError("smoke output must be empty")
    original_steps = args.steps
    args.steps = 2
    try:
        _train(args, scene, arm, protocol_scene)
    finally:
        args.steps = original_steps
    checkpoint = train_dir / "geometry_step_0002.pt"
    _evaluate(
        args,
        target=args.output_root / scene / arm / "selection" / "step_0002",
        scene_root=Path(protocol_scene["scene_root"]),
        phase="selection",
        checkpoint_step=2,
        protocol_scene=protocol_scene,
        arm=arm,
        checkpoint=checkpoint,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("preflight", "smoke", "search", "test", "analyze"),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--lq-source", type=Path)
    parser.add_argument("--lq-checkpoint", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--depth-manifest-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.mode == "preflight":
        protocol = _ensure_protocol(args)
        print(json.dumps(protocol, indent=2))
        return
    if args.mode == "analyze":
        _analyze(args)
        return
    _required_paths(args)
    args.model_dir = args.model_dir.resolve()
    args.lq_source = args.lq_source.resolve()
    args.lq_checkpoint = args.lq_checkpoint.resolve()
    args.parent_checkpoint = args.parent_checkpoint.resolve()
    if args.depth_manifest_root is not None:
        args.depth_manifest_root = args.depth_manifest_root.resolve()
    if args.mode == "smoke":
        _smoke(args)
    elif args.mode == "search":
        _search(args)
    else:
        _test(args)


if __name__ == "__main__":
    main()
