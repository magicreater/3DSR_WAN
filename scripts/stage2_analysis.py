#!/usr/bin/env python3
"""Render Stage 2 training and geometry diagnostic plots from retained logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _campaign_rows(root: Path, name: str) -> list[dict]:
    rows = []
    for path in root.glob(f"*/*/eval/{name}"):
        parts = path.relative_to(root).parts
        if len(parts) < 4:
            continue
        scene, arm = parts[0], parts[1]
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"scene_id": scene, "arm": arm, **row})
    return rows


def _numeric_summary(rows: list[dict], group_keys: tuple[str, ...], value_keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in group_keys)].append(row)
    result = []
    for group, values in sorted(grouped.items()):
        output = dict(zip(group_keys, group))
        for key in value_keys:
            numeric = []
            for value in values:
                try:
                    parsed = float(value.get(key, "nan"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(parsed):
                    numeric.append(parsed)
            if numeric:
                output[f"{key}_mean"] = statistics.fmean(numeric)
                output[f"{key}_median"] = statistics.median(numeric)
                output[f"{key}_count"] = len(numeric)
            else:
                output[f"{key}_mean"] = ""
                output[f"{key}_median"] = ""
                output[f"{key}_count"] = 0
        result.append(output)
    return result


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _protocol_rows(root: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(root.glob(pattern)):
        parts = path.relative_to(root).parts
        if parts[0] == "final_test":
            scene, arm = parts[1], parts[2]
        else:
            scene, arm = parts[0], parts[1]
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(
                {"scene_id": scene, "arm": arm, **row}
                for row in csv.DictReader(handle)
            )
    return rows


def _add_header(fig, axis, title: str, subtitle: str) -> None:
    axis.set_title("")
    left = axis.get_position().x0
    fig.text(
        left,
        0.985,
        textwrap.fill(title, 78),
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color="#1F2430",
    )
    fig.text(
        left,
        0.945,
        textwrap.fill(subtitle, 112),
        ha="left",
        va="top",
        fontsize=9,
        color="#6F768A",
    )


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=180, facecolor="#FCFCFD")
    fig.savefig(path.with_suffix(".svg"), facecolor="#FCFCFD")


def _qualitative_contact_sheets(root: Path, frozen: dict, final_complete: bool) -> None:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    candidate_steps = {
        (row["scene"], row["arm"]): int(row["step"])
        for row in frozen["candidates"]
    }

    def load(scene: str, arm: str, group: str):
        if group == "far-held-out":
            path = root / "final_test" / scene / arm / f"qualitative_{group}.pt"
        elif arm == "baseline_no_geometry":
            path = (
                root
                / scene
                / arm
                / "selection"
                / "baseline"
                / f"qualitative_{group}.pt"
            )
        else:
            step = candidate_steps[(scene, arm)]
            path = (
                root
                / scene
                / arm
                / "selection"
                / f"step_{step:04d}"
                / f"qualitative_{group}.pt"
            )
        return torch.load(path, map_location="cpu", weights_only=True)

    def image_from_tensor(value):
        array = (
            value.detach()
            .float()
            .clamp(-1, 1)
            .add(1)
            .mul(127.5)
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )
        return Image.fromarray(np.ascontiguousarray(array), "RGB")

    def error_image(value, target):
        error = (value.detach().float() - target.detach().float()).abs().mean(dim=0)
        array = error.clamp(0, 1).mul(255).byte().numpy()
        return Image.fromarray(array, "L").convert("RGB")

    groups = ["seen", "near-held-out"]
    if final_complete:
        groups.append("far-held-out")
    output = root / "qualitative"
    output.mkdir(exist_ok=True)
    for scene in ("chair", "lego", "drums"):
        for group in groups:
            baseline = load(scene, "baseline_no_geometry", group)
            rre = load(scene, "rre_geometry", group)
            plucker = load(scene, "plucker_geometry", group)
            hr = baseline["hr"]
            rows = [
                ("HR target", hr, False),
                ("Bicubic", baseline["bicubic"], False),
                ("Stage 1 baseline", baseline["baseline"], False),
                ("RRE selected", rre["correct"], False),
                ("RRE absolute error", rre["correct"], True),
                ("Plucker selected", plucker["correct"], False),
                ("Plucker absolute error", plucker["correct"], True),
            ]
            tile_height, tile_width = hr.shape[-2:]
            header = 26
            canvas = Image.new(
                "RGB",
                (tile_width * hr.shape[2], (tile_height + header) * len(rows)),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            for row_index, (label, tensor, is_error) in enumerate(rows):
                for view_position, view_index in enumerate(baseline["view_indices"]):
                    value = tensor[0, :, view_position]
                    tile = (
                        error_image(value, hr[0, :, view_position])
                        if is_error
                        else image_from_tensor(value)
                    )
                    y = row_index * (tile_height + header)
                    canvas.paste(tile, (view_position * tile_width, y + header))
                    draw.text(
                        (view_position * tile_width + 4, y + 6),
                        f"{label} / view {view_index}",
                        fill="#1F2430",
                    )
            canvas.save(output / f"{scene}_{group}.png")


def _near_select_report(root: Path) -> None:
    protocol = json.loads(
        (root / "protocol_manifest.json").read_text(encoding="utf-8")
    )
    frozen = json.loads(
        (root / "frozen_candidates.json").read_text(encoding="utf-8")
    )
    final_complete = (root / "final_test_complete.json").is_file()
    selection_image = _protocol_rows(
        root, "*/*/selection/*/evaluation_rows.csv"
    )
    selection_pose = _protocol_rows(
        root, "*/*/selection/*/pose_sensitivity.csv"
    )
    final_image = (
        _protocol_rows(root, "final_test/*/*/evaluation_rows.csv")
        if final_complete
        else []
    )
    final_pose = (
        _protocol_rows(root, "final_test/*/*/pose_sensitivity.csv")
        if final_complete
        else []
    )
    final_geometry = (
        _protocol_rows(root, "final_test/*/*/geometry_rows.csv")
        if final_complete
        else []
    )
    _write(root / "selection_image_rows.csv", selection_image)
    _write(root / "selection_pose_rows.csv", selection_pose)
    if final_image:
        _write(root / "far_image_rows.csv", final_image)
        _write(root / "far_pose_rows.csv", final_pose)
    if final_geometry:
        _write(root / "far_geometry_rows.csv", final_geometry)
    else:
        (root / "far_geometry_rows.csv").write_text(
            "availability,reason\n"
            "unavailable,No validated metric depth manifest was supplied\n",
            encoding="utf-8",
        )

    selection_summary = _numeric_summary(
        selection_image,
        ("scene_id", "arm", "evaluation_group", "checkpoint_step", "condition"),
        ("mae", "psnr", "ssim", "lpips"),
    )
    final_summary = _numeric_summary(
        final_image,
        ("scene_id", "arm", "evaluation_group", "checkpoint_step", "condition"),
        ("mae", "psnr", "ssim", "lpips"),
    )
    pose_summary = _numeric_summary(
        selection_pose + final_pose,
        ("scene_id", "arm", "evaluation_group", "checkpoint_step"),
        (
            "correct_flow_loss",
            "shuffled_flow_loss",
            "baseline_flow_loss",
            "correct_shuffled_delta",
            "correct_disabled_delta",
            "pose_perturbation_delta",
            "pose_response_ratio",
        ),
    )
    _write(root / "selection_image_summary.csv", selection_summary)
    if final_summary:
        _write(root / "far_image_summary.csv", final_summary)
    _write(root / "pose_summary.csv", pose_summary)

    training_resources = []
    for scene in ("chair", "lego", "drums"):
        for arm in ("rre_geometry", "plucker_geometry"):
            train_dir = root / scene / arm / "train"
            rows = _read(train_dir / "train_steps.csv")
            has_step_timing = bool(rows) and all(
                row.get("step_seconds", "") != "" for row in rows
            )
            if has_step_timing:
                wall_seconds = sum(float(row["step_seconds"]) for row in rows)
                timing_source = "per_step_monotonic"
            else:
                wall_seconds = max(
                    0.0,
                    (train_dir / "result.json").stat().st_mtime
                    - (train_dir / "run_manifest.json").stat().st_mtime,
                )
                timing_source = "filesystem_mtime_boundary_estimate"
            training_resources.append(
                {
                    "scene_id": scene,
                    "arm": arm,
                    "steps": len(rows),
                    "loss_mean": statistics.fmean(float(row["loss"]) for row in rows),
                    "gradient_norm_median": statistics.median(
                        float(row["gradient_norm"]) for row in rows
                    ),
                    "sigma_min": min(float(row["sigma"]) for row in rows),
                    "sigma_max": max(float(row["sigma"]) for row in rows),
                    "peak_gpu_memory_mib": max(
                        float(row["peak_gpu_memory_mib"]) for row in rows
                    ),
                    "observed_run_wall_seconds": wall_seconds,
                    "mean_step_seconds": wall_seconds / len(rows),
                    "per_step_timing_available": has_step_timing,
                    "timing_source": timing_source,
                }
            )
    _write(root / "training_resource_summary.csv", training_resources)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.grid": True,
            "grid.color": "#E6E8F0",
            "text.color": "#1F2430",
        }
    )
    colors = {"rre_geometry": "#A3BEFA", "plucker_geometry": "#FFE15B"}
    for field, ylabel, output_name in (
        ("loss", "Flow loss", "training_loss_curves"),
        ("gradient_norm", "Gradient norm", "training_gradient_curves"),
        ("sigma", "Training sigma", "training_sigma_curves"),
        ("peak_gpu_memory_mib", "Peak GPU memory (MiB)", "training_memory_curves"),
    ):
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for axis, scene in zip(axes, ("chair", "lego", "drums")):
            for arm in ("rre_geometry", "plucker_geometry"):
                path = root / scene / arm / "train" / "train_steps.csv"
                rows = _read(path)
                axis.plot(
                    [int(row["step"]) for row in rows],
                    [float(row[field]) for row in rows],
                    color=colors[arm],
                    label=arm.replace("_geometry", ""),
                    linewidth=1.0,
                )
            axis.set_ylabel(f"{scene} {ylabel}")
            axis.grid(True, alpha=0.5)
            if scene == "chair":
                axis.legend(loc="upper right", frameon=False)
        axes[-1].set_xlabel("Training step")
        fig.subplots_adjust(top=0.89, hspace=0.18)
        _add_header(
            fig,
            axes[0],
            f"Stage 2 {ylabel.lower()} across six geometry runs",
            "Three NeRF Synthetic scenes, seed 42, 2000 steps; RRE and Plucker are shown with explicit colors.",
        )
        _save_figure(fig, root / output_name)
        plt.close(fig)

    selected_steps = {
        (row["scene"], row["arm"]): int(row["step"])
        for row in frozen["candidates"]
    }
    for group in ("seen", "near-held-out"):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
        for row_index, scene in enumerate(("chair", "lego", "drums")):
            for column_index, (metric, label) in enumerate(
                (("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("lpips", "LPIPS"))
            ):
                axis = axes[row_index, column_index]
                baseline = [
                    row
                    for row in selection_summary
                    if row["scene_id"] == scene
                    and row["arm"] == "baseline_no_geometry"
                    and row["evaluation_group"] == group
                    and row["condition"] == "correct"
                ][0]
                axis.axhline(
                    float(baseline[f"{metric}_mean"]),
                    color="#7A828F",
                    linestyle="--",
                    linewidth=1.0,
                    label="Stage 1 baseline",
                )
                for arm in ("rre_geometry", "plucker_geometry"):
                    rows = sorted(
                        (
                            row
                            for row in selection_summary
                            if row["scene_id"] == scene
                            and row["arm"] == arm
                            and row["evaluation_group"] == group
                            and row["condition"] == "correct"
                        ),
                        key=lambda row: int(row["checkpoint_step"]),
                    )
                    x = [int(row["checkpoint_step"]) for row in rows]
                    y = [float(row[f"{metric}_mean"]) for row in rows]
                    axis.plot(
                        x,
                        y,
                        marker="o",
                        color=colors[arm],
                        label=arm.replace("_geometry", ""),
                        linewidth=1.0,
                    )
                    selected = selected_steps[(scene, arm)]
                    selected_value = y[x.index(selected)]
                    axis.scatter(
                        [selected],
                        [selected_value],
                        s=80,
                        facecolors="none",
                        edgecolors="#1F2430",
                        linewidths=1.2,
                        zorder=4,
                    )
                if row_index == 0 and column_index == 0:
                    axis.legend(loc="best", frameon=False, fontsize=8)
                axis.set_ylabel(f"{scene} {label}")
                axis.set_xticks([500, 1000, 1500, 2000])
        for axis in axes[-1]:
            axis.set_xlabel("Checkpoint step")
        fig.subplots_adjust(top=0.90, hspace=0.25, wspace=0.25)
        _add_header(
            fig,
            axes[0, 0],
            f"{group} checkpoint quality",
            "Four retained checkpoints are discrete evaluation anchors; outlined markers identify checkpoints selected only from near-held-out PSNR.",
        )
        _save_figure(fig, root / f"{group}_checkpoint_quality")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5))
    labels = [
        f"{row['scene_id']}\n{row['arm'].replace('_geometry', '')}"
        for row in training_resources
    ]
    values = [float(row["mean_step_seconds"]) for row in training_resources]
    axis.bar(
        range(len(values)),
        values,
        color=[colors[row["arm"]] for row in training_resources],
    )
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("Mean seconds per step")
    fig.subplots_adjust(top=0.82, bottom=0.20)
    _add_header(
        fig,
        axis,
        "Stage 2 training timing summary",
        "This campaign lacked per-step timestamps; values are run-wall estimates from manifest/result file boundaries and are not step-level observations.",
    )
    _save_figure(fig, root / "training_timing_summary")
    plt.close(fig)

    if final_summary:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, (metric, label) in zip(
            axes,
            (("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("lpips", "LPIPS")),
        ):
            rows = [
                row
                for row in final_summary
                if row["condition"] == "correct"
            ]
            labels = [
                f"{row['scene_id']}\n{row['arm'].replace('_geometry', '')}"
                for row in rows
            ]
            values = [float(row[f"{metric}_mean"]) for row in rows]
            palette = [
                "#C5CAD3"
                if row["arm"] == "baseline_no_geometry"
                else colors[row["arm"]]
                for row in rows
            ]
            axis.scatter(
                range(len(rows)),
                values,
                c=palette,
                s=55,
                edgecolors="#464C55",
            )
            axis.set_xticks(range(len(rows)), labels, rotation=55, ha="right")
            axis.set_ylabel(label)
        fig.subplots_adjust(top=0.84, bottom=0.30, wspace=0.28)
        _add_header(
            fig,
            axes[0],
            "Frozen-candidate performance on far-held-out cameras",
            "Each point is the mean across four pre-registered far views; no checkpoint trajectory is shown for final test.",
        )
        _save_figure(fig, root / "far_quality_comparison")
        plt.close(fig)

    pose_plot_rows = []
    for row in pose_summary:
        if row["arm"] not in {"rre_geometry", "plucker_geometry"}:
            continue
        group = row["evaluation_group"]
        if group == "near-held-out":
            if int(row["checkpoint_step"]) != selected_steps[
                (row["scene_id"], row["arm"])
            ]:
                continue
        elif group != "far-held-out":
            continue
        pose_plot_rows.append(row)
    if pose_plot_rows:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        labels = [
            f"{row['evaluation_group']}\n{row['scene_id']}/"
            f"{row['arm'].replace('_geometry', '')}"
            for row in pose_plot_rows
        ]
        palette = [colors[row["arm"]] for row in pose_plot_rows]
        gaps = [
            float(row["shuffled_flow_loss_mean"])
            - float(row["correct_flow_loss_mean"])
            for row in pose_plot_rows
        ]
        ratios = [
            float(row["pose_response_ratio_mean"])
            for row in pose_plot_rows
        ]
        axes[0].bar(
            range(len(labels)),
            gaps,
            color=palette,
            edgecolor="#464C55",
        )
        axes[0].axhline(0, color="#464C55", linewidth=1.0)
        axes[0].set_ylabel("Shuffled minus correct flow loss")
        axes[1].bar(
            range(len(labels)),
            ratios,
            color=palette,
            edgecolor="#464C55",
        )
        axes[1].axhline(
            10,
            color="#464C55",
            linewidth=1.0,
            linestyle="--",
        )
        axes[1].set_ylabel("Pose response ratio")
        for axis in axes:
            axis.set_xticks(
                range(len(labels)),
                labels,
                rotation=55,
                ha="right",
            )
        fig.subplots_adjust(top=0.84, bottom=0.35, wspace=0.25)
        _add_header(
            fig,
            axes[0],
            "Camera intervention response for frozen candidates",
            "Positive flow-loss gaps favor correct cameras; the dark reference line marks the pre-registered 10x response-ratio requirement.",
        )
        _save_figure(fig, root / "pose_interventions")
        plt.close(fig)

    _qualitative_contact_sheets(root, frozen, final_complete)

    far_pose_rows = [
        row
        for row in pose_summary
        if row["evaluation_group"] == "far-held-out"
        and row["arm"] in {"rre_geometry", "plucker_geometry"}
    ]
    usage_wins = sum(
        float(row["correct_flow_loss_mean"])
        < float(row["shuffled_flow_loss_mean"])
        for row in far_pose_rows
    )
    usage_pass = (
        bool(far_pose_rows)
        and usage_wins / len(far_pose_rows) >= 0.75
        and all(
            float(row["pose_response_ratio_mean"]) >= 10
            for row in far_pose_rows
        )
    )
    quality_checks = []
    for scene in ("chair", "lego", "drums"):
        baseline = [
            row
            for row in final_summary
            if row["scene_id"] == scene
            and row["arm"] == "baseline_no_geometry"
            and row["condition"] == "correct"
        ]
        for arm in ("rre_geometry", "plucker_geometry"):
            current = [
                row
                for row in final_summary
                if row["scene_id"] == scene
                and row["arm"] == arm
                and row["condition"] == "correct"
            ]
            if baseline and current:
                quality_checks.append({
                    "scene": scene,
                    "arm": arm,
                    "psnr_delta": (
                        float(current[0]["psnr_mean"])
                        - float(baseline[0]["psnr_mean"])
                    ),
                    "ssim_delta": (
                        float(current[0]["ssim_mean"])
                        - float(baseline[0]["ssim_mean"])
                    ),
                    "lpips_delta": (
                        float(current[0]["lpips_mean"])
                        - float(baseline[0]["lpips_mean"])
                    ),
                })
    quality_pass = (
        bool(quality_checks)
        and all(
            row["psnr_delta"] >= -0.1
            and row["ssim_delta"] >= -0.002
            and row["lpips_delta"] <= 0.01
            for row in quality_checks
        )
    )
    verdict = {
        "final_test_complete": final_complete,
        "usage_pass": usage_pass if final_complete else None,
        "quality_pass": quality_pass if final_complete else None,
        "consistency_pass": False if final_complete else None,
        "consistency_availability": (
            "available" if final_geometry else "unavailable"
        ),
        "usage_wins": usage_wins,
        "usage_total": len(far_pose_rows),
        "quality_checks": quality_checks,
    }
    (root / "campaign_verdict.json").write_text(
        json.dumps(verdict, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stage 2 Near-Selected Geometry Results",
        "",
        "## Technical Summary",
        "",
        "- Six geometry runs use four fixed seen cameras and select checkpoints only from four scene-specific near-held-out cameras.",
        (
            "- Final far-held-out evaluation is complete."
            if final_complete
            else "- Final far-held-out evaluation has not been run."
        ),
        "- The experiment uses one seed and supports descriptive, not statistical, conclusions.",
        "- Reprojection and cross-view reconstruction remain unavailable without a validated metric depth scale.",
        "",
        "## Protocol",
        "",
        "| Scene | Seen views | Near-held-out views | Far-held-out views |",
        "|---|---|---|---|",
    ]
    for scene in ("chair", "lego", "drums"):
        item = protocol["scenes"][scene]
        lines.append(
            f"| {scene} | {item['seen_view_indices']} | "
            f"{item['near_view_indices']} | {item['far_view_indices']} |"
        )
    lines.extend([
        "",
        "Checkpoint selection maximizes near-held-out mean correct PSNR. Candidates within 0.01 dB use SSIM, LPIPS, then earlier step as deterministic tie-breakers.",
        "",
        "## Frozen Checkpoints",
        "",
        "| Scene | Arm | Step | Near PSNR | Near SSIM | Near LPIPS | SHA256 prefix |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for row in frozen["candidates"]:
        metrics = row["near_metrics"]
        lines.append(
            f"| {row['scene']} | {row['arm']} | {row['step']} | "
            f"{metrics['mean_correct_psnr']:.4f} | "
            f"{metrics['mean_correct_ssim']:.6f} | "
            f"{metrics['mean_correct_lpips']:.6f} | "
            f"{row['checkpoint_sha256'][:12]} |"
        )
    lines.extend([
        "",
        "## Near-Held-Out Quality",
        "",
        "| Scene | Arm | Selected step | PSNR | SSIM | LPIPS |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for scene in ("chair", "lego", "drums"):
        for arm in ("baseline_no_geometry", "rre_geometry", "plucker_geometry"):
            step = (
                0
                if arm == "baseline_no_geometry"
                else selected_steps[(scene, arm)]
            )
            rows = [
                row
                for row in selection_summary
                if row["scene_id"] == scene
                and row["arm"] == arm
                and row["evaluation_group"] == "near-held-out"
                and row["condition"] == "correct"
                and int(row["checkpoint_step"]) == step
            ]
            row = rows[0]
            lines.append(
                f"| {scene} | {arm} | {step} | "
                f"{float(row['psnr_mean']):.4f} | "
                f"{float(row['ssim_mean']):.6f} | "
                f"{float(row['lpips_mean']):.6f} |"
            )
    if final_summary:
        lines.extend([
            "",
            "## Final Far-Held-Out Quality",
            "",
            "| Scene | Arm | Frozen step | PSNR | SSIM | LPIPS | PSNR vs baseline |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for scene in ("chair", "lego", "drums"):
            baseline = [
                row
                for row in final_summary
                if row["scene_id"] == scene
                and row["arm"] == "baseline_no_geometry"
                and row["condition"] == "correct"
            ][0]
            for arm in (
                "baseline_no_geometry",
                "rre_geometry",
                "plucker_geometry",
            ):
                row = [
                    value
                    for value in final_summary
                    if value["scene_id"] == scene
                    and value["arm"] == arm
                    and value["condition"] == "correct"
                ][0]
                step = (
                    0
                    if arm == "baseline_no_geometry"
                    else selected_steps[(scene, arm)]
                )
                delta = float(row["psnr_mean"]) - float(baseline["psnr_mean"])
                lines.append(
                    f"| {scene} | {arm} | {step} | "
                    f"{float(row['psnr_mean']):.4f} | "
                    f"{float(row['ssim_mean']):.6f} | "
                    f"{float(row['lpips_mean']):.6f} | {delta:+.4f} |"
                )
        lines.extend([
            "",
            "## Final Verdicts",
            "",
            f"- USAGE_PASS={usage_pass}: {usage_wins}/{len(far_pose_rows)} frozen geometry evaluations have lower correct-camera than shuffled-camera flow loss; the 10x response-ratio condition is evaluated separately.",
            f"- QUALITY_PASS={quality_pass}: evaluated against the pre-registered PSNR/SSIM/LPIPS non-regression thresholds.",
            "- CONSISTENCY_PASS=False: no validated metric-depth reprojection or cross-view reconstruction evidence is available.",
        ])
    lines.extend([
        "",
        "## Limitations",
        "",
        "- The four far cameras were already evaluated in the 2026-09-02 campaign, so they are run once in this protocol but are not pristine project-history holdouts.",
        "- One seed cannot establish statistical significance or robustness across initialization.",
        "- Camera-use evidence does not substitute for decoded image quality or metric-depth geometric consistency.",
        "- The six completed runs predate step-level timing instrumentation. training_resource_summary.csv therefore labels filesystem-boundary timing as an estimate; no per-step timing values were reconstructed.",
        "",
        "## Artifacts",
        "",
        "- checkpoint_selection_audit.csv and frozen_candidates.json: complete selection audit and immutable checkpoint hashes.",
        "- training_loss_curves.*, training_gradient_curves.*, training_sigma_curves.*, and training_memory_curves.*: 2000-step training diagnostics.",
        "- training_resource_summary.csv and training_timing_summary.*: resource statistics and explicitly labeled run-level timing estimates.",
        "- seen_checkpoint_quality.* and near-held-out_checkpoint_quality.*: discrete checkpoint trajectories with selected markers.",
        "- far_quality_comparison.*: final frozen-candidate comparison, present only after final test.",
        "- qualitative/: selected seen, near-held-out, and final far-held-out contact sheets.",
    ])
    (root / "stage2_near_select_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _campaign_report(root: Path) -> None:
    image = _campaign_rows(root, "evaluation_rows.csv")
    geometry = _campaign_rows(root, "geometry_rows.csv")
    pose = _campaign_rows(root, "pose_sensitivity.csv")
    image_summary = _numeric_summary(image, ("scene_id", "arm", "split", "condition"), ("mae", "psnr", "ssim", "lpips"))
    geometry_summary = _numeric_summary(geometry, ("scene_id", "arm", "split", "metric_group", "condition"), ("coverage", "photometric_mae", "normalized_error", "psnr", "ssim"))
    pose_summary = _numeric_summary(pose, ("scene_id", "arm", "split"), ("correct_flow_loss", "shuffled_flow_loss", "baseline_flow_loss", "correct_shuffled_delta", "correct_disabled_delta", "pose_perturbation_delta", "pose_response_ratio"))
    _write(root / "campaign_image_summary.csv", image_summary)
    if geometry_summary:
        _write(root / "campaign_geometry_summary.csv", geometry_summary)
    else:
        (root / "campaign_geometry_summary.csv").write_text(
            "scene_id,arm,split,metric_group,condition,availability,reason\n"
            ",,,,,unavailable,No validated depth manifest was supplied\n",
            encoding="utf-8",
        )
    _write(root / "campaign_pose_summary.csv", pose_summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_paths = sorted(root.glob("*/*/train/train_steps.csv"))
    if train_paths:
        fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
        for path in train_paths:
            rows = _read(path)
            label = "/".join(path.relative_to(root).parts[:2])
            axis.plot([int(row["step"]) for row in rows], [float(row["loss"]) for row in rows], label=label)
        axis.set(title="Stage 2 training loss by scene and adapter", xlabel="step", ylabel="flow loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        fig.savefig(root / "campaign_training_curves.png", dpi=160)
        plt.close(fig)

    snapshot_paths = sorted(root.glob("*/*/eval_snapshots/step_*/evaluation_rows.csv"))
    if snapshot_paths:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for path in snapshot_paths:
            parts = path.relative_to(root).parts
            scene, arm, step_name = parts[0], parts[1], parts[3]
            snapshot_rows = _read(path)
            selected = [row for row in snapshot_rows if row.get("split") == "test" and row.get("condition") == "correct"]
            if not selected:
                continue
            step = int(step_name.removeprefix("step_"))
            label = f"{scene}/{arm}"
            for axis, key in zip(axes, ("psnr", "ssim", "lpips")):
                values = [float(row[key]) for row in selected if row.get(key, "") != ""]
                if values:
                    axis.scatter([step], [statistics.fmean(values)], label=label, s=18)
        axes[0].set_title("Held-out PSNR")
        axes[1].set_title("Held-out SSIM")
        axes[2].set_title("Held-out LPIPS")
        for axis in axes:
            axis.set_xlabel("step")
            axis.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, fontsize=7, ncol=2)
        fig.savefig(root / "campaign_heldout_quality_curves.png", dpi=160)
        plt.close(fig)

    if geometry_summary:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for index, group in enumerate(("reprojection", "cross_view_reconstruction")):
            selected = [row for row in geometry_summary if row.get("metric_group") == group and row.get("condition") in {"baseline", "correct"}]
            labels = [f"{row['scene_id']}\n{row['arm']}\n{row['condition']}" for row in selected]
            values = [float(row["photometric_mae_mean"]) if row["photometric_mae_mean"] != "" else math.nan for row in selected]
            axes[index].bar(range(len(values)), values)
            axes[index].set_title(group)
            axes[index].set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
            axes[index].grid(axis="y", alpha=0.25)
        fig.savefig(root / "campaign_geometry_metrics.png", dpi=160)
        plt.close(fig)

    pose_plot_rows = [row for row in pose_summary if row.get("split") == "test" and row.get("arm") != "baseline_no_geometry"]
    if pose_plot_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        labels = [f"{row['scene_id']}\n{row['arm']}" for row in pose_plot_rows]
        x = list(range(len(labels)))
        axes[0].bar(x, [float(row["correct_shuffled_delta_mean"]) for row in pose_plot_rows], color="tab:blue")
        axes[0].set_title("Correct vs shuffled latent delta")
        axes[1].bar(x, [float(row["pose_perturbation_delta_mean"]) for row in pose_plot_rows], color="tab:orange")
        axes[1].set_title("Pose perturbation response")
        for axis in axes:
            axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
            axis.grid(axis="y", alpha=0.25)
        fig.savefig(root / "campaign_pose_interventions.png", dpi=160)
        plt.close(fig)

    def mean(rows, key):
        values = [float(row[key]) for row in rows if row.get(key, "") not in ("", None)]
        return statistics.fmean(values) if values else math.nan

    usage_rows = [row for row in pose_summary if row.get("split") == "test" and row.get("arm") in {"rre_geometry", "plucker_geometry"}]
    usage_wins = sum(float(row.get("correct_flow_loss_mean", math.inf)) < float(row.get("shuffled_flow_loss_mean", -math.inf)) for row in usage_rows)
    usage_pass = bool(usage_rows) and usage_wins / len(usage_rows) >= 0.75 and all(float(row.get("pose_response_ratio_mean", 0)) >= 10 for row in usage_rows)

    consistency_checks = []
    for scene in sorted({row.get("scene_id") for row in geometry_summary}):
        for arm in ("rre_geometry", "plucker_geometry"):
            checks = []
            for group in ("reprojection", "cross_view_reconstruction"):
                correct = [row for row in geometry_summary if row.get("scene_id") == scene and row.get("arm") == arm and row.get("metric_group") == group and row.get("condition") == "correct"]
                baseline = [row for row in geometry_summary if row.get("scene_id") == scene and row.get("arm") == arm and row.get("metric_group") == group and row.get("condition") == "baseline"]
                if correct and baseline:
                    c = float(correct[0]["photometric_mae_mean"]); b = float(baseline[0]["photometric_mae_mean"])
                    checks.append((b - c) / max(b, 1e-12))
            if checks:
                consistency_checks.append(max(checks))
    consistency_pass = len(consistency_checks) >= 2 and sum(value >= 0.05 for value in consistency_checks) >= 2

    quality_checks = []
    for scene in sorted({row.get("scene_id") for row in image_summary}):
        for arm in ("rre_geometry", "plucker_geometry"):
            correct = [row for row in image_summary if row.get("scene_id") == scene and row.get("arm") == arm and row.get("split") == "test" and row.get("condition") == "correct"]
            baseline = [row for row in image_summary if row.get("scene_id") == scene and row.get("arm") == arm and row.get("split") == "test" and row.get("condition") == "baseline"]
            if correct and baseline:
                quality_checks.append({
                    "psnr_delta": float(correct[0]["psnr_mean"]) - float(baseline[0]["psnr_mean"]),
                    "ssim_delta": float(correct[0]["ssim_mean"]) - float(baseline[0]["ssim_mean"]),
                    "lpips_delta": float(correct[0].get("lpips_mean", 0) or 0) - float(baseline[0].get("lpips_mean", 0) or 0),
                })
    quality_pass = bool(quality_checks) and all(value["psnr_delta"] >= -0.1 and value["ssim_delta"] >= -0.002 and value["lpips_delta"] <= 0.01 for value in quality_checks)
    verdict = {"usage_pass": usage_pass, "consistency_pass": consistency_pass, "quality_pass": quality_pass, "usage_wins": usage_wins, "usage_total": len(usage_rows), "consistency_scene_gains": consistency_checks, "quality_checks": quality_checks}
    (root / "campaign_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage 2 Extension Results",
        "",
        "Date: 2026-09-02",
        "",
        f"Image rows: {len(image)}; geometry rows: {len(geometry)}; pose rows: {len(pose)}.",
        "",
        "This report is descriptive and paired by scene, split, condition, view, and view pair. It does not claim statistical significance from one seed.",
        "",
        "Reprojection is reported only where an explicit GT depth manifest was supplied and validated. Missing depth coverage is an unavailable metric, not a zero.",
        "",
        "Quality, geometry consistency, and camera usage are separate verdicts; one passing category does not imply the others passed.",
        "",
        f"Verdicts: USAGE_PASS={usage_pass}; CONSISTENCY_PASS={consistency_pass}; QUALITY_PASS={quality_pass}.",
        "",
        "## Held-out Image Quality",
        "",
        "| Scene | Arm | PSNR (dB) | SSIM | LPIPS | PSNR vs baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scene in ("chair", "lego", "drums"):
        for arm in ("baseline_no_geometry", "rre_geometry", "plucker_geometry"):
            current = [row for row in image_summary if row.get("scene_id") == scene and row.get("arm") == arm and row.get("split") == "test" and row.get("condition") == "correct"]
            baseline = [row for row in image_summary if row.get("scene_id") == scene and row.get("arm") == arm and row.get("split") == "test" and row.get("condition") == "baseline"]
            if current and baseline:
                psnr = float(current[0]["psnr_mean"]); ssim = float(current[0]["ssim_mean"]); lpips = float(current[0]["lpips_mean"])
                delta = psnr - float(baseline[0]["psnr_mean"])
                lines.append(f"| {scene} | {arm} | {psnr:.4f} | {ssim:.6f} | {lpips:.6f} | {delta:+.4f} |")
    lines.extend([
        "",
        "## Pose Sensitivity (held-out)",
        "",
        "Correct-camera flow loss is compared with shuffled-camera flow loss. The response ratio uses a fixed small pose jitter; values near 1 indicate a response comparable to that jitter, not a 10x intervention margin.",
        "",
        "| Scene | Arm | Correct flow | Shuffled flow | Gap | Response ratio |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in sorted(usage_rows, key=lambda value: (value.get("scene_id", ""), value.get("arm", ""))):
        correct_loss = float(row["correct_flow_loss_mean"]); shuffled_loss = float(row["shuffled_flow_loss_mean"])
        ratio = float(row.get("pose_response_ratio_mean", "nan"))
        lines.append(f"| {row['scene_id']} | {row['arm']} | {correct_loss:.6f} | {shuffled_loss:.6f} | {shuffled_loss - correct_loss:+.6f} | {ratio:.3f} |")
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `campaign_image_summary.csv`: per-scene, split, condition and view image metrics.",
        "- `campaign_pose_summary.csv`: per-scene train/test pose interventions.",
        "- `campaign_training_curves.png`: retained 2000-step loss curves for geometry arms.",
        "- `campaign_heldout_quality_curves.png`: held-out PSNR/SSIM/LPIPS at 500/1000/1500/2000 checkpoints.",
        "- `campaign_pose_interventions.png`: correct/shuffled and perturbation response bars.",
        "- `*/<arm>/eval/qualitative_contact.png`: fixed-noise held-out qualitative contact sheets.",
        "- `campaign_geometry_summary.csv`: unavailable because no validated metric depth manifest was supplied.",
    ])
    (root / "stage2_extension_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--campaign-root", type=Path)
    args = parser.parse_args()
    if bool(args.experiment_dir) == bool(args.campaign_root):
        raise SystemExit("provide exactly one of --experiment-dir or --campaign-root")
    if args.campaign_root is not None:
        if (args.campaign_root / "protocol_manifest.json").is_file():
            _near_select_report(args.campaign_root)
        else:
            _campaign_report(args.campaign_root)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = _read(args.experiment_dir / "train_steps.csv")
    metrics = _read(args.experiment_dir / "geometry_metrics.csv")
    if not train:
        raise RuntimeError("train_steps.csv is empty")
    steps = [int(row["step"]) for row in train]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, [float(row["loss"]) for row in train], label="train loss")
    axes[0, 0].set_title("Flow loss")
    axes[0, 1].plot(steps, [float(row["sigma"]) for row in train], label="sigma", color="tab:orange")
    axes[0, 1].set_title("Training sigma")
    axes[1, 0].plot(steps, [float(row["gradient_norm"]) for row in train], label="gradient", color="tab:green")
    axes[1, 0].set_title("Geometry gradient norm")
    axes[1, 1].plot(steps, [float(row["peak_gpu_memory_mib"]) for row in train], label="MiB", color="tab:red")
    axes[1, 1].set_title("Peak GPU memory")
    for axis in axes.flat:
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
    fig.savefig(args.experiment_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    if metrics:
        eval_steps = [int(row["step"]) for row in metrics]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        axes[0].plot(eval_steps, [float(row["correct_loss"]) for row in metrics], label="correct")
        axes[0].plot(eval_steps, [float(row["shuffled_loss"]) for row in metrics], label="shuffled")
        axes[0].plot(eval_steps, [float(row["baseline_loss"]) for row in metrics], label="baseline")
        axes[0].set_title("Fixed-noise flow loss")
        axes[1].plot(eval_steps, [float(row["correct_shuffled_delta"]) for row in metrics], label="correct-shuffled")
        axes[1].plot(eval_steps, [float(row["correct_baseline_delta"]) for row in metrics], label="correct-baseline")
        axes[1].set_title("Camera intervention")
        if "correct_psnr" in metrics[0]:
            axes[2].plot(eval_steps, [float(row["correct_psnr"]) for row in metrics], label="geometry")
            axes[2].plot(eval_steps, [float(row["baseline_psnr"]) for row in metrics], label="baseline")
            axes[2].set_title("Decoded PSNR")
        else:
            axes[2].axis("off")
        for axis in axes[:2]:
            axis.set_xlabel("step")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.savefig(args.experiment_dir / "geometry_diagnostics.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
