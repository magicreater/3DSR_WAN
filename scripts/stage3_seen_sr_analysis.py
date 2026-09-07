#!/usr/bin/env python3
"""Validate and summarize the fixed Stage 3 seen-view SR campaign."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ARMS = ("A0", "A1", "A2", "A3")
TRAIN_SEEDS = (42, 43)
SCENES = ("chair", "lego", "drums", "hotdog", "mic")
CHECKPOINT_STEPS = tuple(range(500, 4001, 500))
METRICS = ("psnr", "ssim", "lpips", "mae")
INFERENCE_SEEDS = (3302, 3303, 3304)
GOOD_SIGN = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0, "mae": -1.0}
QUALITATIVE_CROP_SIZE = 96


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_metrics(row: dict) -> None:
    for metric in METRICS:
        value = row.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"missing or nonfinite {metric}: {row}")


def validate_rows(rows: list[dict], expected: int, conditions: set[str]) -> None:
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    keys = set()
    for row in rows:
        finite_metrics(row)
        if row.get("scene") not in SCENES or row.get("condition") not in conditions:
            raise ValueError(f"unexpected scene or condition: {row}")
        key = (row.get("condition"), row.get("inference_seed"), row.get("scene"), row.get("view_index"))
        if key in keys:
            raise ValueError(f"duplicate evaluation identity: {key}")
        keys.add(key)


def load_campaign(root: Path) -> dict:
    data = {"train": {}, "probe": {}, "full": {}, "baseline": {}, "intervention": {}}
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            run = (arm, seed)
            train = read_jsonl(root / "train" / arm / f"seed{seed}" / "train_steps.jsonl")
            if [row.get("step") for row in train] != list(range(1, 4001)):
                raise ValueError(f"non-contiguous training log for {run}")
            data["train"][run] = train
            probes = []
            for step in CHECKPOINT_STEPS:
                rows = read_jsonl(root / "probe" / arm / f"seed{seed}" / f"step{step:04d}" / "evaluation_rows.jsonl")
                validate_rows(rows, 20, {"correct"})
                probes.extend(rows)
            data["probe"][run] = probes
            full_root = root / "full" / arm / f"seed{seed}"
            full = read_jsonl(full_root / "evaluation_rows.jsonl")
            baseline = read_jsonl(full_root / "baseline_rows.jsonl")
            validate_rows(full, 1500, {"correct"})
            validate_rows(baseline, 1000, {"bicubic", "vae_ceiling"})
            data["full"][run] = full
            data["baseline"][run] = baseline
            interventions = read_jsonl(root / "intervention" / arm / f"seed{seed}" / "evaluation_rows.jsonl")
            validate_rows(
                interventions,
                1200,
                {"correct", "correct_repeat", "remove", "duplicate", "shuffle_camera"},
            )
            data["intervention"][run] = interventions
    return data


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {metric: statistics.mean(float(row[metric]) for row in rows) for metric in METRICS}


def scene_equal(rows: list[dict]) -> dict[str, float]:
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    if set(by_scene) != set(SCENES):
        raise ValueError("scene-equal aggregation lacks complete scene coverage")
    per_scene = {scene: mean_metrics(values) for scene, values in by_scene.items()}
    return {metric: statistics.mean(per_scene[scene][metric] for scene in SCENES) for metric in METRICS}


def indexed(rows: list[dict], include_condition: bool = False) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = (row.get("inference_seed"), row["scene"], row["view_index"])
        if include_condition:
            key = (row["condition"], *key)
        if key in result:
            raise ValueError(f"duplicate paired row: {key}")
        result[key] = row
    return result


def paired_gain(candidate: list[dict], reference: list[dict]) -> tuple[dict, list[dict]]:
    left, right = indexed(candidate), indexed(reference)
    if left.keys() != right.keys():
        raise ValueError("paired populations do not match")
    rows = []
    for key in sorted(left):
        row = {"inference_seed": key[0], "scene": key[1], "view_index": key[2]}
        for metric in METRICS:
            row[metric] = (float(left[key][metric]) - float(right[key][metric])) * GOOD_SIGN[metric]
        rows.append(row)
    return scene_equal(rows), rows


def baseline_gain(candidate: list[dict], baseline: list[dict], condition: str) -> tuple[dict, list[dict]]:
    base = {(row["scene"], row["view_index"]): row for row in baseline if row["condition"] == condition}
    rows = []
    for row in candidate:
        reference = base[(row["scene"], row["view_index"])]
        gain = {"inference_seed": row["inference_seed"], "scene": row["scene"], "view_index": row["view_index"]}
        for metric in METRICS:
            gain[metric] = (float(row[metric]) - float(reference[metric])) * GOOD_SIGN[metric]
        rows.append(gain)
    return scene_equal(rows), rows


def condition_gain(rows: list[dict], comparator: str) -> tuple[dict, list[dict]]:
    values = indexed(rows, include_condition=True)
    paired = []
    identities = sorted({key[1:] for key in values if key[0] == "correct"})
    for identity in identities:
        correct = values[("correct", *identity)]
        reference = values[(comparator, *identity)]
        row = {"inference_seed": identity[0], "scene": identity[1], "view_index": identity[2]}
        for metric in METRICS:
            row[metric] = (float(correct[metric]) - float(reference[metric])) * GOOD_SIGN[metric]
        paired.append(row)
    return scene_equal(paired), paired


def jitter(rows: list[dict]) -> dict[str, float]:
    values = indexed(rows, include_condition=True)
    identities = sorted({key[1:] for key in values if key[0] == "correct"})
    return {
        metric: max(abs(float(values[("correct", *key)][metric]) - float(values[("correct_repeat", *key)][metric])) for key in identities)
        for metric in METRICS
    }


def directional_pass(gain: dict, noise: dict) -> bool:
    return (
        gain["psnr"] > noise["psnr"]
        and gain["ssim"] >= -noise["ssim"]
        and gain["lpips"] >= -noise["lpips"]
        and gain["mae"] >= -noise["mae"]
    )


def invariant_pass(gain: dict, noise: dict) -> bool:
    return all(abs(gain[metric]) <= noise[metric] + 1e-8 for metric in METRICS)


def select_qualitative(data: dict) -> tuple[dict, list[str]]:
    selection, group_ids = {}, []
    for scene in SCENES:
        deltas = defaultdict(list)
        for seed in TRAIN_SEEDS:
            _, paired = paired_gain(data["full"][("A3", seed)], data["full"][("A0", seed)])
            for row in paired:
                if row["scene"] == scene:
                    deltas[row["view_index"]].append(row["psnr"])
        ranked = sorted((statistics.mean(values), view) for view, values in deltas.items())
        chosen = {"worst": ranked[0][1], "median": ranked[len(ranked) // 2][1], "best": ranked[-1][1]}
        selection[scene] = chosen
        group_ids.extend(f"{scene}:{view:03d}" for view in chosen.values())
    return selection, group_ids


def ema(values: list[float], alpha: float = 0.05) -> list[float]:
    output = []
    current = values[0]
    for value in values:
        current = alpha * value + (1 - alpha) * current
        output.append(current)
    return output


def plots(root: Path, output: Path, data: dict, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"A0": "#4C4C4C", "A1": "#0072B2", "A2": "#D55E00", "A3": "#009E73"}
    train_fields = (
        "loss", "gradient_norm", "bridge_gradient_norm", "rre_gradient_norm",
        "fusion_gradient_norm", "step_seconds", "peak_gpu_memory_mib", "learning_rate", "sigma_mean",
    )
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            rows = data["train"][(arm, seed)]
            steps = [row["step"] for row in rows]
            for axis, field in zip(axes.flat, train_fields):
                if field == "sigma_mean":
                    values = [statistics.mean(row["sigmas"]) for row in rows]
                else:
                    values = [row.get(field) for row in rows]
                if any(value is None for value in values):
                    continue
                values = [float(value) for value in values]
                axis.plot(steps, values, color=colors[arm], alpha=0.08)
                axis.plot(steps, ema(values), color=colors[arm], linestyle="-" if seed == 42 else "--",
                          label=f"{arm}/s{seed}")
                axis.set_title(field)
                axis.set_xlabel("optimizer step")
                axis.grid(alpha=0.2)
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7, ncol=2)
    fig.suptitle("Stage 3 seen-view training curves: raw + EMA(alpha=0.05)")
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            rows = data["probe"][(arm, seed)]
            by_step = {step: [row for row in rows if row["step"] == step] for step in CHECKPOINT_STEPS}
            for axis, metric in zip(axes, METRICS):
                values = [scene_equal(by_step[step])[metric] for step in CHECKPOINT_STEPS]
                axis.plot(CHECKPOINT_STEPS, values, marker="o", color=colors[arm],
                          linestyle="-" if seed == 42 else "--", label=f"{arm}/s{seed}")
                axis.set_title(metric.upper())
                axis.set_xlabel("checkpoint step")
                axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Fixed 20-view real 50-step probe")
    fig.savefig(output / "probe_quality_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    x = list(range(len(ARMS)))
    width = 0.34
    for axis, metric in zip(axes, METRICS):
        for offset, seed in ((-width / 2, 42), (width / 2, 43)):
            values = [summary["full_metrics"][arm][str(seed)][metric] for arm in ARMS]
            axis.bar([value + offset for value in x], values, width=width, label=f"seed {seed}")
        axis.set_xticks(x, ARMS)
        axis.set_title(metric.upper())
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle("All 500 train views, three inference seeds")
    fig.savefig(output / "final_sr_metrics.png", dpi=160)
    plt.close(fig)

    comparisons = ("A2-A1", "A3-A2", "A3-bicubic")
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    for axis, metric in zip(axes, METRICS):
        for offset, seed in ((-width / 2, 42), (width / 2, 43)):
            values = [summary["comparisons"][name][str(seed)][metric] for name in comparisons]
            axis.bar([value + offset for value in range(len(comparisons))], values, width=width, label=f"seed {seed}")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(comparisons)), comparisons, rotation=15)
        axis.set_title(f"good-direction delta {metric.upper()}")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.savefig(output / "contribution_deltas.png", dpi=160)
    plt.close(fig)


def _contact_canvas(rows: list[tuple[str, list]], labels: tuple[str, ...]):
    from PIL import Image, ImageDraw

    width, height = rows[0][1][0].size
    canvas = Image.new("RGB", (130 + width * len(labels), 32 + len(rows) * (height + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((130 + column * width + 4, 8), label, fill="black")
    for row_index, (label, images) in enumerate(rows):
        y = 32 + row_index * (height + 24)
        draw.text((6, y + 5), label, fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (130 + column * width, y))
    return canvas


def _center_crop(image, size: int = QUALITATIVE_CROP_SIZE):
    width, height = image.size
    if width < size or height < size:
        raise ValueError(f"qualitative image {image.size} is smaller than {size}x{size}")
    left, top = (width - size) // 2, (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def contact_sheets(root: Path, output: Path, selection: dict) -> None:
    from PIL import Image, ImageChops, ImageEnhance

    labels = ("HR", "LR", "Bicubic", "VAE", "A0", "A1", "A2", "A3", "A3 remove", "A3 shuffle", "A0 error", "A3 error")
    for scene, choices in selection.items():
        rows = []
        for rank in ("best", "median", "worst"):
            view = choices[rank]
            base = root / "full" / "A3" / "seed42" / "images" / scene / f"view_{view:03d}"
            refs = base / "reference"
            images = [
                Image.open(refs / "hr.png").convert("RGB"),
                Image.open(refs / "lr_nearest.png").convert("RGB"),
                Image.open(refs / "bicubic.png").convert("RGB"),
                Image.open(refs / "vae_ceiling.png").convert("RGB"),
            ]
            for arm in ARMS:
                path = root / "full" / arm / "seed42" / "images" / scene / f"view_{view:03d}" / "seed_3302" / "correct.png"
                images.append(Image.open(path).convert("RGB"))
            qualitative = root / "qualitative" / "A3" / "seed42" / "images" / scene / f"view_{view:03d}" / "seed_3302"
            images.extend([Image.open(qualitative / "remove.png").convert("RGB"),
                           Image.open(qualitative / "shuffle_camera.png").convert("RGB")])
            images.extend([
                ImageEnhance.Brightness(ImageChops.difference(images[0], images[4])).enhance(4),
                ImageEnhance.Brightness(ImageChops.difference(images[0], images[7])).enhance(4),
            ])
            rows.append((f"{rank} v{view}", images))
        _contact_canvas(rows, labels).save(output / f"qualitative_{scene}_full.png")
        crop_rows = [(label, [_center_crop(image) for image in images]) for label, images in rows]
        _contact_canvas(crop_rows, labels).save(output / f"qualitative_{scene}.png")


def build_summary(data: dict) -> dict:
    full_metrics = {arm: {} for arm in ARMS}
    comparisons = {name: {} for name in ("A2-A1", "A3-A2", "A3-A0", "A3-bicubic")}
    jitters, interventions = {}, {}
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            full_metrics[arm][str(seed)] = scene_equal(data["full"][(arm, seed)])
            jitters[f"{arm}/seed{seed}"] = jitter(data["intervention"][(arm, seed)])
            interventions[f"{arm}/seed{seed}"] = {
                condition: condition_gain(data["intervention"][(arm, seed)], condition)[0]
                for condition in ("remove", "duplicate", "shuffle_camera")
            }
    for seed in TRAIN_SEEDS:
        comparisons["A2-A1"][str(seed)] = paired_gain(data["full"][("A2", seed)], data["full"][("A1", seed)])[0]
        comparisons["A3-A2"][str(seed)] = paired_gain(data["full"][("A3", seed)], data["full"][("A2", seed)])[0]
        comparisons["A3-A0"][str(seed)] = paired_gain(data["full"][("A3", seed)], data["full"][("A0", seed)])[0]
        comparisons["A3-bicubic"][str(seed)] = baseline_gain(
            data["full"][("A3", seed)], data["baseline"][("A3", seed)], "bicubic"
        )[0]
    noise = {metric: max(value[metric] for value in jitters.values()) for metric in METRICS}
    sr_fit = all(directional_pass(comparisons["A3-bicubic"][str(seed)], noise) for seed in TRAIN_SEEDS)
    cross_quality = all(directional_pass(comparisons["A2-A1"][str(seed)], noise) for seed in TRAIN_SEEDS)
    cross_intervention = all(
        directional_pass(interventions[f"A2/seed{seed}"][condition], noise)
        for seed in TRAIN_SEEDS for condition in ("remove", "duplicate")
    )
    controls = all(
        invariant_pass(interventions[f"{arm}/seed{seed}"][condition], noise)
        for arm in ("A0", "A1") for seed in TRAIN_SEEDS
        for condition in ("remove", "duplicate", "shuffle_camera")
    )
    geometry_quality = all(directional_pass(comparisons["A3-A2"][str(seed)], noise) for seed in TRAIN_SEEDS)
    geometry_intervention = all(
        directional_pass(interventions[f"A3/seed{seed}"]["shuffle_camera"], noise)
        and invariant_pass(interventions[f"A2/seed{seed}"]["shuffle_camera"], noise)
        for seed in TRAIN_SEEDS
    )
    verdicts = {
        "SR_FIT_PASS": sr_fit,
        "CROSS_VIEW_PASS": cross_quality and cross_intervention and controls,
        "GEOMETRY_PASS": geometry_quality and geometry_intervention,
    }
    verdicts["SEEN_SR_EFFECTIVE"] = all(verdicts.values())
    return {
        "scope": "seen_train_sr_only",
        "full_metrics": full_metrics,
        "comparisons": comparisons,
        "interventions": interventions,
        "repeat_jitter_max": noise,
        "verdicts": verdicts,
        "claim_limit": "Training-view SR effectiveness only; no held-out generalization, NVS or 3D consistency claim.",
    }


def report(output: Path, summary: dict) -> None:
    verdict = summary["verdicts"]
    lines = [
        "# Stage 3 训练视图 SR 有效性报告",
        "",
        "## 结论",
        "",
        f"- `SR_FIT_PASS`: **{verdict['SR_FIT_PASS']}**",
        f"- `CROSS_VIEW_PASS`: **{verdict['CROSS_VIEW_PASS']}**",
        f"- `GEOMETRY_PASS`: **{verdict['GEOMETRY_PASS']}**",
        f"- `SEEN_SR_EFFECTIVE`: **{verdict['SEEN_SR_EFFECTIVE']}**",
        "",
        "本报告只判断五个共享训练场景中官方 train 视图的 SR 拟合与因果使用，不判断未见视图、未见场景、新视角合成、3DGS 或三维一致性。",
        "",
        "## 定量结果",
        "",
        "场景等权结果、逐 seed 对照、干预差值和重复运行抖动见 `summary.json` 与 `summary_rows.csv`。正向差值统一表示候选更好。",
        "",
        "![训练曲线](training_curves.png)",
        "",
        "![真实 50-step probe](probe_quality_curves.png)",
        "",
        "![最终 SR 指标](final_sr_metrics.png)",
        "",
        "![贡献差值](contribution_deltas.png)",
        "",
        "## 判定规则",
        "",
        "- 两个训练 seed 必须方向一致；PSNR 增益必须超过相同输入重复解码的最大抖动。",
        "- SSIM 不下降，LPIPS 与 MAE 不恶化；不设置额外 0.1/0.2 dB 门槛。",
        "- A2-A1 判断跨视角信息，A3-A2 与正确/打乱 fusion-camera 判断几何贡献。",
        "- 任一关口失败均保持原结论，不追加调参或修改阈值。",
    ]
    (output / "stage3_seen_sr_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_campaign(args.campaign_root.resolve())
    selection, group_ids = select_qualitative(data)
    write_json(output / "qualitative_selection.json", selection)
    (output / "qualitative_group_ids.txt").write_text("\n".join(group_ids) + "\n", encoding="utf-8")
    if args.selection_only:
        return
    summary = build_summary(data)
    write_json(output / "summary.json", summary)
    rows = []
    for comparison, seeds in summary["comparisons"].items():
        for seed, metrics in seeds.items():
            rows.append({"kind": "comparison", "name": comparison, "seed": seed, **metrics})
    for name, metrics in summary["interventions"].items():
        for condition, values in metrics.items():
            rows.append({"kind": "intervention", "name": name, "condition": condition, **values})
    write_csv(output / "summary_rows.csv", rows)
    plots(args.campaign_root.resolve(), output, data, summary)
    contact_sheets(args.campaign_root.resolve(), output, selection)
    report(output, summary)


if __name__ == "__main__":
    main()
