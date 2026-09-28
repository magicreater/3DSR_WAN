#!/usr/bin/env python3
"""Read-only, native-sampler trajectory audit for the frozen Lego Stage 3 comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

import numpy as np
import torch

import stage3_experiment as st
import stage3_loss_3dgs as gs
import stage3_loss_campaign as campaign
import stage3_loss_experiment as experiment
from rl3dsr.models.wan.sampling import FlowSamplingConfig, _scheduler
from rl3dsr.validation.decoded_space import frame_metrics
from rl3dsr.validation.sequence_matters import sha256_file
from rl3dsr.validation.stage3_protocol import load_stage3_config


ROOT = campaign.ROOT
OUT = ROOT / "artifacts/stage3_trajectory_diagnosis_20260928"
SEEDS = (3302, 3303)
MODES = ("correct", "target_drop", "remove", "shuffle_all")
IMAGE_STEPS = (0, 9, 19, 29, 39, 49)
ARMS = {
    "full_ordinary": (campaign.config_path("full"), campaign.OUT / "train/full_ordinary/stage3_step_4000.pt", "candidate"),
    "a5": (gs.A5_CONFIG, gs.A5_CHECKPOINT, "a5"),
}


def payload() -> dict:
    view_file = gs.OUT / "view_protocol.json"
    source_file = gs.OUT / "source_manifest.json"
    views = json.loads(view_file.read_text())
    source = json.loads(source_file.read_text())
    if source["view_protocol_sha256"] != sha256_file(view_file):
        raise RuntimeError("frozen 3DGS view protocol changed")
    if len(views["nested_indices"]) != 16 or any(
        len(views["sr_contexts"][str(index)]) != 4 or
        views["sr_contexts"][str(index)][0] != index
        for index in views["nested_indices"]
    ):
        raise RuntimeError("frozen 16-target/four-view context protocol invalid")
    arms = {}
    for arm, (config_path, checkpoint, image_arm) in ARMS.items():
        if not config_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"missing {arm} config/checkpoint")
        config = load_stage3_config(config_path)
        if (config.sampling_steps, config.sampling_shift, config.views, config.image_size) != (50, 5.0, 4, 256):
            raise RuntimeError(f"{arm} native sampling protocol changed")
        if source[f"{image_arm if image_arm == 'a5' else 'candidate'}_checkpoint_sha256"] != sha256_file(checkpoint):
            raise RuntimeError(f"{arm} source checkpoint hash changed")
        arms[arm] = {"config": str(config_path), "config_sha256": sha256_file(config_path),
                     "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                     "source_image_arm": image_arm}
    return {
        "purpose": "read-only true-HQ-noised versus native-UniPC free-run trajectory audit",
        "code_sha256": sha256_file(Path(__file__)),
        "stage3_experiment_sha256": sha256_file(ROOT / "scripts/stage3_experiment.py"),
        "sampling_sha256": sha256_file(ROOT / "src/rl3dsr/models/wan/sampling.py"),
        "metrics_sha256": sha256_file(ROOT / "src/rl3dsr/validation/decoded_space.py"),
        "view_protocol_sha256": sha256_file(view_file),
        "source_manifest_sha256": sha256_file(source_file),
        "dataset_train_sha256": views["source_train_transforms_sha256"],
        "seeds": SEEDS, "targets": views["nested_indices"],
        "contexts": views["sr_contexts"], "modes": MODES,
        "image_steps": IMAGE_STEPS, "noise_seed_rule": "seed*1000000+target_index",
        "sigma_path": "step0 pure noise matches native initial state; later z_true=(1-sigma)*VAE(HQ)+sigma*initial_noise; native sigma0=0.9997998",
        "clean_estimate": "z_hat0=state-sigma*predicted_velocity",
        "metrics": "rl3dsr.validation.decoded_space.frame_metrics; target view only; full-view VAE decoding",
        "arms": arms, "stage4": "HOLD",
    }


def protocol() -> dict:
    value = payload()
    OUT.mkdir(parents=True, exist_ok=True)
    experiment.frozen_json(OUT / "protocol.json", value)
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("no trajectory rows")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_metrics(runtime, latent, hr, lpips, save: Path | None) -> dict:
    decoded = runtime.vae.decode_multiview(latent.to(torch.bfloat16))[:, :, :1]
    metrics = frame_metrics(decoded, hr[:, :, :1], perceptual_metric=lpips)[0]
    if save is not None:
        st._save_frame(save, decoded)
    return {key: float(metrics[key]) for key in ("psnr", "ssim", "mae", "lpips") if key in metrics}


def row_for(runtime, clean, hr, state, noise, velocity, sigma, step, t, branch, mode, lpips, image_path):
    zhat = state.float() - float(sigma) * velocity.float()
    true_state = noise if step == 0 else ((1 - float(sigma)) * clean.float() + float(sigma) * noise.float()).to(noise.dtype)
    result = {
        "step": step, "timestep": float(t), "sigma": float(sigma),
        "branch": branch, "mode": mode,
        "latent_clean_mse_all": float((zhat - clean.float()).square().mean()),
        "latent_clean_mse_target": float((zhat[:, :, :1] - clean[:, :, :1].float()).square().mean()),
        "state_true_mse_all": float((state.float() - true_state.float()).square().mean()),
        "state_true_mse_target": float((state[:, :, :1].float() - true_state[:, :, :1].float()).square().mean()),
    }
    result.update(prediction_metrics(runtime, zhat, hr, lpips, image_path))
    result["fusion"] = st._scalar_diagnostics(getattr(runtime.module.fusion, "last_diagnostics", {}))
    result["rre_layers"] = st._scalar_diagnostics(runtime.module.geometry.last_diagnostics)
    if not all(np.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise RuntimeError(f"nonfinite diagnostic at {branch}/{mode}/{step}")
    return result


def unit(runtime, arm: str, index: int, seed: int, proto: dict, *, smoke: bool) -> None:
    unit_dir = OUT / ("smoke" if smoke else "units") / arm / f"view_{index:03d}_seed_{seed}"
    done = unit_dir / "manifest.json"
    if done.is_file():
        saved = json.loads(done.read_text())
        for name, digest in saved["files"].items():
            if sha256_file(unit_dir / name) != digest:
                raise RuntimeError(f"completed unit drift: {unit_dir / name}")
        return
    if unit_dir.exists():
        raise RuntimeError(f"partial unit needs audit: {unit_dir}")
    unit_dir.mkdir(parents=True)
    config_path, _, image_arm = ARMS[arm]
    config = load_stage3_config(config_path)
    context_indices = proto["contexts"][str(index)]
    _, hr, lr, camera = st._load_indices(
        experiment.RUNTIME_ROOT / "datasets/nerf_synthetic", "lego", "train",
        context_indices, config, runtime.device,
    )
    generator = torch.Generator(device="cpu").manual_seed(770000 + index)
    clean = runtime.vae.encode_multiview(hr)
    lpips = st._PerFrameLPIPS(runtime.device)
    rows: list[dict] = []
    sample_seed = seed * 1_000_000 + index
    correct_noise = None
    correct_features = None
    correct_context = None
    for mode in MODES:
        changed, fusion_camera, geometry_camera, mask, intervention = st._intervention(
            lr, camera, mode, generator, return_metadata=True,
        )
        holder = {}

        def tracing_sampler(noise, features, text_context, *, predict_velocity, config):
            nonlocal correct_noise, correct_features, correct_context
            scheduler = _scheduler(config, noise.device)
            if len(scheduler.timesteps) != 50 or not 0.999 < float(scheduler.sigmas[0]) < 1:
                raise RuntimeError("unexpected native schedule")
            if mode == "correct":
                correct_noise, correct_features, correct_context = noise.clone(), features, text_context
            elif not torch.equal(noise, correct_noise):
                raise RuntimeError("intervention initial noise differs from correct condition")
            holder["noise_sha256"] = hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest()
            state = noise.clone()
            if runtime.module.fusion is not None:
                runtime.module.fusion.record_diagnostics = True
            for step, t in enumerate(scheduler.timesteps):
                model_t = torch.full((state.shape[0],), float(t), device=state.device, dtype=torch.float32)
                velocity = predict_velocity(state, model_t, text_context, features)
                if velocity.shape != state.shape or not torch.isfinite(velocity).all():
                    raise RuntimeError("nonfinite native velocity")
                selected = step in IMAGE_STEPS
                image = unit_dir / "images" / f"{mode}_free_step_{step:02d}.png" if selected else None
                metric = row_for(runtime, clean, hr, state, noise, velocity,
                                 scheduler.sigmas[step], step, t, "free", mode,
                                 lpips if selected else None, image)
                rows.append(metric)
                state = scheduler.step(velocity, t, state, return_dict=False)[0]
            if runtime.module.fusion is not None:
                runtime.module.fusion.record_diagnostics = False
            return state

        sampled = st.sample_latents(
            runtime, changed, camera, tuple(clean.shape), config.sampling_steps,
            config.image_size, fusion_camera=fusion_camera, geometry_camera=geometry_camera,
            source_mask=mask, sampler=tracing_sampler, seed=sample_seed,
            sampling_shift=config.sampling_shift, dtype=torch.bfloat16,
            return_diagnostics=not config.dynamic_fusion,
        )
        final = sampled[0] if isinstance(sampled, tuple) else sampled
        final_image = unit_dir / "images" / f"{mode}_final.png"
        endpoint = prediction_metrics(runtime, final, hr, lpips, final_image)
        if mode == "correct" and seed == 3302:
            reference = gs.OUT / "source" / image_arm / f"view_{index:03d}.png"
            source_hash = json.loads((gs.OUT / "source_manifest.json").read_text())["images"][image_arm][str(index)]
            if sha256_file(reference) != source_hash or sha256_file(final_image) != source_hash:
                raise RuntimeError(f"native 50-step endpoint failed frozen source-image reproduction: {arm}/{index}")
        (unit_dir / f"{mode}_endpoint.json").write_text(json.dumps({
            "metrics": endpoint, "noise_sha256": holder["noise_sha256"],
            "intervention": intervention, "final_png_sha256": sha256_file(final_image),
        }, indent=2, sort_keys=True), encoding="utf-8")
        if mode == "correct":
            schedule = _scheduler(FlowSamplingConfig(config.sampling_steps, config.sampling_shift), runtime.device)
            noise = correct_noise
            if not torch.equal(noise, correct_noise):
                raise RuntimeError("true and free paths differ at initial pure noise")
            if runtime.module.fusion is not None:
                runtime.module.fusion.record_diagnostics = True
            for step, t in enumerate(schedule.timesteps):
                sigma = float(schedule.sigmas[step])
                state = noise if step == 0 else ((1 - sigma) * clean.float() + sigma * noise.float()).to(noise.dtype)
                model_t = torch.full((state.shape[0],), float(t), device=state.device, dtype=torch.float32)
                velocity = runtime.module.predict(runtime.dit, state, model_t,
                                                  correct_context, correct_features,
                                                  camera, tuple(clean.shape[2:]))
                if velocity.shape != state.shape or not torch.isfinite(velocity).all():
                    raise RuntimeError("nonfinite teacher-path velocity")
                selected = step in IMAGE_STEPS
                image = unit_dir / "images" / f"correct_true_step_{step:02d}.png" if selected else None
                rows.append(row_for(runtime, clean, hr, state, noise, velocity, sigma,
                                    step, t, "true", "correct", lpips if selected else None, image))
            if runtime.module.fusion is not None:
                runtime.module.fusion.record_diagnostics = False
        del final
        torch.cuda.empty_cache()
    if len(rows) != 50 * (len(MODES) + 1):
        raise RuntimeError("trajectory row count mismatch")
    with (unit_dir / "steps.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    flat = [{key: value for key, value in row.items() if key not in ("fusion", "rre_layers")}
            for row in rows]
    write_csv(unit_dir / "steps.csv", flat)
    files = {str(path.relative_to(unit_dir)): sha256_file(path) for path in unit_dir.rglob("*") if path.is_file()}
    done.write_text(json.dumps({"arm": arm, "view": index, "seed": seed,
                                "context_indices": context_indices,
                                "protocol_sha256": sha256_file(OUT / "protocol.json"),
                                "files": files}, indent=2, sort_keys=True), encoding="utf-8")


def run(arm: str, gpu: int, *, smoke: bool):
    campaign.gpu_free(gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select the checked GPU")
    proto = protocol()
    config_path, checkpoint, _ = ARMS[arm]
    config = load_stage3_config(config_path)
    st._seed_all(3302)
    runtime = experiment.runtime(config, checkpoint)
    runtime.module.eval()
    runtime.dit.model.eval().requires_grad_(False)
    runtime.vae.model.model.eval().requires_grad_(False)
    indices = proto["targets"][:1] if smoke else proto["targets"]
    seeds = SEEDS[:1] if smoke else SEEDS
    with torch.inference_mode():
        for index in indices:
            for seed in seeds:
                unit(runtime, arm, index, seed, proto, smoke=smoke)
                print(f"completed {arm} target={index} seed={seed} smoke={smoke}", flush=True)


def review():
    proto = protocol()
    summary = {}
    all_rows = []
    for arm in ARMS:
        units = []
        for index in proto["targets"]:
            for seed in SEEDS:
                directory = OUT / "units" / arm / f"view_{index:03d}_seed_{seed}"
                saved = json.loads((directory / "manifest.json").read_text())
                if saved["protocol_sha256"] != sha256_file(OUT / "protocol.json"):
                    raise RuntimeError("unit protocol drift")
                for name, digest in saved["files"].items():
                    if sha256_file(directory / name) != digest:
                        raise RuntimeError(f"unit artifact drift: {directory / name}")
                rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
                if len(rows) != 250:
                    raise RuntimeError("unit has incomplete trajectory")
                for row in rows:
                    row.update({"arm": arm, "view": index, "seed": seed})
                all_rows.extend(rows)
                true = {r["step"]: r for r in rows if r["branch"] == "true"}
                free = {r["step"]: r for r in rows if r["branch"] == "free" and r["mode"] == "correct"}
                gap = [true[i]["psnr"] - free[i]["psnr"] for i in range(50)]
                first_sustained = next((i for i in range(48) if all(gap[j] > 1 for j in range(i, i + 3))), None)
                endpoints = {mode: json.loads((directory / f"{mode}_endpoint.json").read_text())["metrics"]
                             for mode in MODES}
                units.append({"view": index, "seed": seed, "first_sustained_gap_step": first_sustained,
                              "max_true_minus_free_psnr": max(gap),
                              "true_psnr_step_49": true[49]["psnr"],
                              "free_psnr_step_49": free[49]["psnr"],
                              **{f"final_{mode}_psnr": endpoints[mode]["psnr"] for mode in MODES}})
        summary[arm] = {
            "units": len(units), "sustained_gap_count": sum(x["first_sustained_gap_step"] is not None for x in units),
            "mean_final_psnr": statistics.mean(x["final_correct_psnr"] for x in units),
            "mean_true_step_49_psnr": statistics.mean(x["true_psnr_step_49"] for x in units),
            "mean_free_step_49_psnr": statistics.mean(x["free_psnr_step_49"] for x in units),
            "mean_intervention_delta_psnr": {
                mode: statistics.mean(x["final_correct_psnr"] - x[f"final_{mode}_psnr"] for x in units)
                for mode in MODES[1:]
            }, "per_unit": units,
        }
    experiment.frozen_json(OUT / "review.json", summary)
    with (OUT / "all_steps.jsonl").open("x", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(OUT / "all_steps.csv", [
        {key: value for key, value in row.items() if key not in ("fusion", "rre_layers")}
        for row in all_rows
    ])
    import matplotlib.pyplot as plt
    for arm in ARMS:
        plt.figure(figsize=(9, 5))
        for branch, mode in (("true", "correct"), ("free", "correct"),
                             ("free", "target_drop"), ("free", "remove"), ("free", "shuffle_all")):
            values = [statistics.mean(row["psnr"] for row in all_rows
                                      if row["arm"] == arm and row["branch"] == branch
                                      and row["mode"] == mode and row["step"] == step)
                      for step in range(50)]
            plt.plot(range(50), values, label=f"{branch}/{mode}")
        plt.xlabel("native UniPC step")
        plt.ylabel("target-view decoded clean estimate PSNR (dB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"{arm}_trajectory_psnr.png", dpi=180)
        plt.close()


def self_check():
    assert IMAGE_STEPS[-1] == 49 and len(MODES) == 4
    gap = [0.0, 1.1, 1.2, 1.3] + [0.0] * 46
    assert next((i for i in range(48) if all(gap[j] > 1 for j in range(i, i + 3))), None) == 1
    assert len(set(SEEDS)) == 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "smoke", "run", "review", "self-check"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--gpu", type=int, choices=(1, 3))
    args = parser.parse_args()
    if args.action == "self-check":
        self_check()
    elif args.action == "preflight":
        print(json.dumps(protocol(), indent=2))
    elif args.action == "review":
        review()
    else:
        if args.arm is None or args.gpu is None:
            parser.error("smoke/run require --arm and --gpu")
        run(args.arm, args.gpu, smoke=args.action == "smoke")
