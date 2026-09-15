"""CPU-only contracts for optional A5 direct patch-pair supervision."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch

from rl3dsr.models.wan.geometry_conditioning import CameraBatch
from rl3dsr.models.wan.lr_fusion import (
    LRViewFusion,
    mutual_epipolar_patch_pairs,
    pairing_info_nce_loss,
)
from rl3dsr.models.wan.stage3 import Stage3Conditioning
from rl3dsr.validation.stage3_protocol import Stage3Config


def camera(views=3):
    k = torch.tensor([[24., 0., 16.], [0., 20., 12.], [0., 0., 1.]]).repeat(1, views, 1, 1)
    transform = torch.eye(4).repeat(1, views, 1, 1)
    transform[0, :, 0, 3] = torch.arange(views) * .25
    return CameraBatch(k, transform, (24, 32), "multiview")


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/stage3_experiment.py"
    spec = importlib.util.spec_from_file_location("stage3_pairing_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def a5_config(**changes):
    config = Stage3Config(
        arm="A5",
        fusion_dim=192,
        fusion_heads=1,
        epipolar_attention="local_band",
        allow_self_view_source=False,
        **changes,
    )
    return config


def test_a5_is_explicit_and_uses_the_fixed_pairing_protocol():
    config = a5_config()
    assert config.fusion_mode == "rre_epipolar"
    assert config.pairing_supervision is True
    assert config.pairing_protocol == {
        "temperature": 0.07,
        "minimum_coverage": 0.05,
        "calibration_batches": 8,
        "target_gradient_ratio": 0.25,
        "weight_clip": [0.01, 10.0],
    }
    assert replace(config, arm="A4").pairing_supervision is False
    with pytest.raises(ValueError, match="temperature=0.07"):
        replace(config, pairing_temperature=0.08)
    with pytest.raises(ValueError, match="minimum_coverage=0.05"):
        replace(config, pairing_minimum_coverage=0.1)
    with pytest.raises(ValueError, match="calibration_batches=8"):
        replace(config, pairing_calibration_batches=7)
    with pytest.raises(ValueError, match="A5 requires fusion_dim=192"):
        replace(config, fusion_dim=96)
    with pytest.raises(ValueError, match="pairing_target_gradient_ratio=0.25"):
        replace(config, pairing_target_gradient_ratio=float("nan"))


def test_mutual_nearest_pairs_are_selected_per_ordered_view_pair():
    features = torch.tensor([[[[1., 0.], [0., 1.], [-1., 0.]],
                              [[1., 0.], [0., 1.], [-1., 0.]]]])
    allowed = torch.zeros(1, 6, 6, dtype=torch.bool)
    for patch in range(3):
        allowed[0, patch, 3 + patch] = True
        allowed[0, 3 + patch, patch] = True
    usable = torch.ones(1, 6, 2, dtype=torch.bool)

    pairs = mutual_epipolar_patch_pairs(
        features, allowed, usable, minimum_coverage=0.05
    )

    assert [(row["target_view"], row["source_view"]) for row in pairs] == [(0, 1), (1, 0)]
    assert all(row["coverage"] == 1.0 for row in pairs)
    assert all(torch.equal(row["target_patches"], torch.arange(3)) for row in pairs)
    assert all(torch.equal(row["source_patches"], torch.arange(3)) for row in pairs)
    assert all(not row["target_patches"].requires_grad for row in pairs)


def test_pair_coverage_fails_closed_for_any_valid_view_pair():
    features = torch.tensor([[[[1., 0.], [0., 1.], [-1., 0.]],
                              [[1., 0.], [0., 1.], [-1., 0.]]]])
    allowed = torch.zeros(1, 6, 6, dtype=torch.bool)
    allowed[0, 0, 3] = True
    allowed[0, 3:, :3] = True
    usable = torch.ones(1, 6, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"batch 0 target 0 source 1.*0\.333.*0\.500"):
        mutual_epipolar_patch_pairs(
            features, allowed, usable, minimum_coverage=0.5
        )


def test_unusable_query_rows_cannot_form_pairs_and_remain_in_coverage_denominator():
    features = torch.tensor([[[[1., 0.], [0., 1.], [-1., 0.]],
                              [[1., 0.], [0., 1.], [-1., 0.]]]])
    allowed = torch.zeros(1, 6, 6, dtype=torch.bool)
    for patch in range(3):
        allowed[0, patch, 3 + patch] = True
        allowed[0, 3 + patch, patch] = True
    usable = torch.ones(1, 6, 2, dtype=torch.bool)
    usable[0, 1, 1] = False

    pairs = mutual_epipolar_patch_pairs(
        features, allowed, usable, minimum_coverage=0.5
    )
    forward = next(
        row for row in pairs
        if row["target_view"] == 0 and row["source_view"] == 1
    )
    assert forward["coverage"] == pytest.approx(2 / 3)
    assert torch.equal(forward["target_patches"], torch.tensor([0, 2]))
    assert torch.equal(forward["source_patches"], torch.tensor([0, 2]))


def _manual_pairing_state(correct_positive: bool):
    q = torch.tensor([[[[1., 0.], [0., 1.], [0., 0.], [0., 0.]]]], requires_grad=True)
    if correct_positive:
        k = torch.tensor([[[[0., 0.], [0., 0.], [1., 0.], [0., 1.]]]], requires_grad=True)
        wrong_k = torch.tensor([[[[0., 0.], [0., 0.], [-1., 0.], [0., -1.]]]], requires_grad=True)
    else:
        k = torch.tensor([[[[0., 0.], [0., 0.], [-1., 0.], [0., -1.]]]], requires_grad=True)
        wrong_k = torch.tensor([[[[0., 0.], [0., 0.], [1., 0.], [0., 1.]]]], requires_grad=True)
    wrong_allowed = torch.zeros(1, 4, 4, dtype=torch.bool)
    wrong_allowed[0, :2, 2:] = True
    return {
        "query": q,
        "key": k,
        "wrong_key": wrong_k,
        "wrong_allowed": wrong_allowed,
        "patches": 2,
        "pairs": [{
            "batch_index": 0,
            "target_view": 0,
            "source_view": 1,
            "target_patches": torch.tensor([0, 1]),
            "source_patches": torch.tensor([0, 1]),
            "coverage": 1.0,
        }],
    }


def test_info_nce_prefers_correct_qk_over_wrong_camera_hard_negatives():
    good = pairing_info_nce_loss(_manual_pairing_state(True), temperature=0.07)
    bad = pairing_info_nce_loss(_manual_pairing_state(False), temperature=0.07)
    assert good < 1e-5
    assert bad > 10


def test_pairing_state_detaches_lr_features_but_keeps_qk_projection_gradients():
    torch.manual_seed(3)
    model = LRViewFusion(
        12, 8, 1, mode="rre_epipolar", allow_self_view_source=False
    )
    features = torch.randn(1, 3, 6, 12, requires_grad=True)
    correct = camera()
    order = torch.tensor([0, 2, 1])
    wrong = replace(
        correct,
        K=correct.K[:, order],
        T_world_from_camera=correct.T_world_from_camera[:, order],
    )
    state = model.build_pairing_state(
        features, correct, wrong, (2, 3), minimum_coverage=0.05
    )
    loss = pairing_info_nce_loss(state, temperature=0.07)
    loss.backward()
    assert features.grad is None
    assert model.qkv.weight.grad is not None
    assert float(model.qkv.weight.grad.abs().sum()) > 0
    assert model.output.weight.grad is None
    assert len(state["pairs"]) == 6


def test_pairing_state_rejects_distorted_ucm_epipolar_geometry():
    model = LRViewFusion(
        12, 8, 1, mode="rre_epipolar", allow_self_view_source=False
    )
    features = torch.randn(1, 3, 6, 12)
    correct = camera()
    distorted = replace(
        correct,
        camera_model="ucm",
        xi=torch.full((1, 3), 0.25),
    )
    with pytest.raises(ValueError, match="nonzero UCM distortion"):
        model.build_pairing_state(features, correct, distorted, (2, 3))


def test_prepare_multiview_pairing_is_opt_in_and_keeps_default_return_type():
    torch.manual_seed(5)
    features = torch.randn(1, 18, 12)

    class Conditioner(torch.nn.Module):
        def multiview_features(self, *_args, **_kwargs):
            return features

    fusion = LRViewFusion(
        12, 8, 1, mode="rre_epipolar", allow_self_view_source=False
    )
    module = Stage3Conditioning(Conditioner(), fusion=fusion)
    correct = camera()
    order = torch.tensor([0, 2, 1])
    wrong = replace(
        correct,
        K=correct.K[:, order],
        T_world_from_camera=correct.T_world_from_camera[:, order],
    )
    lr = torch.zeros(1, 3, 3, 8, 8)
    default = module.prepare_multiview(lr, correct, (3, 4, 6), (24, 32))
    prepared, state = module.prepare_multiview(
        lr,
        correct,
        (3, 4, 6),
        (24, 32),
        pairing_camera=wrong,
        pairing_minimum_coverage=0.05,
    )
    assert isinstance(default, torch.Tensor)
    assert torch.equal(default, prepared)
    assert state["pairs"]


def test_flow_target_dropout_does_not_change_pairing_matches_or_qk_state():
    class Conditioner(torch.nn.Module):
        def multiview_features(self, lr, **_kwargs):
            values = lr[:, 0, :, 0, 0]
            return values[:, :, None, None].expand(-1, -1, 6, 12).reshape(1, 18, 12)

    torch.manual_seed(13)
    module = Stage3Conditioning(
        Conditioner(),
        fusion=LRViewFusion(12, 8, 1, mode="rre_epipolar", allow_self_view_source=False),
    )
    correct = camera()
    wrong = replace(
        correct,
        K=correct.K[:, torch.tensor([0, 2, 1])],
        T_world_from_camera=correct.T_world_from_camera[:, torch.tensor([0, 2, 1])],
    )
    original = torch.ones(1, 3, 3, 8, 8)
    original[:, 0, 1] = 2
    original[:, 0, 2] = 3
    dropped = original.clone()
    dropped[:, :, 0] = 0

    original_prepared, original_state = module.prepare_multiview(
        original, correct, (3, 4, 6), (24, 32), pairing_camera=wrong
    )
    dropped_prepared, dropped_state = module.prepare_multiview(
        dropped,
        correct,
        (3, 4, 6),
        (24, 32),
        pairing_camera=wrong,
        pairing_lr=original,
    )

    assert not torch.equal(original_prepared, dropped_prepared)
    for key in ("query", "key", "wrong_key"):
        assert torch.equal(original_state[key], dropped_state[key])
    assert [
        (row["batch_index"], row["target_view"], row["source_view"],
         row["target_patches"].tolist(), row["source_patches"].tolist())
        for row in original_state["pairs"]
    ] == [
        (row["batch_index"], row["target_view"], row["source_view"],
         row["target_patches"].tolist(), row["source_patches"].tolist())
        for row in dropped_state["pairs"]
    ]


def test_calibration_is_deterministic_and_clips_without_parameter_mutation(runner):
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    def factory(index):
        scale = float(index + 1)
        return {
            "flow_loss": parameter * scale,
            "pairing_loss": parameter * (2 * scale),
            "metadata": {"seed": 6000 + index, "coverage_min": 1.0, "batch_sha256": str(index)},
        }

    first = runner.calibrate_pairing_weight(
        factory, [parameter], batch_count=8, target_gradient_ratio=0.25,
        weight_clip=(0.01, 10.0),
    )
    second = runner.calibrate_pairing_weight(
        factory, [parameter], batch_count=8, target_gradient_ratio=0.25,
        weight_clip=(0.01, 10.0),
    )
    assert first == second
    assert first["raw_weight"] == pytest.approx(0.125)
    assert first["weight"] == pytest.approx(0.125)
    assert first["gradient_reduction"] == "mean_gradient_over_8_batches_then_global_l2"
    assert first["flow_gradient_norm"] == pytest.approx(4.5)
    assert first["pairing_gradient_norm"] == pytest.approx(9.0)
    assert [row["seed"] for row in first["batches"]] == list(range(6000, 6008))
    assert parameter.item() == 1.0 and parameter.grad is None

    maximum = runner.calibrate_pairing_weight(
        lambda i: {"flow_loss": parameter * 1000, "pairing_loss": parameter,
                   "metadata": {"seed": i, "coverage_min": 1.0, "batch_sha256": str(i)}},
        [parameter], batch_count=8, target_gradient_ratio=0.25,
        weight_clip=(0.01, 10.0),
    )
    minimum = runner.calibrate_pairing_weight(
        lambda i: {"flow_loss": parameter, "pairing_loss": parameter * 1000,
                   "metadata": {"seed": i, "coverage_min": 1.0, "batch_sha256": str(i)}},
        [parameter], batch_count=8, target_gradient_ratio=0.25,
        weight_clip=(0.01, 10.0),
    )
    assert maximum["weight"] == 10.0
    assert minimum["weight"] == 0.01


def test_calibration_norms_the_reduced_gradient_vector_not_mean_batch_norms(runner):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="pairing gradient"):
        runner.calibrate_pairing_weight(
            lambda index: {
                "flow_loss": parameter,
                "pairing_loss": parameter * (1 if index < 4 else -1),
                "metadata": {
                    "seed": index,
                    "coverage_min": 1.0,
                    "batch_sha256": str(index),
                },
            },
            [parameter],
            batch_count=8,
            target_gradient_ratio=0.25,
            weight_clip=(0.01, 10.0),
        )


def test_resume_preserves_pairing_weight_and_rng_while_init_recalibrates(runner):
    manifest = {"pairing": {"weight": 0.4}}
    checkpoint = {"training_state": {"pairing_weight": 0.4}}
    calls = []
    assert runner.resolve_pairing_weight(
        True, "resume", manifest, checkpoint, lambda: calls.append("calibrate")
    ) == 0.4
    assert calls == []
    assert runner.resolve_pairing_weight(
        True, "model_only", None, checkpoint, lambda: calls.append("calibrate") or {"weight": 0.7}
    ) == 0.7
    assert calls == ["calibrate"]
    assert runner.resolve_pairing_weight(
        False, "scratch", None, None, lambda: pytest.fail("calibrated A4")
    ) == 0.0

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))])

    class Cycle:
        def state_dict(self):
            return {"value": 1}

    pairing = torch.Generator().manual_seed(17)
    state = runner._training_state(
        optimizer, Cycle(), torch.Generator(), torch.Generator(), 3,
        pairing_generator=pairing, pairing_weight=0.4,
    )
    assert state["pairing_weight"] == 0.4
    assert torch.equal(state["pairing_generator_state"], pairing.get_state())


def test_a5_resume_validates_protocol_weight_and_preflight_hash(runner, tmp_path):
    config = a5_config()
    artifact = tmp_path / "pairing_preflight.json"
    artifact.write_text(runner.json.dumps({
        "protocol": config.pairing_protocol,
        "calibration": {"weight": 0.4},
    }))
    manifest = {"pairing": {
        "enabled": True,
        "weight": 0.4,
        "protocol": config.pairing_protocol,
        "preflight": str(artifact),
        "preflight_sha256": runner._sha256(artifact),
    }}
    checkpoint = {"training_state": {"pairing_weight": 0.4}}
    assert runner.validate_pairing_resume(manifest, checkpoint, config) == 0.4

    artifact.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        runner.validate_pairing_resume(manifest, checkpoint, config)
    artifact.unlink()
    with pytest.raises(ValueError, match="missing"):
        runner.validate_pairing_resume(manifest, checkpoint, config)

    artifact.write_text(runner.json.dumps({
        "protocol": config.pairing_protocol,
        "calibration": {"weight": 0.4},
    }))
    manifest["pairing"]["preflight_sha256"] = runner._sha256(artifact)
    manifest["pairing"]["protocol"] = {**config.pairing_protocol, "temperature": 0.08}
    with pytest.raises(ValueError, match="protocol"):
        runner.validate_pairing_resume(manifest, checkpoint, config)
    manifest["pairing"]["protocol"] = config.pairing_protocol
    manifest["pairing"]["weight"] = 11.0
    with pytest.raises(ValueError, match="weight"):
        runner.validate_pairing_resume(manifest, checkpoint, config)


def test_pairing_preflight_is_immutable_and_manifest_binds_its_hash(runner, tmp_path):
    calibration = {
        "batches": [{"seed": 6000, "coverage_min": 0.2}],
        "flow_gradient_norm_mean": 2.0,
        "pairing_gradient_norm_mean": 1.0,
        "raw_weight": 0.5,
        "weight": 0.5,
    }
    config = a5_config()
    section = runner.persist_pairing_preflight(tmp_path, config, calibration)
    artifact = tmp_path / "pairing_preflight.json"
    payload = runner.json.loads(artifact.read_text())
    assert payload["protocol"] == config.pairing_protocol
    assert payload["calibration"] == calibration
    assert section == {
        "enabled": True,
        "weight": 0.5,
        "protocol": config.pairing_protocol,
        "preflight": str(artifact.resolve()),
        "preflight_sha256": runner._sha256(artifact),
    }
    with pytest.raises(FileExistsError):
        runner.persist_pairing_preflight(tmp_path, config, calibration)


def test_a5_preflight_uses_eight_private_seeds_without_mutating_parameters(
    runner, tmp_path, monkeypatch
):
    parameter = torch.nn.Parameter(torch.tensor(0.7))
    calls = []

    class Fusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = parameter

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion = Fusion()
            self.register_buffer("running_probe", torch.tensor(0.0))
            self.cache = {"token": torch.tensor(3.0)}
            self.last_diagnostics = {"before": torch.tensor(4.0)}

        def prepare_multiview(self, lr, camera, latent_shape, size, **kwargs):
            self.running_probe.add_(1)
            self.cache = {"token": torch.tensor(30.0)}
            self.last_diagnostics = {"after": torch.tensor(40.0)}
            self.training = False
            torch.rand(())
            runner.random.random()
            runner.np.random.rand()
            calls.append(kwargs["pairing_camera"])
            q = torch.stack((self.fusion.weight, torch.ones_like(self.fusion.weight))).reshape(1, 1, 1, 2)
            return torch.zeros(1, 3, 1), {
                "query": q,
                "key": torch.tensor([[[[1., 0.], [1., 0.], [1., 0.]]]]),
                "wrong_key": torch.tensor([[[[0., 1.], [0., 1.], [0., 1.]]]]),
                "wrong_allowed": torch.tensor([[[False, True, False],
                                                  [False, False, True],
                                                  [True, False, False]]]),
                "patches": 1,
                "pairs": [{
                    "batch_index": 0, "target_view": 0, "source_view": 1,
                    "target_patches": torch.tensor([0]),
                    "source_patches": torch.tensor([0]), "coverage": 1.0,
                }],
            }

        def predict(self, _dit, noisy, *_args, **_kwargs):
            _dit.running_probe.add_(1)
            _dit.last_diagnostics = {"after": torch.tensor(80.0)}
            _dit.last_injection_stats = {"after": torch.tensor(90.0)}
            _dit.training = False
            return noisy * 0 + self.fusion.weight

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_probe", torch.tensor(5.0))
            self.cache = {"token": torch.tensor(6.0)}

        def encode_multiview(self, hr):
            self.running_probe.add_(1)
            self.cache = {"token": torch.tensor(60.0)}
            self.training = False
            return hr

    class DIT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_probe", torch.tensor(7.0))
            self.last_diagnostics = {"before": torch.tensor(8.0)}
            self.last_injection_stats = {"before": torch.tensor(9.0)}

    transform = torch.eye(4).repeat(1, 3, 1, 1)
    transform[0, :, 0, 3] = torch.arange(3)
    cam = CameraBatch(torch.eye(3).repeat(1, 3, 1, 1), transform, (16, 16), "multiview")
    hr = torch.ones(1, 1, 3, 1, 1)
    monkeypatch.setattr(
        runner, "_load_group",
        lambda _root, _scene, _split, _config, generator, _device:
            ([int(torch.randint(0, 1000, (), generator=generator))], hr, hr, cam),
    )
    runtime = runner.Runtime(Module(), VAE(), DIT(), torch.device("cpu"))
    args = type("Args", (), {"seed": 42, "dataset_root": tmp_path})()
    before = parameter.detach().clone()
    buffer_before = runtime.module.running_probe.detach().clone()
    module_cache_before = runtime.module.cache["token"].clone()
    module_diagnostics_before = runtime.module.last_diagnostics["before"].clone()
    vae_buffer_before = runtime.vae.running_probe.clone()
    vae_cache_before = runtime.vae.cache["token"].clone()
    dit_buffer_before = runtime.dit.running_probe.clone()
    dit_diagnostics_before = runtime.dit.last_diagnostics["before"].clone()
    dit_injection_before = runtime.dit.last_injection_stats["before"].clone()
    python_rng_before = runner.random.getstate()
    numpy_rng_before = runner.np.random.get_state()
    torch_rng_before = torch.get_rng_state()

    calibration = runner._calibrate_a5_pairing(runtime, args, a5_config(
        image_size=16, views=3, train_scenes=("chair",)
    ))

    assert len(calibration["batches"]) == 8
    assert [row["seed"] for row in calibration["batches"]] == list(range(6042, 6050))
    assert len(calls) == 8
    assert torch.equal(parameter, before)
    assert parameter.grad is None
    assert torch.equal(runtime.module.running_probe, buffer_before)
    assert runtime.module.training is True
    assert torch.equal(runtime.module.cache["token"], module_cache_before)
    assert torch.equal(runtime.module.last_diagnostics["before"], module_diagnostics_before)
    assert runtime.vae.training is True
    assert torch.equal(runtime.vae.running_probe, vae_buffer_before)
    assert torch.equal(runtime.vae.cache["token"], vae_cache_before)
    assert runtime.dit.training is True
    assert torch.equal(runtime.dit.running_probe, dit_buffer_before)
    assert torch.equal(runtime.dit.last_diagnostics["before"], dit_diagnostics_before)
    assert torch.equal(runtime.dit.last_injection_stats["before"], dit_injection_before)
    assert runner.random.getstate() == python_rng_before
    numpy_rng_after = runner.np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert torch.equal(torch.from_numpy(numpy_rng_after[1]), torch.from_numpy(numpy_rng_before[1]))
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert all(row["coverage_min"] == 1.0 for row in calibration["batches"])
    assert all(len(row["batch_sha256"]) == 64 for row in calibration["batches"])


def test_a5_step_telemetry_keeps_losses_and_fusion_gradients_separate(runner):
    row = runner.a5_step_telemetry(
        flow_losses=[1.0, 3.0],
        rank_losses=[0.1, 0.3],
        pairing_losses=[0.2, 0.4],
        pair_counts=[5, 7],
        coverage_minima=[0.1, 0.2],
        pairing_weight=0.5,
        flow_gradient_norm=4.0,
        rank_gradient_norm=0.4,
        pairing_gradient_norm=3.0,
        final_fusion_gradient_norm=6.0,
    )
    assert row == {
        "main_flow_loss": 2.0,
        "output_rank_loss": pytest.approx(0.2),
        "pairing_loss": pytest.approx(0.3),
        "weighted_pairing_loss": pytest.approx(0.15),
        "pair_count": 12,
        "pairing_coverage_min": 0.1,
        "pairing_weight": 0.5,
        "flow_fusion_gradient_norm": 4.0,
        "rank_fusion_gradient_norm": 0.4,
        "pairing_fusion_gradient_norm": 3.0,
        "final_fusion_gradient_norm": 6.0,
        "fusion_gradient_reduction": "sum_of_microbatch_mean_loss_gradients_then_global_l2",
    }


def test_calibration_rejects_zero_and_nonfinite_gradients(runner):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="pairing gradient"):
        runner.calibrate_pairing_weight(
            lambda i: {"flow_loss": parameter, "pairing_loss": parameter * 0,
                       "metadata": {"seed": i, "coverage_min": 1.0, "batch_sha256": str(i)}},
            [parameter], batch_count=8, target_gradient_ratio=0.25,
            weight_clip=(0.01, 10.0),
        )
