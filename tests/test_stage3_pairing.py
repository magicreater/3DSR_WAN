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


def test_target_camera_rank_keeps_correct_flow_on_all_views(runner):
    target = torch.zeros(1, 1, 4, 2, 2)
    correct = target.clone()
    correct[:, :, 1:] = 1
    wrong = correct.clone()
    wrong[:, :, 0] = 2
    all_flow, all_correct, _, _ = runner._camera_pair_training_losses(
        correct, wrong, target, margin_ratio=0.05, target_view_only=False,
    )
    target_flow, target_correct, _, _ = runner._camera_pair_training_losses(
        correct, wrong, target, margin_ratio=0.05, target_view_only=True,
    )
    assert torch.equal(all_flow, target_flow)
    assert all_correct.item() > target_correct.item() == 0


def valid_preflight_payload(config, *, training_seed=42):
    batches = []
    for index in range(8):
        valid_pair_identities = [
            {"batch_index": 0, "target_view": target, "source_view": source}
            for target in range(config.views)
            for source in range(config.views)
            if target != source
        ]
        coverage = [{
            **identity,
            "coverage": 0.25,
            "pair_count": 1,
            "target_patch_count": 4,
        } for identity in valid_pair_identities]
        batches.append({
            "seed": training_seed + 6000 + index,
            "scene": config.train_scenes[index % len(config.train_scenes)],
            "view_indices": list(range(config.views)),
            "sigma": 0.5,
            "target_lr_dropped": False,
            "view_count": config.views,
            "target_patch_count": 4,
            "valid_pair_identities": valid_pair_identities,
            "pair_count": len(coverage),
            "coverage_min": 0.25,
            "coverage": coverage,
            "camera_sha256": "a" * 64,
            "wrong_camera_sha256": "b" * 64,
            "batch_sha256": f"{index:064x}",
            "flow_gradient_norm": 2.0,
            "pairing_gradient_norm": 1.0,
        })
    return {
        "protocol": config.pairing_protocol,
        "calibration": {
            "batches": batches,
            "gradient_reduction": "mean_gradient_over_8_batches_then_global_l2",
            "flow_gradient_norm": 2.0,
            "pairing_gradient_norm": 1.0,
            "flow_batch_gradient_norm_mean": 2.0,
            "pairing_batch_gradient_norm_mean": 1.0,
            "raw_weight": 0.5,
            "weight": 0.5,
        },
    }


def valid_resume_payload(runner, config, *, step=1):
    class Conditioner(torch.nn.Module):
        bridge_blocks = (0, 1, 2, 3)
        bridge_time_conditioning = True

        def __init__(self):
            super().__init__()
            self.bridge = torch.nn.Linear(2, 2, bias=False)

    module = Stage3Conditioning(
        Conditioner(),
        geometry=torch.nn.Linear(2, 2, bias=False),
        fusion=torch.nn.Linear(2, 2, bias=False),
    )
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for parameter in module.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    cuda_generator_state = (
        torch.Generator(device="cuda").get_state()
        if torch.cuda.is_available()
        else torch.zeros(16, dtype=torch.uint8)
    )
    numpy_state = runner.np.random.get_state()
    training_state = {
        "format_version": 2,
        "step": step,
        "optimizer": optimizer.state_dict(),
        "sigma_cycle": {
            "steps": 50,
            "shift": 5.0,
            "strategy": "balanced",
            "generator_state": torch.Generator().get_state(),
            "order": torch.arange(113),
            "position": (
                0 if step == 0 else
                ((step * config.gradient_accumulation - 1) % 113) + 1
            ),
            "cycle": (
                0 if step == 0 else
                (step * config.gradient_accumulation - 1) // 113
            ),
        },
        "view_generator_state": torch.Generator().get_state(),
        "noise_generator_state": cuda_generator_state.clone(),
        "dropout_generator_state": torch.Generator().get_state(),
        "pairing_generator_state": torch.Generator().get_state(),
        "python_rng_state": runner.random.getstate(),
        "numpy_rng_state": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [
            cuda_generator_state.clone()
            for _ in range(max(torch.cuda.device_count(), 1))
        ],
        "pairing_weight": 0.5,
    }
    return module, {
        "format": "rl3dsr-stage3",
        "format_version": 1,
        "step": step,
        "config": runner.json.loads(runner.json.dumps(config.to_dict())),
        "bridge_architecture": {
            "blocks": [0, 1, 2, 3],
            "time_conditioning": True,
        },
        "adapters": {
            "bridge": module.conditioner.bridge.state_dict(),
            "geometry": module.geometry.state_dict(),
            "fusion": module.fusion.state_dict(),
        },
        "training_state": training_state,
    }


def truncate_expected_and_coverage(payload):
    batch = payload["calibration"]["batches"][0]
    batch["valid_pair_identities"].pop()
    batch["coverage"].pop()
    batch["pair_count"] -= 1


def calibration_coverage_rows():
    return [
        {
            "batch_index": 0,
            "target_view": target,
            "source_view": source,
            "coverage": 0.25,
            "pair_count": 1,
            "target_patch_count": 4,
        }
        for target in range(3)
        for source in range(3)
        if target != source
    ]


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
    assert state["target_patch_count"] == 6
    assert len(state["valid_pair_identities"]) == 6


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
    artifact.write_text(runner.json.dumps(valid_preflight_payload(config)))
    manifest = {"pairing": {
        "enabled": True,
        "weight": 0.5,
        "protocol": config.pairing_protocol,
        "preflight": str(artifact),
        "preflight_sha256": runner._sha256(artifact),
    }, "provenance": {"training_seed": 42}}
    checkpoint = {"training_state": {"pairing_weight": 0.5}}
    assert runner.validate_pairing_resume(manifest, checkpoint, config) == 0.5

    artifact.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        runner.validate_pairing_resume(manifest, checkpoint, config)
    artifact.unlink()
    with pytest.raises(ValueError, match="missing"):
        runner.validate_pairing_resume(manifest, checkpoint, config)

    artifact.write_text(runner.json.dumps(valid_preflight_payload(config)))
    manifest["pairing"]["preflight_sha256"] = runner._sha256(artifact)
    manifest["pairing"]["protocol"] = {**config.pairing_protocol, "temperature": 0.08}
    with pytest.raises(ValueError, match="protocol"):
        runner.validate_pairing_resume(manifest, checkpoint, config)
    manifest["pairing"]["protocol"] = config.pairing_protocol
    manifest["pairing"]["weight"] = 11.0
    with pytest.raises(ValueError, match="weight"):
        runner.validate_pairing_resume(manifest, checkpoint, config)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload["calibration"].update({
            "gradient_reduction": "mean_of_batch_gradient_norms"
        }), "reduction"),
        (lambda payload: payload["calibration"]["batches"].pop(), "8 batch"),
        (lambda payload: payload["calibration"].update({
            "pairing_gradient_norm": float("nan")
        }), "gradient norm"),
        (lambda payload: payload["calibration"]["batches"][0].update({
            "seed": 999
        }), "seed"),
        (lambda payload: payload["calibration"]["batches"][0]["coverage"][0].update({
            "coverage": 0.01
        }), "coverage"),
        (lambda payload: payload["calibration"]["batches"][0]["coverage"].pop(),
         "identities"),
        (lambda payload: payload["calibration"]["batches"][0][
            "valid_pair_identities"
        ].pop(), "identities"),
        (truncate_expected_and_coverage, "identities"),
        (lambda payload: payload["calibration"]["batches"][0]["coverage"][0].update({
            "batch_index": 1
        }), "batch_index"),
        (lambda payload: payload["calibration"]["batches"][0]["coverage"][0].update({
            "pair_count": 2
        }), "coverage"),
        (lambda payload: payload["calibration"]["batches"][0]["coverage"][0].update({
            "coverage": 0.25 + 1e-11
        }), "coverage"),
        (lambda payload: payload["calibration"]["batches"][0].update({
            "batch_sha256": "not-a-hash"
        }), "hash"),
        (lambda payload: payload["calibration"].update({
            "weight": 0.6
        }), "clipped weight"),
    ],
)
def test_a5_resume_rejects_invalid_corrected_preflight_schema(
    runner, tmp_path, mutation, match
):
    config = a5_config()
    payload = valid_preflight_payload(config)
    mutation(payload)
    artifact = tmp_path / "pairing_preflight.json"
    artifact.write_text(runner.json.dumps(payload))
    manifest = {
        "provenance": {"training_seed": 42},
        "pairing": {
            "enabled": True,
            "weight": 0.5,
            "protocol": config.pairing_protocol,
            "preflight": str(artifact),
            "preflight_sha256": runner._sha256(artifact),
        },
    }
    checkpoint = {"training_state": {"pairing_weight": 0.5}}
    with pytest.raises(ValueError, match=match):
        runner.validate_pairing_resume(manifest, checkpoint, config)


def test_a5_resume_rejects_truncated_legacy_mean_of_norms_artifact(runner, tmp_path):
    config = a5_config()
    artifact = tmp_path / "pairing_preflight.json"
    artifact.write_text(runner.json.dumps({
        "protocol": config.pairing_protocol,
        "calibration": {
            "batches": [],
            "flow_gradient_norm_mean": 2.0,
            "pairing_gradient_norm_mean": 1.0,
            "raw_weight": 0.5,
            "weight": 0.5,
        },
    }))
    manifest = {
        "provenance": {"training_seed": 42},
        "pairing": {
            "enabled": True,
            "weight": 0.5,
            "protocol": config.pairing_protocol,
            "preflight": str(artifact),
            "preflight_sha256": runner._sha256(artifact),
        },
    }
    with pytest.raises(ValueError, match="schema"):
        runner.validate_pairing_resume(
            manifest, {"training_state": {"pairing_weight": 0.5}}, config
        )


@pytest.mark.parametrize(
    "defect,match",
    [
        ("missing_pairing_generator_state", "pairing_generator_state"),
        ("missing_dropout_generator_state", "dropout_generator_state"),
        ("missing_cuda_rng_state", "cuda_rng_state_all"),
        ("wrong_cuda_rng_count", "CUDA RNG"),
        ("legacy_training_state", "format v2"),
        ("short_torch_rng_state", "torch RNG"),
        ("bad_numpy_bit_generator", "NumPy RNG"),
        ("empty_optimizer_state", "optimizer"),
        ("empty_sigma_cycle_state", "sigma cycle"),
        ("optimizer_string_lr", "optimizer"),
        ("optimizer_mismatched_lr", "optimizer"),
        ("optimizer_wrong_moment_shape", "optimizer"),
        ("optimizer_nonfinite_moment", "optimizer"),
        ("sigma_float_steps", "sigma cycle"),
        ("sigma_bool_position", "sigma cycle"),
        ("config_mismatch", "config mismatch"),
        ("adapter_architecture", "fusion architecture"),
        ("nonfinite_adapter", "nonfinite"),
    ],
)
def test_failed_resume_checkpoint_prevalidation_preserves_every_rng(
    runner, tmp_path, monkeypatch, defect, match
):
    config = a5_config(steps=2)
    module, payload = valid_resume_payload(runner, config)
    if defect == "missing_pairing_generator_state":
        payload["training_state"].pop("pairing_generator_state")
    elif defect == "missing_dropout_generator_state":
        payload["training_state"].pop("dropout_generator_state")
    elif defect == "missing_cuda_rng_state":
        payload["training_state"].pop("cuda_rng_state_all")
    elif defect == "wrong_cuda_rng_count":
        payload["training_state"]["cuda_rng_state_all"].append(
            payload["training_state"]["cuda_rng_state_all"][0].clone()
        )
    elif defect == "legacy_training_state":
        payload["training_state"]["format_version"] = 1
    elif defect == "short_torch_rng_state":
        payload["training_state"]["torch_rng_state"] = torch.zeros(1, dtype=torch.uint8)
    elif defect == "bad_numpy_bit_generator":
        payload["training_state"]["numpy_rng_state"]["bit_generator"] = "NOT_MT19937"
    elif defect == "empty_optimizer_state":
        payload["training_state"]["optimizer"] = {}
    elif defect == "empty_sigma_cycle_state":
        payload["training_state"]["sigma_cycle"] = {}
    elif defect == "optimizer_string_lr":
        payload["training_state"]["optimizer"]["param_groups"][0]["lr"] = "oops"
    elif defect == "optimizer_mismatched_lr":
        payload["training_state"]["optimizer"]["param_groups"][0]["lr"] *= 2
    elif defect == "optimizer_wrong_moment_shape":
        first = next(iter(payload["training_state"]["optimizer"]["state"].values()))
        first["exp_avg"] = torch.zeros(1)
    elif defect == "optimizer_nonfinite_moment":
        first = next(iter(payload["training_state"]["optimizer"]["state"].values()))
        first["exp_avg"].flatten()[0] = float("nan")
    elif defect == "sigma_float_steps":
        payload["training_state"]["sigma_cycle"]["steps"] = 50.0
    elif defect == "sigma_bool_position":
        payload["training_state"]["sigma_cycle"]["position"] = True
    elif defect == "config_mismatch":
        payload["config"]["learning_rate"] *= 2
    elif defect == "adapter_architecture":
        payload["adapters"]["fusion"].pop("weight")
    else:
        payload["adapters"]["fusion"]["weight"][0, 0] = float("nan")

    checkpoint = tmp_path / f"{defect}.pt"
    runner.torch.save(payload, checkpoint)
    monkeypatch.setattr(runner, "_checkpoint_validation_module", lambda _config: module)
    monkeypatch.setattr(runner, "_validate_runtime_inputs", lambda *_args, **_kwargs: None)
    cuda_state = [torch.arange(16, dtype=torch.uint8)]
    monkeypatch.setattr(
        runner.torch.cuda, "get_rng_state_all", lambda: [value.clone() for value in cuda_state]
    )
    args = type("Args", (), {
        "seed": 42,
        "resume": checkpoint,
        "init_checkpoint": None,
        "init_reset_fusion": False,
        "output_dir": tmp_path / "run",
    })()
    python_before = runner.random.getstate()
    numpy_before = runner.np.random.get_state()
    torch_before = runner.torch.get_rng_state()
    cuda_before = runner.torch.cuda.get_rng_state_all()

    with pytest.raises(ValueError, match=match):
        runner._train(args, config)

    assert runner.random.getstate() == python_before
    numpy_after = runner.np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert runner.np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert runner.torch.equal(runner.torch.get_rng_state(), torch_before)
    assert all(
        runner.torch.equal(after, before)
        for after, before in zip(runner.torch.cuda.get_rng_state_all(), cuda_before)
    )


@pytest.mark.parametrize(
    "defect",
    [
        "integer_lr_equal_to_float_config",
        "missing_parameter_state",
        "unexpected_amsgrad_moment",
        "parameter_step_tensor",
        "integer_step_tensor",
        "wrong_finite_step",
        "parameter_moment_tensor",
    ],
)
def test_adamw_v2_resume_schema_requires_exact_runtime_types_and_state(
    runner, defect
):
    config = a5_config(learning_rate=1.0, weight_decay=1.0)
    module, payload = valid_resume_payload(runner, config)
    optimizer = payload["training_state"]["optimizer"]
    first_id = next(iter(optimizer["state"]))
    first_state = optimizer["state"][first_id]
    if defect == "integer_lr_equal_to_float_config":
        optimizer["param_groups"][0]["lr"] = 1
    elif defect == "missing_parameter_state":
        optimizer["state"].pop(first_id)
    elif defect == "unexpected_amsgrad_moment":
        first_state["max_exp_avg_sq"] = first_state["exp_avg_sq"].clone()
    elif defect == "parameter_step_tensor":
        first_state["step"] = torch.nn.Parameter(first_state["step"].clone())
    elif defect == "integer_step_tensor":
        first_state["step"] = first_state["step"].to(torch.int64)
    elif defect == "wrong_finite_step":
        first_state["step"] = first_state["step"] + 1
    else:
        first_state["exp_avg"] = torch.nn.Parameter(first_state["exp_avg"].clone())

    with pytest.raises(ValueError, match="optimizer"):
        runner._validate_resume_optimizer_sigma(
            payload["training_state"], module, config
        )


@pytest.mark.parametrize(
    "defect", ["integer_shift_equal_to_float_config", "tensor_subclass_order"]
)
def test_sigma_cycle_v2_resume_schema_requires_exact_scalar_and_tensor_types(
    runner, defect
):
    config = a5_config(sampling_shift=5.0)
    module, payload = valid_resume_payload(runner, config)
    sigma = payload["training_state"]["sigma_cycle"]
    if defect == "integer_shift_equal_to_float_config":
        sigma["shift"] = 5
    else:
        class TensorSubclass(torch.Tensor):
            pass

        sigma["order"] = sigma["order"].as_subclass(TensorSubclass)

    with pytest.raises(ValueError, match="sigma cycle"):
        runner._validate_resume_optimizer_sigma(
            payload["training_state"], module, config
        )


def test_sigma_cycle_resume_state_must_match_consumed_microbatches(runner):
    config = a5_config(gradient_accumulation=2)
    module, payload = valid_resume_payload(runner, config, step=1)
    sigma = payload["training_state"]["sigma_cycle"]
    sigma["cycle"] = 99
    sigma["position"] = 0

    with pytest.raises(ValueError, match="sigma cycle"):
        runner._validate_resume_optimizer_sigma(
            payload["training_state"], module, config
        )


def test_prevalidate_accepts_legacy_a4_training_state(runner, monkeypatch):
    config = Stage3Config(
        arm="A4",
        fusion_dim=192,
        fusion_heads=1,
        epipolar_attention="local_band",
        allow_self_view_source=False,
        steps=2,
    )
    module, payload = valid_resume_payload(runner, config, step=1)
    state = payload["training_state"]
    state["format_version"] = 1
    state.pop("dropout_generator_state")
    state.pop("cuda_rng_state_all")
    state.pop("pairing_generator_state")
    state.pop("pairing_weight")
    monkeypatch.setattr(runner, "_checkpoint_validation_module", lambda _config: module)

    runner._prevalidate_resume_payload(payload, config, expected_step=1)


def test_resume_schema_factory_is_meta_only_and_preserves_global_rng(runner):
    python_before = runner.random.getstate()
    numpy_before = runner.np.random.get_state()
    torch_before = runner.torch.get_rng_state()

    module = runner._checkpoint_validation_module(a5_config())

    assert all(parameter.is_meta for parameter in module.parameters())
    assert runner.random.getstate() == python_before
    numpy_after = runner.np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert runner.np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert runner.torch.equal(runner.torch.get_rng_state(), torch_before)


def test_failed_a5_resume_integrity_does_not_mutate_rng(runner, tmp_path, monkeypatch):
    config = a5_config(steps=2)
    output = tmp_path / "run"
    output.mkdir()
    checkpoint = tmp_path / "resume.pt"
    runner.torch.save({
        "format": "rl3dsr-stage3",
        "format_version": 1,
        "step": 1,
        "training_state": {"pairing_weight": 0.5},
    }, checkpoint)
    (output / "train_steps.jsonl").write_text('{"step": 1}\n')
    artifact = output / "pairing_preflight.json"
    artifact.write_text("{}")
    (output / "run_manifest.json").write_text(runner.json.dumps({
        "config": config.to_dict(),
        "provenance": {"training_seed": 42},
        "pairing": {
            "enabled": True,
            "weight": 0.5,
            "protocol": config.pairing_protocol,
            "preflight": str(artifact),
            "preflight_sha256": runner._sha256(artifact),
        },
    }))
    monkeypatch.setattr(runner, "_validate_runtime_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_prevalidate_resume_payload", lambda *_args: None)
    args = type("Args", (), {
        "seed": 42,
        "resume": checkpoint,
        "init_checkpoint": None,
        "init_reset_fusion": False,
        "output_dir": output,
    })()
    python_before = runner.random.getstate()
    numpy_before = runner.np.random.get_state()
    torch_before = runner.torch.get_rng_state()

    with pytest.raises(ValueError, match="schema"):
        runner._train(args, config)

    assert runner.random.getstate() == python_before
    numpy_after = runner.np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert runner.np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert runner.torch.equal(runner.torch.get_rng_state(), torch_before)


def test_pairing_preflight_is_immutable_and_manifest_binds_its_hash(runner, tmp_path):
    config = a5_config()
    calibration = valid_preflight_payload(config)["calibration"]
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


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate", "missing", "wrong_identity", "unhashable_identity",
        "perturbed_coverage",
    ],
)
def test_pairing_preflight_rejects_bad_cartesian_coverage_before_writing(
    runner, tmp_path, defect
):
    config = a5_config()
    calibration = valid_preflight_payload(config)["calibration"]
    rows = calibration["batches"][0]["coverage"]
    if defect == "duplicate":
        rows[-1] = dict(rows[0])
    elif defect == "missing":
        rows.pop()
    elif defect == "wrong_identity":
        rows[-1]["target_view"] = rows[-1]["source_view"]
    elif defect == "unhashable_identity":
        calibration["batches"][0]["valid_pair_identities"][0]["target_view"] = [0]
    else:
        rows[0]["coverage"] += 1e-11

    with pytest.raises(ValueError, match="calibration coverage"):
        runner.persist_pairing_preflight(tmp_path, config, calibration)
    assert not (tmp_path / "pairing_preflight.json").exists()


def test_calibration_coverage_requires_the_complete_directed_cartesian_set(runner):
    assert runner._validate_calibration_coverage(
        calibration_coverage_rows(), view_count=3, target_patch_count=4
    ) == [
        {"batch_index": 0, "target_view": 0, "source_view": 1},
        {"batch_index": 0, "target_view": 0, "source_view": 2},
        {"batch_index": 0, "target_view": 1, "source_view": 0},
        {"batch_index": 0, "target_view": 1, "source_view": 2},
        {"batch_index": 0, "target_view": 2, "source_view": 0},
        {"batch_index": 0, "target_view": 2, "source_view": 1},
    ]


@pytest.mark.parametrize("defect", ["duplicate", "missing", "perturbed_coverage"])
def test_calibration_coverage_rejects_incomplete_or_inexact_rows(runner, defect):
    rows = calibration_coverage_rows()
    if defect == "duplicate":
        rows.append(dict(rows[0]))
    elif defect == "missing":
        rows.pop()
    else:
        rows[0]["coverage"] += 1e-11
    with pytest.raises(ValueError, match="calibration coverage"):
        runner._validate_calibration_coverage(
            rows, view_count=3, target_patch_count=4
        )


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
            q = torch.stack((self.fusion.weight, torch.ones_like(self.fusion.weight))).reshape(
                1, 1, 1, 2
            ).expand(1, 1, 3, 2)
            identities = [
                {"batch_index": 0, "target_view": 0, "source_view": 1},
                {"batch_index": 0, "target_view": 0, "source_view": 2},
                {"batch_index": 0, "target_view": 1, "source_view": 0},
                {"batch_index": 0, "target_view": 1, "source_view": 2},
                {"batch_index": 0, "target_view": 2, "source_view": 0},
                {"batch_index": 0, "target_view": 2, "source_view": 1},
            ]
            return torch.zeros(1, 3, 1), {
                "query": q,
                "key": torch.tensor([[[[1., 0.], [1., 0.], [1., 0.]]]]),
                "wrong_key": torch.tensor([[[[0., 1.], [0., 1.], [0., 1.]]]]),
                "wrong_allowed": torch.tensor([[[False, True, False],
                                                  [False, False, True],
                                                  [True, False, False]]]),
                "patches": 1,
                "target_patch_count": 1,
                "view_count": 3,
                # Deliberately truncated: calibration must derive its oracle.
                "valid_pair_identities": identities[:-1],
                "pairs": [{**identity,
                    "target_patches": torch.tensor([0]),
                    "source_patches": torch.tensor([0]), "coverage": 1.0,
                } for identity in identities],
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
    assert all(row["view_count"] == 3 for row in calibration["batches"])
    assert all(row["target_patch_count"] == 1 for row in calibration["batches"])
    assert all(len(row["valid_pair_identities"]) == 6 for row in calibration["batches"])
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
