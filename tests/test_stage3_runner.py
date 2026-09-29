"""Runner contracts using synthetic objects only; never load model assets."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/stage3_experiment.py"
    spec = importlib.util.spec_from_file_location("stage3_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspect_has_no_runtime_or_output_side_effects(runner, tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text('{"arm":"A3"}')
    monkeypatch.setattr(runner, "load_runtime", lambda *a: pytest.fail("loaded model"))
    runner.main(["inspect", "--config", str(config)])
    assert json.loads(capsys.readouterr().out)["arm"] == "A3"
    assert list(tmp_path.iterdir()) == [config]


def test_scene_index_and_cpu_camera_are_reused(runner, tmp_path, monkeypatch):
    from rl3dsr.data import Split
    from rl3dsr.validation.stage3_protocol import Stage3Config

    calls = []
    observation = SimpleNamespace(K=np.eye(3, dtype=np.float32),
                                  T_world_from_camera=np.eye(4, dtype=np.float32),
                                  width=16, height=16)

    class Adapter:
        def __init__(self, root):
            self.root = root

        def index(self, split):
            calls.append(split)
            return SimpleNamespace(observations=(observation,))

    monkeypatch.setattr(runner, "NeRFSyntheticAdapter", Adapter)
    config = Stage3Config()
    runner._scene_data.cache_clear()
    runner._scene_camera.cache_clear()
    try:
        first = runner._scene_camera(tmp_path, config, Split.TRAIN)
        second = runner._scene_camera(tmp_path, config, Split.TRAIN)
        assert first is second
        assert len(calls) == 1
        assert runner._scene_data(tmp_path, config, Split.TRAIN)[1].observations[0] is observation
        assert len(calls) == 1
    finally:
        runner._scene_camera.cache_clear()
        runner._scene_data.cache_clear()


def test_paired_wan_prediction_keeps_correct_wrong_order(runner):
    from rl3dsr.models.wan.geometry_conditioning import CameraBatch

    class Module:
        def predict(self, dit, noisy, timestep, context, features, camera, shape):
            assert dit == "dit" and context is None and shape == (4, 2, 2)
            assert noisy.shape == (2, 16, 4, 2, 2)
            assert timestep.tolist() == [500, 500]
            assert features[:, 0, 0].tolist() == [3, 7]
            assert camera.K.shape == (2, 4, 3, 3)
            return noisy + torch.arange(2).view(2, 1, 1, 1, 1)

    camera = CameraBatch(torch.eye(3).repeat(1, 4, 1, 1),
                         torch.eye(4).repeat(1, 4, 1, 1), (16, 16), "multiview")
    runtime = SimpleNamespace(module=Module(), dit="dit")
    correct, wrong = runner._predict_camera_pair(
        runtime, torch.zeros(1, 16, 4, 2, 2), torch.tensor([500]),
        torch.full((1, 4, 2), 3.0), torch.full((1, 4, 2), 7.0), camera, (4, 2, 2),
    )
    assert torch.count_nonzero(correct) == 0
    assert torch.all(wrong == 1)


def test_missing_inputs_fail_before_runtime(runner, tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{}')
    monkeypatch.setattr(runner, "load_runtime", lambda *a: pytest.fail("loaded model"))
    with pytest.raises((ValueError, SystemExit)):
        runner.main(["train", "--config", str(config)])


def test_train_init_checkpoint_is_mutually_exclusive_with_resume(runner):
    parser = runner._parser()
    common = [
        "train", "--config", "config.json", "--dataset-root", "data",
        "--model-dir", "model", "--lq-source", "lq.py",
        "--lq-checkpoint", "lq.pt", "--bridge-checkpoint", "bridge.pt",
        "--output-dir", "out", "--seed", "42",
    ]
    parsed = parser.parse_args(common + ["--init-checkpoint", "parent.pt", "--init-reset-fusion"])
    assert parsed.init_checkpoint == Path("parent.pt") and parsed.resume is None
    assert parsed.init_reset_fusion is True
    with pytest.raises(SystemExit):
        parser.parse_args(common + ["--resume", "resume.pt", "--init-checkpoint", "parent.pt"])


def test_initialization_provenance_records_resolved_parent_and_saved_config(runner, tmp_path):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"stage3-parent")
    payload = {"step": 1000, "config": {"camera_rank_weight": 0.1, "arm": "A3"}}
    result = runner._initialization_provenance(parent, payload, reset_fusion=True)
    assert result == {
        "initialization_mode": "model_only",
        "parent_checkpoint": str(parent.resolve()),
        "parent_checkpoint_sha256": runner._sha256(parent),
        "parent_saved_step": 1000,
        "parent_saved_config_sha256": runner._json_sha256(payload["config"]),
        "reset_fusion": True,
        "copied_modules": ["bridge", "geometry"],
        "reset_modules": ["fusion"],
    }
    copied = runner._initialization_provenance(parent, payload, reset_fusion=False)
    assert copied["copied_modules"] == ["bridge", "geometry", "fusion"]
    assert copied["reset_modules"] == []
    runner._assert_manifest_initialization(result, "model_only", True)
    with pytest.raises(RuntimeError, match="copied_modules"):
        runner._assert_manifest_initialization(
            {**result, "copied_modules": ["fusion"]}, "model_only", True
        )


def test_init_reset_fusion_requires_model_only_parent(runner):
    assert runner._initialization_mode(None, Path("parent.pt"), True) == "model_only"
    with pytest.raises(ValueError, match="init-reset-fusion"):
        runner._initialization_mode(None, None, True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner._initialization_mode(Path("resume.pt"), Path("parent.pt"), False)


def test_model_only_initialization_never_restores_poisoned_training_state(runner, tmp_path):
    parameter = torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.2)

    class Cycle:
        def __init__(self):
            self.value = 11

        def load_state_dict(self, _state):
            self.value = -1

    cycle = Cycle()
    view = torch.Generator().manual_seed(101)
    noise = torch.Generator().manual_seed(102)
    expected_view = torch.rand(3, generator=torch.Generator().manual_seed(101))
    expected_noise = torch.rand(3, generator=torch.Generator().manual_seed(102))
    poisoned = {
        "format": "rl3dsr-stage3",
        "format_version": 1,
        "step": 1000,
        "config": {"arm": "A3"},
        "training_state": {
            "step": 1000,
            "optimizer": {"poison": True},
            "sigma_cycle": {"poison": True},
            "view_generator_state": torch.Generator().manual_seed(999).get_state(),
            "noise_generator_state": torch.Generator().manual_seed(999).get_state(),
        }
    }
    runner._maybe_restore_training_progress(
        poisoned, "model_only", optimizer, cycle, view, noise
    )
    assert optimizer.param_groups[0]["lr"] == 0.2
    assert cycle.value == 11
    assert torch.equal(torch.rand(3, generator=view), expected_view)
    assert torch.equal(torch.rand(3, generator=noise), expected_noise)

    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    (parent_dir / "train_steps.jsonl").write_text('{"step":1000,"poison":true}\n')
    parent_checkpoint = parent_dir / "stage3_step_1000.pt"
    torch.save(poisoned, parent_checkpoint)
    fresh_output = tmp_path / "fresh"
    start_step, rows, loaded = runner._fresh_training_start(
        fresh_output, parent_checkpoint
    )
    assert start_step == 1 and rows == [] and loaded["step"] == 1000
    assert list(fresh_output.iterdir()) == []


def test_split_routes_do_not_touch_test_during_development(runner):
    from rl3dsr.validation.stage3_protocol import Stage3Config
    config = Stage3Config()
    assert runner.scene_routes(config, "train") == [(s, "train") for s in config.train_scenes]
    assert runner.scene_routes(config, "validate") == [(s, "val") for s in config.validation_scene_names]
    assert runner.scene_routes(config, "validate", validation_only=True) == [(s, "val") for s in config.validation_scenes]
    assert runner.scene_routes(config, "test") == [(s, "test") for s in config.test_scenes]


def test_fusion_prepared_once_and_camera_intervention_isolated(runner):
    calls = []
    original, fusion_camera = object(), object()
    features = torch.ones(1, 8, 4)
    module = SimpleNamespace(
        prepare_multiview=lambda lr, cam, shape, size, **kw: calls.append(("prepare", cam)) or features,
        predict=lambda dit, x, t, c, f, cam, shape: calls.append(("predict", cam)) or torch.zeros_like(x),
    )
    runtime = SimpleNamespace(module=module, dit=object(), device=torch.device("cpu"))
    def sampler(noise, prepared, context, *, predict_velocity, config):
        assert prepared is features
        for _ in range(3):
            predict_velocity(noise, torch.zeros(1), context, prepared)
        return noise
    lr = torch.zeros(1, 3, 2, 2, 2)
    runner.sample_latents(runtime, lr, original, (1, 16, 2, 2, 2), 5, 16,
                          fusion_camera=fusion_camera, sampler=sampler, dtype=torch.float32)
    assert calls == [("prepare", fusion_camera)] + [("predict", original)] * 3


def test_sample_latents_uses_explicit_geometry_camera_and_captures_diagnostics(runner):
    calls = []
    from rl3dsr.models.wan.geometry_conditioning import CameraBatch
    original = CameraBatch(
        torch.eye(3).repeat(1, 2, 1, 1),
        torch.eye(4).repeat(1, 2, 1, 1),
        (16, 16),
        "multiview",
    )
    fusion_camera = original
    geometry_camera = original
    features = torch.ones(1, 8, 4)
    module = SimpleNamespace(
        prepare_multiview=lambda lr, cam, shape, size, **kw: calls.append(("prepare", cam)) or features,
        predict=lambda dit, x, t, c, f, cam, shape: calls.append(("predict", cam)) or torch.zeros_like(x),
        geometry=SimpleNamespace(last_diagnostics={}),
    )
    dit = SimpleNamespace(last_injection_stats={})
    runtime = SimpleNamespace(module=module, dit=dit, device=torch.device("cpu"))

    def sampler(noise, prepared, context, *, predict_velocity, config):
        predict_velocity(noise, torch.zeros(1), context, prepared)
        return noise

    result, diagnostics = runner.sample_latents(
        runtime,
        torch.zeros(1, 3, 2, 2, 2),
        original,
        (1, 16, 2, 2, 2),
        5,
        16,
        fusion_camera=fusion_camera,
        geometry_camera=geometry_camera,
        sampler=sampler,
        dtype=torch.float32,
        return_diagnostics=True,
    )
    assert result.shape == (1, 16, 2, 2, 2)
    assert calls == [("prepare", fusion_camera), ("predict", geometry_camera)]
    assert len(diagnostics["velocity_trace"]) == 1
    assert diagnostics["prepared_features"].shape == (1, 8, 4)


def test_resume_requires_matching_log_tail(runner, tmp_path):
    path = tmp_path / "train_steps.jsonl"
    path.write_text('{"step":1}\n{"step":2}\n')
    assert len(runner.resume_rows(path, 2)) == 2
    with pytest.raises(ValueError, match="log"):
        runner.resume_rows(path, 1)


def test_training_state_restores_all_rng_streams(runner, tmp_path):
    class Cycle:
        value = 3

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state):
            self.value = state["value"]

    runner._seed_all(19)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    cycle = Cycle()
    view_generator = torch.Generator().manual_seed(29)
    noise_generator = torch.Generator().manual_seed(39)
    state = runner._training_state(optimizer, cycle, view_generator, noise_generator, 7)
    checkpoint = tmp_path / "state.pt"
    torch.save(state, checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = (
        runner.random.random(),
        float(runner.np.random.rand()),
        torch.rand(2),
        torch.rand(2, generator=view_generator),
        torch.rand(2, generator=noise_generator),
    )
    runner._seed_all(99)
    view_generator.manual_seed(99)
    noise_generator.manual_seed(99)
    cycle.value = 99

    runner._restore_training_state(
        state, optimizer, cycle, view_generator, noise_generator
    )
    actual = (
        runner.random.random(),
        float(runner.np.random.rand()),
        torch.rand(2),
        torch.rand(2, generator=view_generator),
        torch.rand(2, generator=noise_generator),
    )

    assert cycle.value == 3
    assert actual[:2] == expected[:2]
    assert all(torch.equal(left, right) for left, right in zip(actual[2:], expected[2:]))


def test_training_state_restores_target_dropout_rng(runner):
    class Cycle:
        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            assert state == {}

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))])
    cycle = Cycle()
    view_generator = torch.Generator().manual_seed(29)
    noise_generator = torch.Generator().manual_seed(39)
    dropout_generator = torch.Generator().manual_seed(49)
    state = runner._training_state(
        optimizer, cycle, view_generator, noise_generator, 7, dropout_generator
    )
    expected = torch.rand(3, generator=dropout_generator)
    dropout_generator.manual_seed(99)
    runner._restore_training_state(
        state, optimizer, cycle, view_generator, noise_generator, dropout_generator
    )
    assert torch.equal(torch.rand(3, generator=dropout_generator), expected)


def test_current_training_state_requires_dropout_and_cuda_rng_before_mutation(runner, monkeypatch):
    class Cycle:
        def state_dict(self):
            return {}

        def load_state_dict(self, _state):
            pytest.fail("validation must precede mutation")

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.3)
    view = torch.Generator().manual_seed(1)
    noise = torch.Generator().manual_seed(2)
    dropout = torch.Generator().manual_seed(3)
    state = runner._training_state(optimizer, Cycle(), view, noise, 7, dropout)
    assert state["format_version"] == 2
    state.pop("dropout_generator_state")
    with pytest.raises(ValueError, match="dropout_generator_state"):
        runner._restore_training_state(state, optimizer, Cycle(), view, noise, dropout)
    assert optimizer.param_groups[0]["lr"] == 0.3

    state = runner._training_state(optimizer, Cycle(), view, noise, 7)
    state.pop("cuda_rng_state_all", None)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="cuda_rng_state_all"):
        runner._restore_training_state(state, optimizer, Cycle(), view, noise)


def test_legacy_training_state_documents_optional_dropout_rng(runner):
    class Cycle:
        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            assert state == {}

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))])
    view = torch.Generator().manual_seed(1)
    noise = torch.Generator().manual_seed(2)
    dropout = torch.Generator().manual_seed(3)
    state = runner._training_state(optimizer, Cycle(), view, noise, 7, dropout)
    state.pop("format_version")
    state.pop("dropout_generator_state")
    before = dropout.get_state().clone()
    runner._restore_training_state(state, optimizer, Cycle(), view, noise, dropout)
    assert torch.equal(dropout.get_state(), before)


def test_training_state_restores_pairing_rng(runner):
    class Cycle:
        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            assert state == {}

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))])
    cycle = Cycle()
    view_generator = torch.Generator().manual_seed(29)
    noise_generator = torch.Generator().manual_seed(39)
    dropout_generator = torch.Generator().manual_seed(49)
    pairing_generator = torch.Generator().manual_seed(59)
    state = runner._training_state(
        optimizer, cycle, view_generator, noise_generator, 7,
        dropout_generator, pairing_generator,
    )
    expected = torch.rand(3, generator=pairing_generator)
    pairing_generator.manual_seed(99)
    runner._restore_training_state(
        state, optimizer, cycle, view_generator, noise_generator,
        dropout_generator, pairing_generator,
    )
    assert torch.equal(torch.rand(3, generator=pairing_generator), expected)


def test_camera_rank_hinge_and_gradients(runner):
    correct = torch.ones(2, 1, 1, 1, 1, requires_grad=True)
    wrong = torch.full_like(correct, 0.5, requires_grad=True)
    target = torch.zeros_like(correct)
    e_correct, e_wrong, rank = runner.camera_pair_ranking_loss(
        correct, wrong, target, margin_ratio=0.05
    )
    assert torch.allclose(e_correct, torch.ones(2))
    assert torch.allclose(e_wrong, torch.full((2,), 0.25))
    assert torch.allclose(rank, torch.full((2,), 0.8))
    (e_correct.mean() + 0.1 * rank.mean()).backward()
    assert correct.grad is not None and correct.grad.gt(0).all()
    assert wrong.grad is not None and wrong.grad.lt(0).all()


def test_camera_rank_hinge_is_inactive_when_wrong_is_worse(runner):
    correct = torch.ones(1, 1, 1, 1, 1, requires_grad=True)
    wrong = torch.full_like(correct, 2.0, requires_grad=True)
    target = torch.zeros_like(correct)
    _, _, rank = runner.camera_pair_ranking_loss(correct, wrong, target, margin_ratio=0.05)
    assert rank.item() == 0


def test_a6_ranking_uses_only_target_view_but_flow_uses_all_views(runner):
    target = torch.zeros(1, 1, 3, 1, 1)
    correct = torch.tensor([1.0, 2.0, 3.0]).reshape_as(target)
    wrong = torch.tensor([0.5, 20.0, 30.0]).reshape_as(target)
    flow, e_correct, e_wrong, rank = runner._camera_pair_training_losses(
        correct,
        wrong,
        target,
        margin_ratio=0.05,
        target_view_only=True,
    )
    changed = wrong.clone()
    changed[:, :, 1:] = -100
    changed_flow, changed_correct, changed_wrong, changed_rank = runner._camera_pair_training_losses(
        correct,
        changed,
        target,
        margin_ratio=0.05,
        target_view_only=True,
    )
    assert flow == changed_flow
    assert torch.equal(e_correct, changed_correct)
    assert torch.equal(e_wrong, changed_wrong)
    assert torch.equal(rank, changed_rank)
    assert torch.equal(runner.per_view_flow_losses(correct, target), torch.tensor([[1.0, 4.0, 9.0]]))


def test_camera_rank_gradient_groups_split_existing_modules(runner):
    from types import SimpleNamespace

    fusion = SimpleNamespace(
        hidden_dim=2,
        qkv=torch.nn.Linear(2, 6, bias=False),
        output=torch.nn.Linear(2, 2, bias=False),
    )
    bridge = torch.nn.Linear(2, 2)
    module = SimpleNamespace(
        fusion=fusion,
        conditioner=SimpleNamespace(bridge=bridge),
    )
    parameters = [*fusion.qkv.parameters(), *fusion.output.parameters(), *bridge.parameters()]
    gradients = [torch.ones_like(parameter) for parameter in parameters]
    groups = runner._camera_rank_gradient_groups(module, parameters, gradients)
    assert groups["camera_rank_qk_gradient_norm"] > 0
    assert groups["camera_rank_value_gradient_norm"] > 0
    assert groups["camera_rank_output_gradient_norm"] > 0
    assert groups["camera_rank_bridge_gradient_norm"] > 0


def test_pairing_derangement_preserves_target_and_changes_every_auxiliary(runner):
    from rl3dsr.models.wan.geometry_conditioning import CameraBatch
    k = torch.eye(3).repeat(1, 4, 1, 1)
    t = torch.eye(4).repeat(1, 4, 1, 1)
    t[0, :, 0, 3] = torch.arange(4)
    camera = CameraBatch(k, t, (32, 32), "multiview")
    wrong = runner.derange_auxiliary_fusion_camera(
        camera, torch.Generator().manual_seed(11)
    )
    assert torch.equal(wrong.K[:, 0], camera.K[:, 0])
    assert torch.equal(wrong.T_world_from_camera[:, 0], camera.T_world_from_camera[:, 0])
    for view in range(1, 4):
        assert not torch.equal(
            wrong.T_world_from_camera[:, view], camera.T_world_from_camera[:, view]
        )


def test_target_drop_intervention_only_hides_target_lr(runner):
    lr = torch.ones(1, 3, 4, 2, 2)
    camera = object()
    changed, fusion_camera, geometry_camera, mask = runner._intervention(
        lr, camera, "target_drop", torch.Generator().manual_seed(1)
    )
    assert fusion_camera is camera and geometry_camera is camera
    assert changed[:, :, 0].count_nonzero() == 0
    assert torch.equal(changed[:, :, 1:], lr[:, :, 1:])
    assert mask.shape == (1, 4) and mask.all()


def test_camera_intervention_scopes_are_explicit(runner):
    lr = torch.ones(1, 3, 4, 2, 2)
    camera = SimpleNamespace()
    camera.K = torch.eye(3).repeat(1, 4, 1, 1)
    camera.T_world_from_camera = torch.eye(4).repeat(1, 4, 1, 1)
    camera.T_world_from_camera[0, :, 0, 3] = torch.arange(4)
    camera.image_size = (2, 2)
    camera.sequence_kind = "multiview"
    camera.reference_index = 0
    from rl3dsr.models.wan.geometry_conditioning import CameraBatch
    camera = CameraBatch(camera.K, camera.T_world_from_camera, camera.image_size, camera.sequence_kind)
    for mode in ("shuffle_fusion", "shuffle_geometry", "shuffle_all"):
        changed, fusion_camera, geometry_camera, mask = runner._intervention(
            lr, camera, mode, torch.Generator().manual_seed(4)
        )
        assert torch.equal(changed, lr)
        assert mask.all()
        if mode == "shuffle_fusion":
            assert fusion_camera.T_world_from_camera.equal(geometry_camera.T_world_from_camera) is False
            assert geometry_camera.T_world_from_camera.equal(camera.T_world_from_camera)
        elif mode == "shuffle_geometry":
            assert fusion_camera.T_world_from_camera.equal(camera.T_world_from_camera)
            assert geometry_camera.T_world_from_camera.equal(fusion_camera.T_world_from_camera) is False
        else:
            assert fusion_camera.T_world_from_camera.equal(geometry_camera.T_world_from_camera)
            assert fusion_camera.T_world_from_camera.equal(camera.T_world_from_camera) is False


def test_new_mispairing_interventions_have_exact_scopes(runner):
    from rl3dsr.models.wan.geometry_conditioning import CameraBatch

    lr = torch.arange(1 * 1 * 4 * 1 * 1).reshape(1, 1, 4, 1, 1).float()
    transforms = torch.eye(4).repeat(1, 4, 1, 1)
    transforms[0, :, 0, 3] = torch.arange(4)
    camera = CameraBatch(torch.eye(3).repeat(1, 4, 1, 1), transforms, (4, 4), "multiview")

    changed, fusion, geometry, _ = runner._intervention(
        lr, camera, "mispaired_lr", torch.Generator().manual_seed(7)
    )
    assert torch.equal(changed[:, :, :1], lr[:, :, :1])
    assert all(not torch.equal(changed[:, :, i], lr[:, :, i]) for i in range(1, 4))
    assert fusion is camera and geometry is camera

    changed, fusion, geometry, _ = runner._intervention(
        lr, camera, "mispaired_camera", torch.Generator().manual_seed(7)
    )
    assert torch.equal(changed, lr)
    assert geometry is camera
    assert torch.equal(fusion.T_world_from_camera[:, :1], camera.T_world_from_camera[:, :1])
    assert not torch.equal(fusion.T_world_from_camera[:, 1:], camera.T_world_from_camera[:, 1:])

    changed, fusion, geometry, _ = runner._intervention(
        lr, camera, "target_drop_shuffle_fusion", torch.Generator().manual_seed(7)
    )
    assert changed[:, :, 0].count_nonzero() == 0
    assert torch.equal(changed[:, :, 1:], lr[:, :, 1:])
    assert geometry is camera
    assert not torch.equal(fusion.T_world_from_camera[:, 1:], camera.T_world_from_camera[:, 1:])


def test_joint_permutation_covers_all_view_slots_and_roundtrips(runner):
    from rl3dsr.models.wan.geometry_conditioning import CameraBatch

    permutation = torch.tensor([2, 0, 3, 1])
    inverse = runner.inverse_view_permutation(permutation)
    lr = torch.arange(4).reshape(1, 1, 4, 1, 1)
    latent = torch.arange(8).reshape(1, 1, 4, 1, 2)
    mask = torch.tensor([[True, False, True, False]])
    transforms = torch.eye(4).repeat(1, 4, 1, 1)
    transforms[0, :, 0, 3] = torch.arange(4)
    camera = CameraBatch(torch.eye(3).repeat(1, 4, 1, 1), transforms, (4, 4), "multiview")
    result = runner.apply_joint_view_permutation(
        permutation,
        lr=lr,
        fusion_camera=camera,
        geometry_camera=camera,
        target=latent,
        latent=latent,
        source_mask=mask,
        noise=latent,
    )
    assert result["target_index"] == 1
    assert torch.equal(result["lr"], lr.index_select(2, permutation))
    assert torch.equal(result["source_mask"], mask.index_select(1, permutation))
    assert torch.equal(result["fusion_camera"].T_world_from_camera, transforms.index_select(1, permutation))
    assert torch.equal(runner.permute_view_tensor(result["latent"], inverse), latent)
    assert result["permutation_sha256"] == runner._permutation_digest(permutation)
    with pytest.raises(ValueError, match="permutation"):
        runner.inverse_view_permutation(torch.tensor([0, 0, 2, 3]))


def test_joint_permutation_sample_uses_permuted_paired_noise_and_inverse_output(runner):
    permutation = torch.tensor([2, 0, 1])
    captured = {}
    module = SimpleNamespace(
        fusion=None,
        geometry=SimpleNamespace(last_diagnostics={}),
        prepare_multiview=lambda *args, **kwargs: torch.zeros(1, 3, 1),
        predict=lambda _dit, sample, *_args, **_kwargs: torch.zeros_like(sample),
    )
    runtime = SimpleNamespace(module=module, dit=SimpleNamespace(last_injection_stats={}), device=torch.device("cpu"))

    def sampler(noise, prepared, context, *, predict_velocity, config):
        captured["noise"] = noise.clone()
        return noise

    shape = (1, 1, 3, 1, 2)
    base = torch.randn(shape, generator=torch.Generator().manual_seed(19))
    sampled = runner.sample_latents(
        runtime, torch.zeros(1, 1, 3, 1, 1), object(), shape, 1, 4,
        sampler=sampler, seed=19, dtype=torch.float32, view_permutation=permutation,
    )
    assert torch.equal(captured["noise"], base.index_select(2, permutation))
    assert torch.equal(sampled, base.index_select(2, permutation))
    assert torch.equal(
        runner.inverse_permute_joint_output(sampled, {"permutation": permutation.tolist()}),
        base,
    )


def test_joint_permutation_feature_diagnostics_compare_canonical_view_order(runner):
    from rl3dsr.models.wan.geometry_conditioning import CameraBatch
    from rl3dsr.validation.stage3_protocol import Stage3Config

    permutation = torch.tensor([2, 0, 1])
    lr = torch.arange(3).reshape(1, 1, 3, 1, 1).float()
    transforms = torch.eye(4).repeat(1, 3, 1, 1)
    transforms[0, :, 0, 3] = torch.arange(3)
    camera = CameraBatch(torch.eye(3).repeat(1, 3, 1, 1), transforms, (16, 16), "multiview")
    permuted = runner.apply_joint_view_permutation(
        permutation, lr=lr, fusion_camera=camera, geometry_camera=camera
    )

    class Conditioner:
        @staticmethod
        def bridge_residuals(features, _timestep):
            return {0: features}

    module = SimpleNamespace(
        fusion=None,
        conditioner=Conditioner(),
        prepare_multiview=lambda value, *_args, **_kwargs: value.flatten(2).transpose(1, 2),
    )
    result = runner._feature_diagnostics(
        SimpleNamespace(module=module),
        lr,
        camera,
        (1, 1, 3, 1, 1),
        Stage3Config(image_size=16, views=3),
        permuted["lr"],
        permuted["fusion_camera"],
        permuted["geometry_camera"],
        torch.ones(1, 3, dtype=torch.bool),
        {"id": "chair:000", "scene": "chair", "anchor": 0},
        "joint_permute",
        view_permutation=permutation,
    )
    assert result["prepared_feature_delta"] == 0
    assert result["bridge"]["block_0"]["delta_norm"] == 0


def test_fusion_camera_dose_uses_valid_se3_and_exact_endpoints(runner):
    from rl3dsr.models.wan.geometry_conditioning import CameraBatch

    k = torch.eye(3).repeat(1, 3, 1, 1)
    k[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    correct_t = torch.eye(4).repeat(1, 3, 1, 1)
    angle = torch.tensor(0.37)
    correct_t[0, 0, :3, :3] = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0.0],
        [torch.sin(angle), torch.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    correct_t[0, 0, :3, 3] = torch.tensor([0.25, -0.5, 1.5])
    wrong_t = correct_t.clone()
    wrong_t[0, 1, :3, :3] = torch.tensor([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
    wrong_t[0, 2, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    wrong_t[0, 1:, :3, 3] = torch.tensor([[2., 4., 6.], [4., 8., 12.]])
    xi = torch.tensor([[0.2, 0.3, 0.4]])
    correct = CameraBatch(k, correct_t, (8, 8), "multiview", camera_model="ucm", xi=xi)
    wrong_k = k.clone()
    wrong_k[:, 1:] = k[:, [2, 1]]
    wrong_xi = xi.clone()
    wrong_xi[:, 1:] = xi[:, [2, 1]]
    wrong = CameraBatch(
        wrong_k, wrong_t, (8, 8), "multiview", camera_model="ucm", xi=wrong_xi
    )
    dose0 = runner.interpolate_fusion_camera_dose(correct, wrong, 0.0)
    dose_half = runner.interpolate_fusion_camera_dose(correct, wrong, 0.5)
    dose1 = runner.interpolate_fusion_camera_dose(correct, wrong, 1.0)
    assert torch.equal(dose0.K, correct.K) and torch.equal(dose0.T_world_from_camera, correct.T_world_from_camera)
    assert torch.equal(dose1.K, wrong.K) and torch.equal(dose1.T_world_from_camera, wrong.T_world_from_camera)
    assert torch.equal(dose_half.T_world_from_camera[:, :1], correct.T_world_from_camera[:, :1])
    assert torch.equal(dose_half.K[:, :1], correct.K[:, :1])
    assert torch.equal(dose_half.xi[:, :1], correct.xi[:, :1])
    rotation = dose_half.T_world_from_camera[..., :3, :3]
    identity = torch.eye(3).expand_as(rotation)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones_like(torch.linalg.det(rotation)), atol=1e-5)
    axis_scale = 2 ** -0.5
    expected_mixed_axis_half = torch.tensor([
        [0.5, -0.5, -axis_scale],
        [-0.5, 0.5, -axis_scale],
        [axis_scale, axis_scale, 0.0],
    ])
    assert torch.allclose(rotation[0, 1], expected_mixed_axis_half, atol=1e-5)
    scores = [
        (dose.T_world_from_camera - correct.T_world_from_camera).square().sum().item()
        + (dose.K - correct.K).square().sum().item()
        for dose in (dose0, dose_half, dose1)
    ]
    assert scores[0] < scores[1] < scores[2]


def test_auxiliary_permutation_keeps_target_and_relabels_full_model_inputs(runner):
    from rl3dsr.models.wan.geometry_conditioning import CameraBatch

    lr = torch.arange(4).reshape(1, 1, 4, 1, 1).float()
    pose = torch.eye(4).repeat(1, 4, 1, 1)
    pose[0, :, 0, 3] = torch.arange(4)
    camera = CameraBatch(torch.eye(3).repeat(1, 4, 1, 1), pose, (16, 16), "multiview")
    changed, fusion, geometry, mask, metadata = runner._intervention(
        lr, camera, "aux_permute", torch.Generator().manual_seed(7), return_metadata=True
    )
    permutation = torch.tensor(metadata["permutation"])
    assert metadata["target_index"] == 0
    assert permutation[0] == 0 and not torch.equal(permutation, torch.arange(4))
    assert torch.equal(changed, lr[:, :, permutation])
    assert torch.equal(fusion.T_world_from_camera, camera.T_world_from_camera[:, permutation])
    assert torch.equal(geometry.T_world_from_camera, camera.T_world_from_camera[:, permutation])
    assert mask.all()
    assert torch.equal(runner.inverse_permute_joint_output(changed, metadata), lr)


def test_far_fusion_camera_intervention_keeps_lr_target_and_geometry(runner):
    from rl3dsr.models.wan.geometry_conditioning import CameraBatch

    lr = torch.ones(1, 3, 4, 2, 2)
    camera = CameraBatch(
        torch.eye(3).repeat(1, 4, 1, 1),
        torch.eye(4).repeat(1, 4, 1, 1),
        (2, 2),
        "multiview",
    )
    far = CameraBatch(
        camera.K.clone(), camera.T_world_from_camera.clone(), (2, 2), "multiview"
    )
    far.T_world_from_camera[0, 1:, 0, 3] = torch.tensor([10.0, 20.0, 30.0])
    changed, fusion_camera, geometry_camera, mask = runner._intervention(
        lr, camera, "far_shuffle_fusion", torch.Generator().manual_seed(4),
        far_camera=far,
    )
    assert torch.equal(changed, lr)
    assert torch.equal(fusion_camera.T_world_from_camera, far.T_world_from_camera)
    assert torch.equal(geometry_camera.T_world_from_camera, camera.T_world_from_camera)
    assert mask.all()


def test_seen_eval_parser_accepts_frozen_camera_donors_and_no_self_modes(runner):
    parser = runner._parser()
    args = parser.parse_args([
        "seen-eval", "--config", "a.json", "--dataset-root", "data",
        "--model-dir", "model", "--lq-source", "lq.py", "--lq-checkpoint", "lq.pt",
        "--bridge-checkpoint", "bridge.pt", "--checkpoint", "stage3.pt",
        "--output-dir", "out", "--seen-manifest", "seen.json", "--subset", "probe",
        "--inference-seeds", "3302", "--camera-donor-manifest", "donors.json",
        "--modes", "correct", "far_shuffle_fusion", "no_self_correct",
        "no_self_far_shuffle_fusion",
    ])
    assert args.camera_donor_manifest == Path("donors.json")
    assert args.modes[-1] == "no_self_far_shuffle_fusion"


def test_camera_donor_manifest_is_bound_to_seen_groups(runner, tmp_path):
    from rl3dsr.validation.stage3_protocol import Stage3Config

    config = Stage3Config(train_scenes=("chair",), views=4)
    seen = tmp_path / "seen.json"
    seen.write_text("{}")
    group = {"id": "chair:000", "scene": "chair", "anchor": 0,
             "indices": [0, 1, 2, 3]}
    payload = {
        "version": 1,
        "scope": "stage3_2_far_camera_donors",
        "seen_manifest_sha256": runner._sha256(seen),
        "sampling_signature": runner._sampling_signature(config),
        "groups": [{
            "id": "chair:000",
            "scene": "chair",
            "anchor": 0,
            "source_indices": [0, 1, 2, 3],
            "fusion_camera_indices": [0, 9, 8, 7],
            "donor_angles_deg": [120.0, 110.0, 100.0],
        }],
    }
    donors = tmp_path / "donors.json"
    donors.write_text(json.dumps(payload))
    loaded = runner._load_camera_donors(donors, seen, config, [group])
    assert loaded["chair:000"]["fusion_camera_indices"] == [0, 9, 8, 7]
    payload["groups"][0]["source_indices"] = [0, 4, 5, 6]
    donors.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source indices"):
        runner._load_camera_donors(donors, seen, config, [group])


def test_output_directory_never_overwrites(runner, tmp_path):
    (tmp_path / "existing").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        runner.prepare_output(tmp_path)
    assert (tmp_path / "existing").read_text() == "keep"


def test_seed_all_reproduces_torch_and_numpy(runner):
    runner._seed_all(17)
    first = (torch.rand(3), runner.np.random.rand(3))
    runner._seed_all(17)
    second = (torch.rand(3), runner.np.random.rand(3))
    assert torch.equal(first[0], second[0])
    assert (first[1] == second[1]).all()


def test_seen_manifest_selects_exact_groups(runner, tmp_path):
    from rl3dsr.validation.stage3_protocol import Stage3Config

    config = Stage3Config()
    payload = {
        "version": 1,
        "scope": "seen_train_sr",
        "sampling_signature": runner._sampling_signature(config),
        "groups": [
            {"id": "chair:000", "scene": "chair", "anchor": 0, "indices": [0, 1, 2, 3]},
            {"id": "chair:001", "scene": "chair", "anchor": 1, "indices": [1, 0, 2, 3]},
        ],
        "subsets": {"probe": ["chair:000"], "full": ["chair:000", "chair:001"]},
    }
    path = tmp_path / "seen.json"
    path.write_text(json.dumps(payload))
    _, groups = runner._load_seen_groups(path, config, "full", ["chair:001"])
    assert [group["id"] for group in groups] == ["chair:001"]
    with pytest.raises(ValueError, match="requested"):
        runner._load_seen_groups(path, config, "probe", ["chair:001"])


def test_seen_eval_parser_requires_explicit_scope_inputs(runner):
    parser = runner._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["seen-eval", "--config", "a.json"])


def test_trace_delta_rejects_different_initial_noise(runner):
    reference = {
        "initial_noise": torch.zeros(1),
        "velocity_trace": [],
        "geometry_trace": [],
        "injection_trace": [],
    }
    changed = {**reference, "initial_noise": torch.ones(1)}
    with pytest.raises(RuntimeError, match="different initial noise"):
        runner._trace_delta(reference, changed)


def test_seen_eval_writes_target_only_rows_and_images(runner, tmp_path, monkeypatch):
    from rl3dsr.validation.stage3_protocol import Stage3Config

    class Frozen:
        def eval(self):
            return self

        def requires_grad_(self, _enabled):
            return self

    class VAE:
        model = SimpleNamespace(model=Frozen())

        @staticmethod
        def encode_multiview(value):
            return value

        @staticmethod
        def decode_multiview(value):
            return value

    config = Stage3Config(arm="A3", image_size=16)
    checkpoint = tmp_path / "stage3_step_4000.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "seen_groups.json"
    manifest.write_text("{}")
    output = tmp_path / "output"
    group = {"id": "chair:007", "scene": "chair", "anchor": 7, "indices": [7, 2, 3, 4]}
    hr = torch.zeros(1, 3, 4, 16, 16)
    lr = torch.zeros(1, 3, 4, 4, 4)
    runtime = SimpleNamespace(
        module=Frozen(),
        dit=SimpleNamespace(model=Frozen()),
        vae=VAE(),
        device=torch.device("cpu"),
        checkpoint={"step": 4000, "provenance": {"training_seed": 42}},
    )
    sample_seeds = []

    monkeypatch.setattr(runner, "_validate_runtime_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_load_seen_groups", lambda *args, **kwargs: ({"datasets": {}}, [group]))
    monkeypatch.setattr(runner, "load_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(runner, "_load_indices", lambda *args, **kwargs: (group["indices"], hr, lr, object()))
    monkeypatch.setattr(runner, "_PerFrameLPIPS", lambda device: object())
    monkeypatch.setattr(
        runner,
        "_intervention",
        lambda value, camera, mode, generator, **kwargs: (value, camera, camera, None),
    )

    def sample_latents(*args, seed, **kwargs):
        sample_seeds.append(seed)
        return hr

    def frame_metrics(prediction, target, *, perceptual_metric):
        assert prediction.shape[2] == target.shape[2] == 1
        return [{"batch_index": 0, "frame_index": 0, "psnr": 30.0,
                 "ssim": 0.9, "lpips": 0.1, "mae": 0.05}]

    monkeypatch.setattr(runner, "sample_latents", sample_latents)
    monkeypatch.setattr(runner, "frame_metrics", frame_metrics)
    args = SimpleNamespace(
        checkpoint=checkpoint,
        inference_seeds=[3302, 3303],
        seen_manifest=manifest,
        subset="probe",
        group_ids=None,
        output_dir=output,
        model_dir=tmp_path,
        lq_source=tmp_path,
        lq_checkpoint=tmp_path,
        bridge_checkpoint=tmp_path,
        rre_checkpoint=None,
        device="cpu",
        dataset_root=tmp_path,
        modes=("correct", "correct_repeat"),
        save_images=True,
    )
    runner._evaluate_seen(args, config)

    rows = [json.loads(line) for line in (output / "evaluation_rows.jsonl").read_text().splitlines()]
    baselines = [json.loads(line) for line in (output / "baseline_rows.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and len(baselines) == 2
    assert {(row["view_index"], tuple(row["view_indices"])) for row in rows} == {(7, (7, 2, 3, 4))}
    assert all("frame_index" not in row and row["split"] == "train" for row in rows)
    assert all(row["inference_seed"] is None for row in baselines)
    assert sample_seeds == [3302000007, 3302000007, 3303000007, 3303000007]
    assert len(list((output / "images").rglob("*.png"))) == 8
