"""Runner contracts using synthetic objects only; never load model assets."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


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


def test_missing_inputs_fail_before_runtime(runner, tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{}')
    monkeypatch.setattr(runner, "load_runtime", lambda *a: pytest.fail("loaded model"))
    with pytest.raises((ValueError, SystemExit)):
        runner.main(["train", "--config", str(config)])


def test_split_routes_do_not_touch_test_during_development(runner):
    from rl3dsr.validation.stage3_protocol import Stage3Config
    config = Stage3Config()
    assert runner.scene_routes(config, "train") == [(s, "train") for s in config.train_scenes]
    assert runner.scene_routes(config, "validate") == [(s, "val") for s in config.validation_scene_names]
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
        lambda value, camera, mode, generator: (value, camera, camera, None),
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
