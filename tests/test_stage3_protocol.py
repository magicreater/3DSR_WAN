from dataclasses import replace
import json

import pytest
import torch

from rl3dsr.models.wan.geometry_conditioning import CameraBatch
from rl3dsr.validation.stage3_protocol import (
    Stage3Config, claim_final_evaluation, evenly_spaced_indices, freeze_candidate,
    intervene_lr, load_frozen_candidate, load_stage3_config, nearest_view_indices,
    sample_view_indices, select_candidate, shuffle_auxiliary_pairs,
)


def camera(views=4):
    k = torch.eye(3).repeat(1, views, 1, 1)
    t = torch.eye(4).repeat(1, views, 1, 1)
    t[0, :, 0, 3] = torch.arange(views)
    return CameraBatch(k, t, (8, 8), "multiview")


def test_strict_config_and_disjoint_scenes(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(Stage3Config(arm="A3").to_dict()))
    assert load_stage3_config(path).fusion_mode == "epipolar"
    with pytest.raises(ValueError, match="disjoint"):
        replace(Stage3Config(), test_scenes=("chair",))
    with pytest.raises(ValueError):
        replace(Stage3Config(), steps=True)
    path.write_text('{"unknown": 1}')
    with pytest.raises(ValueError, match="Unknown"):
        load_stage3_config(path)
    assert Stage3Config().final_inference_seeds == (3302, 3303, 3304)
    assert Stage3Config(target_lr_dropout=0.5).target_lr_dropout == 0.5
    assert Stage3Config(epipolar_attention="local_band").epipolar_band == 1.5
    assert Stage3Config().allow_self_view_source is True
    assert Stage3Config(allow_self_view_source=False).allow_self_view_source is False
    with pytest.raises(ValueError, match="epipolar_attention"):
        replace(Stage3Config(), epipolar_attention="invalid")
    with pytest.raises(ValueError, match="target_lr_dropout"):
        replace(Stage3Config(), target_lr_dropout=1.0)
    with pytest.raises(ValueError, match="target_lr_dropout"):
        replace(Stage3Config(), target_lr_dropout=-0.1)
    with pytest.raises(ValueError, match="allow_self_view_source"):
        replace(Stage3Config(), allow_self_view_source=1)


def test_interventions_preserve_target_and_inputs():
    lr = torch.arange(1 * 3 * 4 * 8 * 8).reshape(1, 3, 4, 8, 8).float()
    cam = camera()
    original = lr.clone()
    for mode in ("correct", "remove", "duplicate", "shuffle_camera", "local_patch"):
        changed, changed_cam, mask = intervene_lr(
            lr, cam, mode, target=1, source=2, box=(1, 2, 3, 4),
            generator=torch.Generator().manual_seed(1),
        )
        assert torch.equal(changed[:, :, 1], original[:, :, 1])
        assert torch.equal(changed_cam.T_world_from_camera[:, 1], cam.T_world_from_camera[:, 1])
        assert mask.shape == (1, 4) and mask[:, 1].all()
        if mode == "remove":
            assert mask.sum() == 1 and changed[:, :, 0].count_nonzero() == 0
        if mode == "duplicate":
            assert torch.equal(changed[:, :, 0], lr[:, :, 1])
            assert torch.equal(changed_cam.T_world_from_camera[:, 0], cam.T_world_from_camera[:, 1])
        if mode == "shuffle_camera":
            assert not torch.equal(changed_cam.T_world_from_camera[:, 0], cam.T_world_from_camera[:, 0])
        if mode == "local_patch":
            assert changed[:, :, 2, 1:3, 2:4].count_nonzero() == 0
    assert torch.equal(lr, original)
    with pytest.raises(ValueError):
        intervene_lr(lr, cam, "local_patch", target=1, source=1, box=(0, 0, 2, 2))
    with pytest.raises(ValueError):
        intervene_lr(lr[:, :, :2], camera(2), "shuffle_camera")


def test_shuffle_auxiliary_pairs_keeps_target_and_reorders_matching_camera():
    lr = torch.arange(1 * 3 * 4 * 2 * 2).reshape(1, 3, 4, 2, 2).float()
    cam = camera()
    original = lr.clone()
    changed, changed_cam, mask = shuffle_auxiliary_pairs(
        lr, cam, target=0, generator=torch.Generator().manual_seed(3)
    )
    assert torch.equal(changed[:, :, 0], original[:, :, 0])
    assert torch.equal(changed_cam.T_world_from_camera[:, 0], cam.T_world_from_camera[:, 0])
    assert mask.shape == (1, 4) and mask.all()
    for view in range(1, 4):
        source = int(changed_cam.T_world_from_camera[0, view, 0, 3].item())
        assert torch.equal(changed[:, :, view], original[:, :, source])


def test_view_sampling_is_reproducible():
    cam = camera(20)
    first = sample_view_indices(cam, 3, torch.Generator().manual_seed(5))
    assert first == sample_view_indices(cam, 3, torch.Generator().manual_seed(5))
    assert first[0] == 3 and len(set(first)) == 4
    with pytest.raises(ValueError):
        sample_view_indices(camera(2), 0, torch.Generator())


def test_seen_view_sampling_is_deterministic_and_inclusive():
    cam = camera(20)
    assert nearest_view_indices(cam, 3) == nearest_view_indices(cam, 3)
    assert nearest_view_indices(cam, 3)[0] == 3
    assert evenly_spaced_indices(100, 4) == [0, 33, 66, 99]
    selected = evenly_spaced_indices(100, 16)
    assert selected[0] == 0 and selected[-1] == 99 and len(set(selected)) == 16
    with pytest.raises(ValueError):
        nearest_view_indices(camera(2), 0, 4)
    with pytest.raises(ValueError):
        evenly_spaced_indices(3, 4)


def row(checkpoint, step, scene, psnr, ssim=.8, lpips=.1):
    return dict(checkpoint=str(checkpoint), step=step, scene=scene, split="validation",
                psnr=psnr, ssim=ssim, lpips=lpips)


def test_selection_scene_weight_and_validation_only():
    rows = [row("a", 1, "a", 40)] * 10 + [row("a", 1, "b", 10)]
    rows += [row("b", 2, "a", 26), row("b", 2, "b", 26)]
    assert select_candidate(rows, ("a", "b"))["checkpoint"] == "b"
    rows = [row("a", 1, "a", 26), row("b", 2, "a", 26.005, ssim=.81)]
    assert select_candidate(rows, ("a",))["checkpoint"] == "b"
    with pytest.raises(ValueError, match="validation"):
        select_candidate([dict(rows[0], split="test")], ("a",))
    with pytest.raises(ValueError, match="coverage"):
        select_candidate(rows, ("a", "missing"))
    with pytest.raises(ValueError, match="mix"):
        select_candidate([dict(rows[0], train_seed=42), dict(rows[1], train_seed=43)], ("a",))


def test_freeze_and_final_guard_bind_checkpoint_and_protocol(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    config = Stage3Config()
    candidate = select_candidate([row(checkpoint, 500, scene, 26)
                                  for scene in config.validation_scene_names], config.validation_scene_names)
    manifest = tmp_path / "frozen.json"
    with pytest.raises(ValueError, match="validation scene"):
        freeze_candidate(manifest, dict(candidate, scenes=["ficus"]), config)
    freeze_candidate(manifest, candidate, config)
    with pytest.raises(FileExistsError):
        freeze_candidate(manifest, candidate, config)
    assert load_frozen_candidate(manifest, config)["checkpoint_sha256"]
    with pytest.raises(ValueError, match="protocol"):
        load_frozen_candidate(manifest, replace(config, arm="A3"))
    claim_final_evaluation(manifest, config, tmp_path / "final")
    with pytest.raises(FileExistsError):
        claim_final_evaluation(manifest, config, tmp_path / "another")
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checkpoint"):
        load_frozen_candidate(manifest, config)
