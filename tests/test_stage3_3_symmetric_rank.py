import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def stage3():
    path = Path(__file__).resolve().parents[1] / "scripts/stage3_experiment.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("stage3_experiment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def driver(stage3):
    path = Path(__file__).resolve().parents[1] / "scripts/stage3_3_symmetric_rank.py"
    spec = importlib.util.spec_from_file_location("stage3_3_symmetric_rank", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_symmetric_rank_uses_one_target_preserving_permutation_and_exact_mean(stage3):
    generator = torch.Generator().manual_seed(7)
    permutation = stage3._deranged_auxiliary_indices(4, generator)
    lr = torch.arange(1 * 1 * 4 * 2 * 2).reshape(1, 1, 4, 2, 2)
    wrong_lr = stage3.permute_view_tensor(lr, permutation)
    assert permutation[0].item() == 0
    assert torch.equal(wrong_lr[:, :, 0], lr[:, :, 0])
    assert not torch.equal(wrong_lr[:, :, 1:], lr[:, :, 1:])

    camera_rank = torch.tensor([1.0, 3.0])
    lr_rank = torch.tensor([5.0, 7.0])
    combined = stage3._symmetric_correspondence_rank(camera_rank, lr_rank)
    assert torch.equal(combined, torch.tensor([3.0, 5.0]))
    assert 0.5 * combined.mean() == 0.25 * camera_rank.mean() + 0.25 * lr_rank.mean()
    weighted = stage3._symmetric_correspondence_rank(
        camera_rank, lr_rank, camera_fraction=0.8
    )
    assert torch.allclose(weighted, 0.8 * camera_rank + 0.2 * lr_rank)
    with pytest.raises(ValueError, match="matching shapes"):
        stage3._symmetric_correspondence_rank(camera_rank, lr_rank[:1])
    with pytest.raises(ValueError, match="camera_fraction"):
        stage3._symmetric_correspondence_rank(camera_rank, lr_rank, 1.0)


def test_target_only_ranking_keeps_global_flow(stage3):
    target = torch.zeros(1, 1, 4, 1, 1)
    correct = torch.tensor([[[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]]])
    wrong = correct.clone()
    wrong[:, :, 0] = 2.0
    flow, e_correct, e_wrong, rank = stage3._camera_pair_training_losses(
        correct, wrong, target, margin_ratio=0.05, target_view_only=True
    )
    assert flow == correct.square().mean()
    assert e_correct.shape == e_wrong.shape == rank.shape == (1,)
    assert e_correct.item() == 1.0
    assert e_wrong.item() == 4.0


def test_target_flow_fraction_changes_only_correct_flow(stage3):
    correct = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4, 1, 1).requires_grad_()
    target = torch.zeros_like(correct)
    wrong = correct.detach().clone()
    wrong[:, :, 0] = 2.0
    original = stage3._camera_pair_training_losses(
        correct, wrong, target, margin_ratio=0.05, target_view_only=True
    )
    weighted = stage3._camera_pair_training_losses(
        correct, wrong, target, margin_ratio=0.05, target_view_only=True,
        target_view_flow_fraction=0.5,
    )
    assert original[0] == correct.square().mean()
    assert weighted[0] == torch.tensor(0.5 + (4 + 9 + 16) / 6)
    assert all(torch.equal(a, b) for a, b in zip(original[1:], weighted[1:]))
    gradient = torch.autograd.grad(weighted[0], correct)[0].flatten()
    assert torch.allclose(gradient, torch.tensor([1.0, 2 / 3, 1.0, 4 / 3]))


def test_campaign_preserves_original_order_and_stops_on_hold(driver, tmp_path, monkeypatch):
    calls = []

    def run_cell(_args, name):
        calls.append(name)
        return {
            "status": "HOLD",
            "pass": False,
            "next": "REVIEW_ATTENTION_CORRESPONDENCE",
        }

    monkeypatch.setattr(driver, "run_cell", run_cell)
    monkeypatch.setattr(
        driver,
        "_write_final_review",
        lambda _args, _results: {
            "next": "REVIEW_ATTENTION_CORRESPONDENCE",
            "STAGE4_READY": False,
        },
    )
    args = SimpleNamespace(campaign_root=tmp_path)
    result = driver.run(args)
    assert calls == ["a6_lego_seed42"]
    assert result["a6_lego_seed43"] is None
    assert result["a6_chair_seed42"] is None
    assert result["STAGE4_READY"] is False


def test_campaign_runs_all_cells_only_after_passes(driver, tmp_path, monkeypatch):
    calls = []

    def run_cell(_args, name):
        calls.append(name)
        return {"status": "PILOT_PASS", "pass": True, "next": "NEXT"}

    monkeypatch.setattr(driver, "run_cell", run_cell)
    monkeypatch.setattr(
        driver,
        "_write_final_review",
        lambda _args, _results: {"next": "RUN_STAGE4", "STAGE4_READY": True},
    )
    result = driver.run(SimpleNamespace(campaign_root=tmp_path))
    assert calls == list(driver.CELL_ORDER)
    assert result["STAGE4_READY"] is True
