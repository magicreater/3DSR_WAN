"""Small structural checks for the isolated Stage 3 loss experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage3_loss_experiment as experiment


def test_recovery_target_is_exact_for_perfect_velocity():
    clean = torch.tensor([[[[[2.0]]]]])
    noise = torch.tensor([[[[[-1.0]]]]])
    sigma, lower = 0.8, 0.5
    noisy = (1 - sigma) * clean + sigma * noise
    velocity = noise - clean
    next_state = noisy + (lower - sigma) * velocity
    assert torch.allclose((next_state - clean) / lower, velocity)
    assert torch.allclose(next_state - lower * velocity, clean)


def test_next_sigma_skips_near_zero():
    schedule = torch.tensor([1.0, 0.8, 0.5, 0.09, 0.0])
    assert experiment.next_sigma(torch.tensor(0.8), schedule) == pytest.approx(0.5)
    assert experiment.next_sigma(torch.tensor(0.5), schedule) is None


def test_second_target_keeps_auxiliaries_fixed():
    angles = torch.deg2rad(torch.tensor([0., 5., 10., 15., 30., 60.]))
    centers = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), -1)
    assert experiment.second_target(0, [1, 2, 3], centers) == 4
    with pytest.raises(RuntimeError, match="second target"):
        experiment.second_target(0, [1, 2, 3, 4], centers)
