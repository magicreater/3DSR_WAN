#!/usr/bin/env python3
"""Compute-matched zero-pair control for the selected Stage 3 arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import stage3_loss_campaign as campaign
import stage3_loss_experiment as experiment


NAME = "full_compute_match"


def zero_pair(original, *args, **kwargs):
    loss, info = original(*args, **kwargs)
    return loss * 0, info


def self_check():
    value = torch.tensor(2.0, requires_grad=True)
    loss, _ = zero_pair(lambda: (value.square(), {}))
    loss.backward()
    assert loss.requires_grad and value.grad is not None and value.grad.item() == 0


def train():
    campaign.assert_protocol()
    screen = json.loads((campaign.OUT / "screen_review.json").read_text())
    full = json.loads((campaign.OUT / "full_review.json").read_text())
    if screen["selected_loss"] != "both" or full["best_candidate"] != "full_ordinary":
        raise RuntimeError("selected experiment changed")
    output = campaign.OUT / "train" / NAME
    wrapper_sha = experiment.sha256(Path(__file__))
    experiment.frozen_json(campaign.OUT / "compute_match_protocol.json", {
        "purpose": "same two paired forwards and backward graph every fourth step; paired gradient zero",
        "parent_sha256": experiment.sha256(experiment.PARENT),
        "trainer_sha256": experiment.sha256(Path(experiment.__file__)),
        "wrapper_sha256": wrapper_sha,
        "config_sha256": experiment.sha256(campaign.config_path("full")),
        "calibration_sha256": experiment.sha256(campaign.OUT / "calibration.json"),
        "seed": 42, "loss_mode": "both", "pair_mode": "ordinary",
        "paired_gradient_scale": 0.0,
    })
    prior = sorted(output.glob("stage3_step_*.pt"))
    if prior:
        payload = torch.load(prior[-1], map_location="cpu", weights_only=True)
        if payload["provenance"].get("compute_match_wrapper_sha256") != wrapper_sha:
            raise RuntimeError("compute-matched checkpoint provenance drift")
    original_pair = experiment.pair_loss
    original_save = experiment.save_stage3_checkpoint

    def save(path, module, **kwargs):
        provenance = dict(kwargs["provenance"])
        provenance["compute_match_wrapper_sha256"] = wrapper_sha
        provenance["paired_gradient_scale"] = 0.0
        kwargs["provenance"] = provenance
        return original_save(path, module, **kwargs)

    experiment.pair_loss = lambda *args, **kwargs: zero_pair(original_pair, *args, **kwargs)
    experiment.save_stage3_checkpoint = save
    try:
        experiment.train(SimpleNamespace(
            config=campaign.config_path("full"),
            calibration=campaign.OUT / "calibration.json",
            output=output, loss_mode="both", pair_mode="ordinary",
            experiment="stage3_loss_compute_match_20260928", seed=42,
        ))
    finally:
        experiment.pair_loss = original_pair
        experiment.save_stage3_checkpoint = original_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-check", "train"))
    args = parser.parse_args()
    {"self-check": self_check, "train": train}[args.command]()
