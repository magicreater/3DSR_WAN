"""Training-aligned pure-noise sampling for the frozen Wan Stage 1 adapter."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class FlowSamplingConfig:
    steps: int = 50
    shift: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.shift <= 0:
            raise ValueError("shift must be positive")


class SigmaCycle:
    """Deterministically shuffle complete training-aligned sigma cycles."""

    def __init__(self, config: FlowSamplingConfig, *, seed: int, strategy: str = "permutation"):
        if strategy not in {"permutation", "balanced"}:
            raise ValueError("strategy must be permutation or balanced")
        self.config = config
        self.strategy = strategy
        self.values = training_sigmas(config, device="cpu", strategy=strategy)
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(self.values), generator=self.generator)
        self.position = 0
        self.cycle = 0

    def next(self) -> Tensor:
        if self.position == len(self.order):
            self.order = torch.randperm(len(self.values), generator=self.generator)
            self.position = 0
            self.cycle += 1
        value = self.values[self.order[self.position]].clone()
        self.position += 1
        return value

    def state_dict(self) -> dict[str, object]:
        return {
            "steps": self.config.steps,
            "shift": self.config.shift,
            "strategy": self.strategy,
            "generator_state": self.generator.get_state(),
            "order": self.order.clone(),
            "position": self.position,
            "cycle": self.cycle,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("steps") != self.config.steps or state.get("shift") != self.config.shift or state.get("strategy", "permutation") != self.strategy:
            raise ValueError("sigma cycle sampling config mismatch")
        order = state.get("order")
        position = state.get("position")
        if not isinstance(order, Tensor) or order.shape != (len(self.values),):
            raise ValueError("invalid sigma cycle order")
        if not isinstance(position, int) or not 0 <= position <= len(order):
            raise ValueError("invalid sigma cycle position")
        self.generator.set_state(state["generator_state"])
        self.order = order.clone()
        self.position = position
        self.cycle = int(state["cycle"])


def _scheduler(config: FlowSamplingConfig, device: str | torch.device):
    root = Path(__file__).resolve().parents[4] / "third_party" / "wan2_1"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(config.steps, device=device, shift=config.shift)
    return scheduler


def unipc_schedule(
    config: FlowSamplingConfig,
    *,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Return the exact pinned-Wan UniPC timesteps and sigma endpoints."""

    scheduler = _scheduler(config, device)
    return scheduler.timesteps.detach().clone(), scheduler.sigmas.detach().clone()


def training_sigmas(
    config: FlowSamplingConfig,
    *,
    device: str | torch.device,
    strategy: str = "permutation",
) -> Tensor:
    """Return a deterministic training sigma cycle with optional low-sigma balancing."""

    _, inference_sigmas = unipc_schedule(config, device=device)
    values = inference_sigmas[:-1]
    if strategy == "permutation":
        return torch.cat((torch.ones(1, device=device), values))
    if strategy != "balanced":
        raise ValueError("strategy must be permutation or balanced")
    bins = (
        values[(values < 0.25)],
        values[(values >= 0.25) & (values < 0.5)],
        values[(values >= 0.5) & (values < 0.8)],
        values[(values >= 0.8)],
    )
    if any(not len(bucket) for bucket in bins):
        raise ValueError("inference schedule must populate all balanced sigma bins")
    count = max(len(bucket) for bucket in bins)
    balanced = torch.stack(tuple(bucket[index % len(bucket)] for index in range(count) for bucket in bins))
    return torch.cat((torch.ones(1, device=device), balanced))


def sample_conditioned_flow(
    initial_noise: Tensor,
    condition_features: Tensor | None,
    context: Tensor,
    *,
    predict_velocity: Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor],
    config: FlowSamplingConfig = FlowSamplingConfig(),
) -> Tensor:
    """Run the official Wan UniPC trajectory without accepting a target latent."""

    if not isinstance(initial_noise, Tensor) or initial_noise.ndim != 5:
        raise ValueError("initial_noise must have shape [B,C,F,H,W]")
    if not torch.is_floating_point(initial_noise) or not torch.isfinite(initial_noise).all():
        raise ValueError("initial_noise must be finite floating point")
    if context.shape[0] != initial_noise.shape[0]:
        raise ValueError("context batch must match initial_noise")
    if condition_features is not None and condition_features.shape[0] != initial_noise.shape[0]:
        raise ValueError("condition feature batch must match initial_noise")

    scheduler = _scheduler(config, initial_noise.device)
    sample = initial_noise.clone()
    for timestep in scheduler.timesteps:
        model_timestep = torch.full(
            (sample.shape[0],),
            float(timestep),
            device=sample.device,
            dtype=torch.float32,
        )
        velocity = predict_velocity(sample, model_timestep, context, condition_features)
        if velocity.shape != sample.shape or not torch.isfinite(velocity).all():
            raise RuntimeError("velocity prediction must be finite and match the sample shape")
        sample = scheduler.step(
            velocity,
            timestep,
            sample,
            return_dict=False,
        )[0]
    return sample


def oracle_sampling_audit(
    clean_latent: Tensor,
    initial_noise: Tensor,
    *,
    config: FlowSamplingConfig = FlowSamplingConfig(),
) -> dict[str, object]:
    """Audit the pinned UniPC trajectory with the analytic flow velocity.

    This diagnostic intentionally accepts clean/target tensors; the production
    sampler above does not. It reports error before and after every scheduler
    update so sigma alignment failures are localized.
    """

    if clean_latent.shape != initial_noise.shape:
        raise ValueError("clean_latent and initial_noise must have the same shape")
    if clean_latent.ndim != 5 or not torch.is_floating_point(clean_latent):
        raise ValueError("oracle audit tensors must have shape [B,C,F,H,W]")
    scheduler = _scheduler(config, initial_noise.device)
    sample = initial_noise.clone()
    rows = []
    velocity = initial_noise - clean_latent
    for index, timestep in enumerate(scheduler.timesteps):
        sigma = float(scheduler.sigmas[index])
        next_sigma = float(scheduler.sigmas[index + 1])
        analytic = (1 - sigma) * clean_latent + sigma * initial_noise
        updated = scheduler.step(velocity, timestep, sample, return_dict=False)[0]
        analytic_next = (1 - next_sigma) * clean_latent + next_sigma * initial_noise
        rows.append({
            "step": index,
            "sigma": sigma,
            "next_sigma": next_sigma,
            "pre_mean_abs_error": float((sample - analytic).abs().float().mean()),
            "pre_max_abs_error": float((sample - analytic).abs().float().max()),
            "post_mean_abs_error": float((updated - analytic_next).abs().float().mean()),
            "post_max_abs_error": float((updated - analytic_next).abs().float().max()),
        })
        sample = updated
    final_delta = (sample - clean_latent).abs().float()
    return {
        "config": {"steps": config.steps, "shift": config.shift},
        "rows": rows,
        "final_max_abs_error": float(final_delta.max()),
        "final_mean_abs_error": float(final_delta.mean()),
    }
