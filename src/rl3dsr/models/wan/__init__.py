"""Frozen Wan2.1 integrations and Stage 1 LR conditioning."""

from rl3dsr.models.wan.checkpoint import WanCheckpoint
from rl3dsr.models.wan.dit import WanDiT
from rl3dsr.models.wan.geometry_conditioning import CameraBatch, GeometryConditioner, load_geometry_checkpoint
from rl3dsr.models.wan.flow import flow_matching_loss, flow_matching_pair
from rl3dsr.models.wan.lq_conditioning import (
    CausalLQ4xProjector,
    FrozenLQConditioner,
    Stage1Degradation,
    load_adapter_checkpoint,
    load_flashvsr_projector,
    save_adapter_checkpoint,
)
from rl3dsr.models.wan.vae import WanVAE

__all__ = [
    "CameraBatch",
    "GeometryConditioner",
    "load_geometry_checkpoint",
    "CausalLQ4xProjector",
    "FrozenLQConditioner",
    "Stage1Degradation",
    "WanCheckpoint",
    "WanDiT",
    "WanVAE",
    "flow_matching_loss",
    "flow_matching_pair",
    "load_adapter_checkpoint",
    "load_flashvsr_projector",
    "save_adapter_checkpoint",
]
