"""Frozen Wan2.1 integrations and Stage 1 LR conditioning."""

from rl3dsr.models.wan.checkpoint import WanCheckpoint
from rl3dsr.models.wan.dit import WanDiT
from rl3dsr.models.wan.geometry_conditioning import (
    CameraBatch,
    FullRREConditioner,
    FullRREContext,
    GeometryConditioner,
    build_world_to_ray,
    load_geometry_checkpoint,
    save_geometry_checkpoint,
)
from rl3dsr.models.wan.flow import flow_matching_loss, flow_matching_pair
from rl3dsr.models.wan.lq_conditioning import (
    CausalLQ4xProjector,
    FrozenLQConditioner,
    Stage1Degradation,
    load_adapter_checkpoint,
    load_flashvsr_projector,
    save_adapter_checkpoint,
)
from rl3dsr.models.wan.lr_fusion import LRViewFusion, patch_fundamental_matrices
from rl3dsr.models.wan.sampling import FlowSamplingConfig, sample_conditioned_flow
from rl3dsr.models.wan.stage3 import (
    Stage3Conditioning,
    load_stage3_checkpoint,
    save_stage3_checkpoint,
)
from rl3dsr.models.wan.vae import WanVAE

__all__ = [
    "CameraBatch",
    "FullRREConditioner",
    "FullRREContext",
    "GeometryConditioner",
    "build_world_to_ray",
    "load_geometry_checkpoint",
    "save_geometry_checkpoint",
    "CausalLQ4xProjector",
    "FrozenLQConditioner",
    "FlowSamplingConfig",
    "LRViewFusion",
    "Stage1Degradation",
    "Stage3Conditioning",
    "WanCheckpoint",
    "WanDiT",
    "WanVAE",
    "flow_matching_loss",
    "flow_matching_pair",
    "load_adapter_checkpoint",
    "load_flashvsr_projector",
    "load_stage3_checkpoint",
    "patch_fundamental_matrices",
    "sample_conditioned_flow",
    "save_adapter_checkpoint",
    "save_stage3_checkpoint",
]
