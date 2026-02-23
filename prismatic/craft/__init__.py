"""CRaFT scaffolding package for OpenVLA.

This package contains integration stubs for Constrained Retention Fine-Tuning (CRaFT)
on top of the existing OpenVLA training flow.
"""

from .anchor_cache import AnchorCacheDataset, build_anchor_dataloader, craft_anchor_collate
from .craft_config import CraftConfig
from .grad_surgery import compute_dot, merge_grads, project_if_conflict
from .primal_dual import epsilon_schedule, update_lambda
from .retention_loss import compute_retention_loss

__all__ = [
    "AnchorCacheDataset",
    "CraftConfig",
    "build_anchor_dataloader",
    "compute_dot",
    "craft_anchor_collate",
    "epsilon_schedule",
    "merge_grads",
    "project_if_conflict",
    "update_lambda",
    "compute_retention_loss",
]
