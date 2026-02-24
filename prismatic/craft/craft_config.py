"""Configuration schema for CRaFT training in OpenVLA.

This file intentionally provides a minimal but explicit dataclass surface so later prompts
can wire full CRaFT behavior without changing baseline OpenVLA model definitions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CraftConfig:
    """Config container for CRaFT-specific controls.

    Fields are intentionally broad to cover upcoming integration stages:
    - retention loss mode and budget scheduling
    - gradient surgery projection controls
    - primal-dual lambda update controls
    - offline anchor-cache loading controls
    """

    enabled: bool = True
    retention_mode: str = "mse_hidden"

    epsilon: float = 0.05
    epsilon_final: Optional[float] = None
    epsilon_schedule_type: str = "constant"

    lambda_lr: float = 1e-4
    lambda_init: float = 0.0
    lambda_max: Optional[float] = None

    hidden_layer: int = -2
    pooling: str = "mean_image_tokens"
    retention_loss_type: str = "mse"
    retention_mask_key: Optional[str] = None
    delta: float = 1e-12

    anchor_cache_dir: Path = Path("cache/craft/anchors")
    anchor_batch_size: int = 16
    anchor_num_workers: int = 0
    anchor_cache_format: str = "hidden"

    retain_every_n_steps: int = 1
    dry_run_max_batches: int = 1
