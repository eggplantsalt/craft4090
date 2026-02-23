"""Retention loss entrypoints for CRaFT.

This module intentionally centralizes retention objective dispatch to avoid scattering
mode-branching logic across training scripts.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def compute_retention_loss(
    current_features: torch.Tensor,
    anchor_features: torch.Tensor,
    retention_mode: str,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute CRaFT retention loss from current and cached anchor features.

    Supported scaffold modes:
    - `mse_hidden`: plain MSE between feature tensors
    - `mse_token`: optional token-level MSE (same formula; mask can be applied)
    """
    if retention_mode not in {"mse_hidden", "mse_token"}:
        raise ValueError(f"Unsupported retention_mode: {retention_mode}")

    if mask is None:
        return F.mse_loss(current_features, anchor_features)

    masked_current = current_features[mask]
    masked_anchor = anchor_features[mask]
    return F.mse_loss(masked_current, masked_anchor)
