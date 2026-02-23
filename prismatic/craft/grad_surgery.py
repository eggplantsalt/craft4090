"""Gradient surgery utilities for CRaFT.

Current scope is a thin, reusable utility layer. Full training-loop integration
(`two backward passes`, `gradient projection`, `merged step`) is deferred.
"""

from typing import Iterable, List, Optional, Sequence

import torch


def _valid_pairs(
    grads_a: Sequence[Optional[torch.Tensor]],
    grads_b: Sequence[Optional[torch.Tensor]],
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is None or grad_b is None:
            continue
        yield grad_a, grad_b


def compute_dot(
    grads_a: Sequence[Optional[torch.Tensor]],
    grads_b: Sequence[Optional[torch.Tensor]],
) -> torch.Tensor:
    """Compute dot product between two gradient lists, skipping `None` entries."""
    dot = None
    for grad_a, grad_b in _valid_pairs(grads_a, grads_b):
        value = torch.sum(grad_a * grad_b)
        dot = value if dot is None else dot + value

    if dot is None:
        return torch.tensor(0.0)
    return dot


def project_if_conflict(
    grad_act: Sequence[Optional[torch.Tensor]],
    grad_ret: Sequence[Optional[torch.Tensor]],
    delta: float,
) -> List[Optional[torch.Tensor]]:
    """Project action gradients if they conflict with retention gradients.

    Projection rule (per CRaFT paper idea):
        if dot(grad_act, grad_ret) < 0:
            grad_act <- grad_act - dot / (||grad_ret||^2 + delta) * grad_ret
    """
    dot = compute_dot(grad_act, grad_ret)
    if dot >= 0:
        return [g.clone() if g is not None else None for g in grad_act]

    norm_sq = compute_dot(grad_ret, grad_ret)
    scale = dot / (norm_sq + delta)

    projected: List[Optional[torch.Tensor]] = []
    for grad_a, grad_b in zip(grad_act, grad_ret):
        if grad_a is None:
            projected.append(None)
        elif grad_b is None:
            projected.append(grad_a.clone())
        else:
            projected.append(grad_a - scale * grad_b)
    return projected


def merge_grads(
    grad_act_projected: Sequence[Optional[torch.Tensor]],
    grad_ret: Sequence[Optional[torch.Tensor]],
    lambda_value: float,
) -> List[Optional[torch.Tensor]]:
    """Merge projected action gradients with retention gradients.

    Intended merged update direction:
        g_total = g_act_projected + lambda * g_ret
    """
    merged: List[Optional[torch.Tensor]] = []
    for grad_a, grad_r in zip(grad_act_projected, grad_ret):
        if grad_a is None and grad_r is None:
            merged.append(None)
        elif grad_a is None:
            merged.append(lambda_value * grad_r)
        elif grad_r is None:
            merged.append(grad_a.clone())
        else:
            merged.append(grad_a + lambda_value * grad_r)
    return merged
