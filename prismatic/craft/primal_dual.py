"""Primal-dual helper utilities for CRaFT.

These helpers are designed to stay independent from the training engine so they can
be unit-tested in isolation and integrated into the OpenVLA trainer later.
"""

from typing import Optional


def epsilon_schedule(
    step: int,
    max_steps: Optional[int],
    epsilon_start: float,
    epsilon_final: Optional[float],
    schedule_type: str = "constant",
) -> float:
    """Return active retention budget epsilon at `step`."""
    if schedule_type == "constant" or epsilon_final is None or max_steps in (None, 0):
        return epsilon_start

    if schedule_type == "linear":
        ratio = min(max(step / max_steps, 0.0), 1.0)
        return epsilon_start + ratio * (epsilon_final - epsilon_start)

    raise ValueError(f"Unsupported epsilon schedule_type: {schedule_type}")


def update_lambda(
    lambda_value: float,
    lambda_lr: float,
    retention_loss: float,
    epsilon: float,
    lambda_max: Optional[float] = None,
) -> float:
    """Projected gradient-ascent update for dual variable lambda.

    Rule:
        lambda <- [lambda + lambda_lr * (retention_loss - epsilon)]_+
    """
    updated = max(lambda_value + lambda_lr * (retention_loss - epsilon), 0.0)
    if lambda_max is not None:
        updated = min(updated, lambda_max)
    return updated
