"""Retention loss functions for CRaFT hidden-feature alignment.

Primary API:
    compute_hidden_retention_loss(model_or_policy, anchor_batch, craft_config)
"""

from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _extract_model_inputs(anchor_batch: Dict[str, Any]) -> Dict[str, Any]:
    if "model_inputs" in anchor_batch and isinstance(anchor_batch["model_inputs"], dict):
        return anchor_batch["model_inputs"]

    fallback: Dict[str, Any] = {}
    for key in ("pixel_values", "input_ids", "attention_mask", "multimodal_indices", "image_token_mask"):
        if key in anchor_batch:
            fallback[key] = anchor_batch[key]
    if len(fallback) == 0:
        raise KeyError("anchor_batch must contain `model_inputs` or flat forward keys")
    return fallback


def _extract_target(anchor_batch: Dict[str, Any]) -> torch.Tensor:
    if "target_features" not in anchor_batch or anchor_batch["target_features"] is None:
        raise KeyError("anchor_batch must contain non-empty `target_features`")
    return anchor_batch["target_features"]


def _infer_num_image_tokens(model_or_policy: Any, hidden_len: int, attention_len: Optional[int]) -> int:
    if hasattr(model_or_policy, "vision_backbone") and hasattr(model_or_policy.vision_backbone, "num_patches"):
        return int(model_or_policy.vision_backbone.num_patches)
    if attention_len is None:
        return 0
    return max(hidden_len - attention_len, 0)


def _expand_attention_mask(attention_mask: torch.Tensor, num_image_tokens: int) -> torch.Tensor:
    if num_image_tokens <= 0:
        return attention_mask.bool()
    prefix = attention_mask[:, :1]
    suffix = attention_mask[:, 1:]
    mid = torch.ones(
        (attention_mask.shape[0], num_image_tokens),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([prefix, mid, suffix], dim=1).bool()


def _build_image_token_mask_from_layout(attention_mask: torch.Tensor, num_image_tokens: int) -> torch.Tensor:
    expanded = _expand_attention_mask(attention_mask, num_image_tokens)
    mask = torch.zeros_like(expanded, dtype=torch.bool)
    if num_image_tokens > 0:
        mask[:, 1 : 1 + num_image_tokens] = True
    return mask


def _fit_mask_to_hidden(mask: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    target_len = hidden_states.shape[1]
    if mask.shape[1] == target_len:
        return mask.bool()
    if mask.shape[1] > target_len:
        return mask[:, :target_len].bool()

    pad = torch.zeros((mask.shape[0], target_len - mask.shape[1]), dtype=torch.bool, device=mask.device)
    return torch.cat([mask.bool(), pad], dim=1)


def _resolve_pool_mask(
    model_or_policy: Any,
    model_inputs: Dict[str, Any],
    anchor_batch: Dict[str, Any],
    craft_config: Any,
    hidden_states: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    attention_mask = model_inputs.get("attention_mask")
    mask_key = getattr(craft_config, "retention_mask_key", None)
    pooling = getattr(craft_config, "pooling", "mean_image_tokens")

    if pooling == "mean_image_tokens":
        if "image_token_mask" in model_inputs:
            mask = model_inputs["image_token_mask"]
            return _fit_mask_to_hidden(mask, hidden_states), "model_inputs.image_token_mask"
        if "image_token_mask" in anchor_batch:
            mask = anchor_batch["image_token_mask"]
            return _fit_mask_to_hidden(mask, hidden_states), "anchor_batch.image_token_mask"

        if attention_mask is None:
            raise KeyError("mean_image_tokens requires attention_mask or image_token_mask")
        num_image_tokens = _infer_num_image_tokens(model_or_policy, hidden_states.shape[1], attention_mask.shape[1])
        derived = _build_image_token_mask_from_layout(attention_mask.bool(), num_image_tokens)
        return _fit_mask_to_hidden(derived, hidden_states), "derived_from_layout"

    if pooling == "mean_masked":
        if mask_key is not None:
            if mask_key in model_inputs:
                return _fit_mask_to_hidden(model_inputs[mask_key], hidden_states), f"model_inputs.{mask_key}"
            if mask_key in anchor_batch:
                return _fit_mask_to_hidden(anchor_batch[mask_key], hidden_states), f"anchor_batch.{mask_key}"

        if "retention_mask" in model_inputs:
            return _fit_mask_to_hidden(model_inputs["retention_mask"], hidden_states), "model_inputs.retention_mask"
        if "retention_mask" in anchor_batch:
            return _fit_mask_to_hidden(anchor_batch["retention_mask"], hidden_states), "anchor_batch.retention_mask"

        if attention_mask is None:
            raise KeyError("mean_masked requires retention_mask/mask_key or attention_mask")
        num_image_tokens = _infer_num_image_tokens(model_or_policy, hidden_states.shape[1], attention_mask.shape[1])
        expanded = _expand_attention_mask(attention_mask, num_image_tokens)
        return _fit_mask_to_hidden(expanded, hidden_states), "expanded_attention_mask"

    raise ValueError(f"Unsupported pooling mode: {pooling}")


def _pool_hidden(hidden_states: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
    pool_mask = pool_mask.bool()
    masked_hidden = hidden_states * pool_mask.unsqueeze(-1)
    denom = pool_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden_states.dtype)
    return masked_hidden.sum(dim=1) / denom


def _forward_with_hidden_states(model_or_policy: Any, model_inputs: Dict[str, Any]) -> Any:
    try:
        return model_or_policy(
            output_hidden_states=True,
            return_dict=True,
            **model_inputs,
        )
    except TypeError:
        llm_module = getattr(getattr(model_or_policy, "llm_backbone", None), "llm", model_or_policy)
        captured: Dict[str, Any] = {}

        def _hook(_: Any, __: Any, output: Any) -> None:
            if hasattr(output, "hidden_states") and output.hidden_states is not None:
                captured["hidden_states"] = output.hidden_states
            elif hasattr(output, "last_hidden_state"):
                captured["hidden_states"] = [output.last_hidden_state]
            elif isinstance(output, (list, tuple)) and len(output) > 0 and torch.is_tensor(output[0]):
                captured["hidden_states"] = [output[0]]

        handle = llm_module.register_forward_hook(_hook)
        try:
            output = model_or_policy(**model_inputs)
        finally:
            handle.remove()

        if not hasattr(output, "hidden_states") or output.hidden_states is None:
            if "hidden_states" not in captured:
                raise RuntimeError("Unable to capture hidden states via fallback hook")
            output = SimpleNamespace(hidden_states=captured["hidden_states"])
        return output


def compute_hidden_retention_loss(
    model_or_policy: Any,
    anchor_batch: Dict[str, Any],
    craft_config: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute hidden-state retention loss for one anchor batch.

    Required anchor_batch fields:
    - model_inputs: dict containing at least input_ids, attention_mask, pixel_values
    - target_features: Tensor with shape [B, D] or [D]
    Optional fields:
    - model_inputs.image_token_mask (preferred for mean_image_tokens)
    - model_inputs.retention_mask or custom key from craft_config.retention_mask_key (for mean_masked)
    """
    model_inputs = _extract_model_inputs(anchor_batch)
    target_features = _extract_target(anchor_batch)

    output = _forward_with_hidden_states(model_or_policy, model_inputs)
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model forward did not return hidden_states")

    hidden_layer = int(getattr(craft_config, "hidden_layer", -2))
    current_hidden = hidden_states[hidden_layer]
    pool_mask, mask_source = _resolve_pool_mask(model_or_policy, model_inputs, anchor_batch, craft_config, current_hidden)
    current_features = _pool_hidden(current_hidden, pool_mask)

    if target_features.dim() == 1:
        target_features = target_features.unsqueeze(0)
    target_features = target_features.to(device=current_features.device, dtype=current_features.dtype)

    loss_type = getattr(craft_config, "retention_loss_type", "mse").lower()
    current_float = current_features.float()
    target_float = target_features.float()

    if loss_type == "mse":
        loss = F.mse_loss(current_float, target_float)
    elif loss_type == "cosine":
        cosine = F.cosine_similarity(current_float, target_float, dim=-1)
        loss = (1.0 - cosine).mean()
    else:
        raise ValueError(f"Unsupported retention loss type: {loss_type}")

    metrics = {
        "retention_loss_value": float(loss.detach().cpu().item()),
        "hidden_layer": hidden_layer,
        "pooling": getattr(craft_config, "pooling", "mean_image_tokens"),
        "feature_dim": int(current_features.shape[-1]),
        "retention_loss_type": loss_type,
        "mask_source": mask_source,
    }
    return loss, metrics


def compute_retention_loss(
    current_features: torch.Tensor,
    anchor_features: torch.Tensor,
    retention_mode: str,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Backward-compatible generic retention loss helper."""
    if retention_mode not in {"mse_hidden", "mse_token"}:
        raise ValueError(f"Unsupported retention_mode: {retention_mode}")

    if mask is None:
        return F.mse_loss(current_features, anchor_features)

    masked_current = current_features[mask]
    masked_anchor = anchor_features[mask]
    return F.mse_loss(masked_current, masked_anchor)
