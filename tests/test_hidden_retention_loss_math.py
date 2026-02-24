"""CPU tests for hidden-state retention loss math and backward behavior."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.craft.retention_loss import compute_hidden_retention_loss


class TinyHiddenModel(nn.Module):
    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vision_backbone = SimpleNamespace(num_patches=2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: Dict[str, Any]) -> Any:
        token_features = F.one_hot((input_ids % 4), num_classes=4).float()
        h0 = self.proj(token_features)
        h1 = h0 + 1.0
        return SimpleNamespace(hidden_states=[h0, h1])


@dataclass
class DummyCraftConfig:
    hidden_layer: int = -1
    pooling: str = "mean_image_tokens"
    retention_loss_type: str = "mse"
    retention_mask_key: str = "retention_mask"


def test_hidden_retention_mse_value() -> None:
    model = TinyHiddenModel()
    cfg = DummyCraftConfig(hidden_layer=-1, pooling="mean_image_tokens", retention_loss_type="mse")

    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    image_token_mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        h = out.hidden_states[-1]
        pooled = (h * image_token_mask.unsqueeze(-1)).sum(dim=1) / image_token_mask.sum(dim=1, keepdim=True)
        target = pooled + 0.25
        expected = F.mse_loss(pooled.float(), target.float())

    anchor_batch = {
        "model_inputs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_token_mask": image_token_mask,
        },
        "target_features": target.detach(),
    }

    loss, metrics = compute_hidden_retention_loss(model, anchor_batch, cfg)
    assert torch.allclose(loss.detach(), expected.detach(), atol=1e-7)
    assert metrics["pooling"] == "mean_image_tokens"
    assert metrics["feature_dim"] == 4


def test_hidden_retention_cosine_value() -> None:
    model = TinyHiddenModel()
    cfg = DummyCraftConfig(hidden_layer=-1, pooling="mean_masked", retention_loss_type="cosine")

    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    retention_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        h = out.hidden_states[-1]
        pooled = (h * retention_mask.unsqueeze(-1)).sum(dim=1) / retention_mask.sum(dim=1, keepdim=True)
        target = -pooled
        expected = (1.0 - F.cosine_similarity(pooled.float(), target.float(), dim=-1)).mean()

    anchor_batch = {
        "model_inputs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "retention_mask": retention_mask,
        },
        "target_features": target.detach(),
    }

    loss, _ = compute_hidden_retention_loss(model, anchor_batch, cfg)
    assert torch.allclose(loss.detach(), expected.detach(), atol=1e-7)


def test_hidden_retention_backward() -> None:
    model = TinyHiddenModel()
    cfg = DummyCraftConfig(hidden_layer=-1, pooling="mean_image_tokens", retention_loss_type="mse")

    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    image_token_mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        h = out.hidden_states[-1]
        pooled = (h * image_token_mask.unsqueeze(-1)).sum(dim=1) / image_token_mask.sum(dim=1, keepdim=True)
        target = pooled + 0.5

    anchor_batch = {
        "model_inputs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_token_mask": image_token_mask,
        },
        "target_features": target,
    }

    loss, _ = compute_hidden_retention_loss(model, anchor_batch, cfg)
    loss.backward()

    assert model.proj.weight.grad is not None
    assert torch.isfinite(model.proj.weight.grad).all()
