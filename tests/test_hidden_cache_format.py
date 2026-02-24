"""Tests for CRaFT hidden anchor cache schema and loader behavior.

These tests are CPU-only and validate:
- shard loading
- target feature dtype/shape
- dataloader collate output dimensions
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prismatic.craft.anchor_cache import AnchorCacheDataset, craft_anchor_collate


def _write_fake_hidden_shard(path: Path) -> None:
    records = []
    hidden_dim = 8
    for index in range(3):
        records.append(
            {
                "model_inputs": {
                    "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                    "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.bool),
                    "pixel_values": torch.randn(3, 16, 16, dtype=torch.float16),
                },
                "target_features": torch.randn(hidden_dim, dtype=torch.float16),
                "meta": {"sample_id": index, "pooling": "mean_image_tokens"},
            }
        )

    torch.save({"records": records, "meta": {"schema_version": 1}}, path)


def test_hidden_cache_dataset_and_collate(tmp_path: Path) -> None:
    shard_path = tmp_path / "hidden_anchor_shard_000000.pt"
    _write_fake_hidden_shard(shard_path)

    dataset = AnchorCacheDataset(cache_dir=tmp_path)
    assert len(dataset) == 3

    sample = dataset[0]
    assert sample["target_features"].shape == (8,)
    assert sample["target_features"].dtype == torch.float16
    assert isinstance(sample["model_inputs"], dict)
    assert "input_ids" in sample["model_inputs"]

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=craft_anchor_collate)
    batch = next(iter(dataloader))

    assert batch["target_features"].shape == (2, 8)
    assert batch["target_features"].dtype == torch.float16
    assert batch["model_inputs"]["input_ids"].shape == (2, 4)
    assert batch["model_inputs"]["attention_mask"].shape == (2, 4)
