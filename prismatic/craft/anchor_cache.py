"""Anchor cache loading scaffolding for CRaFT.

The CRaFT pipeline will use offline cached reference representations (hidden/token).
This module defines minimal dataset and loader interfaces for those cache shards.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


class AnchorCacheDataset(Dataset):
    """Dataset over cached anchor shards.

    Expected future shard format (placeholder):
    - dict with keys like `anchor_id`, `input_ids`, `pixel_values`, `anchor_features`
    """

    def __init__(self, cache_dir: Path, file_pattern: str = "*.pt") -> None:
        self.cache_dir = cache_dir
        self.paths = sorted(cache_dir.glob(file_pattern))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if not self.paths:
            raise IndexError("AnchorCacheDataset is empty. Populate cache directory first.")
        # TODO: validate schema and support memory-mapped / shard-level indexing.
        item = torch.load(self.paths[index], map_location="cpu")
        if not isinstance(item, dict):
            raise ValueError(f"Anchor cache file should contain dict, got: {type(item)}")
        return item


def craft_anchor_collate(instances: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Minimal collate function for cached anchors.

    Current behavior stacks tensor keys and keeps non-tensor keys as lists.
    """
    if len(instances) == 0:
        return {}

    output: Dict[str, torch.Tensor] = {}
    keys = instances[0].keys()
    for key in keys:
        values = [instance[key] for instance in instances]
        if torch.is_tensor(values[0]):
            output[key] = torch.stack(values)
        else:
            output[key] = values
    return output


def build_anchor_dataloader(
    cache_dir: Path,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    file_pattern: str = "*.pt",
) -> DataLoader:
    """Build dataloader for offline anchor cache shards."""
    dataset = AnchorCacheDataset(cache_dir=cache_dir, file_pattern=file_pattern)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=craft_anchor_collate,
    )
