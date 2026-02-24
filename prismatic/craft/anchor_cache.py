"""Anchor cache loading utilities for CRaFT.

Unified dataset return schema:
    {
        "model_inputs": Dict[str, Tensor | Dict[str, Tensor]],
        "target_features": Tensor | None,
        "token_features": Tensor | None,   # optional backward-compat for token cache
        "meta": Dict | None,
    }

Compatibility notes:
- New hidden-cache shards may contain:
  - a single record dict, or
  - a shard dict with `records: List[record]`
- Older flat formats are normalized into the schema above.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def _stack_nested(values: Sequence[Any]) -> Any:
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(values)
    if isinstance(first, dict):
        return {key: _stack_nested([value[key] for value in values]) for key in first.keys()}
    return list(values)


def _to_unified_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "model_inputs" in raw:
        model_inputs = raw["model_inputs"]
    else:
        model_inputs = {}
        for key in ("pixel_values", "input_ids", "attention_mask", "multimodal_indices", "image_token_mask"):
            if key in raw:
                model_inputs[key] = raw[key]

    if not isinstance(model_inputs, dict):
        raise ValueError("Expected `model_inputs` to be a dict")

    target_features = raw.get("target_features")
    token_features = raw.get("token_features", raw.get("target_tokens"))
    meta = raw.get("meta")

    return {
        "model_inputs": model_inputs,
        "target_features": target_features,
        "token_features": token_features,
        "meta": meta,
    }


class AnchorCacheDataset(Dataset):
    """Dataset over cached anchor shards with hidden/token cache compatibility."""

    def __init__(self, cache_dir: Path, file_pattern: str = "*.pt") -> None:
        self.cache_dir = cache_dir
        self.paths = sorted(cache_dir.glob(file_pattern))
        self._records: List[Dict[str, Any]] = []

        for path in self.paths:
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                for record in payload["records"]:
                    if not isinstance(record, dict):
                        raise ValueError(f"Invalid record type in shard {path}: {type(record)}")
                    self._records.append(_to_unified_record(record))
            elif isinstance(payload, list):
                for record in payload:
                    if not isinstance(record, dict):
                        raise ValueError(f"Invalid list record type in shard {path}: {type(record)}")
                    self._records.append(_to_unified_record(record))
            elif isinstance(payload, dict):
                self._records.append(_to_unified_record(payload))
            else:
                raise ValueError(f"Unsupported shard payload type in {path}: {type(payload)}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self._records:
            raise IndexError("AnchorCacheDataset is empty. Populate cache directory first.")
        return self._records[index]


def craft_anchor_collate(instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate CRaFT anchor records into a batched dictionary."""
    if len(instances) == 0:
        return {}

    output: Dict[str, Any] = {}
    output["model_inputs"] = _stack_nested([instance["model_inputs"] for instance in instances])

    target_features = [instance.get("target_features") for instance in instances]
    output["target_features"] = None if target_features[0] is None else _stack_nested(target_features)

    token_features = [instance.get("token_features") for instance in instances]
    output["token_features"] = None if token_features[0] is None else _stack_nested(token_features)

    output["meta"] = [instance.get("meta") for instance in instances]
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
