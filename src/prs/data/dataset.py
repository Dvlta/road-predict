"""PyTorch Dataset over the processed cache (design doc §11, §29)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from prs.data.cache import CACHE_FIELDS, open_split_caches
from prs.data.types import SampleMeta
from prs.utils.config import DataConfig

# Ground truth needed to score any model.
LABEL_FIELDS = ("target_future", "target_future_valid", "target_type")


class CachedDataset(Dataset):
    """One split's samples. Reads only `fields`, so a model that ignores the map never pays to
    load it."""

    def __init__(self, cfg: DataConfig, split: str, fields: Sequence[str] | None = None) -> None:
        self.fields = tuple(CACHE_FIELDS if fields is None else fields)
        unknown = sorted(set(self.fields) - set(CACHE_FIELDS))
        if unknown:
            raise ValueError(f"unknown sample fields {unknown}")
        self.shards = open_split_caches(cfg, split)
        self._offsets = np.cumsum([0] + [len(s) for s in self.shards])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_number = int(np.searchsorted(self._offsets, index, side="right") - 1)
        shard, local = self.shards[shard_number], index - self._offsets[shard_number]
        item: dict[str, Any] = {name: np.array(shard.array(name)[local]) for name in self.fields}
        if "map_polylines" in item:
            item["map_polylines"] = item["map_polylines"].astype(np.float32)
        item["index"] = index
        return item

    def metas(self) -> list[SampleMeta]:
        return [meta for shard in self.shards for meta in shard.metas()]

    def filter_counts(self) -> dict[str, dict[str, int]]:
        """Target-filter outcomes summed over the split's shards."""
        total: dict[str, dict[str, int]] = {}
        for shard in self.shards:
            for type_name, counts in shard.info["filter_counts"].items():
                for reason, n in counts.items():
                    total.setdefault(type_name, {}).setdefault(reason, 0)
                    total[type_name][reason] += n
        return {name: dict(sorted(total[name].items())) for name in sorted(total)}


def collate_batch(items: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """DataLoader collate: stack each field into a tensor (leading axis = sample)."""
    return {key: torch.as_tensor(np.stack([item[key] for item in items])) for key in items[0]}
