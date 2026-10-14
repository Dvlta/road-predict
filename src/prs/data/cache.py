"""On-disk processed-sample cache (design doc §11, Layer B).

Layout, one directory per raw shard:

    <cache_root>/<fingerprint>/<shard file name>/
        meta.json       what was hashed into the fingerprint, t0s, counts, shapes, dtypes
        samples.jsonl   one SampleMeta per line
        <field>.npy     one array per schema field; first axis = sample

Arrays are memory-mapped and opened lazily, so a PyTorch DataLoader can pickle the cache into
worker processes without copying data. A cache whose fingerprint does not match the current
data config is never loaded (design doc §31).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from prs.data import PREPROCESS_VERSION
from prs.data.collate import collate_samples
from prs.data.preprocess import iter_shard_samples, split_t0s
from prs.data.splits import shard_paths
from prs.data.targets import FilterStats
from prs.data.types import ProcessedSample, SampleMeta
from prs.utils.config import DataConfig
from prs.utils.fingerprint import data_fingerprint, fingerprint_payload

CACHE_FIELDS = tuple(f.name for f in dataclasses.fields(ProcessedSample) if f.name != "meta")


class CacheError(RuntimeError):
    """A missing, stale or incompatible cache."""


def cache_dir(cfg: DataConfig) -> Path:
    return cfg.cache_root / data_fingerprint(cfg, PREPROCESS_VERSION)


def shard_cache_dir(cfg: DataConfig, shard_name: str) -> Path:
    return cache_dir(cfg) / shard_name


def write_shard_cache(
    out_dir: Path, samples: Sequence[ProcessedSample], info: dict[str, Any], map_dtype: str
) -> None:
    """Write one shard's samples atomically: a crash never leaves a half-written cache."""
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    field_info = {}
    if samples:
        batch = collate_samples(samples)
        for name in CACHE_FIELDS:
            if name not in batch:
                raise CacheError(f"samples lack {name!r}; the cache needs complete samples")
            array = batch[name].astype(map_dtype) if name == "map_polylines" else batch[name]
            np.save(tmp / f"{name}.npy", array)
            field_info[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    with (tmp / "samples.jsonl").open("w") as f:
        for sample in samples:
            f.write(json.dumps(dataclasses.asdict(sample.meta)) + "\n")
    (tmp / "meta.json").write_text(
        json.dumps({**info, "num_samples": len(samples), "fields": field_info}, indent=2)
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)


def build_shard_cache(path: Path, cfg: DataConfig, split: str) -> dict[str, Any]:
    """Build and write the cache for one raw shard; returns its meta.json contents."""
    start = time.perf_counter()
    t0s = split_t0s(cfg, split)
    stats = FilterStats()
    samples = list(iter_shard_samples(path, cfg, t0s, stats))
    metas = [s.meta for s in samples]
    info = {
        "fingerprint": data_fingerprint(cfg, PREPROCESS_VERSION),
        "fingerprint_payload": fingerprint_payload(cfg, PREPROCESS_VERSION),
        "source_shard": path.name,
        "source_bytes": path.stat().st_size,
        "t0s": list(t0s),
        "filter_counts": stats.as_dict(),
        "neighbor_cap_hit_fraction": _fraction(
            [m.neighbors_within_radius > cfg.max_neighbors for m in metas]
        ),
        "map_cap_hit_fraction": _fraction(
            [m.map_chunks_within_radius > cfg.max_map_polylines for m in metas]
        ),
        "build_seconds": round(time.perf_counter() - start, 1),
    }
    write_shard_cache(shard_cache_dir(cfg, path.name), samples, info, cfg.map_dtype)
    return {**info, "num_samples": len(samples)}


class ShardCache:
    """Read access to one cached shard."""

    def __init__(self, directory: Path, cfg: DataConfig, t0s: tuple[int, ...]) -> None:
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise CacheError(
                f"no cache at {directory} for the current data config; build it with "
                "`uv run python scripts/preprocess_waymo.py --config <data config>`"
            )
        info = json.loads(meta_path.read_text())
        expected = data_fingerprint(cfg, PREPROCESS_VERSION)
        if info["fingerprint"] != expected:
            raise CacheError(f"{directory}: fingerprint {info['fingerprint']} != {expected}")
        if tuple(info["t0s"]) != tuple(t0s):
            raise CacheError(f"{directory}: built with t0s {info['t0s']}, split needs {t0s}")
        self.directory = directory
        self.info = info
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return int(self.info["num_samples"])

    def array(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            self._arrays[name] = np.load(self.directory / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    def metas(self) -> list[SampleMeta]:
        with (self.directory / "samples.jsonl").open() as f:
            return [SampleMeta(**json.loads(line)) for line in f]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = {}  # memory maps are reopened in each process
        return state


def open_split_caches(cfg: DataConfig, split: str) -> list[ShardCache]:
    t0s = split_t0s(cfg, split)
    return [ShardCache(shard_cache_dir(cfg, p.name), cfg, t0s) for p in shard_paths(cfg, split)]


def _fraction(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None
