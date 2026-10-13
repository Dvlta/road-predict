"""Processed cache and Dataset (design doc §11, §27)."""

import dataclasses
import json
import pickle

import numpy as np
import pytest
import torch
from conftest import CYCLIST, PED

from prs.data import PREPROCESS_VERSION
from prs.data.cache import CACHE_FIELDS, CacheError, ShardCache, shard_cache_dir, write_shard_cache
from prs.data.dataset import CachedDataset, collate_batch
from prs.data.preprocess import build_sample
from prs.data.splits import shard_paths
from prs.data.windows import make_window
from prs.utils.fingerprint import data_fingerprint


@pytest.fixture
def cfg(data_cfg, tmp_path):
    return dataclasses.replace(data_cfg, cache_root=tmp_path / "cache")


@pytest.fixture
def samples(scenario, cfg):
    window = make_window(cfg)
    return [build_sample(scenario, i, cfg, window, shard="s") for i in (PED, CYCLIST)]


def _write(cfg, samples, shard_name, map_dtype="float32"):
    """Write samples as the cache of `shard_name`, as preprocessing would."""
    info = {
        "fingerprint": data_fingerprint(cfg, PREPROCESS_VERSION),
        "t0s": [20],
        "filter_counts": {"pedestrian": {"accepted": 1}, "cyclist": {"accepted": 1}},
    }
    directory = shard_cache_dir(cfg, shard_name)
    write_shard_cache(directory, samples, info, map_dtype)
    return directory


def test_cache_roundtrip(cfg, samples):
    shard = ShardCache(_write(cfg, samples, "s"), cfg, (20,))
    assert len(shard) == 2 and shard.metas() == [s.meta for s in samples]
    for name in CACHE_FIELDS:
        np.testing.assert_array_equal(
            shard.array(name), np.stack([np.asarray(getattr(s, name)) for s in samples]), name
        )


def test_cache_write_is_atomic_and_replaces(cfg, samples):
    directory = _write(cfg, samples, "s")
    _write(cfg, samples[:1], "s")
    assert not directory.with_name("s.tmp").exists()
    assert ShardCache(directory, cfg, (20,)).info["num_samples"] == 1


def test_cache_fingerprint_mismatch_fails(cfg, samples):
    directory = _write(cfg, samples, "s")
    changed = dataclasses.replace(cfg, map_radius_m=60.0)
    with pytest.raises(CacheError, match="no cache"):  # different settings, different directory
        ShardCache(shard_cache_dir(changed, "s"), changed, (20,))
    info = json.loads((directory / "meta.json").read_text())
    (directory / "meta.json").write_text(json.dumps({**info, "fingerprint": "0" * 16}))
    with pytest.raises(CacheError, match="fingerprint"):  # a copied or stale directory
        ShardCache(directory, cfg, (20,))


def test_cache_t0_mismatch_fails(cfg, samples):
    directory = _write(cfg, samples, "s")
    with pytest.raises(CacheError, match="t0s"):
        ShardCache(directory, cfg, (20, 25))


def test_float16_map_storage(cfg, samples):
    shard = ShardCache(_write(cfg, samples, "s", map_dtype="float16"), cfg, (20,))
    assert shard.array("map_polylines").dtype == np.float16
    np.testing.assert_allclose(
        shard.array("map_polylines"), np.stack([s.map_polylines for s in samples]), atol=0.05
    )


def test_dataset_reads_selected_fields(cfg, samples):
    for path in shard_paths(cfg, "val"):
        _write(cfg, samples, path.name, map_dtype="float16")
    dataset = CachedDataset(cfg, "val", fields=("target_history", "map_polylines"))
    assert len(dataset) == 2
    item = dataset[1]
    assert set(item) == {"target_history", "map_polylines", "index"} and item["index"] == 1
    assert item["map_polylines"].dtype == np.float32  # float16 storage is widened on load
    np.testing.assert_array_equal(item["target_history"], samples[1].target_history)
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch["target_history"].shape == (2, 21, 9) and batch["index"].tolist() == [0, 1]
    assert isinstance(batch["map_polylines"], torch.Tensor)
    with pytest.raises(IndexError):
        dataset[2]
    with pytest.raises(ValueError, match="unknown sample fields"):
        CachedDataset(cfg, "val", fields=("target_futures",))


def test_dataset_pickles_without_array_data(cfg, samples):
    """DataLoader workers receive a pickled dataset; memory maps must not travel with it."""
    for path in shard_paths(cfg, "val"):
        _write(cfg, samples, path.name)
    dataset = CachedDataset(cfg, "val")
    dataset[0]  # opens memory maps in this process
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.shards[0]._arrays == {}
    np.testing.assert_array_equal(restored[0]["map_polylines"], dataset[0]["map_polylines"])
