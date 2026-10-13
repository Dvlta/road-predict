"""In-memory sample building, batching and rendering."""

import dataclasses

import numpy as np
import pytest
from conftest import CYCLIST, PED, TRAINING_SHARD_0

from prs.data.collate import collate_samples
from prs.data.preprocess import iter_shard_samples, split_t0s
from prs.data.targets import FilterStats, build_target_sample
from prs.data.windows import make_window
from prs.visualization.predictions import render_sample


def test_split_t0s(data_cfg):
    augmented = dataclasses.replace(data_cfg, train_current_indices=(20, 25, 30))
    assert split_t0s(augmented, "train") == (20, 25, 30)
    for split in ("val", "test"):  # never several t0 per scene outside training (§5.1)
        assert split_t0s(augmented, split) == (20,)


def test_collate_stacks_fields_and_omits_missing_context(scenario, data_cfg):
    window = make_window(data_cfg)
    samples = [build_target_sample(scenario, i, data_cfg, window) for i in (PED, CYCLIST)]
    batch = collate_samples(samples)
    assert batch["target_history"].shape == (2, 21, 9)
    assert batch["target_type"].tolist() == [2, 3]
    assert batch["heading_global"].dtype == np.float64
    assert [m.target_track_id for m in batch["meta"]] == [7, 9]
    assert "neighbors" not in batch and "map_polylines" not in batch
    with pytest.raises(ValueError, match="empty"):
        collate_samples([])


def test_render_sample_with_predictions(scenario, data_cfg, tmp_path):
    sample = build_target_sample(scenario, PED, data_cfg, make_window(data_cfg))
    out = tmp_path / "figure.png"
    predictions = np.stack([sample.target_future, sample.target_future + [0.0, 1.0]])
    render_sample(scenario, sample, data_cfg, out, predictions, np.array([0.7, 0.3]), "test")
    assert out.stat().st_size > 10_000


@pytest.mark.data
@pytest.mark.skipif(not TRAINING_SHARD_0.exists(), reason="training shard 0 not downloaded")
def test_shard_samples_match_filter_counts(data_cfg):
    stats = FilterStats()
    samples = list(iter_shard_samples(TRAINING_SHARD_0, data_cfg, (20,), stats))
    accepted = sum(stats.accepted(name) for name in data_cfg.target_types)
    assert len(samples) == accepted > 0
