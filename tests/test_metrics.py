"""Displacement metrics, miss rate, breakdowns and manifests (design doc §16, §21, §27)."""

import numpy as np
import pytest

from prs.metrics.breakdowns import breakdown, per_sample_metrics
from prs.metrics.displacement import ade, displacement_errors, fde
from prs.metrics.manifest import manifest_hash
from prs.utils.config import EvaluationConfig

T = 60


def _straight(length_m: float = 6.0) -> np.ndarray:
    """Ground truth: straight along +x, [T, 2]."""
    return np.stack([np.linspace(length_m / T, length_m, T), np.zeros(T)], axis=1)


def test_zero_error_ade():
    truth = _straight()[None]
    errors = displacement_errors(truth[:, None], truth)
    valid = np.ones((1, T), dtype=bool)
    assert ade(errors, valid)[0, 0] == 0.0 and fde(errors, valid)[0, 0] == 0.0


def test_known_fde():
    truth = _straight()[None]
    prediction = truth.copy()
    prediction[0, -1] += [3.0, 4.0]  # final point off by a 3-4-5 triangle
    errors = displacement_errors(prediction[:, None], truth)
    valid = np.ones((1, T), dtype=bool)
    assert fde(errors, valid)[0, 0] == pytest.approx(5.0)
    assert ade(errors, valid)[0, 0] == pytest.approx(5.0 / T)


def test_masked_metric():
    truth = _straight()[None]
    prediction = truth.copy()
    prediction[0, 10] += 1000.0  # a wild error at a frame with no ground truth
    valid = np.ones((1, T), dtype=bool)
    valid[0, 10] = False
    errors = displacement_errors(prediction[:, None], truth)
    assert ade(errors, valid)[0, 0] == 0.0


def test_fde_requires_valid_final_frame():
    truth = _straight()[None]
    valid = np.ones((1, T), dtype=bool)
    valid[0, -1] = False
    with pytest.raises(ValueError, match="FDE is undefined"):
        fde(displacement_errors(truth[:, None], truth), valid)


def test_ade_rejects_samples_without_valid_frames():
    truth = _straight()[None]
    with pytest.raises(ValueError, match="no valid future"):
        ade(displacement_errors(truth[:, None], truth), np.zeros((1, T), dtype=bool))


def test_minade_selects_best_mode():
    truth = _straight()[None]
    modes = np.stack([truth[0] + [0.0, 2.0], truth[0] + [0.0, 0.5], truth[0] + [0.0, 1.0]])[None]
    logits = np.array([[3.0, 0.0, 1.0]])  # top-1 is mode 0, the worst one
    metrics = per_sample_metrics(modes, logits, truth, np.ones((1, T), dtype=bool))
    assert metrics["min_ade"][0] == pytest.approx(0.5) and metrics["min_fde"][0] == pytest.approx(
        0.5
    )
    assert metrics["ade"][0] == pytest.approx(2.0) and metrics["fde"][0] == pytest.approx(2.0)
    assert metrics["first_step_error"][0] == pytest.approx(2.0)


def test_breakdown_groups_and_miss_rate():
    truth = np.stack([_straight()] * 4)
    offsets = np.array([0.5, 3.0, 1.0, 2.5])  # FDE per sample; misses beyond 2 m: samples 1, 3
    prediction = truth.copy()
    prediction[:, :, 1] += offsets[:, None]
    per_sample = per_sample_metrics(
        prediction[:, None], np.zeros((4, 1)), truth, np.ones((4, T), dtype=bool)
    )
    result = breakdown(
        per_sample,
        target_types=np.array([2, 2, 3, 3]),  # pedestrian, pedestrian, cyclist, cyclist
        current_speeds=np.array([0.1, 1.4, 4.0, 0.2]),
        class_names=("pedestrian", "cyclist"),
        eval_cfg=EvaluationConfig(miss_threshold_m=2.0, stationary_speed_threshold_mps=0.5),
        num_modes=1,
    )
    assert result["overall"]["count"] == 4 and result["overall"]["miss_rate"] == 0.5
    assert "min_ade" not in result["overall"]  # deterministic models report top-1 only
    assert result["by_class"]["pedestrian"]["fde"] == pytest.approx(1.75)
    speed = result["by_class_and_speed"]
    assert speed["pedestrian/stationary"] == {"count": 1, "ade": 0.5, "fde": 0.5, "miss_rate": 0.0}
    assert speed["cyclist/stationary"]["miss_rate"] == 1.0
    assert speed["cyclist/moving"]["count"] == 1


def test_breakdown_reports_empty_groups():
    truth = _straight()[None]
    per_sample = per_sample_metrics(truth[:, None], np.zeros((1, 1)), truth, np.ones((1, T), bool))
    result = breakdown(
        per_sample, np.array([2]), np.array([1.0]), ("pedestrian", "cyclist"),
        EvaluationConfig(2.0, 0.5), 1,
    )  # fmt: skip
    assert result["by_class"]["cyclist"] == {"count": 0}


def test_manifest_hash_order_independent_and_rejects_duplicates():
    assert manifest_hash(["a", "b", "c"]) == manifest_hash(["c", "a", "b"])
    assert manifest_hash(["a", "b"]) != manifest_hash(["a", "c"])
    with pytest.raises(ValueError, match="duplicate"):
        manifest_hash(["a", "a"])
