"""Displacement metrics, miss rate, breakdowns, intervals, manifests and tables (§16, §21, §24)."""

import json
from pathlib import Path

import numpy as np
import pytest

from prs.data.types import SampleMeta
from prs.metrics.breakdowns import (
    breakdown,
    clustered_ci95,
    group_masks,
    per_sample_metrics,
    summarize,
)
from prs.metrics.displacement import ade, displacement_errors, fde
from prs.metrics.manifest import manifest_hash, save_manifest
from prs.metrics.paired import align, paired_comparison, paired_table
from prs.metrics.tables import ManifestMismatchError, breakdown_table, main_table
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


def test_clustered_ci_matches_classic_ci_for_independent_samples():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    expected = 1.959964 * values.std(ddof=1) / np.sqrt(len(values))
    assert clustered_ci95(values, np.arange(200)) == pytest.approx(expected)


def test_clustered_ci_does_not_count_duplicates_as_evidence():
    """Two identical samples from one scene carry no more information than one."""
    values = np.random.default_rng(1).normal(size=100)
    once = clustered_ci95(values, np.arange(100))
    twice = clustered_ci95(np.repeat(values, 2), np.repeat(np.arange(100), 2))
    naive = clustered_ci95(np.repeat(values, 2), np.arange(200))
    assert twice == pytest.approx(once) and naive < 0.75 * once
    assert clustered_ci95(values, np.zeros(100)) is None  # one scene: no interval


def test_summarize_adds_intervals_with_clusters():
    truth = np.stack([_straight()] * 4)
    per_sample = per_sample_metrics(
        truth[:, None] + [0.0, 1.0], np.zeros((4, 1)), truth, np.ones((4, T), bool)
    )
    summary = summarize(per_sample, np.ones(4, bool), 2.0, 1, clusters=np.array([1, 1, 2, 3]))
    assert summary["ade"] == pytest.approx(1.0) and summary["ade_ci95"] == pytest.approx(0.0)
    assert set(summary) == {"count", "ade", "ade_ci95", "fde", "fde_ci95", "miss_rate",
                            "miss_rate_ci95"}  # fmt: skip


def test_save_manifest_is_content_addressed(tmp_path):
    metas = [_meta("s", "a", 1), _meta("s", "b", 2)]
    first = save_manifest(tmp_path, metas, "test", {"tier": "x"})
    again = save_manifest(tmp_path, metas[::-1], "test", {"tier": "x"})
    assert first == again and first["num_samples"] == 2
    saved = json.loads(Path(first["path"]).read_text())
    assert saved["keys"] == ["s/a/1/20", "s/b/2/20"] and saved["sha256"] == first["sha256"]


def _meta(shard, scenario_id, track_id):
    return SampleMeta(shard, scenario_id, track_id, 2, 20, 1.0, 5.0, False, 10.0, "velocity")


def _report(model, sha, num_modes=1):
    summary = {"count": 10, "ade": 1.0, "ade_ci95": 0.1, "fde": 2.0, "fde_ci95": 0.2,
               "miss_rate": 0.3, "miss_rate_ci95": 0.05}  # fmt: skip
    if num_modes > 1:
        summary |= {"min_ade": 0.5, "min_fde": 1.0}
    return {
        "model": model, "tier": "data_exp", "split": "test",
        "manifest": {"sha256": sha, "num_samples": 10},
        "model_info": {"uses_map": False, "uses_neighbors": False, "num_modes": num_modes},
        "overall": summary, "by_class": {"pedestrian": summary},
        "by_class_and_speed": {"pedestrian/moving": summary},
    }  # fmt: skip


def test_results_tables():
    reports = [_report("constant_velocity", "abc"), _report("multimodal", "abc", num_modes=6)]
    table = main_table(reports)
    assert "| constant_velocity | no | no | 1 | 1.00 ± 0.10 | 2.00 ± 0.20 | N/A | N/A |" in table
    assert "| 0.50 | 1.00 | 30.0% ± 5.0% |" in table
    assert len(breakdown_table(reports).splitlines()) == 2 + 3 * 2  # header + 3 groups x 2


def test_results_tables_refuse_different_sample_sets():
    with pytest.raises(ManifestMismatchError, match="different sample sets"):
        main_table([_report("a", "abc"), _report("b", "def")])


def _per_sample(n: int = 40, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    fde_values = rng.uniform(0.5, 4.0, n)
    return {
        "keys": np.array([f"s/{i // 2}/{i}/20" for i in range(n)]),
        "ade": fde_values / 2,
        "fde": fde_values,
        "min_ade": fde_values / 2,
        "min_fde": fde_values,
    }


def test_group_masks_order_and_buckets():
    masks = group_masks(
        np.array([2, 2, 3]), np.array([0.1, 1.0, 3.0]), ["pedestrian", "cyclist"], 0.5
    )
    assert list(masks) == ["overall", "pedestrian", "pedestrian/stationary", "pedestrian/moving",
                           "cyclist", "cyclist/stationary", "cyclist/moving"]  # fmt: skip
    assert masks["pedestrian/stationary"].tolist() == [True, False, False]
    assert masks["cyclist/moving"].tolist() == [False, False, True]


def test_paired_comparison_of_a_uniformly_better_model():
    a = _per_sample()
    b = {**a, "ade": a["ade"] - 0.25, "fde": a["fde"] - 0.5, "min_fde": a["fde"] - 0.5}
    clusters = np.array([key.split("/")[1] for key in a["keys"]])
    masks = {"overall": np.ones(40, dtype=bool)}
    result = paired_comparison(a, b, masks, clusters, miss_threshold_m=2.0)["overall"]
    assert result["count"] == 40
    assert result["fde"]["diff"] == pytest.approx(-0.5)
    assert result["fde"]["diff_ci95"] == pytest.approx(0.0, abs=1e-12)  # identical shift
    assert result["miss_rate"]["diff"] <= 0
    assert "*" in paired_table({"overall": result}, "A", "B")


def test_align_reorders_and_refuses_different_samples():
    a = _per_sample()
    order = np.random.default_rng(1).permutation(40)
    b = {name: values[order] for name, values in a.items()}
    aligned = align(a, b)
    np.testing.assert_array_equal(aligned["keys"], a["keys"])
    np.testing.assert_array_equal(aligned["fde"], a["fde"])
    with pytest.raises(ValueError, match="different sample sets"):
        align(a, {name: values[:-1] for name, values in a.items()})
