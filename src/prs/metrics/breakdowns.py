"""Per-sample metrics and the required breakdowns (design doc §16, §24.3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from prs.data.types import OBJECT_TYPE_BY_NAME
from prs.metrics.displacement import ade, displacement_errors, fde, take_mode, top1_index
from prs.metrics.miss_rate import misses
from prs.utils.config import EvaluationConfig

Z_95 = 1.959964


def per_sample_metrics(
    trajectories: np.ndarray, logits: np.ndarray, future: np.ndarray, future_valid: np.ndarray
) -> dict[str, np.ndarray]:
    """Top-1 and best-of-K errors per sample, plus the first-frame error (§14.3 check 4)."""
    errors = displacement_errors(trajectories, future)
    ade_k, fde_k = ade(errors, future_valid), fde(errors, future_valid)
    top1 = top1_index(logits)
    first_valid = np.asarray(future_valid, dtype=bool)[:, 0]
    return {
        "ade": take_mode(ade_k, top1),
        "fde": take_mode(fde_k, top1),
        "min_ade": ade_k.min(axis=1),
        "min_fde": fde_k.min(axis=1),
        "first_step_error": np.where(first_valid, take_mode(errors[:, :, 0], top1), np.nan),
    }


def clustered_ci95(values: np.ndarray, clusters: np.ndarray) -> float | None:
    """Half-width of a 95% confidence interval for the mean, robust to correlation within
    clusters. Samples from one scene share place, moment and neighbors, so treating them as
    independent would make intervals too narrow. None with fewer than two clusters."""
    values = np.asarray(values, dtype=np.float64)
    _, cluster_index = np.unique(np.asarray(clusters), return_inverse=True)
    num_clusters = int(cluster_index.max()) + 1 if len(values) else 0
    if num_clusters < 2:
        return None
    sums = np.bincount(cluster_index, weights=values - values.mean())
    variance = num_clusters / (num_clusters - 1) * (sums**2).sum() / len(values) ** 2
    return float(Z_95 * np.sqrt(variance))


def summarize(
    per_sample: dict[str, np.ndarray],
    mask: np.ndarray,
    miss_threshold_m: float,
    num_modes: int,
    clusters: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Means over the masked samples; with `clusters` (one scene id per sample), also 95%
    confidence-interval half-widths as `<metric>_ci95`."""
    count = int(mask.sum())
    if count == 0:
        return {"count": 0}
    values = {"ade": per_sample["ade"][mask], "fde": per_sample["fde"][mask]}
    if num_modes > 1:
        values["min_ade"] = per_sample["min_ade"][mask]
        values["min_fde"] = per_sample["min_fde"][mask]
    values["miss_rate"] = misses(per_sample["min_fde"][mask], miss_threshold_m).astype(float)
    summary: dict[str, float | int | None] = {"count": count}
    for name, v in values.items():
        summary[name] = float(v.mean())
        if clusters is not None:
            summary[f"{name}_ci95"] = clustered_ci95(v, np.asarray(clusters)[mask])
    return summary


def group_masks(
    target_types: np.ndarray,
    current_speeds: np.ndarray,
    class_names: Sequence[str],
    stationary_speed_threshold_mps: float,
) -> dict[str, np.ndarray]:
    """One boolean sample mask per reporting group, in table order: overall, then each class
    followed by its speed buckets ("pedestrian", "pedestrian/stationary", ...)."""
    moving = np.asarray(current_speeds) >= stationary_speed_threshold_mps
    masks = {"overall": np.ones(len(moving), dtype=bool)}
    for name in class_names:
        is_class = np.asarray(target_types) == OBJECT_TYPE_BY_NAME[name]
        masks[name] = is_class
        masks[f"{name}/stationary"] = is_class & ~moving
        masks[f"{name}/moving"] = is_class & moving
    return masks


def breakdown(
    per_sample: dict[str, np.ndarray],
    target_types: np.ndarray,
    current_speeds: np.ndarray,
    class_names: Sequence[str],
    eval_cfg: EvaluationConfig,
    num_modes: int,
    clusters: np.ndarray | None = None,
) -> dict:
    """Overall, per-class and per-class x speed-bucket summaries, each with its sample count."""
    masks = group_masks(
        target_types, current_speeds, class_names, eval_cfg.stationary_speed_threshold_mps
    )
    result: dict = {"by_class": {}, "by_class_and_speed": {}}
    for group, mask in masks.items():
        summary = summarize(per_sample, mask, eval_cfg.miss_threshold_m, num_modes, clusters)
        if group == "overall":
            result["overall"] = summary
        else:
            result["by_class_and_speed" if "/" in group else "by_class"][group] = summary
    result = {"overall": result["overall"], **result}
    first = per_sample["first_step_error"]
    result["sanity"] = {
        "first_step_error_mean_m": float(np.nanmean(first)),
        "first_step_error_median_m": float(np.nanmedian(first)),
    }
    return result
