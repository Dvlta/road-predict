"""Per-sample metrics and the required breakdowns (design doc §16, §24.3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from prs.data.types import OBJECT_TYPE_BY_NAME
from prs.metrics.displacement import ade, displacement_errors, fde, take_mode, top1_index
from prs.metrics.miss_rate import misses
from prs.utils.config import EvaluationConfig


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


def summarize(
    per_sample: dict[str, np.ndarray], mask: np.ndarray, miss_threshold_m: float, num_modes: int
) -> dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        return {"count": 0}
    summary: dict[str, float | int] = {
        "count": count,
        "ade": float(per_sample["ade"][mask].mean()),
        "fde": float(per_sample["fde"][mask].mean()),
    }
    if num_modes > 1:
        summary["min_ade"] = float(per_sample["min_ade"][mask].mean())
        summary["min_fde"] = float(per_sample["min_fde"][mask].mean())
    summary["miss_rate"] = float(misses(per_sample["min_fde"][mask], miss_threshold_m).mean())
    return summary


def breakdown(
    per_sample: dict[str, np.ndarray],
    target_types: np.ndarray,
    current_speeds: np.ndarray,
    class_names: Sequence[str],
    eval_cfg: EvaluationConfig,
    num_modes: int,
) -> dict:
    """Overall, per-class and per-class x speed-bucket summaries, each with its sample count."""
    moving = np.asarray(current_speeds) >= eval_cfg.stationary_speed_threshold_mps

    def summary(mask: np.ndarray) -> dict[str, float | int]:
        return summarize(per_sample, mask, eval_cfg.miss_threshold_m, num_modes)

    result: dict = {
        "overall": summary(np.ones(len(moving), dtype=bool)),
        "by_class": {},
        "by_class_and_speed": {},
    }
    for name in class_names:
        is_class = np.asarray(target_types) == OBJECT_TYPE_BY_NAME[name]
        result["by_class"][name] = summary(is_class)
        for bucket, in_bucket in (("stationary", ~moving), ("moving", moving)):
            result["by_class_and_speed"][f"{name}/{bucket}"] = summary(is_class & in_bucket)
    first = per_sample["first_step_error"]
    result["sanity"] = {
        "first_step_error_mean_m": float(np.nanmean(first)),
        "first_step_error_median_m": float(np.nanmedian(first)),
    }
    return result
