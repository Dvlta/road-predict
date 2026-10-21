"""Ground-truth conflict labels and how well conflict scores find them (design doc §22.6).

The label applies the conflict rule to the target's logged future: one mode with probability
1, skipping frames where the target or the ego is unobserved. This is open loop: the logged ego
already reacted to the VRU, so ground-truth "conflicts" are mostly near-misses the driver or AV
accepted, not collisions.

Intervals come from a scene bootstrap: whole scenes are resampled with replacement, because
samples from one scene share the ego and the moment. Given the same scene ids, every call draws
the same resamples, so per-draw differences between two models are paired.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from prs.metrics.multimodal import mode_probabilities
from prs.risk.conflict import conflict_scores, mode_conflicts
from prs.utils.config import RiskConfig

SCORE_THRESHOLD = 0.5  # precision and recall are reported at conflict_score >= this (§22.6)
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 0
CHUNK = 2048  # samples per batch of separation arrays ([chunk, K, T] float64)


def analyzed(sdc_distance_m: np.ndarray, risk: RiskConfig) -> np.ndarray:
    """Samples inside the analysis radius (§22.5): the SDC is observed at t0 and at most
    max_sdc_distance_m from the target."""
    distance = np.asarray(sdc_distance_m, dtype=np.float64)
    return np.isfinite(distance) & (distance <= risk.max_sdc_distance_m)


def ground_truth_conflicts(
    target_future: np.ndarray,
    target_future_valid: np.ndarray,
    ego_future: np.ndarray,
    ego_future_valid: np.ndarray,
    ego_size: np.ndarray,
    risk: RiskConfig,
    dt: float,
) -> np.ndarray:
    """Whether the logged target future conflicts with the logged ego future: bool [B]."""
    return ground_truth_approaches(
        target_future, target_future_valid, ego_future, ego_future_valid, ego_size, risk, dt
    )["conflict"]


def ground_truth_approaches(
    target_future: np.ndarray,
    target_future_valid: np.ndarray,
    ego_future: np.ndarray,
    ego_future_valid: np.ndarray,
    ego_size: np.ndarray,
    risk: RiskConfig,
    dt: float,
) -> dict[str, np.ndarray]:
    """`mode_conflicts` of the logged target future, one value per sample [B]: whether it
    conflicts, and its closest approach (separation, index, time, ego speed). Frames where the
    target or the ego is invalid are skipped."""
    parts: dict[str, list[np.ndarray]] = {}
    for part in _chunks(len(target_future)):
        frame_valid = np.asarray(target_future_valid[part]) & np.asarray(ego_future_valid[part])
        modes = mode_conflicts(
            np.asarray(target_future[part])[:, None], ego_future[part], frame_valid,
            ego_size[part], risk, dt,
        )  # fmt: skip
        for name, values in modes.items():
            parts.setdefault(name, []).append(values[:, 0])
    return {name: np.concatenate(values) for name, values in parts.items()}


def predicted_conflict_scores(
    trajectories: np.ndarray,
    logits: np.ndarray,
    ego_future: np.ndarray,
    ego_future_valid: np.ndarray,
    ego_size: np.ndarray,
    risk: RiskConfig,
    dt: float,
) -> np.ndarray:
    """conflict_score per sample [B]: the probability mass of the predicted modes that
    conflict. In {0, 1} for single-mode models."""
    scores = []
    for part in _chunks(len(trajectories)):
        modes = mode_conflicts(
            trajectories[part], ego_future[part], ego_future_valid[part], ego_size[part], risk, dt
        )
        scores.append(conflict_scores(modes["conflict"], mode_probabilities(logits[part])))
    return np.concatenate(scores)


def precision_recall_curve(scores: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Precision and recall when flagging every sample with score >= each distinct score,
    highest threshold first. Recall is NaN without positives."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    ranked, hits = scores[order], labels[order]
    last = np.r_[np.flatnonzero(np.diff(ranked) != 0), len(ranked) - 1]  # end of each tie
    true_positives = np.cumsum(hits)[last]
    flagged = last + 1
    positives = labels.sum()
    return {
        "thresholds": ranked[last],
        "precision": true_positives / flagged,
        "recall": true_positives / positives if positives else np.full(len(last), np.nan),
    }


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the step precision-recall curve: the sum over thresholds of the recall gained
    times the precision there (no interpolation). None without positives."""
    if not np.asarray(labels, dtype=bool).any():
        return None
    curve = precision_recall_curve(scores, labels)
    recall_gained = np.diff(np.r_[0.0, curve["recall"]])
    return float((recall_gained * curve["precision"]).sum())


def precision_recall_at(
    scores: np.ndarray, labels: np.ndarray, threshold: float = SCORE_THRESHOLD
) -> tuple[float | None, float | None]:
    """Precision and recall when flagging score >= threshold; None when undefined (nothing
    flagged, or no positives)."""
    flagged = np.asarray(scores) >= threshold
    labels = np.asarray(labels, dtype=bool)
    true_positives = int((flagged & labels).sum())
    precision = true_positives / flagged.sum() if flagged.any() else None
    recall = true_positives / labels.sum() if labels.any() else None
    return precision, recall


def scene_bootstrap(
    clusters: np.ndarray, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> Iterator[np.ndarray]:
    """Sample indices of `draws` resamples of whole scenes, with replacement."""
    _, cluster_index = np.unique(np.asarray(clusters), return_inverse=True)
    members = np.argsort(cluster_index, kind="stable")  # samples grouped by scene
    counts = np.bincount(cluster_index)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        chosen = rng.integers(0, len(counts), len(counts))
        lengths = counts[chosen]
        ends = np.cumsum(lengths)
        position = np.arange(ends[-1]) - np.repeat(ends - lengths - starts[chosen], lengths)
        yield members[position]


def bootstrap_statistic(
    statistic: Callable[[np.ndarray], float | None],
    clusters: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """`statistic(sample indices)` on every scene resample: [draws], NaN where undefined."""
    values = [statistic(index) for index in scene_bootstrap(clusters, draws, seed)]
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def percentile_interval(values: np.ndarray) -> list[float] | None:
    """Central 95% of the finite bootstrap values; None if there are none."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return None
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def conflict_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    threshold: float = SCORE_THRESHOLD,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Count, ground-truth positives and base rate; average precision of the score; precision
    and recall at `threshold`; each metric with a 95% scene-bootstrap interval
    (`<metric>_ci95` = [low, high])."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    count = len(scores)
    summary: dict = {"count": count, "positives": int(labels.sum())}
    if count == 0:
        return summary
    summary["base_rate"] = float(labels.mean())
    summary["flagged"] = int((scores >= threshold).sum())
    statistics = {
        "average_precision": lambda i: average_precision(scores[i], labels[i]),
        "precision": lambda i: precision_recall_at(scores[i], labels[i], threshold)[0],
        "recall": lambda i: precision_recall_at(scores[i], labels[i], threshold)[1],
    }
    everything = np.arange(count)
    for name, statistic in statistics.items():
        summary[name] = statistic(everything)
        summary[f"{name}_ci95"] = percentile_interval(
            bootstrap_statistic(statistic, clusters, draws, seed)
        )
    return summary


def compare_conflict_scores(
    scores: dict[str, np.ndarray],
    labels: np.ndarray,
    clusters: np.ndarray,
    threshold: float = SCORE_THRESHOLD,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict]:
    """`conflict_summary` for each model's scores on the same samples. Every model after the
    first also gets `average_precision_diff`, its average precision minus the first model's,
    with a paired 95% scene-bootstrap interval."""
    labels = np.asarray(labels, dtype=bool)
    names = list(scores)
    reference = names[0]
    draws_by_model = {
        name: _average_precision_draws(np.asarray(scores[name]), labels, clusters, draws, seed)
        for name in names
    }
    result = {}
    for name in names:
        result[name] = conflict_summary(scores[name], labels, clusters, threshold, draws, seed)
        if name != reference and result[name].get("average_precision") is not None:
            result[name]["average_precision_diff"] = (
                result[name]["average_precision"] - result[reference]["average_precision"]
            )
            result[name]["average_precision_diff_ci95"] = percentile_interval(
                draws_by_model[name] - draws_by_model[reference]
            )
    return result


def conflict_table(groups: dict[str, dict[str, dict]], positives: str = "GT conflicts") -> str:
    """Markdown: one row per group and model, from `compare_conflict_scores` results keyed by
    group; `positives` heads the label count. Intervals are 95% scene-bootstrap; `*` marks AP
    differences whose interval excludes zero."""
    reference = next(iter(next(iter(groups.values()))))
    lines = [
        f"| Group | Model | n | {positives} | Base rate | AP | AP − {reference} "
        "| Flagged | Precision | Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, models in groups.items():
        for name, s in models.items():
            diff = "—"
            if s.get("average_precision_diff") is not None:
                diff = _with_interval(
                    s["average_precision_diff"], s["average_precision_diff_ci95"], "+.3f"
                )
                interval = s["average_precision_diff_ci95"]
                if interval is not None and (interval[0] > 0 or interval[1] < 0):
                    diff += "*"
            base_rate = "N/A" if s["count"] == 0 else format(s["base_rate"], ".2%")
            lines.append(
                f"| {group} | {name} | {s['count']} | {s['positives']} | {base_rate} "
                f"| {_cell(s, 'average_precision', '.3f')} | {diff} | {s.get('flagged', 0)} "
                f"| {_cell(s, 'precision', '.2f')} | {_cell(s, 'recall', '.2f')} |"
            )
    return "\n".join(lines)


def _average_precision_draws(
    scores: np.ndarray, labels: np.ndarray, clusters: np.ndarray, draws: int, seed: int
) -> np.ndarray:
    return bootstrap_statistic(
        lambda i: average_precision(scores[i], labels[i]), clusters, draws, seed
    )


def _cell(summary: dict, name: str, fmt: str) -> str:
    return _with_interval(summary.get(name), summary.get(f"{name}_ci95"), fmt)


def _with_interval(value: float | None, interval: list[float] | None, fmt: str) -> str:
    if value is None:
        return "N/A"
    if interval is None:
        return format(value, fmt)
    plain = fmt.replace("+", "")
    return f"{format(value, fmt)} ({format(interval[0], plain)} to {format(interval[1], plain)})"


def _chunks(n: int) -> Iterator[slice]:
    for start in range(0, n, CHUNK):
        yield slice(start, min(start + CHUNK, n))
