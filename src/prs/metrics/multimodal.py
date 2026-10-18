"""Mode statistics for multimodal forecasts (design doc §20 collapse monitoring, §21).

minADE@K, minFDE@K and the top-1 errors are per-sample metrics (`breakdowns.per_sample_metrics`);
these describe the set of modes itself, and how well the probabilities rank the modes: the
"winner" is the mode closest to the ground truth, the same rule as the winner-takes-all loss.
"""

from __future__ import annotations

import numpy as np

from prs.metrics.breakdowns import clustered_ci95
from prs.metrics.displacement import ade, displacement_errors, fde

CALIBRATION_BINS = 10


def mode_probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax over modes: [B, K]."""
    shifted = np.asarray(logits, dtype=np.float64)
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def mode_entropy(logits: np.ndarray) -> np.ndarray:
    """Entropy of the mode probabilities in nats, per sample: [B]. 0 when one mode takes all
    the probability; log K when all modes are equally likely."""
    p = mode_probabilities(logits)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)


def endpoint_spread(trajectories: np.ndarray) -> np.ndarray:
    """Mean distance between the final points of every pair of modes, per sample: [B]. Near 0
    when the modes have collapsed onto one future."""
    ends = np.asarray(trajectories, dtype=np.float64)[:, :, -1]  # [B, K, 2]
    k = ends.shape[1]
    if k < 2:
        return np.zeros(len(ends))
    distances = np.linalg.norm(ends[:, :, None] - ends[:, None], axis=-1)  # [B, K, K]
    return distances.sum(axis=(1, 2)) / (k * (k - 1))


def winners(
    trajectories: np.ndarray, future: np.ndarray, future_valid: np.ndarray, selection: str
) -> np.ndarray:
    """The mode closest to the ground truth by ADE or FDE, per sample: [B]. The same rule as
    the winner-takes-all loss (`prs.losses.multimodal.select_winners`)."""
    errors = displacement_errors(trajectories, future)
    if selection == "ade":
        return ade(errors, future_valid).argmin(axis=1)
    if selection == "fde":
        return fde(errors, future_valid).argmin(axis=1)
    raise ValueError(f"unknown selection {selection!r}")


def ranking(logits: np.ndarray, won: np.ndarray) -> dict[str, np.ndarray]:
    """Per sample: whether the most probable mode is the winner, the probability given to the
    winner and its negative log, and the most probable mode's probability (its confidence)."""
    p = mode_probabilities(logits)
    winner_probability = p[np.arange(len(won)), won]
    return {
        "top1_hit": p.argmax(axis=1) == won,
        "winner_probability": winner_probability,
        "winner_nll": -np.log(np.clip(winner_probability, 1e-12, None)),
        "top1_confidence": p.max(axis=1),
    }


def calibration(
    confidence: np.ndarray, hit: np.ndarray, bins: int = CALIBRATION_BINS
) -> tuple[float, list[dict[str, float | int]]]:
    """Expected calibration error of the top-1 probability, and the reliability bins it comes
    from: samples grouped by confidence, each bin's mean confidence against how often the
    top-1 mode actually won. 0 means the probabilities can be taken at face value."""
    confidence, hit = np.asarray(confidence), np.asarray(hit, dtype=np.float64)
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    table, error = [], 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        mean_confidence, accuracy = float(confidence[mask].mean()), float(hit[mask].mean())
        error += mask.mean() * abs(accuracy - mean_confidence)
        table.append(
            {"count": int(mask.sum()), "confidence": mean_confidence, "accuracy": accuracy}
        )
    return float(error), table


def mode_statistics(
    trajectories: np.ndarray,
    logits: np.ndarray,
    future: np.ndarray,
    future_valid: np.ndarray,
    selection: str,
    clusters: np.ndarray | None = None,
) -> dict:
    """The collapse monitors (how often each mode wins, how far apart the modes end, how
    spread out their probabilities are) and the ranking metrics: top-1 accuracy (how often
    the most probable mode wins), the winner's mean probability and negative log-likelihood
    (log K for uniform probabilities), and the calibration error. With `clusters` (one scene
    id per sample), also 95% intervals for the ranking means as `<metric>_ci95`."""
    num_modes = np.asarray(trajectories).shape[1]
    won = winners(trajectories, future, future_valid, selection)
    ranked = ranking(logits, won)
    calibration_error, reliability = calibration(ranked["top1_confidence"], ranked["top1_hit"])
    stats: dict = {
        "win_rates": (np.bincount(won, minlength=num_modes) / len(won)).tolist(),
        "endpoint_spread_m": float(endpoint_spread(trajectories).mean()),
        "entropy": float(mode_entropy(logits).mean()),
        "uniform_nll": float(np.log(num_modes)),
    }
    for name, key in (
        ("top1_accuracy", "top1_hit"),
        ("winner_probability", "winner_probability"),
        ("winner_nll", "winner_nll"),
        ("top1_confidence", "top1_confidence"),
    ):
        values = ranked[key].astype(np.float64)
        stats[name] = float(values.mean())
        if clusters is not None:
            stats[f"{name}_ci95"] = clustered_ci95(values, clusters)
    stats["calibration_error"] = calibration_error
    stats["reliability"] = reliability
    return stats
