"""Displacement errors over K predicted modes (design doc §16, §21)."""

from __future__ import annotations

import numpy as np


def displacement_errors(trajectories: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Euclidean error per mode and frame: [B, K, T], from predictions [B, K, T, 2] and
    ground truth [B, T, 2]."""
    pred = np.asarray(trajectories, dtype=np.float64)
    truth = np.asarray(future, dtype=np.float64)
    if pred.ndim != 4 or truth.ndim != 3 or pred.shape[:1] + pred.shape[2:] != truth.shape:
        raise ValueError(f"shape mismatch: predictions {pred.shape}, ground truth {truth.shape}")
    return np.linalg.norm(pred - truth[:, None], axis=-1)


def ade(errors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean error over each sample's valid frames, per mode: [B, K]."""
    valid = np.asarray(valid, dtype=bool)
    counts = valid.sum(axis=1)
    if (counts == 0).any():
        raise ValueError("a sample has no valid future frames")
    return np.where(valid[:, None], errors, 0.0).sum(axis=2) / counts[:, None]


def fde(errors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Error at the final frame, per mode: [B, K]. Targets guarantee a valid final frame (§5.2)."""
    if not np.asarray(valid, dtype=bool)[:, -1].all():
        raise ValueError("final future frame is invalid for some samples; FDE is undefined")
    return errors[:, :, -1]


def top1_index(logits: np.ndarray) -> np.ndarray:
    """Index of the highest-probability mode per sample: [B]."""
    return np.asarray(logits).argmax(axis=1)


def take_mode(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """values[b, index[b]] for every sample b."""
    return values[np.arange(len(index)), index]
