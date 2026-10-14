"""Batched inference over a DataLoader (shared by validation and scripts/evaluate.py)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from prs.data.dataset import LABEL_FIELDS
from prs.models.base import ForecastModel, model_inputs


def predict(
    model: ForecastModel, loader: DataLoader, device: torch.device | str = "cpu"
) -> dict[str, np.ndarray]:
    """Model outputs, labels and dataset indices for every sample, in loader order."""
    collected: dict[str, list[np.ndarray]] = {
        key: [] for key in ("trajectories", "logits", *LABEL_FIELDS, "index")
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(**model_inputs(batch, device))
            for key in ("trajectories", "logits"):
                collected[key].append(output[key].float().cpu().numpy())
            for key in (*LABEL_FIELDS, "index"):
                collected[key].append(batch[key].numpy())
    return {key: np.concatenate(values) for key, values in collected.items()}
