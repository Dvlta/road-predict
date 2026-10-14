"""Stack processed samples into batch arrays (leading axis = sample)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from typing import Any

import numpy as np

from prs.data.types import ProcessedSample


def collate_samples(samples: Sequence[ProcessedSample]) -> dict[str, Any]:
    """Batch dict: one stacked array per schema field plus `meta` (a list of SampleMeta).

    Context fields that every sample leaves as None (before Stage 3) are omitted.
    """
    if not samples:
        raise ValueError("cannot collate an empty list of samples")
    batch: dict[str, Any] = {"meta": [s.meta for s in samples]}
    for field in fields(ProcessedSample):
        if field.name == "meta":
            continue
        values = [getattr(s, field.name) for s in samples]
        present = [v is not None for v in values]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"{field.name}: set in some samples and None in others")
        batch[field.name] = np.stack([np.asarray(v) for v in values])
    return batch
