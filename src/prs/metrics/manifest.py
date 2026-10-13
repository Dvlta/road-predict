"""Sample identity for fair comparisons (design doc §24.1).

Every metrics file records the hash of the exact sample set it was computed on.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from prs.data.types import SampleMeta


def sample_key(meta: SampleMeta) -> str:
    return f"{meta.shard}/{meta.scenario_id}/{meta.target_track_id}/{meta.t0}"


def manifest_hash(keys: Iterable[str]) -> str:
    """Order-independent hash of a sample set. Duplicate keys indicate a bug and fail loudly."""
    keys = sorted(keys)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate sample keys in manifest")
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]
