"""Processed-cache fingerprint (design doc §11)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from prs.utils.config import DataConfig

# Fields that say where data lives or which shards a split uses, not how samples are built.
# Each cached shard is keyed by its own name, so changing them must not invalidate caches.
_NON_CONTENT_FIELDS = frozenset({"raw_root", "cache_root", "splits"})


def fingerprint_payload(cfg: DataConfig, preprocess_version: int) -> dict[str, Any]:
    """Every setting that changes processed samples (what the fingerprint hashes)."""
    payload = {
        f.name: getattr(cfg, f.name)
        for f in dataclasses.fields(cfg)
        if f.name not in _NON_CONTENT_FIELDS
    }
    payload["preprocess_version"] = preprocess_version
    return payload


def data_fingerprint(cfg: DataConfig, preprocess_version: int) -> str:
    blob = json.dumps(fingerprint_payload(cfg, preprocess_version), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
