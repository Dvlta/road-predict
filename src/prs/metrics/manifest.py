"""Sample identity for fair comparisons (design doc §24.1).

Every metrics file records the hash of the exact sample set it was computed on, and the full
key list is saved once per sample set (content-addressed by that hash).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from prs.data.types import SampleMeta


def sample_key(meta: SampleMeta) -> str:
    return f"{meta.shard}/{meta.scenario_id}/{meta.target_track_id}/{meta.t0}"


def manifest_hash(keys: Iterable[str]) -> str:
    """Order-independent hash of a sample set. Duplicate keys indicate a bug and fail loudly."""
    keys = sorted(keys)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate sample keys in manifest")
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def save_manifest(
    out_dir: Path, metas: Sequence[SampleMeta], split: str, details: dict[str, Any]
) -> dict[str, Any]:
    """Write the sample set to `<out_dir>/<split>_<hash>.json` (once; identical sets share a
    file) and return the summary that metrics files embed."""
    keys = sorted(sample_key(m) for m in metas)
    sha = manifest_hash(keys)
    path = out_dir / f"{split}_{sha}.json"
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        content = {"split": split, "sha256": sha, "num_samples": len(keys), **details}
        path.write_text(json.dumps({**content, "keys": keys}, indent=1))
    return {"num_samples": len(keys), "sha256": sha, "path": str(path)}
