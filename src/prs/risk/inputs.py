"""Saved predictions of several models, aligned with the cached logged futures they are scored
against (design doc §22.6)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prs.data.dataset import CachedDataset
from prs.data.types import OBJECT_TYPE_BY_NAME, SampleMeta
from prs.metrics.manifest import manifest_hash, sample_key
from prs.metrics.paired import align
from prs.metrics.tables import check_comparable
from prs.utils.config import Config

LOGGED_FIELDS = ("target_future", "target_future_valid", "ego_future", "ego_future_valid",
                 "ego_size")  # fmt: skip


@dataclass(frozen=True)
class ScoringInputs:
    split: str
    tier: str
    manifest: dict
    metrics_files: dict[str, str]  # model name -> metrics JSON
    metas: list[SampleMeta]
    columns: dict[str, np.ndarray]  # cached fields, one row per sample
    predictions: dict[str, dict[str, np.ndarray]]  # model name -> trajectories, logits

    @property
    def models(self) -> list[str]:
        return list(self.predictions)

    @property
    def scenes(self) -> np.ndarray:
        return np.array([m.scenario_id for m in self.metas])

    def groups(self, inside: np.ndarray, target_types: tuple[str, ...]) -> dict[str, np.ndarray]:
        """Sample masks: `inside` overall and per target type."""
        types = np.array([m.target_type for m in self.metas])
        by_type = {name: inside & (types == OBJECT_TYPE_BY_NAME[name]) for name in target_types}
        return {"overall": inside, **by_type}


def load_scoring_inputs(
    metrics_paths: list[Path], cfg: Config, fields: tuple[str, ...] = LOGGED_FIELDS
) -> ScoringInputs:
    """Read metrics files (from scripts/evaluate.py) and the predictions they point to, check
    that they share one sample set, the config's tier and the cached split, and align every
    model's predictions with the cache. Raises ValueError on any mismatch."""
    reports = [json.loads(Path(path).read_text()) for path in metrics_paths]
    check_comparable(reports)
    names = [r["model"] for r in reports]
    if len(set(names)) != len(names):
        raise ValueError(f"two files hold the same model {names}")
    first = reports[0]
    if cfg.tier != first["tier"]:
        raise ValueError(f"the config uses tier {cfg.tier!r}, the metrics {first['tier']!r}")
    dataset = CachedDataset(cfg.data, first["split"], fields=[])
    metas = dataset.metas()
    keys = np.array([sample_key(m) for m in metas])
    if manifest_hash(keys.tolist()) != first["manifest"]["sha256"]:
        raise ValueError("the cached split is not the sample set of the metrics files")
    predictions = {
        report["model"]: align({"keys": keys}, dict(np.load(report["predictions"])))
        for report in reports
    }
    return ScoringInputs(
        split=first["split"],
        tier=first["tier"],
        manifest=first["manifest"],
        metrics_files={r["model"]: str(p) for r, p in zip(reports, metrics_paths, strict=True)},
        metas=metas,
        columns={name: dataset.column(name) for name in fields},
        predictions=predictions,
    )
