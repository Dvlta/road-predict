"""Paired model comparisons on one sample set (design doc §24.3).

Two models scored on the same samples share each sample's difficulty, so the per-sample
difference B - A is far less noisy than either model's own metric. Its scene-clustered 95%
interval is the sharp test of whether B beats A; comparing two separate intervals is
conservative.
"""

from __future__ import annotations

import numpy as np

from prs.metrics.breakdowns import clustered_ci95
from prs.metrics.miss_rate import misses

# Per-sample arrays saved next to every metrics JSON (scripts/evaluate.py).
PER_SAMPLE_FIELDS = (
    "keys", "scenario_id", "target_type", "current_speed_mps", "ade", "fde", "min_ade", "min_fde",
)  # fmt: skip
PAIRED_METRICS = ("ade", "fde", "miss_rate")


def _metric_values(per_sample: dict[str, np.ndarray], miss_threshold_m: float):
    return {
        "ade": np.asarray(per_sample["ade"], dtype=np.float64),
        "fde": np.asarray(per_sample["fde"], dtype=np.float64),
        "miss_rate": misses(per_sample["min_fde"], miss_threshold_m).astype(np.float64),
    }


def align(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reorder B's per-sample arrays to A's sample order; fails unless both hold exactly the
    same samples."""
    keys_a, keys_b = np.asarray(a["keys"]), np.asarray(b["keys"])
    if len(keys_a) != len(keys_b) or set(keys_a.tolist()) != set(keys_b.tolist()):
        raise ValueError("per-sample files hold different sample sets")
    order = {key: i for i, key in enumerate(keys_b.tolist())}
    index = np.array([order[key] for key in keys_a.tolist()])
    return {name: np.asarray(values)[index] for name, values in b.items()}


def paired_comparison(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    clusters: np.ndarray,
    miss_threshold_m: float,
) -> dict[str, dict]:
    """Per group: sample count and, per metric, A's mean, B's mean, the mean of B - A, and
    the clustered 95% half-width of that difference (None with fewer than two scenes)."""
    values_a = _metric_values(a, miss_threshold_m)
    values_b = _metric_values(b, miss_threshold_m)
    result: dict[str, dict] = {}
    for group, mask in masks.items():
        count = int(mask.sum())
        result[group] = {"count": count}
        if count == 0:
            continue
        for name in PAIRED_METRICS:
            diff = values_b[name][mask] - values_a[name][mask]
            result[group][name] = {
                "a": float(values_a[name][mask].mean()),
                "b": float(values_b[name][mask].mean()),
                "diff": float(diff.mean()),
                "diff_ci95": clustered_ci95(diff, np.asarray(clusters)[mask]),
            }
    return result


def paired_table(comparison: dict[str, dict], name_a: str, name_b: str) -> str:
    """Markdown: one row per group; `B - A` cells are means ± 95% half-widths, starred when
    the interval excludes zero."""
    lines = [
        f"| Group | n | ADE {name_a} | ADE {name_b} | ADE B − A | FDE {name_a} | FDE {name_b} "
        f"| FDE B − A | Miss {name_a} | Miss {name_b} | Miss B − A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in comparison.items():
        cells = [group, str(row["count"])]
        for name, fmt in (("ade", "+.3f"), ("fde", "+.3f"), ("miss_rate", "+.1%")):
            if name not in row:
                cells += ["N/A"] * 3
                continue
            m = row[name]
            plain = fmt.replace("+", "")
            ci = m["diff_ci95"]
            diff = format(m["diff"], fmt)
            if ci is not None:
                diff += f" ± {format(ci, plain)}" + ("*" if abs(m["diff"]) > ci else "")
            cells += [format(m["a"], plain), format(m["b"], plain), diff]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
