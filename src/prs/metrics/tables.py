"""Results tables from metrics files (design doc §24.2-§24.3).

Rows are only combined when every file was computed on the identical sample set.
"""

from __future__ import annotations

from collections.abc import Sequence

GROUP_ORDER = ("overall", "pedestrian", "pedestrian/stationary", "pedestrian/moving",
               "cyclist", "cyclist/stationary", "cyclist/moving")  # fmt: skip


class ManifestMismatchError(ValueError):
    """Metrics files computed on different sample sets."""


def check_comparable(reports: Sequence[dict]) -> None:
    hashes = {(r["split"], r["manifest"]["sha256"]) for r in reports}
    if len(hashes) > 1:
        listing = "; ".join(
            f"{r['model']} ({r['tier']}/{r['split']}): {r['manifest']['sha256']}" for r in reports
        )
        raise ManifestMismatchError(f"metrics were computed on different sample sets: {listing}")


def _cell(summary: dict, name: str, fmt: str) -> str:
    if summary.get("count", 0) == 0 or name not in summary:
        return "N/A"
    ci = summary.get(f"{name}_ci95")
    value = format(summary[name], fmt)
    return value if ci is None else f"{value} ± {format(ci, fmt)}"


def _groups(report: dict) -> dict[str, dict]:
    return {"overall": report["overall"], **report["by_class"], **report["by_class_and_speed"]}


def main_table(reports: Sequence[dict]) -> str:
    """One row per model: the §24.2 table (top-1 ADE/FDE, best-of-K, miss rate)."""
    check_comparable(reports)
    lines = [
        "| Model | Map | Neighbors | Modes | ADE (m) | FDE (m) | minADE@K (m) | minFDE@K (m) "
        "| Miss rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        info, s = r["model_info"], r["overall"]
        lines.append(
            f"| {r['model']} | {'yes' if info['uses_map'] else 'no'} "
            f"| {'yes' if info['uses_neighbors'] else 'no'} | {info['num_modes']} "
            f"| {_cell(s, 'ade', '.2f')} | {_cell(s, 'fde', '.2f')} "
            f"| {_cell(s, 'min_ade', '.2f')} | {_cell(s, 'min_fde', '.2f')} "
            f"| {_cell(s, 'miss_rate', '.1%')} |"
        )
    return "\n".join(lines)


def breakdown_table(reports: Sequence[dict]) -> str:
    """Class x speed-bucket rows with sample counts: the §24.3 table."""
    check_comparable(reports)
    lines = [
        "| Group | n | Model | ADE (m) | FDE (m) | Miss rate |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for group in GROUP_ORDER:
        for r in reports:
            s = _groups(r).get(group)
            if s is None:
                continue
            lines.append(
                f"| {group} | {s['count']} | {r['model']} | {_cell(s, 'ade', '.2f')} "
                f"| {_cell(s, 'fde', '.2f')} | {_cell(s, 'miss_rate', '.1%')} |"
            )
    return "\n".join(lines)


def results_markdown(reports: Sequence[dict]) -> str:
    first = reports[0]
    manifest = first["manifest"]
    header = (
        f"Split `{first['split']}` of tier `{first['tier']}`: {manifest['num_samples']} samples, "
        f"manifest `{manifest['sha256']}`. Values are means ± 95% confidence intervals "
        "clustered by scene."
    )
    return "\n\n".join([header, main_table(reports), breakdown_table(reports)]) + "\n"
