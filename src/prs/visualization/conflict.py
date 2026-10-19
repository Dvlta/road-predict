"""Conflict figures (design doc §22, §23): closest approaches drawn on the BEV figure, and
precision-recall curves of conflict scores against ground-truth conflicts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from prs.risk.geometry import closest_point_on_box
from prs.visualization.bev import draw_box


@dataclass(frozen=True)
class ClosestApproach:
    """Where a VRU path comes closest to the ego footprint: future index `index` ((index + 1)
    * dt after t0) and the VRU position `point` then, annotated with `text` (offset from the
    point, in points)."""

    index: int
    point: np.ndarray
    text: str
    color: str
    label: str
    text_offset: tuple[float, float] = (8, -14)

    def legend_handle(self) -> Line2D:
        return Line2D([], [], color=self.color, ls="--", marker="o", mec="black",
                      label=self.label)  # fmt: skip


def draw_closest_approach(
    ax: Axes, ego_future: np.ndarray, ego_size: np.ndarray, approach: ClosestApproach
) -> Line2D:
    """The ego footprint at the approach's index, the VRU position then, and the gap between
    them. Returns a legend handle."""
    x, y, heading = ego_future[approach.index, :3]
    length, width = ego_size
    color = approach.color
    draw_box(ax, (x, y), heading, length, width, facecolor="none", edgecolor=color, ls="--",
             lw=1.5, zorder=9)  # fmt: skip
    point = approach.point
    nearest = closest_point_on_box(point, np.array([x, y]), heading, length, width)
    ax.plot([point[0], nearest[0]], [point[1], nearest[1]], color=color, lw=1.5, zorder=9.5)
    ax.plot(*point, marker="o", ms=7, color=color, mec="black", zorder=9.6)
    ax.annotate(approach.text, point, xytext=approach.text_offset, textcoords="offset points",
                fontsize=8, color=color, zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85))  # fmt: skip
    return approach.legend_handle()


def draw_time_ticks(
    ax: Axes, xy: np.ndarray, valid: np.ndarray, dt: float, color: str, every_s: float = 1.0
) -> None:
    """Dots and "<t> s" labels every `every_s` seconds along a future [T, 2], where index i is
    (i + 1) * dt after t0."""
    step = int(round(every_s / dt))
    for index in range(step - 1, len(xy), step):
        if valid[index]:
            ax.plot(*xy[index], marker="o", ms=3.5, color=color, zorder=8.7)
            ax.annotate(f"{(index + 1) * dt:g}s", xy[index], xytext=(3, 3),
                        textcoords="offset points", fontsize=6.5, color=color)  # fmt: skip


def render_precision_recall(
    curves: dict[str, dict[str, np.ndarray]],
    operating_points: dict[str, tuple[float | None, float | None]],
    base_rate: float,
    out: Path,
    title: str,
) -> None:
    """One step curve per model (`prs.risk.evaluation.precision_recall_curve`), each model's
    (recall, precision) at the score threshold as a dot, and the base rate: the precision of
    flagging samples at random."""
    fig = Figure(figsize=(6.5, 5))
    ax = fig.subplots()
    for name, curve in curves.items():
        recall, precision = curve["recall"], curve["precision"]
        (line,) = ax.step(np.r_[0.0, recall], np.r_[precision[0], precision], where="pre",
                          lw=1.8, label=name)  # fmt: skip
        point_precision, point_recall = operating_points[name]
        if point_precision is not None and point_recall is not None:
            ax.plot(point_recall, point_precision, "o", color=line.get_color(), ms=7,
                    mec="black", zorder=5)  # fmt: skip
    ax.axhline(base_rate, color="gray", ls="--", lw=1.0, label=f"base rate {base_rate:.2%}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall (share of ground-truth conflicts flagged)")
    ax.set_ylabel("precision (share of flagged samples that conflict)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
    ax.plot([], [], "o", color="white", mec="black", label="dot: flagged at the threshold")
    ax.legend(fontsize=8, loc="lower left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
