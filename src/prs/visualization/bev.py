"""Bird's-eye-view drawing primitives (design doc §23).

Colors here are visualization defaults only; core logic never depends on them.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

from prs.data.map_types import MapType
from prs.data.types import ObjectType

AGENT_COLORS = {
    ObjectType.VEHICLE: "#4C72B0",
    ObjectType.PEDESTRIAN: "#DD8452",
    ObjectType.CYCLIST: "#55A868",
    ObjectType.OTHER: "#8C8C8C",
    ObjectType.UNSET: "#8C8C8C",
}
TARGET_COLOR = "#C44E52"
SDC_COLOR = "#8172B3"
CONFLICT_COLOR = "#E7298A"  # predicted modes that conflict with the ego

LINE_STYLES = {
    "lane": dict(color="#B5B5B5", lw=0.7, ls=(0, (4, 3)), zorder=1),
    "road_line": dict(color="#9A9A9A", lw=0.9, zorder=1),
    "road_edge": dict(color="#3F3F3F", lw=1.5, zorder=2),
}
POLYGON_STYLES = {
    "crosswalk": dict(facecolor="#F3E7B3", edgecolor="#C9B458", lw=0.8, zorder=0),
    "speed_bump": dict(facecolor="none", edgecolor="#8C6D46", hatch="////", lw=0.8, zorder=0),
    "driveway": dict(facecolor="#E6E6E6", edgecolor="#BDBDBD", lw=0.6, zorder=0),
}
MUTED_COLOR = "#DCDCDC"  # map geometry that is not a model input

_LANE = dict(color="#9E9E9E", lw=0.9, ls=(0, (4, 3)), zorder=2)
_ROAD_LINE = dict(color="#7F7F7F", lw=1.0, zorder=2)
_ROAD_EDGE = dict(color="#2F2F2F", lw=1.6, zorder=2)
# Map chunks as the model sees them (polylines; polygons are closed outlines).
CHUNK_STYLES = {
    MapType.LANE_UNDEFINED: _LANE,
    MapType.LANE_FREEWAY: _LANE,
    MapType.LANE_SURFACE_STREET: _LANE,
    MapType.LANE_BIKE: dict(_LANE, color="#3C8D53"),
    MapType.ROAD_LINE_BROKEN: dict(_ROAD_LINE, ls=(0, (6, 4))),
    MapType.ROAD_LINE_SOLID: _ROAD_LINE,
    MapType.ROAD_LINE_UNKNOWN: _ROAD_LINE,
    MapType.ROAD_EDGE_BOUNDARY: _ROAD_EDGE,
    MapType.ROAD_EDGE_MEDIAN: _ROAD_EDGE,
    MapType.ROAD_EDGE_UNKNOWN: _ROAD_EDGE,
    MapType.CROSSWALK: dict(color="#C9A227", lw=2.0, zorder=2),
    MapType.SPEED_BUMP: dict(color="#8C6D46", lw=2.0, zorder=2),
    MapType.DRIVEWAY: dict(color="#A0A0A0", lw=1.2, ls=":", zorder=2),
}


def box_corners(center: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    """Corners [4, 2] of an oriented rectangle."""
    forward = np.array([np.cos(heading), np.sin(heading)]) * (length / 2)
    left = np.array([-np.sin(heading), np.cos(heading)]) * (width / 2)
    center = np.asarray(center, dtype=np.float64)
    return np.stack(
        [center + forward + left, center - forward + left, center - forward - left,
         center + forward - left]
    )  # fmt: skip


def draw_box(
    ax: Axes, center: np.ndarray, heading: float, length: float, width: float, **kwargs
) -> None:
    ax.add_patch(Polygon(box_corners(center, heading, length, width), closed=True, **kwargs))


def draw_trajectory(ax: Axes, xy: np.ndarray, valid: np.ndarray, **kwargs) -> None:
    """Plot a trajectory, breaking the line at invalid frames instead of bridging them."""
    points = np.where(np.asarray(valid)[:, None], xy, np.nan)
    ax.plot(points[:, 0], points[:, 1], **kwargs)


def draw_map_feature(ax: Axes, kind: str, points: np.ndarray, muted: bool = False) -> None:
    """Draw one raw map feature whose points are already in plot coordinates. `muted` draws
    it as light-gray background."""
    if kind == "stop_sign":
        color = MUTED_COLOR if muted else "#C0392B"
        ax.plot(points[:, 0], points[:, 1], marker="8", color=color, ms=8, ls="none", zorder=3)
    elif muted:
        if kind in POLYGON_STYLES and len(points) >= 3:
            ax.add_patch(Polygon(points, closed=True, facecolor="none", edgecolor=MUTED_COLOR,
                                 lw=0.7, zorder=0))  # fmt: skip
        else:
            ax.plot(points[:, 0], points[:, 1], color=MUTED_COLOR, lw=0.7, zorder=0)
    elif kind in POLYGON_STYLES:
        if len(points) >= 3:
            ax.add_patch(Polygon(points, closed=True, **POLYGON_STYLES[kind]))
    else:
        ax.plot(points[:, 0], points[:, 1], **LINE_STYLES[kind])


def draw_map_chunk(ax: Axes, map_type: int, points: np.ndarray) -> None:
    """Draw one model-input map chunk (valid points only)."""
    if MapType(map_type) == MapType.STOP_SIGN or len(points) == 1:
        ax.plot(points[:, 0], points[:, 1], marker="8", color="#C0392B", ms=8, ls="none", zorder=3)
    else:
        ax.plot(points[:, 0], points[:, 1], **CHUNK_STYLES[MapType(map_type)])
