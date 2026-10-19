"""Target-centric BEV figure of one sample, optionally with predicted futures (design doc §23).

Everything the model sees is drawn from the ProcessedSample arrays, so the figure checks the
sample builder itself:

- complete samples (model view): the selected map chunks in color over the raw map in light
  gray, the selected neighbors, and agents dropped by the neighbor cap as dashed outlines;
- target-only samples: the raw map and every agent within the neighbor radius.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from prs.data.map_types import MapType
from prs.data.neighbor_selection import neighbor_candidates
from prs.data.types import ObjectType, ProcessedSample, RawScenario
from prs.data.windows import Window, make_window
from prs.utils.config import DataConfig
from prs.visualization.bev import (
    AGENT_COLORS,
    CHUNK_STYLES,
    CONFLICT_COLOR,
    MUTED_COLOR,
    POLYGON_STYLES,
    SDC_COLOR,
    TARGET_COLOR,
    draw_box,
    draw_map_chunk,
    draw_map_feature,
    draw_trajectory,
)

PREDICTION_COLOR = "#1B1B1B"


def render_sample(
    scenario: RawScenario,
    sample: ProcessedSample,
    cfg: DataConfig,
    out: Path,
    predictions: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    subtitle: str | None = None,
    highlighted_modes: Sequence[int] = (),
    overlay: Callable[[Axes], list] | None = None,
) -> None:
    """Save a BEV figure. `predictions` is [K, T, 2] in the target frame; `highlighted_modes`
    are drawn in the conflict color. `overlay(ax)` draws extra layers and returns their legend
    handles."""
    meta = sample.meta
    window = make_window(cfg, meta.t0)
    model_view = sample.neighbors is not None and sample.map_polylines is not None
    fig = Figure(figsize=(9, 9))
    ax = fig.subplots()

    if model_view:
        _draw_model_context(ax, scenario, sample, cfg, window)
    else:
        _draw_raw_context(ax, scenario, sample, cfg, window)

    # Ego (SDC): history solid, future dashed, footprint at t0.
    draw_trajectory(ax, sample.ego_history[:, :2], sample.ego_history_valid,
                    color=SDC_COLOR, lw=2.0, zorder=6)  # fmt: skip
    draw_trajectory(ax, sample.ego_future[:, :2], sample.ego_future_valid,
                    color=SDC_COLOR, lw=2.0, ls="--", zorder=6)  # fmt: skip
    if sample.ego_history_valid[-1]:
        x, y, _, _, s, c, length, width, _ = sample.ego_history[-1]
        draw_box(ax, (x, y), np.arctan2(s, c), length, width,
                 facecolor=SDC_COLOR, edgecolor="black", lw=0.8, alpha=0.85, zorder=7)  # fmt: skip

    # Target: history with frame dots, ground-truth future dashed, footprint at t0.
    draw_trajectory(ax, sample.target_history[:, :2], sample.target_history_valid,
                    color=TARGET_COLOR, lw=2.2, marker="o", ms=2.5, zorder=8)  # fmt: skip
    draw_trajectory(ax, sample.target_future, sample.target_future_valid,
                    color=TARGET_COLOR, lw=2.0, ls="--", zorder=8)  # fmt: skip
    _, _, _, _, s, c, length, width, _ = sample.target_history[-1]
    draw_box(ax, (0.0, 0.0), np.arctan2(s, c), length, width,
             facecolor=TARGET_COLOR, edgecolor="black", lw=0.8, zorder=9)  # fmt: skip

    if predictions is not None:
        draw_predictions(
            ax, sample.target_history[-1, :2], predictions, probabilities, highlighted_modes
        )
    extra_handles = overlay(ax) if overlay is not None else []

    for direction, label in (((1, 0), "+x forward"), ((0, 1), "+y left")):
        tip = np.array(direction) * 3.0
        ax.annotate("", xy=tip, xytext=(0, 0), zorder=10,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.4))  # fmt: skip
        ax.text(*(tip * 1.12), label, fontsize=8, zorder=10)

    shown = [
        sample.target_history[sample.target_history_valid, :2],
        sample.target_future[sample.target_future_valid],
    ]
    if predictions is not None:
        shown.append(predictions.reshape(-1, 2))
    reach = np.abs(np.concatenate(shown)).max()
    radius = float(np.clip(reach + 10.0, 20.0, cfg.map_radius_m))
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x [m]  (target forward)")
    ax.set_ylabel("y [m]  (target left)")
    lines = [
        f"{ObjectType(meta.target_type).name.lower()} track {meta.target_track_id} · "
        f"scenario {meta.scenario_id}",
        f"t0 = frame {window.t0} · speed {meta.current_speed_mps:.2f} m/s · "
        f"frame heading: {meta.frame_heading_source_used} · SDC at {meta.sdc_distance_m:.1f} m",
    ]
    if model_view:
        num_neighbors = int(sample.neighbor_valid.any(axis=1).sum())
        num_chunks = int(sample.map_valid.any(axis=1).sum())
        lines.append(
            f"model input: {num_neighbors} of {meta.neighbors_within_radius} agents · "
            f"{num_chunks} of {meta.map_chunks_within_radius} map chunks "
            f"(within {cfg.neighbor_radius_m:.0f} / {cfg.map_radius_m:.0f} m)"
        )
    if subtitle:
        lines.append(subtitle)
    ax.set_title("\n".join(lines), fontsize=10)
    handles = _legend(predictions is not None, model_view, bool(highlighted_modes))
    ax.legend(handles=handles + extra_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=8)  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")


def _draw_raw_context(
    ax: Axes, scenario: RawScenario, sample: ProcessedSample, cfg: DataConfig, window: Window
) -> None:
    """Raw map and every agent valid at t0 within the neighbor radius (target-only samples)."""
    frame, t0, hist = sample.frame, window.t0, window.history_slice
    target_index = scenario.track_index(sample.meta.target_track_id)
    draw_sample_map(ax, scenario, sample)
    for i in range(scenario.num_tracks):
        if i in (target_index, scenario.sdc_index) or not scenario.valid[i, t0]:
            continue
        position = frame.points_to_local(scenario.xy[i, t0])
        if np.hypot(*position) > cfg.neighbor_radius_m:
            continue
        color = AGENT_COLORS[ObjectType(scenario.track_types[i])]
        draw_trajectory(
            ax, frame.points_to_local(scenario.xy[i, hist]), scenario.valid[i, hist],
            color=color, lw=1.0, alpha=0.8, zorder=4,
        )  # fmt: skip
        length, width = scenario.size[i, t0]
        heading = float(frame.headings_to_local(scenario.heading[i, t0]))
        draw_box(ax, position, heading, length, width, facecolor=color, alpha=0.55, zorder=5)


def _draw_model_context(
    ax: Axes, scenario: RawScenario, sample: ProcessedSample, cfg: DataConfig, window: Window
) -> None:
    """What the model sees: selected chunks over the muted raw map, selected neighbors, and
    agents dropped by the neighbor cap."""
    frame, t0 = sample.frame, window.t0
    draw_sample_map(ax, scenario, sample)
    target_index = scenario.track_index(sample.meta.target_track_id)
    dropped = neighbor_candidates(scenario, target_index, cfg, t0)[cfg.max_neighbors :]
    for i in dropped:
        length, width = scenario.size[i, t0]
        heading = float(frame.headings_to_local(scenario.heading[i, t0]))
        draw_box(ax, frame.points_to_local(scenario.xy[i, t0]), heading, length, width,
                 facecolor="none", edgecolor="#7F7F7F", ls="--", lw=0.8, zorder=4)  # fmt: skip

    for slot in np.flatnonzero(sample.neighbor_valid.any(axis=1) & ~sample.neighbor_is_sdc):
        history, valid = sample.neighbors[slot], sample.neighbor_valid[slot]
        color = AGENT_COLORS[ObjectType(sample.neighbor_types[slot])]
        draw_trajectory(ax, history[:, :2], valid, color=color, lw=1.0, alpha=0.8, zorder=4)
        if valid[-1]:
            x, y, _, _, s, c, length, width, _ = history[-1]
            draw_box(ax, (x, y), np.arctan2(s, c), length, width, facecolor=color, alpha=0.55,
                     zorder=5)  # fmt: skip


def draw_sample_map(ax: Axes, scenario: RawScenario, sample: ProcessedSample) -> None:
    """The map in the target frame: for complete samples, the model's map chunks over the
    muted raw map; for target-only samples, the raw map."""
    frame = sample.frame
    model_view = sample.map_polylines is not None
    for feature in scenario.map_features:
        draw_map_feature(ax, feature.kind, frame.points_to_local(feature.points), muted=model_view)
    if model_view:
        for c in np.flatnonzero(sample.map_valid.any(axis=1)):
            points = sample.map_polylines[c, sample.map_valid[c], :2]
            draw_map_chunk(ax, sample.map_types[c], points)


def draw_predictions(
    ax: Axes,
    start: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    highlighted: Sequence[int] = (),
) -> None:
    """Predicted modes; more likely modes are more opaque and drawn on top."""
    num_modes = len(predictions)
    probs = np.full(num_modes, 1.0 / num_modes) if probabilities is None else probabilities
    alphas = mode_alphas(probs)
    for k in np.argsort(probs):
        alpha = alphas[k]
        color = CONFLICT_COLOR if k in highlighted else PREDICTION_COLOR
        path = np.vstack([start, predictions[k]])
        ax.plot(path[:, 0], path[:, 1], color=color, lw=2.0, ls=":", alpha=alpha, zorder=8.5)
        ax.plot(*predictions[k, -1], marker="x", color=color, ms=7, mew=2, alpha=alpha,
                zorder=8.5)  # fmt: skip
        if num_modes > 1:
            ax.text(*predictions[k, -1], f" {probs[k]:.2f}", fontsize=7, zorder=8.5)


def mode_alphas(probabilities: np.ndarray) -> np.ndarray:
    """Line opacity per mode, rising with its probability: the most likely mode is opaque and
    the least likely stays visible."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return 0.3 + 0.7 * probabilities / probabilities.max()


def _legend(with_predictions: bool, model_view: bool, with_conflicts: bool = False) -> list:
    handles = [
        Line2D([], [], color=TARGET_COLOR, lw=2.2, marker="o", ms=3, label="target history"),
        Line2D([], [], color=TARGET_COLOR, lw=2.0, ls="--", label="target future (GT)"),
    ]
    if with_predictions:
        handles.append(Line2D([], [], color=PREDICTION_COLOR, lw=2.0, ls=":", marker="x",
                              label="prediction"))  # fmt: skip
    if with_conflicts:
        handles.append(Line2D([], [], color=CONFLICT_COLOR, lw=2.0, ls=":", marker="x",
                              label="prediction that conflicts"))  # fmt: skip
    handles += [
        Line2D([], [], color=SDC_COLOR, lw=2.0, label="SDC history"),
        Line2D([], [], color=SDC_COLOR, lw=2.0, ls="--", label="SDC future"),
        *(Patch(color=AGENT_COLORS[t], alpha=0.55, label=t.name.lower())
          for t in (ObjectType.VEHICLE, ObjectType.PEDESTRIAN, ObjectType.CYCLIST)),
    ]  # fmt: skip
    if model_view:
        crosswalk = CHUNK_STYLES[MapType.CROSSWALK]
        handles += [
            Patch(facecolor="none", edgecolor="#7F7F7F", ls="--", label="agent beyond cap"),
            Line2D([], [], color=crosswalk["color"], lw=crosswalk["lw"], label="crosswalk chunk"),
            Line2D([], [], color=CHUNK_STYLES[MapType.ROAD_EDGE_BOUNDARY]["color"], lw=1.6,
                   label="road edge chunk"),
            Line2D([], [], color=MUTED_COLOR, lw=1.5, label="map not in model input"),
        ]  # fmt: skip
    else:
        handles += [
            Patch(**{k: v for k, v in POLYGON_STYLES[kind].items() if k != "zorder"},
                  label=kind.replace("_", " "))
            for kind in ("crosswalk", "speed_bump", "driveway")
        ]  # fmt: skip
    return handles
