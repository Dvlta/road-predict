"""Animated BEV figure of one sample (design doc §23): history -> current frame -> predicted
future, as a GIF.

Frames follow the data at 10 Hz, in real time. Every agent moves along its logged track. From
t0 on, the SDC's logged future is drawn as its plan, which the conflict analysis treats as
known, while the target's true future is revealed frame by frame. The forecast appears at t0
(more likely modes more opaque) and a dot then runs along each mode, so the forecast and the
truth can be compared at the same moment. Each closest approach is drawn when its time comes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from PIL import Image

from prs.data.types import ObjectType, ProcessedSample, RawScenario
from prs.data.windows import make_window
from prs.utils.config import DataConfig
from prs.visualization.bev import (
    AGENT_COLORS,
    CONFLICT_COLOR,
    SDC_COLOR,
    TARGET_COLOR,
    box_corners,
    draw_trajectory,
)
from prs.visualization.conflict import ClosestApproach, draw_closest_approach
from prs.visualization.predictions import (
    PREDICTION_COLOR,
    draw_predictions,
    draw_sample_map,
    mode_alphas,
)

FRAME_MS = 100  # one 0.1 s data step per frame: real time
T0_HOLD_MS = 1500  # pause when the forecast appears
END_HOLD_MS = 2500
TAIL_FRAMES = 10  # other agents trail their last second
PALETTE_COLORS = 256
PALETTE_SAMPLE_EVERY = 8  # frames read when choosing the palette


def render_animation(
    scenario: RawScenario,
    sample: ProcessedSample,
    cfg: DataConfig,
    out: Path,
    predictions: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    title: str = "",
    highlighted_modes: Sequence[int] = (),
    approaches: Sequence[ClosestApproach] = (),
    dpi: int = 80,
) -> int:
    """Save a GIF and return its number of frames. `predictions` [K, T, 2], `probabilities`
    and `highlighted_modes` are as for `render_sample`."""
    meta = sample.meta
    window = make_window(cfg, meta.t0)
    now = window.history_steps  # t0's position among the animated frames
    frames = np.arange(window.t0 - now, window.final_future_index + 1)
    frame = sample.frame
    xy = frame.points_to_local(scenario.xy[:, frames])
    heading = frame.headings_to_local(scenario.heading[:, frames])
    size, valid = scenario.size[:, frames], scenario.valid[:, frames]
    target, sdc = scenario.track_index(meta.target_track_id), scenario.sdc_index

    shown = [
        sample.target_history[sample.target_history_valid, :2],
        sample.target_future[sample.target_future_valid],
        *(np.asarray(a.point)[None] for a in approaches),
    ]
    if predictions is not None:
        shown.append(predictions.reshape(-1, 2))
    radius = float(np.clip(np.abs(np.concatenate(shown)).max() + 8.0, 15.0, cfg.map_radius_m))

    fig = Figure(figsize=(6.4, 7.6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.2)
    draw_sample_map(ax, scenario, sample)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x [m]  (target forward at t0)", fontsize=7)
    ax.set_title(title, fontsize=8)

    near = valid & (np.abs(xy) <= radius + 5.0).all(axis=-1)
    agents = [a for a in np.flatnonzero(near.any(axis=1)) if a not in (target, sdc)] + [sdc, target]
    boxes, tails = {}, {}
    for a in agents:
        if a == target:
            style = dict(facecolor=TARGET_COLOR, edgecolor="black", lw=0.8, zorder=9)
        elif a == sdc:
            style = dict(facecolor=SDC_COLOR, edgecolor="black", lw=0.8, alpha=0.85, zorder=7)
        else:
            color = AGENT_COLORS[ObjectType(scenario.track_types[a])]
            style = dict(facecolor=color, alpha=0.6, zorder=5)
            (tails[a],) = ax.plot([], [], color=color, lw=1.0, alpha=0.6, zorder=4)
        boxes[a] = ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, visible=False, **style))
    (target_past,) = ax.plot([], [], color=TARGET_COLOR, lw=2.2, marker="o", ms=2.5, zorder=8)
    (target_future,) = ax.plot([], [], color=TARGET_COLOR, lw=2.0, ls="--", zorder=8)
    (sdc_trail,) = ax.plot([], [], color=SDC_COLOR, lw=2.0, zorder=6)
    clock = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=11, zorder=11,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#7F7F7F"))  # fmt: skip

    dots: list[Line2D] = []
    if predictions is not None:
        num_modes = len(predictions)
        probs = np.full(num_modes, 1.0 / num_modes) if probabilities is None else probabilities
        for k, alpha in enumerate(mode_alphas(probs)):
            color = CONFLICT_COLOR if k in highlighted_modes else PREDICTION_COLOR
            dots += ax.plot([], [], "o", color=color, ms=5, alpha=alpha, zorder=8.6)
    ax.legend(handles=_legend(predictions is not None, bool(highlighted_modes), approaches),
              loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=7)  # fmt: skip

    def path(track: int, start: int, stop: int) -> np.ndarray:
        """Positions of frames start..stop (inclusive), NaN where invalid."""
        points = xy[track, start : stop + 1]
        return np.where(valid[track, start : stop + 1, None], points, np.nan)

    steps = [(j, False, FRAME_MS) for j in range(now + 1)]
    if predictions is not None:
        steps.append((now, True, T0_HOLD_MS))
    steps += [(j, True, FRAME_MS) for j in range(now + 1, len(frames))]
    steps[-1] = (*steps[-1][:2], END_HOLD_MS)
    dt, drawn = cfg.dt, set()
    images, durations = [], []
    for j, forecast, duration in steps:
        for a in agents:
            boxes[a].set_visible(bool(valid[a, j]))
            if valid[a, j]:
                boxes[a].set_xy(box_corners(xy[a, j], heading[a, j], *size[a, j]))
            if a in tails:
                tails[a].set_data(*path(a, max(0, j - TAIL_FRAMES), j).T)
        target_past.set_data(*path(target, 0, min(j, now)).T)
        target_future.set_data(*(path(target, now, j).T if j > now else ([], [])))
        sdc_trail.set_data(*path(sdc, 0, j).T)
        if forecast and j == now:
            draw_trajectory(ax, sample.ego_future[:, :2], sample.ego_future_valid,
                            color=SDC_COLOR, lw=2.0, ls="--", zorder=6)  # fmt: skip
            draw_predictions(
                ax, sample.target_history[-1, :2], predictions, probabilities, highlighted_modes
            )
        for k, dot in enumerate(dots):
            dot.set_data(*(predictions[k, j - now - 1, :, None] if j > now else ([], [])))
        for i, approach in enumerate(approaches):
            if i not in drawn and j - now - 1 >= approach.index:
                draw_closest_approach(ax, sample.ego_future, sample.ego_size, approach)
                drawn.add(i)
        if j < now:
            clock.set_text(f"history  {(j - now) * dt:+.1f} s")
        elif j == now:
            clock.set_text("t0: forecast" if forecast else "t0: now")
        else:
            clock.set_text(f"future  {(j - now) * dt:+.1f} s")
        canvas.draw()
        images.append(Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy()))
        durations.append(duration)

    quantized = _quantize(images)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(out, save_all=True, append_images=quantized[1:], duration=durations, loop=0)
    return len(images)


def _quantize(images: list[Image.Image]) -> list[Image.Image]:
    """Map every frame onto one palette, so colors do not flicker: the most frequent exact
    colors of a sample of frames. Flat fills (agents, legend patches) keep their true color,
    however small; anti-aliased edges snap to the nearest palette color."""
    sampled = images[::PALETTE_SAMPLE_EVERY] + [images[-1]]
    counts: dict[tuple[int, int, int], int] = {}
    for image in sampled:
        for count, color in image.getcolors(maxcolors=image.width * image.height):
            counts[color] = counts.get(color, 0) + count
    top = sorted(counts, key=counts.get, reverse=True)[:PALETTE_COLORS]
    palette = Image.new("P", (1, 1))
    palette.putpalette([channel for color in top for channel in color])
    return [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]


def _legend(
    with_predictions: bool, with_conflicts: bool, approaches: Sequence[ClosestApproach]
) -> list:
    handles = [
        Line2D([], [], color=TARGET_COLOR, lw=2.2, marker="o", ms=3, label="target history"),
        Line2D([], [], color=TARGET_COLOR, lw=2.0, ls="--", label="target future (GT)"),
        Line2D([], [], color=SDC_COLOR, lw=2.0, label="SDC (dashed: logged plan)"),
    ]
    if with_predictions:
        handles.append(Line2D([], [], color=PREDICTION_COLOR, lw=2.0, ls=":", marker="x",
                              label="prediction"))  # fmt: skip
    if with_conflicts:
        handles.append(Line2D([], [], color=CONFLICT_COLOR, lw=2.0, ls=":", marker="x",
                              label="prediction that conflicts"))  # fmt: skip
    handles += [a.legend_handle() for a in approaches]
    handles += [
        Patch(color=AGENT_COLORS[t], alpha=0.6, label=t.name.lower())
        for t in (ObjectType.VEHICLE, ObjectType.PEDESTRIAN, ObjectType.CYCLIST)
    ]
    return handles
