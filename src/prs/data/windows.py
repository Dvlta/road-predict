"""History / future frame windows around t0 (design doc §5.1)."""

from __future__ import annotations

from dataclasses import dataclass

from prs.utils.config import DataConfig


@dataclass(frozen=True)
class Window:
    t0: int
    history_steps: int  # past frames before t0, excluding t0
    future_steps: int  # T

    @property
    def history_len(self) -> int:
        """H: includes t0."""
        return self.history_steps + 1

    @property
    def history_slice(self) -> slice:
        """Frames t0 - history_steps .. t0 (inclusive): the only frames a model may see."""
        return slice(self.t0 - self.history_steps, self.t0 + 1)

    @property
    def future_slice(self) -> slice:
        """Frames t0 + 1 .. t0 + T: labels only."""
        return slice(self.t0 + 1, self.t0 + 1 + self.future_steps)

    @property
    def final_future_index(self) -> int:
        return self.t0 + self.future_steps

    def check_fits(self, num_frames: int) -> None:
        if self.t0 - self.history_steps < 0 or self.final_future_index > num_frames - 1:
            raise ValueError(
                f"window t0={self.t0} (history {self.history_steps}, future {self.future_steps}) "
                f"does not fit in {num_frames} frames"
            )


def make_window(cfg: DataConfig, t0: int | None = None) -> Window:
    window = Window(
        t0=cfg.current_index if t0 is None else t0,
        history_steps=cfg.history_steps,
        future_steps=cfg.future_steps,
    )
    window.check_fits(cfg.scene_num_frames)
    return window
