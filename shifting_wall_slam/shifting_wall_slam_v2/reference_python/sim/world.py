"""
World geometry for the Shifting Wall & Slipping Base challenge.

The world is a 20 m straight corridor with flat, featureless walls running
along the +X axis. This axis-aligned, feature-poor geometry is what creates
the *structural axial degeneracy*: a LiDAR scan can pin down the robot's
lateral (Y) position and heading from the two long walls, but nothing in the
geometry constrains motion ALONG the corridor (X). Detecting that
unobservability is the core of Task 2/3.

Everything is represented as 2D line segments. A 16-channel 3D LiDAR looking
at flat vertical walls collapses, for the purpose of planar localization, to a
2D range scan against these segments -- so a 2D segment world captures the
exact observability structure of the real problem while staying testable
without a GPU or a physics engine.

Segment convention: each segment is a row [x1, y1, x2, y2].
"""

from __future__ import annotations

import numpy as np


class CorridorWorld:
    def __init__(
        self,
        length: float = 20.0,
        width: float = 3.0,
        panel_center_x: float = 10.0,
        panel_size_x: float = 1.0,
        panel_size_y: float = 0.3,
        panel_y_initial: float = 0.6,
        panel_shift: float = -1.5,
        shift_time: float = 15.0,
        shift_period: float | None = None,
    ):
        """
        Parameters
        ----------
        length, width : corridor dimensions (m). Walls sit at y = +/- width/2.
        panel_*       : geometry of the shifting panel (a thin box).
        panel_y_initial : starting lateral position of the panel center (m).
        panel_shift   : how far the panel jumps sideways in Y (m). Negative
                        moves it toward -Y. Default -1.5 as specified.
        shift_time    : first shift happens at t >= shift_time (s).
        shift_period  : if set, the panel keeps shifting every `shift_period`
                        seconds (interpreting "at T=15s intervals" as periodic).
                        If None, it shifts exactly once at shift_time.

        NOTE on ambiguity: the brief says the panel "shifts sideways by 1.5 m
        precisely at T=15 seconds intervals". That can mean (a) once at t=15s,
        or (b) every 15s. We default to (a) and expose shift_period for (b).
        """
        self.length = length
        self.width = width
        self.half_w = width / 2.0

        self.panel_center_x = panel_center_x
        self.panel_size_x = panel_size_x
        self.panel_size_y = panel_size_y
        self.panel_y_initial = panel_y_initial
        self.panel_shift = panel_shift
        self.shift_time = shift_time
        self.shift_period = shift_period

        # Static walls, precomputed once (they never move).
        L, hw = self.length, self.half_w
        # Capped corridor (v3): two long walls + two end caps.
        self._static = np.array(
            [
                [0.0, +hw, L, +hw],    # left wall  (top)
                [0.0, -hw, L, -hw],    # right wall (bottom)
                [0.0, -hw, 0.0, +hw],  # near end cap
                [L, -hw, L, +hw],      # far end cap
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------ #
    # Panel state
    # ------------------------------------------------------------------ #
    def n_shifts(self, t: float) -> int:
        """How many times the panel has shifted by time t."""
        if t < self.shift_time:
            return 0
        if self.shift_period is None:
            return 1
        return 1 + int((t - self.shift_time) // self.shift_period)

    def panel_y(self, t: float) -> float:
        """Lateral center of the panel at time t."""
        return self.panel_y_initial + self.panel_shift * self.n_shifts(t)

    def panel_segments(self, t: float) -> np.ndarray:
        """The 4 segments of the panel box at time t."""
        cx = self.panel_center_x
        cy = self.panel_y(t)
        dx = self.panel_size_x / 2.0
        dy = self.panel_size_y / 2.0
        x0, x1 = cx - dx, cx + dx
        y0, y1 = cy - dy, cy + dy
        return np.array(
            [
                [x0, y0, x1, y0],  # bottom edge
                [x1, y0, x1, y1],  # right edge
                [x1, y1, x0, y1],  # top edge
                [x0, y1, x0, y0],  # left edge
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------ #
    # Full world
    # ------------------------------------------------------------------ #
    def segments(self, t: float) -> np.ndarray:
        """All segments (static walls + panel) at time t, shape (N, 4)."""
        return np.vstack([self._static, self.panel_segments(t)])


if __name__ == "__main__":
    w = CorridorWorld()
    for t in (0.0, 14.9, 15.0, 30.0):
        print(f"t={t:5.1f}s  n_shifts={w.n_shifts(t)}  panel_y={w.panel_y(t):+.2f}  "
              f"segments={w.segments(t).shape[0]}")
