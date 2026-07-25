"""
2D LiDAR sensor: raycasting against the world's line segments.

A 16-channel 3D LiDAR spinning in a corridor of flat vertical walls provides,
for planar localization, a horizontal ring of ranges. We model exactly that
ring: N beams spread over a field of view, each returning the distance to the
nearest world segment (walls + panel). The 3D channels add nothing to the
planar observability problem against vertical flat walls, so 2D is a faithful
and testable reduction.

Output of a scan:
    angles  : (N,) beam bearings in the BODY frame (rad)
    ranges  : (N,) measured range per beam (m); == max_range where no hit
    hit     : (N,) bool, True if a real surface was struck within max_range

The vectorized ray-segment intersection below solves, for every (beam, segment)
pair simultaneously:
        o + t*d = a + u*(b - a),   t >= 0,  0 <= u <= 1
where o is the sensor origin, d the beam direction, [a,b] the segment.
"""

from __future__ import annotations

import numpy as np


class Lidar2D:
    def __init__(
        self,
        n_beams: int = 360,
        fov: float = 2.0 * np.pi,   # full 360-degree ring (VLP-16-like horizontal sweep)
        max_range: float = 30.0,
        range_noise_std: float = 0.02,
        dropout_prob: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.n_beams = n_beams
        self.max_range = max_range
        self.range_noise_std = range_noise_std
        self.dropout_prob = dropout_prob
        self.rng = rng if rng is not None else np.random.default_rng(0)

        # Body-frame beam bearings. For a full ring we exclude the duplicate
        # endpoint so 0 and 2*pi aren't both present.
        if abs(fov - 2.0 * np.pi) < 1e-9:
            self.angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
        else:
            self.angles = np.linspace(-fov / 2.0, fov / 2.0, n_beams)

    # ------------------------------------------------------------------ #
    def scan(self, pose: np.ndarray, segments: np.ndarray, add_noise: bool = True):
        """
        Cast all beams from `pose` against `segments`.

        pose     : (3,) [px, py, theta]
        segments : (M, 4) rows [x1, y1, x2, y2]
        returns  : (angles, ranges, hit)
        """
        px, py, th = pose
        o = np.array([px, py])

        # Beam directions in the WORLD frame: (N, 2)
        world_ang = self.angles + th
        d = np.stack([np.cos(world_ang), np.sin(world_ang)], axis=1)  # (N,2)

        a = segments[:, 0:2]                 # (M,2) segment starts
        e = segments[:, 2:4] - segments[:, 0:2]  # (M,2) segment edge vectors

        N, M = self.n_beams, segments.shape[0]

        # Broadcast to (N, M): solve the 2x2 system per (beam, segment).
        dx = d[:, 0][:, None]                # (N,1)
        dy = d[:, 1][:, None]
        ex = e[:, 0][None, :]                # (1,M)
        ey = e[:, 1][None, :]
        rx = (a[:, 0][None, :] - px)         # (1->N,M) rhs = a - o
        ry = (a[:, 1][None, :] - py)

        det = (-dx) * ey + ex * dy           # (N,M)
        det_safe = np.where(np.abs(det) < 1e-12, np.nan, det)

        t = (-rx * ey + ex * ry) / det_safe  # distance along the beam
        u = (dx * ry - dy * rx) / det_safe   # position along the segment

        valid = (t > 1e-6) & (u >= 0.0) & (u <= 1.0) & np.isfinite(t)
        t = np.where(valid, t, np.inf)

        ranges = t.min(axis=1)               # nearest hit per beam
        hit = np.isfinite(ranges) & (ranges <= self.max_range)
        ranges = np.where(hit, ranges, self.max_range)

        if add_noise:
            if self.range_noise_std > 0:
                ranges = ranges + self.rng.normal(0.0, self.range_noise_std, size=N)
                ranges = np.clip(ranges, 0.0, self.max_range)
            if self.dropout_prob > 0:
                dropped = self.rng.random(N) < self.dropout_prob
                hit = hit & ~dropped
                ranges = np.where(dropped, self.max_range, ranges)

        return self.angles.copy(), ranges, hit

    # ------------------------------------------------------------------ #
    @staticmethod
    def to_points_body(angles: np.ndarray, ranges: np.ndarray, hit: np.ndarray) -> np.ndarray:
        """Return (K,2) hit points in the BODY frame (only real hits)."""
        a = angles[hit]
        r = ranges[hit]
        return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)

    @staticmethod
    def to_points_world(pose: np.ndarray, angles: np.ndarray,
                        ranges: np.ndarray, hit: np.ndarray) -> np.ndarray:
        """Return (K,2) hit points in the WORLD frame (only real hits)."""
        px, py, th = pose
        a = angles[hit] + th
        r = ranges[hit]
        return np.stack([px + r * np.cos(a), py + r * np.sin(a)], axis=1)


if __name__ == "__main__":
    from sim.world import CorridorWorld
    w = CorridorWorld()
    lidar = Lidar2D(n_beams=360, range_noise_std=0.0)
    pose = np.array([8.0, 0.0, 0.0])
    for t in (0.0, 15.0):
        ang, rng, hit = lidar.scan(pose, w.segments(t), add_noise=False)
        print(f"t={t:4.1f}s  hits={hit.sum():3d}/{len(hit)}  "
              f"min_range={rng[hit].min():.2f}  max_hit_range={rng[hit].max():.2f}")
