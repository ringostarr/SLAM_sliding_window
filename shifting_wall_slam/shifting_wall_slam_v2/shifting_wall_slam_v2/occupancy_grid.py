"""Log-odds occupancy grid with ray-cast updates and time decay (v2).

v2 adds occupancy decay: each scan every cell's log-odds is pulled toward 0
(unknown) by `decay`. Cells that are still observed get re-reinforced and stay
put; cells no longer seen -- e.g. a moved panel's old footprint that beam
divergence can't fully clear at range -- fade to unknown on their own. This is
the lifelong-mapping fix for stale dynamic obstacles that pure ray-casting
leaves behind. Cost: distant, no-longer-observed structure also slowly fades.
"""
from __future__ import annotations

import math
import numpy as np


class OccupancyGridMap:
    def __init__(self, resolution=0.1, xmin=-1.0, xmax=21.0, ymin=-2.0, ymax=2.0,
                 l_occ=0.85, l_free=-0.9, l_min=-2.0, l_max=2.0, decay=0.95):
        # decay in (0, 1]: per-scan multiplier toward unknown. 1.0 = no decay
        # (v1 behaviour); 0.95 fades an unobserved saturated cell to ~unknown
        # in ~15 scans (~2 s at 7 Hz).
        self.res = resolution
        self.xmin, self.ymin = xmin, ymin
        self.nx = int(math.ceil((xmax - xmin) / resolution))
        self.ny = int(math.ceil((ymax - ymin) / resolution))
        self.L = np.zeros((self.nx, self.ny))
        self.l_occ, self.l_free = l_occ, l_free
        self.l_min, self.l_max = l_min, l_max
        self.decay = decay
        self.res = resolution
        self.xmin, self.ymin = xmin, ymin
        self.nx = int(math.ceil((xmax - xmin) / resolution))
        self.ny = int(math.ceil((ymax - ymin) / resolution))
        self.L = np.zeros((self.nx, self.ny))
        self.l_occ, self.l_free = l_occ, l_free
        self.l_min, self.l_max = l_min, l_max

    def cell(self, x, y):
        return int((x - self.xmin) / self.res), int((y - self.ymin) / self.res)

    def in_bounds(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    def _bump(self, i, j, dl):
        if self.in_bounds(i, j):
            self.L[i, j] = min(self.l_max, max(self.l_min, self.L[i, j] + dl))

    @staticmethod
    def _bresenham(i0, j0, i1, j1):
        cells = []
        di, dj = abs(i1 - i0), abs(j1 - j0)
        si = 1 if i0 < i1 else -1
        sj = 1 if j0 < j1 else -1
        err = di - dj
        i, j = i0, j0
        while True:
            cells.append((i, j))
            if i == i1 and j == j1:
                break
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj
        return cells

    def integrate_scan(self, pose, angles, ranges, hit, max_range):
        if self.decay < 1.0:
            self.L *= self.decay          # relax unobserved cells toward unknown
        px, py, th = pose
        i0, j0 = self.cell(px, py)
        for k in range(len(angles)):
            r = ranges[k] if hit[k] else max_range
            a = th + angles[k]
            i1, j1 = self.cell(px + r * math.cos(a), py + r * math.sin(a))
            for ci, cj in self._bresenham(i0, j0, i1, j1)[:-1]:
                self._bump(ci, cj, self.l_free)
            self._bump(i1, j1, self.l_occ if hit[k] else self.l_free)

    def prob(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.L))

    def occupancy_int8(self):
        p = self.prob()
        out = np.full(self.L.shape, -1, dtype=np.int8)
        known = np.abs(self.L) > 1e-3
        out[known] = (p[known] * 100).astype(np.int8)
        return out
