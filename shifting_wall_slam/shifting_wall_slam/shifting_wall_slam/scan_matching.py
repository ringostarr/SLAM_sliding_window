"""Point-to-line scan matching and information-matrix degeneracy analysis."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class CorridorModel:
    half_width: float = 1.5
    length: float = 20.0


def estimate_normals(points, jump=0.3):
    """Per-point 2D normals for a ring-ordered scan, oriented toward the sensor.

    Returns (normals, valid); points near a range discontinuity are invalid.
    """
    n = points.shape[0]
    normals = np.zeros((n, 2))
    valid = np.zeros(n, dtype=bool)
    if n < 3:
        return normals, valid

    prev = np.roll(points, 1, axis=0)
    nxt = np.roll(points, -1, axis=0)
    tang = nxt - prev
    tlen = np.linalg.norm(tang, axis=1)
    ok = ((tlen > 1e-6)
          & (np.linalg.norm(points - prev, axis=1) < jump)
          & (np.linalg.norm(nxt - points, axis=1) < jump))

    t = tang / np.where(tlen[:, None] < 1e-9, 1.0, tlen[:, None])
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    nrm[np.sum(nrm * -points, axis=1) < 0] *= -1.0

    normals[ok] = nrm[ok]
    valid[ok] = True
    return normals, valid


def information_matrix(points, normals, valid, sigma=0.02):
    """3x3 scan-match information in [x, y, theta] (sensor frame)."""
    P = points[valid]
    N = normals[valid]
    if P.shape[0] == 0:
        return np.zeros((3, 3))
    a = np.stack([N[:, 0], N[:, 1], N[:, 1] * P[:, 0] - N[:, 0] * P[:, 1]], axis=1)
    return (a.T @ a) / (sigma * sigma)


def degeneracy(H):
    """Eigen-analysis of H: smallest eigenvalue, its alignment with X, condition."""
    w, V = np.linalg.eigh(H)
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    e_min = V[:, 0]
    x_align = abs(e_min[0]) / (np.linalg.norm(e_min) + 1e-12)
    cond = max(w[-1], 1e-12) / max(w[0], 1e-12)
    return dict(eigvals=w, eigvecs=V, lambda_min=float(w[0]),
                x_alignment=float(x_align), condition=float(cond))


def _corridor_lines(model):
    # Closed corridor: two side walls + two end caps. The end-cap lines
    # (normals along X) supply X information when in range, so lambda_min rises
    # near the ends and collapses in the blind middle -> a genuine spike.
    hw, L = model.half_width, model.length
    return [(np.array([0.0, hw]), np.array([0.0, 1.0])),
            (np.array([0.0, -hw]), np.array([0.0, 1.0])),
            (np.array([0.0, 0.0]), np.array([1.0, 0.0])),
            (np.array([L, 0.0]), np.array([1.0, 0.0]))]


def icp_to_corridor(points_sensor, pose_guess, model,
                    iters=8, gate=0.5, sigma=0.02, damping=1e-6):
    """Point-to-line ICP against the known corridor walls.

    Returns (pose, H, n_inliers). Points beyond `gate` from every wall (the
    panel, outliers) are dropped. Damping keeps the unobservable X bounded.
    """
    x, y, th = map(float, pose_guess)
    lines = _corridor_lines(model)
    inv_var = 1.0 / (sigma * sigma)
    H = np.zeros((3, 3))
    inliers = 0

    for _ in range(iters):
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        pw = points_sensor @ R.T + np.array([x, y])

        H = np.zeros((3, 3))
        g = np.zeros(3)
        inliers = 0
        for q, nrm in lines:
            r = pw @ nrm - (q @ nrm)
            m = np.abs(r) < gate
            if not m.any():
                continue
            p = pw[m]
            a = np.stack([np.full(p.shape[0], nrm[0]),
                          np.full(p.shape[0], nrm[1]),
                          nrm[1] * (p[:, 0] - x) - nrm[0] * (p[:, 1] - y)], axis=1)
            H += inv_var * (a.T @ a)
            g += inv_var * (a.T @ r[m])
            inliers += int(m.sum())

        step = np.linalg.solve(H + (damping + 1e-9) * np.eye(3), -g)
        x, y, th = x + step[0], y + step[1], th + step[2]
        th = (th + math.pi) % (2 * math.pi) - math.pi
        if np.linalg.norm(step) < 1e-5:
            break

    return np.array([x, y, th]), H, inliers


def covariance_from_information(H, min_var=0.035, max_var=1e9):
    """R = H^-1 with an eigenvalue clip: min_var floors the observable axes to
    realistic scan-match precision, max_var caps the degenerate (X) axis."""
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    var = np.clip(1.0 / np.clip(w, 1e-9, None), min_var, max_var)
    R = V @ np.diag(var) @ V.T
    return 0.5 * (R + R.T)
