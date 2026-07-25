"""EKF for corridor localization. State: [x, y, theta, gyro_bias]."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class EKFParams:
    vel_noise_frac: float = 0.50     # velocity process noise ~ frac*|v| + floor
    vel_noise_floor: float = 0.02
    gyro_std: float = 0.010
    bias_rw_std: float = 0.0008


class EKF:
    def __init__(self, x0, P0=None, params=None):
        self.x = np.array(x0, dtype=float)
        self.P = np.diag([0.05, 0.05, 0.02, 0.001]) ** 2 if P0 is None else np.array(P0, float)
        self.p = params or EKFParams()

    def predict(self, v, omega, dt):
        px, py, th, b = self.x
        c, s = math.cos(th), math.sin(th)
        self.x = np.array([px + v * c * dt, py + v * s * dt,
                           wrap(th + (omega - b) * dt), b])

        G = np.array([[1, 0, -v * s * dt, 0],
                      [0, 1, v * c * dt, 0],
                      [0, 0, 1, -dt],
                      [0, 0, 0, 1]], float)
        V = np.array([[c * dt, 0], [s * dt, 0], [0, dt], [0, 0]], float)
        sv = self.p.vel_noise_frac * abs(v) + self.p.vel_noise_floor
        M = np.diag([sv ** 2, self.p.gyro_std ** 2 / max(dt, 1e-6)])
        Q = V @ M @ V.T
        Q[3, 3] += self.p.bias_rw_std ** 2 * dt

        self.P = G @ self.P @ G.T + Q
        return self.x.copy(), self.P.copy()

    def update(self, z, R):
        z, R = np.asarray(z, float), np.asarray(R, float)
        H = np.zeros((3, 4))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        y = z - H @ self.x
        y[2] = wrap(y[2])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        A = np.eye(4) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        return self.x.copy(), self.P.copy()

    def pose(self):
        return self.x[:3].copy()

    def pose_cov(self):
        return self.P[:3, :3].copy()
