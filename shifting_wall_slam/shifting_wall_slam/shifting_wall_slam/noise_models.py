"""Section 3 noise models: odometry slip and gyro bias drift."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def integrate_pose(px, py, th, v, omega, dt):
    if abs(omega) < 1e-9:
        px += v * math.cos(th) * dt
        py += v * math.sin(th) * dt
    else:
        r = v / omega
        px += r * (math.sin(th + omega * dt) - math.sin(th))
        py += -r * (math.cos(th + omega * dt) - math.cos(th))
        th = wrap_angle(th + omega * dt)
    return px, py, th


@dataclass
class NoiseParams:
    slip_scale: float = 0.70
    patch_x_min: float = 10.0
    patch_x_max: float = 15.0
    vel_noise_std: float = 0.02
    gyro_white_std: float = 0.035
    gyro_bias_rw_std: float = 7e-5
    bias0: float = 0.001
    # False = literal Section 3 draws N(0, sigma^2) per sample (spec default).
    # True = rate-invariant continuous-time scaling (sigma/sqrt(dt), sigma*sqrt(dt)).
    dt_scaling: bool = False


class NoiseInjector:
    def __init__(self, params=None, seed=0):
        self.p = params or NoiseParams()
        self.rng = np.random.default_rng(seed)
        self.bias = float(self.p.bias0)

    def slip_factor(self, x):
        return self.p.slip_scale if self.p.patch_x_min <= x <= self.p.patch_x_max else 1.0

    def slip_velocity(self, v_true, x_true):
        s = self.slip_factor(x_true)
        return s * v_true + float(self.rng.normal(0.0, self.p.vel_noise_std)), s

    def corrupt_gyro(self, omega_true, dt):
        p = self.p
        if p.dt_scaling and dt > 0:
            eta_g = float(self.rng.normal(0.0, p.gyro_white_std / math.sqrt(dt)))
            db = float(self.rng.normal(0.0, p.gyro_bias_rw_std * math.sqrt(dt)))
        else:
            eta_g = float(self.rng.normal(0.0, p.gyro_white_std))
            db = float(self.rng.normal(0.0, p.gyro_bias_rw_std))
        self.bias += db
        return omega_true + self.bias + eta_g

    def corrupt(self, v_true, omega_true, x_true, dt):
        v_rec, s = self.slip_velocity(v_true, x_true)
        omega_meas = self.corrupt_gyro(omega_true, dt)
        return v_rec, omega_meas, dict(s=s, bias=self.bias)
