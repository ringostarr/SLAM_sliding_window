"""
In-situ noise injection node (Section 3 of the brief).

This is the standalone module that intercepts ground-truth odometry and
degrades it into what the robot actually *records*. It implements the two
mandated models:

(1) Linear Odometry Slip -- Dynamic Intermittent Scale Slip
        v_recorded(t) = s(t) * v_true(t) + w_v(t)
        w_v ~ N(0, sigma_v^2)
        s(t) = 0.70   for 10 <= X <= 15   (30% slip over a 5 m dusty patch)
               1.00   otherwise
    The patch straddles the panel at x=10, so odometry silently UNDER-reports
    forward distance exactly while the robot passes the moving wall -- this is
    what injects an along-corridor (X) error into dead reckoning, the very axis
    scan matching cannot correct. That coupling is the crux of the assignment.

(2) IMU Gyro Deterioration -- Random Walk + Bias Drift
        omega_measured(t) = omega_true(t) + b(t) + eta_g(t)
        b_dot(t)          = eta_b(t)
        eta_g ~ N(0, sigma_g^2)   gyroscope white noise (angle random walk)
        eta_b ~ N(0, sigma_b^2)   bias instability (rate random walk)
    The bias b is an integrated random walk, so heading error grows without
    bound -- pure white noise would average out, which is why the brief insists
    on drifting bias instead.

Discretization note (important, and a common bug source): continuous white
noise must be scaled by the timestep so the *effect* is loop-rate independent.
We treat sigma_g as a continuous density and draw the per-step gyro noise with
std sigma_g / sqrt(dt); the bias increment is a Wiener step with std
sigma_b * sqrt(dt). Set `dt_scaling=False` to reproduce the brief's equations
literally (per-sample N(0, sigma^2)), which is fine at fixed rate but couples
noise magnitude to dt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.robot import integrate_pose


@dataclass
class NoiseParams:
    # --- odometry slip ---
    slip_scale: float = 0.70          # s(t) inside the patch
    patch_x_min: float = 10.0
    patch_x_max: float = 15.0
    vel_noise_std: float = 0.02       # sigma_v [m/s], white measurement noise

    # --- gyro deterioration ---
    # Tuned so heading drift over a ~40 s run is a few degrees: visible and
    # correctable by fusion, not a catastrophic spin-out. sigma_g drives
    # short-term angle random walk; sigma_b drives the slow bias wander.
    gyro_white_std: float = 0.035    # sigma_g (per-sample, literal mode)
    gyro_bias_rw_std: float = 7e-5   # sigma_b (per-sample, literal mode)
    bias0: float = 0.001              # small nonzero starting bias [rad/s]

    dt_scaling: bool = False          # literal Section 3 per-sample draws (spec)


class NoiseInjector:
    """Stateful because the gyro bias integrates over time."""

    def __init__(self, params: NoiseParams | None = None,
                 rng: np.random.Generator | None = None):
        self.p = params if params is not None else NoiseParams()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.bias = float(self.p.bias0)

    # ------------------------------------------------------------------ #
    def slip_factor(self, x_true: float) -> float:
        """s(t): 0.70 on the dusty patch, else 1.00 (position-dependent)."""
        if self.p.patch_x_min <= x_true <= self.p.patch_x_max:
            return self.p.slip_scale
        return 1.0

    def corrupt(self, v_true: float, omega_true: float,
                x_true: float, dt: float) -> tuple[float, float, dict]:
        """
        Return (v_recorded, omega_measured, diagnostics).

        x_true is the ground-truth X used to locate the physical dusty patch.
        In a real rig the patch is a property of the FLOOR, not something the
        odometry knows about; in sim we know where it is, so we key off truth.
        """
        p = self.p

        # (1) odometry slip
        s = self.slip_factor(x_true)
        w_v = self.rng.normal(0.0, p.vel_noise_std)
        v_rec = s * v_true + w_v

        # (2) gyro white noise + drifting bias
        if p.dt_scaling:
            eta_g = self.rng.normal(0.0, p.gyro_white_std / np.sqrt(dt))
            db = self.rng.normal(0.0, p.gyro_bias_rw_std * np.sqrt(dt))
        else:
            eta_g = self.rng.normal(0.0, p.gyro_white_std)
            db = self.rng.normal(0.0, p.gyro_bias_rw_std)
        self.bias += db
        omega_meas = omega_true + self.bias + eta_g

        diag = dict(s=s, bias=self.bias, w_v=w_v, eta_g=eta_g,
                    v_true=v_true, v_rec=v_rec,
                    omega_true=omega_true, omega_meas=omega_meas)
        return v_rec, omega_meas, diag


class DeadReckoner:
    """
    Integrates the *recorded* (corrupted) velocities into an odometry pose.
    This is the 'Slipping Unfused Odometry' baseline for the deliverable plot:
    what the robot would believe using wheel odometry + gyro alone, no fusion.
    """

    def __init__(self, pose0):
        self.pose = np.array(pose0, dtype=float)

    def update(self, v_rec: float, omega_meas: float, dt: float) -> np.ndarray:
        self.pose = integrate_pose(self.pose, v_rec, omega_meas, dt)
        return self.pose.copy()


if __name__ == "__main__":
    from sim.robot import DiffDriveRobot, CorridorDriver

    rng = np.random.default_rng(42)
    robot = DiffDriveRobot(pose0=(0.5, 0.0, 0.0))
    driver = CorridorDriver(0.5)
    noise = NoiseInjector(NoiseParams(), rng)
    odom = DeadReckoner(robot.pose)

    dt = 0.02
    t = 0.0
    while robot.pose[0] < 19.5 and t < 60:
        v, w = driver.command(robot.pose)
        vr, wm, d = noise.corrupt(v, w, robot.pose[0], dt)
        robot.step(v, w, dt)
        odom.update(vr, wm, dt)
        t += dt
    gt, od = robot.pose, odom.pose
    print(f"ground truth : x={gt[0]:.2f} y={gt[1]:+.3f} th={gt[2]:+.3f}")
    print(f"slipping odom: x={od[0]:.2f} y={od[1]:+.3f} th={od[2]:+.3f}")
    print(f"X error from slip: {gt[0]-od[0]:+.2f} m   final gyro bias: {noise.bias:+.4f} rad/s")
