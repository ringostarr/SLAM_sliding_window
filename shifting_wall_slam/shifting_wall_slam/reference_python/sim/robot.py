"""
Differential-drive robot: kinematics + ground-truth motion.

State x = [px, py, theta]  (world frame, theta measured from +X, CCW positive).
Control u = [v, omega]      (body linear speed, body yaw rate).

Continuous kinematics:
    px_dot    = v * cos(theta)
    py_dot    = v * sin(theta)
    theta_dot = omega

We integrate the GROUND TRUTH with the *true* commanded velocities. The noise
node (sim/noise.py) is what later corrupts the *recorded* odometry -- ground
truth here is deliberately clean, because it represents what the physics
engine's joints actually did.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def integrate_pose(pose: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    """
    Exact unicycle integration over dt with constant (v, omega).

    Using the exact arc solution (not Euler) keeps ground truth clean even at
    larger dt, so any trajectory error we see later is attributable to the
    injected noise / estimator, not to sloppy integration.
    """
    px, py, th = pose
    if abs(omega) < 1e-9:
        # Straight-line limit
        px += v * np.cos(th) * dt
        py += v * np.sin(th) * dt
    else:
        r = v / omega
        px += r * (np.sin(th + omega * dt) - np.sin(th))
        py += -r * (np.cos(th + omega * dt) - np.cos(th))
        th = wrap_angle(th + omega * dt)
    return np.array([px, py, th], dtype=float)


class DiffDriveRobot:
    def __init__(self, pose0=(0.5, 0.0, 0.0)):
        self.pose = np.array(pose0, dtype=float)

    def step(self, v: float, omega: float, dt: float) -> np.ndarray:
        self.pose = integrate_pose(self.pose, v, omega, dt)
        return self.pose.copy()


class CorridorDriver:
    """
    Simple open-loop 'controller' that drives the robot straight down the
    corridor at constant speed, with a mild lateral correction so it stays
    near the centerline. Emits the TRUE (v, omega) at each step.

    The near-straight motion is intentional: it is exactly the regime in which
    the corridor's along-axis direction is unobservable to scan matching.
    """

    def __init__(self, target_speed: float = 0.5, ky: float = 0.4, kth: float = 1.2):
        self.v = target_speed
        self.ky = ky      # gain pulling py toward 0
        self.kth = kth    # gain pulling heading toward 0

    def command(self, pose: np.ndarray) -> tuple[float, float]:
        _, py, th = pose
        # Desired heading nudges back toward centerline, then omega tracks it.
        th_des = np.clip(-self.ky * py, -0.2, 0.2)
        omega = self.kth * wrap_angle(th_des - th)
        return self.v, omega


if __name__ == "__main__":
    robot = DiffDriveRobot(pose0=(0.5, 0.3, 0.05))  # start slightly off-center
    driver = CorridorDriver(target_speed=0.5)
    dt = 0.02
    t = 0.0
    while robot.pose[0] < 19.5:
        v, w = driver.command(robot.pose)
        robot.step(v, w, dt)
        t += dt
    print(f"reached x={robot.pose[0]:.2f} y={robot.pose[1]:+.3f} "
          f"theta={robot.pose[2]:+.3f} at t={t:.2f}s")
