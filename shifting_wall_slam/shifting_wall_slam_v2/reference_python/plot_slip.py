"""Visualize that the odometry slip matches Section 3: v_rec = s(x)*v_true + w_v,
s=0.70 on 10<=x<=15. Three panels: recorded vs true velocity, the scale factor
s(x), and the accumulated X error."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from sim.robot import DiffDriveRobot, CorridorDriver
from sim.noise import NoiseInjector, NoiseParams, DeadReckoner

rng = np.random.default_rng(0)
robot = DiffDriveRobot((0.5, 0.0, 0.0))
driver = CorridorDriver(0.5)
noise = NoiseInjector(NoiseParams(), rng)
odom = DeadReckoner(robot.pose)

xs, vt, vr, gt_x, od_x, ts = [], [], [], [], [], []
dt, t = 0.02, 0.0
while robot.pose[0] < 19.5 and t < 60:
    v, w = driver.command(robot.pose)
    v_rec, wm, info = noise.corrupt(v, w, robot.pose[0], dt)
    s = info['s']
    robot.step(v, w, dt); odom.update(v_rec, wm, dt)
    xs.append(robot.pose[0]); vt.append(v); vr.append(v_rec)
    gt_x.append(robot.pose[0]); od_x.append(odom.pose[0]); ts.append(t)
    t += dt
xs = np.array(xs); vt = np.array(vt); vr = np.array(vr)
gt_x = np.array(gt_x); od_x = np.array(od_x); ts = np.array(ts)

fig, ax = plt.subplots(3, 1, figsize=(11, 9))

ax[0].axvspan(10, 15, color="orange", alpha=0.15, label="dusty patch 10<=x<=15")
ax[0].plot(xs, vt, "g", lw=2, label="true velocity (physics)")
ax[0].plot(xs, vr, "r", lw=1, alpha=0.7, label="recorded velocity (slipping)")
ax[0].axhline(0.5, color="g", ls=":", lw=1)
ax[0].axhline(0.35, color="r", ls=":", lw=1)
ax[0].annotate("0.70x -> 0.35 m/s", (12.5, 0.36), fontsize=9, ha="center")
ax[0].set_ylabel("v (m/s)"); ax[0].set_xlabel("X position (m)"); ax[0].legend(fontsize=8)
ax[0].set_title("(a) Recorded forward velocity drops to 0.70x inside the patch")

s_meas = vr / np.maximum(vt, 1e-6)
ax[1].axvspan(10, 15, color="orange", alpha=0.15)
ax[1].plot(xs, s_meas, "b", lw=0.8, alpha=0.6, label="measured s = v_rec / v_true")
ax[1].axhline(1.0, color="k", ls="--", lw=1, label="s=1.00 (no slip)")
ax[1].axhline(0.70, color="red", ls="--", lw=1, label="s=0.70 (spec)")
ax[1].set_ylim(0.4, 1.3); ax[1].set_ylabel("scale factor s")
ax[1].set_xlabel("X position (m)"); ax[1].legend(fontsize=8)
ax[1].set_title("(b) State-dependent scale factor s(x) matches Section 3")

err = gt_x - od_x
ax[2].axvspan(10, 15, color="orange", alpha=0.15)
ax[2].plot(xs, err, "purple", lw=2, label="ground-truth X  -  odometry X")
ax[2].axhline(1.5, color="k", ls=":", lw=1, label="predicted 0.30 x 5 m = 1.5 m")
ax[2].set_ylabel("X error (m)"); ax[2].set_xlabel("X position (m)"); ax[2].legend(fontsize=8)
ax[2].set_title(f"(c) Accumulated dead-reckoning X error -> {err[-1]:.2f} m (matches prediction)")

fig.tight_layout()
out = os.path.expanduser("~/slip_validation.png")
try:
    fig.savefig(out, dpi=130)
except Exception:
    out = "/home/claude/slip_validation.png"; fig.savefig(out, dpi=130)
print("saved", out)
print(f"mean s inside patch = {s_meas[(xs>=10)&(xs<=15)].mean():.3f}  (want 0.70)")
print(f"mean s outside patch = {s_meas[(xs<10)|(xs>15)].mean():.3f}  (want 1.00)")
print(f"final accumulated X error = {err[-1]:.2f} m  (want ~1.5)")
