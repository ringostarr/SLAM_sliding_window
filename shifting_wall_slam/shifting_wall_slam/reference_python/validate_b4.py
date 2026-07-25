"""Validate B4 ghost clearing: map the corridor, move the panel, show the
stale ('ghost') occupancy at the old panel location gets cleared by ray-casting."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "shifting_wall_slam"))

from sim.world import CorridorWorld
from sim.robot import DiffDriveRobot, CorridorDriver
from sim.lidar import Lidar2D
from occupancy_grid import OccupancyGridMap

rng = np.random.default_rng(1)
world = CorridorWorld()
lidar = Lidar2D(n_beams=360, range_noise_std=0.01, max_range=8.0, rng=rng)
robot = DiffDriveRobot(pose0=(0.5, 0.0, 0.0))
driver = CorridorDriver(0.5)
grid = OccupancyGridMap(resolution=0.1)

dt = 0.02; t = 0.0; step = 0
snap_before = None
panel_y_before = world.panel_y(0.0)
while robot.pose[0] < 19.3 and t < 60:
    v, w = driver.command(robot.pose)
    robot.step(v, w, dt); t += dt; step += 1
    if step % 5 == 0:                       # map at 10 Hz, using TRUE pose
        ang, rg, hit = lidar.scan(robot.pose, world.segments(t), add_noise=True)
        grid.integrate_scan(robot.pose, ang, rg, hit, lidar.max_range)
    if snap_before is None and t >= 14.5:   # just before the shift
        snap_before = grid.prob().copy()
final = grid.prob()

# measure occupancy over the panel FOOTPRINT (the faces the LiDAR strikes),
# not the interior center cell which is never directly hit.
def footprint_max(prob, cx, cy, hx=0.6, hy=0.25):
    i0, j0 = grid.cell(cx - hx, cy - hy)
    i1, j1 = grid.cell(cx + hx, cy + hy)
    return float(prob[i0:i1 + 1, j0:j1 + 1].max())

cx = world.panel_center_x
py_old = panel_y_before
py_new = world.panel_y(30.0)
print("=== B4 ghost-clearing validation ===")
print(f"OLD panel footprint (x={cx}, y={py_old:+.2f}):")
print(f"   max P(occupied) before shift = {footprint_max(snap_before, cx, py_old):.2f}  (mapped as obstacle)")
print(f"   max P(occupied) at end        = {footprint_max(final, cx, py_old):.2f}  (want LOW -> ghost cleared)")
print(f"NEW panel footprint (x={cx}, y={py_new:+.2f}):")
print(f"   max P(occupied) at end        = {footprint_max(final, cx, py_new):.2f}  (want HIGH -> real obstacle)")

fig, ax = plt.subplots(2, 1, figsize=(12, 4.6))
extent = [grid.xmin, grid.xmin + grid.nx * grid.res,
          grid.ymin, grid.ymin + grid.ny * grid.res]
for a, (imgp, title, py) in zip(ax, [
        (snap_before, f"(a) t=14.5 s — panel mapped at y={panel_y_before:+.2f} (before shift)", panel_y_before),
        (final,       f"(b) end — panel now at y={world.panel_y(30.0):+.2f}; old y={panel_y_before:+.2f} obstacle cleared",
         world.panel_y(30.0))]):
    a.imshow(imgp.T, origin="lower", extent=extent, cmap="gray_r",
             vmin=0, vmax=1, aspect="equal")
    a.axvline(world.panel_center_x, color="tab:blue", lw=0.6, alpha=0.5)
    a.plot(world.panel_center_x, panel_y_before, "rx", ms=9, label="old panel spot")
    a.plot(world.panel_center_x, world.panel_y(30.0), "g+", ms=11, label="new panel spot")
    a.set_title(title, fontsize=9.5); a.set_ylabel("Y (m)"); a.legend(fontsize=7, loc="upper right")
ax[-1].set_xlabel("X (m)")
fig.tight_layout()
out = os.path.join(HERE, "plot_stepB4_ghost_clearing.png")
fig.savefig(out, dpi=130); print("saved", out)
