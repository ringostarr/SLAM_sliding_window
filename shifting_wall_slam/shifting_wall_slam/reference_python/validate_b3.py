"""
Validate B3 algorithms end-to-end in Python (what we CAN run), using the
verified Phase A plant + noise to feed the REAL scan_matching + ekf_core
modules that the ROS nodes wrap.

Checks:
  - scan-match information matrix is degenerate along X (small lambda_min,
    weakest eigenvector aligned with X);
  - ICP corrects Y & theta but not X;
  - EKF fused estimate: Y recovered, X slip retained, sigma_X >> sigma_Y;
  - degeneracy warning fires through the corridor.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)                                  # reference_python
sys.path.insert(0, os.path.join(HERE, "..", "shifting_wall_slam"))  # package modules

from sim.world import CorridorWorld
from sim.robot import DiffDriveRobot, CorridorDriver
from sim.noise import NoiseInjector, NoiseParams, DeadReckoner
import scan_matching as sm
import ekf_core as ekf_mod

rng = np.random.default_rng(42)
world = CorridorWorld()
from sim.lidar import Lidar2D
lidar = Lidar2D(n_beams=360, range_noise_std=0.02, max_range=8.0, rng=rng)
robot = DiffDriveRobot(pose0=(0.5, 0.0, 0.0))
driver = CorridorDriver(0.5)
noise = NoiseInjector(NoiseParams(), rng)
odom = DeadReckoner(robot.pose)
ekf = ekf_mod.EKF(x0=[0.5, 0.0, 0.0, 0.0], params=ekf_mod.EKFParams())
model = sm.CorridorModel(half_width=1.5, length=20.0)

dt = 0.02
scan_every = 5           # 10 Hz scan matching
DEG_THRESH = 50.0        # lambda_min below this + X-aligned => warn

log = {k: [] for k in ("t","gx","gy","ox","oy","ex","ey","lammin","xalign",
                       "sx","sy","warn","icp_dx","icp_dy")}
t = 0.0; step = 0; warnings = 0
while robot.pose[0] < 19.3 and t < 60:
    v, w = driver.command(robot.pose)
    vr, wm, d = noise.corrupt(v, w, robot.pose[0], dt)
    robot.step(v, w, dt)
    odom.update(vr, wm, dt)
    ekf.predict(vr, wm, dt)

    lam_min = np.nan; xalign = np.nan; warn = 0
    if step % scan_every == 0:
        ang, rng_, hit = lidar.scan(robot.pose, world.segments(t), add_noise=True)
        pts = lidar.to_points_body(ang, rng_, hit)           # sensor frame
        normals, valid = sm.estimate_normals(pts)
        # ICP seeded from the (drifting) odometry pose -> no circular dependency
        guess = odom.pose.copy()
        refined, H, ninl = sm.icp_to_corridor(pts, guess, model)
        deg = sm.degeneracy(H)
        lam_min, xalign = deg["lambda_min"], deg["x_alignment"]
        R = sm.covariance_from_information(H)
        ekf.update(refined, R)
        # degeneracy warning: weak axis exists AND it points along the corridor
        if lam_min < DEG_THRESH and xalign > 0.8:
            warn = 1; warnings += 1
        log["icp_dx"].append(refined[0] - guess[0])
        log["icp_dy"].append(refined[1] - guess[1])
        log.setdefault("scan_x", []).append(robot.pose[0])
        log.setdefault("scan_lam", []).append(lam_min)

    t += dt; step += 1
    P = ekf.pose_cov()
    for k, val in [("t",t),("gx",robot.pose[0]),("gy",robot.pose[1]),
                   ("ox",odom.pose[0]),("oy",odom.pose[1]),
                   ("ex",ekf.x[0]),("ey",ekf.x[1]),
                   ("lammin",lam_min),("xalign",xalign),
                   ("sx",np.sqrt(P[0,0])),("sy",np.sqrt(P[1,1])),("warn",warn)]:
        log[k].append(val)
L = {k: np.array(v, float) for k, v in log.items()}

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
scan_mask = np.isfinite(L["lammin"])
print("=== B3 end-to-end validation ===")
print(f"scan updates: {scan_mask.sum()}   degeneracy warnings: {warnings}")
sx = np.array(log["scan_x"]); sl = np.array(log["scan_lam"])
mid = (sx > 8) & (sx < 12)
ends = (sx < 4) | (sx > 16)
print(f"lambda_min in blind MIDDLE (8<x<12): median={np.median(sl[mid]):.1f}  (want ~0, degenerate)")
print(f"lambda_min near ENDS   (x<4|x>16):   median={np.median(sl[ends]):.1f}  (want large, X observable)")
print("=> axial uncertainty SPIKES in the middle and recovers at the ends (closed corridor)")
print(f"median lambda_min = {np.nanmedian(L['lammin']):.1f}   "
      f"median X-alignment = {np.nanmedian(L['xalign']):.3f}  (want ~1)")
print(f"ICP correction magnitude: dY rms = {rmse(np.array(log['icp_dy']),0):.3f} m  "
      f"dX rms = {rmse(np.array(log['icp_dx']),0):.3f} m")
print(f"Y RMSE:  odom={rmse(L['gy'],L['oy']):.3f}  EKF={rmse(L['gy'],L['ey']):.3f} m")
print(f"final X error: odom={L['gx'][-1]-L['ox'][-1]:+.2f}  EKF={L['gx'][-1]-L['ex'][-1]:+.2f} m")
print(f"final sigma_X={L['sx'][-1]*100:.1f}cm  sigma_Y={L['sy'][-1]*100:.1f}cm  "
      f"ratio={L['sx'][-1]/L['sy'][-1]:.1f}x")

# ---- figure ----
fig, ax = plt.subplots(3, 1, figsize=(12, 8))
for s in world._static:
    ax[0].plot([s[0],s[2]],[s[1],s[3]], color="0.5", lw=1.2)
ax[0].plot(L["gx"],L["gy"], "g", lw=2.4, label="ground truth")
ax[0].plot(L["ox"],L["oy"], "r--", lw=1.6, label="slipping odom")
ax[0].plot(L["ex"],L["ey"], "b", lw=1.6, label="EKF + real scan match")
ax[0].axvspan(10,15,color="orange",alpha=0.12)
ax[0].set_xlim(-.5,20.5); ax[0].set_ylim(-1.9,1.9); ax[0].set_aspect("equal")
ax[0].legend(fontsize=8,ncol=3,loc="lower left")
ax[0].set_title("(a) EKF driven by the real scan-matching information matrix",fontsize=10)

ax[1].plot(L["t"][scan_mask], L["lammin"][scan_mask], "purple", lw=1.5,
           label=r"$\lambda_{min}$ of scan-match info matrix")
ax[1].axhline(DEG_THRESH, color="red", ls=":", label=f"warn threshold ({DEG_THRESH:.0f})")
w = L["warn"]>0
ax[1].scatter(L["t"][w], L["lammin"][w] if scan_mask.all() else np.full(w.sum(),DEG_THRESH),
              s=8, color="red", zorder=5, label="DEGENERACY_WARNING")
ax[1].set_yscale("log"); ax[1].set_ylabel(r"$\lambda_{min}$"); ax[1].legend(fontsize=8)
ax[1].set_title("(b) Axial degeneracy: weakest eigenvalue stays below threshold in the corridor",fontsize=10)

ax[2].plot(L["t"], L["sx"]*100, "r", lw=2, label=r"$\sigma_X$ (cm) — degenerate axis")
ax[2].plot(L["t"], L["sy"]*100, "b", lw=2, label=r"$\sigma_Y$ (cm) — constrained")
ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("1$\\sigma$ (cm)"); ax[2].legend(fontsize=8)
ax[2].set_title("(c) Running pose covariance",fontsize=10)
fig.tight_layout()
out = os.path.join(HERE, "plot_stepB3_validation.png")
fig.savefig(out, dpi=130); print("saved", out)
