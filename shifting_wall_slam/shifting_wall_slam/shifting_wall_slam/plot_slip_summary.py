"""Deliverable summary plot: Ground-truth X-Y trajectory vs slipping unfused
odometry vs fused EKF estimate, from a logged run (slam_run.csv).

Reads the logger CSV and produces two panels:
  (top)    the literal X-Y trajectory of the three (with the slip patch shaded);
  (bottom) X position vs time, where the odometry slip and the EKF's inability
           to correct the degenerate X axis are actually visible.

Usage:
  ros2 run shifting_wall_slam plot_slip_summary          # reads ~/slam_run.csv
  python3 plot_slip_summary.py /path/slam_run.csv out.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/slam_run.csv")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/slip_summary.png")
    d = np.genfromtxt(csv, delimiter=",", names=True)
    if d.size == 0:
        print("empty log:", csv)
        return
    t = d["t"] - d["t"][0]

    fig, ax = plt.subplots(2, 1, figsize=(11, 7))

    # (top) literal X-Y trajectory
    ax[0].axvspan(10, 15, color="orange", alpha=0.12, label="dusty slip patch (10<=x<=15)")
    ax[0].plot(d["gt_x"], d["gt_y"], "g", lw=2.6, label="ground truth")
    ax[0].plot(d["odom_x"], d["odom_y"], "r--", lw=1.8, label="slipping unfused odometry")
    ax[0].plot(d["ekf_x"], d["ekf_y"], "b", lw=1.6, label="fused EKF estimate")
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Y (m)")
    ax[0].set_ylim(-1.6, 1.6)
    ax[0].axhline(1.5, color="0.7", lw=0.8); ax[0].axhline(-1.5, color="0.7", lw=0.8)
    ax[0].legend(fontsize=8, loc="lower left", ncol=2)
    ax[0].set_title("Ground truth vs slipping unfused odometry vs fused estimate (X-Y)")

    # (bottom) X vs time -- the slip and the degenerate-axis behaviour
    ax[1].axvspan(t[(d["gt_x"] >= 10).argmax()], t[(d["gt_x"] >= 15).argmax()],
                  color="orange", alpha=0.12)
    ax[1].plot(t, d["gt_x"], "g", lw=2.6, label="ground truth X")
    ax[1].plot(t, d["odom_x"], "r--", lw=1.8, label="slipping odom X")
    ax[1].plot(t, d["ekf_x"], "b", lw=1.6, label="fused EKF X")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("X (m)")
    ax[1].legend(fontsize=8, loc="upper left")
    dx_odom = d["gt_x"][-1] - d["odom_x"][-1]
    dx_ekf = d["gt_x"][-1] - d["ekf_x"][-1]
    ax[1].set_title(f"X vs time: slip opens a {dx_odom:.2f} m gap in odom; "
                    f"EKF retains {dx_ekf:.2f} m (X is unobservable, fusion can't correct it)")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)
    print(f"Y RMSE: odom={np.sqrt(np.mean((d['gt_y']-d['odom_y'])**2)):.3f}  "
          f"EKF={np.sqrt(np.mean((d['gt_y']-d['ekf_y'])**2)):.3f} m")
    print(f"final X gap: odom={dx_odom:+.2f}  EKF={dx_ekf:+.2f} m")


if __name__ == "__main__":
    main()
