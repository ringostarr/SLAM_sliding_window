"""Render the GT / odom / fused summary figure from slam_run.csv."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/slam_run.csv")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/slam_summary.png")
    d = np.genfromtxt(csv, delimiter=",", names=True)
    if d.size == 0:
        print("empty log:", csv)
        return

    t = d["t"] - d["t"][0]
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    ax[0].plot(d["gt_x"], d["gt_y"], "g", lw=2.4, label="ground truth")
    ax[0].plot(d["odom_x"], d["odom_y"], "r--", lw=1.6, label="slipping unfused odom")
    ax[0].plot(d["ekf_x"], d["ekf_y"], "b", lw=1.6, label="fused EKF estimate")
    ax[0].axhline(1.5, color="0.6", lw=1)
    ax[0].axhline(-1.5, color="0.6", lw=1)
    ax[0].axvspan(10, 15, color="orange", alpha=0.12, label="dusty slip patch")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("X (m)")
    ax[0].set_ylabel("Y (m)")
    ax[0].legend(fontsize=8, ncol=4, loc="lower left")
    ax[0].set_title("Ground truth vs slipping odometry vs fused estimate")

    lam = d["lambda_min"]
    finite = np.isfinite(lam) & (lam > 0)
    ax[1].semilogy(t[finite], lam[finite], "purple", lw=1.3, label=r"scan-match $\lambda_{min}$")
    w = d["warn"] > 0.5
    ax[1].scatter(t[w], np.clip(lam[w], 1e-2, None), s=6, color="red",
                  label="degeneracy warning", zorder=5)
    ax[1].set_ylabel(r"$\lambda_{min}$")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Axial degeneracy monitor")

    ax[2].plot(t, np.sqrt(np.clip(d["Pxx"], 0, None)) * 100, "r", lw=2, label=r"$\sigma_X$ (cm)")
    ax[2].plot(t, np.sqrt(np.clip(d["Pyy"], 0, None)) * 100, "b", lw=2, label=r"$\sigma_Y$ (cm)")
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel(r"1$\sigma$ (cm)")
    ax[2].legend(fontsize=8)
    ax[2].set_title("Running pose covariance")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"samples {len(t)}  warnings {int(w.sum())}")
    print(f"Y RMSE: odom={rmse(d['gt_y'], d['odom_y']):.3f}  "
          f"EKF={rmse(d['gt_y'], d['ekf_y']):.3f} m")
    print(f"final X err: odom={d['gt_x'][-1]-d['odom_x'][-1]:+.2f}  "
          f"EKF={d['gt_x'][-1]-d['ekf_x'][-1]:+.2f} m")


if __name__ == "__main__":
    main()
