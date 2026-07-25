"""
Rigorous EKF tuning via Monte Carlo consistency analysis (NEES / NIS).

A filter is CONSISTENT when its reported covariance matches its actual error
statistics -- not over/under-confident. We test this with:

  NEES (needs ground truth, which sim gives us):
      eps_k = (x_true - x_hat)^T P^-1 (x_true - x_hat)
      For an n-DOF consistent filter, E[eps] = n. Averaged over M runs,
      M*eps_bar ~ chi2(M*n): eps_bar must lie in [chi2_.025(Mn)/M, chi2_.975(Mn)/M].

  NIS (innovation-based, no ground truth):
      nu_k = z - H x_pred ,  S = H P H^T + R ,  nis = nu^T S^-1 nu
      For an m-DOF measurement, E[nis] = m, same chi2 test.

We evaluate NEES on the OBSERVABLE 2-DOF subspace (y, theta) -- the honest
target, since X is unobservable in the corridor -- and also report full 3-DOF
and the X-bracket check (does true X error stay within +/-3 sigma_X?). NIS uses
the observable (y, theta) part of the scan-match innovation.
"""
import os, sys
import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "shifting_wall_slam"))

from sim.world import CorridorWorld
from sim.robot import DiffDriveRobot, CorridorDriver
from sim.noise import NoiseInjector, NoiseParams, DeadReckoner
from sim.lidar import Lidar2D
import scan_matching as sm
import ekf_core as ekf_mod


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def run_once(seed, params, scan_every=8, min_var=2.5e-3):
    rng = np.random.default_rng(seed)
    world = CorridorWorld()
    lidar = Lidar2D(360, range_noise_std=0.02, max_range=8.0, rng=rng)
    robot = DiffDriveRobot((0.5, 0.0, 0.0))
    driver = CorridorDriver(0.5)
    noise = NoiseInjector(NoiseParams(), rng)
    odom = DeadReckoner(robot.pose)
    ekf = ekf_mod.EKF([0.5, 0.0, 0.0, 0.0], params=params)
    model = sm.CorridorModel(1.5, 20.0)

    dt = 0.02; t = 0.0; step = 0
    nees2, nees3, nis2, xbracket = [], [], [], []
    while robot.pose[0] < 19.3 and t < 60:
        v, w = driver.command(robot.pose)
        vr, wm, _ = noise.corrupt(v, w, robot.pose[0], dt)
        robot.step(v, w, dt); odom.update(vr, wm, dt); ekf.predict(vr, wm, dt)
        if step % scan_every == 0:
            a, r, h = lidar.scan(robot.pose, world.segments(t), True)
            p = lidar.to_points_body(a, r, h)
            refined, H, _ = sm.icp_to_corridor(p, odom.pose.copy(), model)
            R = sm.covariance_from_information(H, min_var=min_var)
            # --- NIS on observable (y,theta) BEFORE the update ---
            Hj = np.zeros((3, 4)); Hj[0, 0] = Hj[1, 1] = Hj[2, 2] = 1.0
            nu = refined - Hj @ ekf.x; nu[2] = wrap(nu[2])
            S = Hj @ ekf.P @ Hj.T + R
            idx = [1, 2]                      # y, theta
            nu2 = nu[idx]; S2 = S[np.ix_(idx, idx)]
            nis2.append(float(nu2 @ np.linalg.solve(S2, nu2)))
            ekf.update(refined, R)
            # --- NEES AFTER the update ---
            e = ekf.x[:3] - robot.pose; e[2] = wrap(e[2])
            P3 = ekf.P[:3, :3]
            nees3.append(float(e @ np.linalg.solve(P3 + 1e-12*np.eye(3), e)))
            e2 = e[[1, 2]]; P2 = ekf.P[np.ix_([1, 2], [1, 2])]
            nees2.append(float(e2 @ np.linalg.solve(P2, e2)))
            sx = np.sqrt(ekf.P[0, 0])
            xbracket.append(abs(e[0]) <= 3 * sx)
        t += dt; step += 1
    n = min(len(nees2), len(nis2))
    return (np.array(nees2[:n]), np.array(nees3[:n]),
            np.array(nis2[:n]), np.array(xbracket[:n]))


def monte_carlo(params, M=20, scan_every=8, min_var=2.5e-3, label="", verbose=True):
    N2, N3, NI, XB = [], [], [], []
    for s in range(M):
        a, b, c, d = run_once(1000 + s, params, scan_every, min_var)
        N2.append(a); N3.append(b); NI.append(c); XB.append(d)
    K = min(len(a) for a in N2)
    N2 = np.array([x[:K] for x in N2]); N3 = np.array([x[:K] for x in N3])
    NI = np.array([x[:K] for x in NI]); XB = np.array([x[:K] for x in XB])

    def stat(arr, dof):
        mean = arr.mean()
        lo = chi2.ppf(0.025, M * dof) / M
        hi = chi2.ppf(0.975, M * dof) / M
        return mean, lo, hi

    nis = stat(NI, 2); n2 = stat(N2, 2); n3 = stat(N3, 3)
    if verbose:
        print(f"[{label}] M={M}, {K} updates")
        for name, (m, lo, hi), dof in [("NIS(y,th)", nis, 2), ("NEES(y,th)", n2, 2),
                                       ("NEES(xyth)", n3, 3)]:
            v = "CONSISTENT" if lo <= m <= hi else ("OVERCONF" if m > hi else "conserv")
            print(f"   {name:11s} mean={m:7.2f} target={dof} band=[{lo:.2f},{hi:.2f}] -> {v}")
        print(f"   X within 3-sigma: {XB.mean()*100:.0f}%\n")
    return nis[0], n2[0], nis[1], nis[2], XB.mean(), n3[0]


if __name__ == "__main__":
    print("=== BEFORE: raw H^-1 covariance, fixed sigma_v (the original filter) ===")
    old = ekf_mod.EKFParams(vel_noise_frac=0.0, vel_noise_floor=0.05)
    monte_carlo(old, M=15, min_var=1e-9, label="raw R, fixed Q")

    print("=== GRID SEARCH: R-floor (min_var) x velocity-noise frac ===")
    print(f"   {'min_var':<8}{'frac':<6}{'NIS':>7}{'NEES2':>8}{'NEES3':>9}{'X<3sig':>9}")
    best = None
    for min_var in [0.02, 0.035, 0.05, 0.08]:
        for frac in [0.4, 0.6, 0.8]:
            p = ekf_mod.EKFParams(vel_noise_frac=frac, vel_noise_floor=0.02)
            nis, n2, lo, hi, xb, n3 = monte_carlo(p, M=15, min_var=min_var,
                                                  label="", verbose=False)
            print(f"   {min_var:<8}{frac:<6}{nis:>7.2f}{n2:>8.2f}{n3:>9.1f}{xb*100:>8.0f}%")
            # score: NIS near 2 (measurement model) + X well bracketed
            score = abs(np.log(nis / 2.0)) + 2.0 * max(0.0, 0.95 - xb)
            if best is None or score < best[0]:
                best = (score, min_var, frac)
    print(f"\n>>> best (NIS~2 & X bracketed): min_var={best[1]}, frac={best[2]}\n")

    print("=== VERIFY best config (M=25) ===")
    p = ekf_mod.EKFParams(vel_noise_frac=best[2], vel_noise_floor=0.02)
    monte_carlo(p, M=25, min_var=best[1], label=f"TUNED min_var={best[1]} frac={best[2]}")

