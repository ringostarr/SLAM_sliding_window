# The Shifting Wall & Slipping Base

A SLAM stress-test for a **degenerate corridor**: a differential-drive robot with
a 16-channel 3D LiDAR and a 6-axis IMU must localize and map a long, featureless
hallway while a wall panel shifts sideways at fixed intervals and injected floor
slip and gyroscope drift corrupt its odometry. The corridor is deliberately
constructed so that motion **along its axis is unobservable** in the middle
section — the estimator must detect and flag this degeneracy rather than fail
silently.

Built for **ROS 2 Jazzy** + **Gazebo Harmonic**, CPU-only (developed under WSL2
with software rendering).

---

## How it works

The pipeline is a set of ROS 2 nodes connected by topics:

| Node | Role | Publishes |
| --- | --- | --- |
| `panel_mover_node` | shifts the wall panel 1.5 m every 15 s | `/panel/cmd_pos` |
| `noise_node` | applies the Section 3 slip + gyro-drift models to ground-truth odometry | `/odom_noisy`, `/imu_noisy` |
| `scan_match_node` | point-to-line scan matching; builds the scan-match **information matrix** and flags axial degeneracy | `/scan_match/pose`, `/scan_match/lambda_min`, `/diagnostics` |
| `ekf_node` | EKF fusing corrupted odometry with the scan-match pose; publishes running covariance | `/ekf/odom`, TF `map→odom` |
| `mapping_node` | log-odds occupancy grid with ray-cast ghost clearing | `/map` |
| `logger_node` | timestamped ground-truth / odom / fused pose, full covariance, λ_min, warning flag | `~/slam_run.csv`, `~/slam_warnings.log` |
| `autodrive_node` | drives a repeatable pass down the corridor (forward-only or there-and-back) | `/cmd_vel` |

**Degeneracy detection.** In a straight corridor the wall normals all point
laterally, so the along-corridor component of every scan constraint is ~zero and
the information matrix `H` has a near-zero eigenvalue. The smallest eigenvalue
`λ_min` and its alignment with the corridor axis are monitored each scan; when
`λ_min` collapses and the weak eigenvector aligns with the axis, the node emits a
`LOCALIZATION_DEGENERACY_WARNING`. Feeding `R = H⁻¹` (eigen-floored) into the EKF
makes the Kalman gain along the degenerate axis vanish automatically, so the
filter correctly declines to "correct" a direction it cannot observe.

---

## Two variants

The two packages differ **only** in the mapping layer; the world, robot, noise
model, EKF, scan matching, and degeneracy detection are identical, so a
side-by-side run isolates the mapping change.

| Package | Mapping |
| --- | --- |
| `shifting_wall_slam` | ray-cast occupancy clearing (baseline) |
| `shifting_wall_slam_v2` | full-beam free-space carving + occupancy decay, so a moved panel's stale footprint fades even when beam divergence at range cannot re-sweep every cell |

## The corridor and its degeneracy

The corridor is **closed at both ends** and the LiDAR range is limited to 8 m.
Near an end the robot sees an end wall, which constrains position along the
corridor; in the middle neither end is within range, so only the parallel side
walls are visible and the along-corridor axis becomes unobservable. Axial
uncertainty therefore **spikes in the blind middle and recovers near the ends**,
and the degeneracy warning fires only in that middle section.

---

## Build & run

Requires a ROS 2 Jazzy + Gazebo Harmonic environment
(see `setup_wsl.sh` for a WSL2 environment bootstrap).

```bash
mkdir -p ~/ros2_ws/src
cp -r shifting_wall_slam shifting_wall_slam_v2 ~/ros2_ws/src/
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Run the full system (auto-drives the robot, RViz optional). `render_engine:=ogre`
selects software rendering for CPU-only / WSL setups.

```bash
# there-and-back pass (re-observes the corridor from both directions)
ros2 launch shifting_wall_slam slam.launch.py render_engine:=ogre rviz:=true

# single forward pass (used for the slip summary; slip is not cancelled by a return trip)
ros2 launch shifting_wall_slam slam_forward.launch.py render_engine:=ogre rviz:=true
```

Generate the summary figures from a logged run:

```bash
ros2 run shifting_wall_slam plot_slip_summary   # -> ~/slip_summary.png  (GT vs odom vs fused)
ros2 run shifting_wall_slam plot_summary        # -> ~/slam_summary.png  (+ λ_min and covariance)
```

Swap `shifting_wall_slam` for `shifting_wall_slam_v2` to run the improved-mapping variant.

---

## Repository layout

```
shifting_wall_slam/           baseline package
  worlds/corridor.sdf         closed corridor + shifting panel
  urdf/robot.urdf.xacro       diff-drive robot, 16-ch 3D LiDAR, 6-axis IMU
  launch/                     bringup / pipeline / slam / slam_forward
  config/bridge.yaml          ros_gz topic bridge
  shifting_wall_slam/         nodes + estimation/mapping/noise modules
  reference_python/           pure-Python reference sim + validation scripts
shifting_wall_slam_v2/        same, with full-beam + decay mapping
figures/                      validation and summary plots
```

## Validation

Every estimation algorithm is validated numerically in each package's
`reference_python/`, a pure-Python re-implementation of the simulator and
estimators independent of ROS/Gazebo:

- `validate_b3.py` — degeneracy (λ_min ≈ 0 in the blind middle, large near the ends) and lateral RMSE, odom vs EKF.
- `validate_b4.py` — ghost clearing (panel-footprint occupancy before/after a shift).
- `tune_ekf.py` — Monte Carlo **NEES/NIS consistency** analysis used to tune the filter.
- `plot_slip.py` — the Section 3 slip-model figure.

## Design decisions & limitations

- **2D planar reduction.** A real 16-channel 3D LiDAR is instantiated; the scan
  matcher filters to a near-horizontal ring and solves the planar problem, which
  is the one relevant to corridor-axis degeneracy.
- **Degeneracy depends on the range assumption.** A closed corridor with
  unlimited range is *not* degenerate — the robot always sees an end wall. The
  8 m effective range models low-reflectivity, dusty concrete giving no reliable
  distant returns; it is the mechanism that makes the mid-corridor blind.
- **EKF tuned for consistency, not by eye.** `tune_ekf.py` uses NEES/NIS against
  chi-square bounds. It surfaced that the raw scan-match Hessian implies
  sub-millimetre precision (overconfident by ~10⁴×); the fix floors the
  scan-match covariance and uses velocity-proportional process noise.
- **The along-corridor axis is deliberately not made consistent.** The slip is a
  deterministic bias on an unobservable axis, which a Kalman filter cannot
  correct or bound; the fused estimate retains the ~1.5 m error, which is why the
  degeneracy warning exists. Fusion recovers the observable directions (lateral,
  heading) to a few centimetres.
- **Section 3 noise is implemented literally** (per-sample `N(0, σ²)` draws as
  written; a rate-invariant variant is available behind a flag). The σ constants
  are not specified by the brief and were chosen for realistic drift.

## Requirements

- ROS 2 Jazzy
- Gazebo Harmonic (`ros-jazzy-ros-gz`)
- Python: `numpy`, `scipy`, `matplotlib` (reference sim / plots)
