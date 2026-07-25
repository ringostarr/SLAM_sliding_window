"""Logs timestamped pose, covariance, lambda_min, and degeneracy warnings."""
import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus

from shifting_wall_slam_v2.conversions import yaw_from_quat


COLS = ["t", "gt_x", "gt_y", "odom_x", "odom_y", "ekf_x", "ekf_y", "ekf_theta",
        "Pxx", "Pxy", "Pxt", "Pyy", "Pyt", "Ptt", "lambda_min", "warn"]


class LoggerNode(Node):
    def __init__(self):
        super().__init__("logger_node")
        self.declare_parameter("csv_path", os.path.expanduser("~/slam_run.csv"))
        self.declare_parameter("warn_path", os.path.expanduser("~/slam_warnings.log"))
        self.csv_path = self.get_parameter("csv_path").value
        self.warn_path = self.get_parameter("warn_path").value

        self.gt = (0.0, 0.0)
        self.odom = (0.0, 0.0)
        self.lam = float("nan")
        self.warn = 0
        self.csv = open(self.csv_path, "w")
        self.csv.write(",".join(COLS) + "\n")
        self.wlog = open(self.warn_path, "w")

        self.create_subscription(Odometry, "/ground_truth/odom", self._gt_cb, 20)
        self.create_subscription(Odometry, "/odom_noisy", self._odom_cb, 20)
        self.create_subscription(Float64, "/scan_match/lambda_min", self._lam_cb, 20)
        self.create_subscription(DiagnosticArray, "/diagnostics", self._diag_cb, 20)
        self.create_subscription(Odometry, "/ekf/odom", self._ekf_cb, 20)
        self.get_logger().info(f"logging to {self.csv_path}")

    def _gt_cb(self, m):
        self.gt = (m.pose.pose.position.x, m.pose.pose.position.y)

    def _odom_cb(self, m):
        self.odom = (m.pose.pose.position.x, m.pose.pose.position.y)

    def _lam_cb(self, m):
        self.lam = m.data

    def _diag_cb(self, m):
        self.warn = 0
        for st in m.status:
            if st.level == DiagnosticStatus.WARN and "DEGENERACY" in st.message:
                self.warn = 1
                t = self.get_clock().now().nanoseconds * 1e-9
                vals = {kv.key: kv.value for kv in st.values}
                self.wlog.write(f"[t={t:8.3f}] {st.message} "
                                f"lambda_min={vals.get('lambda_min', '?')} "
                                f"x_alignment={vals.get('x_alignment', '?')}\n")
                self.wlog.flush()

    def _ekf_cb(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        c = m.pose.covariance
        px, py = m.pose.pose.position.x, m.pose.pose.position.y
        th = yaw_from_quat(m.pose.pose.orientation)
        row = [t, self.gt[0], self.gt[1], self.odom[0], self.odom[1],
               px, py, th,
               c[0], c[1], c[5], c[7], c[11], c[35], self.lam, self.warn]
        self.csv.write(",".join(f"{v:.6g}" for v in row) + "\n")
        self.csv.flush()

    def destroy_node(self):
        try:
            self.csv.close()
            self.wlog.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
