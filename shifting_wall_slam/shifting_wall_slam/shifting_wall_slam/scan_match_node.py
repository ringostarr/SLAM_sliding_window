"""Scan matching, information-matrix degeneracy detection, and the warning."""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseWithCovarianceStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from shifting_wall_slam import scan_matching as sm
from shifting_wall_slam.conversions import yaw_from_quat


class ScanMatchNode(Node):
    def __init__(self):
        super().__init__("scan_match_node")
        self.declare_parameter("half_width", 1.5)
        self.declare_parameter("length", 20.0)
        self.declare_parameter("z_band", 0.15)
        self.declare_parameter("max_range", 8.0)
        self.declare_parameter("lambda_threshold", 50.0)
        self.declare_parameter("x_align_threshold", 0.8)
        self.model = sm.CorridorModel(self.get_parameter("half_width").value,
                                      self.get_parameter("length").value)
        self.z_band = self.get_parameter("z_band").value
        self.max_range = self.get_parameter("max_range").value
        self.lam_thr = self.get_parameter("lambda_threshold").value
        self.xal_thr = self.get_parameter("x_align_threshold").value

        self.guess = None
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, "/scan_match/pose", 10)
        self.pub_lam = self.create_publisher(Float64, "/scan_match/lambda_min", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.create_subscription(Odometry, "/odom_noisy", self._odom_cb, 20)
        self.create_subscription(PointCloud2, "/lidar/points", self._cloud_cb, 5)

    def _odom_cb(self, msg):
        p = msg.pose.pose
        self.guess = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def _cloud_cb(self, msg):
        if self.guess is None:
            return
        raw = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.array([[r[0], r[1], r[2]] for r in raw], dtype=float)
        if pts.shape[0] < 20:
            return
        rng = np.hypot(pts[:, 0], pts[:, 1])
        keep = (np.abs(pts[:, 2]) < self.z_band) & (rng < self.max_range) & (rng > 0.1)
        p2 = pts[keep, :2]
        if p2.shape[0] < 20:
            return
        p2 = p2[np.argsort(np.arctan2(p2[:, 1], p2[:, 0]))]

        refined, H, _ = sm.icp_to_corridor(p2, self.guess, self.model)
        deg = sm.degeneracy(H)
        R = sm.covariance_from_information(H)

        self._publish_pose(msg.header.stamp, refined, R)
        self.pub_lam.publish(Float64(data=float(deg["lambda_min"])))
        self._publish_diag(msg.header.stamp, deg)

    def _publish_pose(self, stamp, pose, R):
        m = PoseWithCovarianceStamped()
        m.header.stamp = stamp
        m.header.frame_id = "map"
        m.pose.pose.position.x = float(pose[0])
        m.pose.pose.position.y = float(pose[1])
        m.pose.pose.orientation.z = math.sin(pose[2] / 2)
        m.pose.pose.orientation.w = math.cos(pose[2] / 2)
        cov = [0.0] * 36
        idx = [0, 1, 5]
        for a in range(3):
            for b in range(3):
                cov[idx[a] * 6 + idx[b]] = float(R[a, b])
        m.pose.covariance = cov
        self.pub_pose.publish(m)

    def _publish_diag(self, stamp, deg):
        degenerate = deg["lambda_min"] < self.lam_thr and deg["x_alignment"] > self.xal_thr
        st = DiagnosticStatus()
        st.name = "localization/scan_match_degeneracy"
        st.hardware_id = "corridor_bot"
        if degenerate:
            st.level = DiagnosticStatus.WARN
            st.message = "LOCALIZATION_DEGENERACY_WARNING"
            self.get_logger().warn(
                f"LOCALIZATION_DEGENERACY_WARNING lambda_min={deg['lambda_min']:.1f} "
                f"x_align={deg['x_alignment']:.2f} cond={deg['condition']:.1e}")
        else:
            st.level = DiagnosticStatus.OK
            st.message = "observable"
        st.values = [KeyValue(key="lambda_min", value=f"{deg['lambda_min']:.3f}"),
                     KeyValue(key="x_alignment", value=f"{deg['x_alignment']:.3f}"),
                     KeyValue(key="condition_number", value=f"{deg['condition']:.3e}")]
        arr = DiagnosticArray()
        arr.header.stamp = stamp
        arr.status = [st]
        self.pub_diag.publish(arr)


def main():
    rclpy.init()
    node = ScanMatchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
