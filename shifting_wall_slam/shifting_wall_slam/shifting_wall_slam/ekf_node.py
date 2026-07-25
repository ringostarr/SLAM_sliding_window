"""EKF fusion node: predict on corrupted odom/gyro, correct on scan match."""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from shifting_wall_slam import ekf_core
from shifting_wall_slam.conversions import yaw_from_quat, stamp_sec


class EKFNode(Node):
    def __init__(self):
        super().__init__("ekf_node")
        self.declare_parameter("x0", 0.5)
        self.declare_parameter("y0", 0.0)
        self.ekf = ekf_core.EKF(
            x0=[self.get_parameter("x0").value, self.get_parameter("y0").value, 0.0, 0.0],
            params=ekf_core.EKFParams())
        self.last_omega = 0.0
        self._t_pred = None
        self._odom_base = None

        self.pub = self.create_publisher(Odometry, "/ekf/odom", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Imu, "/imu_noisy", self._imu_cb, 50)
        self.create_subscription(Odometry, "/odom_noisy", self._odom_cb, 50)
        self.create_subscription(PoseWithCovarianceStamped, "/scan_match/pose", self._scan_cb, 10)
        self.create_subscription(Odometry, "/diff_odom", self._diff_cb, 50)

    def _imu_cb(self, msg):
        self.last_omega = msg.angular_velocity.z

    def _diff_cb(self, msg):
        p = msg.pose.pose
        self._odom_base = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def _odom_cb(self, msg):
        t = stamp_sec(msg.header)
        if self._t_pred is None:
            self._t_pred = t
            return
        dt = t - self._t_pred
        self._t_pred = t
        if dt <= 0 or dt > 1.0:
            return
        self.ekf.predict(msg.twist.twist.linear.x, self.last_omega, dt)
        self._publish(msg.header.stamp)

    def _scan_cb(self, msg):
        p = msg.pose.pose
        z = np.array([p.position.x, p.position.y, yaw_from_quat(p.orientation)])
        c = msg.pose.covariance
        idx = [0, 1, 5]
        R = np.array([[c[idx[a] * 6 + idx[b]] for b in range(3)] for a in range(3)])
        self.ekf.update(z, R)

    def _publish(self, stamp):
        x, y, th, _ = self.ekf.x
        P = self.ekf.P
        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = "map"
        od.child_frame_id = "base_link"
        od.pose.pose.position.x = float(x)
        od.pose.pose.position.y = float(y)
        od.pose.pose.orientation.z = math.sin(th / 2)
        od.pose.pose.orientation.w = math.cos(th / 2)
        cov = [0.0] * 36
        idx = [0, 1, 5]
        for a in range(3):
            for b in range(3):
                cov[idx[a] * 6 + idx[b]] = float(P[a, b])
        od.pose.covariance = cov
        self.pub.publish(od)
        self._broadcast_map_odom(stamp, x, y, th)

    def _broadcast_map_odom(self, stamp, xe, ye, the):
        if self._odom_base is None:
            return
        xo, yo, tho = self._odom_base
        th_mo = the - tho
        c, s = math.cos(th_mo), math.sin(th_mo)
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "map"
        tf.child_frame_id = "odom"
        tf.transform.translation.x = float(xe - (c * xo - s * yo))
        tf.transform.translation.y = float(ye - (s * xo + c * yo))
        tf.transform.rotation.z = math.sin(th_mo / 2)
        tf.transform.rotation.w = math.cos(th_mo / 2)
        self.tf.sendTransform(tf)


def main():
    rclpy.init()
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
