"""Injects Section 3 noise: slip on wheel odometry, bias drift on the IMU gyro."""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from shifting_wall_slam_v2.noise_models import NoiseInjector, NoiseParams, integrate_pose
from shifting_wall_slam_v2.conversions import yaw_from_quat, quat_from_yaw, stamp_sec


class NoiseNode(Node):
    def __init__(self):
        super().__init__("noise_node")
        p = self._declare_params()
        self.inj = NoiseInjector(p, seed=self.get_parameter("seed").value)
        self.last_omega_meas = 0.0
        self._t_imu = None
        self._t_odom = None
        self._dr = None

        self.pub_odom = self.create_publisher(Odometry, "/odom_noisy", 10)
        self.pub_imu = self.create_publisher(Imu, "/imu_noisy", 10)
        self.create_subscription(Imu, "/imu", self._imu_cb, 50)
        self.create_subscription(Odometry, "/ground_truth/odom", self._odom_cb, 50)

    def _declare_params(self):
        self.declare_parameter("seed", 42)
        self.declare_parameter("slip_scale", 0.70)
        self.declare_parameter("patch_x_min", 10.0)
        self.declare_parameter("patch_x_max", 15.0)
        self.declare_parameter("vel_noise_std", 0.02)
        self.declare_parameter("gyro_white_std", 0.035)
        self.declare_parameter("gyro_bias_rw_std", 7e-5)
        self.declare_parameter("bias0", 0.001)
        g = self.get_parameter
        return NoiseParams(
            slip_scale=g("slip_scale").value,
            patch_x_min=g("patch_x_min").value,
            patch_x_max=g("patch_x_max").value,
            vel_noise_std=g("vel_noise_std").value,
            gyro_white_std=g("gyro_white_std").value,
            gyro_bias_rw_std=g("gyro_bias_rw_std").value,
            bias0=g("bias0").value)

    def _imu_cb(self, msg):
        t = stamp_sec(msg.header)
        dt = 0.0 if self._t_imu is None else max(0.0, t - self._t_imu)
        self._t_imu = t
        w_meas = self.inj.corrupt_gyro(msg.angular_velocity.z, dt if dt > 0 else 0.01)
        self.last_omega_meas = w_meas

        out = Imu()
        out.header = msg.header
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance
        out.angular_velocity.x = msg.angular_velocity.x
        out.angular_velocity.y = msg.angular_velocity.y
        out.angular_velocity.z = w_meas
        out.angular_velocity_covariance = msg.angular_velocity_covariance
        out.linear_acceleration = msg.linear_acceleration
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance
        self.pub_imu.publish(out)

    def _odom_cb(self, msg):
        t = stamp_sec(msg.header)
        dt = 0.0 if self._t_odom is None else max(0.0, t - self._t_odom)
        self._t_odom = t

        v_true = msg.twist.twist.linear.x
        x_true = msg.pose.pose.position.x
        v_rec, _ = self.inj.slip_velocity(v_true, x_true)

        if self._dr is None:
            self._dr = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                        yaw_from_quat(msg.pose.pose.orientation))
        if dt > 0:
            self._dr = integrate_pose(*self._dr, v_rec, self.last_omega_meas, dt)

        px, py, th = self._dr
        qx, qy, qz, qw = quat_from_yaw(th)
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = "base_link"
        out.pose.pose.position.x = px
        out.pose.pose.position.y = py
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.twist.twist.linear.x = v_rec
        out.twist.twist.angular.z = self.last_omega_meas
        self.pub_odom.publish(out)


def main():
    rclpy.init()
    node = NoiseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
