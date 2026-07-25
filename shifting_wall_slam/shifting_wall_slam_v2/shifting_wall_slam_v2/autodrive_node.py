"""Auto-drive: forward down the corridor, turn 180 in place, drive back.

State machine: fwd -> turn_to_back -> back -> (turn_to_fwd -> fwd)*  -> done.
Holds lane_y in both directions (the cross-track sign flips when facing -x).
Steers off ground-truth pose, matching the reference CorridorDriver profile.
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from shifting_wall_slam_v2.conversions import yaw_from_quat, wrap


class AutoDrive(Node):
    def __init__(self):
        super().__init__("autodrive")
        self.declare_parameter("speed", 0.5)
        self.declare_parameter("lane_y", 0.0)
        self.declare_parameter("ky", 0.4)
        self.declare_parameter("kth", 1.2)
        self.declare_parameter("stop_x", 19.0)      # far turnaround
        self.declare_parameter("start_x", 1.0)      # near turnaround / finish
        self.declare_parameter("round_trips", 1)    # 0 = loop forever
        self.declare_parameter("return_trip", True) # False = forward-only, stop at stop_x
        self.declare_parameter("rate", 50.0)
        g = self.get_parameter
        self.v = g("speed").value
        self.lane_y = g("lane_y").value
        self.ky = g("ky").value
        self.kth = g("kth").value
        self.stop_x = g("stop_x").value
        self.start_x = g("start_x").value
        self.round_trips = g("round_trips").value
        self.return_trip = g("return_trip").value

        self.pose = None
        self.state = "fwd"
        self.trips = 0
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self._odom_cb, 20)
        self.create_timer(1.0 / g("rate").value, self._tick)

    def _odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def _set_state(self, s):
        self.state = s
        self.get_logger().info(f"[autodrive] -> {s}")

    def _tick(self):
        cmd = Twist()
        if self.pose is None:
            self.pub.publish(cmd)
            return
        x, y, th = self.pose
        e = y - self.lane_y

        if self.state == "fwd":
            if x >= self.stop_x:
                self._set_state("turn_to_back" if self.return_trip else "done")
            else:
                th_des = max(-0.2, min(0.2, -self.ky * e))
                cmd.linear.x = self.v
                cmd.angular.z = self.kth * wrap(th_des - th)

        elif self.state == "turn_to_back":
            err = wrap(math.pi - th)                 # face -x
            if abs(err) < 0.05:
                self._set_state("back")
            else:
                cmd.angular.z = max(-1.0, min(1.0, self.kth * err))

        elif self.state == "back":
            if x <= self.start_x:
                self.trips += 1
                if self.round_trips and self.trips >= self.round_trips:
                    self._set_state("done")
                else:
                    self._set_state("turn_to_fwd")
            else:
                th_des = wrap(math.pi + max(-0.2, min(0.2, self.ky * e)))
                cmd.linear.x = self.v
                cmd.angular.z = self.kth * wrap(th_des - th)

        elif self.state == "turn_to_fwd":
            err = wrap(0.0 - th)                      # face +x
            if abs(err) < 0.05:
                self._set_state("fwd")
            else:
                cmd.angular.z = max(-1.0, min(1.0, self.kth * err))

        # "done": zero cmd -> hold
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = AutoDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
