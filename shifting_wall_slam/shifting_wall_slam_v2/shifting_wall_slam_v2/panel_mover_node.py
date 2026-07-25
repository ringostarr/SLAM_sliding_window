"""Commands the sliding panel: toggles position every shift_period seconds."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class PanelMover(Node):
    def __init__(self):
        super().__init__("panel_mover")
        self.declare_parameter("shift_time", 15.0)
        self.declare_parameter("shift_distance", -1.5)
        self.declare_parameter("shift_period", 0.0)
        self.declare_parameter("publish_rate", 10.0)
        self.shift_time = self.get_parameter("shift_time").value
        self.shift_distance = self.get_parameter("shift_distance").value
        self.shift_period = self.get_parameter("shift_period").value

        self.pub = self.create_publisher(Float64, "/panel/cmd_pos", 10)
        self.timer = self.create_timer(
            1.0 / self.get_parameter("publish_rate").value, self._tick)
        self._announced = 0

    def _n_shifts(self, t):
        if t < self.shift_time:
            return 0
        if self.shift_period and self.shift_period > 0:
            return 1 + int((t - self.shift_time) // self.shift_period)
        return 1

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        n = self._n_shifts(t)
        target = self.shift_distance * (n % 2)
        self.pub.publish(Float64(data=float(target)))
        if n > self._announced:
            self._announced = n
            side = "shifted" if (n % 2) else "start"
            self.get_logger().info(f"[t={t:5.1f}s] shift #{n} -> {side}, cmd {target:+.2f} m")


def main():
    rclpy.init()
    node = PanelMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
