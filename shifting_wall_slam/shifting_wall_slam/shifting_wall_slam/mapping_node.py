"""Occupancy-grid mapping with ray-cast clearing of the moved panel."""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry, OccupancyGrid

from shifting_wall_slam.occupancy_grid import OccupancyGridMap
from shifting_wall_slam.conversions import yaw_from_quat


class MappingNode(Node):
    def __init__(self):
        super().__init__("mapping_node")
        self.declare_parameter("resolution", 0.1)
        self.declare_parameter("z_band", 0.15)
        self.declare_parameter("max_range", 8.0)
        self.declare_parameter("publish_period", 1.0)
        self.z_band = self.get_parameter("z_band").value
        self.max_range = self.get_parameter("max_range").value
        self.grid = OccupancyGridMap(resolution=self.get_parameter("resolution").value)
        self.pose = None

        self.pub = self.create_publisher(OccupancyGrid, "/map", 1)
        self.create_subscription(Odometry, "/ekf/odom", self._pose_cb, 20)
        self.create_subscription(PointCloud2, "/lidar/points", self._cloud_cb, 5)
        self.create_timer(self.get_parameter("publish_period").value, self._publish)

    def _pose_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def _cloud_cb(self, msg):
        if self.pose is None:
            return
        raw = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.array([[r[0], r[1], r[2]] for r in raw], dtype=float)
        if pts.shape[0] < 10:
            return
        rng = np.hypot(pts[:, 0], pts[:, 1])
        keep = (np.abs(pts[:, 2]) < self.z_band) & (rng < self.max_range) & (rng > 0.1)
        p2 = pts[keep, :2]
        if p2.shape[0] < 10:
            return
        angles = np.arctan2(p2[:, 1], p2[:, 0])
        ranges = np.hypot(p2[:, 0], p2[:, 1])
        self.grid.integrate_scan(self.pose, angles, ranges,
                                 np.ones(len(angles), bool), self.max_range)

    def _publish(self):
        og = OccupancyGrid()
        og.header.stamp = self.get_clock().now().to_msg()
        og.header.frame_id = "map"
        og.info.resolution = self.grid.res
        og.info.width = self.grid.nx
        og.info.height = self.grid.ny
        og.info.origin.position.x = self.grid.xmin
        og.info.origin.position.y = self.grid.ymin
        og.info.origin.orientation.w = 1.0
        og.data = self.grid.occupancy_int8().T.reshape(-1).astype(np.int8).tolist()
        self.pub.publish(og)


def main():
    rclpy.init()
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
