"""Occupancy-grid mapping (v2): full-beam ray-casting + occupancy decay.

Two changes vs the v1 package:
  1. Full-beam reconstruction -- points are binned by azimuth; bins with no
     return become max-range misses that carve free space. Clearing happens
     along every beam direction, not just where a point was returned.
  2. The grid decays toward unknown each scan, so a moved panel's old footprint
     fades even where beam divergence at range can't sweep every cell.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry, OccupancyGrid

from shifting_wall_slam_v2.occupancy_grid import OccupancyGridMap
from shifting_wall_slam_v2.conversions import yaw_from_quat


class MappingNode(Node):
    def __init__(self):
        super().__init__("mapping_node")
        self.declare_parameter("resolution", 0.1)
        self.declare_parameter("z_band", 0.15)
        self.declare_parameter("max_range", 8.0)
        self.declare_parameter("n_beams", 360)
        self.declare_parameter("decay", 0.95)
        self.declare_parameter("publish_period", 1.0)
        self.z_band = self.get_parameter("z_band").value
        self.max_range = self.get_parameter("max_range").value
        self.n_beams = self.get_parameter("n_beams").value
        self.grid = OccupancyGridMap(resolution=self.get_parameter("resolution").value,
                                     decay=self.get_parameter("decay").value)
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
        r2 = rng[keep]
        if p2.shape[0] < 10:
            return

        n = self.n_beams
        az = np.arctan2(p2[:, 1], p2[:, 0])
        idx = ((az + np.pi) / (2 * np.pi) * n).astype(int) % n
        bin_range = np.full(n, np.inf)
        np.minimum.at(bin_range, idx, r2)
        angles = -np.pi + (np.arange(n) + 0.5) * (2 * np.pi / n)
        hit = np.isfinite(bin_range)
        ranges = np.where(hit, bin_range, self.max_range)
        self.grid.integrate_scan(self.pose, angles, ranges, hit, self.max_range)

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
