"""
Pipeline launch (B2): world + robot + bridge (bringup) plus the panel mover
and the Section 3 noise node.

    ros2 launch shifting_wall_slam_v2 pipeline.launch.py render_engine:=ogre

Adds on top of bringup:
  panel_mover  -> shifts the wall panel at t=15 s
  noise_node   -> /odom_noisy (slip) and /imu_noisy (gyro drift)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("shifting_wall_slam_v2")
    render_engine = LaunchConfiguration("render_engine")
    rviz = LaunchConfiguration("rviz")

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "bringup.launch.py")),
        launch_arguments={"render_engine": render_engine, "rviz": rviz}.items(),
    )

    common = [{"use_sim_time": True}]

    panel_mover = Node(
        package="shifting_wall_slam_v2", executable="panel_mover",
        name="panel_mover", output="screen",
        parameters=common + [{"shift_time": 15.0, "shift_distance": -1.5,
                              "shift_period": 15.0}],
    )

    noise_node = Node(
        package="shifting_wall_slam_v2", executable="noise_node",
        name="noise_node", output="screen",
        parameters=common + [{"seed": 42}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("render_engine", default_value="ogre2"),
        DeclareLaunchArgument("rviz", default_value="false"),
        bringup,
        panel_mover,
        noise_node,
    ])
