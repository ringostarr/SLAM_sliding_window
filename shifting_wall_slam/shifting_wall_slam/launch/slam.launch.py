"""
Full SLAM launch (B3): world + robot + bridge + panel mover + noise node
(pipeline) plus the scan matcher (info matrix + degeneracy) and the EKF.

    ros2 launch shifting_wall_slam slam.launch.py render_engine:=ogre rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("shifting_wall_slam")
    render_engine = LaunchConfiguration("render_engine")
    rviz = LaunchConfiguration("rviz")

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "pipeline.launch.py")),
        launch_arguments={"render_engine": render_engine, "rviz": rviz}.items(),
    )

    common = [{"use_sim_time": True}]

    scan_match = Node(
        package="shifting_wall_slam", executable="scan_match",
        name="scan_match_node", output="screen",
        parameters=common + [{"half_width": 1.5, "length": 20.0,
                              "max_range": 8.0, "lambda_threshold": 50.0}],
    )

    ekf = Node(
        package="shifting_wall_slam", executable="ekf_node",
        name="ekf_node", output="screen",
        parameters=common + [{"x0": 0.5, "y0": 0.9}],
    )

    mapping = Node(
        package="shifting_wall_slam", executable="mapping",
        name="mapping_node", output="screen",
        parameters=common + [{"resolution": 0.1, "max_range": 8.0}],
    )

    logger = Node(
        package="shifting_wall_slam", executable="logger",
        name="logger_node", output="screen",
        parameters=common,
    )

    autodrive = Node(
        package="shifting_wall_slam", executable="autodrive",
        name="autodrive", output="screen",
        parameters=common + [{"speed": 0.5, "lane_y": 0.9, "stop_x": 16.0, "start_x": 1.0, "round_trips": 1}],
        condition=IfCondition(LaunchConfiguration("autodrive")),
    )

    return LaunchDescription([
        DeclareLaunchArgument("render_engine", default_value="ogre2"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("autodrive", default_value="true",
                              description="auto-drive the validated straight pass; "
                                          "set false to teleop manually"),
        pipeline,
        scan_match,
        ekf,
        mapping,
        logger,
        autodrive,
    ])
