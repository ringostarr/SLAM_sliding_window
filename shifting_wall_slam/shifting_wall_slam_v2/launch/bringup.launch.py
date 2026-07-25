"""
Bring up the corridor world + robot + ROS<->Gazebo bridge.

    ros2 launch shifting_wall_slam_v2 bringup.launch.py
    ros2 launch shifting_wall_slam_v2 bringup.launch.py render_engine:=ogre  # no-GPU
    ros2 launch shifting_wall_slam_v2 bringup.launch.py rviz:=true

On a machine without a real GPU (e.g. WSL2 software rendering), pass
render_engine:=ogre -- OGRE1 is far more reliable under llvmpipe than OGRE2.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = get_package_share_directory("shifting_wall_slam_v2")
    ros_gz_sim = get_package_share_directory("ros_gz_sim")

    world = LaunchConfiguration("world")
    rviz = LaunchConfiguration("rviz")
    render_engine = LaunchConfiguration("render_engine")

    world_path = PathJoinSubstitution([pkg, "worlds", world])
    urdf_path = os.path.join(pkg, "urdf", "robot.urdf.xacro")
    bridge_yaml = os.path.join(pkg, "config", "bridge.yaml")
    rviz_cfg = os.path.join(pkg, "config", "view.rviz")

    robot_desc = ParameterValue(Command(["xacro ", urdf_path]), value_type=str)

    # Let Gazebo find models referenced from this package (model:// URIs).
    set_resource = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.pathsep.join([os.path.join(pkg, ".."), os.environ.get("GZ_SIM_RESOURCE_PATH", "")]),
    )

    gz_args = [world_path, " -r -v4 --render-engine ", render_engine]
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_desc, "use_sim_time": True}],
    )

    # Delay the spawn: under WSL software rendering the gz server takes several
    # seconds to accept spawn requests, so spawning immediately races ahead of
    # it and the robot silently fails to appear. spawn_delay is tunable.
    spawn = TimerAction(
        period=LaunchConfiguration("spawn_delay"),
        actions=[Node(
            package="ros_gz_sim",
            executable="create",
            output="screen",
            arguments=[
                "-topic", "robot_description",
                "-name", "corridor_bot",
                "-x", "0.5", "-y", "0.9", "-z", "0.12",
            ],
        )],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[{"config_file": bridge_yaml, "use_sim_time": True}],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(rviz),
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="corridor.sdf"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("render_engine", default_value="ogre2",
                              description="ogre2 (GPU) or ogre (software/no-GPU)"),
        DeclareLaunchArgument("spawn_delay", default_value="3.0",
                              description="seconds to wait for gz server before spawning"),
        set_resource,
        gz_sim,
        rsp,
        spawn,
        bridge,
        rviz_node,
    ])
