#!/usr/bin/env bash
# One-shot setup for ROS 2 Jazzy + Gazebo Harmonic on Ubuntu 24.04 (WSL2).
# Run inside your WSL2 Ubuntu 24.04 shell:  bash setup_wsl.sh
set -e

echo "==> Locale"
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "==> ROS 2 apt repository"
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

echo "==> ROS 2 Jazzy + Gazebo Harmonic + tools"
sudo apt install -y ros-dev-tools ros-jazzy-desktop
# ros-jazzy-ros-gz pulls in Gazebo Harmonic + bridge + sim + interfaces:
sudo apt install -y ros-jazzy-ros-gz
sudo apt install -y ros-jazzy-xacro ros-jazzy-robot-state-publisher \
  ros-jazzy-teleop-twist-keyboard ros-jazzy-tf-transformations python3-transforms3d

echo "==> rosdep"
sudo rosdep init 2>/dev/null || true
rosdep update

echo "==> Done. Add to your ~/.bashrc:"
echo "    source /opt/ros/jazzy/setup.bash"
echo "If sensors fail to render under WSL2 software GL, launch with render_engine:=ogre"
