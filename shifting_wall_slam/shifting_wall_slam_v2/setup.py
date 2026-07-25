import os
from glob import glob

from setuptools import find_packages, setup

package_name = "shifting_wall_slam_v2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.sdf")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="candidate",
    maintainer_email="you@example.com",
    description="Degenerate corridor SLAM stress-test.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "panel_mover = shifting_wall_slam_v2.panel_mover_node:main",
            "noise_node  = shifting_wall_slam_v2.noise_node:main",
            "ekf_node    = shifting_wall_slam_v2.ekf_node:main",
            "scan_match  = shifting_wall_slam_v2.scan_match_node:main",
            "mapping     = shifting_wall_slam_v2.mapping_node:main",
            "logger      = shifting_wall_slam_v2.logger_node:main",
            "autodrive   = shifting_wall_slam_v2.autodrive_node:main",
            "plot_summary = shifting_wall_slam_v2.plot_summary:main",
            "plot_slip_summary = shifting_wall_slam_v2.plot_slip_summary:main",
        ],
    },
)
