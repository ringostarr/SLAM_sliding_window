import os
from glob import glob

from setuptools import find_packages, setup

package_name = "shifting_wall_slam"

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
            "panel_mover = shifting_wall_slam.panel_mover_node:main",
            "noise_node  = shifting_wall_slam.noise_node:main",
            "ekf_node    = shifting_wall_slam.ekf_node:main",
            "scan_match  = shifting_wall_slam.scan_match_node:main",
            "mapping     = shifting_wall_slam.mapping_node:main",
            "logger      = shifting_wall_slam.logger_node:main",
            "autodrive   = shifting_wall_slam.autodrive_node:main",
            "plot_summary = shifting_wall_slam.plot_summary:main",
            "plot_slip_summary = shifting_wall_slam.plot_slip_summary:main",
        ],
    },
)
