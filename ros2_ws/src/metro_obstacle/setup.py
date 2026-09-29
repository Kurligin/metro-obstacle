from glob import glob

from setuptools import find_packages, setup

PACKAGE = "metro_obstacle"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE}/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ivan Kurligin",
    maintainer_email="kurligindevelopment@gmail.com",
    description="ROS 2 node: obstacles inside the metro train gauge from a 3D lidar point cloud.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [f"detector_node = {PACKAGE}.detector_node:main"]},
)
