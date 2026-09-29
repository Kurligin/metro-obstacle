#!/bin/bash
# Окружение ROS и рабочего пространства — для любой команды в контейнере.
set -e
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
exec "$@"
