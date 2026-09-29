"""Запуск детектора; по желанию — проигрывание bag и RViz.

    ros2 launch metro_obstacle detect.launch.py                      # слушать живой топик
    ros2 launch metro_obstacle detect.launch.py bag:=/data/<bag>     # проиграть bag
    ros2 launch metro_obstacle detect.launch.py bag:=/data/<bag> rviz:=true rate:=0.5

Пустой аргумент параметра ноды = значение из params_file (config/default.yaml).
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnProcessIO
from launch.events import Shutdown
from launch.events.process import ProcessIO
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SHARE = get_package_share_directory("metro_obstacle")
READY_MARKER = "detector ready"  # см. detector_node.READY_MARKER
TRUE = ("1", "true", "yes", "on")

# Аргументы, которые переопределяют одноимённые параметры ноды, если не пустые.
NODE_ARGS = {
    "input_topic": ("", "PointCloud2 topic; empty = auto-detect the first one"),
    "mode": ("", "detector mode: default | geometry | strict | soft"),
    "verify_model": (
        "",
        "verifier model file (LightGBM text); empty = the one shipped with the core",
    ),
    "forward_axis": ("", "lidar forward axis: auto | x | -x | y | -y"),
    # В Docker путь по умолчанию задаёт переменная окружения образа (каталог /data/out/).
    "results_path": (
        os.environ.get("METRO_RESULTS_PATH", ""),
        "JSON Lines output, one line per frame: a directory (a file per source, "
        "results_<bag or topic>_<time>.jsonl), a template with {name}/{time}, or a file "
        "(appended); none = do not write",
    ),
    "fake_core": (
        os.environ.get("METRO_FAKE_CORE", ""),
        "true = fake detector, to test the ROS wiring without the core",
    ),
}


def _setup(context: LaunchContext) -> list:
    def arg(name: str) -> str:
        return context.launch_configurations.get(name, "").strip()

    overrides: dict[str, object] = {}
    for name in NODE_ARGS:
        value = arg(name)
        if value:
            # Строки — явно строками: иначе launch угадывает тип по YAML ("1" стал бы int).
            overrides[name] = (
                value.lower() in TRUE
                if name == "fake_core"
                else ParameterValue(value, value_type=str)
            )

    bag = arg("bag")
    if bag:
        # Имя файла результатов — по имени bag, а не по топику.
        name = os.path.basename(os.path.normpath(bag))
        overrides["results_tag"] = ParameterValue(name, value_type=str)

    node = Node(
        package="metro_obstacle",
        executable="detector_node",
        name="metro_obstacle_detector",
        output="screen",
        parameters=[arg("params_file"), overrides],
        # Нода упала (нет ядра, неверный параметр) — гасим весь launch, а не висим.
        on_exit=[EmitEvent(event=Shutdown(reason="detector exited"))],
    )
    actions: list = [node]

    if bag:
        cmd = [
            "ros2",
            "bag",
            "play",
            bag,
            "--rate",
            arg("rate"),
            # Сообщения по ~9 МБ: очередь по умолчанию (1000) съела бы гигабайты памяти.
            "--read-ahead-queue-size",
            "10",
            # Пауза после создания издателей: нода успевает найти топик и подписаться,
            # поэтому первые кадры (на них идёт калибровка) не теряются.
            "--delay",
            "1.0",
            # Не выходить, пока подписчики не подтвердили приём: иначе последний кадр
            # (~9 МБ, ещё в пути) теряется вместе с процессом плеера.
            "--wait-for-all-acked",
            "5000",
            "--disable-keyboard-controls",
        ]
        if arg("loop").lower() in TRUE:
            cmd.append("--loop")
        play = ExecuteProcess(cmd=cmd, output="screen", name="bag_play")
        started = []

        def start_play_when_ready(event: ProcessIO) -> list | None:
            # Bag стартует, только когда нода прогрела ядро и ждёт топик: иначе
            # кадры, пришедшие во время компиляции numba, были бы потеряны.
            if started or READY_MARKER not in event.text.decode(errors="replace"):
                return None
            started.append(True)
            return [play]

        actions.append(
            RegisterEventHandler(
                OnProcessIO(
                    target_action=node,
                    on_stdout=start_play_when_ready,
                    on_stderr=start_play_when_ready,
                )
            )
        )
        if arg("exit_after_bag").lower() in TRUE:
            # Дать ноде дообработать последний кадр и дописать results.jsonl.
            actions.append(
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=play,
                        on_exit=[
                            TimerAction(
                                period=2.0,
                                actions=[EmitEvent(event=Shutdown(reason="bag finished"))],
                            )
                        ],
                    )
                )
            )

    if arg("rviz").lower() in TRUE:
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", os.path.join(SHARE, "config", "metro_obstacle.rviz")],
                output="log",
            )
        )
    return actions


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(n, default_value=d, description=h) for n, (d, h) in NODE_ARGS.items()
    ]
    args += [
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(SHARE, "config", "default.yaml"),
            description="node parameters file",
        ),
        DeclareLaunchArgument("bag", default_value="", description="rosbag2 to play; empty = none"),
        DeclareLaunchArgument("rate", default_value="1.0", description="bag playback rate"),
        DeclareLaunchArgument("loop", default_value="false", description="loop the bag"),
        DeclareLaunchArgument(
            "exit_after_bag",
            default_value="true",
            description="stop everything when the bag is over (ignored with loop)",
        ),
        DeclareLaunchArgument("rviz", default_value="false", description="start RViz"),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=_setup)])
