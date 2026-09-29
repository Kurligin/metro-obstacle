"""Нода целиком: вход сменился на запущенной ноде. Нужен ROS (запускается в Docker-образе)."""

import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("metro_obstacle_msgs.msg")

import rclpy
import rclpy.executors
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

from metro_obstacle import detector_node
from metro_obstacle.detector_node import DetectorNode


def _cloud(stamp: float, frame_id: str) -> PointCloud2:
    rng = np.random.default_rng(0)
    xyz = np.c_[rng.uniform(-2, 2, 2000), -rng.uniform(1, 60, 2000), rng.uniform(-1.5, 2, 2000)]
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int((stamp % 1) * 1e9)
    msg.height, msg.width = 1, len(xyz)
    msg.fields = [
        PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1)
        for k, n in enumerate("xyz")
    ]
    msg.point_step, msg.row_step = 12, 12 * len(xyz)
    msg.data = xyz.astype("<f4").tobytes()
    return msg


@pytest.fixture
def ros(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("METRO_FAKE_CORE", "1")  # ядро здесь не проверяем — только обвязку
    rclpy.init()
    node = DetectorNode()
    other = Node("bag_player")
    yield node, other
    other.destroy_node()
    node.destroy_node()
    rclpy.shutdown()


def _spin_until(nodes: tuple, cond, timeout: float = 10.0) -> bool:  # type: ignore[no-untyped-def]
    ex = rclpy.executors.SingleThreadedExecutor()
    for n in nodes:
        ex.add_node(n)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < timeout:
            if cond():
                return True
            ex.spin_once(timeout_sec=0.05)
        return cond()
    finally:
        for n in nodes:
            ex.remove_node(n)


def _feed(nodes: tuple, pub, stamps: list[float], frame_id: str) -> bool:  # type: ignore[no-untyped-def]
    """Кадры по одному: у подписки ноды очередь глубины 1."""
    node = nodes[0]
    for t in stamps:
        n = node._frames
        pub.publish(_cloud(t, frame_id))
        if not _spin_until(nodes, lambda n=n: node._frames == n + 1):
            return False
    return True


def test_second_bag_on_another_topic_is_picked_up(ros) -> None:  # type: ignore[no-untyped-def]
    node, player = ros
    resets: list[str] = []
    real_reset = node._reset

    def spy(reason: str) -> None:
        resets.append(reason)
        real_reset(reason)

    node._reset = spy

    # первый bag: свой топик, своё время
    pub = player.create_publisher(PointCloud2, "/sensing/lidar/hesai128/pointcloud", 10)
    assert _spin_until((node, player), lambda: node._topic == pub.topic_name)
    assert _feed((node, player), pub, [946692903.0 + 0.1 * k for k in range(3)], "hesai_lidar")
    assert node._source_frames == 3
    first = node._detector

    # bag закончился: издателя нет; простой дольше IDLE_S — подписка снята
    player.destroy_publisher(pub)
    node._last_input -= detector_node.IDLE_S + 0.5
    assert _spin_until((node, player), lambda: node._sub is None)

    # второй bag — другой топик: нода находит его сама, ядро сброшено
    pub2 = player.create_publisher(PointCloud2, "/lidar_points", 10)
    assert _spin_until((node, player), lambda: node._topic == "/lidar_points")
    assert resets == ["new input source"] and node._detector is not first
    assert _feed((node, player), pub2, [1700000000.0 + 0.1 * k for k in range(3)], "lidar")
    assert node._source_frames == 3
    assert node._frames == 6 and resets == ["new input source"]
    assert node._tf_parent == "lidar"  # lidar_link перенесён на фрейм нового источника

    # тот же источник: скачок времени вперёд > 5 с — сброс
    assert _feed((node, player), pub2, [1700000010.0], "lidar")
    assert len(resets) == 2 and "forward" in resets[-1]
    # смена frame_id — тоже сброс
    assert _feed((node, player), pub2, [1700000010.1], "lidar_2")
    assert len(resets) == 3 and "frame_id" in resets[-1]


def test_silent_publisher_keeps_subscription(ros) -> None:  # type: ignore[no-untyped-def]
    """Живой лидар замолчал, но его издатель на месте — подписку не дёргаем."""
    node, player = ros
    player.create_publisher(PointCloud2, "/lidar_points", 10)
    assert _spin_until((node, player), lambda: node._topic == "/lidar_points")
    sub = node._sub
    node._last_input -= detector_node.IDLE_S + 0.5
    _spin_until((node, player), lambda: False, timeout=1.0)
    assert node._sub is sub and node._topic == "/lidar_points"


@pytest.fixture
def sigint_handler():  # type: ignore[no-untyped-def]
    """main() меняет обработчик SIGINT процесса — после теста возвращаем прежний."""
    old = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, old)


def test_ctrl_c_shuts_down_quietly_and_keeps_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sigint_handler  # type: ignore[no-untyped-def]
) -> None:
    """Ctrl-C под ros2 launch: SIGINT приходит и из терминала, и от launch — второй
    попадает в закрытие ноды. main() завершается без исключения, журнал дописан и закрыт."""
    monkeypatch.setenv("METRO_FAKE_CORE", "1")
    out = tmp_path / "results.jsonl"
    nodes: list[DetectorNode] = []

    def spin(node: DetectorNode, *args: object, **kwargs: object) -> None:
        nodes.append(node)
        for k in range(3):
            node._on_cloud(_cloud(946692903.0 + 0.1 * k, "lidar"))
        raise KeyboardInterrupt  # первый SIGINT

    real_destroy = DetectorNode.destroy_node

    def destroy(self: DetectorNode) -> None:
        os.kill(os.getpid(), signal.SIGINT)  # второй SIGINT — во время закрытия
        time.sleep(0.05)
        real_destroy(self)

    monkeypatch.setattr(detector_node.rclpy, "spin", spin)
    monkeypatch.setattr(DetectorNode, "destroy_node", destroy)

    detector_node.main(["--ros-args", "-p", f"results_path:={out}"])

    assert not rclpy.ok()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["frame"] for line in lines] == [0, 1, 2]
    assert nodes[0]._results is None  # журнал закрыт


def test_ctrl_c_inside_numba_is_not_a_skipped_frame(ros) -> None:  # type: ignore[no-untyped-def]
    """SIGINT внутри функции numba приходит как SystemError с KeyboardInterrupt в причине —
    это остановка, а не сбой кадра: нода не пишет «frame skipped», а завершается."""
    node, _ = ros

    class Interrupted:
        def process(self, *args: object) -> None:
            try:
                raise KeyboardInterrupt
            except KeyboardInterrupt as e:
                raise SystemError("returned a result with an exception set") from e

    node._detector = Interrupted()
    with pytest.raises(KeyboardInterrupt):
        node._on_cloud(_cloud(946692903.0, "lidar"))
    assert node._errors == 0 and node._frames == 0
