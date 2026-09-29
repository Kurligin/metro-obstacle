"""Сборка ROS-сообщений из FrameResult. Нужен ROS (запускается в Docker-образе)."""

import math

import numpy as np
import pytest

pytest.importorskip("visualization_msgs.msg")
pytest.importorskip("metro_obstacle_msgs.msg")

from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from metro_obstacle.fake_core import FrameResult, Obstacle
from metro_obstacle.messages import (
    detections_msg,
    markers_msg,
    obstacles_msg,
    status_msg,
)


def _result(obstacle: bool) -> FrameResult:
    axis = np.stack([np.zeros(30), -np.arange(30.0) * 2, np.full(30, -1.6)], 1)
    obs = [
        Obstacle(56.3, 0.1, 1.7, 0.4, 0.6, 40, 0.9, True, (0.1, -56.3, -0.8), 1.5),
        Obstacle(20.0, -0.3, 0.3, 0.3, 0.2, 5, 0.4, False, (-0.3, -20.0, -1.4)),
    ]
    return FrameResult(
        obstacle=obstacle,
        distance=56.3 if obstacle else math.nan,
        confidence=0.9 if obstacle else 0.0,
        level=2 if obstacle else 0,
        visible_range=58.0,
        processing_ms=12.0,
        mode="default",
        calibrated=True,
        obstacles=obs if obstacle else [],
        corridor=axis,
        corridor_half_width=np.full(30, 1.05),
    )


def test_status_and_obstacles() -> None:
    h = Header(frame_id="hesai_lidar")
    st = status_msg(_result(True), h, 15.5)
    assert st.obstacle and st.level == 2 and abs(st.distance - 56.3) < 1e-4
    assert st.header.frame_id == "hesai_lidar" and st.mode == "default"
    assert st.calibrated
    arr = obstacles_msg(_result(True), h)
    assert [round(o.distance, 1) for o in arr.obstacles] == [20.0, 56.3]  # ближние первыми
    assert arr.obstacles[1].z_extent == pytest.approx(1.5)
    warming = _result(False)
    warming.calibrated = False  # ядро ещё калибруется — статус говорит об этом честно
    clear = status_msg(warming, h, 1.0)
    assert not clear.obstacle and math.isnan(clear.distance) and not clear.calibrated


def test_detections_follow_track_axis() -> None:
    det = detections_msg(_result(True), Header(frame_id="f"))
    assert len(det.detections) == 2
    d = det.detections[1]
    assert d.results[0].hypothesis.score == pytest.approx(0.9)
    assert d.results[0].hypothesis.class_id == "obstacle"
    # Ось пути идёт в −Y: бокс повёрнут на −90° вокруг Z.
    q = d.bbox.center.orientation
    assert 2 * math.atan2(q.z, q.w) == pytest.approx(-math.pi / 2, abs=1e-6)
    # высота рамки — вертикальный размер точек, а не верх над опорой
    assert d.bbox.size.z == pytest.approx(1.5) and d.bbox.center.position.z == pytest.approx(-0.8)
    assert det.detections[0].bbox.size.z == pytest.approx(0.1)  # z_extent 0 → минимум для RViz


def test_markers() -> None:
    m = markers_msg(_result(True), Header(frame_id="f"), 3.0).markers
    assert m[0].action == Marker.DELETEALL
    texts = [x.text for x in m if x.type == Marker.TEXT_VIEW_FACING]
    assert "OBSTACLE 56.3 m" in texts and "56.3 m" in texts
    assert all(t.isascii() for t in texts)
    cubes = [x for x in m if x.type == Marker.CUBE]
    assert len(cubes) == 2
    assert sorted(round(c.scale.z, 2) for c in cubes) == [0.1, 1.5]
    assert {(round(c.color.r, 2), round(c.color.g, 2)) for c in cubes} == {(1.0, 0.15), (1.0, 0.85)}
    edges = [x for x in m if x.ns == "corridor" and x.type == Marker.LINE_STRIP]
    assert len(edges) == 4 and all(len(e.points) == 30 for e in edges)
    assert all(e.header.frame_id == "f" for e in m)


def test_markers_survive_bad_corridor() -> None:
    res = _result(False)
    res.corridor[3] = np.nan
    m = markers_msg(res, Header(frame_id="f"), 3.0).markers
    assert [x.text for x in m if x.ns == "status"] == ["CLEAR"]
    pts = [p for x in m for p in x.points]
    assert all(math.isfinite(p.x) and math.isfinite(p.y) for p in pts)
