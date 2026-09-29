"""Геометрия маркеров без ROS (общая для ноды и конвертера в MCAP)."""

import math

import numpy as np
import pytest

from metro_obstacle.fake_core import FrameResult, Obstacle
from metro_obstacle.viz import (
    GREEN,
    RED,
    YELLOW,
    box_size,
    box_yaw,
    corridor_lines,
    label_position,
    level_color,
    obstacle_color,
    sorted_obstacles,
    status_text,
)


def _result(obstacle: bool = True, level: int | None = None) -> FrameResult:
    axis = np.stack([np.zeros(30), -np.arange(30.0) * 2, np.full(30, -1.6)], 1)
    obs = [
        Obstacle(56.3, 0.1, 1.7, 0.4, 0.6, 40, 0.9, True, (0.1, -56.3, -0.8), 1.5),
        Obstacle(20.0, -0.3, 0.3, 0.3, 0.2, 5, 0.4, False, (-0.3, -20.0, -1.4)),
    ]
    return FrameResult(
        obstacle=obstacle,
        distance=56.3 if obstacle else math.nan,
        confidence=0.9 if obstacle else 0.0,
        level=(2 if obstacle else 0) if level is None else level,
        visible_range=58.0,
        processing_ms=12.0,
        mode="default",
        calibrated=True,
        obstacles=obs if obstacle or level == 1 else [],
        corridor=axis,
        corridor_half_width=np.full(30, 1.05),
    )


def test_status_text() -> None:
    assert status_text(_result(True)) == "OBSTACLE 56.3 m"
    assert status_text(_result(False, level=1)) == "CANDIDATE 20.0 m"
    assert status_text(_result(False)) == "CLEAR"


def test_colors() -> None:
    near, far = sorted_obstacles(_result(True))
    assert near.distance == pytest.approx(20.0)  # ближние первыми
    assert obstacle_color(far) == RED and obstacle_color(near) == YELLOW
    assert level_color(_result(True)) == RED and level_color(_result(False)) == GREEN


def test_box_follows_track_axis() -> None:
    res = _result(True)
    near, far = sorted_obstacles(res)
    # Ось пути идёт в −Y: бокс повёрнут на −90° вокруг Z.
    assert box_yaw(res, far) == pytest.approx(-math.pi / 2, abs=1e-6)
    assert box_size(far) == pytest.approx((0.4, 0.6, 1.5))  # высота рамки — z_extent
    assert box_size(near)[2] == pytest.approx(0.1)  # плоский объект — минимум, чтобы был виден
    x, y, z = label_position(far, lift=0.6)
    assert (x, y) == pytest.approx((0.1, -56.3)) and z == pytest.approx(-0.8 + 0.75 + 0.6)


def test_corridor_lines() -> None:
    lines = corridor_lines(_result(True), gauge_height=3.0)
    assert lines is not None
    left, right, left_top, right_top = lines.edges
    # Едем в −Y, значит «влево» — это +X.
    np.testing.assert_allclose(left[:, 0], 1.05, atol=1e-9)
    np.testing.assert_allclose(right[:, 0], -1.05, atol=1e-9)
    np.testing.assert_allclose(left_top[:, 2] - left[:, 2], 3.0)
    np.testing.assert_allclose(right_top[:, 2], -1.6 + 3.0)
    # Ось 58 м, рамки раз в 10 м: s = 0, 10, …, 50 → 6 рамок по 4 отрезка.
    assert lines.sections.shape == (24, 2, 3)


def test_corridor_lines_degenerate() -> None:
    res = _result(True)
    res.corridor = np.full((5, 3), np.nan)
    res.corridor_half_width = np.full(5, 1.05)
    assert corridor_lines(res, 3.0) is None
    res.corridor = np.zeros((0, 3))
    res.corridor_half_width = np.zeros(0)
    assert corridor_lines(res, 3.0) is None
