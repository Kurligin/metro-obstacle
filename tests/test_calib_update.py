"""Плоскость полотна уточняется на ходу: медиана оценок по кадрам (plane_every)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from metro_obstacle_core.params import BENCH_COMPAT, MODES, Params
from metro_obstacle_core.synthetic import LIDAR_H, to_message_axes, tunnel_points

from metro_obstacle_core import Detector

BAD_TILT = 4.0  # град: крен ближней зоны на «неудачных» стартовых кадрах


def _frame(rng: np.random.Generator, bad: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Прямой тоннель; bad — ближняя зона (до 25 м) наклонена поперёк пути на BAD_TILT,
    как на стрелке или у платформы: по такому кадру RANSAC находит кривую плоскость."""
    xyz, ring, inten = tunnel_points(rng)
    if bad:
        near = xyz[:, 0] < 25.0
        xyz[near, 2] += np.float32(math.tan(math.radians(BAD_TILT))) * xyz[near, 1]
    return to_message_axes(xyz, "-y"), ring, inten


def _tilt(det: Detector) -> float:
    assert det.plane is not None
    return math.degrees(math.acos(min(1.0, float(det.plane[2]))))


def _run(frames: int, bad_first: int, **over: object) -> Detector:
    rng = np.random.default_rng(1)
    det = Detector("default", "-y", **over)
    for k in range(frames):
        det.process(*_frame(rng, k < bad_first), 0.1 * k)
    return det


def test_bad_start_is_forgotten_by_median() -> None:
    legacy = _run(30, bad_first=3, plane_every=0)
    assert _tilt(legacy) > 0.7 * BAD_TILT  # по первому кадру — кривая навсегда
    det = _run(30, bad_first=3)
    assert _tilt(det) < 0.5
    assert det.plane is not None and det.plane[3] == pytest.approx(LIDAR_H, abs=0.03)


def test_good_start_stays_put() -> None:
    det = _run(40, bad_first=0)
    assert _tilt(det) < 0.5
    assert det.plane is not None and det.plane[3] == pytest.approx(LIDAR_H, abs=0.03)
    assert np.linalg.norm(det.plane[:3]) == pytest.approx(1.0, abs=1e-6)


def test_good_start_keeps_first_plane_with_hysteresis() -> None:
    """Удачный старт: медиана оценок в пределах plane_tol — плоскость первого кадра не
    трогается (как при калибровке по первому кадру); без гистерезиса — меняется."""
    first = _run(1, bad_first=0, plane_every=0)
    det = _run(40, bad_first=0)
    assert det.plane is not None and first.plane is not None
    assert np.array_equal(det.plane, first.plane)
    free = _run(40, bad_first=0, plane_tol_deg=0.0)
    assert free.plane is not None and not np.array_equal(free.plane, first.plane)


def test_first_frame_calibration_unchanged() -> None:
    """Первый кадр калибрует как раньше: старт не задерживается, плоскость та же."""
    rng = np.random.default_rng(2)
    f = _frame(rng, bad=False)
    a, b = Detector("default", "-y"), Detector("default", "-y", plane_every=0)
    a.process(*f, 0.0)
    b.process(*f, 0.0)
    assert a.plane is not None and np.array_equal(a.plane, b.plane)


def test_schedule_dense_then_every_kth() -> None:
    det = _run(40, bad_first=0, plane_dense=20, plane_every=5)
    # 1 (первый кадр) + 20 подряд + кадры 25, 30, 35 после калибровки (39 кадров после неё)
    assert len(det._plane_est) == 1 + 20 + 3
    small = _run(12, bad_first=0, plane_buf=4)
    assert len(small._plane_est) == 4  # буфер — последние plane_buf оценок


def test_plane_update_off_in_bench_compat_and_validated() -> None:
    assert BENCH_COMPAT["plane_every"] == 0
    assert all(p.plane_every > 0 for p in MODES.values())
    with pytest.raises(ValueError):
        Params(plane_every=-1)
    with pytest.raises(ValueError):
        Params(plane_buf=0)
