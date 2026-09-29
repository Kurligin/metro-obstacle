"""Автокалибровка: направление «вперёд» и плоскость полотна. Ручных extrinsics нет."""

from __future__ import annotations

import math

import numpy as np

# ось «вперёд» → код для kernels.normalize
AXIS_CODE = {"x": 0, "-x": 1, "y": 2, "-y": 3}
_AXIS_ANGLE = {"x": 0.0, "y": 90.0, "-x": 180.0, "-y": -90.0}


def guess_forward_axis(xyz: np.ndarray, min_points: int = 200) -> str | None:
    """Направление «вперёд» по медиане азимута дальних точек, с округлением до оси.

    Лидар на лобовой части видит тоннель впереди, поэтому дальние точки (> 20 м по
    горизонтали) лежат вдоль направления движения; ближние — стены сбоку, они
    смещают медиану (в bag с препятствием для > 5 м медиана −54°, для > 20 м — −84°).
    Медиана — относительно кругового среднего, чтобы не ломаться на переходе ±180°.
    None — если дальних точек мало (решим на следующем кадре).
    """
    p = np.asarray(xyz, dtype=np.float64)
    p = p[np.isfinite(p).all(axis=1)]
    r = np.hypot(p[:, 0], p[:, 1])
    for r_min in (20.0, 5.0):
        far = p[r > r_min]
        if len(far) >= min_points:
            break
    else:
        return None
    az = np.arctan2(far[:, 1], far[:, 0])
    mean = math.atan2(float(np.sin(az).mean()), float(np.cos(az).mean()))
    rel = np.angle(np.exp(1j * (az - mean)))
    med = math.degrees(mean + float(np.median(rel)))

    def dist(a: float) -> float:
        return abs((med - a + 180.0) % 360.0 - 180.0)

    return min(_AXIS_ANGLE, key=lambda k: dist(_AXIS_ANGLE[k]))


CALIB_MIN_POINTS = 50  # меньше точек в зоне калибровки — кадр для калибровки непригоден


def calib_zone(xyz: np.ndarray) -> np.ndarray:
    """Маска зоны калибровки: 3–20 м впереди, ±2 м вбок (нормализованные оси)."""
    return calib_zone_xy(xyz[:, 0], xyz[:, 1])


def calib_zone_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """То же по отдельным массивам x, y — без сборки массива (N, 3) всего кадра."""
    return (x > 3) & (x < 20) & (np.abs(y) < 2.0)


def ground_plane(xyz: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float] | None:
    """Плоскость полотна в ближней зоне (RANSAC): n·p + d = 0, n вверх.

    Точки — в нормализованных осях, после кропа коридора, в исходном порядке
    (от порядка зависят выборки RANSAC). Нижняя половина по высоте — полотно
    и рельсы, стены и ниши не мешают.

    None — если в зоне калибровки меньше CALIB_MIN_POINTS точек (пустой или почти
    пустой кадр): плоскость не угадываем, калибруемся на следующем кадре.
    """
    p = xyz[calib_zone(xyz)]
    if len(p) < CALIB_MIN_POINTS:
        return None
    p = p[p[:, 2] < np.percentile(p[:, 2], 50)]
    best, bn = (np.array([0, 0, 1.0]), -np.median(p[:, 2])), -1
    for _ in range(100):
        s = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn * (1 if n[2] > 0 else -1)
        if n[2] < 0.97:
            continue
        d = -n @ s[0]
        c = np.sum(np.abs(p @ n + d) < 0.06)
        if c > bn:
            best, bn = (n, d), c
    return best


def refine_plane(
    xyz: np.ndarray, n: np.ndarray, d: float, tol: float = 0.06
) -> tuple[np.ndarray, float]:
    """Уточнение плоскости МНК по инлайерам RANSAC (одна выборка RANSAC шумит на ±1°)."""
    p = xyz[calib_zone(xyz)].astype(np.float64)
    inl = p[np.abs(p @ n + d) < tol]
    if len(inl) < CALIB_MIN_POINTS:
        return n, d
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    nn = vt[2] if vt[2][2] > 0 else -vt[2]
    return nn, float(-nn @ c)


def leveling_rotation(n: np.ndarray) -> np.ndarray:
    """Поворот R, переводящий нормаль полотна n в ось z (формула Родрига)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c = float(n @ z)
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
