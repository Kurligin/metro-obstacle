"""Геометрия для визуализации: кромки коридора габарита, направление оси пути."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _tangents(corridor: np.ndarray) -> np.ndarray:
    """Единичные касательные к оси в горизонтальной плоскости (z лидара — вверх)."""
    d = np.gradient(corridor[:, :2].astype(np.float64), axis=0)
    norm = np.linalg.norm(d, axis=1)
    good = norm > 1e-9
    if not good.any():
        return np.tile([1.0, 0.0], (len(corridor), 1))
    # Для повторяющихся точек берём касательную ближайшей нормальной точки.
    idx = np.flatnonzero(good)
    nearest = idx[np.clip(np.searchsorted(idx, np.arange(len(d))), 0, len(idx) - 1)]
    return d[nearest] / norm[nearest][:, None]


def corridor_edges(corridor: np.ndarray, half_width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Левая и правая кромки коридора на уровне оси (головок рельсов).

    Меньше двух точек оси — направление не определено, кромок нет.
    """
    if len(corridor) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3))
    t = _tangents(corridor)
    normal = np.zeros((len(t), 3))
    normal[:, 0], normal[:, 1] = -t[:, 1], t[:, 0]  # поворот касательной на +90° — «влево»
    hw = np.asarray(half_width, np.float64)[:, None]
    c = corridor.astype(np.float64)
    return c + normal * hw, c - normal * hw


def corridor_length(corridor: np.ndarray) -> float:
    if len(corridor) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(corridor.astype(np.float64), axis=0), axis=1).sum())


def yaw_at(corridor: np.ndarray, point: Sequence[float]) -> float:
    """Курс оси пути в ближайшей к точке вершине — чтобы бокс объекта шёл вдоль пути."""
    if len(corridor) < 2:
        return 0.0
    p = np.asarray(point, np.float64)[:2]
    k = int(np.argmin(np.linalg.norm(corridor[:, :2] - p, axis=1)))
    t = _tangents(corridor)[k]
    return float(np.arctan2(t[1], t[0]))
