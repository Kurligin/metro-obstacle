"""Простая синтетическая сцена тоннеля: прогрев JIT и юнит-тесты без реальных данных.

Не модель лидара (для экспериментов — трассировка лучей стенда, bench/synth.py):
равномерные точки полотна, головок рельсов и стен, кольца — по углу места.
"""

from __future__ import annotations

import numpy as np

LIDAR_H = 1.35  # высота лидара над полотном, м


def tunnel_points(
    rng: np.random.Generator,
    length: float = 120.0,
    n: int = 60000,
    center: float = 0.0,
    box: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Точки прямого тоннеля в нормализованных осях (x-вперёд, y-влево, z-вверх).

    center — смещение оси пути по y; box = (s, lat, размер, высота) — куб на пути.
    Возвращает (xyz float32 (N, 3), ring int64, intensity float32).
    """
    k = n // 4
    parts = []
    # полотно
    s = rng.uniform(3.0, length, k)
    parts.append(np.c_[s, center + rng.uniform(-1.4, 1.4, k), np.full(k, -LIDAR_H)])
    # головки рельсов (≈ 0.17 м над полотном)
    for side in (-1.0, 1.0):
        s = rng.uniform(3.0, length, k // 2)
        lat = center + side * 0.76 + rng.uniform(-0.03, 0.03, k // 2)
        parts.append(np.c_[s, lat, np.full(k // 2, -LIDAR_H + 0.17)])
    # стены
    for side in (-1.0, 1.0):
        s = rng.uniform(3.0, length, k)
        z = rng.uniform(-LIDAR_H, 3.0, k)
        parts.append(np.c_[s, np.full(k, center + side * 2.2), z])
    if box is not None:
        bs, blat, size, bh = box
        m = 400
        parts.append(
            np.c_[
                np.full(m, bs),
                center + blat + rng.uniform(-size / 2, size / 2, m),
                -LIDAR_H + 0.17 + rng.uniform(0.05, bh, m),
            ]
        )
    xyz = np.concatenate(parts).astype(np.float32)
    xyz[:, 2] += rng.normal(0, 0.005, len(xyz)).astype(np.float32)
    elev = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    ring = np.clip(((elev + 25.0) / 0.4).astype(np.int64), 0, 127)
    inten = rng.uniform(0, 255, len(xyz)).astype(np.float32)
    return xyz, ring, inten


def to_message_axes(xyz: np.ndarray, forward: str = "-y") -> np.ndarray:
    """Нормализованные оси → оси сообщения для заданного направления «вперёд»."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if forward == "x":
        mx, my = x, y
    elif forward == "-x":
        mx, my = -x, -y
    elif forward == "y":
        mx, my = -y, x
    elif forward == "-y":
        mx, my = y, -x
    else:
        raise ValueError(forward)
    return np.stack([mx, my, z], 1).astype(np.float32)
