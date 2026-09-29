"""Геометрия маркеров без ROS: коридор габарита, рамки объектов, подписи, цвета.

Общая часть для ноды (messages.py собирает из неё visualization_msgs) и для стенда
(bench/to_mcap.py пишет те же маркеры в MCAP без ROS). Всё — в осях и frame_id
входного облака.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from metro_obstacle.geometry import corridor_edges, yaw_at

RGBA = tuple[float, float, float, float]

RED: RGBA = (1.0, 0.15, 0.1, 0.9)
YELLOW: RGBA = (1.0, 0.85, 0.1, 0.8)
GREEN: RGBA = (0.2, 0.9, 0.3, 0.9)
WHITE: RGBA = (1.0, 1.0, 1.0, 1.0)

LEVEL_COLORS: dict[int, RGBA] = {0: GREEN, 1: YELLOW, 2: RED}
MIN_BOX = 0.1  # м — чтобы плоский объект был виден в просмотрщике
SECTION_STEP = 10.0  # м между поперечными рамками габарита


@dataclass
class CorridorLines:
    """Коридор габарита: кромки и поперечные П-рамки.

    edges — 4 ломаные (K, 3): левая и правая кромки на уровне оси (головок рельсов),
    затем они же, поднятые на высоту габарита. sections — отрезки (M, 2, 3)
    поперечных рамок раз в SECTION_STEP м вдоль оси.
    """

    edges: list[np.ndarray]
    sections: np.ndarray


def sorted_obstacles(res: Any) -> list[Any]:
    """Объекты кадра, ближние первыми (порядок id маркеров и детекций)."""
    return sorted(res.obstacles, key=lambda o: o.distance)


def status_text(res: Any) -> str:
    """Подпись над лидаром. Только ASCII: RViz не рисует кириллицу в TEXT-маркерах."""
    if res.obstacle:
        return f"OBSTACLE {res.distance:.1f} m"
    if res.level == 1 and res.obstacles:
        return f"CANDIDATE {min(o.distance for o in res.obstacles):.1f} m"
    return "CLEAR"


def level_color(res: Any, default: RGBA = GREEN) -> RGBA:
    return LEVEL_COLORS.get(int(res.level), default)


def obstacle_color(o: Any) -> RGBA:
    """Красный — подтверждённый объект, жёлтый — кандидат."""
    return RED if o.confirmed else YELLOW


def box_yaw(res: Any, o: Any) -> float:
    """Курс рамки объекта — вдоль оси пути у объекта."""
    return yaw_at(res.corridor, o.position)


def box_size(o: Any) -> tuple[float, float, float]:
    # Высота рамки — вертикальный размер точек (z_extent), а не height (верх над опорой):
    # центр рамки — середина точек, и у висящего кабеля рамка иначе тянулась бы вниз и вверх.
    return (
        max(float(o.length), MIN_BOX),
        max(float(o.width), MIN_BOX),
        max(float(o.z_extent), MIN_BOX),
    )


def label_position(o: Any, lift: float) -> tuple[float, float, float]:
    """Точка подписи: над верхом рамки объекта на lift м."""
    x, y, z = (float(v) for v in o.position)
    return x, y, z + box_size(o)[2] / 2 + lift


def corridor_lines(res: Any, gauge_height: float) -> CorridorLines | None:
    """Кромки и рамки коридора габарита; None, если оси нет (меньше двух точек)."""
    corridor = np.asarray(res.corridor, np.float64).reshape(-1, 3)
    half = np.broadcast_to(np.asarray(res.corridor_half_width, np.float64), (len(corridor),))
    ok = np.isfinite(corridor).all(axis=1) & np.isfinite(half)
    corridor, half = corridor[ok], half[ok]
    left, right = corridor_edges(corridor, half)
    if len(left) == 0:
        return None
    up = np.array([0.0, 0.0, gauge_height])
    # Поперечные П-рамки раз в SECTION_STEP м: по ним глазом видно, как далеко
    # ось пути прослежена и где она поворачивает.
    steps = np.linalg.norm(np.diff(corridor, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    idx = np.searchsorted(s, np.arange(0.0, s[-1] + 1e-6, SECTION_STEP))
    sections = [
        (a, b)
        for i in np.unique(np.clip(idx, 0, len(s) - 1))
        for lb, rb in [(left[i], right[i])]
        for a, b in ((lb, lb + up), (lb + up, rb + up), (rb + up, rb), (rb, lb))
    ]
    return CorridorLines(
        edges=[left, right, left + up, right + up],
        sections=np.asarray(sections, np.float64).reshape(-1, 2, 3),
    )
