"""Ядро детектора посторонних объектов в габарите поезда метро по 3D-лидару.

Без ROS: на вход — точки кадра (оси сообщения), на выход — FrameResult.
Одна и та же библиотека используется ROS-нодой и стендом (bench), поэтому цифры
экспериментов относятся ровно к этому коду.
"""

from __future__ import annotations

from metro_obstacle_core.detector import Detector, FrameResult, Obstacle
from metro_obstacle_core.params import MODES, Params

__version__ = "0.1.0"

__all__ = ["MODES", "Detector", "FrameResult", "Obstacle", "Params", "warmup"]


def warmup() -> float:
    """Скомпилировать/загрузить из кэша все numba-ядра на синтетической сцене.

    Первый вызов после установки компилирует ядра (секунды), дальше они берутся
    из дискового кэша numba. Вызывать при старте ноды, до первого кадра, чтобы
    первый реальный кадр не ждал компиляции. Режимы default и geometry вместе проходят
    все ветви (раздвоенные лучи, отбраковка у края, классификатор). Возвращает
    затраченное время, с.
    """
    import time

    import numpy as np

    from metro_obstacle_core.synthetic import to_message_axes, tunnel_points

    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    for mode in ("default", "geometry"):
        det = Detector(mode=mode, forward_axis="-y")
        for k in range(3):
            xyz, ring, inten = tunnel_points(rng, box=(40.0 - k, 0.0, 0.5, 0.5))
            det.process(to_message_axes(xyz, "-y"), ring, inten, float(k) * 0.1)
    return time.perf_counter() - t0
