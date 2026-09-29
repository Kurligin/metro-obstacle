"""Ядро решения (metro_obstacle_core) под интерфейс детектора стенда.

Конфиги «core:default», «core:geometry», «core:strict», «core:soft» гоняют сдаваемый код через тот
же evaluate, что и прототипы: цифры финального прогона относятся ровно к ядру.
Кадры кэша (x-вперёд, y-влево) переводятся обратно в оси сообщения (вперёд = −Y),
ось «вперёд» ядро определяет само (auto) — как в ROS-ноде по умолчанию.
"""

from __future__ import annotations

import numpy as np

from bench.detect import Config, Detection

PREFIX = "core:"
CORE_CONFIGS = [Config(f"{PREFIX}{m}", axis="walls") for m in ("default", "geometry", "strict", "soft")]


def is_core(name: str) -> bool:
    return name.startswith(PREFIX)


class CoreDetector:
    """process(pts) → (подтверждённые детекции, info) — как bench.detect.Detector."""

    def __init__(self, cfg: Config, forward_axis: str = "auto"):
        from metro_obstacle_core import Detector

        self.cfg = cfg
        self.core = Detector(mode=cfg.name[len(PREFIX) :], forward_axis=forward_axis)
        self.last = None

    def process(self, pts: np.ndarray, oracle_axis: object = None) -> tuple[list[Detection], dict]:
        xyz = np.stack([pts["y"], -pts["x"], pts["z"]], 1).astype(np.float32)
        res = self.core.process(xyz, pts["ring"], pts["i"], 0.0)
        self.last = res
        dets = [
            Detection(o.distance, o.lateral, o.height, o.points, o.distance)
            for o in res.obstacles
            if o.confirmed
        ]
        return dets, {"axis_len": res.visible_range, "core_ms": res.processing_ms}


def make_detector(cfg: Config):  # type: ignore[no-untyped-def]
    """Детектор по конфигу: ядро для «core:*», иначе прототип стенда."""
    from bench.detect import Detector

    return CoreDetector(cfg) if is_core(cfg.name) else Detector(cfg)
