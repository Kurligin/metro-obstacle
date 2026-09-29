"""Фиктивное ядро с тем же API, что у `metro_obstacle_core`.

Нужно, чтобы проверять ROS-обвязку (топики, маркеры, лог, Docker) без настоящего
детектора. Алгоритм намеренно простой: прямой коридор |lat| < 1.05 м вдоль оси
«вперёд», пол — нижний перцентиль высот в ближней зоне, «препятствие» — первое
скопление точек в коридоре выше пола. Никаких претензий на качество.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

MODES = ("default", "geometry", "strict", "soft")
FORWARD_AXES = ("auto", "x", "-x", "y", "-y")

_HALF_WIDTH = 1.05  # полуширина габарита, м
_MIN_HEIGHT = 0.3  # ниже — считаем полотном/рельсами
_GAUGE_HEIGHT = 3.0
_MAX_RANGE = 60.0  # дальше прямой коридор уже не совпадает с кривым тоннелем
_MIN_POINTS = 5
_CONFIRM_FRAMES = 3
_MAX_POINTS = 150_000
_UP = np.array([0.0, 0.0, 1.0])


def forward_basis(axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Единичные векторы «вперёд» и «влево» в осях сообщения (z — вверх)."""
    sign = -1.0 if axis.startswith("-") else 1.0
    f = np.zeros(3)
    f["xy".index(axis[-1])] = sign
    return f, np.cross(_UP, f)


@dataclass
class Obstacle:
    distance: float
    lateral: float
    height: float
    length: float
    width: float
    points: int
    confidence: float
    confirmed: bool
    position: tuple[float, float, float]
    z_extent: float = 0.0  # вертикальный размер точек объекта, м


@dataclass
class FrameResult:
    obstacle: bool
    distance: float
    confidence: float
    level: int
    visible_range: float
    processing_ms: float
    mode: str
    calibrated: bool
    obstacles: list[Obstacle] = field(default_factory=list)
    corridor: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    corridor_half_width: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_json(self) -> dict:
        def num(v: float) -> float | None:
            return None if math.isnan(v) else round(float(v), 3)

        return {
            "obstacle": self.obstacle,
            "distance": num(self.distance),
            "confidence": num(self.confidence),
            "level": self.level,
            "visible_range": num(self.visible_range),
            "processing_ms": num(self.processing_ms),
            "mode": self.mode,
            "calibrated": self.calibrated,
            "obstacles": [
                {k: (list(v) if k == "position" else v) for k, v in asdict(o).items()}
                for o in self.obstacles
            ],
        }


class FakeDetector:
    def __init__(self, mode: str = "default", forward_axis: str = "auto") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if forward_axis not in FORWARD_AXES:
            raise ValueError(f"forward_axis must be one of {FORWARD_AXES}, got {forward_axis!r}")
        self.mode = mode
        self._axis = None if forward_axis == "auto" else forward_axis
        self._frames = 0
        self._streak = 0
        self._last_distance = math.nan

    @staticmethod
    def _guess_axis(p: np.ndarray) -> str:
        # Лидар стоит на лбу поезда: впереди тоннель, сзади вагон. Берём направление,
        # где больше всего точек на 10–60 м в узкой полосе.
        best, best_n = "x", -1
        for axis in FORWARD_AXES[1:]:
            f, left = forward_basis(axis)
            s, lat = p @ f, p @ left
            n = int(np.count_nonzero((s > 10) & (s < 60) & (np.abs(lat) < 3)))
            if n > best_n:
                best, best_n = axis, n
        return best

    def process(
        self,
        xyz: np.ndarray,
        ring: np.ndarray,
        intensity: np.ndarray | None,
        stamp: float,
    ) -> FrameResult:
        t0 = time.perf_counter()
        ok = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
        p = xyz[ok]
        # Заглушка не должна сама становиться узким местом на кадрах в 900k точек.
        if len(p) > _MAX_POINTS:
            p = p[:: -(-len(p) // _MAX_POINTS)]
        p = p.astype(np.float64)
        if self._axis is None:
            self._axis = self._guess_axis(p)
        f, left = forward_basis(self._axis)
        s, lat, z = p @ f, p @ left, p[:, 2]
        self._frames += 1

        near = (s > 5) & (s < 30) & (np.abs(lat) < 1.5)
        ground = float(np.percentile(z[near], 10)) if near.any() else float(np.min(z, initial=0))
        band = np.abs(lat) < 3
        seen = float(np.percentile(s[band & (s > 0)], 98)) if (band & (s > 0)).any() else 0.0
        visible = min(_MAX_RANGE, seen)

        h = z - ground
        cand = (
            (np.abs(lat) < _HALF_WIDTH)
            & (h > _MIN_HEIGHT)
            & (h < _GAUGE_HEIGHT)
            & (s > 3)
            & (s < visible)
        )
        obstacles: list[Obstacle] = []
        if np.count_nonzero(cand) >= _MIN_POINTS:
            cs, cl, ch = s[cand], lat[cand], h[cand]
            bins = np.bincount(np.floor(cs).astype(np.int64))
            pair = bins + np.concatenate([bins[1:], [0]])
            hits = np.flatnonzero(pair >= _MIN_POINTS)
            if len(hits):
                b = hits[0]
                m = (cs >= b) & (cs < b + 2)
                self._streak = self._streak + 1 if abs(b - self._last_distance) < 3 else 1
                self._last_distance = float(b)
                confirmed = self._streak >= _CONFIRM_FRAMES
                h_lo, h_hi = float(ch[m].min()), float(ch[m].max())
                centre_local = (
                    float(cs[m].mean()),
                    float(cl[m].mean()),
                    ground + (h_lo + h_hi) / 2,
                )
                centre = centre_local[0] * f + centre_local[1] * left + centre_local[2] * _UP
                obstacles.append(
                    Obstacle(
                        distance=float(cs[m].min()),
                        lateral=float(cl[m].mean()),
                        height=float(ch[m].max()),
                        length=float(np.ptp(cs[m])),
                        width=float(np.ptp(cl[m])),
                        points=int(np.count_nonzero(m)),
                        confidence=min(1.0, np.count_nonzero(m) / 50.0),
                        confirmed=confirmed,
                        position=(float(centre[0]), float(centre[1]), float(centre[2])),
                        z_extent=h_hi - h_lo,
                    )
                )
        if not obstacles:
            self._streak, self._last_distance = 0, math.nan

        confirmed_obs = [o for o in obstacles if o.confirmed]
        nearest = confirmed_obs[0] if confirmed_obs else None
        axis_s = np.arange(0.0, visible + 1e-6, 2.0)
        corridor = axis_s[:, None] * f + ground * _UP
        return FrameResult(
            obstacle=nearest is not None,
            distance=nearest.distance if nearest else math.nan,
            confidence=obstacles[0].confidence if obstacles else 0.0,
            level=2 if nearest else (1 if obstacles else 0),
            visible_range=visible,
            processing_ms=(time.perf_counter() - t0) * 1e3,
            mode=self.mode,
            calibrated=self._frames >= _CONFIRM_FRAMES,
            obstacles=obstacles,
            corridor=corridor.astype(np.float32),
            corridor_half_width=np.full(len(axis_s), _HALF_WIDTH, np.float32),
        )
