"""Пройденный за кадр путь без ICP: сдвиг продольного профиля интенсивности стен.

Профиль — средняя интенсивность точек стен (|y| > 1.3 м) в бинах 5 см вдоль x,
8 каналов (сторона × полоса высоты). Текстура стены движется с миром, а густота
точек (рисунок колец лидара) привязана к сенсору — поэтому средняя, а не сумма.
Одометрия по облакам (KISS-ICP) в однородном тоннеле вырождается, а этот сдвиг
работает на всех bag.
"""

from __future__ import annotations

import numpy as np

from metro_obstacle_core import kernels as K

BIN = 0.05
X0, X1 = 4.0, 40.0
NB = int((X1 - X0) / BIN)
# края бинов как у np.histogram для float32-данных (linspace в float32)
_EDGES = np.linspace(X0, X1, NB + 1, dtype=np.float32)
_KERNEL = np.ones(41) / 41


def profile(x: np.ndarray, y: np.ndarray, z: np.ndarray, inten: np.ndarray) -> np.ndarray:
    """Профиль стен (8, NB): центрированный скользящим средним 2 м и нормированный."""
    cnt, sm = K.profile_hist(x, y, z, inten, _EDGES)
    out = np.zeros((8, NB))
    ar = np.arange(NB)
    for ch in range(8):
        ok = cnt[ch] > 0
        h = np.zeros(NB)
        if ok.sum() > 10:
            h = np.interp(ar, np.flatnonzero(ok), sm[ch][ok] / cnt[ch][ok])
        h = h - np.convolve(h, _KERNEL, mode="same")
        out[ch] = h / (np.linalg.norm(h) + 1e-9)
    return out


class SpeedEstimator:
    """Онлайн-сдвиг за кадр: максимум корреляции профилей в окне 0–3 м.

    Штраф за скачок относительно прошлого сдвига: изменение до 0.3 м за кадр
    бесплатно (шум оценки), дальше — мягкий штраф 0.05 z на бин. Первые три кадра —
    без штрафа, чтобы захватить реальную скорость. legacy=True — поведение прототипа
    стенда (жёсткий штраф от нуля; на движущемся поезде «залипает» на нуле) — только
    для сверки со стендом.
    """

    def __init__(self, legacy: bool = False) -> None:
        self.prev: np.ndarray | None = None
        self.shift = 0.0
        self.legacy = legacy
        self.n = 0

    def step(self, pr: np.ndarray) -> float:
        if self.prev is None:
            self.prev = pr
            return 0.0
        n = pr.shape[1]
        ks = np.arange(0, int(3.0 / BIN) + 1)
        sc = np.array([np.sum(self.prev[:, k:] * pr[:, : n - k]) for k in ks])
        sc = (sc - sc.mean()) / (sc.std() + 1e-9)
        prev_k = self.shift / BIN
        if self.legacy:
            sc = sc - 0.5 * np.maximum(np.abs(ks - prev_k) - 1.5, 0)
        elif self.n >= 3:
            sc = sc - 0.05 * np.maximum(np.abs(ks - prev_k) - 6, 0)
        k = int(np.argmax(sc))
        self.prev = pr
        self.shift = k * BIN
        self.n += 1
        return self.shift
