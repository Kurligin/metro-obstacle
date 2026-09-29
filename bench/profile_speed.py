"""Скорость без ICP: сдвиг продольного профиля стен между соседними кадрами.

Профиль — гистограмма точек стен (|y| > 1.3 м) вдоль x с весом по интенсивности
и по высоте (разные полосы высоты — разные каналы). Сдвиг ищется
кросс-корреляцией в окне 0..max_shift с субпиксельной параболой.
"""

from __future__ import annotations

import sys

import numpy as np

from bench.data import Bag

BIN = 0.05
X0, X1 = 4.0, 40.0
BANDS = [(-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 3.0)]


def profile(fr) -> np.ndarray:
    p = fr.pts
    m = (np.abs(p["y"]) > 1.3) & (p["x"] > X0) & (p["x"] < X1)
    nb = int((X1 - X0) / BIN)
    out = []
    for side in (-1, 1):
        for z0, z1 in BANDS:
            mm = m & (np.sign(p["y"]) == side) & (p["z"] > z0) & (p["z"] < z1)
            # средняя интенсивность в ячейке: текстура стены движется с миром,
            # а густота точек (рисунок колец лидара) привязана к сенсору
            cnt, _ = np.histogram(p["x"][mm], bins=nb, range=(X0, X1))
            s, _ = np.histogram(p["x"][mm], bins=nb, range=(X0, X1), weights=p["i"][mm].astype(float))
            ok = cnt > 0
            h = np.zeros(nb)
            if ok.sum() > 10:
                h = np.interp(np.arange(nb), np.flatnonzero(ok), s[ok] / cnt[ok])
            h = h - np.convolve(h, np.ones(41) / 41, mode="same")
            out.append(h / (np.linalg.norm(h) + 1e-9))
    return np.array(out)


def shift(a: np.ndarray, b: np.ndarray, max_shift: float = 3.0) -> tuple[float, float]:
    """Насколько b сдвинут к лидару относительно a (поезд проехал вперёд), м."""
    ks = np.arange(0, int(max_shift / BIN) + 1)
    n = a.shape[1]
    score = np.array([np.sum(a[:, k:] * b[:, : n - k]) for k in ks])
    k = int(np.argmax(score))
    if 0 < k < len(ks) - 1:
        y0, y1, y2 = score[k - 1 : k + 2]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    srt = np.sort(score)
    return k * BIN, float(srt[-1] - np.median(score))


def run(name: str):
    bag = Bag(name)
    prof = [profile(bag[k]) for k in range(len(bag))]
    d = np.array([shift(prof[k], prof[k + 1])[0] for k in range(len(bag) - 1)])
    v = d / np.diff(bag.stamps) * 3.6
    return v


if __name__ == "__main__":
    for n in sys.argv[1:]:
        v = run(n)
        print(n, "км/ч p10/p50/p90:", np.percentile(v, [10, 50, 90]).round(1), "путь м:", round(float((v / 3.6 * 0.1).sum()), 1))
        np.save(f"out/odom/{n}.profile_speed.npy", v)
