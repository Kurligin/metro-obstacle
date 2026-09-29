"""Эталонная ось пути для стенда (офлайн, с визуальной проверкой).

Ближняя зона (до NEAR м): будущая траектория лидара — точна до 0.1–0.3 м.
Дальше: стены тоннеля. Для каждого бина по x ищем ближайшие к прогнозу оси
левую и правую стены в полосе высот над полотном, переводим каждую в положение
оси через смещение, измеренное в ближней зоне, и сглаживаем. Высота полотна —
нижние точки рядом с осью.
"""

from __future__ import annotations

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from bench.synth import track_frame

NEAR = 25.0
STEP = 2.0
BAND = (1.2, 2.6)  # полоса высот над головкой рельса: выше платформы, ниже свода


def _walls(xyz, x, c, z0):
    m = (np.abs(xyz[:, 0] - x) < STEP / 2) & (xyz[:, 2] > z0 + BAND[0]) & (xyz[:, 2] < z0 + BAND[1])
    y = xyz[m, 1] - c
    left = y[(y > 1.1) & (y < 5.0)]
    right = y[(y < -1.1) & (y > -5.0)]
    wl = c + np.percentile(left, 10) if len(left) >= 3 else np.nan
    wr = c + np.percentile(right, 90) if len(right) >= 3 else np.nan
    return wl, wr


def truth_axis(xyz: np.ndarray, poses: np.ndarray, k: int, lidar_h: float, xmax: float = 200.0):
    """Возвращает xs, ys, zs — ось пути (головка рельса) в системе кадра k, и флаг качества."""
    s, p, _ = track_frame(poses, k, lidar_h)
    near = s <= NEAR
    xs = np.arange(4.0, xmax, STEP)
    ys = np.full(len(xs), np.nan)
    zs = np.full(len(xs), np.nan)
    if near.sum() >= 3 and p[near, 0].max() > 10:
        ynear = np.interp(xs, p[near, 0], p[near, 1], right=np.nan)
        znear = np.interp(xs, p[near, 0], p[near, 2], right=np.nan)
    else:  # поезд стоит — ближняя зона вдоль оси x на высоте −lidar_h
        ynear = np.where(xs <= NEAR, 0.0, np.nan)
        znear = np.where(xs <= NEAR, -lidar_h, np.nan)
    offl, offr = [], []
    for i, x in enumerate(xs):
        if not np.isnan(ynear[i]):
            wl, wr = _walls(xyz, x, ynear[i], znear[i])
            if not np.isnan(wl):
                offl.append(wl - ynear[i])
            if not np.isnan(wr):
                offr.append(wr - ynear[i])
            ys[i], zs[i] = ynear[i], znear[i]
    ol = np.median(offl) if len(offl) >= 3 else np.nan
    orr = np.median(offr) if len(offr) >= 3 else np.nan
    # дальняя зона: трекинг по бинам, оценки от каждой стены отдельно
    est_l, est_r = np.full(len(xs), np.nan), np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        if not np.isnan(ys[i]):
            continue
        good = np.flatnonzero(~np.isnan(ys[:i]))
        if len(good) < 3:
            break
        g = good[-6:]
        c = np.polyval(np.polyfit(xs[g], ys[g], 1), x)
        zc = np.polyval(np.polyfit(xs[g], zs[g], 1), x)
        wl, wr = _walls(xyz, x, c, zc)
        if not np.isnan(wl) and not np.isnan(ol):
            est_l[i] = wl - ol
        if not np.isnan(wr) and not np.isnan(orr):
            est_r[i] = wr - orr
        cand = [e for e in (est_l[i], est_r[i]) if not np.isnan(e) and abs(e - c) < 0.8]
        if not cand:
            continue
        ys[i] = float(np.mean(cand))
        mg = (np.abs(xyz[:, 0] - x) < STEP / 2) & (np.abs(xyz[:, 1] - ys[i]) < 1.0) & (xyz[:, 2] < zc + 0.5)
        zs[i] = np.percentile(xyz[mg, 2], 20) if mg.sum() >= 3 else zc
    # глобальная гладкая кривая: RANSAC-кубика по ближней оси + оценкам стен
    nearm = ~np.isnan(ynear)
    X = np.r_[xs[nearm], xs, xs]
    Y = np.r_[ynear[nearm], est_l, est_r]
    W = np.r_[np.full(nearm.sum(), 3.0), np.ones(2 * len(xs))]
    v = ~np.isnan(Y)
    X, Y, W = X[v], Y[v], W[v]
    if len(X) >= 8:
        rng = np.random.default_rng(k)
        best, bestn = None, -1
        for _ in range(200):
            idx = rng.choice(len(X), 4, replace=False)
            if np.ptp(X[idx]) < 20:
                continue
            c3 = np.polyfit(X[idx], Y[idx], 3)
            inl = np.abs(np.polyval(c3, X) - Y) < 0.3
            n = W[inl].sum()
            if n > bestn:
                best, bestn = inl, n
        if best is not None and best.sum() >= 6:
            c3 = np.polyfit(X[best], Y[best], 3, w=W[best])
            xmax_ok = X[best].max()
            yfit = np.polyval(c3, xs)
            keep = xs <= xmax_ok
            ys = np.where(keep, yfit, np.nan)
            zs = np.where(keep, zs, np.nan)
    ok = ~np.isnan(ys)
    if ok.sum() < 3:
        return xs[:0], xs[:0], xs[:0]
    last = np.flatnonzero(ok)[-1]
    xs, ys, zs = xs[: last + 1], ys[: last + 1], zs[: last + 1]
    ok = ok[: last + 1]
    ys = np.interp(xs, xs[ok], ys[ok])
    zs = np.interp(xs, xs[ok], np.nan_to_num(zs[ok], nan=np.nanmedian(zs)))
    # сглаживание скользящим средним по 5 бинам (10 м)
    ker = np.ones(5) / 5
    ys = np.convolve(np.pad(ys, 2, mode="edge"), ker, mode="valid")
    zs = np.convolve(np.pad(zs, 2, mode="edge"), ker, mode="valid")
    return xs, ys, zs
