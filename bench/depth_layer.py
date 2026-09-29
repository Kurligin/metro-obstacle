"""Слой «скачки глубины»: независимый от модели полотна признак для дальней зоны.

«Картинка дальностей» — кольцо × азимут (0.1°), в пикселе ближнее отражение.
Предмет на пути — компактная группа пикселей, заметно БЛИЖЕ соседей по азимуту
(там дальше пол/стены) и соседа сверху (луч над предметом уходит дальше). Стена,
пол, свод дают гладкую картинку; кронштейны и кабели на стенах отсекает грубая
маска коридора (полуширина 1.5 м вокруг оси без сужения по дальности).

Слой не решает сам: в слиянии он только подтверждает дальние кандидаты ядра.
"""

from __future__ import annotations

import numpy as np

AZ_STEP = 0.1  # град
AZ_MIN, AZ_MAX = -65.0, 65.0
NCOL = int((AZ_MAX - AZ_MIN) / AZ_STEP)
SIDE = (4, 6, 8)  # соседи по азимуту, пикселей: дальше ширины предмета на 100 м
H_MIN = 0.35  # выше головок рельсов: рельсы под скользящим углом тоже «ближе соседей»


def range_image(xyz: np.ndarray, ring: np.ndarray):
    """xyz — нормализованные оси (x вперёд, y влево). → (R[128,NCOL], idx[128,NCOL])."""
    r = np.linalg.norm(xyz, axis=1)
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    col = ((az - AZ_MIN) / AZ_STEP).astype(np.int64)
    ok = (col >= 0) & (col < NCOL) & (r > 1.0)
    key = ring[ok].astype(np.int64) * NCOL + col[ok]
    rr = r[ok]
    idx_all = np.flatnonzero(ok)
    order = np.lexsort((rr, key))  # в пикселе — ближнее отражение первым
    k_s = key[order]
    first = np.r_[True, k_s[1:] != k_s[:-1]]
    R = np.full(128 * NCOL, np.inf, np.float32)
    I = np.full(128 * NCOL, -1, np.int64)
    R[k_s[first]] = rr[order][first]
    I[k_s[first]] = idx_all[order][first]
    return R.reshape(128, NCOL), I.reshape(128, NCOL)


def _ring_order(xyz, ring) -> np.ndarray:
    """Номера колец по возрастанию угла места (номер кольца ≠ порядок по высоте)."""
    el = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    med = np.full(128, np.nan)
    for q in np.unique(ring):
        med[q] = np.median(el[ring == q])
    med = np.where(np.isnan(med), np.inf, med)
    return np.argsort(med)


def depth_candidates(xyz: np.ndarray, ring: np.ndarray, axis, s_min: float = 40.0, jump_rel: float = 0.06, jump_abs: float = 1.5, min_pix: int = 2):
    """Кандидаты (s, lat, n_pix) по скачкам глубины внутри грубого коридора."""
    xs, ys, zs = axis
    if len(xs) < 3:
        return []
    R, I = range_image(xyz, ring)
    R = R[_ring_order(xyz, ring)]  # строки снизу вверх по углу места
    I = I[_ring_order(xyz, ring)]
    fin = np.isfinite(R)
    Rn = np.where(fin, R, np.inf)
    jump = jump_rel * Rn + jump_abs
    # соседи по азимуту: ±2..4 пикселя (соседний ±1 может быть тем же предметом)
    side = np.full_like(Rn, np.inf)
    for d in SIDE:
        lft = np.roll(Rn, d, axis=1)
        rgt = np.roll(Rn, -d, axis=1)
        side = np.minimum(side, np.maximum(lft, rgt))  # обе стороны дальше
    above = np.roll(Rn, -1, axis=0)
    above[-1] = np.inf
    above2 = np.roll(Rn, -2, axis=0)
    above2[-2:] = np.inf
    up = np.maximum(above, above2)
    fg = fin & (Rn < side - jump) & (Rn < up - jump)
    pix = np.argwhere(fg)
    if not len(pix):
        return []
    idx = I[fg]
    p = xyz[idx]
    inside = (p[:, 0] > s_min) & (p[:, 0] < xs[-1])
    lat = p[:, 1] - np.interp(p[:, 0], xs, ys)
    h = p[:, 2] - np.interp(p[:, 0], xs, zs)
    inside &= (np.abs(lat) < 1.5) & (h > H_MIN) & (h < 3.2)
    if not inside.any():
        return []
    s, lat = p[inside, 0], lat[inside]
    order = np.argsort(s)
    s, lat = s[order], lat[order]
    # группы вдоль s с разрывом > 1.5 м
    cuts = np.flatnonzero(np.diff(s) > 1.5) + 1
    out = []
    for g in np.split(np.arange(len(s)), cuts):
        if len(g) >= min_pix:
            out.append((float(s[g].min()), float(np.median(lat[g])), int(len(g))))
    return out
