"""Модель полотна: профиль поперёк пути, опора — головки рельсов, поправка Δ(s) вдали."""

from __future__ import annotations

import numpy as np

from metro_obstacle_core import kernels as K

LAT_BINS = np.linspace(-1.3, 1.3, 53)
_Q_UPDATE = np.array([97.0, 30.0])
_Q_DELTA = np.array([80.0])


class FloorModel:
    """Профиль полотна поперёк пути (верхняя огибающая «нормы») + сдвиг по s.

    Профиль U(lat) обучается на ближней зоне (там плотно) с EMA по кадрам.
    Вдали вычитается локальный сдвиг Δ(s): верх рельсов в 5-метровых бинах
    относительно выученного уровня головок (уклоны, неровность, ошибка оси по z).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        fast: float | None = None,
        dil_k: float = 0.0,
        delta_clip: float = 0.3,
    ) -> None:
        self.delta_clip = delta_clip  # предел поправки Δ(s), м
        self.dil_k = dil_k  # доп. расширение профиля вбок, м на метр дальности
        self.alpha = alpha  # вес нового кадра в EMA
        self.fast = fast  # порог «профиль сменился» (м); None — выкл.
        self.n_updates = 0
        self.U: np.ndarray | None = None
        self.L: np.ndarray | None = None
        self.rail_head = 0.0
        self.last_delta: tuple[np.ndarray, np.ndarray] | None = None  # (s-бины, Δ) — для выхода

    def update(self, lat: np.ndarray, h: np.ndarray, s: np.ndarray) -> None:
        m = (s > 4) & (s < 25) & (np.abs(lat) < 1.3) & (h < 0.6) & (h > -0.8)
        if m.sum() < 200:
            return
        b = np.digitize(lat[m], LAT_BINS) - 1
        U, L = K.group_percentiles(h[m], b, len(LAT_BINS) - 1, _Q_UPDATE, 10)
        self.n_updates += 1
        if self.U is None or self.L is None:
            self.U, self.L = U, L
        else:
            a = self.alpha
            if self.fast is not None:
                # смена типа тоннеля: профиль ближней зоны заметно другой — не ждать
                # десятки кадров EMA, а переучиться за несколько
                both = ~np.isnan(U) & ~np.isnan(self.U)
                if both.sum() >= 10 and np.median(np.abs(U[both] - self.U[both])) > self.fast:
                    a = 0.5
            for arr, new in ((self.U, U), (self.L, L)):
                ok = ~np.isnan(new)
                fresh = ok & np.isnan(arr)
                arr[fresh] = new[fresh]
                both = ok & ~fresh
                keep = 0.9 if a == 0.1 else 1.0 - a  # 0.9 литералом: бит-в-бит со стендом
                arr[both] = keep * arr[both] + a * new[both]

    def reference(self, extra: int = 0) -> np.ndarray:
        """Опорный профиль: max(профиль нормы с расширением ±(0.15 м + extra бинов),
        уровень головок)."""
        assert self.U is not None
        U = np.nan_to_num(self.U, nan=np.nanmax(self.U))
        w = 3 + extra
        dil = np.array([U[max(j - w, 0) : j + w + 1].max() for j in range(len(U))])
        c = 0.5 * (LAT_BINS[:-1] + LAT_BINS[1:])
        rail = (np.abs(c) > 0.6) & (np.abs(c) < 0.95)
        self.rail_head = float(np.max(U[rail]))
        return np.maximum(dil, self.rail_head)

    def height(self, lat: np.ndarray, h: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Высота над опорным профилем с поправкой Δ(s) по рельсам вдали."""
        if self.U is None:
            return h
        ref = self.reference()
        b = np.clip(np.digitize(lat, LAT_BINS) - 1, 0, len(ref) - 1)
        sb = (s // 5).astype(int)
        band = (
            (np.abs(lat) > 0.6)
            & (np.abs(lat) < 0.95)
            & (h > self.rail_head - 0.4)
            & (h < self.rail_head + 0.4)
        )
        ub = np.unique(sb)
        grp = np.searchsorted(ub, sb[band])
        delta = K.group_percentiles(h[band], grp, len(ub), _Q_DELTA, 3)[0] - self.rail_head
        ok = ~np.isnan(delta)
        if ok.sum() >= 2:
            dc = self.delta_clip
            delta = np.interp(ub, ub[ok], np.clip(delta[ok], -dc, dc))
        else:
            delta = np.zeros(len(ub))
        # без точек у оси бинов нет: поправку для коридора не обновляем «пустой»
        self.last_delta = (ub, delta) if len(ub) else None
        d = delta[np.searchsorted(ub, sb)]
        if self.dil_k <= 0:
            return h - d - ref[b]
        # вдали ось ошибается вбок сильнее — гребни профиля (края лотка, рельсы)
        # могут попасть в «ячейку лотка»; расширяем профиль пропорционально дальности
        step = float(LAT_BINS[1] - LAT_BINS[0])
        extra = np.minimum((self.dil_k * s / step).astype(np.int64), 12)
        out = np.empty_like(h)
        for e in np.unique(extra):
            m = extra == e
            out[m] = h[m] - d[m] - (ref if e == 0 else self.reference(int(e)))[b[m]]
        return out
