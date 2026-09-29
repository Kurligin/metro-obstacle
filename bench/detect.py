"""Кандидаты детектора: ось коридора × модель полотна × накопление кадров.

Все кандидаты работают онлайн (только текущий и прошлые кадры), кроме Oracle —
он берёт эталонную ось (будущая траектория) и служит верхней границей.

Координаты пути: s ≈ x (вдоль), lat = y − y_axis(x), h = z − z_axis(x).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

GAUGE_HALF_W = 1.05
GAUGE_H = 3.0
STEP = 2.0
XMAX = 200.0
S_MIN = 5.0  # ближе — мёртвая зона у лобовой части поезда


# ----------------------------------------------------------------- утилиты


class XIndex:
    """Точки, отсортированные по x, для быстрых срезов по бинам."""

    def __init__(self, xyz: np.ndarray, extra: dict[str, np.ndarray] | None = None):
        o = np.argsort(xyz[:, 0], kind="stable")
        self.xyz = xyz[o]
        self.order = o
        self.extra = {k: v[o] for k, v in (extra or {}).items()}

    def slab(self, x0: float, x1: float) -> slice:
        xs = self.xyz[:, 0]
        return slice(int(np.searchsorted(xs, x0)), int(np.searchsorted(xs, x1)))


def ground_plane(xyz: np.ndarray, rng=np.random.default_rng(0)):
    """Плоскость полотна в ближней зоне (RANSAC). n·p + d = 0, n вверх."""
    m = (xyz[:, 0] > 3) & (xyz[:, 0] < 20) & (np.abs(xyz[:, 1]) < 2.0)
    p = xyz[m]
    if len(p) < 50:
        return np.array([0, 0, 1.0]), 1.35
    p = p[p[:, 2] < np.percentile(p[:, 2], 50)]
    best, bn = (np.array([0, 0, 1.0]), -np.median(p[:, 2])), -1
    for _ in range(100):
        s = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn * (1 if n[2] > 0 else -1)
        if n[2] < 0.97:
            continue
        d = -n @ s[0]
        c = np.sum(np.abs(p @ n + d) < 0.06)
        if c > bn:
            best, bn = (n, d), c
    return best


def robust_cubic(X, Y, W, rng, tol=0.3, min_span=20.0):
    if len(X) < 8:
        return None, None
    best, bestn = None, -1
    for _ in range(150):
        idx = rng.choice(len(X), 4, replace=False)
        if np.ptp(X[idx]) < min_span:
            continue
        with np.errstate(all="ignore"):
            c = np.polyfit(X[idx], Y[idx], 3)
        inl = np.abs(np.polyval(c, X) - Y) < tol
        n = W[inl].sum()
        if n > bestn:
            best, bestn = inl, n
    if best is None or best.sum() < 6:
        return None, None
    with np.errstate(all="ignore"):
        c = np.polyfit(X[best], Y[best], 3, w=W[best])
    return c, X[best].max()


# ----------------------------------------------------------------- оси коридора


@dataclass
class Calib:
    """Автокалибровка: плоскость полотна в системе лидара."""

    n: np.ndarray
    d: float

    @property
    def height(self) -> float:
        return float(self.d)

    def z_floor(self, x, y):
        # n·p + d = 0 → z = −(d + nx x + ny y)/nz
        return -(self.d + self.n[0] * x + self.n[1] * y) / self.n[2]


def axis_straight(ix: XIndex, cal: Calib, state: dict):
    xs = np.arange(4.0, XMAX, STEP)
    return xs, np.zeros_like(xs), cal.z_floor(xs, 0 * xs)


def axis_near_straight(ix: XIndex, cal: Calib, state: dict):
    """Ближняя ось по рельсам, дальше — прямая по касательной (без стен)."""
    xs = np.arange(4.0, XMAX, STEP)
    _near_axis(cal, xs, ix, state)
    c = state.get("near_coef", np.zeros(3))
    ys = np.where(xs <= NEAR_RAILS, np.polyval(c, xs), np.polyval(c, NEAR_RAILS) + np.polyval(np.polyder(c), NEAR_RAILS) * (xs - NEAR_RAILS))
    return xs, ys, _floor_along(ix, xs, ys, cal)


def _rail_center(ix: "XIndex", x: float, c: float, zc: float, search: float = 0.35):
    """Центр пути в бине: шаблон «пара головок рельсов через 1.52 м».

    Перебираем центр в ±search от прогноза; счёт = число точек в узких полосах
    ±4 см вокруг обеих головок на высоте 0.08–0.4 м над полотном, минус точки
    между рельсами на той же высоте (там должно быть пусто). Нужны обе головки.
    """
    sl = ix.slab(x - STEP / 2, x + STEP / 2)
    p = ix.xyz[sl]
    h = p[:, 2] - zc
    m = (h > 0.08) & (h < 0.4) & (np.abs(p[:, 1] - c) < RAIL_LAT + search + 0.1)
    y = p[m, 1]
    if len(y) < 4:
        return np.nan
    # точки выше 0.5 м — над путём габарит должен быть свободен
    yhi = p[(h > 0.5) & (h < 2.0) & (np.abs(p[:, 1] - c) < 1.0 + search), 1]
    best, bs = np.nan, 0
    for cc in np.arange(c - search, c + search + 1e-9, 0.02):
        l = np.sum(np.abs(y - (cc + RAIL_LAT)) < 0.04)
        r = np.sum(np.abs(y - (cc - RAIL_LAT)) < 0.04)
        mid = np.sum(np.abs(y - cc) < RAIL_LAT - 0.12)
        pen = np.sum(np.abs(yhi - cc) < 1.0)
        sc = min(l, r) * 2 - mid - 3 * pen
        if l >= 1 and r >= 1 and sc > bs:
            best, bs = cc, sc
    return float(best) if bs >= 2 else np.nan


NEAR_RAILS = 30.0


def _near_axis(cal: Calib, xs, ix: "XIndex | None" = None, state: dict | None = None):
    """Ближняя ось: парабола по рельсам в 4–30 м, сглаженная во времени.

    Без неё ось лидара считается осью пути — это неверно при повороте лидара
    относительно пути и в кривых (смещение 0.3 м уже на 10 м).
    """
    coef = None if state is None else state.get("near_coef")
    if ix is not None:
        prior = coef if coef is not None else np.zeros(3)
        X, Y = [], []
        for x in np.arange(5.0, NEAR_RAILS, STEP):
            # прогноз: по уже найденным центрам (слежение), иначе — прошлый кадр
            if len(X) >= 3:
                deg = 2 if len(X) >= 5 else 1
                c = np.polyval(np.polyfit(X[-6:], Y[-6:], deg), x)
            else:
                c = np.polyval(prior, x)
            yc = _rail_center(ix, x, c, cal.z_floor(x, c))
            if not np.isnan(yc) and abs(yc - c) < 0.4:
                X.append(x)
                Y.append(yc)
        if len(X) >= 5:
            X, Y = np.array(X), np.array(Y)
            new = np.polyfit(X, Y, 2)
            r = np.abs(np.polyval(new, X) - Y)
            if (r < 0.1).sum() >= 5:
                new = np.polyfit(X[r < 0.1], Y[r < 0.1], 2)
            coef = new if coef is None else 0.5 * coef + 0.5 * new
            if state is not None:
                state["near_coef"] = coef
    if coef is None:
        coef = np.zeros(3)
    ys = np.polyval(coef, xs)
    return ys, cal.z_floor(xs, ys)


BAND = (1.2, 2.6)
NEAR = 20.0


def _walls(ix: XIndex, x, c, z0):
    sl = ix.slab(x - STEP / 2, x + STEP / 2)
    p = ix.xyz[sl]
    m = (p[:, 2] > z0 + BAND[0]) & (p[:, 2] < z0 + BAND[1])
    y = p[m, 1] - c
    left = y[(y > 1.1) & (y < 5.0)]
    right = y[(y < -1.1) & (y > -5.0)]
    wl = c + np.percentile(left, 10) if len(left) >= 3 else np.nan
    wr = c + np.percentile(right, 90) if len(right) >= 3 else np.nan
    return wl, wr


def axis_walls(ix: XIndex, cal: Calib, state: dict):
    """A: ближние NEAR м — прямо по оси лидара; дальше — стены + смещения из ближней зоны."""
    rng = state.setdefault("rng", np.random.default_rng(0))
    xs = np.arange(4.0, XMAX, STEP)
    ys = np.full(len(xs), np.nan)
    zs = np.full(len(xs), np.nan)
    yn, zn = _near_axis(cal, xs, ix, state)
    near = xs <= NEAR
    ys[near], zs[near] = yn[near], zn[near]
    offl, offr = [], []
    for i in np.flatnonzero(near):
        wl, wr = _walls(ix, xs[i], ys[i], zs[i])
        offl.append(wl - ys[i])
        offr.append(wr - ys[i])
    # смещения стен сглаживаем во времени (стена у пути почти не меняется)
    ol = np.nanmedian(offl) if np.sum(~np.isnan(offl)) >= 3 else np.nan
    orr = np.nanmedian(offr) if np.sum(~np.isnan(offr)) >= 3 else np.nan
    for key, v in (("ol", ol), ("or", orr)):
        prev = state.get(key)
        if not np.isnan(v):
            state[key] = v if prev is None or np.isnan(prev) else 0.8 * prev + 0.2 * v
    ol, orr = state.get("ol", np.nan), state.get("or", np.nan)
    el, er = np.full(len(xs), np.nan), np.full(len(xs), np.nan)
    for i in np.flatnonzero(~near):
        good = np.flatnonzero(~np.isnan(ys[:i]))
        if len(good) < 3:
            break
        g = good[-6:]
        c = np.polyval(np.polyfit(xs[g], ys[g], 1), xs[i])
        zc = np.polyval(np.polyfit(xs[g], zs[g], 1), xs[i])
        wl, wr = _walls(ix, xs[i], c, zc)
        if not np.isnan(ol) and not np.isnan(wl):
            el[i] = wl - ol
        if not np.isnan(orr) and not np.isnan(wr):
            er[i] = wr - orr
        cand = [e for e in (el[i], er[i]) if not np.isnan(e) and abs(e - c) < 0.8]
        if cand:
            ys[i] = float(np.mean(cand))
            zs[i] = zc
    X = np.r_[xs[near], xs, xs]
    Y = np.r_[ys[near], el, er]
    W = np.r_[np.full(near.sum(), 3.0), np.ones(2 * len(xs))]
    v = ~np.isnan(Y)
    c3, xok = robust_cubic(X[v], Y[v], W[v], rng)
    if c3 is not None:
        ext = state.get("axis_ext", 0.0)
        state["x_ok"] = float(xok)
        if ext > 0:
            # продление за последнюю надёжную точку: по касательной и кривизне в xok
            d1 = np.polyval(np.polyder(c3), xok)
            d2 = np.polyval(np.polyder(c3, 2), xok)
            y0 = np.polyval(c3, xok)
            dx = xs - xok
            yext = y0 + d1 * dx + 0.5 * d2 * dx * dx
            ys = np.where(xs <= xok, np.polyval(c3, xs), np.where(xs <= xok + ext, yext, np.nan))
        else:
            keep = xs <= xok
            ys = np.where(keep, np.polyval(c3, xs), np.nan)
    ok = ~np.isnan(ys)
    xs, ys = xs[ok], ys[ok]
    zs = _floor_along(ix, xs, ys, cal)
    return xs, ys, zs


RAIL_LAT = 0.76


def axis_kalman(ix: XIndex, cal: Calib, state: dict):
    """K: фильтр Калмана вдоль пути. Состояние [y, y', y''] (смещение, наклон, кривизна).

    Старт — ближняя ось по рельсам. Измерения в каждом бине: левая стена, правая
    стена (через их смещения, выученные в ближней зоне) и рельсы. Стробирование
    3σ отбрасывает колонны, ниши, переходы типа тоннеля. σ(x) оси возвращается
    в state["sigma"] — по ней сужается рабочий габарит и обрывается ось.
    """
    xs = np.arange(4.0, XMAX, STEP)
    yn, zn = _near_axis(cal, xs, ix, state)
    c = state.get("near_coef", np.zeros(3))
    x0 = NEAR_RAILS
    # смещения стен относительно ближней оси (сглажены во времени)
    offl, offr = [], []
    for x in np.arange(5.0, NEAR_RAILS, STEP):
        yc = np.polyval(c, x)
        wl, wr = _walls(ix, x, yc, cal.z_floor(x, yc))
        offl.append(wl - yc)
        offr.append(wr - yc)
    for key, arr in (("ol", offl), ("or", offr)):
        v = np.array(arr, float)
        v = v[~np.isnan(v)]
        if len(v) >= 3:
            med = float(np.median(v))
            prev = state.get(key)
            state[key] = med if prev is None else 0.8 * prev + 0.2 * med
    ol, orr = state.get("ol"), state.get("or")
    # старт: положение и направление ближней оси; кривизну из короткой параболы
    # не берём (шумная) — ноль с широкой неопределённостью, её уточнят стены
    X = np.array([np.polyval(c, x0), np.polyval(np.polyder(c), x0), 0.0])
    P = np.diag([0.03**2, 0.005**2, 0.003**2])
    q = 3e-7  # шум изменения кривизны на метр (метро: радиусы ≥ 300 м, плавные переходы)
    ys = np.full(len(xs), np.nan)
    sig = np.full(len(xs), np.nan)
    near = xs <= x0
    ys[near] = np.polyval(c, xs[near])
    sig[near] = 0.05
    miss = 0
    for i in np.flatnonzero(~near):
        dx = STEP
        F = np.array([[1, dx, dx * dx / 2], [0, 1, dx], [0, 0, 1]])
        X = F @ X
        P = F @ P @ F.T + np.diag([0, 0, q * dx])
        zc = cal.z_floor(xs[i], X[0])
        meas = []
        wl, wr = _walls(ix, xs[i], X[0], zc)
        if ol is not None and not np.isnan(wl):
            meas.append((wl - ol, 0.25**2))
        if orr is not None and not np.isnan(wr):
            meas.append((wr - orr, 0.25**2))
        yr = _rail_center(ix, xs[i], X[0], zc)
        if not np.isnan(yr):
            meas.append((yr, 0.08**2))
        used = 0
        for z, R in meas:
            S = P[0, 0] + R
            inn = z - X[0]
            if inn * inn > 9 * S:
                continue
            K = P[:, 0] / S
            X = X + K * inn
            P = P - np.outer(K, P[0, :])
            used += 1
        miss = 0 if used else miss + 1
        ys[i] = X[0]
        sig[i] = np.sqrt(P[0, 0])
        if sig[i] > 0.45 or miss > 8:
            break
    ok = ~np.isnan(ys)
    xs, ys, sig = xs[ok], ys[ok], sig[ok]
    state["sigma"] = (xs, sig)
    return xs, ys, _floor_along(ix, xs, ys, cal)




def axis_rails(ix: XIndex, cal: Calib, state: dict):
    """B: ось по головкам рельсов — пики высоты на ±0.76 м от прогноза оси."""
    rng = state.setdefault("rng", np.random.default_rng(0))
    xs = np.arange(4.0, XMAX, STEP)
    ys = np.full(len(xs), np.nan)
    yn, zn = _near_axis(cal, xs, ix, state)
    miss = 0
    for i, x in enumerate(xs):
        good = np.flatnonzero(~np.isnan(ys[:i]))
        if len(good) >= 3:
            g = good[-6:]
            c = np.polyval(np.polyfit(xs[g], ys[g], 1), x)
        else:
            c = yn[i]
        zc = cal.z_floor(x, c)
        sl = ix.slab(x - STEP / 2, x + STEP / 2)
        p = ix.xyz[sl]
        lat = p[:, 1] - c
        h = p[:, 2] - zc
        est = []
        for side in (-1, 1):
            m = (np.abs(lat - side * RAIL_LAT) < 0.25) & (h > 0.05) & (h < 0.45)
            if m.sum() >= 2:
                # головка рельса — самые высокие точки узкой полосы
                top = h[m] > np.percentile(h[m], 60)
                est.append(np.median(lat[m][top]) - side * RAIL_LAT + c)
        if est:
            ys[i] = float(np.mean(est))
            miss = 0
        else:
            miss += 1
            if miss > 5 and x > NEAR:
                break
    v = ~np.isnan(ys)
    c3, xok = robust_cubic(xs[v], ys[v], np.ones(v.sum()), rng, tol=0.2)
    if c3 is not None:
        ys = np.where(xs <= xok, np.polyval(c3, xs), np.nan)
    ok = ~np.isnan(ys)
    xs, ys = xs[ok], ys[ok]
    return xs, ys, _floor_along(ix, xs, ys, cal)


def _floor_along(ix: XIndex, xs, ys, cal: Calib):
    """Высота полотна вдоль оси: последовательное слежение от ближней зоны.

    В бине — 30-й перцентиль точек у оси в окне ±0.35 м от прогноза (прогноз —
    по последним бинам, уклон ограничен 6%). Так пол у гермозатвора или на
    уклоне не уходит из окна, а стены/объекты не тянут его вверх.
    """
    zs = np.full(len(xs), np.nan)
    z_prev = None
    slope = 0.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        pred = cal.z_floor(x, y) if z_prev is None else z_prev + slope * STEP
        sl = ix.slab(x - STEP, x + STEP)
        p = ix.xyz[sl]
        m = (np.abs(p[:, 1] - y) < 1.2) & (np.abs(p[:, 2] - pred) < 0.35)
        if m.sum() >= 5:
            z = float(np.percentile(p[m, 2], 30))
            if z_prev is not None:
                slope = float(np.clip(0.7 * slope + 0.3 * (z - z_prev) / STEP, -0.06, 0.06))
            zs[i] = z
            z_prev = z
        else:
            zs[i] = pred
            z_prev = pred
    if len(zs) >= 5:
        zs = np.convolve(np.pad(zs, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return zs


# ----------------------------------------------------------------- модель полотна

LAT_BINS = np.linspace(-1.3, 1.3, 53)


class FloorModel:
    """D: профиль полотна поперёк пути (верхняя огибающая «нормы») + сдвиг по s.

    Профиль U(lat) обучается на ближней зоне (плотно) с EMA по кадрам.
    Вдали вычитается локальный сдвиг Δ(s) — медиана невязки нижних точек в бине.
    """

    def __init__(self, fixed: float | None = None):
        self.fixed = fixed  # если задан — простой порог по высоте над осью
        self.U: np.ndarray | None = None
        self.L: np.ndarray | None = None

    def update(self, lat, h, s):
        m = (s > 4) & (s < 25) & (np.abs(lat) < 1.3) & (h < 0.6) & (h > -0.8)
        if m.sum() < 200:
            return
        b = np.digitize(lat[m], LAT_BINS) - 1
        U = np.full(len(LAT_BINS) - 1, np.nan)
        L = np.full(len(LAT_BINS) - 1, np.nan)
        hm = h[m]
        for j in range(len(U)):
            v = hm[b == j]
            if len(v) >= 10:
                U[j] = np.percentile(v, 97)
                L[j] = np.percentile(v, 30)
        if self.U is None:
            self.U, self.L = U, L
        else:
            for arr, new in ((self.U, U), (self.L, L)):
                ok = ~np.isnan(new)
                fresh = ok & np.isnan(arr)
                arr[fresh] = new[fresh]
                both = ok & ~fresh
                arr[both] = 0.9 * arr[both] + 0.1 * new[both]

    def reference(self):
        """Опорный профиль: max(профиль нормы с расширением ±0.15 м, уровень головок рельсов)."""
        U = np.nan_to_num(self.U, nan=np.nanmax(self.U))
        dil = np.array([U[max(j - 3, 0) : j + 4].max() for j in range(len(U))])
        c = 0.5 * (LAT_BINS[:-1] + LAT_BINS[1:])
        rail = (np.abs(c) > 0.6) & (np.abs(c) < 0.95)
        self.rail_head = float(np.max(U[rail]))
        return np.maximum(dil, self.rail_head)

    def height(self, lat, h, s):
        """Высота над опорным профилем с поправкой Δ(s) по рельсам вдали."""
        if self.fixed is not None or self.U is None:
            return h
        ref = self.reference()
        b = np.clip(np.digitize(lat, LAT_BINS) - 1, 0, len(ref) - 1)
        # Δ(s): верх рельсов в 5-м бинах относительно выученного уровня головок
        sb = (s // 5).astype(int)
        band = (np.abs(lat) > 0.6) & (np.abs(lat) < 0.95) & (h > self.rail_head - 0.4) & (h < self.rail_head + 0.4)
        ub = np.unique(sb)
        delta = np.full(len(ub), np.nan)
        for i, q in enumerate(ub):
            v = h[band & (sb == q)]
            if len(v) >= 3:
                delta[i] = np.percentile(v, 80) - self.rail_head
        ok = ~np.isnan(delta)
        if ok.sum() >= 2:
            delta = np.interp(ub, ub[ok], np.clip(delta[ok], -0.3, 0.3))
        else:
            delta = np.zeros(len(ub))
        d = delta[np.searchsorted(ub, sb)]
        return h - d - ref[b]


# ----------------------------------------------------------------- детектор


@dataclass
class Config:
    name: str
    axis: str = "walls"  # straight | walls | rails | oracle
    floor: str = "profile"  # profile | fixed
    h_min: float = 0.08  # порог высоты над нормой (profile) или над осью (fixed → 0.25)
    accumulate: int = 1  # сколько кадров накапливать (C)
    persist: int = 1  # в скольких кадрах из accumulate должен быть объект
    min_pts: int = 3
    min_rings: int = 2
    cell_s: float = 0.6
    cell_lat: float = 0.35
    guards: bool = True  # сужение габарита/порог по дальности/рельсы/конец оси
    confirm: tuple[int, int] = (1, 1)  # подтверждение M из N кадров (C)
    margin_k: float = 0.004  # сужение габарита на метр дальности (запас на ошибку оси)
    # «чистая зона»: верх габарита вблизи в норме пуст — там хватает 1–2 точек
    clean_h: float | None = None  # нижняя граница зоны над нормой, м (None — выкл.)
    clean_smax: float = 60.0
    clean_min_pts: int = 2
    axis_ext: float = 0.0  # продление оси за последнюю надёжную точку, м
    split: bool = False  # S: раздвоенные лучи (ближнее отражение в габарите) — сильные точки
    beam_bg: bool = False  # B: луч короче своего фона за последнюю секунду — сильные точки
    evidence: bool = False  # E: сетка уверенности вдоль пути вместо M из N
    ev_thr: float = 3.0


@dataclass
class Detection:
    s: float
    lat: float
    h_top: float
    n: int
    rng: float


def split_mask(xyz: np.ndarray, ring: np.ndarray, min_dr: float = 1.0) -> np.ndarray:
    """Ближние отражения «раздвоенных» лучей: в том же кольце и азимуте есть отражение
    дальше на min_dr — луч частично задел что-то тонкое перед фоном."""
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    r = np.linalg.norm(xyz, axis=1)
    o = np.lexsort((az, ring))
    ro, ao, rr = ring[o], az[o], r[o]
    same = (ro[1:] == ro[:-1]) & (np.abs(ao[1:] - ao[:-1]) < 0.03)
    out = np.zeros(len(r), bool)
    i = np.flatnonzero(same & (np.abs(rr[1:] - rr[:-1]) > min_dr))
    near = np.where(rr[i] < rr[i + 1], i, i + 1)
    out[o[near]] = True
    return out


class BeamBackground:
    """B: фон по лучам (кольцо × азимут 0.1°) — EMA дальности последнего отражения.

    В однородном тоннеле дальность каждого луча от кадра к кадру почти постоянна.
    Луч, ставший короче фона, — кандидат. Аномальные лучи фон не обновляют.
    """

    NAZ = 1800  # −90…+90° с шагом 0.1°

    def __init__(self):
        self.bg = np.full(128 * self.NAZ, np.nan)
        self.cnt = np.zeros(128 * self.NAZ, np.int16)

    def keys(self, xyz, ring):
        az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
        b = np.clip(np.round((az + 90) / 0.1).astype(int), 0, self.NAZ - 1)
        return ring.astype(int) * self.NAZ + b

    def step(self, xyz, ring) -> np.ndarray:
        k = self.keys(xyz, ring)
        r = np.linalg.norm(xyz, axis=1)
        bg = self.bg[k]
        known = self.cnt[k] >= 5
        anom = known & (r < bg * 0.95 - 0.5)
        # обновление фона дальним отражением луча, только по неаномальным лучам
        order = np.lexsort((-r, k))
        first = np.r_[True, k[order][1:] != k[order][:-1]]
        kk, rmax = k[order][first], r[order][first]
        bad = np.zeros(len(self.bg), bool)
        bad[k[anom]] = True
        upd = ~bad[kk]
        kk, rmax = kk[upd], rmax[upd]
        old = self.bg[kk]
        self.bg[kk] = np.where(np.isnan(old), rmax, 0.7 * old + 0.3 * rmax)
        self.cnt[kk] = np.minimum(self.cnt[kk] + 1, 1000)
        return anom


class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cal: Calib | None = None
        self.state: dict = {}
        self.floor = FloorModel(fixed=0.35 if cfg.floor == "fixed" else None)
        self.hist: list[dict] = []  # прошлые кандидаты для накопления
        self.prof_prev = None
        self.speed_shift = 0.0
        self.bgm = BeamBackground() if cfg.beam_bg else None
        self.s_train = 0.0
        self.grid: dict = {}

    def _speed(self, pts) -> float:
        """Онлайн-сдвиг за кадр по профилю интенсивности стен (как в bench.traj)."""
        from bench.profile_speed import BIN, profile

        pr = profile(type("F", (), {"pts": pts})())
        if self.prof_prev is None:
            self.prof_prev = pr
            return 0.0
        n = pr.shape[1]
        ks = np.arange(0, int(3.0 / BIN) + 1)
        sc = np.array([np.sum(self.prof_prev[:, k:] * pr[:, : n - k]) for k in ks])
        sc = (sc - sc.mean()) / (sc.std() + 1e-9)
        prev_k = self.speed_shift / BIN
        sc = sc - 0.5 * np.maximum(np.abs(ks - prev_k) - 1.5, 0)  # плавность
        k = int(np.argmax(sc))
        self.prof_prev = pr
        self.speed_shift = k * BIN
        return self.speed_shift

    def process(self, pts, oracle_axis=None) -> tuple[list[Detection], dict]:
        cfg = self.cfg
        xyz = np.stack([pts["x"], pts["y"], pts["z"]], 1)
        crop = (xyz[:, 0] > 2) & (xyz[:, 0] < XMAX) & (np.abs(xyz[:, 1]) < 8)
        strong = np.zeros(len(xyz), np.int8)
        if cfg.split:
            strong |= split_mask(xyz, pts["ring"]).astype(np.int8)
        xyz, ring, strong = xyz[crop], pts["ring"][crop], strong[crop]
        if self.bgm is not None:
            strong |= (self.bgm.step(xyz, ring).astype(np.int8) * 2)
        if self.cal is None:
            n, d = ground_plane(xyz)
            self.cal = Calib(n, d)
            self.state["axis_ext"] = cfg.axis_ext
        ix = XIndex(xyz, {"ring": ring, "strong": strong})
        if cfg.axis == "oracle":
            xs, ys, zs = oracle_axis
        else:
            fn = {"straight": axis_straight, "walls": axis_walls, "rails": axis_rails, "near": axis_near_straight, "kalman": axis_kalman}[cfg.axis]
            xs, ys, zs = fn(ix, self.cal, self.state)
        info = {"axis_len": float(xs[-1]) if len(xs) else 0.0}
        self.last_axis = (xs, ys, zs)
        shift = self._speed(pts) if cfg.accumulate > 1 else 0.0
        if len(xs) < 3:
            return [], info
        p = ix.xyz
        inside = p[:, 0] <= xs[-1]
        p, rg, st = p[inside], ix.extra["ring"][inside], ix.extra["strong"][inside]
        lat = p[:, 1] - np.interp(p[:, 0], xs, ys)
        h = p[:, 2] - np.interp(p[:, 0], xs, zs)
        s = p[:, 0]
        if cfg.floor == "profile":
            self.floor.update(lat, h, s)
        sel = np.abs(lat) < 1.3
        hh = np.full(len(h), -1.0)
        hh[sel] = self.floor.height(lat[sel], h[sel], s[sel])
        thr = cfg.h_min if cfg.floor == "profile" else 0.35
        half_w = np.full(len(s), GAUGE_HALF_W)
        if cfg.guards:
            if cfg.axis == "kalman" and "sigma" in self.state:
                sx, sv = self.state["sigma"]
                half_w = GAUGE_HALF_W - (0.05 + 2.0 * np.interp(s, sx, sv))  # запас 2σ оси
            else:
                half_w = GAUGE_HALF_W - (0.05 + cfg.margin_k * s)  # запас на ошибку оси вдали
                if cfg.axis_ext > 0 and "x_ok" in self.state:
                    over = np.maximum(s - self.state["x_ok"], 0)
                    half_w = half_w - 0.015 * over  # за надёжной частью оси — уже
            thr = thr + 0.0015 * s  # шум высоты растёт с дальностью
            rails = (np.abs(np.abs(lat) - RAIL_LAT) < 0.12) & (hh < 0.15 + 0.0015 * s)
            hh = np.where(rails, -1.0, hh)
            s_end = xs[-1] - 3.0
        else:
            s_end = np.inf
        cand = (np.abs(lat) < half_w) & (hh > thr) & (h < GAUGE_H) & (s >= S_MIN) & (s < s_end)
        # сильные точки (S/B): в габарите, выше уровня головок рельсов, до 100 м
        sc = (st > 0) & (np.abs(lat) < half_w) & (hh > 0.05) & (h < GAUGE_H) & (s >= S_MIN) & (s < min(s_end, 100.0))
        cand = cand | sc
        cur = {"s": s[cand], "lat": lat[cand], "h": hh[cand], "ring": rg[cand], "y": p[cand, 1], "x": p[cand, 0], "strong": st[cand]}
        if cfg.evidence:
            return self._evidence(cur, pts, float(xs[-1])), info
        # накопление точек (если accumulate > 1): прошлые кандидаты сдвигаем на пройденный путь
        for hst in self.hist:
            hst["s"] = hst["s"] - shift
        self.hist.append({**cur, "age": 0})
        self.hist = self.hist[-cfg.accumulate :]
        for i, hst in enumerate(self.hist):
            hst["age"] = len(self.hist) - 1 - i
        dets = self._cluster()
        M, N = cfg.confirm
        if N > 1:
            if not hasattr(self, "dhist"):
                self.dhist = []
            shift = shift if cfg.accumulate > 1 else self._speed(pts)
            self.dhist = [[s_ - shift for s_ in h_] for h_ in self.dhist]
            self.dhist.append([d.s for d in dets])
            self.dhist = self.dhist[-N:]
            conf = []
            for d in dets:
                votes = sum(any(abs(d.s - q) < 2.0 + 0.02 * d.s for q in h_) for h_ in self.dhist)
                if votes >= M:
                    conf.append(d)
            dets = conf
        return dets, info

    def _evidence(self, cur: dict, pts, axis_end: float) -> list[Detection]:
        """E: уверенность в клетках (0.5 м вдоль пути × 0.35 м поперёк) в мировых координатах.

        Кандидат добавляет +1 (сильный +2), не больше +3 на клетку за кадр;
        все клетки видимой части коридора каждый кадр теряют 0.6. Неподвижный
        предмет копит уверенность, случайный шум — нет.
        """
        cfg = self.cfg
        self.s_train += self._speed(pts)
        sw = cur["s"] + self.s_train
        ks = np.floor(sw / 0.5).astype(int)
        kl = np.floor(cur["lat"] / 0.35).astype(int)
        add: dict = {}
        for a, b, w in zip(ks, kl, 1 + cur["strong"].clip(0, 1)):
            add[(a, b)] = min(add.get((a, b), 0) + w, 3)
        lo = np.floor((self.s_train + S_MIN) / 0.5)
        hi = np.floor((self.s_train + axis_end) / 0.5)
        g = self.grid
        for key in list(g):
            if key[0] < lo - 4:
                del g[key]
            elif key[0] <= hi:
                g[key] = max(g[key] - 0.6, -2.0)
        for key, w in add.items():
            g[key] = min(g.get(key, 0.0) + w, 10.0)
        hot = [(k, v) for k, v in g.items() if v >= cfg.ev_thr and lo <= k[0] <= hi]
        if not hot:
            return []
        # склейка соседних горячих клеток
        hot.sort()
        dets, run = [], [hot[0]]
        for k, v in hot[1:]:
            if k[0] - run[-1][0][0] <= 2:
                run.append((k, v))
            else:
                dets.append(run)
                run = [(k, v)]
        dets.append(run)
        out = []
        for run in dets:
            s0 = run[0][0][0] * 0.5 - self.s_train
            lat = float(np.mean([k[1] * 0.35 + 0.175 for k, _ in run]))
            out.append(Detection(float(s0), lat, 0.0, int(sum(v for _, v in run)), float(s0)))
        return out

    def _cluster(self) -> list[Detection]:
        cfg = self.cfg
        H = self.hist
        s = np.concatenate([h["s"] for h in H])
        if not len(s):
            return []
        lat = np.concatenate([h["lat"] for h in H])
        hh = np.concatenate([h["h"] for h in H])
        ring = np.concatenate([h["ring"] for h in H])
        age = np.concatenate([np.full(len(h["s"]), h["age"]) for h in H])
        strong = np.concatenate([h.get("strong", np.zeros(len(h["s"]), np.int8)) for h in H])
        cs = np.floor(s / cfg.cell_s).astype(int)
        cl = np.floor(lat / cfg.cell_lat).astype(int)
        keys = cs * 1000 + (cl + 500)
        uk, inv = np.unique(keys, return_inverse=True)
        # связные компоненты по соседству клеток (8-связность)
        kset = {k: i for i, k in enumerate(uk)}
        parent = np.arange(len(uk))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, k in enumerate(uk):
            a, b = divmod(k, 1000)
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    j = kset.get((a + da) * 1000 + b + db)
                    if j is not None:
                        ra, rb = find(i), find(j)
                        if ra != rb:
                            parent[ra] = rb
        comp = np.array([find(i) for i in range(len(uk))])[inv]
        dets = []
        for c in np.unique(comp):
            m = comp == c
            n = int(m.sum())
            in_clean = cfg.clean_h is not None and hh[m].min() > cfg.clean_h and s[m].max() < cfg.clean_smax
            in_clean = in_clean or (strong[m] > 0).sum() >= 2  # ≥2 сильных точек — как «чистая зона»
            multi_frame = len(np.unique(age[m])) >= 2
            if in_clean:
                if n < cfg.clean_min_pts:
                    continue
            else:
                if n < cfg.min_pts:
                    continue
                if len(np.unique(ring[m])) < cfg.min_rings and not multi_frame:
                    continue
            if len(np.unique(age[m])) < cfg.persist:
                continue
            cur = m & (age == 0)
            if cfg.accumulate > 1 and not cur.any():
                continue  # объект должен быть и в текущем кадре
            ss = float(np.min(s[m]))
            dets.append(Detection(ss, float(np.median(lat[m])), float(hh[m].max()), n, ss))
        dets.sort(key=lambda d: d.s)
        return dets


CONFIGS = [
    Config("S+fix", axis="straight", floor="fixed", guards=False),
    Config("N+D+C", axis="near", confirm=(3, 5)),
    Config("A+fix", axis="walls", floor="fixed"),
    Config("A+D", axis="walls"),
    Config("B+D", axis="rails"),
    Config("Oracle+D", axis="oracle"),
    Config("A+D+C", axis="walls", confirm=(3, 5)),
    Config("Oracle+D+C", axis="oracle", confirm=(3, 5)),
    *[
        Config(f"A+D+C m{mk:g} c{m}", axis="walls", confirm=(m, 5), margin_k=mk)
        for mk in (0.001, 0.002, 0.003)
        for m in (2, 3)
    ],
    Config("Z c3", axis="walls", confirm=(3, 5), clean_h=0.5),
    Config("Z c2 m0.002", axis="walls", confirm=(2, 5), margin_k=0.002, clean_h=0.5),
    Config("ZA c3", axis="walls", confirm=(3, 5), clean_h=0.5, accumulate=3, persist=2),
    Config("ZE c3", axis="walls", confirm=(3, 5), clean_h=0.5, axis_ext=30.0),
    Config("ZE c2 m0.002", axis="walls", confirm=(2, 5), margin_k=0.002, clean_h=0.5, axis_ext=30.0),
    Config("Z+S c3", axis="walls", confirm=(3, 5), clean_h=0.5, split=True),
    Config("Z+B c3", axis="walls", confirm=(3, 5), clean_h=0.5, beam_bg=True),
    Config("Z+SB c3", axis="walls", confirm=(3, 5), clean_h=0.5, split=True, beam_bg=True),
    Config("E", axis="walls", clean_h=0.5, evidence=True),
    Config("E+SB", axis="walls", clean_h=0.5, evidence=True, split=True, beam_bg=True),
    Config("E5", axis="walls", clean_h=0.5, evidence=True, ev_thr=5.0),
    Config("E5+S", axis="walls", clean_h=0.5, evidence=True, split=True, ev_thr=5.0),
    Config("K+D", axis="kalman"),
    Config("K+D+C", axis="kalman", confirm=(3, 5)),
]
