"""Ось пути: рельсы вблизи (шаблон пары головок + слежение) → стены вдали → RANSAC-кубика.

Ось задаётся в нормализованных осях лидара как y(x), z(x) на сетке x = 4, 6, … м.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from metro_obstacle_core import kernels as K

STEP = K.STEP
NEAR_RAILS = 30.0  # ось по рельсам строится до этой дальности
NEAR = 20.0  # до этой дальности ось берётся из рельсов, дальше — из стен
AXIS_X0 = 4.0


@dataclass
class Points:
    """Точки коридора, разложенные по целым метрам x (срезы — kernels._slab)."""

    x: np.ndarray  # float32
    y: np.ndarray
    z: np.ndarray
    start: np.ndarray  # начало каждого метрового бина


@dataclass
class AxisState:
    """Состояние оси между кадрами (сглаживание во времени, RANSAC)."""

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    near_coef: np.ndarray | None = None
    ol: float = float("nan")  # смещение левой стены от оси
    orr: float = float("nan")  # смещение правой стены от оси
    fit: str = "cubic"  # "cubic" — одна RANSAC-кубика; "smooth" — робастное сглаживание
    smooth_lam: float = 2000.0  # жёсткость сглаживания (штраф на вторую разность)
    smooth_gap: float = 30.0  # разрыв согласованных оценок, после которого ось обрывается, м
    wall_ext: bool = False  # продолжать ось за кубикой по поучастковым оценкам стен


def z_floor(plane: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Высота плоскости калибровки: n·p + d = 0 → z = −(d + nx·x + ny·y)/nz."""
    return -(plane[3] + plane[0] * x + plane[1] * y) / plane[2]


def near_axis(
    pts: Points, plane: np.ndarray, xs: np.ndarray, st: AxisState
) -> tuple[np.ndarray, np.ndarray]:
    """Ближняя ось: парабола по центрам рельсов в 5–30 м, сглаженная во времени.

    Без неё ось лидара считалась бы осью пути — неверно при повороте лидара
    относительно пути и в кривых (смещение 0.3 м уже на 10 м).
    """
    prior = st.near_coef if st.near_coef is not None else np.zeros(3)
    X, Y = K.near_rail_centers(pts.y, pts.z, pts.start, plane, prior, NEAR_RAILS)
    if len(X) >= 5:
        new = np.polyfit(X, Y, 2)
        r = np.abs(np.polyval(new, X) - Y)
        if (r < 0.1).sum() >= 5:
            new = np.polyfit(X[r < 0.1], Y[r < 0.1], 2)
        st.near_coef = new if st.near_coef is None else 0.5 * st.near_coef + 0.5 * new
    coef = st.near_coef if st.near_coef is not None else np.zeros(3)
    ys = np.polyval(coef, xs)
    return ys, z_floor(plane, xs, ys)


def robust_cubic(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    rng: np.random.Generator,
    tol: float = 0.3,
    min_span: float = 20.0,
    iters: int = 150,
) -> tuple[np.ndarray | None, float | None]:
    """RANSAC-кубика y(x) по оценкам оси (вес ближних по рельсам — 3).

    Выборки — те же вызовы rng.choice, что в прототипе (состояние генератора
    переходит между кадрами). Кубики через 4 точки решаются пакетом; выборки
    с повторяющимся x (вырожденные) — через np.polyfit, как в прототипе.
    """
    n = len(X)
    if n < 8:
        return None, None
    idx = np.array([rng.choice(n, 4, replace=False) for _ in range(iters)])
    Xi, Yi = X[idx], Y[idx]
    valid = np.ptp(Xi, axis=1) >= min_span
    srt = np.sort(Xi, axis=1)
    degen = (np.diff(srt, axis=1) == 0).any(axis=1)
    coefs = np.full((iters, 4), np.nan)
    ok = valid & ~degen
    if ok.any():
        V = Xi[ok, :, None] ** np.arange(3, -1, -1)  # как np.vander
        scale = np.sqrt((V * V).sum(axis=1))
        sol = np.linalg.solve(V / scale[:, None, :], Yi[ok][:, :, None])[:, :, 0]
        coefs[ok] = sol / scale
    for i in np.flatnonzero(valid & degen):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            coefs[i] = np.polyfit(Xi[i], Yi[i], 3)
    best, bestn = None, -1.0
    for i in np.flatnonzero(valid):
        c = coefs[i]
        v = ((c[0] * X + c[1]) * X + c[2]) * X + c[3]
        inl = np.abs(v - Y) < tol
        w = W[inl].sum()
        if w > bestn:
            best, bestn = inl, w
    if best is None or best.sum() < 6:
        return None, None
    with np.errstate(all="ignore"):
        c = np.polyfit(X[best], Y[best], 3, w=W[best])
    return c, float(X[best].max())


def axis_walls(
    pts: Points, plane: np.ndarray, st: AxisState, max_range: float
) -> tuple[np.ndarray, np.ndarray]:
    """Ось: ближние 20 м — по рельсам; дальше — по стенам со смещениями из ближней зоны.

    Ось обрывается на последней точке, согласованной с RANSAC-кубикой: дальше
    стены перестают согласовываться (поворот, переход типа тоннеля).
    Возвращает (xs, ys) — без NaN; высоту полотна считает floor_along.
    """
    xs = np.arange(AXIS_X0, max_range, STEP)
    ys = np.full(len(xs), np.nan)
    zs = np.full(len(xs), np.nan)
    yn, zn = near_axis(pts, plane, xs, st)
    near = xs <= NEAR
    ys[near], zs[near] = yn[near], zn[near]
    n_near = int(near.sum())
    el, er, st.ol, st.orr = K.walls_axis(pts.y, pts.z, pts.start, xs, ys, zs, n_near, st.ol, st.orr)
    X = np.r_[xs[near], xs, xs]
    Y = np.r_[ys[near], el, er]
    W = np.r_[np.full(n_near, 3.0), np.ones(2 * len(xs))]
    v = ~np.isnan(Y)
    if st.fit == "smooth":
        ys = smooth_axis(xs, X[v], Y[v], W[v], st.smooth_lam, gap=st.smooth_gap)
    else:
        c3, xok = robust_cubic(X[v], Y[v], W[v], st.rng)
        if c3 is not None:
            ys = np.where(xs <= xok, np.polyval(c3, xs), np.nan)
            if st.wall_ext:
                ys = extend_by_walls(xs, ys, el, er, xok)
    ok = ~np.isnan(ys)
    return xs[ok], ys[ok]


def smooth_axis(
    xs: np.ndarray, X: np.ndarray, Y: np.ndarray, W: np.ndarray, lam: float,
    tol: float = 0.3, iters: int = 4, gap: float = 30.0,
) -> np.ndarray:
    """Ось как робастная сглаживающая кривая на сетке xs (Whittaker–Eilers).

    Минимизируется Σ wᵢ(yᵢ − f(xᵢ))² + λ·Σ(Δ²f)² — кривая следует за S-поворотами и
    переходами, которые одна кубика описать не может. Веса пересчитываются по невязкам
    (Тьюки, масштаб tol): колонны, ниши, край платформы выпадают. Ось кончается на
    последнем бине, где есть согласованная оценка, и не раньше, чем там, где подряд
    идут пропуски длиннее gap (дальше не экстраполируем).
    """
    n = len(xs)
    if len(X) < 8:
        return np.full(n, np.nan)
    b = np.clip(np.round((X - xs[0]) / (xs[1] - xs[0])).astype(int), 0, n - 1)
    D = np.diff(np.eye(n), 2, axis=0)
    P = lam * (D.T @ D)
    w = W.astype(float).copy()
    f = np.zeros(n)
    for _ in range(iters):
        Wd = np.bincount(b, weights=w, minlength=n)
        Zd = np.bincount(b, weights=w * Y, minlength=n)
        f = np.linalg.solve(np.diag(Wd) + P + 1e-9 * np.eye(n), Zd)
        r = np.abs(Y - f[b]) / tol
        w = W * np.where(r < 1.0, (1 - r * r) ** 2, 0.0)
    good = np.zeros(n, bool)
    good[b[w > 0]] = True
    if not good.any():
        return np.full(n, np.nan)
    last = np.flatnonzero(good)
    # обрыв на первом разрыве согласованных оценок длиннее 10 м
    gaps = np.flatnonzero(np.diff(last) * (xs[1] - xs[0]) > gap)
    end = last[gaps[0]] if len(gaps) else last[-1]
    out = f.copy()
    out[end + 1 :] = np.nan
    return out


def floor_along(pts: Points, xs: np.ndarray, ys: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Высота полотна вдоль оси (слежение, см. kernels.floor_along) + сглаживание по 5 бинам."""
    zs = K.floor_along(pts.x, pts.y, pts.z, pts.start, xs, ys, plane)
    if len(zs) >= 5:
        zs = np.convolve(np.pad(zs, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return zs


def extend_by_walls(
    xs: np.ndarray, ys: np.ndarray, el: np.ndarray, er: np.ndarray, xok: float,
    agree: float = 0.3, step: float = 0.4, gap: float = 10.0,
) -> np.ndarray:
    """Продолжение оси за кубикой по поучастковым оценкам стен (без глобальной подгонки).

    Кубика по всей длине не описывает S-поворот и обрывается раньше, чем кончаются
    стены. Каждая стена в бине даёт свою оценку оси (через смещение из ближней зоны).
    Точка принимается, если обе стены согласны (|el − er| ≤ agree) — две независимые
    стены редко ошибаются одинаково, — или одна стена продолжает ось без скачка
    (≤ step от прошлой точки). Разрыв длиннее gap обрывает ось. Между принятыми
    точками — линейная интерполяция.
    """
    out = ys.copy()
    idx = np.flatnonzero(~np.isnan(ys))
    if not len(idx):
        return out
    last_i = int(idx[-1])
    last_y = float(ys[last_i])
    acc_i, acc_y = [last_i], [last_y]
    dx = float(xs[1] - xs[0])
    for i in range(last_i + 1, len(xs)):
        if (i - acc_i[-1]) * dx > gap:
            break
        a, b = el[i], er[i]
        cand = None
        if not np.isnan(a) and not np.isnan(b) and abs(a - b) <= agree:
            cand = 0.5 * (a + b)
            if abs(cand - acc_y[-1]) > step * (i - acc_i[-1]):
                cand = None
        else:
            for e in (a, b):
                if not np.isnan(e) and abs(e - acc_y[-1]) <= step:
                    cand = float(e)
                    break
        if cand is not None:
            acc_i.append(i)
            acc_y.append(float(cand))
    if len(acc_i) > 1:
        ii = np.array(acc_i)
        out[ii[0] : ii[-1] + 1] = np.interp(xs[ii[0] : ii[-1] + 1], xs[ii], np.array(acc_y))
    return out
