"""Горячие циклы детектора на numba (однопоточно, cache=True).

Каждое ядро повторяет арифметику прототипа (стенд, bench/detect.py) вплоть до типов:
где стенд считает во float32 (сравнение массива float32 с питоновским числом), здесь
тоже float32; где в выражении есть numpy-скаляр float64 — float64. Иначе на границах
порогов точки изредка «перескакивали» бы, и цифры стенда переставали бы относиться
к сдаваемому коду.
"""

from __future__ import annotations

import numpy as np
from numba import njit

RAIL_LAT = 0.76  # полуколея по головкам рельсов, м
BAND_LO, BAND_HI = 1.2, 2.6  # полоса высот стен над полотном, м
STEP = 2.0  # шаг бинов вдоль пути, м

_F32_DEG = np.float32(180.0) / np.float32(np.pi)  # как npy_degreesf: x * (180f / pi_f)
_F32_AZ_TOL = np.float32(0.03)
_F32_1_3 = np.float32(1.3)
_F32_0_35 = np.float32(0.35)


# ----------------------------------------------------------------- перцентили


@njit(cache=True)
def pct_f64(v: np.ndarray, q: float) -> float:
    """np.percentile(v, q) (метод linear) для отсортированного float64-массива."""
    n = v.size
    virt = (n - 1) * (q / 100.0)
    if virt >= n - 1:
        return v[n - 1]
    lo = int(np.floor(virt))
    g = virt - lo
    a = v[lo]
    b = v[lo + 1]
    diff = b - a
    if g >= 0.5:
        return b - diff * (1.0 - g)
    return a + diff * g


@njit(cache=True)
def pct_f32(v: np.ndarray, q: float) -> np.float32:
    """np.percentile(v, q) для отсортированного float32-массива.

    numpy интерполирует во float32 (доля между соседями — «слабое» число), поэтому
    и здесь разность и сдвиг считаются во float32.
    """
    n = v.size
    virt = (n - 1) * (q / 100.0)
    if virt >= n - 1:
        return v[n - 1]
    lo = int(np.floor(virt))
    g = virt - lo
    a = v[lo]
    b = v[lo + 1]
    diff = b - a
    if g >= 0.5:
        return b - diff * np.float32(1.0 - g)
    return a + diff * np.float32(g)


# ----------------------------------------------------------------- вход


@njit(cache=True)
def normalize(xyz: np.ndarray, fwd: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Оси сообщения → x-вперёд, y-влево; отбрасывает невалидные и кроп как у кэша стенда.

    fwd: 0 = x, 1 = −x, 2 = y, 3 = −y. Кроп x ∈ (−3, 230), |y| < 12 — тот же, что при
    кэшировании кадров стенда: так раздвоенные лучи и профиль скорости видят ровно
    те же точки, что и на стенде. Возвращает (x, y, z, индексы исходных точек).
    """
    n = xyz.shape[0]
    xs = np.empty(n, np.float32)
    ys = np.empty(n, np.float32)
    zs = np.empty(n, np.float32)
    idx = np.empty(n, np.int64)
    k = 0
    for i in range(n):
        x = xyz[i, 0]
        y = xyz[i, 1]
        z = xyz[i, 2]
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        if abs(x) + abs(y) + abs(z) <= 0:
            continue
        if fwd == 0:
            xn, yn = x, y
        elif fwd == 1:
            xn, yn = -x, -y
        elif fwd == 2:
            xn, yn = y, -x
        else:
            xn, yn = -y, x
        if not (xn > -3.0 and xn < 230.0 and abs(yn) < 12.0):
            continue
        xs[k] = xn
        ys[k] = yn
        zs[k] = z
        idx[k] = i
        k += 1
    return xs[:k], ys[:k], zs[:k], idx[:k]


@njit(cache=True)
def split_mask(x: np.ndarray, y: np.ndarray, z: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Ближние отражения «раздвоенных» лучей (dual return).

    В том же кольце и азимуте (±0.03°) есть отражение дальше на > 1 м: луч частично
    задел что-то тонкое перед фоном. Порядок как у lexsort((az, ring)) прототипа:
    устойчивая сортировка по кольцу, внутри кольца — устойчивая по азимуту.
    """
    n = x.size
    az = np.empty(n, np.float32)
    r = np.empty(n, np.float32)
    for i in range(n):
        az[i] = np.float32(np.arctan2(y[i], x[i])) * _F32_DEG
        r[i] = np.sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i])
    out = np.zeros(n, np.bool_)
    if n < 2:
        return out
    rmax = 0
    for i in range(n):
        if ring[i] > rmax:
            rmax = ring[i]
    cnt = np.zeros(rmax + 2, np.int64)
    for i in range(n):
        cnt[ring[i] + 1] += 1
    for k in range(rmax + 1):
        cnt[k + 1] += cnt[k]
    pos = cnt[:-1].copy()
    by_ring = np.empty(n, np.int64)
    for i in range(n):
        by_ring[pos[ring[i]]] = i
        pos[ring[i]] += 1
    for k in range(rmax + 1):
        a, b = cnt[k], cnt[k + 1]
        if b - a < 2:
            continue
        ids = by_ring[a:b]
        o = ids[np.argsort(az[ids], kind="mergesort")]
        for j in range(o.size - 1):
            p, q = o[j], o[j + 1]
            if abs(az[q] - az[p]) < _F32_AZ_TOL and abs(r[q] - r[p]) > 1.0:
                out[p if r[p] < r[q] else q] = True
    return out


@njit(cache=True)
def profile_hist(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, inten: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Счётчики и суммы интенсивности для профиля стен (как np.histogram в прототипе).

    8 каналов: сторона (−1, +1) × 4 полосы высоты. Номер бина — тем же способом, что
    быстрый путь np.histogram для float32: (x − 4)·720/36 с поправкой по краям бинов.
    """
    nb = edges.size - 1
    cnt = np.zeros((8, nb), np.int64)
    sm = np.zeros((8, nb), np.float64)
    for i in range(x.size):
        xi = x[i]
        yi = y[i]
        if not (abs(yi) > _F32_1_3 and xi > 4.0 and xi < 40.0):
            continue
        zi = z[i]
        if zi > -1.5 and zi < -0.5:
            band = 0
        elif zi > -0.5 and zi < 0.5:
            band = 1
        elif zi > 0.5 and zi < 1.5:
            band = 2
        elif zi > 1.5 and zi < 3.0:
            band = 3
        else:
            continue
        ch = band if yi < 0 else 4 + band
        f = np.float64(xi - np.float32(4.0)) / 36.0 * nb
        k = int(f)
        if k == nb:
            k -= 1
        if xi < edges[k]:
            k -= 1
        if xi >= edges[k + 1] and k != nb - 1:
            k += 1
        cnt[ch, k] += 1
        sm[ch, k] += np.float64(inten[i])
    return cnt, sm


# ----------------------------------------------------------------- индекс по x


@njit(cache=True)
def bucket_by_x(x: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    """Раскладка точек по целым метрам x (устойчиво). Замена полной сортировке по x.

    Все срезы прототипа (XIndex.slab) — с целыми границами, поэтому срез [x0, x1)
    = бины floor(x) ∈ [x0, x1). Порядок внутри среза на результат не влияет
    (перцентили, счётчики).
    """
    n = x.size
    start = np.zeros(nbins + 1, np.int64)
    for i in range(n):
        start[int(np.floor(x[i])) + 1] += 1
    for k in range(nbins):
        start[k + 1] += start[k]
    pos = start[:-1].copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        k = int(np.floor(x[i]))
        order[pos[k]] = i
        pos[k] += 1
    return order, start


@njit(cache=True)
def _slab(start: np.ndarray, x0: float, x1: float) -> tuple[int, int]:
    nb = start.size - 1
    a = min(max(int(x0), 0), nb)
    b = min(max(int(x1), 0), nb)
    return start[a], start[b]


@njit(cache=True)
def _zfloor(n0: float, n1: float, n2: float, d: float, x: float, y: float) -> float:
    return -(d + n0 * x + n1 * y) / n2


# ----------------------------------------------------------------- ось по рельсам


@njit(cache=True)
def rail_center(
    by: np.ndarray, bz: np.ndarray, i0: int, i1: int, c: float, zc: float, search: float
) -> float:
    """Центр пути в бине: шаблон «пара головок рельсов через 1.52 м».

    Перебор центра в ±search от прогноза; счёт = точки в полосах ±4 см у обеих
    головок на высоте 0.08–0.4 м над полотном, минус точки между рельсами на той же
    высоте, минус точки выше 0.5 м над путём (габарит над путём должен быть свободен).
    """
    m = i1 - i0
    yv = np.empty(m, np.float64)
    yh = np.empty(m, np.float64)
    ny = 0
    nh = 0
    lim = RAIL_LAT + search + 0.1
    lim_hi = 1.0 + search
    for j in range(i0, i1):
        h = bz[j] - zc
        dy = abs(by[j] - c)
        if h > 0.08 and h < 0.4 and dy < lim:
            yv[ny] = by[j]
            ny += 1
        if h > 0.5 and h < 2.0 and dy < lim_hi:
            yh[nh] = by[j]
            nh += 1
    if ny < 4:
        return np.nan
    start = c - search
    stop = c + search + 1e-9
    ncc = int(np.ceil((stop - start) / 0.02))
    second = start + 0.02
    delta = second - start  # как np.arange: a[0]=start, a[1]=start+step, далее start+i·δ
    best = np.nan
    bs = 0
    for k in range(ncc):
        if k == 0:
            cc = start
        elif k == 1:
            cc = second
        else:
            cc = start + k * delta
        left = 0
        right = 0
        mid = 0
        for t in range(ny):
            yy = yv[t]
            if abs(yy - (cc + RAIL_LAT)) < 0.04:
                left += 1
            if abs(yy - (cc - RAIL_LAT)) < 0.04:
                right += 1
            if abs(yy - cc) < RAIL_LAT - 0.12:
                mid += 1
        pen = 0
        for t in range(nh):
            if abs(yh[t] - cc) < 1.0:
                pen += 1
        sc = min(left, right) * 2 - mid - 3 * pen
        if left >= 1 and right >= 1 and sc > bs:
            best = cc
            bs = sc
    return best if bs >= 2 else np.nan


@njit(cache=True)
def _fit_eval(xv: np.ndarray, yv: np.ndarray, deg: int, x: float) -> float:
    """Значение МНК-полинома степени deg (1 или 2) в точке x.

    Эквивалент np.polyval(np.polyfit(xv, yv, deg), x); x центрируется и
    масштабируется, система 3×3 решается с выбором ведущего — ошибка ~1e-14 м.
    """
    n = xv.size
    xm = 0.0
    for i in range(n):
        xm += xv[i]
    xm /= n
    sc = 0.0
    for i in range(n):
        sc = max(sc, abs(xv[i] - xm))
    if sc == 0.0:
        sc = 1.0
    k = deg + 1
    a = np.zeros((3, 4))
    for i in range(n):
        t = (xv[i] - xm) / sc
        pw = np.ones(3)
        pw[1] = t
        pw[2] = t * t
        for r in range(k):
            for cix in range(k):
                a[r, cix] += pw[r] * pw[cix]
            a[r, 3] += pw[r] * yv[i]
    for col in range(k):
        piv = col
        for r in range(col + 1, k):
            if abs(a[r, col]) > abs(a[piv, col]):
                piv = r
        if piv != col:
            for cix in range(4):
                tmp = a[col, cix]
                a[col, cix] = a[piv, cix]
                a[piv, cix] = tmp
        if a[col, col] == 0.0:
            return np.nan
        for r in range(col + 1, k):
            f = a[r, col] / a[col, col]
            for cix in range(col, 4):
                a[r, cix] -= f * a[col, cix]
    coef = np.zeros(3)
    for r in range(k - 1, -1, -1):
        s = a[r, 3]
        for cix in range(r + 1, k):
            s -= a[r, cix] * coef[cix]
        coef[r] = s / a[r, r]
    t = (x - xm) / sc
    return coef[0] + coef[1] * t + coef[2] * t * t


@njit(cache=True)
def near_rail_centers(
    by: np.ndarray,
    bz: np.ndarray,
    start: np.ndarray,
    plane: np.ndarray,
    prior: np.ndarray,
    near_rails: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Центры пути по рельсам в бинах 5, 7, … < near_rails с последовательным слежением.

    Прогноз центра в бине — по уже найденным центрам (прямая/парабола по последним 6),
    пока их меньше 3 — по параболе прошлого кадра (prior).
    """
    n0, n1, n2, d = plane[0], plane[1], plane[2], plane[3]
    nx = int(np.ceil((near_rails - 5.0) / STEP))
    xo = np.empty(nx, np.float64)
    yo = np.empty(nx, np.float64)
    k = 0
    for ix in range(nx):
        x = 5.0 + ix * STEP
        if k >= 3:
            deg = 2 if k >= 5 else 1
            lo = max(0, k - 6)
            c = _fit_eval(xo[lo:k], yo[lo:k], deg, x)
        else:
            c = (0.0 * x + prior[0]) * x + prior[1]
            c = c * x + prior[2]
        zc = _zfloor(n0, n1, n2, d, x, c)
        i0, i1 = _slab(start, x - STEP / 2, x + STEP / 2)
        yc = rail_center(by, bz, i0, i1, c, zc, 0.35)
        if not np.isnan(yc) and abs(yc - c) < 0.4:
            xo[k] = x
            yo[k] = yc
            k += 1
    return xo[:k], yo[:k]


# ----------------------------------------------------------------- ось по стенам


@njit(cache=True)
def walls(
    by: np.ndarray, bz: np.ndarray, i0: int, i1: int, c: float, z0: float
) -> tuple[float, float]:
    """Положение левой/правой стены в бине: 10-й/90-й перцентиль в полосе высот."""
    m = i1 - i0
    lv = np.empty(m, np.float64)
    rv = np.empty(m, np.float64)
    nl = 0
    nr = 0
    lo = z0 + BAND_LO
    hi = z0 + BAND_HI
    for j in range(i0, i1):
        zz = bz[j]
        if zz > lo and zz < hi:
            yy = by[j] - c
            if yy > 1.1 and yy < 5.0:
                lv[nl] = yy
                nl += 1
            if yy < -1.1 and yy > -5.0:
                rv[nr] = yy
                nr += 1
    wl = np.nan
    wr = np.nan
    if nl >= 3:
        wl = c + pct_f64(np.sort(lv[:nl]), 10.0)
    if nr >= 3:
        wr = c + pct_f64(np.sort(rv[:nr]), 90.0)
    return wl, wr


@njit(cache=True)
def _nanmedian(v: np.ndarray) -> float:
    m = 0
    tmp = np.empty(v.size, np.float64)
    for i in range(v.size):
        if not np.isnan(v[i]):
            tmp[m] = v[i]
            m += 1
    s = np.sort(tmp[:m])
    if m % 2 == 1:
        return s[m // 2]
    return (s[m // 2 - 1] + s[m // 2]) / 2


@njit(cache=True)
def _line_eval(xv: np.ndarray, yv: np.ndarray, x: float) -> float:
    """np.polyval(np.polyfit(xv, yv, 1), x) — МНК-прямая в центрированном виде."""
    n = xv.size
    xm = 0.0
    ym = 0.0
    for i in range(n):
        xm += xv[i]
        ym += yv[i]
    xm /= n
    ym /= n
    sxx = 0.0
    sxy = 0.0
    for i in range(n):
        dx = xv[i] - xm
        sxx += dx * dx
        sxy += dx * (yv[i] - ym)
    return ym + sxy / sxx * (x - xm)


@njit(cache=True)
def walls_axis(
    by: np.ndarray,
    bz: np.ndarray,
    start: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    n_near: int,
    ol_prev: float,
    or_prev: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Ось вдали по стенам: смещения стен от оси, выученные в ближней зоне.

    ys/zs на входе заполнены в ближних n_near бинах (ось по рельсам) и меняются на
    месте: в дальних бинах — центр по стенам, если он согласован с прогнозом (< 0.8 м).
    Возвращает оценки оси по левой/правой стене (el, er) и сглаженные смещения стен.
    """
    nx = xs.size
    offl = np.empty(n_near, np.float64)
    offr = np.empty(n_near, np.float64)
    for i in range(n_near):
        i0, i1 = _slab(start, xs[i] - STEP / 2, xs[i] + STEP / 2)
        wl, wr = walls(by, bz, i0, i1, ys[i], zs[i])
        offl[i] = wl - ys[i]
        offr[i] = wr - ys[i]
    nl = 0
    nr = 0
    for i in range(n_near):
        nl += not np.isnan(offl[i])
        nr += not np.isnan(offr[i])
    ol = _nanmedian(offl) if nl >= 3 else np.nan
    orr = _nanmedian(offr) if nr >= 3 else np.nan
    # смещения стен сглаживаем во времени (стена у пути почти не меняется)
    if not np.isnan(ol):
        ol = ol if np.isnan(ol_prev) else 0.8 * ol_prev + 0.2 * ol
    else:
        ol = ol_prev
    if not np.isnan(orr):
        orr = orr if np.isnan(or_prev) else 0.8 * or_prev + 0.2 * orr
    else:
        orr = or_prev
    el = np.full(nx, np.nan)
    er = np.full(nx, np.nan)
    good = np.empty(nx, np.int64)
    ng = 0
    for i in range(n_near):
        if not np.isnan(ys[i]):
            good[ng] = i
            ng += 1
    gx = np.empty(6, np.float64)
    gy = np.empty(6, np.float64)
    gz = np.empty(6, np.float64)
    for i in range(n_near, nx):
        if ng < 3:
            break
        lo = max(0, ng - 6)
        m = ng - lo
        for t in range(m):
            gx[t] = xs[good[lo + t]]
            gy[t] = ys[good[lo + t]]
            gz[t] = zs[good[lo + t]]
        c = _line_eval(gx[:m], gy[:m], xs[i])
        zc = _line_eval(gx[:m], gz[:m], xs[i])
        i0, i1 = _slab(start, xs[i] - STEP / 2, xs[i] + STEP / 2)
        wl, wr = walls(by, bz, i0, i1, c, zc)
        if not np.isnan(ol) and not np.isnan(wl):
            el[i] = wl - ol
        if not np.isnan(orr) and not np.isnan(wr):
            er[i] = wr - orr
        s = 0.0
        nc = 0
        if not np.isnan(el[i]) and abs(el[i] - c) < 0.8:
            s += el[i]
            nc += 1
        if not np.isnan(er[i]) and abs(er[i] - c) < 0.8:
            s += er[i]
            nc += 1
        if nc > 0:
            ys[i] = s / nc
            zs[i] = zc
            good[ng] = i
            ng += 1
    return el, er, ol, orr


# ----------------------------------------------------------------- полотно


@njit(cache=True)
def floor_along(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    start: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    plane: np.ndarray,
) -> np.ndarray:
    """Высота полотна вдоль оси: последовательное слежение от ближней зоны.

    В бине — 30-й перцентиль точек у оси в окне ±0.35 м от прогноза (прогноз по
    последним бинам, уклон ≤ 6%). Прогноз бывает «float64» (от плоскости калибровки)
    или «питоновским» (от найденного уровня) — прототип во втором случае сравнивает
    во float32; повторяем это, иначе редкие точки на границе окна расходятся.
    """
    n0, n1, n2, d = plane[0], plane[1], plane[2], plane[3]
    nx = xs.size
    zs = np.empty(nx, np.float64)
    have_prev = False
    z_prev = 0.0
    prev_weak = False
    slope = 0.0
    buf = np.empty(0, np.float32)
    for i in range(nx):
        x = xs[i]
        y = ys[i]
        if not have_prev:
            pred = _zfloor(n0, n1, n2, d, x, y)
            weak = False
        else:
            pred = z_prev + slope * STEP
            weak = prev_weak
        i0, i1 = _slab(start, x - STEP, x + STEP)
        if buf.size < i1 - i0:
            buf = np.empty(i1 - i0, np.float32)
        m = 0
        pred32 = np.float32(pred)
        for j in range(i0, i1):
            if not abs(by[j] - y) < 1.2:
                continue
            if weak:
                ok = abs(bz[j] - pred32) < _F32_0_35
            else:
                ok = abs(bz[j] - pred) < 0.35
            if ok:
                buf[m] = bz[j]
                m += 1
        if m >= 5:
            z = np.float64(pct_f32(np.sort(buf[:m]), 30.0))
            if have_prev:
                sl = 0.7 * slope + 0.3 * (z - z_prev) / STEP
                slope = min(max(sl, -0.06), 0.06)
            zs[i] = z
            z_prev = z
            prev_weak = True
        else:
            zs[i] = pred
            z_prev = pred
            prev_weak = weak
        have_prev = True
    return zs


@njit(cache=True)
def interp_prefilter(
    x: np.ndarray, y: np.ndarray, xs: np.ndarray, ys: np.ndarray, x_end: float, lat_max: float
) -> np.ndarray:
    """Индексы точек с x ≤ x_end и |y − ось(x)| < lat_max (с запасом 1e-6 м).

    Грубый отбор перед точным np.interp: дальше нужны только точки у оси, а
    интерполяция всех 150 тыс. точек облака — самая дорогая векторная операция.
    """
    n = x.size
    out = np.empty(n, np.int64)
    k = 0
    m = xs.size
    lim = lat_max + 1e-6
    for i in range(n):
        xi = np.float64(x[i])
        if xi > x_end:
            continue
        if xi <= xs[0]:
            yi = ys[0]
        elif xi >= xs[m - 1]:
            yi = ys[m - 1]
        else:
            j = np.searchsorted(xs, xi, side="right") - 1
            yi = ys[j] + (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]) * (xi - xs[j])
        if abs(np.float64(y[i]) - yi) < lim:
            out[k] = i
            k += 1
    return out[:k]


@njit(cache=True)
def group_percentiles(
    vals: np.ndarray, groups: np.ndarray, ngroups: int, qs: np.ndarray, min_count: int
) -> np.ndarray:
    """Перцентили qs по группам (float64); группа с < min_count значений → NaN."""
    cnt = np.zeros(ngroups + 1, np.int64)
    for i in range(vals.size):
        g = groups[i]
        if 0 <= g < ngroups:
            cnt[g + 1] += 1
    for g in range(ngroups):
        cnt[g + 1] += cnt[g]
    pos = cnt[:-1].copy()
    buf = np.empty(cnt[ngroups], np.float64)
    for i in range(vals.size):
        g = groups[i]
        if 0 <= g < ngroups:
            buf[pos[g]] = vals[i]
            pos[g] += 1
    out = np.full((qs.size, ngroups), np.nan)
    for g in range(ngroups):
        a, b = cnt[g], cnt[g + 1]
        if b - a >= min_count:
            s = np.sort(buf[a:b])
            for t in range(qs.size):
                out[t, g] = pct_f64(s, qs[t])
    return out


# ----------------------------------------------------------------- кластеры


@njit(cache=True)
def _find(parent: np.ndarray, a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@njit(cache=True)
def cluster_cells(uk: np.ndarray) -> np.ndarray:
    """Связные компоненты занятых клеток (8-связность). uk — отсортированные ключи
    клеток cs·1000 + (cl + 500). Возвращает номер компоненты (корень) для клетки."""
    n = uk.size
    parent = np.arange(n)
    for i in range(n):
        k = uk[i]
        a = k // 1000
        b = k - a * 1000
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                key = (a + da) * 1000 + b + db
                j = np.searchsorted(uk, key)
                if j < n and uk[j] == key:
                    ra = _find(parent, i)
                    rb = _find(parent, j)
                    if ra != rb:
                        parent[ra] = rb
    comp = np.empty(n, np.int64)
    for i in range(n):
        comp[i] = _find(parent, i)
    return comp


@njit(cache=True)
def component_stats(
    comp: np.ndarray,
    s: np.ndarray,
    lat: np.ndarray,
    hh: np.ndarray,
    ring: np.ndarray,
    strong: np.ndarray,
    ncomp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Признаки компонент: целые (n, колец, сильных) и вещественные (min/max s, медиана/
    min/max lat, min/max hh) — всё, что нужно правилу отбора и выходу."""
    ist = np.zeros((ncomp, 3), np.int64)
    fst = np.empty((ncomp, 7), np.float64)
    for c in range(ncomp):
        fst[c, 0] = np.inf
        fst[c, 1] = -np.inf
        fst[c, 3] = np.inf
        fst[c, 4] = -np.inf
        fst[c, 5] = np.inf
        fst[c, 6] = -np.inf
    order = np.argsort(comp, kind="mergesort")
    lo = 0
    n = comp.size
    while lo < n:
        c = comp[order[lo]]
        hi = lo
        while hi < n and comp[order[hi]] == c:
            hi += 1
        ids = order[lo:hi]
        m = hi - lo
        ist[c, 0] = m
        rs = np.sort(ring[ids])
        nu = 1
        for t in range(1, m):
            if rs[t] != rs[t - 1]:
                nu += 1
        ist[c, 1] = nu
        ns = 0
        for t in range(m):
            if strong[ids[t]] > 0:
                ns += 1
        ist[c, 2] = ns
        lv = np.sort(lat[ids])
        fst[c, 2] = lv[m // 2] if m % 2 == 1 else (lv[m // 2 - 1] + lv[m // 2]) / 2
        fst[c, 3] = lv[0]
        fst[c, 4] = lv[m - 1]
        for t in range(m):
            i = ids[t]
            fst[c, 0] = min(fst[c, 0], np.float64(s[i]))
            fst[c, 1] = max(fst[c, 1], np.float64(s[i]))
            fst[c, 5] = min(fst[c, 5], hh[i])
            fst[c, 6] = max(fst[c, 6], hh[i])
        lo = hi
    return ist, fst
