"""Генератор синтетических препятствий: трассировка лучей по сетке Pandar128.

Истинная ось пути для кадра k — будущая траектория лидара (кадры j > k),
перенесённая в систему кадра k и опущенная на высоту лидара над полотном.
Объект ставится неподвижно в мире на эту ось (или со сдвигом внутри габарита),
поэтому при движении поезда он честно приближается.

Луч = (кольцо, азимут). Кольца и шаг азимута берутся из данных. Если луч
пересёк объект ближе исходной точки (или исходной точки нет), точка заменяется
попаданием в объект; оба отражения (dual return) заменяются одним.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bench.data import CACHE_ROOT, POINT_DTYPE, Bag


@dataclass
class Obj:
    """Ориентированный параллелепипед в системе «ось пути»: s — вдоль пути,
    lat — влево от оси, h — высота над головкой рельса (низ объекта)."""

    kind: str
    size: tuple[float, float, float]  # вдоль пути, поперёк, по высоте, м
    s: float  # положение вдоль пути от начала bag, м
    lat: float = 0.0
    h: float = 0.0
    intensity: int = 20
    yaw: float = 0.0  # поворот вокруг вертикали относительно направления пути, рад


PRESETS = {
    "box_30x10": (0.30, 0.30, 0.10),  # мин. объект ТЗ: 300×100 мм лежит на полотне
    "box_50": (0.5, 0.5, 0.5),
    "person": (0.3, 0.5, 1.7),
    "cable": (0.02, 0.02, 1.0),  # свисающий кабель, низ на h, от свода
    "cart": (1.0, 1.2, 1.0),
}


@dataclass
class BeamModel:
    el: np.ndarray  # угол места каждого кольца, рад
    daz: np.ndarray  # шаг азимута кольца, рад
    az0: np.ndarray  # фаза сетки азимута кольца, рад
    az_lim: tuple[float, float]

    @classmethod
    def from_frame(cls, pts) -> "BeamModel":
        el = np.zeros(128)
        daz = np.full(128, np.radians(0.2))
        az0 = np.zeros(128)
        az = np.arctan2(pts["y"], pts["x"])
        e = np.arctan2(pts["z"], np.hypot(pts["x"], pts["y"]))
        for r in range(128):
            m = pts["ring"] == r
            if m.sum() < 20:
                el[r] = np.nan
                continue
            el[r] = np.median(e[m])
            a = np.unique(np.round(az[m], 6))
            d = np.diff(a)
            d = d[d > 1e-5]
            step = np.median(d) if len(d) else np.radians(0.2)
            daz[r] = step
            az0[r] = np.median(np.mod(a, step))
        return cls(el, daz, az0, (float(az.min()), float(az.max())))


def track_frame(poses: np.ndarray, k: int, lidar_h: float):
    """Ось пути впереди кадра k: массив (s_rel, p_xyz, tangent) в системе кадра k."""
    inv = np.linalg.inv(poses[k])
    rel = np.einsum("ij,njk->nik", inv, poses[k:])
    p = rel[:, :3, 3].copy()
    p[:, 2] -= lidar_h  # от лидара к головке рельса
    ds = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1)]
    s = np.cumsum(ds)
    tan = rel[:, :3, 0]
    return s, p, tan


def obj_corners(o: Obj, s_rel: float, s: np.ndarray, p: np.ndarray, tan: np.ndarray):
    """Центр и базис объекта в системе кадра. None — если ось пути туда не доходит."""
    if s_rel < 0 or s_rel > s[-1] or len(s) < 2:
        return None
    c = np.array([np.interp(s_rel, s, p[:, i]) for i in range(3)])
    t = np.array([np.interp(s_rel, s, tan[:, i]) for i in range(3)])
    t /= np.linalg.norm(t)
    left = np.cross([0, 0, 1.0], t)
    left /= np.linalg.norm(left)
    up = np.cross(t, left)
    if o.yaw:
        cy_, sy_ = np.cos(o.yaw), np.sin(o.yaw)
        t, left = cy_ * t + sy_ * left, -sy_ * t + cy_ * left
    L, W, H = o.size
    left0 = np.cross([0, 0, 1.0], np.array([np.interp(s_rel, s, tan[:, i]) for i in range(3)]))
    left0 /= np.linalg.norm(left0)
    center = c + left0 * o.lat + up * (o.h + H / 2)
    return center, np.stack([t, left, up]), np.array([L, W, H]) / 2


def ray_box(dirs: np.ndarray, center, basis, half) -> np.ndarray:
    """Дальность пересечения лучей из начала координат с ориентированным боксом (inf — мимо)."""
    o = -basis @ center  # начало луча в системе бокса
    d = dirs @ basis.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o) / d
        t2 = (half - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= tmin) & (tmax > 0)
    return np.where(hit, np.maximum(tmin, 0), np.inf)


FOOT_EL = np.radians(0.12)  # размер пятна луча по вертикали ≈ шаг колец в центре
SUB = 5  # подлучей на сторону пятна


def inject(pts: np.ndarray, beams: BeamModel, boxes, dual: bool = True, naive: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Вставить боксы в кадр. Возвращает новый кадр и метки точек (номер бокса или −1).

    Луч имеет пятно (шаг азимута × FOOT_EL), оно сэмплируется SUB×SUB подлучами.
    Доля пятна на объекте f:
      f ≥ 0.8 — оба отражения от объекта (как у лидара на сплошной цели);
      0.2 ≤ f < 0.8 — первое отражение от объекта, последнее — исходный фон
        (частичное попадание: кабель перед стеной);
      f < 0.2 — объект не виден этим лучом.
    """
    if not boxes:
        return pts, np.full(len(pts), -1)
    az = np.arctan2(pts["y"], pts["x"])
    rng = np.sqrt(pts["x"] ** 2 + pts["y"] ** 2 + pts["z"] ** 2)
    new, drop = [], np.zeros(len(pts), bool)
    owner = []
    g = (np.arange(SUB) + 0.5) / SUB - 0.5
    for bi, (center, basis, half) in enumerate(boxes):
        corners = center + (np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * half) @ basis
        caz = np.arctan2(corners[:, 1], corners[:, 0])
        cel = np.arctan2(corners[:, 2], np.hypot(corners[:, 0], corners[:, 1]))
        rings = np.flatnonzero((beams.el >= cel.min() - FOOT_EL) & (beams.el <= cel.max() + FOOT_EL))
        for r in rings:
            st = beams.daz[r]
            a0 = np.ceil((caz.min() - st - beams.az0[r]) / st) * st + beams.az0[r]
            a = np.arange(a0, caz.max() + st + 1e-9, st)
            a = a[(a >= beams.az_lim[0]) & (a <= beams.az_lim[1])]
            e0 = beams.el[r]
            for ai in a:
                # подлучи пятна
                sa = ai + g[:, None] * st
                se = e0 + g[None, :] * FOOT_EL
                sa, se = np.broadcast_arrays(sa, se)
                sa, se = sa.ravel(), se.ravel()
                dirs = np.stack([np.cos(se) * np.cos(sa), np.cos(se) * np.sin(sa), np.sin(se)], 1)
                t = ray_box(dirs, center, basis, half)
                hit = np.isfinite(t)
                f = hit.mean()
                if f < 0.2:
                    continue
                ti = float(np.median(t[hit]))
                di = np.array([np.cos(e0) * np.cos(ai), np.cos(e0) * np.sin(ai), np.sin(e0)])
                m = np.flatnonzero((pts["ring"] == r) & (np.abs(az - ai) < st / 2) & ~drop)
                if naive:
                    # «наивная» вставка: точка объекта добавляется, фон не трогается
                    q = np.zeros(1, POINT_DTYPE)
                    q["x"], q["y"], q["z"] = np.array([np.cos(e0) * np.cos(ai), np.cos(e0) * np.sin(ai), np.sin(e0)]) * ti
                    q["i"], q["ring"] = 20, r
                    new.append(q)
                    owner.append(bi)
                    continue
                if len(m) and rng[m].min() <= ti:
                    continue  # исходная точка ближе — объект закрыт
                q = np.zeros(1, POINT_DTYPE)
                q["x"], q["y"], q["z"] = di * ti
                q["i"], q["ring"] = 20, r
                if f >= 0.8 or not len(m) or not dual:  # dual=False: без второго отражения
                    drop[m] = True
                    new += [q, q.copy()]  # два одинаковых отражения
                    owner += [bi, bi]
                else:
                    # частичное: ближнее исходное отражение заменяется объектом, дальнее остаётся
                    drop[m[np.argmin(rng[m])]] = True
                    new.append(q)
                    owner.append(bi)
    if not new:
        return pts, np.full(len(pts), -1)
    out = np.concatenate([pts[~drop]] + new)
    lab = np.r_[np.full((~drop).sum(), -1), np.array(owner)]
    return out, lab


LIDAR_H = {"doubleT_obstacle": 1.73}


@dataclass
class Scenario:
    bag: str
    objs: list[Obj] = field(default_factory=list)

    @property
    def lidar_h(self) -> float:
        return LIDAR_H.get(self.bag, 1.35)


def axis_arrays(xs, ys, zs):
    """Ось из truth_axis → (s, p, tan) для obj_corners; начало — у лидара."""
    p = np.stack([np.r_[0.0, xs], np.r_[ys[0], ys], np.r_[zs[0], zs]], 1)
    d = np.diff(p, axis=0)
    s = np.r_[0, np.cumsum(np.linalg.norm(d, axis=1))]
    tan = np.vstack([d, d[-1:]])
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    return s, p, tan


def _rail_head_z(xyz, x, y_axis):
    """Уровень головок рельсов у точки оси по точкам кадра (None — не видно)."""
    m = (np.abs(xyz[:, 0] - x) < 3.0) & (np.abs(np.abs(xyz[:, 1] - y_axis) - 0.76) < 0.1)
    z = xyz[m, 2]
    if len(z) < 3:
        return None
    lo = np.percentile(z, 20)
    z = z[z < lo + 0.4]  # без стен/объектов над рельсом
    return float(np.percentile(z, 90))


def _on_rails(xyz, box, o: Obj, xs, ys):
    """Сдвинуть центр объекта по высоте: низ — на уровне головок рельсов (+o.h).

    Высота оси из траектории плывёт (дрейф KISS-ICP по z), поэтому опора —
    сами точки кадра. Если рельсы у объекта не видны — ближайший видимый бин.
    """
    center, basis, half = box
    y_axis = float(np.interp(center[0], xs, ys))
    zr = _rail_head_z(xyz, center[0], y_axis)
    x = center[0]
    while zr is None and x > 5:
        x -= 4.0
        zr = _rail_head_z(xyz, x, float(np.interp(x, xs, ys)))
    if zr is None:
        return center
    c = center.copy()
    c[2] = zr + o.h + half[2]
    return c


def frames_with_objects(sc: Scenario, stride: int = 1, ks=None, dual: bool = True, naive: bool = False):
    """Итератор (k, кадр со вставками, метки точек, [(obj, дальность, n_точек)] ).

    Объекты неподвижны в мире: дистанция по оси = obj.s − пройденный путь.
    Если эталонная ось до объекта не дотягивается — объект в кадр не ставится
    (и в метриках такой кадр по этому объекту не учитывается).
    """
    from bench.truth import truth_axis

    bag = Bag(sc.bag)
    poses = np.load(CACHE_ROOT / f"{sc.bag}.traj.npy")
    s_frames = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    beams = BeamModel.from_frame(bag[0].pts)
    for k in ks if ks is not None else range(0, len(bag), stride):
        fr = bag[k]
        xs, ys, zs = truth_axis(fr.xyz, poses, k, sc.lidar_h)
        boxes, info = [], []
        if len(xs) >= 3:
            s, p, tan = axis_arrays(xs, ys, zs)
            for o in sc.objs:
                b = obj_corners(o, o.s - s_frames[k], s, p, tan)
                if b is not None:
                    b = (_on_rails(fr.xyz, b, o, xs, ys), b[1], b[2])
                    boxes.append(b)
                    info.append([o, float(np.linalg.norm(b[0][:2])), 0])
        pts, lab = inject(fr.pts, beams, boxes, dual, naive)
        for j in range(len(info)):
            info[j][2] = int((lab == j).sum())
        yield k, pts, lab, info
