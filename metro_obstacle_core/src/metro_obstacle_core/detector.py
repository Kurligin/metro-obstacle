"""Детектор посторонних объектов в габарите: конвейер на один кадр.

1. Нормализация осей (x-вперёд, y-влево), отбрасывание невалидных точек.
2. Раздвоенные лучи → «сильные» точки; профиль стен → пройденный путь за кадр.
3. Автокалибровка на первом кадре: плоскость полотна (RANSAC).
4. Ось пути: рельсы вблизи, стены вдали; высота полотна вдоль оси.
5. Модель полотна (профиль, опора — головки рельсов) → высота каждой точки.
6. Кандидаты в габарите с запасом по дальности → кластеры → подтверждение M из N.

Внутри всё в нормализованных осях; наружу — в осях сообщения.
Координаты пути: s ≈ x (вдоль), lat = y − y_оси(x), h = z − z_оси(x).
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from metro_obstacle_core import kernels as K
from metro_obstacle_core.axis import AxisState, Points, axis_walls, floor_along
from metro_obstacle_core.calib import (
    AXIS_CODE,
    CALIB_MIN_POINTS,
    calib_zone,
    calib_zone_xy,
    ground_plane,
    guess_forward_axis,
    leveling_rotation,
    refine_plane,
)
from metro_obstacle_core.floor import FloorModel
from metro_obstacle_core.params import (
    FALLBACK_MODE,
    FORWARD_AXES,
    MODES,
    VERIFY_FIELDS,
    Params,
    make_params,
)
from metro_obstacle_core.speed import SpeedEstimator, profile
from metro_obstacle_core.verifier import VerifierError, load_verifier

log = logging.getLogger(__name__)

RAIL_LAT = K.RAIL_LAT
S_MIN = 5.0  # ближе — мёртвая зона у лобовой части поезда
S_STRONG_MAX = 100.0  # сильные точки (раздвоенные лучи) учитываются до этой дальности
LAT_WORK = 1.3  # дальше от оси точки не нужны ни модели полотна, ни кандидатам
# Меньше точек в рабочей зоне — кадр вырожденный (пустой, обрезанный, сбой драйвера):
# результат «ничего не видно», состояние (ось, полотно, история) не трогаем.
MIN_FRAME_POINTS = 50
PLANE_UPDATE_POINTS = 5000  # точек зоны калибровки на одну оценку плоскости на ходу


@dataclass
class Obstacle:
    """Объект (кластер точек) в габарите.

    confidence — эвристика 0..1, а не вероятность:
        0.5·(голосов / N) + 0.3·(1 − e^(−точек/6)) + 0.2·min(колец/3, 1),
    где голоса — в скольких из последних N кадров объект был на том же месте пути
    (с учётом пройденного пути). Подтверждённый 3 из 5 объект из 10 точек на 3 кольцах
    получает ≈ 0.74, одиночный кандидат из 2 точек одного кольца — ≈ 0.25.
    """

    distance: float  # м, вдоль оси пути от лидара (s ближайшей точки кластера)
    lateral: float  # м, смещение от оси пути (+ влево), медиана по точкам
    height: float  # м, верх объекта над опорой (головками рельсов)
    length: float  # м, протяжённость вдоль пути
    width: float  # м, поперёк пути
    points: int
    confidence: float
    confirmed: bool  # прошёл подтверждение M из N
    position: tuple[float, float, float]  # центр рамки точек в осях сообщения, м
    # м, вертикальная протяжённость точек объекта (у висящего кабеля мала, а height —
    # верх над опорой — велик); высота рамки объекта
    z_extent: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "distance": _num(self.distance),
            "lateral": _num(self.lateral),
            "height": _num(self.height),
            "length": _num(self.length),
            "width": _num(self.width),
            "points": int(self.points),
            "confidence": _num(self.confidence),
            "confirmed": bool(self.confirmed),
            "position": [_num(v) for v in self.position],
            "z_extent": _num(self.z_extent),
        }


@dataclass
class FrameResult:
    """Результат кадра.

    confidence: при подтверждённом объекте — уверенность ближайшего подтверждённого;
    иначе — наибольшая среди неподтверждённых кандидатов (0, если их нет).
    """

    obstacle: bool  # есть подтверждённый объект
    distance: float  # до ближайшего подтверждённого, м; math.nan если нет
    confidence: float
    level: int  # 0 свободно / 1 только неподтверждённые кандидаты / 2 подтверждено
    visible_range: float  # м, докуда построен коридор (длина оси)
    processing_ms: float
    mode: str
    calibrated: bool
    obstacles: list[Obstacle] = field(default_factory=list)
    corridor: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    corridor_half_width: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    def to_json(self) -> dict[str, Any]:
        """Для results.jsonl: без коридора, только встроенные типы; NaN → None."""
        return {
            "obstacle": bool(self.obstacle),
            "distance": _num(self.distance),
            "confidence": _num(self.confidence),
            "level": int(self.level),
            "visible_range": _num(self.visible_range),
            "processing_ms": _num(self.processing_ms),
            "mode": str(self.mode),
            "calibrated": bool(self.calibrated),
            "obstacles": [o.to_json() for o in self.obstacles],
        }


def intensity_u8(iv: np.ndarray) -> np.ndarray:
    """Интенсивность → uint8 0..255 для профиля стен.

    Драйверы отдают её по-разному: 0..255, float 0..1, 16 бит. Без приведения float
    0..1 после перевода в uint8 становится нулями, и профиль стен (скорость) пуст.
    Целые 0..255 не меняются (так на стенде).
    """
    iv = np.asarray(iv)
    if iv.dtype == np.uint8:
        return iv
    v = np.nan_to_num(iv.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(v.max()) if len(v) else 0.0
    if vmax > 255.0:
        v = v * (255.0 / vmax)
    elif 0.0 < vmax <= 1.0 and iv.dtype.kind == "f":
        v = v * 255.0
    return np.clip(v, 0, 255).astype(np.uint8)


def _num(v: float) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


@dataclass
class _Det:
    """Кластер-кандидат текущего кадра (нормализованные оси)."""

    s: float
    lat: float
    h_top: float
    n: int
    rings: int
    s_max: float
    lat_min: float
    lat_max: float
    center: tuple[float, float, float]
    z_extent: float = 0.0
    votes: int = 1
    confirmed: bool = False


def _margin(cfg: Params, s):
    """Запас по ширине габарита: margin0 + margin_k·(s − margin_s0)⁺ — на ошибку оси вдали.

    Для скаляра — питоновский float (не numpy-скаляр: тот поменял бы тип арифметики с
    float32-массивами точек, см. Params.__post_init__), для массива — массив.
    """
    m = cfg.margin0 + cfg.margin_k * np.maximum(s - cfg.margin_s0, 0.0)
    return float(m) if np.ndim(m) == 0 else m


class Detector:
    """Детектор кадра за кадром. Состояние: калибровка, ось, профиль полотна, история.

    mode — «default» | «geometry» | «strict» | «soft» (см. params.MODES); overrides — поля
    Params. forward_axis — ось «вперёд» в системе сообщения: auto | x | -x | y | -y.

    Если в параметрах задан verify_thr, детектор сам загружает проверяющий классификатор
    (verify_model или модель из пакета). Не загрузился (нет lightgbm, нет файла) —
    предупреждение в лог; режим default (пороги которого рассчитаны на классификатор)
    работает как geometry, прочие режимы — как есть, без классификатора. Прочие
    переопределения сохраняются; причина — в verifier_error, фактический режим — в mode.
    """

    def __init__(self, mode: str = "default", forward_axis: str = "auto", **overrides: Any):
        if forward_axis not in FORWARD_AXES:
            raise ValueError(f"forward_axis must be one of {FORWARD_AXES}, got {forward_axis!r}")
        self.mode = mode
        self.params: Params = make_params(mode, **overrides)
        # проверяющий классификатор: f(признаки (n, 7)) → оценки (n,); признаки — как
        # у Obstacle: distance, lateral, height, length, width, z_extent, points
        self.verifier: Callable[[np.ndarray], np.ndarray] | None = None
        self.verifier_note = "off"
        self.verifier_error: str | None = None  # классификатор нужен, но не загрузился
        if self.params.verify_thr is not None:
            try:
                v = load_verifier(self.params.verify_model)
                self.verifier = v
                self.verifier_note = f"on: {v.path} (threshold {self.params.verify_thr:.4f})"
            except VerifierError as e:
                # у default пороги рассчитаны на классификатор — без него это geometry;
                # у прочих режимов классификатор задан переопределением — просто без него
                fb = FALLBACK_MODE if MODES[mode].verify_thr is not None else mode
                log.warning("verifier unavailable (%s); running mode %r", e, fb)
                self.verifier_error = str(e)
                self.verifier_note = f"off: {e}; running mode {fb!r}"
                self.mode = fb
                rest = {k: val for k, val in overrides.items() if k not in VERIFY_FIELDS}
                self.params = make_params(fb, **rest)
        self.forward_axis = forward_axis
        self._fwd: str | None = None if forward_axis == "auto" else forward_axis
        self.plane: np.ndarray | None = None  # [nx, ny, nz, d] в нормализованных осях
        self._R: np.ndarray | None = None  # выравнивание по полотну (level=True)
        self._calib_buf: list[np.ndarray] = []
        self._axis = AxisState(
            fit=self.params.axis_fit,
            smooth_lam=self.params.smooth_lam,
            smooth_gap=self.params.smooth_gap,
            wall_ext=self.params.wall_ext,
        )
        self._plane_rng = np.random.default_rng(0)
        # оценки плоскости [nx, ny, nz, d] для медианы (уточнение на ходу, plane_every)
        self._plane_est: deque[np.ndarray] = deque(maxlen=self.params.plane_buf)
        self._plane_frames = 0  # кадров после калибровки (счётчик для plane_every)
        self.floor = FloorModel(
            self.params.floor_alpha,
            self.params.floor_fast,
            self.params.dil_k,
            self.params.delta_clip,
        )
        self.speed = SpeedEstimator(legacy=self.params.speed_legacy)
        self._dhist: list[list[float]] = []  # s детекций последних N кадров
        self.debug = False  # стенд: сохранять промежуточные массивы кадра в self.dbg
        self.dbg: dict | None = None
        self.last_axis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def resolved_forward_axis(self) -> str | None:
        return self._fwd

    # ------------------------------------------------------------ оси сообщения

    @staticmethod
    def _apply(R: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        Rf = R.astype(np.float32)
        return (
            Rf[0, 0] * x + Rf[0, 1] * y + Rf[0, 2] * z,
            Rf[1, 0] * x + Rf[1, 1] * y + Rf[1, 2] * z,
            Rf[2, 0] * x + Rf[2, 1] * y + Rf[2, 2] * z,
        )

    def _unlevel(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Из выровненных осей обратно в нормализованные (R ортогональна: R⁻¹ = Rᵀ)."""
        if self._R is None:
            return x, y, z
        Rt = self._R.T
        return (
            Rt[0, 0] * x + Rt[0, 1] * y + Rt[0, 2] * z,
            Rt[1, 0] * x + Rt[1, 1] * y + Rt[1, 2] * z,
            Rt[2, 0] * x + Rt[2, 1] * y + Rt[2, 2] * z,
        )

    def _to_msg(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = self._fwd
        if f == "x":
            return x, y
        if f == "-x":
            return -x, -y
        if f == "y":
            return -y, x
        return y, -x  # -y: x_n = −y, y_n = x

    def _empty(self, t0: float, visible: float = 0.0) -> FrameResult:
        return FrameResult(
            obstacle=False,
            distance=math.nan,
            confidence=0.0,
            level=0,
            visible_range=visible,
            processing_ms=(time.perf_counter() - t0) * 1e3,
            mode=self.mode,
            calibrated=self.plane is not None,
        )

    # ------------------------------------------------------------ кадр

    def process(
        self,
        xyz: np.ndarray,
        ring: np.ndarray,
        intensity: np.ndarray | None,
        stamp: float,
    ) -> FrameResult:
        """Обработать кадр. xyz — (N, 3) в осях сообщения (NaN/нули отбрасываются),
        ring — (N,) номер кольца, intensity — (N,) или None, stamp — время кадра, с.

        stamp алгоритм не использует: пройденный путь меряется за кадр (сдвиг профиля
        стен в окне 0–3 м), как на стенде. Пропуск кадров допустим, пока поезд
        проходит между обработанными кадрами меньше 3 м.
        """
        t0 = time.perf_counter()
        cfg = self.params
        xyz = np.ascontiguousarray(xyz, dtype=np.float32).reshape(-1, 3)
        if self._fwd is None:
            self._fwd = guess_forward_axis(xyz)
            if self._fwd is None:
                return self._empty(t0)
        x, y, z, idx = K.normalize(xyz, AXIS_CODE[self._fwd])
        if self._R is not None:
            x, y, z = self._apply(self._R, x, y, z)
        rg = np.asarray(ring)[idx].astype(np.int64)
        if intensity is None:
            # без интенсивности профиль стен пуст: сдвиг за кадр = 0, подтверждение
            # работает без учёта пройденного пути (хуже на скорости)
            it = np.zeros(len(idx), np.uint8)
        else:
            it = intensity_u8(np.asarray(intensity)[idx])

        strong_all = K.split_mask(x, y, z, rg) if cfg.split else None
        crop = (x > 2) & (x < cfg.max_range) & (np.abs(y) < 8)
        cx, cy, cz, crg = x[crop], y[crop], z[crop], rg[crop]
        cst = (
            strong_all[crop].astype(np.int8)
            if strong_all is not None
            else np.zeros(len(cx), np.int8)
        )
        if len(cx) < MIN_FRAME_POINTS:
            return self._empty(t0)
        if self.plane is None:
            # Калибровка — только по пригодным кадрам: на пустом или почти пустом кадре
            # плоскость не угадываем, а ждём следующий (как с осью «вперёд»).
            cp = np.stack([cx, cy, cz], 1)
            if cfg.calib_frames <= 1 and not cfg.level:
                gp = ground_plane(cp, self._plane_rng)
                if gp is None:
                    return self._empty(t0)
                n, d = gp
                self.plane = np.array([n[0], n[1], n[2], d], dtype=np.float64)
                self._plane_est.append(self.plane.copy())
            else:
                if int(np.count_nonzero(calib_zone(cp))) < CALIB_MIN_POINTS:
                    return self._empty(t0)  # непригодный кадр в накопление не идёт
                self._calib_buf.append(cp)
                if len(self._calib_buf) < cfg.calib_frames:
                    return self._empty(t0)
                acc = np.concatenate(self._calib_buf)
                self._calib_buf = []
                gp = ground_plane(acc, self._plane_rng)
                if gp is None:
                    return self._empty(t0)
                n, d = refine_plane(acc, np.asarray(gp[0], float), float(gp[1]))
                if cfg.level:
                    # дальше всё — в осях, где полотно горизонтально: крен и тангаж
                    # установки больше не протекают в высоты, полосы стен и рельсов
                    self._R = leveling_rotation(n)
                    # поворот не меняет расстояние от лидара до плоскости: в выровненных
                    # осях полотно — ровно z = −d
                    n = np.array([0.0, 0.0, 1.0])
                    self.plane = np.array([n[0], n[1], n[2], d], dtype=np.float64)
                    return self._empty(t0)  # кадр калибровки — со следующего уже выровнено
                self.plane = np.array([n[0], n[1], n[2], d], dtype=np.float64)
        elif cfg.plane_every > 0 and cfg.calib_frames <= 1 and not cfg.level:
            self._update_plane(cx, cy, cz)
        plane = self.plane

        order, start = K.bucket_by_x(cx, math.ceil(cfg.max_range) + 1)
        pts = Points(cx[order], cy[order], cz[order], start)
        xs, ys = axis_walls(pts, plane, self._axis, cfg.max_range)
        zs = floor_along(pts, xs, ys, plane)
        self.last_axis = (xs, ys, zs)
        visible = float(xs[-1]) if len(xs) else 0.0
        if len(xs) < 3:
            # как в прототипе: без оси кадр не участвует ни в скорости, ни в истории
            return self._result(t0, [], xs, ys, zs, visible)

        # точки у оси: грубый отбор, затем точные lat/h (как в прототипе, np.interp)
        sel = K.interp_prefilter(pts.x, pts.y, xs, ys, float(xs[-1]), LAT_WORK)
        px, py, pz = pts.x[sel], pts.y[sel], pts.z[sel]
        lat = py - np.interp(px, xs, ys)
        keep = np.abs(lat) < LAT_WORK
        sel, px, py, pz, lat = sel[keep], px[keep], py[keep], pz[keep], lat[keep]
        oi = order[sel]
        prg, pst = crg[oi], cst[oi]
        h = pz - np.interp(px, xs, zs)
        s = px
        self.floor.update(lat, h, s)
        hh = self.floor.height(lat, h, s)
        thr = cfg.h_min
        half_w = cfg.gauge_half_width - _margin(cfg, s)  # запас на ошибку оси вдали
        thr = thr + 0.0015 * s  # шум высоты растёт с дальностью
        rails = (np.abs(np.abs(lat) - RAIL_LAT) < 0.12) & (hh < 0.15 + 0.0015 * s)
        hh = np.where(rails, -1.0, hh)
        s_end = xs[-1] - cfg.end_guard
        # верх габарита над полотном вдоль оси (h от zs); от головки рельса — выше на
        # выученный уровень головок
        top = cfg.gauge_height
        if cfg.gauge_from_rail_head:
            top = top + self.floor.rail_head
        if cfg.top_k > 0:  # высоты вдали занижены (ошибка опоры/тангажа) — верх опускается
            top = top - cfg.top_k * np.maximum(s - cfg.top_s0, 0.0)
        cand = (np.abs(lat) < half_w) & (hh > thr) & (h < top) & (s >= S_MIN) & (s < s_end)
        # сильные точки: в габарите, выше уровня головок рельсов, до 100 м
        sc = (
            (pst > 0)
            & (np.abs(lat) < half_w)
            & (hh > 0.05)
            & (h < top)
            & (s >= S_MIN)
            & (s < min(s_end, S_STRONG_MAX))
        )
        cand = cand | sc
        if self.debug:  # отладка стенда: что с точками у оси на этом кадре
            self.dbg = {
                "px": px, "py": py, "pz": pz, "lat": lat, "h": h, "hh": hh,
                "half_w": half_w, "thr": thr, "top": top, "s_end": s_end, "cand": cand,
            }  # fmt: skip
        dets = self._cluster(s[cand], lat[cand], hh[cand], prg[cand], pst[cand], px, py, pz, cand)
        if cfg.edge_reject:
            dets = self._reject_edge(dets, s, lat, hh, thr)
        if self.verifier is not None and cfg.verify_thr is not None and dets:
            chk = [d for d in dets if d.s >= cfg.verify_smin]
            if chk:
                F = np.array(
                    [
                        [d.s, d.lat, d.h_top, d.s_max - d.s, d.lat_max - d.lat_min, d.z_extent, d.n]
                        for d in chk
                    ]
                )
                ok = np.asarray(self.verifier(F)) >= cfg.verify_thr
                drop = {id(d) for d, o in zip(chk, ok, strict=True) if not o}
                dets = [d for d in dets if id(d) not in drop]
        if cfg.far_ext > 0 and len(xs) >= 10:
            dets += self._far_beyond_axis(pts, order, crg, cst, xs, ys, zs, s_end)
            dets.sort(key=lambda d: d.s)

        M, N = cfg.confirm
        if N > 1:
            pr = profile(x, y, z, it)
            shift = self.speed.step(pr)
            self._dhist = [[q - shift for q in hq] for hq in self._dhist]
            self._dhist.append([d.s for d in dets])
            self._dhist = self._dhist[-N:]
            ga, gb = cfg.gate_a, cfg.gate_b
            for d in dets:
                d.votes = sum(any(abs(d.s - q) < ga + gb * d.s for q in hq) for hq in self._dhist)
                m_need = M
                if cfg.confirm_far is not None and d.s > cfg.far_s:
                    m_need = cfg.confirm_far[0]
                d.confirmed = d.votes >= m_need
        else:
            for d in dets:
                d.confirmed = True
        return self._result(t0, dets, xs, ys, zs, visible)

    def _update_plane(self, cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> None:
        """Ещё одна оценка плоскости полотна (по расписанию plane_dense / plane_every) и
        плоскость = покомпонентная медиана накопленных оценок, нормаль перенормирована.

        Медиана устойчива к кадрам, где ближняя зона — не полотно (стрелка, платформа,
        поезд рядом), и меняется плавно, поэтому ось и модель полотна не скачут.
        """
        cfg = self.params
        if cfg.plane_freeze and len(self._plane_est) >= cfg.plane_buf:
            return
        self._plane_frames += 1
        k = self._plane_frames - cfg.plane_dense
        if k > 0 and k % cfg.plane_every:
            return
        zi = np.flatnonzero(calib_zone_xy(cx, cy))
        if len(zi) < CALIB_MIN_POINTS:
            return
        # прореживание до ~PLANE_UPDATE_POINTS: точность одной оценки задаёт выборка
        # RANSAC из 3 точек, а не их число, — стоимость же падает вдвое-втрое
        zi = zi[:: max(1, len(zi) // PLANE_UPDATE_POINTS)]
        gp = ground_plane(np.stack([cx[zi], cy[zi], cz[zi]], 1), self._plane_rng)
        if gp is None:
            return
        n, d = gp
        self._plane_est.append(np.array([n[0], n[1], n[2], d], dtype=np.float64))
        med = np.median(np.asarray(self._plane_est), axis=0)
        nn = med[:3] / np.linalg.norm(med[:3])
        if cfg.plane_tol_deg > 0 and self.plane is not None:
            # гистерезис: текущая плоскость меняется, только если медиана ушла от неё
            # заметно (неудачный старт), — мелкий шум оценок ось и полотно не трогает
            cur = self.plane[:3] / np.linalg.norm(self.plane[:3])
            ang = math.degrees(math.acos(min(1.0, abs(float(cur @ nn)))))
            if ang < cfg.plane_tol_deg and abs(float(med[3] - self.plane[3])) < cfg.plane_tol_d:
                return
        self.plane = np.array([nn[0], nn[1], nn[2], med[3]], dtype=np.float64)

    def _reject_edge(self, dets: list[_Det], s, lat, hh, thr) -> list[_Det]:
        """Стена или портал, зашедшие в габарит из-за неточной оси, касаются его края и
        продолжаются за ним; настоящий объект в габарите изолирован. Кластер у края
        отбрасывается, если за краем (полоса 0.25–0.55 м) на той же дальности над
        полотном точек не меньше половины его собственных."""
        cfg = self.params
        out = []
        for d in dets:
            if d.s < cfg.edge_smin:
                out.append(d)
                continue
            hw = cfg.gauge_half_width - _margin(cfg, d.s)
            near = (s > d.s - 0.5) & (s < d.s_max + 0.5) & (hh > (thr[0] if np.ndim(thr) else thr))
            reject = False
            for side, edge in ((1, d.lat_max), (-1, -d.lat_min)):
                if edge >= hw - 0.1:
                    # полоса отступает от края на 0.25 м: сам объект (шириной до ~0.5 м),
                    # слегка вылезший за суженный край, в неё не попадает, а стена тянется
                    band = near & (side * lat >= hw + 0.25) & (side * lat < hw + 0.55)
                    if band.sum() >= max(3, 0.5 * d.n):
                        reject = True
            if not reject:
                out.append(d)
        return out

    def _far_beyond_axis(self, pts, order, crg, cst, xs, ys, zs, s_end: float) -> list[_Det]:
        """Компактные объекты за концом оси (параметр far_ext).

        Ось кончается раньше, чем лидар перестаёт видеть тоннель (кубика не описывает
        S-поворот). За концом коридор продолжается по касательной к последним 20 м оси,
        сужаясь на 1 см на метр. Стены поворота попадают в такой коридор, но это длинные
        поверхности, уходящие за его край; человек, ящик, тележка — компактны. Поэтому
        объект принимается, только если он короче far_max_len вдоль пути и за краем
        коридора рядом с ним (та же дальность и высота) меньше точек, чем у него самого.
        """
        cfg = self.params
        x0 = float(xs[-1])
        m = xs >= x0 - 20.0
        if m.sum() < 3:
            return []
        by = np.polyfit(xs[m], ys[m], 1)
        bz = np.polyfit(xs[m], zs[m], 1)
        lo = np.searchsorted(pts.x, s_end)
        hi = np.searchsorted(pts.x, x0 + cfg.far_ext)
        if hi - lo < cfg.min_pts:
            return []
        fx, fy, fz = pts.x[lo:hi], pts.y[lo:hi], pts.z[lo:hi]
        oi = order[lo:hi]
        lat = fy - (by[0] * fx + by[1])
        h = fz - (bz[0] * fx + bz[1])
        w0 = cfg.gauge_half_width - _margin(cfg, x0)
        half_w = w0 - 0.01 * np.maximum(fx - x0, 0.0)
        top = cfg.gauge_height
        band = (h > cfg.far_h_min) & (h < top)
        cand = band & (np.abs(lat) < half_w)
        if cand.sum() < cfg.min_pts:
            return []
        dets = self._cluster(
            fx[cand], lat[cand], h[cand], crg[oi][cand], cst[oi][cand], fx, fy, fz, cand
        )
        out = []
        for d in dets:
            if d.s_max - d.s > cfg.far_max_len or d.n < cfg.min_pts or d.rings < cfg.min_rings:
                continue
            near = (fx > d.s - 1.0) & (fx < d.s_max + 1.0) & band
            outside = near & (np.abs(lat) >= half_w) & (np.abs(lat) < half_w + 1.0)
            if outside.sum() >= d.n:
                continue  # продолжается за край коридора — стена поворота
            out.append(d)
        return out

    def _cluster(
        self,
        s: np.ndarray,
        lat: np.ndarray,
        hh: np.ndarray,
        ring: np.ndarray,
        strong: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        pz: np.ndarray,
        cand: np.ndarray,
    ) -> list[_Det]:
        """Кластеры в сетке cell_s × cell_lat (8-связность) и правило отбора.

        Обычный кластер — ≥ min_pts точек и ≥ min_rings колец. В «чистой зоне»
        (весь кластер выше clean_h над опорой и ближе clean_smax) и при ≥ 2 сильных
        точках — достаточно clean_min_pts точек.
        """
        cfg = self.params
        if not len(s):
            return []
        cs = np.floor(s / cfg.cell_s).astype(int)
        cl = np.floor(lat / cfg.cell_lat).astype(int)
        keys = cs * 1000 + (cl + 500)
        uk, inv = np.unique(keys, return_inverse=True)
        roots = K.cluster_cells(uk)
        _, comp_cell = np.unique(roots, return_inverse=True)
        comp = comp_cell[np.ravel(inv)]
        ncomp = int(comp.max()) + 1
        ist, fst = K.component_stats(comp, s, lat, hh, ring, strong, ncomp)
        cx, cy, cz = px[cand], py[cand], pz[cand]
        dets = []
        for c in range(ncomp):
            n, rings, n_strong = (int(v) for v in ist[c])
            s_min, s_max, lat_med, lat_lo, lat_hi, h_lo, h_hi = (float(v) for v in fst[c])
            in_clean = cfg.clean_h is not None and h_lo > cfg.clean_h and s_max < cfg.clean_smax
            in_clean = in_clean or n_strong >= 2  # ≥ 2 сильных точек — как «чистая зона»
            if in_clean:
                if n < cfg.clean_min_pts:
                    continue
            else:
                if n < cfg.min_pts or rings < cfg.min_rings:
                    continue
            m = comp == c
            center = tuple(
                float(0.5 * (float(v[m].min()) + float(v[m].max()))) for v in (cx, cy, cz)
            )
            z_extent = float(cz[m].max()) - float(cz[m].min())
            dets.append(
                _Det(
                    s=s_min,
                    lat=lat_med,
                    h_top=h_hi,
                    n=n,
                    rings=rings,
                    s_max=s_max,
                    lat_min=lat_lo,
                    lat_max=lat_hi,
                    center=center,  # type: ignore[arg-type]
                    z_extent=z_extent,
                )
            )
        dets.sort(key=lambda d: d.s)
        return dets

    def _result(
        self,
        t0: float,
        dets: list[_Det],
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        visible: float,
    ) -> FrameResult:
        N = self.params.confirm[1]
        obstacles = []
        for d in dets:
            conf = (
                0.5 * min(d.votes / N, 1.0)
                + 0.3 * (1.0 - math.exp(-d.n / 6.0))
                + 0.2 * min(d.rings / 3.0, 1.0)
            )
            ux, uy, uz = self._unlevel(
                np.array([d.center[0]]), np.array([d.center[1]]), np.array([d.center[2]])
            )
            mx, my = self._to_msg(ux, uy)
            obstacles.append(
                Obstacle(
                    distance=d.s,
                    lateral=d.lat,
                    height=d.h_top,
                    length=d.s_max - d.s,
                    width=d.lat_max - d.lat_min,
                    points=d.n,
                    confidence=round(min(max(conf, 0.0), 1.0), 3),
                    confirmed=d.confirmed,
                    position=(float(mx[0]), float(my[0]), float(uz[0])),
                    z_extent=d.z_extent,
                )
            )
        warm = self.floor.n_updates >= self.params.warmup_frames
        if not warm:
            # профиль полотна ещё не выучен: объекты — только кандидаты, без тревоги
            for o in obstacles:
                o.confirmed = False
        confirmed = [o for o in obstacles if o.confirmed]
        if confirmed:
            level, distance, confidence = 2, confirmed[0].distance, confirmed[0].confidence
        elif obstacles:
            level, distance = 1, math.nan
            confidence = max(o.confidence for o in obstacles)
        else:
            level, distance, confidence = 0, math.nan, 0.0
        corridor, half_w = self._corridor(xs, ys, zs)
        return FrameResult(
            obstacle=bool(confirmed),
            distance=distance,
            confidence=confidence,
            level=level,
            visible_range=visible,
            processing_ms=(time.perf_counter() - t0) * 1e3,
            mode=self.mode,
            calibrated=self.plane is not None and warm,
            obstacles=obstacles,
            corridor=corridor,
            corridor_half_width=half_w,
        )

    def _corridor(
        self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ось на уровне головок рельсов (полотно + выученный уровень головок + Δ(s))
        в осях сообщения и рабочая полуширина габарита с учётом запаса по дальности."""
        cfg = self.params
        z = zs.astype(np.float64)
        if self.floor.U is not None:
            z = z + self.floor.rail_head
            if self.floor.last_delta is not None and len(xs):
                ub, delta = self.floor.last_delta
                if len(ub):
                    z = z + np.interp(xs, ub * 5.0 + 2.5, delta)
        ux, uy, uz = self._unlevel(xs.astype(np.float64), ys.astype(np.float64), z)
        mx, my = self._to_msg(ux, uy)
        corridor = np.stack([mx, my, uz], 1).astype(np.float32)
        half_w = (cfg.gauge_half_width - _margin(cfg, xs)).astype(np.float32)
        return corridor, half_w
