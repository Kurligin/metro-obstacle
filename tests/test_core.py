"""Юнит-тесты ядра: раздвоенные лучи, центр пути по рельсам, кластеры, ось «вперёд»,
режимы, сериализация, числовые примитивы (бит-в-бит с numpy) и сквозной синтетический кадр."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from metro_obstacle_core.calib import guess_forward_axis
from metro_obstacle_core.params import Params, make_params
from metro_obstacle_core.speed import _EDGES, NB
from metro_obstacle_core.synthetic import to_message_axes, tunnel_points

from metro_obstacle_core import MODES, Detector, FrameResult, Obstacle
from metro_obstacle_core import kernels as K

AXES = ("x", "-x", "y", "-y")
NUMPY2 = int(np.__version__.split(".")[0]) >= 2


# ----------------------------------------------------------------- раздвоенные лучи


def _beam(az_deg: float, r: float, elev_deg: float = -2.0) -> tuple[float, float, float]:
    a, e = math.radians(az_deg), math.radians(elev_deg)
    return r * math.cos(e) * math.cos(a), r * math.cos(e) * math.sin(a), r * math.sin(e)


def _split(points: list[tuple[tuple[float, float, float], int]]) -> np.ndarray:
    xyz = np.array([p for p, _ in points], np.float32)
    ring = np.array([r for _, r in points], np.int64)
    return K.split_mask(xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy(), ring)


def test_split_marks_near_return_of_dual_beam() -> None:
    m = _split([(_beam(10.0, 20.0), 5), (_beam(10.0, 35.0), 5)])
    assert m.tolist() == [True, False]
    # порядок во входе не важен: помечается именно ближнее отражение
    m = _split([(_beam(10.0, 35.0), 5), (_beam(10.0, 20.0), 5)])
    assert m.tolist() == [False, True]


@pytest.mark.parametrize(
    "second",
    [
        (_beam(10.0, 20.5), 5),  # разница дальностей < 1 м — один объект
        (_beam(10.1, 35.0), 5),  # другой азимут (0.1° > 0.03°)
        (_beam(10.0, 35.0), 6),  # другое кольцо
    ],
)
def test_split_ignores_non_dual(second: tuple[tuple[float, float, float], int]) -> None:
    assert not _split([(_beam(10.0, 20.0), 5), second]).any()


def test_split_matches_prototype_on_random_cloud() -> None:
    bd = pytest.importorskip("bench.detect")
    rng = np.random.default_rng(3)
    n = 20000
    az = rng.choice(np.arange(-90, 90, 0.2), n) + rng.normal(0, 0.01, n)
    r = rng.uniform(2, 80, n)
    xyz = np.stack([r * np.cos(np.radians(az)), r * np.sin(np.radians(az)), rng.normal(0, 1, n)], 1)
    xyz = xyz.astype(np.float32)
    ring = rng.integers(0, 16, n)
    ref = bd.split_mask(xyz, ring)
    got = K.split_mask(xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy(), ring)
    assert ref.sum() > 100
    assert np.array_equal(ref, got)


# ----------------------------------------------------------------- центр пути по рельсам


def _slab_scene(center: float, rng: np.random.Generator, obstacle_above: bool = False):
    """Бин 2 м: полотно, две головки рельсов вокруг center, стены; z полотна = 0."""
    parts = [
        np.c_[rng.uniform(-1.5, 1.5, 300), rng.normal(0.0, 0.01, 300)],  # полотно
        np.c_[center + 0.76 + rng.uniform(-0.02, 0.02, 20), np.full(20, 0.17)],
        np.c_[center - 0.76 + rng.uniform(-0.02, 0.02, 20), np.full(20, 0.17)],
        np.c_[np.full(50, center + 2.2), rng.uniform(0.0, 3.0, 50)],
    ]
    if obstacle_above:
        parts.append(np.c_[center + rng.uniform(-0.3, 0.3, 40), rng.uniform(0.6, 1.8, 40)])
    p = np.concatenate(parts)
    return p[:, 0].astype(np.float32), p[:, 1].astype(np.float32)


@pytest.mark.parametrize("center", [0.0, 0.17, -0.25])
def test_rail_center_finds_track_center(center: float) -> None:
    y, z = _slab_scene(center, np.random.default_rng(1))
    got = K.rail_center(y, z, 0, len(y), 0.0, 0.0, 0.35)
    assert abs(got - center) <= 0.03


def test_rail_center_needs_both_rails() -> None:
    y, z = _slab_scene(0.1, np.random.default_rng(2))
    keep = ~((z > 0.1) & (y < 0))  # убрали правую головку
    assert math.isnan(K.rail_center(y[keep], z[keep], 0, int(keep.sum()), 0.0, 0.0, 0.35))


def test_rail_center_rejects_occupied_track() -> None:
    """Над путём что-то стоит — «пара головок» здесь не ось (так не цепляемся за соседний
    путь, над которым стоит поезд)."""
    y, z = _slab_scene(0.0, np.random.default_rng(4), obstacle_above=True)
    assert math.isnan(K.rail_center(y, z, 0, len(y), 0.0, 0.0, 0.35))


def test_near_axis_follows_offset_track() -> None:
    rng = np.random.default_rng(5)
    xyz, _, _ = tunnel_points(rng, length=40.0, n=80000, center=0.2)
    x, y, z = (np.ascontiguousarray(xyz[:, i]) for i in range(3))
    order, start = K.bucket_by_x(x, 201)
    plane = np.array([0.0, 0.0, 1.0, 1.35])
    X, Y = K.near_rail_centers(y[order], z[order], start, plane, np.zeros(3), 30.0)
    assert len(X) >= 10
    assert np.all(np.abs(Y - 0.2) <= 0.03)


# ----------------------------------------------------------------- кластеры


def test_cluster_cells_8_connectivity() -> None:
    def key(cs: int, cl: int) -> int:
        return cs * 1000 + (cl + 500)

    uk = np.array(sorted([key(10, 0), key(11, 1), key(12, 2), key(20, 0), key(20, 2)]))
    comp = K.cluster_cells(uk)
    lab = dict(zip(uk.tolist(), comp.tolist(), strict=True))
    assert lab[key(10, 0)] == lab[key(11, 1)] == lab[key(12, 2)]  # по диагонали — связны
    assert lab[key(20, 0)] != lab[key(20, 2)]  # через клетку — нет
    assert lab[key(20, 0)] != lab[key(10, 0)]


def _cluster(det: Detector, pts: list[tuple[float, float, float, int, int]]):
    s, lat, hh, ring, strong = (np.array(v) for v in zip(*pts, strict=True))
    s = s.astype(np.float32)
    n = len(s)
    return det._cluster(
        s, lat, hh, ring.astype(np.int64), strong.astype(np.int8),
        s, lat.astype(np.float32), hh.astype(np.float32), np.ones(n, bool),
    )  # fmt: skip


def test_cluster_rules_regular_clean_and_strong() -> None:
    det = Detector("geometry", "-y", warmup_frames=0)
    # обычный: 3 точки на 2 кольцах — да; 2 точки — нет; 3 точки одного кольца — нет
    assert (
        len(_cluster(det, [(30.0, 0.0, 0.3, 1, 0), (30.1, 0.05, 0.3, 2, 0), (30.2, 0, 0.3, 2, 0)]))
        == 1
    )
    assert len(_cluster(det, [(30.0, 0.0, 0.3, 1, 0), (30.1, 0.05, 0.3, 2, 0)])) == 0
    assert (
        len(_cluster(det, [(30.0, 0.0, 0.3, 1, 0), (30.1, 0.0, 0.3, 1, 0), (30.2, 0, 0.3, 1, 0)]))
        == 0
    )
    # «чистая зона» (выше 0.5 м, ближе 60 м): хватает 2 точек одного кольца
    assert len(_cluster(det, [(30.0, 0.0, 0.8, 1, 0), (30.1, 0.0, 0.9, 1, 0)])) == 1
    assert len(_cluster(det, [(70.0, 0.0, 0.8, 1, 0), (70.1, 0.0, 0.9, 1, 0)])) == 0
    # две сильные точки (раздвоенные лучи) — как «чистая зона»
    assert len(_cluster(det, [(30.0, 0.0, 0.2, 1, 1), (30.1, 0.0, 0.2, 1, 1)])) == 1
    # в strict «чистая зона» та же, а clean_h=None её выключает
    off = Detector("strict", "-y", warmup_frames=0, clean_h=None)
    assert len(_cluster(off, [(30.0, 0.0, 0.8, 1, 0), (30.1, 0.0, 0.9, 1, 0)])) == 0


def test_cluster_geometry() -> None:
    det = Detector("geometry", "-y", warmup_frames=0)
    (d,) = _cluster(det, [(30.0, -0.1, 0.3, 1, 0), (30.4, 0.1, 0.5, 2, 0), (30.2, 0.0, 0.4, 3, 0)])
    assert d.n == 3 and d.rings == 3
    assert d.s == pytest.approx(30.0) and d.s_max == pytest.approx(30.4)
    assert d.lat == pytest.approx(0.0) and d.h_top == pytest.approx(0.5)


# ----------------------------------------------------------------- ось «вперёд»


@pytest.mark.parametrize("fwd", AXES)
def test_guess_forward_axis(fwd: str) -> None:
    xyz, _, _ = tunnel_points(np.random.default_rng(6))
    assert guess_forward_axis(to_message_axes(xyz, fwd)) == fwd


def test_guess_forward_axis_needs_far_points() -> None:
    xyz = np.random.default_rng(0).uniform(-3, 3, (1000, 3)).astype(np.float32)
    assert guess_forward_axis(xyz) is None


def test_forward_axis_validation() -> None:
    with pytest.raises(ValueError):
        Detector("geometry", forward_axis="z")


# ----------------------------------------------------------------- режимы


def test_modes_are_bench_configs() -> None:
    """Режимы ядра = конфиги стенда, по которым считались цифры сравнения."""
    bd = pytest.importorskip("bench.detect")
    cfg = {c.name: c for c in bd.CONFIGS}
    for mode, name in (("geometry", "Z+S c3"), ("strict", "Z c3"), ("soft", "Z c2 m0.002")):
        p, c = MODES[mode], cfg[name]
        assert c.axis == "walls" and c.floor == "profile" and c.guards
        assert c.accumulate == 1 and c.persist == 1 and not c.beam_bg and not c.evidence
        assert c.axis_ext == 0.0
        for f in ("h_min", "min_pts", "min_rings", "cell_s", "cell_lat", "confirm", "margin_k",
                  "clean_h", "clean_smax", "clean_min_pts", "split"):  # fmt: skip
            assert getattr(p, f) == getattr(c, f), (mode, f)
        assert p.gauge_half_width == bd.GAUGE_HALF_W and p.gauge_height == bd.GAUGE_H
        assert p.max_range == bd.XMAX
        # отличия режимов от стенда — только осознанные (BENCH_COMPAT их возвращает)
        assert p.delta_clip == 0.05 and p.warmup_frames == 15


def test_mode_differences() -> None:
    assert MODES["geometry"].split and not MODES["strict"].split and not MODES["soft"].split
    assert MODES["soft"].confirm == (2, 5) and MODES["soft"].margin_k == 0.002
    assert MODES["strict"].confirm == (3, 5)


def test_overrides() -> None:
    p = make_params("geometry", confirm=[2, 4], margin_k=np.float64(0.003))
    assert p.confirm == (2, 4)
    assert type(p.margin_k) is float  # numpy-скаляр поменял бы типы арифметики
    with pytest.raises(TypeError):
        make_params("geometry", nonexistent=1)
    with pytest.raises(ValueError):
        make_params("fast")
    with pytest.raises(ValueError):
        Params(confirm=(4, 3))
    assert Detector("soft", "-y", warmup_frames=0, max_range=150.0).params.max_range == 150.0


# ----------------------------------------------------------------- выход


def test_to_json_serializable_without_corridor() -> None:
    o = Obstacle(40.0, 0.1, 0.5, 0.3, 0.4, 12, 0.8, True, (0.1, -40.0, -0.8))
    r = FrameResult(
        obstacle=False, distance=math.nan, confidence=0.0, level=1, visible_range=120.0,
        processing_ms=12.5, mode="geometry", calibrated=True, obstacles=[o],
        corridor=np.zeros((5, 3), np.float32), corridor_half_width=np.ones(5, np.float32),
    )  # fmt: skip
    js = r.to_json()
    assert "corridor" not in js and "corridor_half_width" not in js
    assert js["distance"] is None
    assert js["obstacles"][0]["position"] == [0.1, -40.0, -0.8]
    json.dumps(js, allow_nan=False)


def _run_synth(fwd: str, box: bool, frames: int = 5, mode: str = "geometry") -> FrameResult:
    rng = np.random.default_rng(7)
    det = Detector(mode, forward_axis="auto", warmup_frames=0)
    res = None
    for k in range(frames):
        b = (40.0, 0.2, 0.5, 0.6) if box else None
        xyz, ring, inten = tunnel_points(rng, box=b, center=0.3)
        res = det.process(to_message_axes(xyz, fwd), ring, inten, 0.1 * k)
        assert det.resolved_forward_axis == fwd
    assert res is not None
    return res


@pytest.mark.parametrize("fwd", AXES)
def test_synthetic_box_confirmed_and_reported_in_message_axes(fwd: str) -> None:
    res = _run_synth(fwd, box=True)
    assert res.calibrated and res.obstacle and res.level == 2
    assert res.distance == pytest.approx(40.0, abs=0.05)
    (o,) = res.obstacles
    assert o.confirmed and o.lateral == pytest.approx(0.2, abs=0.05)
    assert o.height == pytest.approx(0.6, abs=0.1)
    assert 0.0 < o.confidence <= 1.0
    # центр объекта в осях сообщения: вперёд 40 м, влево 0.5 м (ось пути смещена на 0.3)
    exp = to_message_axes(np.array([[40.0, 0.5, 0.0]], np.float32), fwd)[0]
    assert np.allclose(o.position[:2], exp[:2], atol=0.05)
    assert res.corridor.shape[1] == 3 and res.corridor.dtype == np.float32
    assert len(res.corridor) == len(res.corridor_half_width)
    assert res.visible_range > 100.0
    json.dumps(res.to_json(), allow_nan=False)


def test_synthetic_needs_confirmation_and_empty_is_free() -> None:
    first = _run_synth("-y", box=True, frames=1)
    assert first.level == 1 and not first.obstacle and math.isnan(first.distance)
    empty = _run_synth("-y", box=False)
    assert empty.level == 0 and not empty.obstacles and math.isnan(empty.distance)


def test_invalid_points_are_dropped() -> None:
    rng = np.random.default_rng(8)
    xyz, ring, inten = tunnel_points(rng, box=(40.0, 0.2, 0.5, 0.6))
    msg = to_message_axes(xyz, "-y")
    bad = np.array([[np.nan, 1, 1], [0, 0, 0], [np.inf, 0, 0]], np.float32)
    a, b = Detector("geometry", "-y", warmup_frames=0), Detector("geometry", "-y", warmup_frames=0)
    for _ in range(3):
        ra = a.process(msg, ring, inten, 0.0)
        rb = b.process(np.r_[msg, bad], np.r_[ring, [1, 1, 1]], np.r_[inten, [1, 1, 1]], 0.0)
    assert [o.distance for o in ra.obstacles] == [o.distance for o in rb.obstacles]


def test_no_intensity() -> None:
    res = _run_synth("-y", box=True)
    rng = np.random.default_rng(7)
    det = Detector("geometry", "-y", warmup_frames=0)
    for k in range(5):
        xyz, ring, _ = tunnel_points(rng, box=(40.0, 0.2, 0.5, 0.6), center=0.3)
        r = det.process(to_message_axes(xyz, "-y"), ring, None, 0.1 * k)
    assert r.obstacle and r.distance == pytest.approx(res.distance)


# ----------------------------------------------------------------- числовые примитивы


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_percentiles_bitexact_with_numpy(dtype: type) -> None:
    rng = np.random.default_rng(9)
    f = K.pct_f32 if dtype is np.float32 else K.pct_f64
    for n in (1, 2, 3, 7, 10, 101, 1000):
        v = rng.normal(0, 3, n).astype(dtype)
        s = np.sort(v)
        for q in (10.0, 30.0, 50.0, 80.0, 90.0, 97.0):
            ref = np.percentile(v, int(q))
            got = f(s, q)
            if NUMPY2 or dtype is np.float64:
                assert ref.dtype == np.dtype(dtype)
                assert got == ref, (n, q)
            else:
                # numpy 1.x интерполирует float32 во float64; ядро фиксирует поведение
                # numpy 2 (как на стенде), расхождение — в последнем знаке float32
                assert got == pytest.approx(ref, rel=1e-6, abs=1e-6), (n, q)


def test_profile_hist_matches_numpy_histogram() -> None:
    rng = np.random.default_rng(10)
    n = 50000
    x = rng.uniform(3.0, 41.0, n).astype(np.float32)
    x[: len(_EDGES)] = _EDGES  # точки ровно на краях бинов
    y = rng.choice([-2.0, 2.0], n).astype(np.float32)
    z = rng.uniform(-1.4, -0.6, n).astype(np.float32)
    it = rng.integers(0, 256, n).astype(np.uint8)
    cnt, sm = K.profile_hist(x, y, z, it, _EDGES)
    m = (x > 4) & (x < 40) & (y < 0)
    ref_c, _ = np.histogram(x[m], bins=NB, range=(4.0, 40.0))
    ref_s, _ = np.histogram(x[m], bins=NB, range=(4.0, 40.0), weights=it[m].astype(float))
    assert np.array_equal(cnt[0], ref_c) and np.array_equal(sm[0], ref_s)


# ----------------------------------------------------------------- вырожденные кадры


def _frame(rng: np.random.Generator, lift: float = 0.0, box: bool = True):
    """Кадр синтетического тоннеля в осях сообщения (вперёд −Y); lift — лидар выше, м."""
    xyz, ring, inten = tunnel_points(rng, box=(40.0, 0.2, 0.5, 0.6) if box else None)
    xyz[:, 2] -= lift
    return to_message_axes(xyz, "-y"), ring, inten


DEGENERATE = {
    "empty": (np.zeros((0, 3), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32)),
    "three": (
        np.array([[0.0, -10.0, -1.3], [0.5, -12.0, -1.3], [0.0, -30.0, 0.0]], np.float32),
        np.array([1, 2, 3]),
        np.ones(3, np.float32),
    ),
    "all_nan": (np.full((500, 3), np.nan, np.float32), np.zeros(500, np.int64), None),
    "zeros": (np.zeros((500, 3), np.float32), np.zeros(500, np.int64), np.ones(500, np.float32)),
    # немного точек только вдали: мимо зоны калибровки и модели полотна
    "far_only": (
        np.c_[np.zeros(300), -np.linspace(60.0, 90.0, 300), np.full(300, -1.3)].astype(np.float32),
        np.arange(300) % 64,
        np.full(300, 100.0, np.float32),
    ),
}


@pytest.mark.parametrize("axis", ["-y", "auto"])
def test_empty_first_frame_does_not_fix_calibration(axis: str) -> None:
    """Пустой первый кадр не калибрует «заглушкой» 1.35 м: калибровка — по первому
    пригодному кадру. Лидар на 1 м выше заглушки — объект всё равно на 40 м, а не на 5 м."""
    det = Detector("geometry", forward_axis=axis, warmup_frames=0)
    for name in ("empty", "far_only"):
        r = det.process(*DEGENERATE[name], 0.0)
        assert not r.calibrated and r.level == 0 and det.plane is None, name
    rng = np.random.default_rng(11)
    for k in range(5):
        res = det.process(*_frame(rng, lift=1.0), 0.1 * (k + 1))
    assert det.plane is not None and det.plane[3] == pytest.approx(2.35, abs=0.05)
    assert res.calibrated and res.obstacle
    assert [round(o.distance) for o in res.obstacles] == [40]


@pytest.mark.parametrize("over", [{"calib_frames": 3}, {"level": True}])
def test_multi_frame_calibration_skips_empty_frames(over: dict) -> None:
    det = Detector("geometry", forward_axis="-y", warmup_frames=0, **over)
    for name in ("empty", "three", "all_nan"):
        assert not det.process(*DEGENERATE[name], 0.0).calibrated
    rng = np.random.default_rng(12)
    for k in range(6):
        res = det.process(*_frame(rng, lift=1.0), 0.1 * (k + 1))
    # плоскость — по пригодным кадрам, а не по пустым
    assert det.plane is not None and det.plane[3] == pytest.approx(2.35, abs=0.05)
    assert res.calibrated and [round(o.distance) for o in res.obstacles] == [40]


def test_degenerate_frames_between_normal_ones() -> None:
    """Вырожденный кадр после калибровки — обычный результат «свободно, ничего не видно»,
    а не исключение; нормальные кадры после него работают как прежде."""
    rng = np.random.default_rng(13)
    det = Detector("geometry", forward_axis="auto", warmup_frames=0)
    for k in range(4):
        assert det.process(*_frame(rng), 0.1 * k).level >= 1
    t = 1.0
    for name, frame in list(DEGENERATE.items()) * 2:
        t += 0.1
        r = det.process(*frame, t)
        assert r.calibrated, name
        assert (r.level, r.obstacles) == (0, []), name
        if name != "far_only":  # 300 точек вдали — не вырожденный кадр, ось из памяти
            assert r.visible_range == 0.0, name
        assert not r.obstacle and math.isnan(r.distance), name
        assert r.corridor.shape[1] == 3 and len(r.corridor) == len(r.corridor_half_width)
        json.dumps(r.to_json(), allow_nan=False)
        r = det.process(*_frame(rng), t + 0.05)
        assert r.visible_range > 100.0 and 40 in [round(o.distance) for o in r.obstacles]


# ----------------------------------------------------------------- интенсивность


@pytest.mark.parametrize("scale", [1.0 / 255.0, 257.0])
def test_intensity_scale_is_normalized(scale: float) -> None:
    """Интенсивность 0..1 (float) или шире 0..255 даёт тот же профиль стен, что и 0..255:
    иначе после перевода в uint8 профиль пуст и скорость не меряется."""

    def run(k: float) -> list[float]:
        rng = np.random.default_rng(14)
        det = Detector("geometry", "-y", warmup_frames=0)
        shifts = []
        for f in range(4):
            xyz, ring, inten = tunnel_points(rng, box=(40.0 - 0.5 * f, 0.2, 0.5, 0.6))
            xyz[:, 0] -= 0.5 * f  # поезд едет 0.5 м за кадр
            det.process(to_message_axes(xyz, "-y"), ring, np.round(inten) * k, 0.1 * f)
            shifts.append(det.speed.shift)
        return shifts

    ref = run(1.0)
    assert max(ref) > 0.0
    assert run(scale) == pytest.approx(ref, abs=0.051)


# ----------------------------------------------------------------- габарит и рамка объекта


def _hanging(rng: np.random.Generator, top: float, bottom: float, lat: float = 0.0):
    """Кадр с «кабелем» на 40 м: точки на высотах bottom..top над полотном."""
    xyz, ring, inten = tunnel_points(rng)
    m = 60
    cable = np.c_[
        np.full(m, 40.0) + rng.uniform(0, 0.2, m),
        lat + rng.uniform(-0.3, 0.3, m),
        -1.35 + rng.uniform(bottom, top, m),
    ].astype(np.float32)
    xyz = np.r_[xyz, cable]
    elev = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    ring = np.clip(((elev + 25.0) / 0.4).astype(np.int64), 0, 127)
    return to_message_axes(xyz, "-y"), ring, np.r_[inten, np.full(m, 100.0, np.float32)]


@pytest.mark.parametrize(("from_rail", "found"), [(False, False), (True, True)])
def test_gauge_height_from_rail_head(from_rail: bool, found: bool) -> None:
    """Верх габарита: по умолчанию 3.0 м над полотном (как на стенде); с
    gauge_from_rail_head — 3.0 м над головкой рельса (≈ 0.17 м выше)."""
    assert Params().gauge_from_rail_head is False
    rng = np.random.default_rng(15)
    det = Detector("geometry", "-y", warmup_frames=0, gauge_from_rail_head=from_rail)
    for k in range(5):
        res = det.process(*_hanging(rng, top=3.12, bottom=3.04), 0.1 * k)
    assert (40 in [round(o.distance) for o in res.obstacles]) is found


def test_obstacle_vertical_extent() -> None:
    """z_extent — вертикальный размер точек объекта (у висящего кабеля мал), height —
    верх над опорой; центр рамки — середина точек по высоте."""
    rng = np.random.default_rng(16)
    det = Detector("geometry", "-y", warmup_frames=0)
    for k in range(5):
        res = det.process(*_hanging(rng, top=2.3, bottom=2.0), 0.1 * k)
    (o,) = [o for o in res.obstacles if round(o.distance) == 40]
    assert o.height == pytest.approx(2.3 - 0.17, abs=0.05)
    assert o.z_extent == pytest.approx(0.3, abs=0.03)
    assert o.position[2] == pytest.approx(-1.35 + 2.15, abs=0.03)
    js = o.to_json()
    assert js["z_extent"] == pytest.approx(o.z_extent) and js["height"] == pytest.approx(o.height)
    # позиционный конструктор с прежними полями по-прежнему работает
    assert Obstacle(40.0, 0.1, 0.5, 0.3, 0.4, 12, 0.8, True, (0.1, -40.0, -0.8)).z_extent == 0.0


def test_warmup_suppresses_alarms_until_floor_learned() -> None:
    """Режимы по умолчанию: пока профиль полотна не прогрет (15 обновлений), тревога не
    подтверждается и calibrated=False; после — та же сцена даёт подтверждённый объект."""
    rng = np.random.default_rng(0)
    det = Detector("geometry", "-y")
    assert det.params.warmup_frames == 15
    res = [det.process(*_frame(rng), 0.1 * k) for k in range(25)]
    assert not any(r.obstacle or r.calibrated for r in res[:14])
    assert res[-1].calibrated and res[-1].obstacle
