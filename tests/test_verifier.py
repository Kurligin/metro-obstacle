"""Режим default: геометрия с пониженными порогами + проверяющий классификатор + 3 из 5.

Классификатор загружается ядром само (модель из пакета), без него default работает как geometry.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace

import numpy as np
import pytest
from metro_obstacle_core.params import MODES, make_params
from metro_obstacle_core.synthetic import LIDAR_H, to_message_axes, tunnel_points

from metro_obstacle_core import Detector
from metro_obstacle_core import verifier as V

pytest.importorskip("lightgbm")


def test_default_params_are_bench_spec() -> None:
    """default = ровно то, что гонял стенд: make(GEOM) + порог модели full|rand."""
    r4 = pytest.importorskip("bench.round4")
    e2e = pytest.importorskip("bench.verifier_e2e")
    bench = r4.make(e2e.GEOM).params
    assert replace(bench, verify_thr=V.DEFAULT_THRESHOLD) == MODES["default"]


def test_default_params_explicit() -> None:
    p = MODES["default"]
    assert p.verify_thr == pytest.approx(0.7978411887547918, abs=0)
    assert p.verify_model is None and p.verify_smin == 0.0
    assert (p.min_pts, p.min_rings, p.margin_k, p.confirm) == (2, 1, 0.001, (3, 5))
    assert p.edge_reject and p.end_guard == 0.0 and not p.split
    assert p.delta_clip == 0.05 and p.warmup_frames == 15
    for m in ("geometry", "strict", "soft"):
        assert MODES[m].verify_thr is None


def test_bench_uses_core_features() -> None:
    e2e = pytest.importorskip("bench.verifier_e2e")
    assert e2e.derive is V.derive


def test_derive() -> None:
    F7 = np.array([[100.0, -0.4, 1.0, 0.5, 0.3, 0.8, 10.0], [0.5, 0.2, 1.0, 0.5, 0.3, 0.8, 4.0]])
    X = V.derive(F7)
    assert X.shape == (2, 9)
    assert np.array_equal(X[:, :7], F7)
    assert X[:, 7].tolist() == [0.4, 0.2]
    assert X[:, 8].tolist() == [10.0 * 4.0, 4.0 * (1.0 / 50.0) ** 2]  # дальность не < 1 м


def test_packaged_model_is_installed() -> None:
    assert V.packaged_model_path().is_file()


def test_default_loads_classifier() -> None:
    det = Detector("default", "-y")
    assert det.mode == "default" and det.verifier is not None
    assert det.verifier_note.startswith("on:")
    scores = det.verifier(np.array([[40.0, 0.1, 0.6, 0.5, 0.5, 0.6, 300.0]]))
    assert scores.shape == (1,) and 0.0 <= float(scores[0]) <= 1.0
    # модель одна на процесс: пересоздание детектора (сброс в ноде) её не перечитывает
    assert Detector("default", "-y").verifier is det.verifier


def test_geometry_modes_do_not_load_classifier() -> None:
    for m in ("geometry", "strict", "soft"):
        det = Detector(m, "-y")
        assert det.verifier is None and det.verifier_note == "off"


def _scene(det: Detector, blob_s: float, frames: int = 6):  # type: ignore[no-untyped-def]
    """Тоннель с кубом 0.5 м на 40 м и крошечным «бликом» (2 точки одного кольца) на
    blob_s: геометрия с пониженными порогами подтверждает оба, классификатор — только куб."""
    rng = np.random.default_rng(7)
    res = None
    blob = np.array([[blob_s, 0.1, 0.4 - LIDAR_H], [blob_s + 0.05, 0.1, 0.4 - LIDAR_H]])
    for k in range(frames):
        xyz, ring, inten = tunnel_points(rng, length=180.0, box=(40.0, 0.2, 0.5, 0.6))
        xyz = np.r_[xyz, blob].astype(np.float32)
        ring = np.r_[ring, [20, 20]]
        inten = np.r_[inten, [50.0, 50.0]].astype(np.float32)
        res = det.process(to_message_axes(xyz, "-y"), ring, inten, 0.1 * k)
    assert res is not None
    return res


@pytest.mark.parametrize("blob_s", [60.0, 90.0])
def test_classifier_filters_candidates(blob_s: float) -> None:
    geom = _scene(Detector("default", "-y", warmup_frames=0, verify_thr=None), blob_s)
    assert sorted(round(o.distance) for o in geom.obstacles if o.confirmed) == sorted(
        [40, round(blob_s)]
    )
    res = _scene(Detector("default", "-y", warmup_frames=0), blob_s)
    assert res.obstacle and res.distance == pytest.approx(40.0, abs=0.05)
    assert [round(o.distance) for o in res.obstacles] == [40]


def test_threshold_above_one_rejects_everything() -> None:
    res = _scene(Detector("default", "-y", warmup_frames=0, verify_thr=1.01), 60.0)
    assert not res.obstacles and res.level == 0


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_fallback_without_model_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # свой обработчик прямо на логгере модуля, а не caplog: в образе pytest-плагин ROS
    # (launch_testing) подменяет класс логгеров, и записи не всплывают к корню
    rec = _Records()
    logger = logging.getLogger("metro_obstacle_core.detector")
    logger.addHandler(rec)
    try:
        det = Detector("default", "-y", warmup_frames=0, verify_model=str(tmp_path / "nope.txt"))
    finally:
        logger.removeHandler(rec)
    assert det.verifier is None and det.mode == "geometry"
    assert det.params == make_params("geometry", warmup_frames=0)  # прочие переопределения — те же
    assert "not found" in det.verifier_note
    assert len(rec.messages) == 1 and "geometry" in rec.messages[0]
    res = _scene(det, 60.0)  # работает как geometry: куб подтверждён
    assert res.obstacle and res.mode == "geometry"


def test_fallback_without_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lightgbm", None)  # import lightgbm → ImportError
    monkeypatch.setattr(V, "_cache", {})
    det = Detector("default", "-y")
    assert det.verifier is None and det.mode == "geometry"
    assert det.params == MODES["geometry"]
    assert "lightgbm" in det.verifier_note


def test_fallback_on_broken_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.txt"
    bad.write_text("not a model\n")
    det = Detector("default", "-y", verify_model=str(bad))
    assert det.verifier is None and det.mode == "geometry"


class _BrokenLib:
    """Поисковик модулей: import lightgbm падает OSError, как без libgomp."""

    def find_spec(self, name: str, path=None, target=None):  # type: ignore[no-untyped-def]
        if name == "lightgbm" or name.startswith("lightgbm."):
            raise OSError("libgomp.so.1: cannot open shared object file")
        return None


def test_fallback_when_lightgbm_library_fails_to_load(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [m for m in sys.modules if m == "lightgbm" or m.startswith("lightgbm.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_BrokenLib(), *sys.meta_path])
    monkeypatch.setattr(V, "_cache", {})
    det = Detector("default", "-y")
    assert det.verifier is None and det.mode == "geometry"
    assert det.verifier_error is not None and "libgomp" in det.verifier_error


def test_other_modes_keep_their_params_without_classifier(tmp_path) -> None:  # type: ignore[no-untyped-def]
    missing = str(tmp_path / "nope.txt")
    det = Detector("strict", "-y", verify_thr=0.5, verify_model=missing, warmup_frames=0)
    assert det.verifier is None and det.mode == "strict" and det.verifier_error
    assert det.params == make_params("strict", warmup_frames=0)
    ok = Detector("strict", "-y", verify_thr=0.5)
    assert ok.verifier is not None and ok.verifier_error is None


def test_margin_scalar_is_python_float() -> None:
    from metro_obstacle_core.detector import _margin

    p = MODES["default"]
    assert type(_margin(p, 50.0)) is float
    assert type(_margin(p, np.float32(50.0))) is float
    arr = _margin(p, np.array([10.0, 50.0], np.float32))
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)
