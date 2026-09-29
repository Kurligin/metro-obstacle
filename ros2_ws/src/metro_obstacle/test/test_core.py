"""Выбор ядра и сквозной путь PointCloud2 → numpy → настоящее ядро (без ROS)."""

from types import SimpleNamespace

import numpy as np
import pytest

from metro_obstacle.cloud import cloud_to_arrays
from metro_obstacle.core import make_detector, verifier_failed, verifier_log
from metro_obstacle.fake_core import FakeDetector

core = pytest.importorskip("metro_obstacle_core")
synthetic = pytest.importorskip("metro_obstacle_core.synthetic")


def _cloud(xyz: np.ndarray, ring: np.ndarray, intensity: np.ndarray) -> SimpleNamespace:
    """PointCloud2 в раскладке bag хакатона (x,y,z,intensity f32; ring u16; step 26)."""
    dt = np.dtype(
        {
            "names": ["x", "y", "z", "intensity", "ring"],
            "formats": ["<f4", "<f4", "<f4", "<f4", "<u2"],
            "offsets": [0, 4, 8, 12, 16],
            "itemsize": 26,
        }
    )
    arr = np.zeros(len(xyz), dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["intensity"], arr["ring"] = intensity, ring
    fields = [
        SimpleNamespace(name=n, offset=o, datatype=t, count=1)
        for n, o, t in (
            ("x", 0, 7),
            ("y", 4, 7),
            ("z", 8, 7),
            ("intensity", 12, 7),
            ("ring", 16, 4),
        )
    ]
    return SimpleNamespace(
        fields=fields,
        point_step=26,
        row_step=26 * len(xyz),
        width=len(xyz),
        height=1,
        is_bigendian=False,
        data=arr.tobytes(),
    )


def test_real_core_is_default_and_takes_overrides() -> None:
    det, kind = make_detector(
        "strict", "-y", fake=False, overrides={"gauge_half_width": 1.2, "max_range": 150.0}
    )
    assert kind == "core" and isinstance(det, core.Detector)
    assert det.params.gauge_half_width == pytest.approx(1.2)
    assert det.params.max_range == pytest.approx(150.0)
    assert det.params.split is False  # strict — без раздвоенных лучей


def test_fake_core_only_on_request() -> None:
    det, kind = make_detector("default", "auto", fake=True, overrides={"max_range": 100.0})
    assert kind == "fake" and isinstance(det, FakeDetector)


@pytest.mark.parametrize(("mode", "overrides"), [("nope", {}), ("default", {"no_such_param": 1.0})])
def test_bad_core_parameters_are_value_errors(mode: str, overrides: dict) -> None:
    with pytest.raises(ValueError):
        make_detector(mode, "auto", fake=False, overrides=overrides)


def test_cloud_through_real_core_finds_box() -> None:
    """Синтетический тоннель вперёд по −Y с кубом на 40 м: подтверждение после прогрева
    модели полотна (15 кадров по умолчанию) и 3 из 5 кадров."""
    det, _ = make_detector("default", "auto", fake=False, overrides={})
    rng = np.random.default_rng(3)
    res = None
    for k in range(20):
        xyz, ring, inten = synthetic.tunnel_points(rng, box=(40.0, 0.0, 0.6, 1.0))
        msg = _cloud(synthetic.to_message_axes(xyz, "-y"), ring, inten)
        res = det.process(*cloud_to_arrays(msg), 100.0 + 0.1 * k)
    assert res is not None and res.calibrated
    assert res.obstacle and res.level == 2
    assert res.distance == pytest.approx(40.0, abs=1.5)
    near = res.obstacles[0]
    assert near.position[1] < -35  # в осях сообщения «вперёд» — это −Y
    assert res.corridor.shape[1] == 3 and len(res.corridor) == len(res.corridor_half_width)


def test_default_mode_runs_with_classifier() -> None:
    pytest.importorskip("lightgbm")
    det, kind = make_detector("default", "auto", fake=False, overrides={})
    assert det.mode == "default" and det.verifier is not None
    level, text = verifier_log(det, kind)
    assert level == "info" and text.startswith("verifier on:")


def test_geometry_mode_logs_verifier_off() -> None:
    det, kind = make_detector("geometry", "auto", fake=False, overrides={})
    assert verifier_log(det, kind) == ("info", "verifier off (mode 'geometry' is geometry only)")


def test_missing_model_falls_back_to_geometry_with_warning(tmp_path) -> None:  # type: ignore[no-untyped-def]
    det, kind = make_detector(
        "default", "auto", fake=False, overrides={"verify_model": str(tmp_path / "none.txt")}
    )
    assert det.mode == "geometry" and det.verifier is None
    level, text = verifier_log(det, kind)
    assert level == "warn" and "geometry" in text
    assert verifier_failed(det, kind)


def test_verify_model_in_mode_without_verifier_is_reported(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = str(tmp_path / "m.txt")
    det, kind = make_detector("strict", "auto", fake=False, overrides={"verify_model": model})
    assert det.verifier is None and not verifier_failed(det, kind)
    level, text = verifier_log(det, kind, model)
    assert level == "warn" and "ignored" in text


def test_fake_core_has_no_verifier_log() -> None:
    det, kind = make_detector("geometry", "auto", fake=True)
    assert kind == "fake" and verifier_log(det, kind) is None
