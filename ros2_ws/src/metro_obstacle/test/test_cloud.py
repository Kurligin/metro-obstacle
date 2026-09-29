from types import SimpleNamespace

import numpy as np
import pytest

from metro_obstacle.cloud import cloud_to_arrays, rings_from_elevation

F32, F64, U16 = 7, 8, 4


def _field(name: str, offset: int, datatype: int) -> SimpleNamespace:
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)


def _hesai_msg(pts: np.ndarray, big_endian: bool = False, height: int = 1) -> SimpleNamespace:
    """Облако в раскладке bag хакатона: x,y,z,intensity f32; ring u16; timestamp f64; step 26."""
    order = ">" if big_endian else "<"
    dt = np.dtype(
        {
            "names": ["x", "y", "z", "intensity", "ring", "timestamp"],
            "formats": [order + t for t in ("f4", "f4", "f4", "f4", "u2", "f8")],
            "offsets": [0, 4, 8, 12, 16, 18],
            "itemsize": 26,
        }
    )
    arr = np.zeros(len(pts), dt)
    for k in dt.names:
        arr[k] = pts[k]
    fields = [
        _field("x", 0, F32),
        _field("y", 4, F32),
        _field("z", 8, F32),
        _field("intensity", 12, F32),
        _field("ring", 16, U16),
        _field("timestamp", 18, F64),
    ]
    width = len(pts) // height
    return SimpleNamespace(
        fields=fields,
        point_step=26,
        row_step=26 * width,
        width=width,
        height=height,
        is_bigendian=big_endian,
        data=arr.tobytes(),
    )


def _points(n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    dt = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("intensity", "f4"),
        ("ring", "u2"),
        ("timestamp", "f8"),
    ]
    p = np.zeros(n, dt)
    p["x"], p["y"], p["z"] = rng.normal(size=(3, n)).astype("f4")
    p["intensity"] = np.arange(n, dtype="f4") * 10
    p["ring"] = np.arange(n) % 128
    p["timestamp"] = 1.7e9 + np.arange(n) * 1e-5
    return p


@pytest.mark.parametrize("big_endian", [False, True])
def test_hesai_layout_roundtrip(big_endian: bool) -> None:
    p = _points()
    xyz, ring, intensity = cloud_to_arrays(_hesai_msg(p, big_endian))
    assert xyz.dtype == np.float32 and xyz.shape == (len(p), 3)
    np.testing.assert_array_equal(xyz[:, 0], p["x"])
    np.testing.assert_array_equal(xyz[:, 2], p["z"])
    np.testing.assert_array_equal(ring, p["ring"])
    assert intensity is not None and intensity.dtype == np.float32
    np.testing.assert_array_equal(intensity, p["intensity"])


def test_nan_and_zero_points_are_passed_through() -> None:
    p = _points()
    p["x"][0] = np.nan
    p["x"][1] = p["y"][1] = p["z"][1] = 0
    xyz, _, _ = cloud_to_arrays(_hesai_msg(p))
    assert len(xyz) == len(p) and np.isnan(xyz[0, 0]) and not xyz[1].any()


def test_padded_rows_are_handled() -> None:
    """row_step больше width*point_step (выравнивание строк) — хвосты строк отбрасываем."""
    p = _points(8)
    msg = _hesai_msg(p, height=2)
    rows = np.frombuffer(msg.data, np.uint8).reshape(2, -1)
    msg.data = np.hstack([rows, np.zeros((2, 6), np.uint8)]).tobytes()
    msg.row_step += 6
    xyz, ring, _ = cloud_to_arrays(msg)
    np.testing.assert_array_equal(xyz[:, 1], p["y"])
    np.testing.assert_array_equal(ring, p["ring"])


def test_missing_ring_and_intensity() -> None:
    p = _points()
    msg = _hesai_msg(p)
    msg.fields = [f for f in msg.fields if f.name not in ("ring", "intensity")]
    _, ring, intensity = cloud_to_arrays(msg)
    assert intensity is None
    assert ring.shape == (len(p),) and ring.dtype.kind in "iu"


def test_rings_from_elevation_recovers_discrete_beams() -> None:
    rng = np.random.default_rng(1)
    beams_deg = np.array([-25.0, -10.0, -3.0, -1.0, -0.8, 0.0, 0.5, 2.0, 14.0])
    beam = rng.integers(0, len(beams_deg), 5000)
    az = rng.uniform(-np.pi, np.pi, 5000)
    r = rng.uniform(2, 150, 5000)
    el = np.radians(beams_deg[beam] + rng.normal(0, 0.005, 5000))
    xyz = np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], 1)
    xyz[:10] = np.nan
    ring = rings_from_elevation(xyz.astype(np.float32))
    ok = np.isfinite(xyz[:, 0])
    # Кольца упорядочены снизу вверх и один-в-один совпадают с лучами.
    np.testing.assert_array_equal(ring[ok], beam[ok])
    assert (ring[~ok] == 0).all()
