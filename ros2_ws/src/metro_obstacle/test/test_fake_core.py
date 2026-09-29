import json
import math

import numpy as np

from metro_obstacle.fake_core import FakeDetector, forward_basis


def _tunnel(forward: str, obstacle_at: float | None, seed: int = 0) -> np.ndarray:
    """Грубый тоннель: пол на -1.5 м, стены на ±2.2 м, лидар смотрит в `forward`."""
    rng = np.random.default_rng(seed)
    n = 40000
    s = rng.uniform(1, 120, n)
    lat = rng.choice([-2.2, 2.2], n) + rng.normal(0, 0.02, n)
    z = rng.uniform(-1.5, 3.0, n)
    floor = rng.random(n) < 0.5
    lat[floor] = rng.uniform(-2.2, 2.2, floor.sum())
    z[floor] = -1.5
    pts = [np.stack([s, lat, z], 1)]
    if obstacle_at is not None:
        m = 300
        pts.append(
            np.stack(
                [
                    obstacle_at + rng.uniform(0, 0.3, m),
                    rng.uniform(-0.3, 0.3, m),
                    rng.uniform(-1.4, 0.2, m),
                ],
                1,
            )
        )
    local = np.concatenate(pts)
    f, left = forward_basis(forward)
    up = np.array([0.0, 0.0, 1.0])
    xyz = local[:, :1] * f + local[:, 1:2] * left + local[:, 2:3] * up
    xyz = np.vstack([xyz, np.full((5, 3), np.nan), np.zeros((5, 3))])
    return xyz.astype(np.float32)


def test_detects_obstacle_on_minus_y_with_auto_axis() -> None:
    det = FakeDetector(mode="default", forward_axis="auto")
    xyz = _tunnel("-y", obstacle_at=25.0)
    ring = np.zeros(len(xyz), np.int64)
    res = None
    for k in range(4):
        res = det.process(xyz, ring, None, 100.0 + 0.1 * k)
    assert res is not None
    assert res.obstacle and res.level == 2
    assert abs(res.distance - 25.0) < 1.0
    ob = res.obstacles[0]
    assert ob.confirmed and ob.points > 0
    # Центр объекта — в осях сообщения: вперёд по −Y.
    assert ob.position[1] < -20 and abs(ob.position[0]) < 1
    assert res.corridor.shape[1] == 3 and len(res.corridor) == len(res.corridor_half_width)
    assert res.corridor[-1, 1] < -10  # ось коридора тоже уходит в −Y
    json.dumps(res.to_json())


def test_clear_track_on_x() -> None:
    det = FakeDetector(mode="strict", forward_axis="x")
    xyz = _tunnel("x", obstacle_at=None)
    res = det.process(xyz, np.zeros(len(xyz), np.int64), np.ones(len(xyz), np.float32), 1.0)
    assert not res.obstacle and res.level == 0 and math.isnan(res.distance)
    assert res.mode == "strict" and res.visible_range > 0
    d = res.to_json()
    assert d["distance"] is None  # NaN в JSON не пишем
    json.dumps(d, allow_nan=False)
