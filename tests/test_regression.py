"""Регрессия на реальных кадрах из кэша стенда (пропускается, если кэша нет).

Полная сверка со стендом — tests/parity.py (минута на 9 процессах); здесь — короткие
куски, чтобы ловить расхождения при каждом прогоне pytest.
"""

from __future__ import annotations

import numpy as np
import pytest

from metro_obstacle_core import Detector

data = pytest.importorskip("bench.data")
if not (data.CACHE_ROOT / "doubleT_obstacle.pts.npy").exists():
    pytest.skip("нет кэша кадров стенда (LCT_CACHE)", allow_module_level=True)

from tests.parity import MODE_CFG, _bench_detector, denormalize  # noqa: E402

pytestmark = pytest.mark.regression


def _frames(bag: str, n: int) -> list:
    b = data.Bag(bag)
    return [b[k] for k in range(min(n, len(b)))]


@pytest.mark.parametrize(
    ("bag", "mode", "n"),
    [
        ("doubleT_obstacle", "geometry", 40),
        ("doubleT_obstacle", "strict", 40),
        ("squareT_platform_squareT_switch", "soft", 40),
    ],
)
def test_same_detections_as_bench(bag: str, mode: str, n: int) -> None:
    from metro_obstacle_core.params import BENCH_COMPAT

    ref = _bench_detector(MODE_CFG[mode])
    core = Detector(mode=mode, forward_axis="auto", **BENCH_COMPAT)
    for k, fr in enumerate(_frames(bag, n)):
        dets, info = ref.process(fr.pts)
        res = core.process(denormalize(fr.pts), fr.pts["ring"], fr.pts["i"], float(fr.stamp))
        assert sorted(d.s for d in dets) == sorted(
            o.distance for o in res.obstacles if o.confirmed
        ), f"кадр {k}"
        assert res.visible_range == info["axis_len"], f"кадр {k}"


@pytest.mark.parametrize("mode", ["default", "geometry"])
def test_real_person_alarm_and_release(mode: str) -> None:
    """Человек в doubleT_obstacle на ~56 м: тревога, пока он в габарите (кадры ~13–60),
    и снятие, когда уходит (дальше на этой дальности подтверждённых нет)."""
    if mode == "default":
        pytest.importorskip("lightgbm")
    core = Detector(mode=mode, forward_axis="auto")
    assert core.mode == mode
    hit = []
    for fr in _frames("doubleT_obstacle", 201):
        res = core.process(denormalize(fr.pts), fr.pts["ring"], fr.pts["i"], float(fr.stamp))
        hit.append(any(o.confirmed and 50 < o.distance < 62 for o in res.obstacles))
    hit_arr = np.array(hit)
    assert hit_arr[15:60].mean() >= 0.95
    assert not hit_arr[70:].any()
