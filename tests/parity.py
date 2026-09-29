"""Сверка ядра со стендом: детекции покадрово на реальных кадрах из кэша.

    python -m tests.parity                    # geometry/strict/soft × 3 bag × оси -y и auto
    python -m tests.parity --quick            # по 60 кадров — быстрая проверка
    python -m tests.parity --bags doubleT_obstacle --modes geometry --axes=-y

Ядру подаются ДЕнормализованные оси (оси сообщения: x = y_норм, y = −x_норм),
стенду — кадры кэша как есть. Кадр совпал, если совпали
подтверждённые детекции (число и s в пределах 0.05 м) — это выход стенда. Отдельно
сверяются все кластеры кадра до голосования (у стенда — Detector._cluster() по
текущему кадру): так расхождение видно сразу, а не через 3 кадра подтверждения.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# режим ядра → конфиг стенда (geometry — прежний основной режим; у default
# с классификатором конфига-прототипа нет)
MODE_CFG = {"geometry": "Z+S c3", "strict": "Z c3", "soft": "Z c2 m0.002"}
BAG_FRAMES = {
    "roundT_pressureGate_roundT": None,
    "doubleT_obstacle": None,
    "squareT_platform_squareT_switch": 300,
}
S_TOL = 0.05
OUT = ROOT / "out" / "parity"


def denormalize(pts: np.ndarray) -> np.ndarray:
    """Кадр кэша (x-вперёд, y-влево) → оси сообщения (вперёд = −Y): x = y_н, y = −x_н."""
    return np.stack([pts["y"], -pts["x"], pts["z"]], 1).astype(np.float32)


def _bench_detector(cfg_name: str):  # type: ignore[no-untyped-def]
    import bench.detect as bd

    # В прототипе генератор RANSAC калибровки — изменяемый аргумент по умолчанию,
    # общий для всех детекторов процесса. Для сверки — свежий на каждый детектор,
    # как у первого детектора в процессе (и как в ядре).
    bd.ground_plane.__defaults__ = (np.random.default_rng(0),)
    return bd.Detector({c.name: c for c in bd.CONFIGS}[cfg_name])


def _same(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(
        abs(p - q) <= S_TOL for p, q in zip(sorted(a), sorted(b), strict=True)
    )


def run_job(job: tuple[str, str, str, int | None]) -> dict:
    bag, mode, axis, limit = job
    from bench.data import Bag
    from metro_obstacle_core import Detector
    from metro_obstacle_core.params import BENCH_COMPAT

    b = Bag(bag)
    n = len(b) if limit is None else min(limit, len(b))
    ref = _bench_detector(MODE_CFG[mode])
    core = Detector(mode=mode, forward_axis=axis, **BENCH_COMPAT)
    rows = []
    for k in range(n):
        fr = b[k]
        dets, info = ref.process(fr.pts)
        res = core.process(denormalize(fr.pts), fr.pts["ring"], fr.pts["i"], float(fr.stamp))
        cs = [o.distance for o in res.obstacles if o.confirmed]
        bs = [d.s for d in dets]
        ca = [o.distance for o in res.obstacles]
        ba = [d.s for d in ref._cluster()] if len(ref.last_axis[0]) >= 3 else []
        rows.append(
            {
                "k": k,
                "same": _same(bs, cs),
                "same_all": _same(ba, ca),
                "bench": bs,
                "core": cs,
                "core_all": [(o.distance, o.confirmed) for o in res.obstacles],
                "axis_bench": info["axis_len"],
                "axis_core": res.visible_range,
            }
        )
    return {"bag": bag, "mode": mode, "axis": axis, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", default=",".join(BAG_FRAMES))
    ap.add_argument("--modes", default="geometry,strict,soft")
    ap.add_argument("--axes", default="-y,auto")
    ap.add_argument("--quick", action="store_true", help="по 60 кадров на bag")
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    from metro_obstacle_core import warmup

    warmup()  # компиляция numba до форка воркеров — в кэш, чтобы не компилировать 9 раз
    jobs = []
    for bag in a.bags.split(","):
        lim = 60 if a.quick else BAG_FRAMES.get(bag)
        for mode in a.modes.split(","):
            for axis in a.axes.split(","):
                jobs.append((bag, mode, axis, lim))
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(min(a.workers, 10, len(jobs))) as ex:
        for r in ex.map(run_job, jobs):
            results.append(r)
            rows = r["rows"]
            ok = sum(x["same"] for x in rows)
            ok_all = sum(x["same_all"] for x in rows)
            with_det = sum(bool(x["bench"]) for x in rows)
            with_all = sum(bool(x["core_all"]) for x in rows)
            dmax = max(
                [
                    abs(p - q)
                    for x in rows
                    if x["same"]
                    for p, q in zip(sorted(x["bench"]), sorted(x["core"]), strict=True)
                ]
                or [0.0]
            )
            print(
                f"{r['bag']:36s} {r['mode']:8s} axis={r['axis']:4s} "
                f"совпало {ok}/{len(rows)} = {100 * ok / len(rows):.2f}% "
                f"(кластеры до голосования {100 * ok_all / len(rows):.2f}%); "
                f"кадров с тревогой {with_det}, с кластерами {with_all}, max|Δs| {dmax:.3g} м",
                flush=True,
            )
            for x in rows:
                if not (x["same"] and x["same_all"]):
                    print(
                        f"    k={x['k']} стенд={np.round(x['bench'], 2).tolist()} "
                        f"ядро={np.round(x['core'], 2).tolist()}"
                    )
    (OUT / "parity.json").write_text(json.dumps(results))
    tot = sum(len(r["rows"]) for r in results)
    ok = sum(x["same"] for r in results for x in r["rows"])
    print(f"ИТОГО: {ok}/{tot} = {100 * ok / tot:.2f}% кадров совпало ({time.time() - t0:.0f} с)")
    for mode in a.modes.split(","):
        rr = [x for r in results if r["mode"] == mode for x in r["rows"]]
        if rr:
            print(f"  {mode}: {100 * sum(x['same'] for x in rr) / len(rr):.2f}% из {len(rr)}")


if __name__ == "__main__":
    main()
