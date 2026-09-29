"""Старт записи с разных файлов new_data: тревоги на общем участке и итоговая плоскость.

Плоскость полотна калибруется по первому кадру; если запись начинается в неудачном месте
(стрелка, платформа), без уточнения на ходу плоскость кривая до конца. Здесь каждый
вариант детектора запускается с нескольких стартовых файлов и идёт до конца участка;
тревоги считаются только на общем участке [from, to), где у всех стартов разогрев позади.

    python -m bench.starts <каталог new_data> [старты] [варианты] [общий участок]
    python -m bench.starts <каталог new_data> 111,130,145,148 \
        core_default,core_default_plane0 148-158   # общий участок — файлы 148–157 включительно
"""

from __future__ import annotations

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ros2_ws/src/metro_obstacle"))
from metro_obstacle.cloud import cloud_to_arrays

from bench.customer import make_variant
from bench.newdata import messages

warnings.filterwarnings("ignore")


def run(job):
    root, name, start, lo, hi = job
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    det = make_variant(name)
    upd: list[float] = []
    f0 = det._update_plane

    def timed(*a):
        p0 = det.plane
        t0 = time.perf_counter()
        f0(*a)
        if det.plane is not p0:  # была оценка: плоскость пересчитана
            upd.append((time.perf_counter() - t0) * 1e3)

    det._update_plane = timed
    alarms: dict[int, int] = {}
    plane_at: dict[int, list[float] | None] = {}
    for i in range(start, hi):
        for t, _, raw in messages(Path(root) / f"new_data_{i}.db3"):
            xyz, ring, it = cloud_to_arrays(ts.deserialize_cdr(raw, "sensor_msgs/msg/PointCloud2"))
            r = det.process(xyz, ring, it, t * 1e-9)
            if i >= lo:
                alarms[i] = alarms.get(i, 0) + int(r.obstacle)
        plane_at[i] = None if det.plane is None else det.plane.round(4).tolist()
    return name, start, alarms, plane_at.get(lo), plane_at.get(hi - 1), upd


def main() -> None:
    root = sys.argv[1]
    starts = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "111,130,145,148").split(",")]
    names = (sys.argv[3] if len(sys.argv) > 3 else "core_default,core_default_plane0").split(",")
    lo, hi = (int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "148-158").split("-"))
    jobs = [(root, n, s, lo, hi) for n in names for s in starts]
    with ProcessPoolExecutor(len(jobs)) as ex:
        for name, start, al, p_lo, p_hi, upd in ex.map(run, jobs):
            est = np.array(upd)
            total = sum(al.values())
            print(
                f"{name:22s} старт {start:4d}: тревог в {lo}–{hi - 1} = {total:4d} {al}\n"
                f"{'':22s} плоскость в начале общего участка {p_lo}, в конце {p_hi}; "
                f"оценок плоскости {len(est)}, "
                f"мс на оценку p50 {np.percentile(est, 50) if len(est) else 0:.2f} "
                f"p95 {np.percentile(est, 95) if len(est) else 0:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
