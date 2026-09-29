"""Время обработки кадра ядром на реальных кадрах из кэша (однопоточно).

    python -m tests.speed                          # 3 bag сверки, режим default
    python -m tests.speed --bags all --modes default,geometry,strict,soft

Первый кадр каждого прогона (калибровка RANSAC) в статистику не входит; JIT
прогревается заранее через warmup() — время прогрева печатается отдельно.
Время — полный Detector.process (вход в осях сообщения → FrameResult).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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

DEFAULT_BAGS = "roundT_pressureGate_roundT,doubleT_obstacle,squareT_platform_squareT_switch"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", default=DEFAULT_BAGS)
    ap.add_argument("--modes", default="default")
    a = ap.parse_args()
    from bench.data import BAGS, Bag
    from metro_obstacle_core import Detector, warmup
    from tests.parity import denormalize

    t = warmup()
    print(f"прогрев numba: {t:.2f} с")
    bags = BAGS if a.bags == "all" else a.bags.split(",")
    for mode in a.modes.split(","):
        for bag in bags:
            b = Bag(bag)
            frames = [b[k] for k in range(len(b))]
            inputs = [
                (denormalize(f.pts), f.pts["ring"].astype(np.int64), f.pts["i"].astype(np.float32))
                for f in frames
            ]
            det = Detector(mode=mode, forward_axis="auto")
            ms = []
            for k, (xyz, ring, inten) in enumerate(inputs):
                t0 = time.perf_counter()
                det.process(xyz, ring, inten, float(frames[k].stamp))
                ms.append((time.perf_counter() - t0) * 1e3)
            m = np.array(ms[1:])
            print(
                f"{mode:8s} {bag:36s} кадров {len(m):4d}  первый {ms[0]:6.1f} мс  "
                f"p50 {np.percentile(m, 50):5.1f}  p95 {np.percentile(m, 95):5.1f}  "
                f"max {m.max():5.1f} мс",
                flush=True,
            )


if __name__ == "__main__":
    main()
