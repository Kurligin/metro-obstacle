"""Веб-прототип: загрузка записи ROS 2 → ядро детектора → просмотр результата в браузере.

Инициализация пакета выполняется до импорта numpy/numba в модулях web: однопоточность
(нагрузка предсказуема, как у стенда) и путь к чистым модулям ROS-пакета
(cloud.py, viz.py, source.py, results_log.py) — те же, что в ноде.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
_NODE_PKG = str(ROOT / "ros2_ws" / "src" / "metro_obstacle")
for _p in (str(ROOT), _NODE_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)
