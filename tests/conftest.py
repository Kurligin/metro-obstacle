"""Общие настройки тестов ядра: однопоточность и путь к стенду (bench) для сверок."""

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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
