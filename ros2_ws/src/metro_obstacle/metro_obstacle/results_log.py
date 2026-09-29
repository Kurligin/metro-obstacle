"""Лог результатов в JSON Lines: одна строка на обработанный кадр.

results_path бывает трёх видов:
    каталог (`/data/out/` или существующий каталог) — на каждый источник свой файл
        `results_<bag или топик>_<YYYYmmdd-HHMMSS>.jsonl`;
    шаблон с `{name}` и/или `{time}` — то же, имя файла задаёт шаблон;
    обычный файл — дописывается (повторный запуск не стирает прошлый прогон).
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

TIME_FORMAT = "%Y%m%d-%H%M%S"
FILE_PATTERN = "results_{name}_{time}.jsonl"


def jsonable(v: Any) -> Any:
    """numpy → встроенные типы; NaN/inf → null (строгий JSON их не допускает)."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def safe_name(name: str) -> str:
    """Имя bag или топика → кусок имени файла: `/lidar_points` → `lidar_points`."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "input"


def per_source(spec: str) -> bool:
    """True — spec задаёт каталог или шаблон: на каждый источник свой файл."""
    return "{" in spec or spec.endswith(("/", os.sep)) or Path(spec).is_dir()


def results_file(spec: str, name: str, now: datetime) -> Path:
    """Путь файла для источника name (bag или топик), открытого в момент now."""
    fields = {"name": safe_name(name), "time": now.strftime(TIME_FORMAT)}
    if "{" in spec:
        return Path(spec.format(**fields))
    if per_source(spec):
        return Path(spec) / FILE_PATTERN.format(**fields)
    return Path(spec)


class ResultsWriter:
    def __init__(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Дописываем, а не перезаписываем: повторный запуск не теряет прошлые данные.
        # Построчная буферизация: файл пригоден для разбора, даже если прогон оборван.
        self._f = p.open("a", buffering=1, encoding="utf-8")
        self.path = str(p)

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")

    def close(self) -> None:
        self._f.close()


class ResultsLog:
    """Лог прогона: файл открывается на первом кадре источника (к этому моменту
    известно имя топика), при смене источника в режиме каталога/шаблона — новый файл."""

    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.per_source = per_source(spec)
        self._writer: ResultsWriter | None = None

    @property
    def path(self) -> str | None:
        return self._writer.path if self._writer is not None else None

    def write(self, record: dict, name: str, now: datetime | None = None) -> bool:
        """Записать кадр; True — открыт новый файл (для лога ноды)."""
        opened = False
        if self._writer is None:
            self._writer = ResultsWriter(str(results_file(self.spec, name, now or datetime.now())))
            opened = True
        self._writer.write(record)
        return opened

    def new_source(self) -> None:
        """Источник сменился: в режиме каталога/шаблона следующий кадр откроет новый файл."""
        if self.per_source:
            self.close()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
