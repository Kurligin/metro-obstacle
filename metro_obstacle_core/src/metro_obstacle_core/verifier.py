"""Проверяющий классификатор кандидатов (LightGBM).

Геометрия с пониженными порогами находит больше кандидатов; классификатор по 9
признакам кластера отсекает ложные (стены, порталы, шум вдали) до подтверждения M из N.
Модель — текстовый файл LightGBM booster; в пакете лежит models/verifier.txt
(обучена на кандидатах всех пустых записей стенда и случайных формах, bench.customer
train), порог — максимум её оценки на ложных кандидатах обучающих записей.
"""

from __future__ import annotations

import threading
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

# Порог модели из пакета (bench: out/verifier/thresholds.json, ключ "full|rand").
DEFAULT_THRESHOLD = 0.7978411887547918
N_RAW = 7  # distance, lateral, height, length, width, z_extent, points


class VerifierError(RuntimeError):
    """Классификатор нельзя загрузить: нет lightgbm, нет файла или файл не читается."""


def derive(F7: np.ndarray) -> np.ndarray:
    """7 сырых признаков (distance, lateral, height, length, width, z_extent, points) → 9.

    Добавляются |lateral| и точки, приведённые к 50 м: points·(max(distance, 1)/50)² —
    плотность облака падает с квадратом дальности.
    """
    d = np.maximum(F7[:, 0], 1.0)
    return np.c_[F7, np.abs(F7[:, 1]), F7[:, 6] * (d / 50.0) ** 2]


def packaged_model_path() -> Path:
    """Путь к модели, установленной вместе с пакетом."""
    return Path(str(resources.files("metro_obstacle_core") / "models" / "verifier.txt"))


class Verifier:
    """f(признаки (n, 7)) → оценки (n,) в 0..1; предсказание в один поток."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else packaged_model_path()
        try:
            import lightgbm as lgb
        except (ImportError, OSError) as e:  # OSError — нет libgomp для lib_lightgbm.so
            raise VerifierError(f"lightgbm is not importable: {e}") from e
        if not self.path.is_file():
            raise VerifierError(f"verifier model not found: {self.path}")
        try:
            self._booster: Any = lgb.Booster(model_file=str(self.path))
        except Exception as e:  # битый или чужой файл
            raise VerifierError(f"cannot load verifier model {self.path}: {e}") from e
        n_feat = int(self._booster.num_feature())
        if n_feat != N_RAW + 2:
            raise VerifierError(
                f"verifier model {self.path} expects {n_feat} features, not {N_RAW + 2}"
            )

    def __call__(self, F7: np.ndarray) -> np.ndarray:
        X = derive(np.asarray(F7, dtype=np.float64).reshape(-1, N_RAW))
        return np.asarray(self._booster.predict(X, num_threads=1), dtype=np.float64)


_cache: dict[str, Verifier] = {}
_lock = threading.Lock()


def load_verifier(path: str | Path | None = None) -> Verifier:
    """Загруженный классификатор (один на процесс для каждого файла: нода пересоздаёт
    детектор при смене источника, модель при этом не перечитывается).

    Ошибка загрузки — VerifierError (не кэшируется).
    """
    key = str(Path(path).resolve()) if path else ""
    with _lock:
        v = _cache.get(key)
        if v is None:
            v = Verifier(path)
            _cache[key] = v
        return v
