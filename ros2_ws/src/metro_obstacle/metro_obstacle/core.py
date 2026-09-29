"""Выбор реализации детектора: настоящее ядро (по умолчанию) или фиктивное (отладка обвязки)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

TRUE_WORDS = ("1", "true", "yes", "on")


def fake_from_env() -> bool:
    return os.environ.get("METRO_FAKE_CORE", "").strip().lower() in TRUE_WORDS


def make_detector(
    mode: str, forward_axis: str, fake: bool, overrides: Mapping[str, Any] | None = None
) -> tuple[Any, str]:
    """Детектор и его происхождение ("core" или "fake").

    overrides — поля `metro_obstacle_core.Params` (габарит, max_range); фиктивное ядро
    их не использует. Неверный режим или параметр — ValueError.

    Без явного fake_core отсутствие ядра — ошибка, а не тихая подмена: иначе можно
    случайно сдать прогон на заглушке.
    """
    if fake:
        from metro_obstacle.fake_core import FakeDetector

        return FakeDetector(mode=mode, forward_axis=forward_axis), "fake"
    try:
        from metro_obstacle_core import Detector
    except ImportError as e:
        raise RuntimeError(
            "metro_obstacle_core is not installed "
            "(pip install ./metro_obstacle_core); use fake_core:=true to test the ROS wiring only"
        ) from e
    try:
        return Detector(mode=mode, forward_axis=forward_axis, **dict(overrides or {})), "core"
    except TypeError as e:  # неизвестное имя параметра ядра
        raise ValueError(str(e)) from e


def verifier_log(detector: Any, kind: str, verify_model: str = "") -> tuple[str, str] | None:
    """Строка лога о проверяющем классификаторе: (уровень "info" | "warn", текст).

    None — у фиктивного ядра классификатора нет. warn — режим требует классификатор,
    а он не загрузился (default работает как geometry), или задан verify_model, а режим
    классификатор не использует.
    """
    if kind != "core":
        return None
    note = str(getattr(detector, "verifier_note", "off"))
    if getattr(detector, "verifier", None) is not None:
        return "info", f"verifier {note}"
    if verifier_failed(detector, kind):
        return "warn", f"verifier {note}"
    if verify_model:
        return "warn", f"verify_model is ignored: mode {detector.mode!r} does not use the verifier"
    return "info", f"verifier off (mode {detector.mode!r} is geometry only)"


def verifier_failed(detector: Any, kind: str) -> bool:
    """Режиму нужен классификатор, но он не загрузился (нет lightgbm, модели)."""
    return kind == "core" and bool(getattr(detector, "verifier_error", None))


def warmup_core(kind: str) -> float | None:
    """Прогрев JIT ядра до первого кадра (секунды) или None, если прогревать нечего.

    Без прогрева первые кадры ждали бы компиляции numba и терялись бы. Если кэш numba
    уже лежит в образе (прогрет при сборке), это доли секунды.
    """
    if kind != "core":
        return None
    from metro_obstacle_core import warmup

    return float(warmup())
