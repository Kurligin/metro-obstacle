"""Смена источника облаков: когда сбрасывать ядро и когда переподписываться.

Модуль без ROS — логика проверяется юнит-тестами. Сценарий, ради которого он есть:
нода запущена один раз, а на её вход по очереди проигрывают разные bag (другой
тоннель, другой топик, другое время). Калибровка, ось и история подтверждений от
прошлого источника к новому не относятся.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# Время кадров пошло назад больше чем на столько (с) — источник начал заново.
RESET_BACK_S = 1.0
# Разрыв вперёд больше этого (с) — другой источник или долгий перерыв.
RESET_FORWARD_S = 5.0
# Столько секунд без кадров — вход простаивает: можно искать другой источник.
IDLE_S = 2.0


class ResetGuard:
    """Решает по заголовку кадра, что источник сменился и ядро надо сбросить."""

    def __init__(self, back_s: float = RESET_BACK_S, forward_s: float = RESET_FORWARD_S) -> None:
        self.back_s = back_s
        self.forward_s = forward_s
        self._stamp: float | None = None
        self._frame_id: str | None = None

    def check(self, stamp: float, frame_id: str) -> str | None:
        """Причина сброса (для лога) или None. Запоминает кадр как последний."""
        reason = None
        if self._stamp is not None:
            dt = stamp - self._stamp
            if dt < -self.back_s:
                reason = f"frame time went back {-dt:.1f} s"
            elif dt > self.forward_s:
                reason = f"frame time jumped forward {dt:.1f} s"
        if reason is None and self._frame_id is not None and frame_id != self._frame_id:
            reason = f"frame_id changed {self._frame_id} -> {frame_id}"
        self._stamp, self._frame_id = stamp, frame_id
        return reason

    def clear(self) -> None:
        self._stamp = self._frame_id = None


def publishers_signature(publishers: Mapping[str, Iterable[Any]]) -> frozenset:
    """Отпечаток издателей по топикам: (топик, gid издателя, надёжность).

    publishers — {топик: список TopicEndpointInfo}. Отпечаток меняется, когда издатель
    исчез или появился новый (следующий `ros2 bag play` — новый gid, даже на том же топике).
    """
    return frozenset(
        (topic, tuple(getattr(p, "endpoint_gid", ())), str(p.qos_profile.reliability))
        for topic, pubs in publishers.items()
        for p in pubs
    )


def should_resubscribe(idle_s: float, now: frozenset, at_subscribe: frozenset) -> bool:
    """Снять подписку и искать вход заново: кадров нет дольше IDLE_S, а издатели
    с момента подписки сменились. Пока издатели те же (живой лидар молчит) —
    подписку не дёргаем, просто ждём."""
    return idle_s > IDLE_S and now != at_subscribe
