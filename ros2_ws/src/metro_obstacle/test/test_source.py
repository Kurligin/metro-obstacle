from types import SimpleNamespace

from metro_obstacle.source import (
    IDLE_S,
    ResetGuard,
    publishers_signature,
    should_resubscribe,
)


def test_reset_on_time_jumps_and_frame_change() -> None:
    g = ResetGuard()
    assert g.check(100.0, "hesai") is None  # первый кадр
    assert g.check(100.1, "hesai") is None
    assert g.check(99.5, "hesai") is None  # перестановка кадров в пределах 1 с
    assert "back" in (g.check(95.0, "hesai") or "")  # bag начался заново (--loop)
    assert g.check(99.0, "hesai") is None  # 4 с вперёд — пропуск кадров, не сброс
    assert "forward" in (g.check(105.5, "hesai") or "")  # другой bag, позже по времени
    assert "frame_id" in (g.check(105.6, "lidar") or "")
    g.clear()
    assert g.check(1.0, "other") is None  # после новой подписки — с чистого листа


def _pub(gid: int, reliability: str = "RELIABLE") -> SimpleNamespace:
    return SimpleNamespace(
        endpoint_gid=[gid, 0], qos_profile=SimpleNamespace(reliability=reliability)
    )


def test_resubscribe_only_when_idle_and_publishers_changed() -> None:
    first = publishers_signature({"/sensing/lidar/hesai128/pointcloud": [_pub(1)]})
    gone = publishers_signature({})
    second = publishers_signature({"/lidar_points": [_pub(2)]})
    same_topic_new_bag = publishers_signature({"/sensing/lidar/hesai128/pointcloud": [_pub(3)]})
    # данные идут — ничего не трогаем, даже если в графе что-то поменялось
    assert not should_resubscribe(0.5, second, first)
    # простой, но издатель тот же (живой лидар молчит) — ждём на той же подписке
    assert not should_resubscribe(IDLE_S + 1, first, first)
    # первый bag закончился: издателя нет — снимаем подписку и ищем снова
    assert should_resubscribe(IDLE_S + 0.1, gone, first)
    assert should_resubscribe(IDLE_S + 0.1, second, first)
    # тот же топик, но другой издатель (следующий bag) — переподписка заново берёт его QoS
    assert should_resubscribe(IDLE_S + 0.1, same_topic_new_bag, first)
