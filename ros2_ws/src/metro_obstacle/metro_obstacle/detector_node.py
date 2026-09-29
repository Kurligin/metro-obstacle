"""ROS 2-нода детектора препятствий в габарите.

Подписывается на PointCloud2 (или сама находит такой топик), на каждый кадр вызывает
ядро и публикует статус, объекты, vision_msgs-детекции, маркеры RViz и диагностику.

Нода рассчитана на долгую жизнь: источники на входе могут сменяться (bag за bag).
Вход молчит дольше 2 с, а издатели сменились — подписка снимается и топик ищется
заново; скачок времени кадров или смена frame_id — ядро сбрасывается (см. source.py).
"""

from __future__ import annotations

import math
import signal
import time
from collections import deque
from typing import Any

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray

from metro_obstacle.cloud import cloud_to_arrays
from metro_obstacle.core import (
    fake_from_env,
    make_detector,
    verifier_failed,
    verifier_log,
    warmup_core,
)
from metro_obstacle.geometry import corridor_length
from metro_obstacle.messages import detections_msg, markers_msg, obstacles_msg, status_msg
from metro_obstacle.results_log import ResultsLog
from metro_obstacle.source import IDLE_S, ResetGuard, publishers_signature, should_resubscribe
from metro_obstacle_msgs.msg import ObstacleArray, ObstacleStatus

POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"
OUT_NS = "/obstacle_detection"
FIXED_FRAME = "lidar_link"
# Строка лога о готовности: по ней launch запускает проигрывание bag.
READY_MARKER = "detector ready"
DISABLED_PATHS = ("", "none", "off", "false")
# Параметры ноды, которые передаются в ядро (имена полей metro_obstacle_core.Params).
CORE_PARAMS: dict[str, float] = {
    "gauge_half_width": 1.05,  # полуширина габарита, м
    "gauge_height": 3.0,  # высота габарита над полотном, м
    "max_range": 200.0,  # дальше ось пути не строится, м
}

RELIABILITY = {
    "reliable": ReliabilityPolicy.RELIABLE,
    "best_effort": ReliabilityPolicy.BEST_EFFORT,
}


def sensor_qos(reliability: ReliabilityPolicy, depth: int = 5) -> QoSProfile:
    """Очередь KEEP_LAST небольшой глубины. Ядро обрабатывает кадр в разы быстрее периода
    лидара (20–30 мс при 100 мс), поэтому очередь в норме пуста и задержку не добавляет;
    глубина нужна, чтобы не терять кадры, когда источник (например, ros2 bag play под
    нехваткой памяти) отдаёт их пачками: при depth=1 из пачки оставался только последний.
    depth=1 — всегда только самый свежий кадр."""
    return QoSProfile(
        reliability=reliability,
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        durability=DurabilityPolicy.VOLATILE,
    )


def _stamp_sec(header: Header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class DropCounter:
    """Счётчик пропущенных кадров по разрывам во времени кадров.

    Считает и то, что вытеснено в очереди ноды, и потери в транспорте. Период
    сенсора — минимальный интервал среди последних: он не завышается пропусками.
    """

    def __init__(self, window: int = 50) -> None:
        self._intervals: deque[float] = deque(maxlen=window)
        self._prev: float | None = None
        self.dropped = 0

    def restart(self) -> None:
        """Новый источник: разрыв между источниками — не пропущенные кадры."""
        self._intervals.clear()
        self._prev = None

    def update(self, stamp: float) -> None:
        if self._prev is not None:
            dt = stamp - self._prev
            if dt <= 0:
                # Время пошло назад — bag начался заново (--loop); историю сбрасываем.
                self._intervals.clear()
            else:
                self._intervals.append(dt)
                period = min(self._intervals)
                self.dropped += max(0, round(dt / period) - 1)
        self._prev = stamp


def _interrupted(e: Exception) -> bool:
    """Ctrl-C внутри функции numba: CPython отдаёт его как SystemError («returned a result
    with an exception set») с KeyboardInterrupt в причине. Это остановка, а не сбой кадра."""
    return isinstance(e, SystemError) and isinstance(e.__cause__, KeyboardInterrupt)


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("metro_obstacle_detector")
        self.declare_parameter("input_topic", "")
        self.declare_parameter("mode", "default")
        # Файл модели проверяющего классификатора; пусто — модель из пакета ядра.
        self.declare_parameter("verify_model", "")
        self.declare_parameter("forward_axis", "auto")
        self.declare_parameter("results_path", "")
        self.declare_parameter("results_tag", "")
        self.declare_parameter("fake_core", False)
        # Геометрия габарита и дальность — параметры ядра (значения по умолчанию = ядра).
        for name, default in CORE_PARAMS.items():
            self.declare_parameter(name, default)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("input_reliability", "auto")
        self.declare_parameter("input_depth", 5)

        # Неверные параметры — ошибка сразу, до долгого прогрева ядра.
        self._wanted_topic = self._str("input_topic")
        self._reliability = self._str("input_reliability").lower()
        if self._reliability not in ("auto", *RELIABILITY):
            raise ValueError(
                f"input_reliability must be auto|reliable|best_effort, got {self._reliability!r}"
            )
        mode = self._str("mode")
        forward_axis = self._str("forward_axis")
        fake = bool(self.get_parameter("fake_core").value) or fake_from_env()
        core_params: dict[str, Any] = {n: float(self.get_parameter(n).value) for n in CORE_PARAMS}
        verify_model = self._str("verify_model")
        if verify_model:
            core_params["verify_model"] = verify_model
        self._core_args = (mode, forward_axis, fake, core_params)
        self._detector, self._core_kind = make_detector(*self._core_args)
        # Фактический режим: без классификатора default откатывается на geometry.
        self._mode = str(getattr(self._detector, "mode", mode))
        self._gauge_height = core_params["gauge_height"]
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        if self._core_kind == "fake":
            self.get_logger().warn("FAKE core in use: results are NOT real detections")
        vlog = verifier_log(self._detector, self._core_kind, verify_model)
        if vlog is not None:
            level, text = vlog
            (self.get_logger().warn if level == "warn" else self.get_logger().info)(text)
        self._warmup_s = warmup_core(self._core_kind)
        if self._warmup_s is not None:
            self.get_logger().info(f"core warm-up done in {self._warmup_s:.1f} s")

        self._pub_status = self.create_publisher(ObstacleStatus, f"{OUT_NS}/status", 10)
        self._pub_obstacles = self.create_publisher(ObstacleArray, f"{OUT_NS}/obstacles", 10)
        self._pub_detections = self.create_publisher(Detection3DArray, f"{OUT_NS}/detections", 10)
        self._pub_markers = self.create_publisher(MarkerArray, f"{OUT_NS}/markers", 10)
        self._pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self._tf = StaticTransformBroadcaster(self)
        self._tf_parent = ""

        path = self._str("results_path")
        # Имя источника в имени файла: bag (передаёт launch) или, если пусто, топик.
        self._results_tag = self._str("results_tag")
        self._results: ResultsLog | None = None
        if path.lower() not in DISABLED_PATHS:
            self._results = ResultsLog(path)
            how = "a file per input source" if self._results.per_source else "appending"
            self.get_logger().info(f"results to {path} ({how})")

        self._topic = ""
        self._frame_id = ""
        self._sub: Any = None
        self._sub_signature: frozenset = frozenset()
        self._last_input = time.monotonic()  # последний кадр или момент подписки
        self._guard = ResetGuard()
        self._source_frames = 0  # кадров от текущего источника (с подписки)
        self._frames = 0
        self._errors = 0
        self._drops = DropCounter()
        self._last_res: Any = None
        self._t_proc: deque[float] = deque(maxlen=200)  # полное время кадра в ноде, мс
        self._t_core: deque[float] = deque(maxlen=200)  # время ядра, мс
        self._t_wall: deque[float] = deque(maxlen=50)  # моменты выдачи результата
        self._t_stamp: deque[float] = deque(maxlen=50)  # время кадров (для частоты входа)
        self._stamp_age_ms = math.nan
        self._discover_started = time.monotonic()
        # Таймер не останавливается: он же следит за простоем входа.
        self.create_timer(0.2, self._discover)
        self.create_timer(1.0, self._publish_diagnostics)
        what = self._wanted_topic or "any PointCloud2 topic"
        self.get_logger().info(
            f"{READY_MARKER}: mode={self._mode} forward_axis={forward_axis} "
            f"core={self._core_kind}; "
            f"waiting for a publisher on {what}; outputs under {OUT_NS}/"
        )

    def _str(self, name: str) -> str:
        return str(self.get_parameter(name).value or "").strip()

    # --- вход ---------------------------------------------------------------

    def _candidates(self) -> list[str]:
        if self._wanted_topic:
            return [self._wanted_topic]
        return sorted(
            name
            for name, types in self.get_topic_names_and_types()
            if POINTCLOUD2 in types and not name.startswith(OUT_NS)
        )

    def _live_publishers(self) -> dict[str, list]:
        """Издатели по топикам-кандидатам (только топики, где издатели есть)."""
        pubs = {t: self.get_publishers_info_by_topic(t) for t in self._candidates()}
        return {t: p for t, p in pubs.items() if p}

    def _discover(self) -> None:
        """Ждём издателя на нужном топике (или на любом PointCloud2) и подписываемся.

        Подписка создаётся, только когда издатель уже виден: нужно знать его QoS.
        Если вход молчит дольше 2 с и издатели сменились (bag закончился, запущен
        следующий), подписка снимается и топик ищется заново. С явным input_topic
        ищется тот же топик: переподписка лишь заново подбирает QoS нового издателя.
        """
        if self._sub is not None:
            idle = time.monotonic() - self._last_input
            if idle <= IDLE_S:
                return  # кадры идут — граф не опрашиваем
            if not should_resubscribe(
                idle, publishers_signature(self._live_publishers()), self._sub_signature
            ):
                return
            self.get_logger().info(
                f"no frames on {self._topic} for {idle:.1f} s and its publishers changed: "
                "unsubscribed, looking for input again"
            )
            self.destroy_subscription(self._sub)
            self._sub, self._topic = None, ""
            self._discover_started = time.monotonic()
        pubs = self._live_publishers()
        live = sorted(pubs)
        if not live:
            waited = time.monotonic() - self._discover_started
            self.get_logger().info(
                f"still waiting for a publisher ({waited:.0f} s)", throttle_duration_sec=5.0
            )
            return
        if len(live) > 1:
            self.get_logger().warn(
                f"several PointCloud2 topics {live}; taking {live[0]} (set input_topic to choose)"
            )
        self._subscribe(live[0], publishers_signature(pubs))

    def _pick_reliability(self, topic: str) -> ReliabilityPolicy:
        """RELIABLE, если все издатели надёжные (так публикует ros2 bag play), иначе
        BEST_EFFORT (типично для драйверов лидаров): надёжный подписчик не получит
        ничего от best-effort издателя.

        Почему не всегда best effort: кадр ~9 МБ передаётся сотнями фрагментов, и при
        best effort потеря одного фрагмента — потеря всего кадра (на проверке —
        ~12% кадров). Надёжная доставка дошлёт фрагменты, а глубина 1 всё равно
        оставляет только свежий кадр.
        """
        if self._reliability in RELIABILITY:
            return RELIABILITY[self._reliability]
        pubs = self.get_publishers_info_by_topic(topic)
        if pubs and all(p.qos_profile.reliability == ReliabilityPolicy.RELIABLE for p in pubs):
            return ReliabilityPolicy.RELIABLE
        return ReliabilityPolicy.BEST_EFFORT

    def _subscribe(self, topic: str, signature: frozenset) -> None:
        if self._source_frames:
            # Новый источник: калибровка и история прошлого к нему не относятся.
            self._reset("new input source")
            if self._results is not None:
                self._results.new_source()
            # lidar_link переносится на фрейм нового источника (/tf_static — latched,
            # новое сообщение заменяет прежнее), иначе RViz не свяжет его выходы.
            self._tf_parent = ""
        self._guard.clear()
        self._topic = topic
        self._sub_signature = signature
        self._last_input = time.monotonic()
        reliability = self._pick_reliability(topic)
        depth = max(1, int(self.get_parameter("input_depth").value))
        self._sub = self.create_subscription(
            PointCloud2, topic, self._on_cloud, sensor_qos(reliability, depth)
        )
        self.get_logger().info(f"subscribed to {topic} ({reliability.name.lower()}, depth {depth})")

    def _reset(self, reason: str) -> None:
        """Сброс ядра: калибровка, ось, модель полотна и история — с чистого листа."""
        self._detector, _ = make_detector(*self._core_args)
        self._mode = str(getattr(self._detector, "mode", self._core_args[0]))
        self._drops.restart()
        self._t_stamp.clear()
        self._source_frames = 0
        self.get_logger().info(f"detector reset: {reason}")

    def _ensure_tf(self, frame_id: str) -> None:
        """Тождественный TF <frame облака> → lidar_link.

        Так фиксированный фрейм RViz всегда lidar_link, как бы ни назывался фрейм
        лидара в данных. lidar_link — дочерний: если у фрейма облака уже есть
        родитель в чужом дереве TF, второго родителя мы ему не создаём. Публикуется
        на первом кадре источника (у lidar_link может быть только один родитель);
        при смене источника — заново, для фрейма нового источника.
        """
        if not self._publish_tf or frame_id == FIXED_FRAME or self._tf_parent == frame_id:
            return
        if self._tf_parent:
            self.get_logger().warn(
                f"cloud frame changed {self._tf_parent} -> {frame_id}; "
                f"{FIXED_FRAME} stays attached to {self._tf_parent}",
                once=True,
            )
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = FIXED_FRAME
        t.transform.rotation.w = 1.0
        self._tf.sendTransform(t)
        self._tf_parent = frame_id
        self.get_logger().info(f"static TF {frame_id} -> {FIXED_FRAME} (identity)")

    def _on_cloud(self, msg: PointCloud2) -> None:
        t0 = time.perf_counter()
        header = msg.header
        if not header.frame_id:
            header.frame_id = FIXED_FRAME
        if header.frame_id != self._frame_id:
            self._frame_id = header.frame_id
            self.get_logger().info(
                f"first cloud: frame_id={header.frame_id} "
                f"fields={[f.name for f in msg.fields]} points={msg.width * msg.height}"
            )
        self._ensure_tf(header.frame_id)
        stamp = _stamp_sec(header)
        self._last_input = time.monotonic()
        # Время назад (bag заново, --loop), большой разрыв вперёд (другой bag) или
        # другой frame_id — калибровка и история относятся к другому месту и лидару.
        reason = self._guard.check(stamp, header.frame_id)
        if reason is not None:
            self._reset(reason)
        self._drops.update(stamp)
        self._stamp_age_ms = (self.get_clock().now().nanoseconds * 1e-9 - stamp) * 1e3

        try:
            xyz, ring, intensity = cloud_to_arrays(msg)
            res = self._detector.process(xyz, ring, intensity, stamp)
        except Exception as e:  # noqa: BLE001 — кадр с ошибкой пропускаем, нода работает дальше
            if _interrupted(e):
                raise KeyboardInterrupt from None
            self._errors += 1
            self.get_logger().error(f"frame skipped: {type(e).__name__}: {e}")
            return
        total_ms = (time.perf_counter() - t0) * 1e3

        self._pub_status.publish(status_msg(res, header, total_ms))
        self._pub_obstacles.publish(obstacles_msg(res, header))
        try:
            self._pub_detections.publish(detections_msg(res, header))
            self._pub_markers.publish(markers_msg(res, header, self._gauge_height))
        except Exception as e:  # noqa: BLE001 — сбой отрисовки не должен ронять детектор
            self.get_logger().error(
                f"visualization failed: {type(e).__name__}: {e}", throttle_duration_sec=5.0
            )

        if self._results is not None:
            self._write_result(stamp, res, total_ms)

        self._frames += 1
        self._source_frames += 1
        self._last_res = res
        self._t_proc.append(total_ms)
        self._t_core.append(float(res.processing_ms))
        self._t_wall.append(time.monotonic())
        self._t_stamp.append(stamp)

    def _write_result(self, stamp: float, res: Any, total_ms: float) -> None:
        assert self._results is not None
        # frame — номер кадра от начала текущего источника.
        record = {"stamp": round(stamp, 6), "frame": self._source_frames, **res.to_json()}
        # processing_ms верхнего уровня — полное время кадра в ноде; время ядра — core_ms.
        record["core_ms"] = record.get("processing_ms")
        record["processing_ms"] = round(total_ms, 3)
        try:
            if self._results.write(record, self._results_tag or self._topic):
                self.get_logger().info(f"writing results to {self._results.path}")
        except OSError as e:
            self.get_logger().error(
                f"cannot write results under {self._results.spec!r}: {e}; not writing"
            )
            self._results.close()
            self._results = None

    # --- диагностика ----------------------------------------------------------

    @staticmethod
    def _rate(times: deque[float]) -> float:
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def _publish_diagnostics(self) -> None:
        st = DiagnosticStatus(name="metro_obstacle: detector", hardware_id=self._frame_id or "")
        res = self._last_res
        idle = not self._t_wall or time.monotonic() - self._t_wall[-1] > 2.0
        calibrated = bool(res.calibrated) if res is not None else False
        if not self._topic:
            st.level, st.message = DiagnosticStatus.WARN, "waiting for PointCloud2 topic"
        elif idle:
            st.level, st.message = DiagnosticStatus.WARN, "no frames in the last 2 s"
        elif not calibrated:
            st.level, st.message = DiagnosticStatus.WARN, "calibrating"
        else:
            st.level, st.message = DiagnosticStatus.OK, "running"
        if self._core_kind == "fake":
            st.level, st.message = DiagnosticStatus.ERROR, f"FAKE core; {st.message}"
        elif verifier_failed(self._detector, self._core_kind):
            if st.level == DiagnosticStatus.OK:
                st.level = DiagnosticStatus.WARN
            st.message = f"verifier unavailable, running {self._mode}; {st.message}"

        proc = np.array(self._t_proc) if self._t_proc else np.array([math.nan])
        corridor = np.asarray(res.corridor).reshape(-1, 3) if res is not None else np.zeros((0, 3))
        values = {
            "input_topic": self._topic,
            "frame_id": self._frame_id,
            "core": self._core_kind,
            "mode": self._mode,
            "verifier": getattr(self._detector, "verifier", None) is not None,
            "calibrated": calibrated,
            "frames_processed": self._frames,
            "frames_dropped": self._drops.dropped,
            "frames_failed": self._errors,
            "input_rate_hz": self._rate(self._t_stamp),
            "output_rate_hz": 0.0 if idle else self._rate(self._t_wall),
            "processing_ms_last": proc[-1],
            "processing_ms_p95": float(np.percentile(proc, 95)),
            "core_ms_last": self._t_core[-1] if self._t_core else math.nan,
            # Возраст кадра по часам ROS; при проигрывании bag без /clock не показателен.
            "stamp_age_ms": self._stamp_age_ms,
            "core_warmup_s": self._warmup_s if self._warmup_s is not None else math.nan,
            "corridor_length_m": corridor_length(corridor),
            "visible_range_m": float(res.visible_range) if res is not None else math.nan,
        }
        st.values = [
            KeyValue(key=k, value=f"{v:.1f}" if isinstance(v, float) else str(v))
            for k, v in values.items()
        ]
        diag = DiagnosticArray(status=[st])
        diag.header.stamp = self.get_clock().now().to_msg()
        self._pub_diag.publish(diag)

    def destroy_node(self) -> None:
        # Журнал — первым: все записанные кадры уже в файле (построчный буфер), close
        # закрывает его до того, как рвётся связь с ROS.
        if self._results is not None:
            self._results.close()
            self._results = None
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    try:
        node = DetectorNode()
    except (RuntimeError, ValueError) as e:
        # Понятное сообщение вместо трейсбека: ядро не установлено, неверный mode и т.п.
        rclpy.logging.get_logger("metro_obstacle_detector").fatal(str(e))
        rclpy.shutdown()
        raise SystemExit(1) from e
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ctrl-C под ros2 launch даёт ноде два SIGINT: от терминала (вся группа
        # процессов) и следом от launch. Второй не должен прервать закрытие журнала и
        # ноды трейсбеком KeyboardInterrupt — до выхода процесса SIGINT больше не нужен.
        signal.signal(signal.SIGINT, _ignore_signal)
        node.destroy_node()
        rclpy.try_shutdown()
        # Дальше — только выгрузка интерпретатора; при ней Python возвращает своему
        # обработчику SIGINT действие по умолчанию, и запоздалый сигнал убил бы процесс
        # (launch: «process has died, exit code -2»). SIG_IGN этот сброс переживает.
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _ignore_signal(signum: int, frame: object) -> None:
    """Обработчик-заглушка: повторный SIGINT во время завершения ничего не делает.

    Заглушка, а не SIG_IGN: rclpy.try_shutdown снимает обработчики rclpy и возвращает
    сохранённый при init обработчик Python, а тот передаёт сигнал сюда.
    """


if __name__ == "__main__":
    main()
