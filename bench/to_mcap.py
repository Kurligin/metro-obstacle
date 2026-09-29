"""Запись ROS 2 (sqlite .db3) → ядро детектора → MCAP для Lichtblick / Foxglove. Без ROS.

Каждый кадр облака прогоняется через ядро (режим default, как в ноде), в MCAP пишутся
стандартные сообщения ROS 2 (схемы ros2msg, сообщения cdr) — просмотрщик читает их
без наших определений:

    /lidar_points                 sensor_msgs/PointCloud2  — исходное облако как есть
    /metro_obstacle/markers       visualization_msgs/MarkerArray — коридор габарита,
                                  рамки объектов (красный — подтверждён, жёлтый —
                                  кандидат), дистанции, строка статуса
    /metro_obstacle/status        std_msgs/String — JSON {obstacle, distance, calibrated}
    /metro_obstacle/in_corridor   sensor_msgs/PointCloud2 — точки внутри габарита выше
                                  полотна (то, что ядро рассматривает как кандидатов)

frame_id — как у входного облака; время — из записи: log_time = время записи в bag,
заголовки — штамп облака. Раскладка для просмотра: docs/lichtblick_layout.json.

Запуск из корня репозитория (облако — первый PointCloud2-топик, либо --topic):

    NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m bench.to_mcap \\
        /path/to/bag --max-frames 100
"""

from __future__ import annotations

import os

# Нагрузка ноутбука: numpy/numba — в один поток (до их импорта).
for _v in (
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Чистые модули ROS-пакета (разбор облака, геометрия маркеров) — те же, что в ноде.
_NODE_PKG = str(ROOT / "ros2_ws" / "src" / "metro_obstacle")
if _NODE_PKG not in sys.path:
    sys.path.insert(0, _NODE_PKG)

from mcap_ros2.writer import Writer  # noqa: E402
from metro_obstacle import viz  # noqa: E402
from metro_obstacle.cloud import cloud_to_arrays  # noqa: E402
from metro_obstacle.source import ResetGuard  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

from metro_obstacle_core import Detector, warmup  # noqa: E402

POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"
MARKER_ARRAY = "visualization_msgs/msg/MarkerArray"
STRING = "std_msgs/msg/String"

TOPIC_CLOUD = "/lidar_points"
TOPIC_MARKERS = "/metro_obstacle/markers"
TOPIC_STATUS = "/metro_obstacle/status"
TOPIC_IN_CORRIDOR = "/metro_obstacle/in_corridor"

MODE = "default"  # как у ноды по умолчанию
FIXED_FRAME = "lidar_link"  # если у облака пустой frame_id (как в ноде)

# visualization_msgs/Marker
CUBE, LINE_STRIP, LINE_LIST, TEXT_VIEW_FACING, TRIANGLE_LIST = 1, 4, 5, 9, 11
ADD, DELETEALL = 0, 3
# Каждый кадр начинается с DELETEALL; lifetime — страховка, чтобы маркеры не копились,
# если просмотрщик пропустит сообщение. Кадры идут с 10 Гц.
MARKER_LIFETIME_NS = 500_000_000
LABEL_SIZE = 1.2  # м, высота букв дистанции у объекта
LABEL_LIFT = 0.8  # м, подпись над верхом рамки
TITLE_SIZE = 1.5  # м, строка статуса над лидаром
TITLE_Z = 3.0
FILL_ALPHA = 0.12  # полупрозрачная лента габарита по полотну

_TS = get_typestore(Stores.ROS2_HUMBLE)


# --------------------------------------------------------------------------- вход


def db3_files(path: Path) -> list[Path]:
    """Файлы .db3 записи: каталог bag (все .db3 по порядку) или сам файл .db3."""
    path = Path(path).expanduser()
    if path.is_file():
        if path.suffix != ".db3":
            raise ValueError(f"not a .db3 file: {path}")
        return [path]
    files = sorted(path.glob("*.db3"))
    if not files:
        raise ValueError(f"no .db3 files in {path}")
    return files


def _connect(path: Path) -> sqlite3.Connection:
    # Только чтение, без блокировок и журнала: запись не трогаем.
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def find_cloud_topic(files: list[Path], topic: str | None = None) -> str:
    """Топик облака: заданный (проверяется тип) или первый PointCloud2 в записи."""
    with _connect(files[0]) as db:
        rows = db.execute("SELECT name, type FROM topics ORDER BY id").fetchall()
    clouds = [name for name, mtype in rows if mtype == POINTCLOUD2]
    if topic is not None:
        if topic not in clouds:
            raise ValueError(f"{topic} is not a {POINTCLOUD2} topic here (have: {clouds})")
        return topic
    if not clouds:
        raise ValueError(f"no {POINTCLOUD2} topic in {files[0]} (topics: {rows})")
    return clouds[0]


def iter_raw(
    files: list[Path], topic: str, start: int = 0, max_frames: int | None = None
) -> Iterator[tuple[int, bytes]]:
    """(время записи, нс; CDR) сообщений топика по порядку записи, начиная с кадра start.

    Пропуск через OFFSET: большие блобы облаков пропущенных кадров не читаются.
    """
    left = max_frames if max_frames is not None else -1
    skip = max(0, start)
    for path in files:
        if left == 0:
            return
        with _connect(path) as db:
            row = db.execute("SELECT id FROM topics WHERE name = ?", (topic,)).fetchone()
            if row is None:
                continue
            (count,) = db.execute(
                "SELECT COUNT(*) FROM messages WHERE topic_id = ?", (row[0],)
            ).fetchone()
            if skip >= count:
                skip -= count
                continue
            cur = db.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? "
                "ORDER BY id LIMIT ? OFFSET ?",
                (row[0], left, skip),
            )
            skip = 0
            for stamp, data in cur:
                yield int(stamp), bytes(data)
                left -= 1


# --------------------------------------------------------------------------- сообщения


def _header(header: Any) -> dict[str, Any]:
    return {
        "stamp": {"sec": int(header.stamp.sec), "nanosec": int(header.stamp.nanosec)},
        "frame_id": header.frame_id or FIXED_FRAME,
    }


def cloud_message(msg: Any) -> dict[str, Any]:
    """Исходное облако без изменений (поля, шаги, байты точек), frame_id — как был."""
    return {
        "header": _header(msg.header),
        "height": int(msg.height),
        "width": int(msg.width),
        "fields": [
            {
                "name": f.name,
                "offset": int(f.offset),
                "datatype": int(f.datatype),
                "count": int(f.count),
            }
            for f in msg.fields
        ],
        "is_bigendian": bool(msg.is_bigendian),
        "point_step": int(msg.point_step),
        "row_step": int(msg.row_step),
        "data": np.asarray(msg.data, np.uint8).tobytes(),
        "is_dense": bool(msg.is_dense),
    }


def points_message(header: dict[str, Any], xyz: np.ndarray) -> dict[str, Any]:
    """Облако из точек (M, 3): поля x, y, z float32."""
    xyz = np.ascontiguousarray(xyz, "<f4").reshape(-1, 3)
    return {
        "header": header,
        "height": 1,
        "width": len(xyz),
        "fields": [
            {"name": n, "offset": 4 * k, "datatype": 7, "count": 1}
            for k, n in enumerate(("x", "y", "z"))
        ],
        "is_bigendian": False,
        "point_step": 12,
        "row_step": 12 * len(xyz),
        "data": xyz.tobytes(),
        "is_dense": True,
    }


def status_json(res: Any) -> str:
    """{obstacle, distance, calibrated}; нет подтверждённого объекта — distance: null."""
    d = float(res.distance)
    return json.dumps(
        {
            "obstacle": bool(res.obstacle),
            "distance": round(d, 2) if math.isfinite(d) else None,
            "calibrated": bool(res.calibrated),
        }
    )


def _xyz(p: Any) -> dict[str, float]:
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


def _rgba(c: viz.RGBA, alpha: float | None = None) -> dict[str, float]:
    return {"r": c[0], "g": c[1], "b": c[2], "a": c[3] if alpha is None else alpha}


def _marker(
    header: dict[str, Any], ns: str, mid: int, mtype: int, color: dict[str, float]
) -> dict[str, Any]:
    return {
        "header": header,
        "ns": ns,
        "id": mid,
        "type": mtype,
        "action": ADD,
        "pose": {
            "position": _xyz((0, 0, 0)),
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "scale": _xyz((1.0, 1.0, 1.0)),
        "color": color,
        "lifetime": {"sec": 0, "nanosec": MARKER_LIFETIME_NS},
        "frame_locked": False,
        "points": [],
    }


def _ribbon(left: np.ndarray, right: np.ndarray) -> list[dict[str, float]]:
    """Треугольники ленты между кромками (TRIANGLE_LIST)."""
    tris = [
        p
        for i in range(len(left) - 1)
        for p in (left[i], right[i], left[i + 1], left[i + 1], right[i], right[i + 1])
    ]
    return [_xyz(p) for p in tris]


def markers_message(res: Any, header: dict[str, Any], gauge_height: float) -> dict[str, Any]:
    """Маркеры кадра — те же, что у ноды (viz.py), плюс полупрозрачная лента габарита."""
    markers = [{**_marker(header, "", 0, 0, _rgba(viz.WHITE)), "action": DELETEALL}]
    lines = viz.corridor_lines(res, gauge_height)
    if lines is not None:
        color = _rgba(viz.level_color(res))
        for k, edge in enumerate(lines.edges):
            m = _marker(header, "corridor", k, LINE_STRIP, color)
            m["scale"] = _xyz((0.06, 0.0, 0.0))
            m["points"] = [_xyz(p) for p in edge]
            markers.append(m)
        frames = _marker(header, "corridor", 4, LINE_LIST, color)
        frames["scale"] = _xyz((0.03, 0.0, 0.0))
        frames["points"] = [_xyz(p) for p in lines.sections.reshape(-1, 3)]
        markers.append(frames)
        fill = _marker(
            header, "corridor_fill", 0, TRIANGLE_LIST, _rgba(viz.level_color(res), FILL_ALPHA)
        )
        fill["points"] = _ribbon(lines.edges[0], lines.edges[1])
        markers.append(fill)

    for k, o in enumerate(viz.sorted_obstacles(res)):
        color = _rgba(viz.obstacle_color(o))
        yaw = viz.box_yaw(res, o)
        box = _marker(header, "obstacles", k, CUBE, color)
        box["pose"] = {
            "position": _xyz(o.position),
            "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2), "w": math.cos(yaw / 2)},
        }
        box["scale"] = _xyz(viz.box_size(o))
        markers.append(box)

        label = _marker(header, "obstacle_labels", k, TEXT_VIEW_FACING, color)
        label["pose"]["position"] = _xyz(viz.label_position(o, lift=LABEL_LIFT))
        label["scale"] = _xyz((0.0, 0.0, LABEL_SIZE))
        label["text"] = f"{o.distance:.1f} m"
        markers.append(label)

    title = _marker(header, "status", 0, TEXT_VIEW_FACING, _rgba(viz.level_color(res, viz.WHITE)))
    title["pose"]["position"] = _xyz((0.0, 0.0, TITLE_Z))
    title["scale"] = _xyz((0.0, 0.0, TITLE_SIZE))
    title["text"] = viz.status_text(res)
    markers.append(title)
    return {"markers": markers}


def corridor_points(det: Detector) -> np.ndarray | None:
    """Точки последнего кадра внутри габарита выше полотна, в осях сообщения (M, 3).

    Берутся из отладочных массивов ядра (det.debug): маска cand — точки у оси пути,
    попавшие в габарит по ширине и высоте и выше порога над полотном. Массивы ядра —
    в выровненных нормализованных осях; обратно — теми же преобразованиями, что
    строят коридор (FrameResult.corridor). None — ось на кадре не построена.
    """
    d = det.dbg
    if d is None:
        return None
    m = np.asarray(d["cand"], bool)
    ux, uy, uz = det._unlevel(
        np.asarray(d["px"], np.float64)[m],
        np.asarray(d["py"], np.float64)[m],
        np.asarray(d["pz"], np.float64)[m],
    )
    mx, my = det._to_msg(ux, uy)
    return np.stack([mx, my, uz], 1).astype(np.float32)


# --------------------------------------------------------------------------- прогон


@dataclass
class Stats:
    frames: int
    seconds: float  # прогон без прогрева JIT
    warmup_seconds: float
    core_ms_mean: float
    obstacle_frames: int
    topic: str
    out: Path


def _schemas(writer: Writer) -> dict[str, Any]:
    return {
        name: writer.register_msgdef(name, _TS.generate_msgdef(name, ros_version=2)[0])
        for name in (POINTCLOUD2, MARKER_ARRAY, STRING)
    }


def convert(
    bag: Path,
    out: Path,
    topic: str | None = None,
    start: int = 0,
    max_frames: int | None = None,
    log: Callable[[str], None] = print,
    progress: Callable[[int], None] | None = None,
) -> Stats:
    """progress(k) — после каждого записанного кадра (k — сколько кадров готово)."""
    files = db3_files(bag)
    topic = find_cloud_topic(files, topic)
    warm = float(warmup())
    log(f"input {bag} topic {topic}; core warm-up {warm:.1f} s; writing {out}")

    det = Detector(mode=MODE, forward_axis="auto")
    gauge_height = float(det.params.gauge_height)
    guard = ResetGuard()
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = obstacles = 0
    core_ms = 0.0
    t0 = time.perf_counter()
    with out.open("wb") as f, Writer(f) as writer:
        sch = _schemas(writer)
        for log_ns, raw in iter_raw(files, topic, start, max_frames):
            msg = _TS.deserialize_cdr(raw, POINTCLOUD2)
            header = _header(msg.header)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            # Как в ноде: скачок времени или смена frame_id — ядро с чистого листа.
            reason = guard.check(stamp, header["frame_id"])
            if reason is not None:
                log(f"detector reset: {reason}")
                det = Detector(mode=MODE, forward_axis="auto")
            det.debug, det.dbg = True, None
            xyz, ring, intensity = cloud_to_arrays(msg)
            res = det.process(xyz, ring, intensity, stamp)

            writer.write_message(TOPIC_CLOUD, sch[POINTCLOUD2], cloud_message(msg), log_ns)
            writer.write_message(
                TOPIC_MARKERS, sch[MARKER_ARRAY], markers_message(res, header, gauge_height), log_ns
            )
            writer.write_message(TOPIC_STATUS, sch[STRING], {"data": status_json(res)}, log_ns)
            inside = corridor_points(det)
            if inside is not None:
                writer.write_message(
                    TOPIC_IN_CORRIDOR, sch[POINTCLOUD2], points_message(header, inside), log_ns
                )

            frames += 1
            obstacles += int(res.obstacle)
            core_ms += float(res.processing_ms)
            if progress is not None:
                progress(frames)
            if frames % 10 == 0:
                dt = time.perf_counter() - t0
                log(
                    f"frame {start + frames}: {frames} done, {frames / dt:.1f} fps; "
                    f"{viz.status_text(res)}; calibrated={res.calibrated}"
                )
    seconds = time.perf_counter() - t0
    return Stats(
        frames=frames,
        seconds=seconds,
        warmup_seconds=warm,
        core_ms_mean=core_ms / frames if frames else math.nan,
        obstacle_frames=obstacles,
        topic=topic,
        out=out,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", type=Path, help="каталог записи ROS 2 или файл .db3")
    ap.add_argument("--out", type=Path, help="файл .mcap (по умолчанию out/<запись>.mcap)")
    ap.add_argument("--topic", help="топик облака (по умолчанию — первый PointCloud2)")
    ap.add_argument("--start", type=int, default=0, help="с какого кадра начать (от 0)")
    ap.add_argument("--max-frames", type=int, help="сколько кадров обработать")
    a = ap.parse_args(argv)
    bag = a.bag.expanduser()
    out = a.out or ROOT / "out" / f"{bag.stem if bag.is_file() else bag.name}.mcap"
    try:
        st = convert(bag, out, a.topic, a.start, a.max_frames, log=lambda s: print(s, flush=True))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    size_mb = st.out.stat().st_size / 1e6
    fps = st.frames / max(st.seconds, 1e-9)
    print(
        f"done: {st.frames} frames in {st.seconds:.1f} s ({fps:.1f} fps, "
        f"core {st.core_ms_mean:.0f} ms/frame), obstacle in {st.obstacle_frames} frames; "
        f"{st.out} {size_mb:.1f} MB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
