"""Обработка записи для веб-прототипа: bag (.db3) → ядро детектора → результаты по кадрам.

Без ROS и без HTTP — только данные, чтобы логику можно было проверить тестами.
Чтение записи и сборка маркеров — те же функции, что у bench/to_mcap.py и ноды
(to_mcap: поиск топика и файлов; cloud.py: разбор облака и кольца по углу места;
viz.py: коридор габарита и рамки объектов). Ядро — metro_obstacle_core.Detector
в режиме default, покадрово, со сбросом при скачке времени (как в ноде).

Облака в памяти не держатся: при обработке запоминается номер сообщения каждого
кадра в sqlite, а облако для 3D-вида читается из записи по запросу и прореживается.
"""

from __future__ import annotations

import json
import math
import shutil
import threading
import time
import uuid
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import numpy as np
from metro_obstacle import viz
from metro_obstacle.cloud import _points_view, cloud_to_arrays
from metro_obstacle.results_log import jsonable, safe_name
from metro_obstacle.source import ResetGuard

from bench import to_mcap
from metro_obstacle_core import Detector, warmup

MODE = to_mcap.MODE
MAX_VIEW_POINTS = 70_000  # точек облака в 3D-виде браузера
EVENT_GAP = 10  # кадров (1 с при 10 Гц) без тревоги внутри одного события тревоги
UPLOAD_CHUNK = 4 << 20


# --------------------------------------------------------------------------- записи


@dataclass(frozen=True)
class BagRef:
    """Запись в одном из корней (каталог BAGS_DIR или загрузки): id вида `bags:путь`."""

    root: str
    rel: str
    path: Path

    @property
    def id(self) -> str:
        return f"{self.root}:{self.rel}"

    @property
    def name(self) -> str:
        return self.path.stem if self.path.is_file() else self.path.name


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.glob("*.db3"))


def list_bags(roots: dict[str, Path], depth: int = 3) -> list[dict[str, Any]]:
    """Записи ROS 2 (sqlite) в корнях: каталоги с .db3 и одиночные файлы .db3."""
    found: list[dict[str, Any]] = []

    def walk(root: str, base: Path, cur: Path, level: int) -> None:
        try:
            entries = sorted(cur.iterdir())
        except OSError:
            return
        if any(e.suffix == ".db3" and e.is_file() for e in entries) and cur != base:
            ref = BagRef(root, cur.relative_to(base).as_posix(), cur)
            found.append(
                {"id": ref.id, "name": ref.name, "root": root, "rel": ref.rel, "size": _size(cur)}
            )
            return
        for e in entries:
            if e.name.startswith("."):
                continue
            if e.is_file() and e.suffix == ".db3":
                ref = BagRef(root, e.relative_to(base).as_posix(), e)
                found.append(
                    {"id": ref.id, "name": ref.name, "root": root, "rel": ref.rel, "size": _size(e)}
                )
            elif e.is_dir() and level < depth:
                walk(root, base, e, level + 1)

    for root, base in roots.items():
        if base.is_dir():
            walk(root, base, base, 0)
    return found


def resolve_bag(roots: dict[str, Path], bag_id: str) -> BagRef:
    """id `корень:путь` → запись; путь не выходит за корень."""
    root, sep, rel = bag_id.partition(":")
    if not sep or root not in roots:
        raise ValueError(f"unknown bag id: {bag_id!r}")
    base = roots[root].resolve()
    path = (base / rel).resolve()
    if path != base and base not in path.parents:
        raise ValueError(f"bag path outside of {root}: {rel!r}")
    to_mcap.db3_files(path)  # ValueError, если это не запись sqlite
    return BagRef(root, path.relative_to(base).as_posix(), path)


def save_upload(stream: BinaryIO, length: int, filename: str, uploads: Path) -> Path:
    """Сохранить загруженный файл (.db3 или .zip с каталогом записи); путь к записи.

    Тело читается кусками — запись в несколько ГБ не попадает в память целиком.
    """
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in (".db3", ".zip"):
        raise ValueError("нужен файл .db3 или .zip с каталогом записи ROS 2")
    target = uploads / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    target.mkdir(parents=True)
    stem = safe_name(Path(name).stem)
    dst = target / f"{stem}{suffix}"
    left = length
    with dst.open("wb") as f:
        while left > 0:
            chunk = stream.read(min(UPLOAD_CHUNK, left))
            if not chunk:
                break
            f.write(chunk)
            left -= len(chunk)
    if left:
        shutil.rmtree(target, ignore_errors=True)
        raise ValueError("загрузка оборвана")
    if suffix == ".db3":
        bag = target / stem
        bag.mkdir()
        dst.rename(bag / dst.name)
        return bag
    try:
        with zipfile.ZipFile(dst) as z:
            # extractall отбрасывает абсолютные пути и «..» в именах архива
            z.extractall(target / stem)
    except zipfile.BadZipFile as e:
        shutil.rmtree(target, ignore_errors=True)
        raise ValueError(f"не zip-архив: {e}") from e
    finally:
        dst.unlink(missing_ok=True)
    # Служебное macOS (__MACOSX/, ._файл) и скрытое — не запись: иначе «первым»
    # окажется каталог с ._*.db3 из архива, собранного в Finder.
    root = target / stem
    dirs = sorted(
        {
            p.parent
            for p in root.rglob("*.db3")
            if p.is_file()
            and not any(
                part.startswith(".") or part == "__MACOSX" for part in p.relative_to(root).parts
            )
        }
    )
    if not dirs:
        shutil.rmtree(target, ignore_errors=True)
        raise ValueError("в архиве нет файлов .db3 (нужна запись ROS 2 в формате sqlite3)")
    return dirs[0]


# --------------------------------------------------------------------------- кадры


@dataclass(frozen=True)
class FrameRef:
    """Где лежит кадр: номер файла .db3 и id сообщения в нём."""

    file: int
    msg_id: int
    log_ns: int


def index_frames(files: list[Path], topic: str) -> list[FrameRef]:
    """Список кадров топика по порядку записи (без чтения облаков)."""
    refs: list[FrameRef] = []
    for k, path in enumerate(files):
        with to_mcap._connect(path) as db:
            row = db.execute("SELECT id FROM topics WHERE name = ?", (topic,)).fetchone()
            if row is None:
                continue
            cur = db.execute(
                "SELECT id, timestamp FROM messages WHERE topic_id = ? ORDER BY id", (row[0],)
            )
            refs.extend(FrameRef(k, int(i), int(t)) for i, t in cur)
    return refs


def read_message(files: list[Path], ref: FrameRef) -> Any:
    """Облако кадра (sensor_msgs/PointCloud2 из rosbags)."""
    db = to_mcap._connect(files[ref.file])
    try:
        row = db.execute("SELECT data FROM messages WHERE id = ?", (ref.msg_id,)).fetchone()
    finally:
        db.close()
    if row is None:
        raise KeyError(f"message {ref.msg_id} not found")
    return to_mcap._TS.deserialize_cdr(bytes(row[0]), to_mcap.POINTCLOUD2)


def view_cloud(msg: Any, max_points: int = MAX_VIEW_POINTS) -> np.ndarray:
    """Облако для браузера: (M, 4) float32 x, y, z, intensity; не больше max_points.

    Кольца здесь не нужны, поэтому поля читаются напрямую (без восстановления колец).
    Прореживание — случайная выборка с фиксированным зерном: у лидара точки идут по
    столбцам, и равномерный шаг дал бы муар.
    """
    pts = _points_view(msg)
    names = pts.dtype.names or ()
    out = np.zeros((len(pts), 4), np.float32)
    for k, axis in enumerate(("x", "y", "z")):
        out[:, k] = pts[axis]
    if "intensity" in names:
        out[:, 3] = pts["intensity"]
    ok = np.isfinite(out).all(axis=1) & (np.abs(out[:, :3]).sum(axis=1) > 0)
    out = out[ok]
    if len(out) > max_points:
        idx = np.random.default_rng(0).choice(len(out), max_points, replace=False)
        out = out[np.sort(idx)]
    return np.ascontiguousarray(out, "<f4")


# --------------------------------------------------------------------------- геометрия


def _obj(d: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**{k: (math.nan if v is None else v) for k, v in d.items()})


def _pts(a: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(a, np.float64), 3).tolist()


def frame_scene(
    record: dict[str, Any], corridor: np.ndarray, half: np.ndarray, gauge_height: float
) -> dict[str, Any]:
    """Коридор габарита и рамки объектов кадра — те же, что в маркерах ноды (viz.py)."""
    res = SimpleNamespace(
        level=record["level"],
        obstacle=record["obstacle"],
        distance=math.nan if record["distance"] is None else record["distance"],
        obstacles=[_obj(o) for o in record["obstacles"]],
        corridor=corridor,
        corridor_half_width=half,
    )
    lines = viz.corridor_lines(res, gauge_height) if len(corridor) >= 2 else None
    boxes = []
    for o, raw in zip(
        viz.sorted_obstacles(res),
        sorted(record["obstacles"], key=lambda o: o["distance"]),
        strict=True,
    ):
        boxes.append(
            {
                **raw,
                "center": [float(v) for v in o.position],
                "size": list(viz.box_size(o)),
                "yaw": float(viz.box_yaw(res, o)) if len(corridor) >= 2 else 0.0,
                "label": f"{o.distance:.1f} м",
            }
        )
    return {
        "corridor": (
            None
            if lines is None
            else {
                "edges": [_pts(e) for e in lines.edges],
                "sections": _pts(lines.sections),
                "axis": _pts(corridor),
            }
        ),
        "color": list(viz.level_color(res)),
        "boxes": boxes,
    }


# --------------------------------------------------------------------------- сводка


def _pct(values: list[float], q: float) -> float | None:
    v = [x for x in values if x is not None and math.isfinite(x)]
    return round(float(np.percentile(v, q)), 2) if v else None


def alarm_events(records: list[dict[str, Any]], gap: int = EVENT_GAP) -> list[dict[str, Any]]:
    """События тревоги: подряд идущие кадры с подтверждённым объектом.

    Разрыв короче gap кадров не делит событие (кадр-другой без подтверждения — дребезг).
    """
    events: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for r in records:
        if not r["obstacle"]:
            continue
        k = r["frame"]
        near = min(
            (o for o in r["obstacles"] if o["confirmed"]), key=lambda o: o["distance"], default=None
        )
        if cur is None or k - cur["end"] > gap:
            cur = {
                "start": k,
                "end": k,
                "frames": 0,
                "first_distance": r["distance"],
                "min_distance": r["distance"],
                "width": None,
                "height": None,
                "length": None,
            }
            events.append(cur)
        cur["end"] = k
        cur["frames"] += 1
        if r["distance"] is not None and (
            cur["min_distance"] is None or r["distance"] <= cur["min_distance"]
        ):
            cur["min_distance"] = r["distance"]
        if near is not None:
            for dim in ("width", "height", "length"):
                v = near.get(dim)
                if v is not None and (cur[dim] is None or v > cur[dim]):
                    cur[dim] = v
    return events


def summarize(
    records: list[dict[str, Any]], total: int, t0: float | None, wall_s: float
) -> dict[str, Any]:
    """Сводка по обработанным кадрам записи."""
    core = [r["core_ms"] for r in records]
    full = [r["processing_ms"] for r in records]
    alarms = [r for r in records if r["obstacle"]]
    first = alarms[0] if alarms else None
    dists = [r["distance"] for r in alarms if r["distance"] is not None]
    span = (records[-1]["stamp"] - records[0]["stamp"]) if len(records) > 1 else 0.0
    return {
        "frames": len(records),
        "total_frames": total,
        "alarm_frames": len(alarms),
        "candidate_frames": sum(1 for r in records if r["level"] == 1),
        "first_alarm": (
            None
            if first is None
            else {
                "frame": first["frame"],
                "t": None if t0 is None else round(first["stamp"] - t0, 2),
                "distance": first["distance"],
            }
        ),
        "min_distance": min(dists) if dists else None,
        "core_ms": {
            "min": _pct(core, 0),
            "median": _pct(core, 50),
            "p95": _pct(core, 95),
            "max": _pct(core, 100),
        },
        "frame_ms": {"median": _pct(full, 50), "p95": _pct(full, 95)},
        "record_s": round(span, 2),
        "wall_s": round(wall_s, 2),
    }


# --------------------------------------------------------------------------- задачи


@dataclass
class Job:
    id: str
    bag: BagRef
    dir: Path
    max_frames: int | None = None
    state: str = "queued"  # queued | running | done | error
    error: str | None = None
    topic: str | None = None
    total: int = 0
    warmup_s: float | None = None
    wall_s: float = 0.0
    files: list[Path] = field(default_factory=list)
    refs: list[FrameRef] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    corridors: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    gauge_height: float = 0.0
    resets: list[dict[str, Any]] = field(default_factory=list)
    mcap_state: str = "none"  # none | queued | running | done | error
    mcap_done: int = 0
    mcap_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def results_path(self) -> Path:
        return self.dir / "results.jsonl"

    @property
    def mcap_path(self) -> Path:
        return self.dir / f"{safe_name(self.bag.name)}.mcap"

    def status(self) -> dict[str, Any]:
        with self.lock:
            records = list(self.records)
        t0 = records[0]["stamp"] if records else None
        return {
            "id": self.id,
            "bag": {"id": self.bag.id, "name": self.bag.name},
            "state": self.state,
            "error": self.error,
            "topic": self.topic,
            "mode": MODE,
            "done": len(records),
            "total": self.total,
            "warmup_s": self.warmup_s,
            "resets": self.resets,
            "summary": summarize(records, self.total, t0, self.wall_s),
            "events": alarm_events(records),
            "mcap": {"state": self.mcap_state, "done": self.mcap_done, "error": self.mcap_error},
        }

    def series(self) -> dict[str, Any]:
        """Компактные ряды по кадрам для графика и слайдера."""
        with self.lock:
            records = list(self.records)
        t0 = records[0]["stamp"] if records else 0.0
        return {
            "t": [round(r["stamp"] - t0, 3) for r in records],
            "distance": [r["distance"] for r in records],
            "nearest": [
                min((o["distance"] for o in r["obstacles"]), default=None) for r in records
            ],
            "level": [r["level"] for r in records],
            "obstacle": [r["obstacle"] for r in records],
            "core_ms": [r["core_ms"] for r in records],
            # дорожки таймлайна: полное время кадра, видимая дальность, уверенность
            "frame_ms": [r["processing_ms"] for r in records],
            "visible_range": [r.get("visible_range") for r in records],
            "confidence": [r.get("confidence") for r in records],
        }

    def frame(self, k: int) -> dict[str, Any]:
        with self.lock:
            if not 0 <= k < len(self.records):
                raise KeyError(k)
            record = self.records[k]
            corridor, half = self.corridors[k]
            t0 = self.records[0]["stamp"]
        return {
            **record,
            "t": round(record["stamp"] - t0, 3),
            **frame_scene(record, corridor, half, self.gauge_height),
        }

    def cloud(self, k: int, max_points: int = MAX_VIEW_POINTS) -> np.ndarray:
        if not 0 <= k < len(self.refs):
            raise KeyError(k)
        return view_cloud(read_message(self.files, self.refs[k]), max_points)


def run_job(job: Job, log: Callable[[str], None] = print) -> None:
    """Прогнать запись через ядро; результаты — в job (по мере обработки) и results.jsonl."""
    try:
        job.state = "running"
        job.files = to_mcap.db3_files(job.bag.path)
        job.topic = to_mcap.find_cloud_topic(job.files)
        refs = index_frames(job.files, job.topic)
        if job.max_frames is not None:
            refs = refs[: job.max_frames]
        job.refs, job.total = refs, len(refs)
        job.warmup_s = round(float(warmup()), 2)
        det = Detector(mode=MODE, forward_axis="auto")
        job.gauge_height = float(det.params.gauge_height)
        guard = ResetGuard()
        job.dir.mkdir(parents=True, exist_ok=True)
        log(f"job {job.id}: {job.bag.path} topic {job.topic}, {job.total} frames")
        t_start = time.perf_counter()
        with job.results_path.open("w", encoding="utf-8") as out:
            for k, ref in enumerate(refs):
                t0 = time.perf_counter()
                msg = read_message(job.files, ref)
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                reason = guard.check(stamp, msg.header.frame_id or to_mcap.FIXED_FRAME)
                if reason is not None:
                    job.resets.append({"frame": k, "reason": reason})
                    det = Detector(mode=MODE, forward_axis="auto")
                xyz, ring, intensity = cloud_to_arrays(msg)
                res = det.process(xyz, ring, intensity, stamp)
                total_ms = (time.perf_counter() - t0) * 1e3
                # Как строка results.jsonl ноды: processing_ms — полное время кадра
                # (чтение из записи + разбор + ядро), время ядра — core_ms.
                record = jsonable({"stamp": round(stamp, 6), "frame": k, **res.to_json()})
                record["core_ms"] = record["processing_ms"]
                record["processing_ms"] = round(total_ms, 3)
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                corridor = np.asarray(res.corridor, np.float32).reshape(-1, 3)
                half = np.broadcast_to(
                    np.asarray(res.corridor_half_width, np.float32), (len(corridor),)
                ).copy()
                with job.lock:
                    job.records.append(record)
                    job.corridors.append((corridor, half))
                    job.wall_s = time.perf_counter() - t_start
        job.state = "done"
        log(f"job {job.id}: done, {job.total} frames in {job.wall_s:.1f} s")
    except Exception as e:
        job.state = "error"
        job.error = f"{type(e).__name__}: {e}"
        log(f"job {job.id}: error {job.error}")


def build_mcap(job: Job, log: Callable[[str], None] = print) -> None:
    """MCAP для Lichtblick/Foxglove тем же конвейером, что bench/to_mcap.py."""
    try:
        job.mcap_state = "running"
        tmp = job.mcap_path.with_suffix(".mcap.part")

        def progress(k: int) -> None:
            job.mcap_done = k

        to_mcap.convert(
            job.bag.path, tmp, job.topic, 0, job.max_frames, log=lambda _: None, progress=progress
        )
        tmp.replace(job.mcap_path)
        job.mcap_state = "done"
    except Exception as e:
        job.mcap_state = "error"
        job.mcap_error = f"{type(e).__name__}: {e}"
        log(f"job {job.id}: mcap error {job.mcap_error}")


class JobManager:
    """Задачи обработки: по одной за раз (ядро однопоточное, нагрузка предсказуема)."""

    def __init__(self, work_dir: Path, log: Callable[[str], None] = print) -> None:
        self.work_dir = work_dir
        self.log = log
        self.jobs: dict[str, Job] = {}
        self._detect = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detect")
        self._mcap = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcap")

    def submit(self, bag: BagRef, max_frames: int | None = None) -> Job:
        jid = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        job = Job(id=jid, bag=bag, dir=self.work_dir / "jobs" / jid, max_frames=max_frames)
        self.jobs[jid] = job
        self._detect.submit(run_job, job, self.log)
        return job

    def submit_mcap(self, job: Job) -> None:
        if job.state != "done":
            raise ValueError("запись ещё не обработана")
        if job.mcap_state in ("queued", "running", "done"):
            return
        job.mcap_state, job.mcap_done, job.mcap_error = "queued", 0, None
        self._mcap.submit(build_mcap, job, self.log)

    def get(self, jid: str) -> Job:
        return self.jobs[jid]

    def list(self) -> Iterable[Job]:
        return self.jobs.values()

    def shutdown(self) -> None:
        self._detect.shutdown(wait=False, cancel_futures=True)
        self._mcap.shutdown(wait=False, cancel_futures=True)
