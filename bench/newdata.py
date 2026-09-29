"""Ложные тревоги на расширенном датасете (new_data: одна длинная запись, ~220 кусков .db3).

Куски читаются напрямую через sqlite (без metadata.yaml), по порядку номера. Для
параллельности вся запись режется на WORKERS непрерывных отрезков; каждый вариант
детектора проходит отрезок последовательно (прогрев в начале отрезка — как после
перезапуска ноды). Пишем по кадрам: путь по оценке скорости, тревогу, объекты.

    python -m bench.newdata run <каталог с new_data_*.db3> [варианты через запятую]
    python -m bench.newdata report
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ros2_ws/src/metro_obstacle"))
from metro_obstacle.cloud import cloud_to_arrays  # noqa: E402

from bench.customer import make_variant  # noqa: E402
from bench.data import WORKERS  # noqa: E402
from bench.verifier_e2e import OUT  # noqa: E402

warnings.filterwarnings("ignore")
RES = OUT.parent / "newdata"


def files(root: Path) -> list[Path]:
    fs = list(root.glob("*.db3"))
    return sorted(fs, key=lambda p: int(re.findall(r"(\d+)\.db3$", p.name)[0]))


def messages(path: Path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    topics = {i: (n, t) for i, n, t in con.execute("select id, name, type from topics")}
    for tid, ts_, data in con.execute("select topic_id, timestamp, data from messages order by timestamp"):
        name, typ = topics[tid]
        if typ.endswith("PointCloud2"):
            yield ts_, typ, data
    con.close()


def run_seg(job):
    name, seg_id, paths = job
    from rosbags.typesys import Stores, get_typestore

    det = make_variant(name)
    travel = [0.0]
    step0 = det.speed.step

    def step(pr):
        sh = step0(pr)
        travel[0] += sh
        return sh

    det.speed.step = step
    ts = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    for p in paths:
        for t, typ, raw in messages(Path(p)):
            msg = ts.deserialize_cdr(raw, "sensor_msgs/msg/PointCloud2")
            xyz, ring, it = cloud_to_arrays(msg)
            res = det.process(xyz, ring, it, t * 1e-9)
            obs = [
                [round(o.distance, 2), round(o.lateral, 2), round(o.height, 2), round(o.width, 2), o.points, bool(o.confirmed)]
                for o in res.obstacles
            ]
            rows.append([Path(p).name, t, round(travel[0], 2), bool(res.obstacle), round(res.processing_ms, 1), obs])
    return name, seg_id, rows


def run():
    root = Path(sys.argv[2])
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else ["geometry", "geometry+top", "low+ver", "low+ver2"]
    fs = [str(p) for p in files(root)]
    nseg = max(1, WORKERS // len(names))
    segs = [list(a) for a in np.array_split(fs, nseg) if len(a)]
    jobs = [(n, i, s) for n in names for i, s in enumerate(segs)]
    RES.mkdir(parents=True, exist_ok=True)
    out = {}
    with ProcessPoolExecutor(WORKERS) as ex:
        for name, i, rows in ex.map(run_seg, jobs):
            out.setdefault(name, {})[i] = rows
            print(f"{name} отрезок {i} готов: {len(rows)} кадров", flush=True)
    old = json.loads((RES / "newdata.json").read_text()) if (RES / "newdata.json").exists() else {}
    out = {**old, **out}
    (RES / "newdata.json").write_text(json.dumps(out))
    report(out)


def report(out=None):
    out = out or json.loads((RES / "newdata.json").read_text())
    for name, segs in out.items():
        fr = al = ev = 0
        km = 0.0
        events = []
        for i in sorted(segs, key=int):
            rows = segs[i]
            prev = False
            for r in rows:
                fr += 1
                al += r[3]
                if r[3] and not prev:
                    ev += 1
                    conf = [o for o in r[5] if o[5]]
                    events.append((r[0], r[1], r[2], conf[0] if conf else None))
                prev = r[3]
            if rows:
                km += (rows[-1][2] - rows[0][2]) / 1000
        ms = np.percentile([r[4] for s in segs.values() for r in s], 95)
        print(f"\n== {name}: кадров {fr}, путь {km:.2f} км, тревога {100 * al / max(fr, 1):.2f}% кадров, "
              f"{ev} событий = {ev / max(km, 1e-9):.1f} на км, p95 {ms:.0f} мс")
        for f, t, s, o in events[:60]:
            print(f"   {f:22s} t={t} путь {s:8.1f}  [дист, бок, выс, шир, точек] {o}")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
