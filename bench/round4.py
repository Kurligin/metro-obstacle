"""Четвёртый круг: хрупкость ядра и варианты модели полотна.

Детектор задаётся строкой «режим|параметр=значение,…» (ядро metro_obstacle_core),
искажение установки — «roll=2,pitch=-0.5,yaw=2,dz=0.3» (градусы, метры; dz > 0 —
лидар выше). Искажение применяется ко всему кадру ПОСЛЕ вставки синтетики — как
если бы тот же тоннель снимал лидар, установленный иначе.

    python -m bench.round4 empty   — ложные тревоги на 5 пустых bag
    python -m bench.round4 synth   — Pd на синтетике (в т.ч. без второго отражения)
    python -m bench.round4 report
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench.data import BAGS, CACHE_ROOT, WORKERS, Bag

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "out" / os.environ.get("ROUND_OUT", "round4")
EMPTY_BAGS = [b for b in BAGS if b != "doubleT_obstacle"]

DETS = [
    "geometry",
    "geometry|edge_reject=1,end_guard=0.0",
    "strict|edge_reject=1,end_guard=0.0",
]
TFS = ["", "roll=2", "roll=-2", "yaw=2", "yaw=-2", "pitch=0.5", "pitch=-0.5"]
# искажения гоняем для базового и одного улучшенного детектора, остальное — без искажений
PAIRS = [(d, "") for d in DETS]


def make(spec: str):
    from metro_obstacle_core import Detector

    mode, _, rest = spec.partition("|")
    kw = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        if "/" in v:
            kw[k] = tuple(int(q) for q in v.split("/"))
        elif v.replace(".", "", 1).lstrip("-").isdigit():
            kw[k] = float(v) if "." in v else int(v)
        else:
            kw[k] = v
    return Detector(mode=mode, forward_axis="auto", **kw)


def transform(pts: np.ndarray, spec: str) -> np.ndarray:
    if not spec:
        return pts
    p = dict((kv.split("=")[0], float(kv.split("=")[1])) for kv in spec.split(","))
    r, pi, y = (np.radians(p.get(k, 0.0)) for k in ("roll", "pitch", "yaw"))
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(pi), 0, np.sin(pi)], [0, 1, 0], [-np.sin(pi), 0, np.cos(pi)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], 1) @ (Rz @ Ry @ Rx).T
    out = pts.copy()
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2] - p.get("dz", 0.0)
    return out


def feed(det, pts):
    # кэш: x-вперёд/y-влево → оси сообщения (вперёд = −Y), как в bench.core_adapter
    xyz = np.stack([pts["y"], -pts["x"], pts["z"]], 1).astype(np.float32)
    return det.process(xyz, pts["ring"], pts["i"], 0.0)


def run_empty(job):
    bag, dspec, tf = job
    b = Bag(bag)
    poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
    s_tr = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    det = make(dspec)
    rows = []
    for k in range(len(b)):
        res = feed(det, transform(b[k].pts, tf))
        rows.append([k, float(s_tr[k]), [[o.distance, o.lateral, o.height, o.points] for o in res.obstacles if o.confirmed]])
    return bag, dspec, tf, rows


OBJ = {
    "box_30x10": ((0.30, 0.30, 0.10), 0.0),
    "post_10x30": ((0.10, 0.10, 0.30), 0.0),
    "box_50": ((0.5, 0.5, 0.5), 0.0),
    "person": ((0.3, 0.5, 1.7), 0.0),
    "cable": ((0.02, 0.02, 1.0), 2.0),
    "cart": ((1.0, 1.2, 1.0), 0.0),
}


def synth_jobs():
    jobs = []
    for bag in EMPTY_BAGS:
        poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
        L = float(np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)))
        for pi, s_obj in enumerate(np.arange(150.0, L - 10, 150.0)):
            for t in OBJ:
                for dual in (True,):
                    jobs.append((bag, float(s_obj), t, 0.0 if pi % 2 == 0 else 0.6, dual))
    return jobs


def run_synth(job):
    try:
        return _run_synth(job)
    except Exception as e:  # один сломанный проезд не должен ронять весь прогон
        print(f"ОШИБКА {job}: {type(e).__name__}: {e}", flush=True)
        return []


def _run_synth(job):
    from bench.synth import Obj, Scenario, frames_with_objects

    bag, s_obj, t, lat, dual = job
    size, h = OBJ[t]
    poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
    s_tr = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    d_ahead = s_obj - s_tr
    ks = [k for k in range(len(s_tr)) if -2 < d_ahead[k] < 215]
    if not ks:
        return []
    ks = list(range(max(ks[0] - 20, 0), ks[-1] + 1))  # разогрев: подтверждение + модель полотна
    pairs = PAIRS if dual else [(d, "") for d in DETS]
    dets = {(d, tf): make(d) for d, tf in pairs}
    out = []
    for k, pts, lab, info in frames_with_objects(Scenario(bag, [Obj(t, size, s=s_obj, lat=lat, h=h)]), ks=ks, dual=dual):
        m = lab == 0
        n_obj = int(m.sum())
        obj_x = float(np.min(pts["x"][m])) if n_obj else None
        for (d, tf), det in dets.items():
            res = feed(det, transform(pts, tf))
            hit = obj_x is not None and any(o.confirmed and abs(o.distance - obj_x) < 2.5 for o in res.obstacles)
            if info:
                out.append([bag, s_obj, t, dual, d, tf, k, float(d_ahead[k]), n_obj, bool(hit)])
    return out


def report():
    lines = ["## Ложные тревоги (5 пустых bag)\n", "| детектор | искажение | кадров с тревогой | событий | событий/км |", "|---|---|---|---|---|"]
    E = json.loads((OUT / "empty.json").read_text()) if (OUT / "empty.json").exists() else []
    agg = {}
    for bag, d, tf, rows in E:
        a = agg.setdefault((d, tf), [0, 0, 0, 0.0])
        prev = False
        for k, s, dets in rows:
            al = bool(dets)
            a[0] += 1
            a[1] += al
            a[2] += al and not prev
            prev = al
        a[3] += (rows[-1][1] - rows[0][1]) / 1000
    for (d, tf), (n, al, ev, km) in agg.items():
        lines.append(f"| {d} | {tf or '—'} | {100 * al / n:.1f}% | {ev} | {ev / km:.1f} |")
    S = json.loads((OUT / "synth.json").read_text()) if (OUT / "synth.json").exists() else []
    bins = [0, 20, 40, 60, 80, 100, 125]
    for t in OBJ:
        lines += [f"\n### Pd при видимом объекте (≥3 точки; кабель ≥1): {t}\n", "| детектор | искажение | 2 отражения | " + " | ".join(f"{a}–{b}" for a, b in zip(bins[:-1], bins[1:])) + " |", "|---" * (len(bins) + 2) + "|"]
        keys = sorted({(r[4], r[5], r[3]) for r in S if r[2] == t}, key=lambda x: (x[0], x[1], not x[2]))
        for d, tf, dual in keys:
            rr = [r for r in S if r[2] == t and r[4] == d and r[5] == tf and r[3] == dual and r[8] >= (1 if t == "cable" else 3)]
            cells = []
            for a, b in zip(bins[:-1], bins[1:]):
                m = [r[9] for r in rr if a <= r[7] < b]
                cells.append(f"{np.mean(m):.2f}" if m else "—")
            lines.append(f"| {d} | {tf or '—'} | {'да' if dual else 'нет'} | " + " | ".join(cells) + " |")
    txt = "\n".join(lines)
    (OUT / "report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1]
    if mode == "empty":
        jobs = [(b, d, tf) for b in EMPTY_BAGS for d, tf in PAIRS]
        res = []
        with ProcessPoolExecutor(WORKERS) as ex:
            for i, r in enumerate(ex.map(run_empty, jobs)):
                res.append(r)
                print(f"empty {i + 1}/{len(jobs)}", flush=True)
        (OUT / "empty.json").write_text(json.dumps(res))
    elif mode == "synth":
        jobs = synth_jobs()
        res = []
        with ProcessPoolExecutor(WORKERS) as ex:
            for i, r in enumerate(ex.map(run_synth, jobs)):
                res += r
                print(f"synth {i + 1}/{len(jobs)}", flush=True)
                if (i + 1) % 10 == 0:
                    (OUT / "synth.partial.json").write_text(json.dumps(res))
        (OUT / "synth.json").write_text(json.dumps(res))
    report()
