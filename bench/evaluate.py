"""Прогон кандидатов: ложные тревоги на пустых проездах и Pd на синтетике.

    python -m bench.evaluate empty      # все кадры 5 пустых bag + bag с препятствием
    python -m bench.evaluate synth      # синтетические объекты
    python -m bench.evaluate report     # сводные таблицы

Ядро решения (metro_obstacle_core) — конфиги «core:default», «core:strict», «core:soft»:
    python -m bench.evaluate empty "core:default,core:strict,core:soft"
    python -m bench.evaluate synth "core:default,core:strict,core:soft"   # → synth__core.json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench.data import WORKERS, BAGS, CACHE_ROOT, Bag
from bench.core_adapter import CORE_CONFIGS, is_core, make_detector
from bench.detect import CONFIGS, Config

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "out" / "eval"
EMPTY_BAGS = [b for b in BAGS if b != "doubleT_obstacle"]
CFG = {c.name: c for c in CONFIGS + CORE_CONFIGS}


def _oracle(bag: str, k: int, xyz, poses):
    from bench.synth import LIDAR_H
    from bench.truth import truth_axis

    return truth_axis(xyz, poses, k, LIDAR_H.get(bag, 1.35))


def run_empty(job):
    bag, cname = job
    cfg = CFG[cname]
    b = Bag(bag)
    poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
    s_frames = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    det = make_detector(cfg)
    rows = []
    for k in range(len(b)):
        fr = b[k]
        orc = _oracle(bag, k, fr.xyz, poses) if cfg.axis == "oracle" else None
        t0 = time.perf_counter()
        dets, info = det.process(fr.pts, orc)
        ms = (time.perf_counter() - t0) * 1e3
        rows.append(
            {
                "k": k,
                "s_train": float(s_frames[k]),
                "ms": ms,
                "axis_len": info["axis_len"],
                "dets": [[d.s, d.lat, d.h_top, d.n] for d in dets],
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"empty__{bag}__{cname}.json").write_text(json.dumps(rows))
    return bag, cname


# ------------------------------------------------------------------ синтетика

OBJ_TYPES = {
    "box_30x10": ((0.30, 0.30, 0.10), 0.0),
    "post_10x30": ((0.10, 0.10, 0.30), 0.0),
    "box_50": ((0.5, 0.5, 0.5), 0.0),
    "person": ((0.3, 0.5, 1.7), 0.0),
    "cable": ((0.02, 0.02, 1.0), 2.0),  # свисает от верха габарита до 1 м над рельсом
    "cart": ((1.0, 1.2, 1.0), 0.0),
}


SYNTH_CFGS = ["Z c3", "Z c2 m0.002", "Z+S c3", "E5", "E5+S"]


def synth_jobs(cfgs: tuple[str, ...] | None = None):
    jobs = []
    for bag in EMPTY_BAGS:
        poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
        L = float(np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)))
        for pi, s_obj in enumerate(np.arange(150.0, L - 10, 150.0)):
            for t in OBJ_TYPES:
                lat = 0.0 if pi % 2 == 0 else 0.6
                jobs.append((bag, float(s_obj), t, lat) + ((cfgs,) if cfgs else ()))
    return jobs


def run_synth(job):
    from bench.synth import Obj, Scenario, frames_with_objects

    bag, s_obj, t, lat = job[:4]
    cfgs = job[4] if len(job) > 4 else SYNTH_CFGS
    size, h = OBJ_TYPES[t]
    sc = Scenario(bag, [Obj(t, size, s=s_obj, lat=lat, h=h)])
    poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
    s_frames = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    d_ahead = s_obj - s_frames
    ks = [k for k in range(len(s_frames)) if -2 < d_ahead[k] < 215]
    if not ks:
        return []
    ks = list(range(max(ks[0] - 8, 0), ks[-1] + 1))  # разогрев для накопления
    dets = {n: make_detector(CFG[n]) for n in cfgs}
    b = Bag(bag)
    res = []
    for k, pts, lab, info in frames_with_objects(sc, ks=ks):
        orc = _oracle(bag, k, b[k].xyz, poses)
        obj_x = None
        n_obj = 0
        if info:
            o, dist, n_obj = info[0]
            m = lab == 0
            if m.any():
                obj_x = float(np.min(pts["x"][m]))
        for cname, dt in dets.items():
            ds, inf = dt.process(pts, orc if CFG[cname].axis == "oracle" else None)
            hit = obj_x is not None and any(abs(d.s - obj_x) < 2.5 for d in ds)
            other = [d.s for d in ds if obj_x is None or abs(d.s - obj_x) >= 2.5]
            res.append(
                {
                    "bag": bag, "s_obj": s_obj, "type": t, "lat": lat, "cfg": cname, "k": k,
                    "dist": float(d_ahead[k]), "inserted": bool(info), "n_obj": n_obj,
                    "hit": bool(hit), "other": other, "axis_len": inf["axis_len"],
                }
            )
    return res


# ------------------------------------------------------------------ отчёт


def report():
    lines = []
    # ложные тревоги
    lines.append("## Пустые проезды: ложные тревоги\n")
    lines.append("| конфиг | кадров | кадров с тревогой, % | событий | событий/км | тревог ≤100 м, % кадров | мс/кадр p50/p95 | ось, м p50 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    total_km = 0.0
    per_cfg = {}
    for c in CONFIGS + CORE_CONFIGS:
        fr = al = ev = al100 = 0
        ms, ax, km = [], [], 0.0
        for bag in EMPTY_BAGS:
            p = OUT / f"empty__{bag}__{c.name}.json"
            if not p.exists():
                continue
            rows = json.loads(p.read_text())
            km += (rows[-1]["s_train"] - rows[0]["s_train"]) / 1000
            prev = False
            for r in rows:
                fr += 1
                a = len(r["dets"]) > 0
                al += a
                al100 += any(d[0] < 100 for d in r["dets"])
                ev += a and not prev
                prev = a
                ms.append(r["ms"])
                ax.append(r["axis_len"])
        if not fr:
            continue
        per_cfg[c.name] = km
        lines.append(
            f"| {c.name} | {fr} | {100 * al / fr:.1f} | {ev} | {ev / max(km, 1e-9):.1f} | {100 * al100 / fr:.1f} | "
            f"{np.percentile(ms, 50):.0f}/{np.percentile(ms, 95):.0f} | {np.median(ax):.0f} |"
        )
        total_km = km
    lines.append(f"\nПройдено по пустым bag: {total_km:.2f} км.\n")
    # синтетика
    # synth.json — прогон прототипов, synth__*.json — отдельные прогоны (ядро); synth_roundN — архив
    synth_files = [p for p in [OUT / "synth.json", *sorted(OUT.glob("synth__*.json"))] if p.exists()]
    if synth_files:
        R = [r for f in synth_files for r in json.loads(f.read_text())]
        synth_cfgs = [c for c in CONFIGS + CORE_CONFIGS if any(r["cfg"] == c.name for r in R)]
        bins = [0, 20, 40, 60, 80, 100, 125, 150, 200]
        types = list(OBJ_TYPES)
        for t in types:
            lines.append(f"\n### Pd по дальности: {t}\n")
            lines.append("| конфиг | " + " | ".join(f"{a}–{b}" for a, b in zip(bins[:-1], bins[1:])) + " |")
            lines.append("|---" * (len(bins)) + "|")
            for c in synth_cfgs:
                rr = [r for r in R if r["cfg"] == c.name and r["type"] == t and r["inserted"]]
                cells = []
                for a, b in zip(bins[:-1], bins[1:]):
                    m = [r["hit"] for r in rr if a <= r["dist"] < b]
                    cells.append(f"{np.mean(m):.2f}" if m else "—")
                lines.append(f"| {c.name} | " + " | ".join(cells) + " |")
        # дальность первого устойчивого обнаружения (3 кадра подряд)
        lines.append("\n### Дальность устойчивого обнаружения (3 кадра подряд), медиана по прогонам, м\n")
        lines.append("| конфиг | " + " | ".join(types) + " |")
        lines.append("|---" * (len(types) + 1) + "|")
        for c in synth_cfgs:
            cells = []
            for t in types:
                rs = {}
                for r in R:
                    if r["cfg"] == c.name and r["type"] == t:
                        rs.setdefault((r["bag"], r["s_obj"]), []).append(r)
                firsts = []
                for key, seq in rs.items():
                    seq.sort(key=lambda r: r["k"])
                    run, fd = 0, 0.0
                    for r in seq:
                        run = run + 1 if r["hit"] else 0
                        if run >= 3:
                            fd = r["dist"]
                            break
                    firsts.append(fd)
                cells.append(f"{np.median(firsts):.0f}" if firsts else "—")
            lines.append(f"| {c.name} | " + " | ".join(cells) + " |")
    txt = "\n".join(lines)
    (OUT / "report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "empty":
        only = sys.argv[2].split(",") if len(sys.argv) > 2 else [c.name for c in CONFIGS]
        jobs = [(b, c) for b in BAGS for c in only]
        with ProcessPoolExecutor(WORKERS) as ex:
            for r in ex.map(run_empty, jobs):
                print("done", r, flush=True)
    elif mode == "synth":
        cfgs = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
        jobs = synth_jobs(cfgs)
        print(len(jobs), "прогонов", flush=True)
        allr = []
        with ProcessPoolExecutor(WORKERS) as ex:
            for i, r in enumerate(ex.map(run_synth, jobs)):
                allr += r
                if i % 10 == 0:
                    print(i, flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        # прогон ядра — в отдельный файл, чтобы не затирать прогоны прототипов
        name = "synth.json" if cfgs is None else f"synth__{'core' if all(map(is_core, cfgs)) else 'custom'}.json"
        (OUT / name).write_text(json.dumps(allr))
    report()
