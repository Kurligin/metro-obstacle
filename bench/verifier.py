"""Проверяющий классификатор кластеров (эксперимент): отличает объект от фона.

1) collect — ядро в «мягком» режиме без подтверждения отдаёт всех кандидатов:
   на пустых bag это настоящие ложные (метка 0), на синтетике — кандидат у объекта
   (метка 1) и прочие (метка 0). Признаки — только из выхода ядра (Obstacle).
2) train — LightGBM с отложенной записью: порог = максимум оценки на отрицательных
   примерах отложенной записи (ложных 0), полнота объектов по дальности при этом пороге.

    python -m bench.verifier collect   # → out/verifier/cands.json
    python -m bench.verifier train
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
from bench.round4 import OBJ, feed, make

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "out" / "verifier"
EMPTY = [b for b in BAGS if b != "doubleT_obstacle"]
SPEC = os.environ.get("LCT_VSPEC", "soft|edge_reject=1,end_guard=0.0,confirm=1/1,min_pts=2,min_rings=1,margin_k=0.001")
VTAG = os.environ.get("LCT_VTAG", "rand")  # суффикс файла кандидатов: cands_<VTAG>.json
VLAT = float(os.environ.get("LCT_VLAT", "0.7"))  # боковой разброс случайных форм, ±м
VHANG = float(os.environ.get("LCT_VHANG", "0"))  # доля тонких стержней, свисающих с потолка
FEATS = ["distance", "lateral", "abs_lat", "height", "length", "width", "z_extent", "points", "pts_norm", "conf"]


def feats(o) -> list[float]:
    d = max(o.distance, 1.0)
    return [
        o.distance,
        o.lateral,
        abs(o.lateral),
        o.height,
        o.length,
        o.width,
        o.z_extent,
        float(o.points),
        o.points * (d / 50.0) ** 2,  # точки, приведённые к 50 м (плотность ~ 1/r²)
        o.confidence,
    ]


def collect_empty(bag: str):
    b = Bag(bag)
    det = make(SPEC)
    rows = []
    for k in range(len(b)):
        res = feed(det, b[k].pts)
        for o in res.obstacles:
            if o.distance >= 20:
                rows.append({"bag": bag, "k": k, "y": 0, "src": "empty", "f": feats(o)})
    return rows


def random_obj(seed: int):
    """Случайная форма: компактный бокс 0.1–1.2 × 0.1–1.2 × 0.1–1.8 м на полотне или
    тонкий вертикальный элемент (кабель/стойка) на случайной высоте; поворот и смещение."""
    r = np.random.default_rng(seed)
    u = r.random()
    if u < VHANG:  # стержень от потолка (~4.3 м) вниз до 2.3–2.9 м над головками
        d = r.uniform(0.02, 0.08)
        h = r.uniform(2.3, 2.9)
        size = (d, d, 4.3 - h)
        kind = "rand_hang"
    elif u < VHANG + 0.25:
        d = r.uniform(0.015, 0.08)
        size = (d, d, r.uniform(0.3, 1.5))
        h = r.uniform(0.0, 3.0 - size[2])
        kind = "rand_thin"
    else:
        size = (r.uniform(0.1, 1.2), r.uniform(0.1, 1.2), r.uniform(0.1, 1.8))
        h = 0.0
        kind = "rand_box"
    return kind, size, h, float(r.uniform(-VLAT, VLAT)), float(r.uniform(0, np.pi))


def collect_synth(job):
    from bench.synth import Obj, Scenario, frames_with_objects

    bag, s_obj, t, lat = job[:4]
    yaw = 0.0
    if t.startswith("rand:"):
        t, size, hgt, lat, yaw = random_obj(int(t.split(":")[1]))
    else:
        size, hgt = OBJ[t]
    poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
    s_tr = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    d_ahead = s_obj - s_tr
    ks = [k for k in range(len(s_tr)) if -2 < d_ahead[k] < 215]
    if not ks:
        return []
    ks = list(range(max(ks[0] - 20, 0), ks[-1] + 1))
    det = make(SPEC)
    rows = []
    for k, pts, lab, info in frames_with_objects(Scenario(bag, [Obj(t, size, s=s_obj, lat=lat, h=hgt, yaw=yaw)]), ks=ks):
        res = feed(det, pts)
        m = lab == 0
        ox = float(np.min(pts["x"][m])) if m.any() else None
        for o in res.obstacles:
            if o.distance < 20:
                continue
            pos = ox is not None and abs(o.distance - ox) < 2.5
            rows.append({"bag": bag, "k": k, "y": int(pos), "src": t if pos else "synth_bg", "d_obj": float(d_ahead[k]), "f": feats(o)})
    return rows


def collect(rand: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    seed = 0
    for bag in EMPTY:
        poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
        L = float(np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)))
        for pi, s_obj in enumerate(np.arange(150.0, L - 10, 150.0)):
            if rand:
                for _ in range(10):  # 10 случайных форм на позицию
                    jobs.append((bag, float(s_obj), f"rand:{seed}", 0.0))
                    seed += 1
            else:
                for t in OBJ:
                    jobs.append((bag, float(s_obj), t, 0.0 if pi % 2 == 0 else 0.6))
    rows = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for r in ex.map(collect_empty, EMPTY):
            rows += r
        print("empty done", len(rows), flush=True)
        for i, r in enumerate(ex.map(collect_synth, jobs)):
            rows += r
            if (i + 1) % 10 == 0:
                print(f"synth {i + 1}/{len(jobs)}", flush=True)
    (OUT / (f"cands_{VTAG}.json" if rand else "cands.json")).write_text(json.dumps(rows))
    print("кандидатов", len(rows), "положительных", sum(r["y"] for r in rows))


def train():
    import lightgbm as lgb

    R = json.loads((OUT / "cands.json").read_text())
    X = np.array([r["f"] for r in R])
    y = np.array([r["y"] for r in R])
    bag = np.array([r["bag"] for r in R])
    src = np.array([r["src"] for r in R])
    d_obj = np.array([r.get("d_obj", np.nan) for r in R])
    dist = X[:, 0]
    print(f"всего {len(R)}: объектов {y.sum()}, фон пустых {np.sum(src == 'empty')}, фон синтетики {np.sum(src == 'synth_bg')}")
    bins = [(20, 60), (60, 80), (80, 100), (100, 130)]
    res = {b: [0, 0, 0, 0] for b in bins}  # объектов прошло / всего; ложных прошло / всего (по пустым)
    for hold in EMPTY:
        tr, te = bag != hold, bag == hold
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, verbose=-1)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        neg_empty = te & (src == "empty")
        thr = float(np.max(m.predict_proba(X[neg_empty])[:, 1])) + 1e-6 if neg_empty.any() else 0.5
        for lo, hi in bins:
            pos = (y[te] == 1) & (dist[te] >= lo) & (dist[te] < hi)
            neg = (src[te] == "empty") & (dist[te] >= lo) & (dist[te] < hi)
            res[(lo, hi)][0] += int((p[pos] >= thr).sum())
            res[(lo, hi)][1] += int(pos.sum())
            res[(lo, hi)][2] += int((p[neg] >= thr).sum())
            res[(lo, hi)][3] += int(neg.sum())
        print(f"  отложена {hold}: порог {thr:.3f}", flush=True)
    print("\nпри пороге «0 ложных на отложенной записи»:")
    for (lo, hi), (a, b, c, d) in res.items():
        print(f"  {lo}-{hi} м: объекты проходят {a}/{b} = {a / max(b, 1):.0%}; ложные кандидаты пустых {c}/{d}")
    imp = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1).fit(X, y).feature_importances_
    print("важность признаков:", dict(zip(FEATS, imp.tolist())))


if __name__ == "__main__":
    {"collect": collect, "collect_rand": lambda: collect(rand=True), "train": train}[sys.argv[1]]()
