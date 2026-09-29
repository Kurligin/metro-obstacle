"""Сквозная проверка проверяющего классификатора (честная: ничего с проверяемой записи).

Для каждой записи X — модель, обученная на кандидатах остальных записей (cands.json из
bench.verifier collect); порог — максимум оценки на ложных кандидатах ОБУЧАЮЩИХ пустых
записей. Детектор: геометрия со сниженными порогами (как при сборе) + классификатор +
подтверждение 3 из 5. Проверки: пустые записи (ложные), синтетика (Pd), отложенная
форма (модель без человека → человек), наивная вставка (фон за объектом не удаляется).

    python -m bench.verifier_e2e models
    python -m bench.verifier_e2e run
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
from metro_obstacle_core.verifier import derive  # признаки — те же, что в ядре

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "out" / "verifier"
EMPTY = [b for b in BAGS if b != "doubleT_obstacle"]
GEOM = os.environ.get("LCT_GEOM", "soft|edge_reject=1,end_guard=0.0,min_pts=2,min_rings=1,margin_k=0.001,confirm=3/5")
VTAG = os.environ.get("LCT_VTAG", "rand")
BASE = "geometry"  # прежний основной режим ядра (только геометрия)


def models(src_file: str = "cands.json", tags=("all", "noperson")):
    import lightgbm as lgb

    R = json.loads((OUT / src_file).read_text())
    # в cands признаки: distance, lateral, abs_lat, height, length, width, z_extent, points, pts_norm, conf
    F7 = np.array([[r["f"][0], r["f"][1], r["f"][3], r["f"][4], r["f"][5], r["f"][6], r["f"][7]] for r in R])
    X = derive(F7)
    y = np.array([r["y"] for r in R])
    bag = np.array([r["bag"] for r in R])
    src = np.array([r["src"] for r in R])
    meta = {}
    for hold in EMPTY:
        opts = {"all": np.ones(len(R), bool), "noperson": src != "person", VTAG: np.ones(len(R), bool)}
        for tag in tags:
            keep = opts[tag]
            tr = (bag != hold) & keep
            m = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
            ).fit(X[tr], y[tr])
            neg = tr & (src == "empty")
            thr = float(np.max(m.predict_proba(X[neg])[:, 1])) + 1e-6
            path = OUT / f"model_{hold}_{tag}.txt"
            m.booster_.save_model(str(path))
            meta[f"{hold}|{tag}"] = thr
            print(f"{hold:38s} {tag:9s} порог {thr:.3f}", flush=True)
    tp = OUT / "thresholds.json"
    old = json.loads(tp.read_text()) if tp.exists() else {}
    tp.write_text(json.dumps({**old, **meta}, indent=1))


def make_det(spec: str, bag: str | None, tag: str | None):
    det = make(spec)
    if bag is not None:
        import lightgbm as lgb

        thr = json.loads((OUT / "thresholds.json").read_text())[f"{bag}|{tag}"]
        booster = lgb.Booster(model_file=str(OUT / f"model_{bag}_{tag}.txt"))
        det.verifier = lambda F: booster.predict(derive(F))
        det.params = type(det.params)(**{**det.params.__dict__, "verify_thr": thr * float(os.environ.get("LCT_VTHR_MUL", "1"))})
    return det


def run_empty(job):
    bag, name, spec, tag = job
    b = Bag(bag)
    poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
    s_tr = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    det = make_det(spec, bag if tag else None, tag)
    rows = []
    for k in range(len(b)):
        res = feed(det, b[k].pts)
        rows.append([k, float(s_tr[k]), bool(res.obstacle)])
    return bag, name, rows


def run_synth(job):
    from bench.synth import Obj, Scenario, frames_with_objects

    bag, s_obj, t, lat, naive, variants = job
    size, hgt = OBJ[t]
    poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
    s_tr = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    d_ahead = s_obj - s_tr
    ks = [k for k in range(len(s_tr)) if -2 < d_ahead[k] < 215]
    if not ks:
        return []
    ks = list(range(max(ks[0] - 20, 0), ks[-1] + 1))
    dets = {name: make_det(spec, bag if tag else None, tag) for name, spec, tag in variants}
    out = []
    sc = Scenario(bag, [Obj(t, size, s=s_obj, lat=lat, h=hgt)])
    for k, pts, lab, info in frames_with_objects(sc, ks=ks, naive=naive):
        m = lab == 0
        n_obj = int(m.sum())
        ox = float(np.min(pts["x"][m])) if n_obj else None
        for name, det in dets.items():
            res = feed(det, pts)
            hit = ox is not None and any(o.confirmed and abs(o.distance - ox) < 2.5 for o in res.obstacles)
            if info:
                out.append([bag, s_obj, t, lat, naive, name, k, float(d_ahead[k]), n_obj, bool(hit)])
    return out


def run():
    rand = len(sys.argv) > 2 and sys.argv[2] == "rand"
    core = len(sys.argv) > 2 and sys.argv[2] == "core"
    if core:  # режимы ядра (классификатор — из пакета, полная модель): сравнение вариантов между собой
        V_ALL = [
            ("geometry", "geometry", None),
            ("default", "default", None),
            ("default(plane0)", "default|plane_every=0", None),
        ]
        V_PERSON = V_ALL
    elif rand:
        V_ALL = [("base", BASE, None), ("default+top", BASE + "|top_k=0.004,top_s0=20.0", None), ("low+ver(случайные формы)", GEOM, VTAG)]
        V_PERSON = V_ALL
    else:
        V_ALL = [("base", BASE, None), ("low", GEOM, None), ("low+ver", GEOM, "all")]
        V_PERSON = V_ALL + [("low+ver(без человека)", GEOM, "noperson")]
    ejobs = [(b, n, s, t) for b in EMPTY for n, s, t in V_ALL]
    sjobs = []
    for bag in EMPTY:
        poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
        L = float(np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)))
        for pi, s_obj in enumerate(np.arange(150.0, L - 10, 150.0)):
            lat = 0.0 if pi % 2 == 0 else 0.6
            for t in OBJ:
                sjobs.append((bag, float(s_obj), t, lat, False, V_PERSON if t == "person" else V_ALL))
                if t in ("person", "box_50", "post_10x30"):
                    sjobs.append((bag, float(s_obj), t, lat, True, V_ALL))
    E, S = [], []
    with ProcessPoolExecutor(WORKERS) as ex:
        for r in ex.map(run_empty, ejobs):
            E.append(r)
        print("empty done", flush=True)
        for i, r in enumerate(ex.map(run_synth, sjobs)):
            S += r
            if (i + 1) % 10 == 0:
                print(f"synth {i + 1}/{len(sjobs)}", flush=True)
                (OUT / "e2e_synth.partial.json").write_text(json.dumps(S))
    suf = "_core" if core else (f"_{VTAG}" if rand else "")
    (OUT / f"e2e_empty{suf}.json").write_text(json.dumps(E))
    (OUT / f"e2e_synth{suf}.json").write_text(json.dumps(S))
    report(E, S)


def report(E=None, S=None):
    suf = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
    E = E or json.loads((OUT / f"e2e_empty{suf}.json").read_text())
    S = S or json.loads((OUT / f"e2e_synth{suf}.json").read_text())
    agg = {}
    for bag, name, rows in E:
        a = agg.setdefault(name, [0, 0, 0, 0.0])
        prev = False
        for k, s, al in rows:
            a[0] += 1
            a[1] += al
            a[2] += al and not prev
            prev = al
        a[3] += (rows[-1][1] - rows[0][1]) / 1000
    print("\nЛожные тревоги (5 пустых записей, модель без проверяемой записи):")
    for n, (fr, al, ev, km) in agg.items():
        print(f"  {n:24s} {100 * al / fr:5.1f}% кадров, {ev / km:5.1f} событий/км")

    def stable(rows):
        runs = {}
        for r in rows:
            runs.setdefault((r[0], r[1]), []).append(r)
        fd = []
        for seq in runs.values():
            seq.sort(key=lambda r: r[6])
            run, f = 0, 0.0
            for r in seq:
                run = run + 1 if r[9] else 0
                if run >= 3:
                    f = r[7]
                    break
            fd.append(f)
        return float(np.median(fd)) if fd else float("nan")

    order = ["geometry", "default", "default(plane0)", "base", "default+top", "low", "low+ver", "low+ver(без человека)", "low+ver(случайные формы)"]
    names = sorted({r[5] for r in S}, key=order.index)
    for naive in (False, True):
        print(f"\nУстойчивое обнаружение (3 подряд), медиана, м — {'наивная вставка' if naive else 'трассировка лучей'}:")
        types = [t for t in OBJ if any(r[2] == t and r[4] == naive for r in S)]
        print("  " + " " * 24 + " ".join(f"{t[:8]:>9s}" for t in types))
        for n in names:
            cells = []
            for t in types:
                rr = [r for r in S if r[2] == t and r[4] == naive and r[5] == n]
                cells.append(f"{stable(rr):9.0f}" if rr else f"{'—':>9s}")
            print(f"  {n:24s}" + " ".join(cells))
    bins = [(40, 60), (60, 80), (80, 100), (100, 125)]
    print("\nPd при видимом (≥3 точек) — человек / коробка 0.5, трассировка:")
    for n in names:
        row = []
        for t in ("person", "box_50"):
            rr = [r for r in S if r[2] == t and not r[4] and r[5] == n and r[8] >= 3]
            row.append(" ".join(f"{np.mean([r[9] for r in rr if a <= r[7] < b]):.2f}" if rr else "—" for a, b in bins))
        print(f"  {n:24s} {row[0]} | {row[1]}")


if __name__ == "__main__":
    {
        "models": models,
        "models_rand": lambda: models(f"cands_{VTAG}.json", (VTAG,)),
        "run": run,
        "report": report,
    }[sys.argv[1]]()
