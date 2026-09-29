"""Классификатор, дообученный на отрицательных примерах расширенной записи new_data.

Положительные и прочие отрицательные — cands_rand.json (случайные формы на 5 записях).
new_data режется на 4 непрерывных отрезка nd0..nd3; кандидаты каждого — отрицательные.
Честная проверка: модель для записи/отрезка X обучена без X; порог — максимум оценки
на пустых отрицательных обучающей части.

    python -m bench.verifier_nd collect <каталог new_data>   # → cands_nd.json
    python -m bench.verifier_nd models                        # → model_<X>_rand3.txt, model_full_rand3.txt
    python -m bench.verifier_nd run <каталог new_data>        # ложные на отрезках своей моделью
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ros2_ws/src/metro_obstacle"))
from metro_obstacle.cloud import cloud_to_arrays

from bench.newdata import files, messages
from bench.round4 import make
from bench.verifier import SPEC, feats
from bench.verifier_e2e import EMPTY, GEOM, OUT, derive

warnings.filterwarnings("ignore")
NSEG = 4
TAG = "rand3"


def segments(root: Path) -> list[list[str]]:
    return [list(a) for a in np.array_split([str(p) for p in files(root)], NSEG)]


def _frames(paths):
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    for p in paths:
        for t, _, raw in messages(Path(p)):
            yield t, cloud_to_arrays(ts.deserialize_cdr(raw, "sensor_msgs/msg/PointCloud2"))


def collect_seg(job):
    i, paths = job
    det = make(SPEC)
    rows = []
    for k, (t, (xyz, ring, it)) in enumerate(_frames(paths)):
        res = det.process(xyz, ring, it, t * 1e-9)
        for o in res.obstacles:
            if o.distance >= 20:
                rows.append({"bag": f"nd{i}", "k": k, "y": 0, "src": "empty", "f": feats(o)})
    return rows


def collect():
    segs = segments(Path(sys.argv[2]))
    rows = []
    with ProcessPoolExecutor(NSEG) as ex:
        for r in ex.map(collect_seg, list(enumerate(segs))):
            rows += r
    (OUT / "cands_nd.json").write_text(json.dumps(rows))
    print("кандидатов new_data", len(rows))


def _xy(R):
    F7 = np.array([[r["f"][0], r["f"][1], r["f"][3], r["f"][4], r["f"][5], r["f"][6], r["f"][7]] for r in R])
    return derive(F7), np.array([r["y"] for r in R])


def models():
    import lightgbm as lgb

    R = json.loads((OUT / "cands_rand.json").read_text()) + json.loads((OUT / "cands_nd.json").read_text())
    X, y = _xy(R)
    bag = np.array([r["bag"] for r in R])
    src = np.array([r["src"] for r in R])
    print(f"всего {len(R)}, положительных {y.sum()}, пустых отрицательных {np.sum(src == 'empty')}")
    meta = {}
    for hold in [*EMPTY, *(f"nd{i}" for i in range(NSEG)), "full"]:
        tr = bag != hold
        m = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
        ).fit(X[tr], y[tr])
        neg = tr & (src == "empty")
        sc = m.predict_proba(X[neg])[:, 1]
        thr = float(np.max(sc)) + 1e-6
        m.booster_.save_model(str(OUT / f"model_{hold}_{TAG}.txt"))
        meta[f"{hold}|{TAG}"] = thr
        print(f"{hold:38s} порог {thr:.3f} (p99.9 {np.percentile(sc, 99.9):.3f})", flush=True)
    tp = OUT / "thresholds.json"
    old = json.loads(tp.read_text()) if tp.exists() else {}
    tp.write_text(json.dumps({**old, **meta}, indent=1))


def run_seg(job):
    i, paths, spec, tag = job
    det = make(spec)
    if tag:
        import lightgbm as lgb

        thr = json.loads((OUT / "thresholds.json").read_text())[f"nd{i}|{tag}"]
        booster = lgb.Booster(model_file=str(OUT / f"model_nd{i}_{tag}.txt"))
        det.verifier = lambda F: booster.predict(derive(F))
        det.params = type(det.params)(**{**det.params.__dict__, "verify_thr": thr * float(os.environ.get("LCT_VTHR_MUL", "1"))})
    fr = al = ev = 0
    prev = False
    events = []
    for t, (xyz, ring, it) in _frames(paths):
        res = det.process(xyz, ring, it, t * 1e-9)
        fr += 1
        al += res.obstacle
        if res.obstacle and not prev:
            ev += 1
            o = next((o for o in res.obstacles if o.confirmed), None)
            events.append([t, o and [round(o.distance, 1), round(o.lateral, 2), round(o.height, 2), o.points]])
        prev = res.obstacle
    return i, tag, fr, al, ev, events


def run():
    segs = segments(Path(sys.argv[2]))
    jobs = [(i, s, GEOM, TAG) for i, s in enumerate(segs)]
    tot = [0, 0, 0]
    with ProcessPoolExecutor(NSEG) as ex:
        for i, tag, fr, al, ev, events in ex.map(run_seg, jobs):
            tot = [tot[0] + fr, tot[1] + al, tot[2] + ev]
            print(f"nd{i}: кадров {fr}, тревога {al}, событий {ev}: {events[:10]}", flush=True)
    print(f"ИТОГО {TAG} на new_data (модель без своего отрезка): {100 * tot[1] / tot[0]:.2f}% кадров, {tot[2]} событий")


if __name__ == "__main__":
    {"collect": collect, "models": models, "run": run}[sys.argv[1]]()
