"""Прогон на синтетике заказчика (cloud_with_fake_obj): облако без ring и timestamp.

Вход читается как в ноде (cloud_to_arrays: кольца по углу места). Для каждого варианта
детектора пишем по кадрам: пройденный путь (по оценке скорости ядра) и все объекты.
Мировая координата объекта вдоль пути = путь + distance — по ней детекции группируются
в места и сопоставляются с 10 объектами заказчика.

    python -m bench.customer train      # полная модель классификатора (все пустые записи)
    python -m bench.customer run [bag_dir]
    python -m bench.customer report
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
from metro_obstacle.cloud import cloud_to_arrays  # noqa: E402

from bench.round4 import make  # noqa: E402
from bench.verifier_e2e import GEOM, OUT, derive  # noqa: E402

warnings.filterwarnings("ignore")
BAG = Path(os.environ.get("LCT_CUSTOMER", Path.home() / "Downloads/archive/customer/cloud_with_fake_obj"))
GEOM2 = os.environ.get(
    "LCT_GEOM2", "soft|end_guard=0.0,min_pts=2,min_rings=1,margin_k=0.001,margin0=0.0,top_k=0.004,top_s0=20.0,confirm=3/5"
)
# имя → (спецификация детектора, тег модели классификатора или None)
VARIANTS = {
    "geometry": ("geometry", None),  # прежний основной режим ядра (раньше — «default»)
    "geometry+top": ("geometry|top_k=0.004,top_s0=20.0", None),
    "strict": ("strict", None),
    "low+ver": (GEOM, "rand"),
    "low+ver2": (GEOM2, "rand2"),
    "low+ver3": (GEOM, "rand3"),  # + отрицательные примеры new_data
    "all_cands": ("soft|min_pts=2,min_rings=1,margin_k=0.001,confirm=1/1,end_guard=0.0", None),
    # новый режим ядра по умолчанию (классификатор из пакета) — должен совпасть с low+ver
    "core_default": ("default", None),
    # классификатор только дальше 20 м (модель обучена на кандидатах ≥ 20 м)
    "core_default_s20": ("default|verify_smin=20.0", None),
    # плоскость полотна только по первому кадру (как до уточнения на ходу) — для сравнения
    "core_default_plane0": ("default|plane_every=0", None),
    # плоскость уточняется по первым plane_buf оценкам и замораживается
    "core_default_freeze": ("default|plane_freeze=1", None),
    # гистерезис: плоскость первого кадра меняется, только если медиана оценок ушла > 1° / 5 см
    "core_default_tol": ("default|plane_tol_deg=1.0", None),
}


def train(tag: str = "rand"):
    import lightgbm as lgb

    R = json.loads((OUT / f"cands_{tag}.json").read_text())
    F7 = np.array([[r["f"][0], r["f"][1], r["f"][3], r["f"][4], r["f"][5], r["f"][6], r["f"][7]] for r in R])
    X, y = derive(F7), np.array([r["y"] for r in R])
    src = np.array([r["src"] for r in R])
    m = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
    ).fit(X, y)
    thr = float(np.max(m.predict_proba(X[src == "empty"])[:, 1])) + 1e-6
    m.booster_.save_model(str(OUT / f"model_full_{tag}.txt"))
    tp = OUT / "thresholds.json"
    old = json.loads(tp.read_text()) if tp.exists() else {}
    tp.write_text(json.dumps({**old, f"full|{tag}": thr}, indent=1))
    print("полная модель", tag, "порог", thr)


def make_variant(name: str):
    spec, tag = VARIANTS[name]
    det = make(spec)
    if tag:
        import lightgbm as lgb

        thr = json.loads((OUT / "thresholds.json").read_text())[f"full|{tag}"]
        booster = lgb.Booster(model_file=str(OUT / f"model_full_{tag}.txt"))
        det.verifier = lambda F: booster.predict(derive(F))
        det.params = type(det.params)(**{**det.params.__dict__, "verify_thr": thr * float(os.environ.get("LCT_VTHR_MUL", "1"))})
    return det


def run_one(args):
    name, bag_dir = args
    from rosbags.rosbag2 import Reader
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
    with Reader(bag_dir) as r:
        conns = [c for c in r.connections if c.msgtype.endswith("PointCloud2")]
        for k, (conn, t, raw) in enumerate(r.messages(connections=conns)):
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            xyz, ring, it = cloud_to_arrays(msg)
            res = det.process(xyz, ring, it, t * 1e-9)
            obs = [
                [o.distance, o.lateral, o.height, o.length, o.width, o.z_extent, o.points, bool(o.confirmed)]
                for o in res.obstacles
            ]
            rows.append([k, travel[0], bool(res.obstacle), res.distance if res.obstacle else None, res.processing_ms, obs])
            if k % 200 == 0:
                print(name, k, flush=True)
    return name, rows


def run():
    bag_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else BAG
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(VARIANTS)
    with ProcessPoolExecutor(len(names)) as ex:
        res = dict(ex.map(run_one, [(n, bag_dir) for n in names]))
    (OUT / "customer.json").write_text(json.dumps(res))
    report(res)


def report(res=None):
    res = res or json.loads((OUT / "customer.json").read_text())
    # путь — по оценке скорости варианта geometry (в старых файлах он назывался default)
    base_name = "geometry" if "geometry" in res else "default"
    base = res[base_name]
    print(f"кадров {len(base)}, путь по оценке ядра {base[-1][1]:.0f} м")
    for name, rows in res.items():
        al = sum(r[2] for r in rows)
        ms = np.percentile([r[4] for r in rows], 95)
        print(f"\n== {name}: тревога в {al}/{len(rows)} кадрах, p95 {ms:.0f} мс")
        # места: мировая координата подтверждённых объектов (у all_cands — всех)
        pts = []
        for k, _s, alarm, d, _, obs in rows:
            s = base[k][1]  # путь — по оценке скорости варианта geometry (у 1/1 её нет)
            for o in obs:
                if o[7] or name == "all_cands":
                    pts.append((s + o[0], k, o[0], o[1], o[2], o[4], o[6]))
        pts.sort()
        groups, cur = [], []
        for p in pts:
            if cur and p[0] - cur[-1][0] > 8:
                groups.append(cur)
                cur = []
            cur.append(p)
        if cur:
            groups.append(cur)
        for g in groups:
            if name == "all_cands" and len(g) < 5:
                continue
            a = np.array(g)
            print(
                f"  место {np.median(a[:, 0]):7.1f} м: кадров {len(set(a[:, 1])):3d} (k {int(a[:, 1].min())}–{int(a[:, 1].max())}), "
                f"первая дист {a[np.argmin(a[:, 1]), 2]:5.1f}, макс дист {a[:, 2].max():5.1f}, "
                f"бок {np.median(a[:, 3]):+.2f}, выс {np.median(a[:, 4]):.2f}, шир {np.median(a[:, 5]):.2f}, точек {np.median(a[:, 6]):.0f}"
            )


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(*sys.argv[2:3])
    else:
        {"run": run, "report": report}[sys.argv[1]]()
