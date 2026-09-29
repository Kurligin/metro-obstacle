"""Воронка потерь: на каком шаге ядро теряет видимый объект на дальности."""
import sys, warnings
import numpy as np
from bench.data import CACHE_ROOT
from bench.round4 import make, feed, OBJ
from bench.synth import Obj, Scenario, frames_with_objects
warnings.filterwarnings("ignore")

def run(bag, s_obj, t, lat, dmin=65, dmax=130, spec="default"):
    size, hgt = OBJ.get(t, ((1.0, 1.2, 1.0), 0.0))
    poses = np.load(CACHE_ROOT / f"{bag}.traj.npy")
    s_tr = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    d_ahead = s_obj - s_tr
    ks = [k for k in range(len(s_tr)) if -2 < d_ahead[k] < 215]
    ks = list(range(max(ks[0] - 20, 0), ks[-1] + 1))
    det = make(spec); det.debug = True
    out = []
    for k, pts, lab, info in frames_with_objects(Scenario(bag, [Obj(t, size, s=s_obj, lat=lat, h=hgt)]), ks=ks):
        res = feed(det, pts)
        m = lab == 0
        if not (info and m.sum() >= 3 and dmin <= d_ahead[k] < dmax):
            continue
        ox = float(pts["x"][m].min())
        D = det.dbg
        axis_end = float(det.last_axis[0][-1]) if len(det.last_axis[0]) else 0.0
        stage = None
        if D is None or ox > axis_end - 3:
            stage = "за концом оси"
        else:
            key = {(round(float(a), 3), round(float(b), 3)) for a, b in zip(pts["x"][m], pts["y"][m])}
            idx = [i for i, (a, b) in enumerate(zip(D["px"], D["py"])) if (round(float(a), 3), round(float(b), 3)) in key]
            if not idx:
                stage = "вне коридора (|lat|>рабочей зоны)"
            else:
                i = np.array(idx)
                lat_ok = np.abs(D["lat"][i]) < D["half_w"][i]
                h_ok = D["hh"][i] > (D["thr"][i] if np.ndim(D["thr"]) else D["thr"])
                if D["cand"][i].sum() >= 1:
                    near = [o for o in res.obstacles if abs(o.distance - ox) < 2.5]
                    if not near:
                        stage = "кластер не прошёл правила"
                    elif not any(o.confirmed for o in near):
                        stage = "не подтверждён (M из N)"
                    else:
                        stage = "ПОЙМАН"
                elif not lat_ok.any():
                    stage = "за суженным краем габарита"
                elif not (lat_ok & h_ok).any():
                    stage = "ниже порога высоты"
                else:
                    stage = "прочее (рельсы/верх/конец)"
        out.append((round(float(d_ahead[k])), stage))
    return out

if __name__ == "__main__":
    from collections import Counter
    jobs = [("roundT_pressureGate_roundT", 300.0), ("roundT_squareT_pressureGate_squareT", 300.0), ("roundT_squareT_pressureGate_squareT", 600.0), ("squareT_platform_squareT_switch", 300.0), ("roundT_doubleT", 300.0)]
    for t in sys.argv[1:] or ["person"]:
        for lat in (0.0, 0.6):
            C = Counter()
            for bag, s in jobs:
                for d, st in run(bag, s, t, lat):
                    C[("65-90" if d < 90 else "90-130", st)] += 1
            print(f"== {t} lat={lat}")
            for band in ("65-90", "90-130"):
                tot = sum(v for (b, _), v in C.items() if b == band)
                print(f"  {band} м ({tot} кадров): " + "; ".join(f"{st} {v/tot:.0%}" for (b, st), v in sorted(C.items(), key=lambda x: -x[1]) if b == band))
