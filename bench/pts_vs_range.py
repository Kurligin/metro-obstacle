"""Сколько точек лидар кладёт на объект в зависимости от дальности (по пресетам)."""
import json, sys
import numpy as np
from bench.synth import PRESETS, Obj, Scenario, frames_with_objects

bag = sys.argv[1] if len(sys.argv) > 1 else "roundT_doubleT"
S0 = float(sys.argv[2]) if len(sys.argv) > 2 else 260.0
objs = []
for i, (name, size) in enumerate(PRESETS.items()):
    h = 3.0 - size[2] if name == "cable" else 0.0  # кабель свисает от верха габарита
    objs.append(Obj(name, size, s=S0 + i * 0.0, lat=(i - 2) * 0.35, h=h))
rows = []
for k, pts, lab, info in frames_with_objects(Scenario(bag, objs), stride=2):
    for o, d, n in info:
        rows.append((o.kind, d, n))
res = {}
for name in PRESETS:
    r = np.array([(d, n) for kk, d, n in rows if kk == name])
    if not len(r):
        continue
    bins = [0, 20, 40, 60, 80, 100, 125, 150, 200, 260]
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (r[:, 0] >= a) & (r[:, 0] < b)
        if m.any():
            out.append((f"{a}-{b}", round(float(r[m, 1].mean()), 1), round(float((r[m, 1] > 0).mean()), 2)))
    res[name] = out
for k2, v in res.items():
    print(k2.ljust(10), "  ".join(f"{a}м:{n}т/{p:.0%}" for a, n, p in v))
