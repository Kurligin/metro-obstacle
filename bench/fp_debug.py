"""Показать ложные тревоги: кадры с детекциями на пустых bag, точки-кандидаты подсвечены."""
import json, sys, warnings
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench.data import Bag, CACHE_ROOT
from bench.detect import CONFIGS, Detector
from bench.evaluate import OUT, _oracle, EMPTY_BAGS
warnings.filterwarnings("ignore")
cname = sys.argv[1] if len(sys.argv) > 1 else "Oracle+D"
cfg = {c.name: c for c in CONFIGS}[cname]
picks = []
rng = np.random.default_rng(1)
for b in (EMPTY_BAGS if len(sys.argv) <= 2 else []):
    rows = json.load(open(OUT / f"empty__{b}__{cname}.json"))
    ks = [r["k"] for r in rows if r["dets"]]
    for k in rng.choice(ks, min(2, len(ks)), replace=False):
        picks.append((b, int(k)))
if len(sys.argv) > 2:
    picks = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[2:]]
fig, ax = plt.subplots(max(len(picks), 2), 2, figsize=(24, 4 * len(picks)))
for row, (b, k) in enumerate(picks):
    bag = Bag(b); P = np.load(CACHE_ROOT / f"{b}.traj.npy")
    det = Detector(cfg)
    for j in range(max(0, k - 25), k + 1):
        fr = bag[j]
        dets, info = det.process(fr.pts, _oracle(b, j, fr.xyz, P) if cfg.axis == "oracle" else None)
    H = det.hist[-1]
    x = fr.xyz
    xs, ys, zs = det.last_axis
    m = (x[:, 0] > 0) & (x[:, 0] < 130) & (np.abs(x[:, 1]) < 6)
    ax[row, 0].scatter(x[m, 0], x[m, 1], s=0.1, c="gray")
    ax[row, 0].plot(xs, ys + 1.05, "b-", xs, ys - 1.05, "b-", lw=0.7)
    ax[row, 0].scatter(H["x"], H["y"], s=6, c="r")
    for d in dets:
        ax[row, 0].axvline(d.s, color="orange", lw=0.8)
    ax[row, 0].set_title(f"{b} k={k} {cname}: dets " + ", ".join(f"{d.s:.0f}м lat{d.lat:+.2f} h{d.h_top:.2f} n{d.n}" for d in dets[:5]))
    ax[row, 0].set_xlim(0, 130)
    yax = np.interp(x[:, 0], xs, ys)
    mm = m & (np.abs(x[:, 1] - yax) < 1.3)
    ax[row, 1].scatter(x[mm, 0], x[mm, 2] - np.interp(x[mm, 0], xs, zs), s=0.3, c="gray")
    ax[row, 1].scatter(H["x"], H["h"], s=6, c="r")
    ax[row, 1].set_ylim(-0.6, 3.2); ax[row, 1].set_xlim(0, 130); ax[row, 1].set_title("сбоку, высота над осью (серые) / над нормой (красные)")
plt.tight_layout(); plt.savefig(f"out/fp_{cname}.png", dpi=45)
print(picks)
