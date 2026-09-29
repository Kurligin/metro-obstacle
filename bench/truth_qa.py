"""Картинки для визуальной проверки эталонной оси: вид сверху и сбоку с габаритом."""
import sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench.data import BAGS, CACHE_ROOT, Bag
from bench.truth import truth_axis

H = {"doubleT_obstacle": 1.73}
for n in sys.argv[1:] or BAGS:
    b = Bag(n); P = np.load(CACHE_ROOT / f"{n}.traj.npy")
    ks = np.linspace(0, len(b) - 1, 6).astype(int)
    fig, ax = plt.subplots(6, 2, figsize=(22, 20))
    for r, k in enumerate(ks):
        x = b[k].xyz
        xs, ys, zs = truth_axis(x, P, k, H.get(n, 1.35))
        m = (x[:, 0] > 0) & (x[:, 0] < 200) & (np.abs(x[:, 1]) < 25)
        ax[r, 0].scatter(x[m, 0], x[m, 1], s=0.1, c="gray")
        for off in (-1.05, 1.05):
            ax[r, 0].plot(xs, ys + off, "r-", lw=1)
        ax[r, 0].set_title(f"{n} k={k}: сверху, габарит ±1.05"); ax[r, 0].set_xlim(0, 200)
        yax = np.interp(x[:, 0], xs, ys) if len(xs) else 0
        mm = m & (np.abs(x[:, 1] - yax) < 1.05)
        ax[r, 1].scatter(x[mm, 0], x[mm, 2], s=0.3, c="gray")
        ax[r, 1].plot(xs, zs, "r-"); ax[r, 1].plot(xs, zs + 3.0, "r--")
        ax[r, 1].set_title("сбоку: точки внутри габарита по ширине"); ax[r, 1].set_xlim(0, 200)
    plt.tight_layout(); plt.savefig(f"out/truth_{n}.png", dpi=45); plt.close(fig)
    print(n, flush=True)
