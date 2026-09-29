"""Траектория поезда для стенда: скорость по профилю стен + поворот/бок из KISS-ICP.

1. Для каждой пары кадров — кривая корреляции профилей стен по сдвигу.
2. Витерби по сдвигам: штраф за изменение скорости (ускорение поезда ≤ ~1.5 м/с²)
   убирает «залипания» на нуле.
3. Позы: вращение и поперечные компоненты шага — из KISS-ICP, продольная — из п.2.
"""

from __future__ import annotations

import sys

import numpy as np

from bench.data import BAGS, CACHE_ROOT, Bag
from bench.profile_speed import BIN, profile

MAX_SHIFT = 3.0


def score_matrix(bag: Bag) -> np.ndarray:
    prof = [profile(bag[k]) for k in range(len(bag))]
    ks = np.arange(0, int(MAX_SHIFT / BIN) + 1)
    n = prof[0].shape[1]
    S = np.zeros((len(bag) - 1, len(ks)))
    for k in range(len(bag) - 1):
        a, b = prof[k], prof[k + 1]
        S[k] = [np.sum(a[:, j:] * b[:, : n - j]) for j in ks]
    return S


def viterbi_speed(S: np.ndarray, dts: np.ndarray, max_acc: float = 1.5) -> np.ndarray:
    """Сдвиг (м) на каждый шаг с ограничением на изменение между шагами."""
    T, K = S.shape
    shifts = np.arange(K) * BIN
    # мягкий штраф: квадрат превышения допустимого изменения сдвига
    lim = max_acc * np.mean(dts) ** 2 + BIN
    dd = np.abs(shifts[:, None] - shifts[None, :])
    trans = -50.0 * np.maximum(dd - lim, 0) / BIN
    Sn = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
    acc = Sn[0].copy()
    back = np.zeros((T, K), int)
    for t in range(1, T):
        cand = acc[:, None] + trans
        back[t] = np.argmax(cand, 0)
        acc = cand[back[t], np.arange(K)] + Sn[t]
    path = np.zeros(T, int)
    path[-1] = int(np.argmax(acc))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return shifts[path]


def fused_poses(bag: Bag, dx: np.ndarray) -> np.ndarray:
    P = np.load(CACHE_ROOT / f"{bag.name}.poses.npy")
    out = [np.eye(4)]
    for k in range(len(P) - 1):
        rel = np.linalg.inv(P[k]) @ P[k + 1]
        rel = rel.copy()
        rel[0, 3] = dx[k]  # продольную компоненту заменяем
        out.append(out[-1] @ rel)
    return np.array(out)


def build(name: str) -> dict:
    bag = Bag(name)
    S = score_matrix(bag)
    dx = viterbi_speed(S, np.diff(bag.stamps))
    poses = fused_poses(bag, dx)
    np.save(CACHE_ROOT / f"{name}.traj.npy", poses)
    v = dx / np.diff(bag.stamps) * 3.6
    np.save(CACHE_ROOT / f"{name}.speed.npy", v)
    return {"bag": name, "path_m": float(dx.sum()), "v_p10_50_90": np.percentile(v, [10, 50, 90]).round(1).tolist()}


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sys.argv[1:] or BAGS
    fig, ax = plt.subplots(len(names), 2, figsize=(16, 3 * len(names)))
    ax = np.atleast_2d(ax)
    for i, n in enumerate(names):
        r = build(n)
        print(r, flush=True)
        b = Bag(n)
        v = np.load(CACHE_ROOT / f"{n}.speed.npy")
        tr = np.load(CACHE_ROOT / f"{n}.traj.npy")[:, :3, 3]
        ax[i, 0].plot(b.stamps[1:] - b.stamps[0], v)
        ax[i, 0].set_title(f"{n}: скорость, км/ч")
        ax[i, 1].plot(tr[:, 0], tr[:, 1])
        ax[i, 1].set_title("траектория XY, м")
    plt.tight_layout()
    plt.savefig("out/odom/traj.png", dpi=55)
