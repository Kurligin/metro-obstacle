"""Обзор каждого bag: положение лидара, сетка лучей, картинки сцен."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench.data import BAGS, Bag

OUT = Path(__file__).resolve().parent.parent / "out" / "survey"


def ground_plane(xyz: np.ndarray, rng=np.random.default_rng(0)):
    """RANSAC-плоскость по нижним точкам в 3–25 м впереди. Возвращает (n, d): n·p + d = 0."""
    m = (xyz[:, 0] > 3) & (xyz[:, 0] < 25) & (np.abs(xyz[:, 1]) < 2.5)
    p = xyz[m]
    p = p[p[:, 2] < np.percentile(p[:, 2], 40)]
    best, best_n = None, -1
    for _ in range(300):
        s = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < 0.95:
            continue
        d = -n @ s[0]
        cnt = np.sum(np.abs(p @ n + d) < 0.05)
        if cnt > best_n:
            best, best_n = (n, d), cnt
    n, d = best
    inl = p[np.abs(p @ n + d) < 0.05]
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c)
    n = vt[2] if vt[2][2] > 0 else -vt[2]
    return n, -n @ c


def ring_elevations(pts) -> np.ndarray:
    el = np.degrees(np.arctan2(pts["z"], np.hypot(pts["x"], pts["y"])))
    out = np.full(128, np.nan)
    for r in range(128):
        v = el[pts["ring"] == r]
        if len(v) > 20:
            out[r] = np.median(v)
    return out


def views(fr, title, path, xmax=120):
    p = fr.pts
    m = (p["x"] > 0) & (p["x"] < xmax)
    fig, ax = plt.subplots(3, 1, figsize=(20, 11))
    mm = m & (np.abs(p["y"]) < 8)
    ax[0].scatter(p["x"][mm], p["y"][mm], s=0.2, c=p["z"][mm], cmap="viridis", vmin=-2.5, vmax=2.5)
    ax[0].set_title(f"{title}: вид сверху (цвет — высота)")
    ax[0].set_aspect("equal")
    m2 = m & (np.abs(p["y"]) < 1.5)
    ax[1].scatter(p["x"][m2], p["z"][m2], s=0.3, c=p["i"][m2], cmap="plasma", vmax=80)
    ax[1].set_title("вид сбоку, |y|<1.5 (цвет — интенсивность)")
    ax[1].set_aspect("equal")
    m3 = (p["x"] > 8) & (p["x"] < 12)
    ax[2].scatter(p["y"][m3], p["z"][m3], s=1, c=p["x"][m3])
    ax[2].set_title("поперечный срез 8–12 м")
    ax[2].set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=60)
    plt.close(fig)


def survey(name: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    bag = Bag(name)
    n = len(bag)
    fr0 = bag[0]
    gn, gd = ground_plane(fr0.xyz)
    pitch = np.degrees(np.arctan2(gn[0], gn[2]))
    roll = np.degrees(np.arctan2(gn[1], gn[2]))
    el = ring_elevations(fr0.pts)
    az = np.degrees(np.arctan2(fr0.pts["y"], fr0.pts["x"]))
    r0 = fr0.pts[fr0.pts["ring"] == 64]
    a0 = np.sort(np.degrees(np.arctan2(r0["y"], r0["x"])))
    daz = np.median(np.diff(a0)) if len(a0) > 10 else float("nan")
    for k in np.linspace(0, n - 1, 5).astype(int):
        views(bag[k], f"{name} #{k}", OUT / f"{name}_{k:04d}.png")
    return {
        "bag": name,
        "frames": n,
        "duration_s": float(bag.stamps[-1] - bag.stamps[0]),
        "pts_per_frame": int(np.median(np.diff(bag.offs))),
        "lidar_height_m": float(gd),
        "pitch_deg": float(pitch),
        "roll_deg": float(roll),
        "az_range_deg": [float(az.min()), float(az.max())],
        "az_step_deg": float(daz),
        "ring_el_deg_min_max": [float(np.nanmin(el)), float(np.nanmax(el))],
        "ring_el_step_mid": float(np.nanmedian(np.abs(np.diff(el[40:90])))),
    }


if __name__ == "__main__":
    res = [survey(n) for n in (sys.argv[1:] or BAGS)]
    print(json.dumps(res, indent=1, ensure_ascii=False))
    (OUT / "survey.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
