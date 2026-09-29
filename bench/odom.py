"""Эго-движение по облакам (KISS-ICP) и проверка вырождения вдоль оси тоннеля."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP

from bench.data import BAGS, CACHE_ROOT, Bag

OUT = Path(__file__).resolve().parent.parent / "out" / "odom"


def run(name: str, max_range: float = 80.0, voxel: float = 0.5, deskew: bool = True) -> dict:
    bag = Bag(name)
    cfg = KISSConfig()
    cfg.data.max_range = max_range
    cfg.data.min_range = 2.0
    cfg.data.deskew = deskew
    cfg.mapping.voxel_size = voxel
    icp = KissICP(cfg)
    poses, t_ms = [], []
    for k in range(len(bag)):
        fr = bag[k]
        xyz = fr.xyz.astype(np.float64)
        dt = fr.pts["dt"].astype(np.float64)
        ts = np.clip(dt / 0.1, 0.0, 1.0)  # доля оборота 10 Гц, а не длительности сектора
        t0 = time.perf_counter()
        icp.register_frame(xyz, ts)
        t_ms.append((time.perf_counter() - t0) * 1e3)
        poses.append(icp.last_pose.copy())
    poses = np.array(poses)
    np.save(CACHE_ROOT / f"{name}.poses.npy", poses)
    tr = poses[:, :3, 3]
    step = np.linalg.norm(np.diff(tr, axis=0), axis=1)
    speed = step / np.maximum(np.diff(bag.stamps), 1e-3) * 3.6
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(14, 7))
    ax[0].plot(tr[:, 0], tr[:, 1])
    ax[0].set_title(f"{name}: траектория XY")
    ax[0].set_aspect("equal")
    ax[1].plot(bag.stamps[1:] - bag.stamps[0], speed)
    ax[1].set_title("скорость, км/ч (по одометрии)")
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=60)
    plt.close(fig)
    return {
        "bag": name,
        "path_m": float(step.sum()),
        "speed_kmh_p50_max": [float(np.median(speed)), float(speed.max())],
        "speed_jitter_kmh": float(np.median(np.abs(np.diff(speed)))),
        "z_drift_m": float(tr[-1, 2] - tr[0, 2]),
        "ms_per_frame_p50_p95": [float(np.median(t_ms)), float(np.percentile(t_ms, 95))],
    }


if __name__ == "__main__":
    res = [run(n) for n in (sys.argv[1:] or BAGS)]
    print(json.dumps(res, indent=1, ensure_ascii=False))
    (OUT / "odom.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
