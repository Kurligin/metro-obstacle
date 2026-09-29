"""Чтение bag-файлов и кэш кадров в удобной системе координат.

Система координат кадра после нормализации (как в ROS REP-103):
x — вперёд по ходу, y — влево, z — вверх. В исходных данных «вперёд» — это −Y.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_ROOT = Path(
    os.environ.get("LCT_DATA", Path.home() / "Downloads/archive/extracted/for_hackathon")
)
CACHE_ROOT = Path(os.environ.get("LCT_CACHE", Path.home() / "Downloads/archive/cache"))

# Потолок параллельности стенда: не больше 70% ядер ноутбука (LCT_WORKERS переопределяет).
WORKERS = int(os.environ.get("LCT_WORKERS", max(1, int((os.cpu_count() or 4) * 0.7))))

BAGS = [
    "doubleT_obstacle",
    "doubleT_platform",
    "roundT_doubleT",
    "roundT_pressureGate_roundT",
    "roundT_squareT_pressureGate_squareT",
    "squareT_platform_squareT_switch",
]

# Кадр в кэше: xyz (float32), intensity (uint8), ring (uint8), dt — время точки
# относительно начала кадра, с (float32).
POINT_DTYPE = np.dtype(
    [("x", "f4"), ("y", "f4"), ("z", "f4"), ("i", "u1"), ("ring", "u1"), ("dt", "f4")]
)

# Кроп при кэшировании: вперёд до 230 м, вбок ±12 м, чуть назад — для калибровки.
CROP_X = (-3.0, 230.0)
CROP_Y = 12.0


@dataclass
class Frame:
    stamp: float  # время кадра, с
    pts: np.ndarray  # POINT_DTYPE

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.pts["x"], self.pts["y"], self.pts["z"]], axis=1)


def _raw_dtype(msg) -> np.dtype:
    types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    return np.dtype(
        {
            "names": [f.name for f in msg.fields],
            "formats": [types[f.datatype] for f in msg.fields],
            "offsets": [f.offset for f in msg.fields],
            "itemsize": msg.point_step,
        }
    )


def convert(msg) -> Frame:
    """PointCloud2 → Frame в нормализованных осях, без невалидных точек."""
    a = np.frombuffer(msg.data, dtype=_raw_dtype(msg))
    x0, y0, z0 = a["x"], a["y"], a["z"]
    ok = np.isfinite(x0) & ((np.abs(x0) + np.abs(y0) + np.abs(z0)) > 0)
    a = a[ok]
    x, y, z = -a["y"], a["x"], a["z"]  # поворот на +90° вокруг z: −Y → +X
    keep = (x > CROP_X[0]) & (x < CROP_X[1]) & (np.abs(y) < CROP_Y)
    a, x, y, z = a[keep], x[keep], y[keep], z[keep]
    out = np.empty(len(a), POINT_DTYPE)
    out["x"], out["y"], out["z"] = x, y, z
    out["i"] = np.clip(a["intensity"], 0, 255).astype(np.uint8)
    out["ring"] = a["ring"].astype(np.uint8)
    t = a["timestamp"].astype(np.float64)
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    t0 = t.min() if len(t) else stamp
    out["dt"] = (t - t0).astype(np.float32)
    return Frame(stamp=float(t0), pts=out)


def iter_bag(name: str):
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(DATA_ROOT / name) as r:
        for conn, _, raw in r.messages():
            yield convert(ts.deserialize_cdr(raw, conn.msgtype))


def cache_bag(name: str) -> Path:
    """Все кадры bag в один .npy (точки подряд) + индекс смещений и времён."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    pts_path = CACHE_ROOT / f"{name}.pts.npy"
    idx_path = CACHE_ROOT / f"{name}.idx.npz"
    if pts_path.exists() and idx_path.exists():
        return pts_path
    chunks, offs, stamps = [], [0], []
    for fr in iter_bag(name):
        chunks.append(fr.pts)
        offs.append(offs[-1] + len(fr.pts))
        stamps.append(fr.stamp)
    np.save(pts_path, np.concatenate(chunks))
    np.savez(idx_path, offs=np.array(offs), stamps=np.array(stamps))
    return pts_path


class Bag:
    """Быстрый доступ к кэшированным кадрам (memory-mapped)."""

    def __init__(self, name: str):
        self.name = name
        cache_bag(name)
        self._pts = np.load(CACHE_ROOT / f"{name}.pts.npy", mmap_mode="r")
        idx = np.load(CACHE_ROOT / f"{name}.idx.npz")
        self.offs, self.stamps = idx["offs"], idx["stamps"]

    def __len__(self) -> int:
        return len(self.stamps)

    def __getitem__(self, k: int) -> Frame:
        return Frame(self.stamps[k], np.asarray(self._pts[self.offs[k] : self.offs[k + 1]]))


if __name__ == "__main__":
    import sys
    from concurrent.futures import ProcessPoolExecutor

    names = sys.argv[1:] or BAGS
    with ProcessPoolExecutor(min(len(names), WORKERS)) as ex:
        for n, p in zip(names, ex.map(cache_bag, names)):
            print(n, p, flush=True)
