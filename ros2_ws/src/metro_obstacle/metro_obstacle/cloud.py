"""PointCloud2 → numpy без поточечного цикла.

Модуль не импортирует ROS: сообщение читается «утиным» способом (fields, point_step,
row_step, width, height, is_bigendian, data), поэтому его можно тестировать без rclpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# sensor_msgs/PointField.datatype → код numpy.
_POINTFIELD_DTYPES: dict[int, str] = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    8: "f8",
}

# Шаг гистограммы углов места при восстановлении колец. Лучи Pandar128 разнесены
# минимум на 0.125°, так что между соседними лучами всегда остаются пустые ячейки.
_RING_BIN_DEG = 0.02


def _structured_dtype(msg: Any) -> np.dtype:
    order = ">" if msg.is_bigendian else "<"
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype not in _POINTFIELD_DTYPES:
            continue
        code = order + _POINTFIELD_DTYPES[f.datatype]
        count = int(getattr(f, "count", 1) or 1)
        names.append(f.name)
        formats.append(code if count == 1 else (code, count))
        offsets.append(f.offset)
    return np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step}
    )


def _points_view(msg: Any) -> np.ndarray:
    """Структурированный вид на буфер сообщения (без копирования, если строки плотные)."""
    dtype = _structured_dtype(msg)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    n = int(msg.width) * int(msg.height)
    dense = int(msg.width) * int(msg.point_step)
    if int(msg.height) > 1 and int(msg.row_step) != dense:
        # Строки с выравниванием: отрезаем хвост каждой строки и склеиваем.
        rows = buf[: int(msg.row_step) * int(msg.height)].reshape(int(msg.height), -1)
        buf = np.ascontiguousarray(rows[:, :dense]).reshape(-1)
    return np.frombuffer(buf.data, dtype=dtype, count=n)


def rings_from_elevation(xyz: np.ndarray) -> np.ndarray:
    """Номер кольца (луча) по углу места, если в облаке нет поля ring.

    Лучи лидара дискретны по углу места: гистограмма углов состоит из узких пиков,
    разделённых пустыми промежутками. Каждая связная группа занятых ячеек — один
    луч; кольца нумеруются снизу вверх. Невалидные точки получают кольцо 0.
    """
    ring = np.zeros(len(xyz), np.int64)
    ok = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
    if not ok.any():
        return ring
    p = xyz[ok].astype(np.float64)
    elev = np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
    idx = np.floor((elev + 90.0) / _RING_BIN_DEG).astype(np.int64)
    counts = np.bincount(idx)
    # Одиночные выбросы не должны склеивать соседние лучи.
    occupied = counts >= max(2, int(1e-4 * len(p)))
    starts = occupied & ~np.concatenate([[False], occupied[:-1]])
    group = np.cumsum(starts) - 1
    r = group[idx]
    # Точка в «пустой» ячейке (выброс) относится к ближайшей группе снизу.
    ring[ok] = np.maximum(r, 0)
    return ring


def cloud_to_arrays(msg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Разбор PointCloud2 в массивы для детектора.

    Возвращает xyz (N,3) float32 в осях сообщения (NaN и нулевые точки сохраняются —
    их отбрасывает детектор), ring (N,) целые, intensity (N,) float32 или None.
    """
    pts = _points_view(msg)
    names = pts.dtype.names or ()
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"PointCloud2 has no '{axis}' field (fields: {list(names)})")
    xyz = np.empty((len(pts), 3), np.float32)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = pts["x"], pts["y"], pts["z"]
    ring = pts["ring"].astype(np.int64) if "ring" in names else rings_from_elevation(xyz)
    intensity = pts["intensity"].astype(np.float32) if "intensity" in names else None
    return xyz, ring, intensity
