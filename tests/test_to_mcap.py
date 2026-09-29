"""bench/to_mcap.py: запись ROS 2 (.db3) → ядро → MCAP (ros2msg/cdr) без ROS."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcap_ros2")
pytest.importorskip("rosbags")

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from metro_obstacle_core.synthetic import to_message_axes, tunnel_points
from rosbags.rosbag2 import Writer as BagWriter
from rosbags.typesys import Stores, get_typestore

from bench import to_mcap

TS = get_typestore(Stores.ROS2_HUMBLE)
FRAMES = 22
T0_NS = 1_700_000_000_000_000_000


def _cloud(xyz: np.ndarray, inten: np.ndarray, sec: int, nsec: int) -> bytes:
    """PointCloud2 как у синтетики заказчика: x, y, z, intensity (16 байт), без ring."""
    pf = TS.types["sensor_msgs/msg/PointField"]
    pts = np.empty(len(xyz), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4")])
    pts["x"], pts["y"], pts["z"], pts["i"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], inten
    msg = TS.types["sensor_msgs/msg/PointCloud2"](
        header=TS.types["std_msgs/msg/Header"](
            stamp=TS.types["builtin_interfaces/msg/Time"](sec=sec, nanosec=nsec),
            frame_id="hesai_lidar",
        ),
        height=1,
        width=len(pts),
        fields=[
            pf(name=n, offset=4 * k, datatype=7, count=1)
            for k, n in enumerate(("x", "y", "z", "intensity"))
        ],
        is_bigendian=False,
        point_step=16,
        row_step=16 * len(pts),
        data=pts.view(np.uint8).reshape(-1),
        is_dense=False,
    )
    return bytes(TS.serialize_cdr(msg, "sensor_msgs/msg/PointCloud2"))


@pytest.fixture(scope="module")
def bag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("bag") / "synthetic"
    rng = np.random.default_rng(0)
    with BagWriter(path, version=5) as w:
        conn = w.add_connection("/lidar_points", "sensor_msgs/msg/PointCloud2", typestore=TS)
        for k in range(FRAMES):
            xyz, _, inten = tunnel_points(rng, box=(40.0, 0.2, 0.5, 0.6))
            raw = _cloud(to_message_axes(xyz, "-y"), inten, 1000 + k // 10, (k % 10) * 100_000_000)
            w.write(conn, T0_NS + k * 100_000_000, raw)
    return path


def _read(path: Path) -> dict[str, list]:
    out: dict[str, list] = {}
    with path.open("rb") as f:
        for schema, channel, message, decoded in make_reader(
            f, decoder_factories=[DecoderFactory()]
        ).iter_decoded_messages():
            assert schema.encoding == "ros2msg" and channel.message_encoding == "cdr"
            out.setdefault(channel.topic, []).append((schema.name, message, decoded))
    return out


def test_bag_files_and_topic(bag: Path) -> None:
    files = to_mcap.db3_files(bag)
    assert len(files) == 1 and files[0].suffix == ".db3"
    assert to_mcap.db3_files(files[0]) == files  # путь к самому .db3 тоже годится
    assert to_mcap.find_cloud_topic(files) == "/lidar_points"
    with pytest.raises(ValueError, match="/nope"):
        to_mcap.find_cloud_topic(files, "/nope")


def test_start_and_max_frames(bag: Path) -> None:
    files = to_mcap.db3_files(bag)
    got = [t for t, _ in to_mcap.iter_raw(files, "/lidar_points", start=5, max_frames=3)]
    assert got == [T0_NS + k * 100_000_000 for k in (5, 6, 7)]


def test_convert_roundtrip(bag: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.mcap"
    stats = to_mcap.convert(bag, out, log=lambda _: None)
    assert stats.frames == FRAMES and out.stat().st_size > 0
    msgs = _read(out)

    clouds = msgs["/lidar_points"]
    assert len(clouds) == FRAMES and clouds[0][0] == "sensor_msgs/msg/PointCloud2"
    first = clouds[0][2]
    assert first.header.frame_id == "hesai_lidar" and first.point_step == 16
    assert [f.name for f in first.fields] == ["x", "y", "z", "intensity"]
    # время — из записи: log_time = время bag, штамп заголовка — как у облака
    assert [m.log_time for _, m, _ in clouds] == [T0_NS + k * 100_000_000 for k in range(FRAMES)]
    assert (first.header.stamp.sec, first.header.stamp.nanosec) == (1000, 0)

    status = [json.loads(d.data) for _, _, d in msgs["/metro_obstacle/status"]]
    assert len(status) == FRAMES
    assert set(status[0]) == {"obstacle", "distance", "calibrated"}
    last = status[-1]
    assert last["calibrated"] and last["obstacle"]
    assert last["distance"] == pytest.approx(40.0, abs=0.1)
    assert status[0]["distance"] is None  # NaN → null: строгий JSON

    arrays = msgs["/metro_obstacle/markers"]
    assert len(arrays) == FRAMES and arrays[0][0] == "visualization_msgs/msg/MarkerArray"
    markers = arrays[-1][2].markers
    assert markers[0].action == to_mcap.DELETEALL
    assert markers[0].header.frame_id == "hesai_lidar"
    cubes = [m for m in markers if m.type == to_mcap.CUBE]
    assert cubes and all(m.color.r == pytest.approx(1.0) for m in cubes)
    assert all(
        m.lifetime.nanosec > 0 or m.lifetime.sec > 0 for m in markers[1:]
    )  # не копятся, даже если DELETEALL пропустят
    texts = [m.text for m in markers if m.type == to_mcap.TEXT_VIEW_FACING]
    assert "40.0 m" in texts and "OBSTACLE 40.0 m" in texts
    ns = {m.ns for m in markers}
    assert {"corridor", "corridor_fill", "obstacles", "obstacle_labels", "status"} <= ns
    strips = [m for m in markers if m.ns == "corridor" and m.type == to_mcap.LINE_STRIP]
    assert len(strips) == 4 and len(strips[0].points) > 10

    inside = msgs["/metro_obstacle/in_corridor"]
    pts = inside[-1][2]
    assert pts.point_step == 12 and pts.width > 0
    xyz = np.frombuffer(bytes(pts.data), "<f4").reshape(-1, 3)
    # точки внутри коридора — это и есть объект на 40 м по ходу (−Y), у оси пути
    assert np.median(xyz[:, 1]) == pytest.approx(-40.0, abs=1.0)
    assert np.abs(xyz[:, 0]).max() < 1.05


def test_status_payload_nan() -> None:
    class R:
        obstacle, distance, calibrated = False, math.nan, True

    assert json.loads(to_mcap.status_json(R())) == {
        "obstacle": False,
        "distance": None,
        "calibrated": True,
    }
