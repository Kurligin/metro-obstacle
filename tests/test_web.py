"""Веб-прототип (web/): API на синтетической записи — загрузка → обработка → кадр."""

from __future__ import annotations

import io
import json
import socket
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mcap_ros2")
pytest.importorskip("rosbags")

from metro_obstacle_core.synthetic import to_message_axes, tunnel_points
from rosbags.rosbag2 import Writer as BagWriter
from test_to_mcap import TS, _cloud

from web import pipeline, server

FRAMES = 22
T0_NS = 1_700_000_000_000_000_000


@pytest.fixture(scope="module")
def bags_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Каталог записей: синтетический тоннель с объектом на 40 м по ходу (−Y), без ring."""
    root = tmp_path_factory.mktemp("bags")
    rng = np.random.default_rng(0)
    with BagWriter(root / "synthetic", version=5) as w:
        conn = w.add_connection("/lidar_points", "sensor_msgs/msg/PointCloud2", typestore=TS)
        for k in range(FRAMES):
            xyz, _, inten = tunnel_points(rng, box=(40.0, 0.2, 0.5, 0.6))
            raw = _cloud(to_message_axes(xyz, "-y"), inten, 1000 + k // 10, (k % 10) * 100_000_000)
            w.write(conn, T0_NS + k * 100_000_000, raw)
    return root


@pytest.fixture(scope="module")
def base(bags_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    srv = server.make_server("127.0.0.1", 0, bags_dir, tmp_path_factory.mktemp("web"))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.app.jobs.shutdown()
    srv.server_close()


def _req(
    url: str, data: bytes | None = None, method: str | None = None, ctype: str = "application/json"
) -> tuple[int, bytes, Any]:
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": ctype} if data is not None else {}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def _get(url: str) -> Any:
    code, body, _ = _req(url)
    assert code == 200, body
    return json.loads(body)


def _post(url: str, payload: dict[str, Any]) -> tuple[int, Any]:
    code, body, _ = _req(url, json.dumps(payload).encode(), "POST")
    return code, json.loads(body)


def _wait(base: str, jid: str, key: str = "state", timeout: float = 120.0) -> dict[str, Any]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = _get(f"{base}/api/jobs/{jid}")
        state = st["state"] if key == "state" else st["mcap"]["state"]
        if state in ("done", "error"):
            return st
        time.sleep(0.2)
    raise TimeoutError(jid)


def test_index_and_vendor(base: str) -> None:
    code, body, headers = _req(f"{base}/")
    assert code == 200 and b'id="view"' in body
    assert headers["Content-Type"].startswith("text/html")
    code, _, headers = _req(f"{base}/static/vendor/three.module.min.js")
    assert code == 200 and headers["Content-Type"].startswith("text/javascript")
    # шрифты — локально, без внешних URL
    code, _, headers = _req(f"{base}/static/vendor/fonts/inter-cyrillic-400-normal.woff2")
    assert code == 200 and headers["Content-Type"] == "font/woff2"
    assert b"http://" not in body and b"https://" not in body
    assert _req(f"{base}/static/../server.py")[0] == 404


def test_list_and_reject(base: str) -> None:
    bags = _get(f"{base}/api/bags")["bags"]
    assert [b["id"] for b in bags] == ["bags:synthetic"] and bags[0]["size"] > 0
    code, err = _post(f"{base}/api/jobs", {"bag": "bags:../.."})
    assert code == 400 and "error" in err
    code, err = _post(f"{base}/api/jobs", {"bag": "nope:x"})
    assert code == 400
    assert _req(f"{base}/api/jobs/unknown")[0] == 404


def test_process_bag_from_catalog(base: str) -> None:
    code, created = _post(f"{base}/api/jobs", {"bag": "bags:synthetic"})
    assert code == 201
    jid = created["id"]
    st = _wait(base, jid)
    assert st["state"] == "done", st["error"]
    assert st["done"] == st["total"] == FRAMES and st["topic"] == "/lidar_points"

    s = st["summary"]
    assert s["frames"] == FRAMES and 0 < s["alarm_frames"] < FRAMES
    assert s["first_alarm"]["distance"] == pytest.approx(40.0, abs=0.3)
    assert s["core_ms"]["median"] > 0 and s["record_s"] == pytest.approx(2.1, abs=0.01)
    ev = st["events"]
    assert len(ev) == 1 and ev[0]["end"] == FRAMES - 1
    assert ev[0]["frames"] == s["alarm_frames"] and ev[0]["width"] > 0

    series = _get(f"{base}/api/jobs/{jid}/series")
    assert len(series["distance"]) == FRAMES and series["distance"][0] is None
    assert series["distance"][-1] == pytest.approx(40.0, abs=0.3)

    last = _get(f"{base}/api/jobs/{jid}/frames/{FRAMES - 1}")
    assert last["obstacle"] and last["level"] == 2 and last["frame"] == FRAMES - 1
    assert len(last["corridor"]["edges"]) == 4 and len(last["corridor"]["edges"][0]) > 10
    box = last["boxes"][0]
    assert box["confirmed"] and box["label"] == "40.0 м"
    assert box["center"][1] == pytest.approx(-40.0, abs=1.0) and min(box["size"]) > 0
    first = _get(f"{base}/api/jobs/{jid}/frames/0")
    # до подтверждения M из N объект — только кандидат (жёлтая рамка)
    assert not first["obstacle"] and not any(b["confirmed"] for b in first["boxes"])

    code, body, headers = _req(f"{base}/api/jobs/{jid}/frames/{FRAMES - 1}/cloud")
    assert code == 200 and len(body) % 16 == 0
    pts = np.frombuffer(body, "<f4").reshape(-1, 4)
    assert len(pts) == int(headers["X-Points"]) and 0 < len(pts) <= pipeline.MAX_VIEW_POINTS
    assert np.isfinite(pts).all()
    assert _req(f"{base}/api/jobs/{jid}/frames/{FRAMES}")[0] == 404

    code, body, headers = _req(f"{base}/api/jobs/{jid}/results.jsonl")
    lines = [json.loads(x) for x in body.decode().splitlines()]
    assert code == 200 and len(lines) == FRAMES and "attachment" in headers["Content-Disposition"]
    assert {"stamp", "frame", "obstacle", "distance", "processing_ms", "core_ms"} <= set(lines[0])

    assert _req(f"{base}/api/jobs/{jid}/mcap")[0] == 409
    code, _ = _post(f"{base}/api/jobs/{jid}/mcap", {})
    assert code == 202
    st = _wait(base, jid, key="mcap")
    assert st["mcap"]["state"] == "done" and st["mcap"]["done"] == FRAMES
    code, body, _ = _req(f"{base}/api/jobs/{jid}/mcap")
    assert code == 200 and body[:5] == b"\x89MCAP"


def test_upload_db3_and_zip(base: str, bags_dir: Path) -> None:
    db3 = next((bags_dir / "synthetic").glob("*.db3"))
    code, body, _ = _req(
        f"{base}/api/upload?name=my%20bag.db3", db3.read_bytes(), "POST", "application/octet-stream"
    )
    assert code == 201, body
    up = json.loads(body)
    assert up["bag"].startswith("uploads:")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in (bags_dir / "synthetic").iterdir():
            z.write(f, f"synthetic/{f.name}")
    code, body, _ = _req(
        f"{base}/api/upload?name=synthetic.zip", buf.getvalue(), "POST", "application/octet-stream"
    )
    assert code == 201, body
    zipped = json.loads(body)
    assert zipped["name"] == "synthetic"
    ids = {b["id"] for b in _get(f"{base}/api/bags")["bags"]}
    assert {up["bag"], zipped["bag"]} <= ids

    code, created = _post(f"{base}/api/jobs", {"bag": zipped["bag"], "max_frames": 5})
    st = _wait(base, created["id"])
    assert st["state"] == "done" and st["total"] == st["done"] == 5

    code, body, _ = _req(
        f"{base}/api/upload?name=x.txt", b"hello", "POST", "application/octet-stream"
    )
    assert code == 400
    code, body, _ = _req(
        f"{base}/api/upload?name=x.zip", b"not a zip", "POST", "application/octet-stream"
    )
    assert code == 400


def test_alarm_events_merge_short_gaps() -> None:
    def rec(k: int, alarm: bool, d: float | None = None) -> dict[str, Any]:
        objs = [{"distance": d, "confirmed": True, "width": 1.0, "height": 1.5, "length": 0.5}]
        return {"frame": k, "obstacle": alarm, "distance": d, "obstacles": objs if alarm else []}

    flags = [0, 1, 1, 0, 1, 0, 0, 0, 0, 1]
    recs = [rec(k, bool(f), 50.0 - k) for k, f in enumerate(flags)]
    ev = pipeline.alarm_events(recs, gap=3)
    assert [(e["start"], e["end"], e["frames"]) for e in ev] == [(1, 4, 3), (9, 9, 1)]
    assert ev[0]["first_distance"] == 49.0 and ev[0]["min_distance"] == 46.0


def test_zip_from_finder_skips_macosx(base: str, bags_dir: Path) -> None:
    """Архив из Finder: __MACOSX/<bag>/._*.db3 сортируется раньше записи — не брать его."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in (bags_dir / "synthetic").iterdir():
            z.write(f, f"synthetic/{f.name}")
            z.writestr(f"__MACOSX/synthetic/._{f.name}", b"\x00\x05\x16\x07junk")
    code, body, _ = _req(
        f"{base}/api/upload?name=finder.zip", buf.getvalue(), "POST", "application/octet-stream"
    )
    assert code == 201, body
    up = json.loads(body)
    assert "__MACOSX" not in up["bag"] and up["name"] == "synthetic"


def test_bad_requests_answer_not_drop(base: str) -> None:
    """Неверный тип поля — 400, а не оборванное соединение; Content-Length < 0 — не виснуть."""
    code, err = _post(f"{base}/api/jobs", {"bag": "bags:synthetic", "max_frames": [1]})
    assert code == 400 and "error" in err

    host, port = base.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as s:
        s.sendall(b"POST /api/jobs HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n")
        head = s.recv(4096)
    assert head.startswith((b"HTTP/1.0 400", b"HTTP/1.1 400")), head


def test_series_has_timeline_tracks(tmp_path: Path) -> None:
    """Ряды для дорожек таймлайна: видимая дальность, время кадра, уверенность."""
    job = pipeline.Job("j", pipeline.BagRef("bags", "x", tmp_path), tmp_path)
    job.records = [
        {
            "stamp": 10.0 + k / 10,
            "frame": k,
            "obstacle": k == 1,
            "distance": 40.0 if k == 1 else None,
            "confidence": 0.7 if k == 1 else 0.0,
            "level": 2 if k == 1 else 0,
            "visible_range": 150.0 + k,
            "processing_ms": 30.0 + k,
            "core_ms": 10.0,
            "calibrated": True,
            "obstacles": [],
        }
        for k in range(3)
    ]
    s = job.series()
    assert s["visible_range"] == [150.0, 151.0, 152.0]
    assert s["frame_ms"] == [30.0, 31.0, 32.0]
    assert s["confidence"] == [0.0, 0.7, 0.0]
    assert s["t"] == pytest.approx([0.0, 0.1, 0.2])
