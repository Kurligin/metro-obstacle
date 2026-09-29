import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from metro_obstacle.results_log import (
    ResultsLog,
    ResultsWriter,
    jsonable,
    per_source,
    results_file,
)


def test_jsonable_handles_numpy_and_nan() -> None:
    d = jsonable(
        {"a": np.float32(1.5), "b": math.nan, "c": [np.int64(3), (1.0, np.inf)], "d": np.arange(2)}
    )
    assert d == {"a": 1.5, "b": None, "c": [3, [1.0, None]], "d": [0, 1]}


def test_writer_one_line_per_frame(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "results.jsonl"
    w = ResultsWriter(str(path))
    w.write({"stamp": 1.0, "frame": 0, "distance": math.nan})
    w.write({"stamp": 1.1, "frame": 1, "distance": 12.5})
    # Строки видны сразу, без закрытия файла: прогон можно оборвать в любой момент.
    lines = path.read_text().splitlines()
    assert [json.loads(s)["frame"] for s in lines] == [0, 1]
    assert json.loads(lines[0])["distance"] is None
    w.close()


def test_rerun_does_not_overwrite_plain_file(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    for run in range(2):
        w = ResultsWriter(str(path))
        w.write({"run": run})
        w.close()
    assert [json.loads(s)["run"] for s in path.read_text().splitlines()] == [0, 1]


def test_directory_gives_file_per_source(tmp_path: Path) -> None:
    t = datetime(2026, 9, 24, 13, 5, 9)
    out = tmp_path / "out"
    spec = str(out) + "/"  # каталога ещё нет — узнаём по косой черте
    assert per_source(spec) and per_source(str(tmp_path)) and not per_source(str(out / "r.jsonl"))
    assert (
        results_file(spec, "doubleT_obstacle", t)
        == out / "results_doubleT_obstacle_20260924-130509.jsonl"
    )
    assert results_file(str(tmp_path), "/sensing/lidar/hesai128/pointcloud", t).name == (
        "results_sensing_lidar_hesai128_pointcloud_20260924-130509.jsonl"
    )
    assert results_file(str(tmp_path / "{name}-{time}.jsonl"), "/lidar_points", t).name == (
        "lidar_points-20260924-130509.jsonl"
    )

    log = ResultsLog(spec)
    assert log.write({"f": 0}, "/lidar_points", t) and not log.write({"f": 1}, "/lidar_points", t)
    first = log.path
    log.new_source()  # следующий bag на ту же ноду — новый файл
    assert log.write({"f": 0}, "doubleT_obstacle", datetime(2026, 9, 24, 13, 6, 0))
    second = log.path
    log.close()
    assert first != second
    assert [len(Path(p).read_text().splitlines()) for p in (first, second)] == [2, 1]


def test_plain_file_keeps_one_file_across_sources(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    log = ResultsLog(str(path))
    log.write({"f": 0}, "/a")
    log.new_source()
    log.write({"f": 0}, "/b")
    log.close()
    assert len(path.read_text().splitlines()) == 2
