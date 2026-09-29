"""HTTP-сервер веб-прототипа (стандартная библиотека, без внешних зависимостей).

    python -m web.server            # http://localhost:8080

Переменные окружения:
    BAGS_DIR      каталог с записями (только чтение), по умолчанию ./data
    WEB_DATA_DIR  загрузки и результаты задач, по умолчанию ./out/web
    WEB_HOST, WEB_PORT  адрес, по умолчанию 0.0.0.0:8080

API (всё JSON, кроме облака и файлов):
    GET  /api/bags                       записи в BAGS_DIR и загруженные
    POST /api/upload?name=<файл>         тело — .db3 или .zip с каталогом записи
    POST /api/jobs {bag, max_frames}     запустить обработку
    GET  /api/jobs/<id>                  состояние, прогресс, сводка, события тревоги
    GET  /api/jobs/<id>/series           ряды по кадрам (дистанция, уровень, время)
    GET  /api/jobs/<id>/frames/<k>       кадр: статус, объекты, коридор, рамки
    GET  /api/jobs/<id>/frames/<k>/cloud облако кадра: float32 x, y, z, intensity
    GET  /api/jobs/<id>/results.jsonl    результаты по кадрам (формат как у ноды)
    POST /api/jobs/<id>/mcap             собрать MCAP для Lichtblick
    GET  /api/jobs/<id>/mcap             скачать MCAP
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from web import pipeline

STATIC = Path(__file__).resolve().parent / "static"
ROOT = Path(__file__).resolve().parents[1]

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("font/woff2", ".woff2")


class App:
    """Состояние сервера: корни записей и задачи."""

    def __init__(self, bags_dir: Path, data_dir: Path) -> None:
        self.uploads = data_dir / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.roots = {"bags": bags_dir, "uploads": self.uploads}
        self.jobs = pipeline.JobManager(data_dir, log=lambda s: print(s, flush=True))


class Handler(BaseHTTPRequestHandler):
    server_version = "metro-obstacle-web"
    app: App  # задаётся в make_server

    # ------------------------------------------------------------------ ответы

    def log_message(self, format: str, *args: Any) -> None:
        # Опрос прогресса идёт часто — в лог только ошибки.
        if len(args) > 1 and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)

    def _send(
        self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data: Any, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json({"error": message}, code)

    def _file(self, path: Path, ctype: str, download: str | None = None) -> None:
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                self.wfile.write(chunk)

    def _body_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n < 0:  # rfile.read(-1) ждал бы закрытия соединения клиентом
            raise ValueError("некорректный Content-Length")
        data = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if not isinstance(data, dict):
            raise ValueError("ожидался JSON-объект")
        return data

    def _job(self, jid: str) -> pipeline.Job | None:
        try:
            return self.app.jobs.get(jid)
        except KeyError:
            self._error(404, f"нет задачи {jid}")
            return None

    # ------------------------------------------------------------------ маршруты

    def do_GET(self) -> None:
        url = urlparse(self.path)
        p = unquote(url.path)
        try:
            if p == "/" or p == "/index.html":
                return self._static("index.html")
            if p.startswith("/static/"):
                return self._static(p[len("/static/") :])
            if p == "/api/bags":
                return self._json({"bags": pipeline.list_bags(self.app.roots)})
            if p == "/api/jobs":
                return self._json({"jobs": [j.status() for j in self.app.jobs.list()]})
            m = re.fullmatch(r"/api/jobs/([\w-]+)(/.*)?", p)
            if m:
                return self._get_job(m.group(1), m.group(2) or "")
            return self._error(404, "не найдено")
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as e:  # noqa: BLE001 — ответить 500, а не оборвать соединение
            return self._internal(e)

    def _internal(self, e: Exception) -> None:
        self.log_error("%s: %s", type(e).__name__, e)
        try:
            self._error(500, f"внутренняя ошибка: {type(e).__name__}")
        except OSError:
            pass  # заголовки уже ушли или клиент отключился

    def _get_job(self, jid: str, rest: str) -> None:
        job = self._job(jid)
        if job is None:
            return
        if rest == "":
            return self._json(job.status())
        if rest == "/series":
            return self._json(job.series())
        m = re.fullmatch(r"/frames/(\d+)(/cloud)?", rest)
        if m:
            k = int(m.group(1))
            try:
                if m.group(2):
                    cloud = job.cloud(k)
                    return self._send(
                        200,
                        cloud.tobytes(),
                        "application/octet-stream",
                        {"X-Points": str(len(cloud))},
                    )
                return self._json(job.frame(k))
            except KeyError:
                return self._error(404, f"кадр {k} ещё не обработан")
        if rest == "/results.jsonl":
            if not job.results_path.exists():
                return self._error(409, "результатов ещё нет")
            name = f"results_{pipeline.safe_name(job.bag.name)}.jsonl"
            return self._file(job.results_path, "application/x-ndjson", name)
        if rest == "/mcap":
            if job.mcap_state != "done":
                return self._error(409, "MCAP ещё не собран")
            return self._file(job.mcap_path, "application/octet-stream", job.mcap_path.name)
        return self._error(404, "не найдено")

    def do_POST(self) -> None:
        url = urlparse(self.path)
        p = unquote(url.path)
        try:
            if p == "/api/upload":
                return self._upload(parse_qs(url.query))
            if p == "/api/jobs":
                data = self._body_json()
                bag = pipeline.resolve_bag(self.app.roots, str(data.get("bag", "")))
                mf = data.get("max_frames")
                max_frames = int(mf) if mf not in (None, "", 0) else None
                if max_frames is not None and max_frames < 1:
                    raise ValueError("max_frames должен быть больше 0")
                job = self.app.jobs.submit(bag, max_frames)
                return self._json({"id": job.id}, 201)
            m = re.fullmatch(r"/api/jobs/([\w-]+)/mcap", p)
            if m:
                job = self._job(m.group(1))
                if job is None:
                    return None
                self.app.jobs.submit_mcap(job)
                return self._json(job.status()["mcap"], 202)
            return self._error(404, "не найдено")
        except (ValueError, TypeError) as e:
            return self._error(400, str(e))
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as e:  # noqa: BLE001
            return self._internal(e)

    def _upload(self, query: dict[str, list[str]]) -> None:
        name = (query.get("name") or [""])[0]
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("пустая загрузка")
        path = pipeline.save_upload(self.rfile, length, name, self.app.uploads)
        ref = pipeline.resolve_bag(
            self.app.roots, f"uploads:{path.relative_to(self.app.uploads).as_posix()}"
        )
        return self._json({"bag": ref.id, "name": ref.name}, 201)

    def _static(self, rel: str) -> None:
        path = (STATIC / rel).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            return self._error(404, "не найдено")
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/"):
            ctype += "; charset=utf-8"
        return self._file(path, ctype)


def make_server(host: str, port: int, bags_dir: Path, data_dir: Path) -> ThreadingHTTPServer:
    app = App(bags_dir, data_dir)
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.app = app  # type: ignore[attr-defined]
    return server


def main() -> int:
    bags = Path(os.environ.get("BAGS_DIR", ROOT / "data")).expanduser()
    data = Path(os.environ.get("WEB_DATA_DIR", ROOT / "out" / "web")).expanduser()
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "8080"))
    server = make_server(host, port, bags, data)
    shown = "localhost" if host in ("0.0.0.0", "") else host
    print(f"metro-obstacle web: http://{shown}:{port}  (bags: {bags}, data: {data})", flush=True)
    # Прогрев ядра заранее: первая обработка не ждёт компиляцию numba.
    threading.Thread(target=pipeline.warmup, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.app.jobs.shutdown()  # type: ignore[attr-defined]
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
