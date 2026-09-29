# Сборка и запуск

Всё решение собирается в один Docker-образ: ROS 2 Humble, ядро детектора
(`metro_obstacle_core`) и ROS-пакеты (`metro_obstacle`, `metro_obstacle_msgs`).
Ставить на машину ничего, кроме Docker, не нужно.

## Сборка

```bash
docker build -t metro-obstacle .
```

Что делает сборка: ставит ядро (`pip install ./metro_obstacle_core` с точными версиями
из `docker/constraints.txt`: numpy 1.26.4, numba 0.67.0 — на них проверена сверка со
стендом; lightgbm 4.6.0 — проверяющий классификатор режима `default`, ему нужен
`libgomp1` из apt), проверяет, что `Detector('default')` загрузил модель классификатора
из пакета (иначе сборка падает — в сдаваемом образе откат в `geometry` исключён),
прогревает numba (компиляция ~5 с, кэш остаётся в образе, в `/opt/numba_cache`),
собирает `ros2_ws` (`colcon build`). Образ x86 — 1.24 ГБ, сборка на i5-10600KF — около
1.5 мин (с загруженным базовым образом `ros:humble-ros-base`).

Образ многоархитектурный: собирается и на x86_64 (стенд проверки), и на arm64.
Кэш numba привязан к процессору машины сборки: если образ собран на одной машине, а
запущен на другой, нода один раз перекомпилирует ядра при старте (~5 с, до начала
проигрывания bag — кадры не теряются).
Собрать amd64-образ на Mac с Apple Silicon (медленно, через эмуляцию):

```bash
docker buildx build --platform linux/amd64 -t metro-obstacle:amd64 --load .
```

## Запуск на bag

```bash
docker run --rm --net=host -v /path/to/bags:/data metro-obstacle \
  ros2 launch metro_obstacle detect.launch.py bag:=/data/<bag>
```

Что происходит:

1. Нода прогревает ядро (из кэша numba — доли секунды) и пишет в лог `detector ready`.
2. Только после этого запускается `ros2 bag play` — первые кадры не теряются.
3. Нода сама находит топик `sensor_msgs/PointCloud2` (в данных хакатона это
   `/lidar_points` или `/sensing/lidar/hesai128/pointcloud`), сама определяет ось
   «вперёд», калибруется и на каждый кадр публикует результат. Плоскость полотна —
   по первому пригодному кадру (пустые и почти пустые кадры пропускаются), обычно это
   первый же кадр bag; дальше она уточняется на ходу (20 кадров подряд, затем каждый
   5-й) и заменяется медианой оценок, только если та отличается больше чем на 1° или
   5 см — так старт записи в неудачном месте не портит калибровку
   ([ALGORITHM.md](ALGORITHM.md#21-уточнение-плоскости-на-ходу)). Модель полотна
   прогревается 15 кадров (`warmup_frames = 15` во всех режимах, около 1.5 с при 10 Гц):
   до тех пор `calibrated=false` в статусе и в results, тревоги не подтверждаются. Зачем
   прогрев — в [EXPERIMENTS.md](EXPERIMENTS.md#история-улучшений).
   `warmup_frames`, `calib_frames`, `plane_every` и прочие поля `Params` — параметры
   ядра: они задаются режимом (`MODES` в `params.py`) и переопределяются только через
   Python API ядра (`Detector(mode=..., warmup_frames=...)`). Через launch и
   `params_file` в ядро передаются только `mode`, `verify_model`, `gauge_half_width`,
   `gauge_height` и `max_range`.
4. Результат пишется в `/path/to/bags/out/results_<bag>_<YYYYmmdd-HHMMSS>.jsonl`
   (строка на кадр). Каждый запуск — новый файл, прошлые прогоны не затираются.
5. Когда bag закончился, контейнер завершается сам.

Если каталог с bag нужно оставить только на чтение, результат выносится отдельно
(так и проверялось; `--cpus` — по желанию, чтобы не занимать всю машину; можно
запускать и не от root — `-u "$(id -u):$(id -g)"`, тогда файлы результатов будут
принадлежать вам):

```bash
docker run --rm --net=host --cpus=10 \
  -v /path/to/bags:/data/bags:ro -v "$PWD/out":/data/out \
  metro-obstacle ros2 launch metro_obstacle detect.launch.py bag:=/data/bags/<bag>
```

## Живой лидар или свой `ros2 bag play`

Без аргумента `bag` нода просто ждёт топик PointCloud2:

```bash
docker run --rm --net=host --ipc=host -v "$PWD/out":/data/out metro-obstacle
# в другом терминале, на хосте с ROS 2 Humble:
ros2 bag play /path/to/bags/<bag>
```

Bag можно проигрывать один за другим на одну и ту же запущенную ноду, в том числе с
разными топиками. Если кадров нет дольше 2 с, а издатели сменились (прошлый bag
закончился), нода снимает подписку и снова ищет топик PointCloud2 (с заданным
`input_topic` — ждёт на нём же). Ядро сбрасывается (калибровка, ось, история — заново;
в логе `detector reset: ...`) при новом источнике, скачке времени кадров назад больше
1 с или вперёд больше 5 с и при смене `frame_id`. Результаты каждого источника — в
своём файле `results_<топик>_<время>.jsonl`.

`--ipc=host` нужен, когда издатель — вне контейнера: Fast DDS на одном хосте
передаёт данные через разделяемую память, и без общего `/dev/shm` большие облака
не доходят. `ROS_DOMAIN_ID` у хоста и контейнера должен совпадать.

## Аргументы launch

| аргумент | по умолчанию | смысл |
|---|---|---|
| `bag` | — | путь к bag; пусто — не проигрывать, слушать живой топик |
| `rate` | `1.0` | скорость проигрывания bag |
| `loop` | `false` | проигрывать bag по кругу |
| `exit_after_bag` | `true` | завершить всё, когда bag закончился (при `loop` не срабатывает) |
| `input_topic` | авто | топик PointCloud2; пусто — первый найденный топик этого типа |
| `mode` | `default` | `default` (геометрия + проверяющий классификатор; без классификатора — `geometry`) / `geometry` (только геометрия) / `strict` (геометрия без раздвоенных лучей) / `soft` (2 из 5, дальше по крупным объектам) |
| `verify_model` | модель из пакета | файл модели классификатора (текст LightGBM booster); путь внутри контейнера |
| `forward_axis` | `auto` | ось «вперёд» в осях лидара: `auto`, `x`, `-x`, `y`, `-y` |
| `results_path` | `/data/out/` | куда писать JSON Lines (см. «Как смотреть результат»); `none` — не писать |
| `fake_core` | `false` | фиктивное ядро — только для проверки ROS-обвязки |
| `rviz` | `false` | запустить RViz (нужен образ с RViz, см. ниже) |
| `params_file` | `config/default.yaml` | файл параметров ноды |

Пустой аргумент = значение из `config/default.yaml`. Там же параметры, которых нет
среди аргументов launch:

| параметр | по умолчанию | смысл |
|---|---|---|
| `gauge_half_width` | `1.05` | полуширина габарита от оси пути, м (передаётся в ядро) |
| `gauge_height` | `3.0` | высота габарита над полотном, м (ядро и отрисовка коридора) |
| `max_range` | `200.0` | дальше ось пути не строится, м (ядро) |
| `results_tag` | пусто | имя источника в имени файла результатов; пусто — имя топика (с `bag:=` launch подставляет имя bag) |
| `input_reliability` | `auto` | `auto` — как у издателя; `reliable`; `best_effort` |
| `input_depth` | `5` | глубина очереди входа (KEEP_LAST). Кадр обрабатывается обычно за 20–40 мс, p95 до ~60 мс на 921 600 точек, при периоде 100 мс, так что очередь в норме пуста; глубина спасает кадры, когда источник отдаёт их пачками (`ros2 bag play` под нехваткой памяти). `1` — только самый свежий кадр |
| `publish_tf` | `true` | публиковать TF `<frame облака>` → `lidar_link` |

Свой файл параметров — через том и `params_file:=`:

```bash
docker run --rm --net=host -v /path/to/bags:/data/bags:ro -v "$PWD/out":/data/out \
  -v "$PWD/my.yaml":/params.yaml:ro metro-obstacle \
  ros2 launch metro_obstacle detect.launch.py bag:=/data/bags/<bag> params_file:=/params.yaml
```

Числа в YAML для этих параметров — с точкой (`1.2`, не `1`): тип параметра — double.

## Режим и классификатор

По умолчанию нода работает в режиме `default`: геометрия с пониженными порогами +
проверяющий классификатор LightGBM (модель `models/verifier.txt` из пакета ядра, порог
0.7978) + подтверждение 3 из 5. В логе при старте:

```
verifier on: /usr/local/lib/python3.10/dist-packages/metro_obstacle_core/models/verifier.txt (threshold 0.7978)
detector ready: mode=default forward_axis=auto core=core; ...
```

Только геометрия, без классификатора: `mode:=geometry` (или `MODE=geometry` для compose).

Своя модель (тот же формат и те же 9 признаков; порог остаётся порогом режима):

```bash
docker run --rm --net=host -v /path/to/bags:/data/bags:ro -v "$PWD/out":/data/out \
  -v "$PWD/my_model.txt":/model.txt:ro metro-obstacle \
  ros2 launch metro_obstacle detect.launch.py bag:=/data/bags/<bag> verify_model:=/model.txt
```

Если модель не загрузилась (нет `lightgbm`, нет файла, файл не читается или рассчитан на
другое число признаков), нода не падает: в логе WARN `verifier off: <причина>; running mode
'geometry'`, в `/diagnostics` — WARN `verifier unavailable, running geometry`, в статусе и
в `results_*.jsonl` поле `mode` = `geometry`. `verify_model` в режимах без классификатора
игнорируется (WARN в логе).

## Что публикуется

| топик | тип | содержание |
|---|---|---|
| `/obstacle_detection/status` | `metro_obstacle_msgs/ObstacleStatus` | на каждый кадр: `obstacle`, `distance` (м вдоль пути), `confidence`, `level` (0 свободно / 1 кандидат / 2 подтверждено), `visible_range`, `processing_ms`, `mode`, `calibrated` |
| `/obstacle_detection/obstacles` | `metro_obstacle_msgs/ObstacleArray` | объекты: дистанция, смещение от оси, высота над рельсом, размеры, число точек, confidence, confirmed, центр |
| `/obstacle_detection/detections` | `vision_msgs/Detection3DArray` | те же объекты в стандартном типе (бокс, `score` = confidence) |
| `/obstacle_detection/markers` | `visualization_msgs/MarkerArray` | коридор габарита, боксы объектов, подписи дистанций, `CLEAR` / `OBSTACLE 56.3 m` |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | частота входа и выхода, время обработки (последнее и p95), пропущенные и сбойные кадры, длина прослеженной оси, калибровка, время прогрева ядра |

Все выходы — в `frame_id` входного облака. Нода публикует тождественный статический
TF `<frame облака>` → `lidar_link`, поэтому в RViz фиксированный фрейм всегда
`lidar_link`, как бы ни назывался фрейм лидара.

Подписка на облако — очередь KEEP_LAST глубины `input_depth` (по умолчанию 5). Кадр
обрабатывается обычно за 20–40 мс, p95 до ~60 мс на 921 600 точек, при периоде 100 мс, поэтому в норме очередь пуста; глубина
нужна, чтобы не терять кадры, когда источник отдаёт их пачками. Если ядро всё же не
успевает, старые кадры вытесняются (их число — `frames_dropped` в `/diagnostics`).

## Как смотреть результат

`results_path` бывает трёх видов:

- каталог (`/data/out/` — по умолчанию в образе; признак — `/` на конце или существующий
  каталог): на каждый источник свой файл `results_<имя bag или топика>_<YYYYmmdd-HHMMSS>.jsonl`;
- шаблон с `{name}` и/или `{time}`, например `results_path:=/data/out/run_{name}.jsonl`;
- обычный файл: дописывается, повторный запуск прошлые строки не стирает.

Файл результатов — одна строка на обработанный кадр:

```json
{"stamp": 946692903.133367, "frame": 5, "obstacle": true, "distance": 56.3,
 "confidence": 0.9, "level": 2, "visible_range": 121.0, "mode": "default",
 "calibrated": true, "obstacles": [{"distance": 56.3, "lateral": 0.2, "height": 1.7,
 "length": 0.4, "width": 0.6, "points": 38, "confidence": 0.9, "confirmed": true,
 "position": [0.1, -56.9, -0.1], "z_extent": 1.6}], "processing_ms": 27.1,
 "core_ms": 21.4}
```

`stamp` — время кадра из заголовка PointCloud2, `frame` — номер кадра от начала
источника, `processing_ms` — полное время кадра в ноде (разбор облака + детектор),
`core_ms` — только детектор. У объекта `height` — верх над головками рельсов,
`z_extent` — вертикальный размер его точек (высота рамки; у висящего кабеля мал),
`position` — центр рамки. `null` — значения нет (например, `distance`, когда путь
свободен).

Быстрая сводка по файлу (на хосте нужен только Python 3):

```bash
python3 - out/results_doubleT_obstacle_*.jsonl <<'PY'
import json, statistics, sys
r = [json.loads(line) for line in open(sys.argv[1])]
ms = sorted(x["processing_ms"] for x in r)
print(f"кадров {len(r)}, тревога в {sum(x['obstacle'] for x in r)}, "
      f"p50 {statistics.median(ms):.1f} мс, p95 {ms[int(0.95 * (len(ms) - 1))]:.1f} мс")
for a, b in zip(r, r[1:]):
    if a["obstacle"] != b["obstacle"]:
        print(f"кадр {b['frame']}: {'тревога' if b['obstacle'] else 'снята'} {b['distance']}")
PY
```

Остановка вручную — Ctrl+C (или `docker stop`): нода закрывает файл результатов и
завершается чисто, в логе launch — `process has finished cleanly`.

Топики — изнутри работающего контейнера (`--name` в `docker run`):

```bash
docker exec -it <container> bash -lc 'source /ws/install/setup.bash && \
  ros2 topic echo /obstacle_detection/status'
docker exec -it <container> bash -lc 'source /ws/install/setup.bash && \
  ros2 topic hz /obstacle_detection/status'
docker exec -it <container> bash -lc 'source /ws/install/setup.bash && \
  ros2 topic echo /diagnostics'
```

## RViz

Конфиг — `ros2_ws/src/metro_obstacle/config/metro_obstacle.rviz`: облако (цвет по
интенсивности), коридор габарита (зелёный — свободно, жёлтый — кандидат, красный —
препятствие), боксы объектов (красные — подтверждённые, жёлтые — кандидаты), подпись
над лидаром. Вид — сзади-сверху по ходу движения (в данных хакатона «вперёд» — это −Y
лидара).

**Linux (X11)** — демо-сервис compose: детектор + bag по кругу + RViz. Каталог с
bag монтируется только на чтение.

```bash
xhost +local:docker
BAGS_DIR=/path/to/bags BAG=doubleT_obstacle docker compose up demo
```

Демо-образ собирается из того же `Dockerfile` на базе `osrf/ros:humble-desktop`
(только amd64). На arm64 — `DEMO_BASE_IMAGE=ros:humble-ros-base`, RViz поставится
из apt. Без GPU в контейнере: `LIBGL_ALWAYS_SOFTWARE=1`. Результат пишется в
`./out/run/results_<bag>_<время>.jsonl` (`OUT_DIR`).

Если ROS 2 Humble стоит на хосте, RViz можно запустить там, а детектор — в
контейнере с `--net=host --ipc=host`:

```bash
rviz2 -d ros2_ws/src/metro_obstacle/config/metro_obstacle.rviz
```

**macOS — ограничения.** В Docker Desktop нет X-сервера и нет аппаратного OpenGL
для контейнеров. Через XQuartz (`xhost + 127.0.0.1`,
`DISPLAY=host.docker.internal:0`, `LIBGL_ALWAYS_SOFTWARE=1`) RViz запускается
только с программной отрисовкой — медленно и не на всех версиях. `--net=host`
в Docker Desktop связывает контейнер с Linux-VM, а не с самим Mac, так что RViz,
установленный на Mac, топики контейнера не увидит. Для демо на Mac надёжнее
записать выход в bag и смотреть его на Linux-машине. Сам детектор на Mac
работает полностью (`docker run ...` выше).

## Без ROS: запись → MCAP для Lichtblick / Foxglove

`bench/to_mcap.py` прогоняет ядро (режим `default`, как в ноде) по записи ROS 2
(sqlite `.db3`) без ROS и пишет `.mcap` со стандартными сообщениями ROS 2 (ros2msg /
cdr) — удобно для демо на Mac. Нужны ядро (`pip install ./metro_obstacle_core`) и пакеты
`mcap`, `mcap-ros2-support`, `rosbags`. Маркеры строятся тем же модулем, что и в ноде
(`ros2_ws/src/metro_obstacle/metro_obstacle/viz.py`), разбор облака — тем же `cloud.py`.

```bash
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m bench.to_mcap \
  /path/to/bags/doubleT_obstacle --max-frames 100
# → out/doubleT_obstacle.mcap; ещё: --start N, --out файл.mcap, --topic /топик_облака
```

Вход — каталог bag или файл `.db3`; облако — первый PointCloud2 (или `--topic`),
поле `ring` не обязательно (восстанавливается по углу места, как в ноде). В MCAP:
`/lidar_points` (исходное облако), `/metro_obstacle/markers` (коридор, рамки, дистанции),
`/metro_obstacle/status` (`std_msgs/String`, JSON `{obstacle, distance, calibrated}`),
`/metro_obstacle/in_corridor` (точки в габарите выше полотна). Раскладка —
`docs/lichtblick_layout.json` (Layouts → Import): 3D-вид сзади-сверху по ходу (−Y),
облако по интенсивности, справа — статус.

## Веб-прототип

```bash
BAGS_DIR=/path/to/bags docker compose up --build web   # → http://localhost:8080
```

Страница: выбрать запись из `BAGS_DIR` (тот же каталог, что у `detect`) или загрузить
`.db3` / `.zip` с каталогом записи → «Обработать» (поле «Кадров» — обработать только
первые N) → прогресс → 3D-вид кадра, статус, сводка, график дистанции, события тревоги,
скачивание `results.jsonl` и `.mcap`. Загрузки и результаты — в `WEB_DATA_DIR`
(по умолчанию `./out/web`), порт — `WEB_PORT`. Задачи живут в памяти сервера: после
перезапуска контейнера запись нужно обработать заново. Без Docker:
`BAGS_DIR=/path/to/bags python -m web.server` (нужны ядро, `rosbags`, `mcap-ros2-support`).

HTTP API (для скрипта вместо страницы):

```bash
curl -s localhost:8080/api/bags                                      # записи
curl -s -X POST localhost:8080/api/jobs -d '{"bag": "bags:doubleT_obstacle"}'   # → {"id": ...}
curl -s localhost:8080/api/jobs/<id>                  # состояние, прогресс, сводка, события
curl -s localhost:8080/api/jobs/<id>/results.jsonl    # результаты по кадрам (как у ноды)
curl -s -X POST --data-binary @bag.db3 'localhost:8080/api/upload?name=bag.db3'  # загрузка
```

Ещё: `GET /api/jobs/<id>/series` (ряды по кадрам), `/frames/<k>` (объекты, коридор,
рамки кадра), `/frames/<k>/cloud` (облако float32 x, y, z, intensity), `POST` и `GET
/api/jobs/<id>/mcap` (собрать и скачать MCAP). В `results.jsonl` `processing_ms` — чтение
кадра из записи + разбор + ядро, `core_ms` — ядро.

## Проверка без ядра

Чтобы проверить ROS-обвязку (топики, маркеры, лог) без настоящего детектора:
`fake_core:=true` или `-e METRO_FAKE_CORE=1`. Фиктивное ядро ищет «препятствия»
в прямом коридоре и годится только для этого. Без него и без установленного ядра
нода завершается с понятной ошибкой, а не подменяет результат молча.

## Тесты ROS-пакета

В образе, с ROS и настоящим ядром (разбор облака, сборка сообщений, сквозной прогон
синтетического кадра через ядро):

```bash
docker run --rm metro-obstacle bash -c \
  'cd /ws/src/metro_obstacle && python3 -m pytest -q -p no:cacheprovider test'
```

Без Docker (только модули без ROS; тесты сообщений пропускаются):

```bash
python3 -m pytest -q ros2_ws/src/metro_obstacle/test
```
