# Образ детектора: ROS 2 Humble + ядро + ROS-пакеты.
#   docker build -t metro-obstacle .
#   docker run --rm --net=host -v /path/to/bags:/data metro-obstacle \
#     ros2 launch metro_obstacle detect.launch.py bag:=/data/<bag>
# Демо-образ с RViz собирается из этого же файла с другой базой (см. docker-compose.yml).
ARG BASE_IMAGE=ros:humble-ros-base
FROM ${BASE_IMAGE}
# Доп. apt-пакеты (демо-образ ставит сюда ros-humble-rviz2). В основном образе пусто.
ARG EXTRA_APT=""

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# vision_msgs — выход detections; sqlite3-плагин rosbag2 — для bag хакатона
# (в ros-base он уже есть, пакет указан явно, чтобы не зависеть от состава базы).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-pytest \
        libgomp1 \
        ros-humble-vision-msgs \
        ros-humble-rosbag2-storage-default-plugins \
        ${EXTRA_APT} \
    && rm -rf /var/lib/apt/lists/*

# Ядро детектора. pip из Ubuntu 22.04 (22.0) собирает pyproject-пакеты как «UNKNOWN» —
# обновляем только pip; системный setuptools не трогаем, на нём собирает colcon.
# Зависимости ядра — точные версии из docker/constraints.txt (numpy 1.26.4: сообщения
# ROS собраны под numpy 1.x; numba 0.67.0; lightgbm 4.6.0 — классификатор режима
# default, ему нужен libgomp1): на них проверена сверка со стендом. Модель
# классификатора (models/verifier.txt) ставится вместе с пакетом ядра; сборка падает,
# если режим default в образе не загрузил её.
COPY docker/constraints.txt /opt/constraints.txt
COPY metro_obstacle_core /opt/metro_obstacle_core/
RUN python3 -m pip install --upgrade "pip>=23" \
    && python3 -m pip install -c /opt/constraints.txt /opt/metro_obstacle_core \
    && python3 -c "import numpy, numba, lightgbm, metro_obstacle_core as c; \
print('metro_obstacle_core', c.__version__, 'numpy', numpy.__version__, 'numba', numba.__version__, \
'lightgbm', lightgbm.__version__); d = c.Detector('default'); print('verifier', d.verifier_note); \
assert d.verifier is not None"

# Прогрев numba при сборке: скомпилированные ядра ложатся в дисковый кэш
# NUMBA_CACHE_DIR, и нода стартует без компиляции. Каталог доступен на запись всем:
# с `docker run -u <uid>` кэш читается, а на машине с другим CPU numba может его
# дописать. numba сверяет кэш с mtime исходника, а слой образа хранит mtime с
# точностью до секунды — поэтому сначала округляем mtime исходников до целых секунд,
# иначе в контейнере кэш считался бы устаревшим. Кэш привязан к процессору: на машине
# с другим CPU numba один раз перекомпилирует ядра при старте ноды — до проигрывания bag.
ENV NUMBA_CACHE_DIR=/opt/numba_cache
RUN pkg="$(python3 -c 'import metro_obstacle_core, os; print(os.path.dirname(metro_obstacle_core.__file__))')" \
    && find "$pkg" -name '*.py' -exec touch -d "@$(date +%s)" {} + \
    && mkdir -p "$NUMBA_CACHE_DIR" \
    && python3 -c "import metro_obstacle_core as c; print(f'numba warm-up {c.warmup():.1f} s')" \
    && python3 -c "import metro_obstacle_core as c; print(f'numba from cache {c.warmup():.2f} s')" \
    && chmod -R a+rwX "$NUMBA_CACHE_DIR"

WORKDIR /ws
COPY ros2_ws/src ./src
RUN source /opt/ros/humble/setup.bash \
    && MAKEFLAGS=-j8 colcon build --parallel-workers 4 --event-handlers console_direct- \
    && rm -rf build log

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# /data — bag на вход и результаты на выход: каталог /data/out/, на каждый bag (или
# топик) свой файл results_<имя>_<YYYYmmdd-HHMMSS>.jsonl — повторный запуск не
# затирает прошлый. Логи ROS — в /tmp: с `docker run -u <uid>` домашний каталог
# (/root) недоступен на запись.
ENV METRO_RESULTS_PATH=/data/out/ \
    ROS_LOG_DIR=/tmp/ros_log
VOLUME ["/data"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "metro_obstacle", "detect.launch.py"]
