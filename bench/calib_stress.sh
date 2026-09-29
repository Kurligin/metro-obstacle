#!/bin/bash
# Проверка уточнения плоскости полотна на ходу на расширенной записи (new_data), ПК.
#   1) старты с разных файлов (111/130/145/148), тревоги в 148–157 — до и после;
#   2) bench.newdata по всей записи, 12 отрезков = 12 разных точек старта, до и после.
# «До» — вариант core_default_plane0 (плоскость только по первому кадру), «после» —
# core_default. Запуск: bash bench/calib_stress.sh <каталог new_data>  (из корня репо).
# Долго: держать ssh-сессию или запускать через setsid nohup … &.
set -e
if [ $# -lt 1 ] || [ ! -d "$1" ]; then
    echo "usage: bash bench/calib_stress.sh <каталог new_data>" >&2
    [ $# -ge 1 ] && echo "нет каталога: $1" >&2
    exit 2
fi
ND="$(cd "$1" && pwd)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/out/calib_stress"
mkdir -p "$LOG"
cd "$ROOT"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$PY" -m bench.starts "$ND" 111,130,145,148 core_default,core_default_plane0 148-158 \
    | tee "$LOG/starts.log"

# по очереди, 12 отрезков на вариант (nseg = LCT_WORKERS / число вариантов); результаты
# копятся в out/newdata/newdata.json по имени варианта, итог — отчётом по обоим
for v in core_default_plane0 core_default; do
    LCT_WORKERS=12 "$PY" -m bench.newdata run "$ND" "$v" > "$LOG/newdata_$v.log"
done
"$PY" -m bench.newdata report | grep "==" | tee "$LOG/newdata.log"
