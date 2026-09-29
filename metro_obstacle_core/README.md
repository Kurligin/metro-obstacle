# metro_obstacle_core

Ядро детектора посторонних объектов в габарите поезда метро по 3D-лидару (без ROS).

```python
from metro_obstacle_core import Detector, warmup

warmup()  # прогрев numba (первый раз — компиляция, дальше — дисковый кэш)
det = Detector(mode="default", forward_axis="auto")  # default | geometry | strict | soft
res = det.process(xyz, ring, intensity, stamp)       # xyz (N, 3) float32 в осях сообщения
print(res.mode, res.level, res.distance, [o.to_json() for o in res.obstacles])
print(det.verifier_note)  # "on: <модель> (threshold 0.7978)" или причина отката
```

Режимы: `default` — геометрия с пониженными порогами + проверяющий классификатор
LightGBM + подтверждение 3 из 5 (основной); `geometry` — только геометрия; `strict`,
`soft` — её варианты. Модель классификатора ставится вместе с пакетом
(`models/verifier.txt`); свой файл — `Detector(mode="default", verify_model="путь")`. Если
`lightgbm` или модель недоступны, `default` работает как `geometry`: предупреждение в лог,
причина — в `det.verifier_error`, фактический режим — в `det.mode` и `res.mode`. Любое поле
`Params` (`params.py`) можно переопределить аргументом: `Detector(mode="default",
plane_every=0)`.

Установка: `pip install ./metro_obstacle_core`. Python ≥ 3.10, numpy, numba, lightgbm
(для lightgbm на Linux нужен `libgomp1`). Тесты: `pytest tests` из корня репозитория.
Алгоритм и параметры — [docs/ALGORITHM.md](../docs/ALGORITHM.md).
