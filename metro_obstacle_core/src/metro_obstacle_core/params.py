"""Параметры детектора и режимы.

Режимы — ровно конфиги стенда, по которым считались цифры экспериментов:
    default  = геометрия с пониженными порогами + проверяющий классификатор + 3 из 5
               (стенд: «soft|edge_reject=1,end_guard=0.0,min_pts=2,min_rings=1,
               margin_k=0.001,confirm=3/5» + модель full|rand); без модели — geometry;
    geometry = «Z+S c3»       — чистая зона + раздвоенные лучи, 3 из 5, запас 0.004/м
               (прежний основной режим, только геометрия);
    strict   = «Z c3»         — без раздвоенных лучей: минимум ложных тревог;
    soft     = «Z c2 m0.002»  — 2 из 5, запас 0.002/м: дальше по крупным объектам, больше ложных.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from metro_obstacle_core.verifier import DEFAULT_THRESHOLD


@dataclass(frozen=True)
class Params:
    h_min: float = 0.08  # порог высоты над опорным профилем полотна, м (+0.0015·s)
    min_pts: int = 3  # точек в обычном кластере
    min_rings: int = 2  # колец в обычном кластере
    cell_s: float = 0.6  # клетка кластеризации вдоль пути, м
    cell_lat: float = 0.35  # клетка поперёк пути, м
    confirm: tuple[int, int] = (3, 5)  # подтверждение: M кадров из N
    margin_k: float = 0.004  # сужение габарита на метр дальности (запас на ошибку оси)
    # «чистая зона»: верх габарита вблизи в норме пуст — там хватает 2 точек
    clean_h: float | None = 0.5  # нижняя граница зоны над опорой, м (None — выкл.)
    clean_smax: float = 60.0
    clean_min_pts: int = 2
    split: bool = True  # раздвоенные лучи (ближнее отражение в габарите) — сильные точки
    gauge_half_width: float = 1.05  # полуширина габарита, м
    gauge_height: float = (
        3.0  # высота габарита, м: над полотном (как на стенде) или над головкой рельса
    )
    # True — верх габарита = полотно + уровень головок рельсов + gauge_height (как в ТЗ:
    # «3.0 м от головки рельса»); False — полотно + gauge_height, как на стенде
    gauge_from_rail_head: bool = False
    max_range: float = 200.0  # дальше ось не строится, м
    # модель полотна: разогрев и скорость переобучения (по умолчанию — как на стенде)
    warmup_frames: int = 0  # до стольких обновлений профиля тревоги не подтверждаются
    floor_alpha: float = 0.1  # вес нового кадра в EMA профиля
    floor_fast: float | None = None  # порог смены профиля, м: выше — учимся с весом 0.5
    dil_k: float = 0.0  # доп. расширение профиля вбок, м на метр дальности (ошибка оси)
    delta_clip: float = 0.3  # предел поправки высоты по рельсам вдали Δ(s), м
    # подтверждение: окно сопоставления |Δs| < gate_a + gate_b·s (м) и мягче вдали
    gate_a: float = 2.0
    gate_b: float = 0.02
    confirm_far: tuple[int, int] | None = None  # M из N дальше far_s (None — как confirm)
    far_s: float = 60.0
    axis_fit: str = "cubic"  # ось вдали: "cubic" (RANSAC-кубика) | "smooth" (робастное сглаживание)
    smooth_lam: float = 2000.0
    smooth_gap: float = 30.0
    # за концом оси: коридор по касательной, тревога только на компактный изолированный объект
    wall_ext: bool = False  # ось за кубикой — по поучастковым оценкам стен
    edge_reject: bool = False  # отбрасывать кластеры, продолжающиеся за край габарита (стены)
    edge_smin: float = 30.0  # ближе этой дальности проверку не делать, м
    verify_thr: float | None = None  # порог проверяющего классификатора (None — выкл.)
    verify_smin: float = 0.0  # проверять кластеры не ближе этой дальности, м
    verify_model: str | None = None  # файл модели классификатора (None — модель из пакета)
    end_guard: float = (
        3.0  # последние N м оси не проверять (стена «в лоб» в конце видимого участка)
    )
    far_ext: float = 0.0  # сколько метров за концом оси проверять (0 — выкл.)
    far_h_min: float = 0.35  # высота над продолженным полом, м
    far_max_len: float = 2.0  # максимальная длина объекта вдоль пути, м
    speed_legacy: bool = False  # оценка скорости как в прототипе стенда (только для сверки)
    # калибровка установки: по нескольким кадрам и с выравниванием облака по полотну
    calib_frames: int = 1  # сколько первых кадров копить для плоскости полотна
    level: bool = False  # поворачивать облако так, чтобы полотно стало горизонтальным
    # Уточнение плоскости полотна на ходу (только при calib_frames=1 и level=False): после
    # калибровки по первому кадру плоскость оценивается ещё на plane_dense кадрах подряд,
    # дальше — на каждом plane_every-м; плоскость = покомпонентная медиана последних
    # plane_buf оценок (нормаль перенормируется). Старт в неудачном месте (стрелка,
    # уклон) больше не портит плоскость навсегда, а медиана меняет её плавно.
    # plane_every=0 — плоскость только по первому кадру (как на стенде).
    plane_every: int = 5
    plane_dense: int = 20
    plane_buf: int = 200
    # True — после plane_buf оценок плоскость замораживается (крепление лидара жёсткое:
    # одна плоскость на поездку); False — скользящая медиана последних plane_buf оценок
    plane_freeze: bool = False
    # гистерезис: плоскость заменяется медианой, только если та отличается от текущей
    # больше чем на plane_tol_deg градусов или plane_tol_d м (0 — менять всегда). Удачный
    # старт так ведёт себя как калибровка по первому кадру, неудачный — исправляется
    plane_tol_deg: float = 1.0
    plane_tol_d: float = 0.05
    # запас по ширине: margin0 + margin_k·(s − margin_s0)⁺
    margin0: float = 0.05
    margin_s0: float = 0.0
    # верх габарита вдали опускается на top_k·(s − top_s0)⁺ (высоты вдали занижены)
    top_k: float = 0.0
    top_s0: float = 0.0

    def __post_init__(self) -> None:
        # Питоновские float, а не numpy-скаляры: арифметика с float32-массивами точек
        # должна идти так же, как на стенде (numpy-скаляр float64 меняет тип результата).
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "confirm_far":
                object.__setattr__(self, f.name, None if v is None else (int(v[0]), int(v[1])))
            elif f.name == "confirm":
                m, n = (int(v[0]), int(v[1]))
                if not 1 <= m <= n:
                    raise ValueError(f"confirm must be (M, N) with 1 <= M <= N, got {v}")
                object.__setattr__(self, f.name, (m, n))
            elif f.name in ("clean_h", "floor_fast", "verify_thr"):
                object.__setattr__(self, f.name, None if v is None else float(v))
            elif f.name in (
                "level",
                "gauge_from_rail_head",
                "speed_legacy",
                "wall_ext",
                "edge_reject",
                "plane_freeze",
            ):
                object.__setattr__(self, f.name, bool(v))
            elif f.name in ("warmup_frames", "calib_frames"):
                object.__setattr__(self, f.name, int(v))
            elif f.name in ("plane_every", "plane_dense", "plane_buf"):
                if int(v) < (1 if f.name == "plane_buf" else 0):
                    raise ValueError(f"{f.name} must be non-negative (plane_buf >= 1), got {v}")
                object.__setattr__(self, f.name, int(v))
            elif f.name == "split":
                object.__setattr__(self, f.name, bool(v))
            elif f.name == "verify_model":
                object.__setattr__(self, f.name, str(v) if v not in (None, "") else None)
            elif f.name == "axis_fit":
                if v not in ("cubic", "smooth"):
                    raise ValueError(f"axis_fit must be 'cubic' or 'smooth', got {v!r}")
            elif f.type in ("int", int):
                object.__setattr__(self, f.name, int(v))
            else:
                object.__setattr__(self, f.name, float(v))


# Поправка Δ(s) ограничена ±0.05 м и профиль полотна прогревается 15 кадров: на стенде
# это в 2.3 раза сокращает ложные тревоги (6.1 → 2.6% кадров) и снимает чувствительность
# к крену/повороту лидара, не ухудшая обнаружение людей, кабелей, тележек
# (research/comparison.md, четвёртый круг).
_FLOOR = {"delta_clip": 0.05, "warmup_frames": 15}

_SOFT = Params(confirm=(2, 5), margin_k=0.002, clean_h=0.5, split=False, **_FLOOR)

MODES: dict[str, Params] = {
    # soft с пониженными порогами (кластер от 2 точек на 1 кольце, запас 0.001/м, без
    # защиты конца оси, с отбраковкой кластеров у края габарита) — кандидатов много,
    # ложные отсекает классификатор; подтверждение 3 из 5
    "default": replace(
        _SOFT,
        edge_reject=True,
        end_guard=0.0,
        min_pts=2,
        min_rings=1,
        margin_k=0.001,
        confirm=(3, 5),
        verify_thr=DEFAULT_THRESHOLD,
    ),
    "geometry": Params(confirm=(3, 5), margin_k=0.004, clean_h=0.5, split=True, **_FLOOR),
    "strict": Params(confirm=(3, 5), margin_k=0.004, clean_h=0.5, split=False, **_FLOOR),
    "soft": _SOFT,
}

# Режим, в котором детектор работает, если классификатор режима недоступен.
FALLBACK_MODE = "geometry"
# Поля классификатора: при откате на FALLBACK_MODE переопределения этих полей отбрасываются.
VERIFY_FIELDS = ("verify_thr", "verify_smin", "verify_model")

# Параметры, при которых ядро повторяет прототип стенда бит-в-бит (сверка tests/parity.py).
BENCH_COMPAT: dict[str, object] = {
    "delta_clip": 0.3,
    "warmup_frames": 0,
    "speed_legacy": True,
    "plane_every": 0,
}

FORWARD_AXES = ("auto", "x", "-x", "y", "-y")


def make_params(mode: str, **overrides: object) -> Params:
    """Параметры режима с переопределениями (имена полей Params)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
    known = {f.name for f in fields(Params)}
    bad = sorted(set(overrides) - known)
    if bad:
        raise TypeError(f"unknown detector parameters: {bad}")
    return replace(MODES[mode], **overrides)  # type: ignore[arg-type]
