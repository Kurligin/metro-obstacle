"""Графики и сводные таблицы для docs/EXPERIMENTS.md по готовым результатам стенда.

Скрипт ничего не прогоняет: читает JSON и логи в out/ и пишет PNG в docs/img/,
а таблицы (markdown) печатает в stdout — из них взяты цифры EXPERIMENTS.md.

Прогон ядра 25.09 (out/eval_final_pc/) сделан до переименования режимов: тогдашний
«default» — нынешний режим geometry (только геометрия, до исправления оценки скорости).
На графиках по этому прогону он подписан как geometry (MODE_LABEL). Итоговое сравнение
default (классификатор) и geometry — график fp_final.png (fig_fp_final).

Откуда данные (команды, которыми они получены):
    out/eval_final_pc/  прогон ядра 25.09, x86 (ключи core:default = нынешний geometry):
        python -m bench.evaluate empty "core:default,core:strict,core:soft"
        python -m bench.evaluate synth "core:default,core:strict,core:soft"
        python -m tests.speed --bags all --modes default,strict,soft > out_fin_speed.log
    out/eval/           прототипы, круги 1–3: python -m bench.evaluate empty|synth
    out/round4_pc/      четвёртый круг: python -m bench.round4 empty|synth

    python -m bench.make_figures
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FINAL = OUT / "eval_final_pc"
EARLY = OUT / "eval"
ROUND4 = OUT / "round4_pc"
IMG = ROOT / "docs" / "img"

EMPTY_BAGS = [
    "doubleT_platform",
    "roundT_doubleT",
    "roundT_pressureGate_roundT",
    "roundT_squareT_pressureGate_squareT",
    "squareT_platform_squareT_switch",
]
ALL_BAGS = ["doubleT_obstacle", *EMPTY_BAGS]
MODES = ["default", "strict", "soft"]
OBJ_LABEL = {
    "box_30x10": "объект 300×100 мм",
    "post_10x30": "столбик 10×30 см",
    "box_50": "коробка 0.5 м",
    "person": "человек",
    "cable": "кабель Ø20 мм",
    "cart": "тележка 1 м",
}
BINS = [0, 20, 40, 60, 80, 100, 125, 150]
MIN_FRAMES = 20  # точка на графике Pd — только если в бине столько видимых кадров
FINAL_R4 = "default|delta_clip=0.05,warmup_frames=15"  # итоговые умолчания в прогоне 4-го круга
MODE_COLOR = {"default": "tab:blue", "strict": "tab:green", "soft": "tab:orange"}
# подписи режимов прогона 25.09: тогдашний default — нынешний geometry
MODE_LABEL = {"default": "geometry", "strict": "strict", "soft": "soft"}
RUN_0925 = "прогон 25.09"

Row = dict[str, Any]


# ------------------------------------------------------------------ общие расчёты


def bin_labels() -> list[str]:
    return [f"{a}–{b}" for a, b in pairwise(BINS)]


def bin_centers() -> np.ndarray:
    return np.array([(a + b) / 2 for a, b in pairwise(BINS)])


def min_visible(obj_type: str) -> int:
    """Объект «видим» в кадре: ≥3 точек на нём (кабель — ≥1), как в отчётах стенда."""
    return 1 if obj_type == "cable" else 3


def alarm_stats(frames: Iterable[tuple[float, bool]]) -> tuple[int, int, int, float]:
    """(кадров, кадров с тревогой, событий, км) по последовательности (s поезда, тревога)."""
    n = alarms = events = 0
    prev = False
    s_first = s_last = None
    for s, alarm in frames:
        n += 1
        alarms += alarm
        events += alarm and not prev
        prev = alarm
        s_first = s if s_first is None else s_first
        s_last = s
    km = 0.0 if s_first is None or s_last is None else (s_last - s_first) / 1000
    return n, alarms, events, km


def sum_stats(
    parts: Iterable[tuple[int, int, int, float]],
) -> tuple[int, int, int, float]:
    a = np.array(list(parts), dtype=float)
    return (
        int(a[:, 0].sum()),
        int(a[:, 1].sum()),
        int(a[:, 2].sum()),
        float(a[:, 3].sum()),
    )


def load_empty(folder: Path, bag: str, cfg: str) -> list[Row]:
    return json.loads((folder / f"empty__{bag}__{cfg}.json").read_text())


def empty_bag_stats(folder: Path, bag: str, cfg: str) -> tuple[int, int, int, float]:
    rows = load_empty(folder, bag, cfg)
    return alarm_stats((r["s_train"], bool(r["dets"])) for r in rows)


def empty_total(folder: Path, cfg: str) -> tuple[int, int, int, float]:
    return sum_stats(empty_bag_stats(folder, b, cfg) for b in EMPTY_BAGS)


def round4_empty() -> dict[tuple[str, str], tuple[int, int, int, float]]:
    """(детектор, искажение) → суммарная статистика ложных тревог по 5 пустым bag."""
    parts: dict[tuple[str, str], list[tuple[int, int, int, float]]] = {}
    for _bag, det, tf, rows in json.loads((ROUND4 / "empty.json").read_text()):
        st = alarm_stats((s, bool(dets)) for _k, s, dets in rows)
        parts.setdefault((det, tf), []).append(st)
    return {key: sum_stats(v) for key, v in parts.items()}


def fmt_stats(st: tuple[int, int, int, float]) -> str:
    n, al, ev, km = st
    return f"{100 * al / n:.1f}% | {ev} | {ev / km:.1f}"


def pd_by_bin(
    items: list[tuple[float, int, bool]], obj_type: str, visible_only: bool = True
) -> list[tuple[float, int]]:
    """Pd по бинам дальности: [(доля кадров с обнаружением, число кадров)]."""
    thr = min_visible(obj_type)
    out = []
    for a, b in pairwise(BINS):
        hits = [h for d, n, h in items if a <= d < b and (n >= thr or not visible_only)]
        out.append((float(np.mean(hits)) if hits else float("nan"), len(hits)))
    return out


def stable_distance(runs: dict[Any, list[tuple[int, float, bool]]]) -> float:
    """Медиана по проездам: дальность, на которой объект впервые найден 3 кадра подряд.

    Проезд без такого обнаружения даёт 0 (как в bench.evaluate.report).
    """
    firsts = []
    for seq in runs.values():
        run, first = 0, 0.0
        for _k, dist, hit in sorted(seq):
            run = run + 1 if hit else 0
            if run >= 3:
                first = dist
                break
        firsts.append(first)
    return float(np.median(firsts)) if firsts else float("nan")


# ------------------------------------------------------------------ финальная синтетика


def load_final_synth() -> list[Row]:
    return json.loads((FINAL / "synth__core.json").read_text())


def final_items(rows: list[Row], mode: str, obj_type: str) -> list[tuple[float, int, bool]]:
    cfg = f"core:{mode}"
    return [
        (r["dist"], r["n_obj"], r["hit"])
        for r in rows
        if r["cfg"] == cfg and r["type"] == obj_type and r["inserted"]
    ]


def final_runs(rows: list[Row], mode: str, obj_type: str) -> dict[Any, list]:
    cfg = f"core:{mode}"
    runs: dict[Any, list] = {}
    for r in rows:
        if r["cfg"] == cfg and r["type"] == obj_type:
            runs.setdefault((r["bag"], r["s_obj"]), []).append((r["k"], r["dist"], r["hit"]))
    return runs


def points_by_bin(rows: list[Row], obj_type: str) -> list[tuple[float, float, float, int]]:
    """Точки на объекте (кадр со вставкой): среднее, медиана, доля видимых, число кадров."""
    thr = min_visible(obj_type)
    items = [
        (r["dist"], r["n_obj"])
        for r in rows
        if r["cfg"] == "core:default" and r["type"] == obj_type and r["inserted"]
    ]
    out = []
    for a, b in pairwise(BINS):
        n = np.array([m for d, m in items if a <= d < b], dtype=float)
        if len(n):
            out.append((float(n.mean()), float(np.median(n)), float((n >= thr).mean()), len(n)))
        else:
            out.append((float("nan"), float("nan"), float("nan"), 0))
    return out


# ------------------------------------------------------------------ 4-й круг: синтетика


def load_round4_synth() -> list[list[Any]]:
    # строка: [bag, s_obj, type, dual, detector, transform, k, dist, n_obj, hit]
    return json.loads((ROUND4 / "synth.json").read_text())


def r4_items(rows: list[list[Any]], det: str, tf: str, dual: bool, obj_type: str) -> list:
    return [
        (r[7], r[8], r[9])
        for r in rows
        if r[2] == obj_type and r[4] == det and r[5] == tf and r[3] == dual
    ]


def r4_runs(rows: list[list[Any]], det: str, tf: str, dual: bool, obj_type: str) -> dict:
    runs: dict[Any, list] = {}
    for r in rows:
        if r[2] == obj_type and r[4] == det and r[5] == tf and r[3] == dual:
            runs.setdefault((r[0], r[1]), []).append((r[6], r[7], r[9]))
    return runs


# ------------------------------------------------------------------ ложные тревоги по причинам


def alarm_events(rows: list[Row]) -> list[tuple[int, int, list[list[float]]]]:
    """События: подряд идущие кадры с тревогой → (первый кадр, последний, детекции)."""
    events: list[tuple[int, int, list[list[float]]]] = []
    cur: list[Any] | None = None
    for r in rows:
        if r["dets"]:
            if cur is None:
                cur = [r["k"], r["k"], []]
            cur[1] = r["k"]
            cur[2] += r["dets"]
        elif cur is not None:
            events.append((cur[0], cur[1], cur[2]))
            cur = None
    if cur is not None:
        events.append((cur[0], cur[1], cur[2]))
    return events


CAUSES = [
    "первые кадры после прогрева",
    "низко (<0.3 м), 2–4 точки у рельса, <40 м",
    "низко (<0.3 м), дальше 40 м",
    "верх габарита, 60–85 м (переход круглый → сдвоенный)",
    "дальняя зона ≥85 м (платформа, стрелка, конец оси)",
]


def classify(k0: int, dets: list[list[float]]) -> str:
    """Причина события по детекциям (s, lat, h, n) — правила из разбора кадров."""
    s_min = min(d[0] for d in dets)
    h_max = max(d[2] for d in dets)
    if k0 < 20:
        return CAUSES[0]
    if h_max < 0.3:
        return CAUSES[1] if s_min < 40 else CAUSES[2]
    return CAUSES[3] if s_min < 85 else CAUSES[4]


# ------------------------------------------------------------------ таблицы (stdout)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def f2(x: float) -> str:
    return "—" if np.isnan(x) else f"{x:.2f}"


def print_tables(synth: list[Row], r4s: list[list[Any]]) -> None:
    print("## Ложные тревоги, финальный прогон: кадров с тревогой | событий | событий/км\n")
    rows = []
    for m in MODES:
        cfg = f"core:{m}"
        rows.append([m, "все 5", fmt_stats(empty_total(FINAL, cfg))])
        rows += [[m, b, fmt_stats(empty_bag_stats(FINAL, b, cfg))] for b in EMPTY_BAGS]
    print(md_table(["режим", "bag", "кадров с тревогой | событий | событий/км"], rows))
    km = empty_total(FINAL, "core:default")[3]
    print(f"\nкм по пустым bag: {km:.3f}")
    for b in EMPTY_BAGS:
        n, _, _, kmb = empty_bag_stats(FINAL, b, "core:default")
        print(f"  {b}: {n} кадров, {kmb:.3f} км")

    print("\n## Причины событий (default, финал)\n")
    for m in ("default", "strict"):
        count: dict[str, list[int]] = {c: [0, 0] for c in CAUSES}
        for b in EMPTY_BAGS:
            for k0, k1, dets in alarm_events(load_empty(FINAL, b, f"core:{m}")):
                c = classify(k0, dets)
                count[c][0] += 1
                count[c][1] += k1 - k0 + 1
        print(m, {c: v for c, v in count.items()})

    print("\n## Pd при видимом объекте (финал) и доля видимых кадров\n")
    for t in OBJ_LABEL:
        rows = []
        for m in MODES:
            pd = pd_by_bin(final_items(synth, m, t), t)
            rows.append([m] + [f"{f2(v)} ({n})" for v, n in pd])
        pd_all = pd_by_bin(final_items(synth, "default", t), t, visible_only=False)
        rows.append(["default, все кадры"] + [f2(v) for v, _ in pd_all])
        pts = points_by_bin(synth, t)
        rows.append(["точек: среднее"] + [f"{p[0]:.1f}" for p in pts])
        rows.append(["точек: медиана"] + [f"{p[1]:.0f}" for p in pts])
        rows.append(["доля видимых"] + [f2(p[2]) for p in pts])
        print(f"### {t}\n")
        print(md_table(["", *bin_labels()], rows) + "\n")

    print("## Устойчивое обнаружение (3 кадра подряд), медиана, м\n")
    rows = [
        [m] + [f"{stable_distance(final_runs(synth, m, t)):.0f}" for t in OBJ_LABEL] for m in MODES
    ]
    print(md_table(["режим", *OBJ_LABEL], rows))

    print("\n## 4-й круг: ложные тревоги\n")
    r4 = round4_empty()
    print(
        md_table(
            ["детектор", "искажение", "кадров | событий | /км"],
            [[d, tf or "—", fmt_stats(st)] for (d, tf), st in r4.items()],
        )
    )

    print("\n## 4-й круг: без второго отражения (итоговые умолчания), Pd при видимом\n")
    rows = []
    for t in ("box_30x10", "post_10x30", "box_50", "person", "cable"):
        for dual in (True, False):
            pd = pd_by_bin(r4_items(r4s, FINAL_R4, "", dual, t), t)
            st = stable_distance(r4_runs(r4s, FINAL_R4, "", dual, t))
            rows.append([t, "да" if dual else "нет"] + [f2(v) for v, _ in pd[:6]] + [f"{st:.0f}"])
        pd = pd_by_bin(r4_items(r4s, "default", "", True, t), t)
        st = stable_distance(r4_runs(r4s, "default", "", True, t))
        rows.append([t, "да, delta_clip 0.3"] + [f2(v) for v, _ in pd[:6]] + [f"{st:.0f}"])
    print(md_table(["объект", "2 отражения", *bin_labels()[:6], "устойч., м"], rows))

    print("\n## История улучшений\n")
    print(
        md_table(
            ["шаг", "кадров | событий | /км"],
            [[s[0], fmt_stats(s[1])] for s in history()],
        )
    )


# ------------------------------------------------------------------ графики


def fig_pd_default(synth: list[Row]) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    x = bin_centers()
    for t, label in OBJ_LABEL.items():
        pd = pd_by_bin(final_items(synth, "default", t), t)
        y = np.array([v if n >= MIN_FRAMES else np.nan for v, n in pd])
        ax[0].plot(x, y, "o-", label=label)
        vis = np.array([p[2] if p[3] >= MIN_FRAMES else np.nan for p in points_by_bin(synth, t)])
        ax[1].plot(x, vis, "o-", label=label)
    ax[0].set_title(f"Обнаружение, когда объект виден (geometry, {RUN_0925})")
    ax[0].set_ylabel("доля кадров с обнаружением")
    ax[1].set_title("Физика: доля кадров, где объект виден\n(≥3 точек, кабель ≥1)")
    ax[1].set_ylabel("доля кадров")
    for a in ax:
        a.set_xlabel("дальность до объекта, м")
        a.set_xticks(BINS)
        a.set_ylim(0, 1.02)
        a.grid(alpha=0.3)
    ax[1].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(IMG / "pd_default.png", dpi=110)
    plt.close(fig)


def fig_pd_modes(synth: list[Row]) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    x = bin_centers()
    for a, t in zip(ax, ("person", "cable", "box_50"), strict=True):
        for m in MODES:
            pd = pd_by_bin(final_items(synth, m, t), t)
            y = np.array([v if n >= MIN_FRAMES else np.nan for v, n in pd])
            a.plot(x, y, "o-", color=MODE_COLOR[m], label=MODE_LABEL[m])
        a.set_title(OBJ_LABEL[t])
        a.set_xlabel("дальность, м")
        a.set_xticks(BINS)
        a.set_ylim(0, 1.02)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("доля кадров с обнаружением\n(объект виден)")
    ax[0].legend(title=RUN_0925, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "pd_modes.png", dpi=110)
    plt.close(fig)


def fig_points(synth: list[Row]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = bin_centers()
    for t, label in OBJ_LABEL.items():
        pts = points_by_bin(synth, t)
        y = np.array([p[0] if p[3] >= MIN_FRAMES else np.nan for p in pts])
        ax.plot(x, np.maximum(y, 0.05), "o-", label=label)
    ax.axhline(3, color="k", ls="--", lw=1)
    ax.text(142, 3.3, "3 точки", ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("дальность до объекта, м")
    ax.set_ylabel("точек на объекте за кадр (среднее)")
    ax.set_title("Сколько точек лидар кладёт на объект")
    ax.set_xticks(BINS)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "points_range.png", dpi=110)
    plt.close(fig)


def history() -> list[tuple[str, tuple[int, int, int, float]]]:
    """Ступеньки улучшений: ложные тревоги на 5 пустых bag после каждого шага."""
    r4 = round4_empty()
    return [
        ("прямая по оси лидара, порог 0.35 м", empty_total(EARLY, "S+fix")),
        ("+ ось по рельсам и стенам", empty_total(EARLY, "A+fix")),
        ("+ выученный профиль полотна", empty_total(EARLY, "A+D")),
        ("+ подтверждение 3 из 5 со сдвигом", empty_total(EARLY, "A+D+C")),
        ("+ «чистая зона»", empty_total(EARLY, "Z c3")),
        ("+ раздвоенные лучи", empty_total(EARLY, "Z+S c3")),
        ("+ предел поправки Δ(s) 0.05 м", r4[("default|delta_clip=0.05", "")]),
        (
            "+ прогрев полотна 15 кадров (geometry, 25.09)",
            empty_total(FINAL, "core:default"),
        ),
        ("strict (без раздвоенных лучей, 25.09)", empty_total(FINAL, "core:strict")),
    ]


def fig_history() -> None:
    steps = history()
    labels = [f"{i + 1}. {s}" for i, (s, _) in enumerate(steps)]
    pct = np.array([100 * st[1] / st[0] for _, st in steps])
    per_km = np.array([st[2] / st[3] for _, st in steps])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    y = np.arange(len(steps))[::-1]
    colors = ["tab:gray"] * (len(steps) - 2) + ["tab:blue", "tab:green"]
    ax[0].barh(y, pct, color=colors)
    ax[1].barh(y, per_km, color=colors)
    for yi, p, e in zip(y, pct, per_km, strict=True):
        ax[0].text(p + 1, yi, f"{p:.1f}%", va="center", fontsize=9)
        ax[1].text(e + 1, yi, f"{e:.1f}", va="center", fontsize=9)
    ax[0].set_yticks(y, labels)
    ax[0].set_xlim(0, 110)
    ax[1].set_xlim(0, 100)
    ax[0].set_xlabel("кадров с тревогой, %")
    ax[1].set_xlabel("событий тревоги на км")
    fig.suptitle(
        "Ложные тревоги на 5 пустых проездах (2.14 км) по шагам улучшений геометрии\n"
        "(до исправления оценки скорости и классификатора; итог — график fp_final)"
    )
    for a in ax:
        a.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "fp_history.png", dpi=110)
    plt.close(fig)


SPEED_RE = re.compile(
    r"^(\w+)\s+(\S+)\s+кадров\s+(\d+)\s+первый\s+([\d.]+) мс\s+p50\s+([\d.]+)\s+p95\s+([\d.]+)"
    r"\s+max\s+([\d.]+)"
)


def load_speed() -> dict[tuple[str, str], tuple[float, float, float]]:
    """(режим, bag) → (p50, p95, max), мс — из out_fin_speed.log (tests.speed на x86)."""
    out = {}
    for line in (FINAL / "out_fin_speed.log").read_text().splitlines():
        m = SPEED_RE.match(line.strip())
        if m:
            out[(m[1], m[2])] = (float(m[5]), float(m[6]), float(m[7]))
    return out


BAG_SHORT = {
    "doubleT_obstacle": "doubleT_obstacle\n(сектор ±124°)",
    "doubleT_platform": "doubleT_platform",
    "roundT_doubleT": "roundT_doubleT",
    "roundT_pressureGate_roundT": "roundT_pressureGate\n_roundT",
    "roundT_squareT_pressureGate_squareT": "roundT_squareT_\npressureGate_squareT",
    "squareT_platform_squareT_switch": "squareT_platform_\nsquareT_switch",
}


def fig_speed() -> None:
    sp = load_speed()
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(ALL_BAGS))
    w = 0.26
    for i, m in enumerate(MODES):
        p50 = [sp[(m, b)][0] for b in ALL_BAGS]
        p95 = [sp[(m, b)][1] for b in ALL_BAGS]
        xs = x + (i - 1) * w
        ax.bar(xs, p95, w, color=MODE_COLOR[m], alpha=0.35)
        ax.bar(
            xs,
            p50,
            w,
            color=MODE_COLOR[m],
            label=f"{MODE_LABEL[m]}: p50 (тёмная), p95 (светлая)",
        )
    ax.axhline(100, color="tab:red", ls="--", lw=1)
    ax.text(-0.4, 97, "бюджет 100 мс (лидар 10 Гц)", ha="left", va="top", color="tab:red")
    ax.set_xticks(x, [BAG_SHORT[b] for b in ALL_BAGS], fontsize=8)
    ax.set_ylabel("время ядра на кадр, мс")
    ax.set_ylim(0, 105)
    ax.set_title(f"Скорость ядра на x86 (i5-10600KF, один поток), {RUN_0925}")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "speed.png", dpi=110)
    plt.close(fig)


TF_LABEL = {
    "": "без искажения",
    "roll=2": "крен +2°",
    "roll=-2": "крен −2°",
    "yaw=2": "поворот +2°",
    "yaw=-2": "поворот −2°",
    "pitch=0.5": "тангаж +0.5°",
    "pitch=-0.5": "тангаж −0.5°",
}


def fig_mount() -> None:
    r4 = round4_empty()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(TF_LABEL))
    w = 0.38
    for i, (det, label, color) in enumerate(
        (
            ("default", "было: поправка Δ(s) до 0.3 м, без прогрева", "tab:gray"),
            (FINAL_R4, "стало: Δ(s) до 0.05 м, прогрев 15 кадров", "tab:blue"),
        )
    ):
        vals = []
        for tf in TF_LABEL:
            n, al, _, _ = r4[(det, tf)]
            vals.append(100 * al / n)
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_xticks(x, list(TF_LABEL.values()))
    ax.set_ylabel("кадров с тревогой, %")
    ax.set_title(
        f"Ложные тревоги при искажённой установке лидара (geometry, 5 пустых bag, {RUN_0925})"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "mount_robustness.png", dpi=110)
    plt.close(fig)


def fig_real_person() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for m, marker in (("default", "o"), ("soft", "x")):
        rows = load_empty(FINAL, "doubleT_obstacle", f"core:{m}")
        ks = [r["k"] for r in rows for _ in r["dets"]]
        ds = [d[0] for r in rows for d in r["dets"]]
        ax.plot(
            ks,
            ds,
            marker,
            ms=4,
            color=MODE_COLOR[m],
            label=f"{MODE_LABEL[m]}: подтверждённые объекты",
        )
    ax.axvspan(0, 14, color="tab:gray", alpha=0.15, label="прогрев модели полотна")
    ax.set_xlabel("кадр (10 Гц), поезд стоит")
    ax.set_ylabel("дальность вдоль пути, м")
    ax.set_title(f"Реальный человек в doubleT_obstacle ({RUN_0925})")
    ax.set_ylim(0, 180)
    ax.set_xlim(0, 201)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(IMG / "real_person.png", dpi=110)
    plt.close(fig)


# Итоговое сравнение режимов (финальный код, 28.09). Числа — из логов прогона на x86:
#   5 пустых записей: out/pc_final/logs/v9/e2e.log (bench.verifier_e2e run rand: для каждой
#       записи модель без неё; «base» = geometry, «low+ver(случайные формы)» = default);
#   новая запись заказчика: out/pc_final/logs/v9/newdata_geometry.log (bench.newdata run,
#       12 отрезков; строки «== geometry» и «== core_default»).
# (кадров с тревогой, %; событий на км)
FP_FINAL = {
    "5 пустых записей хакатона\n(2.14 км, 2287 кадров)": {
        "geometry": (3.7, 22.9),
        "default": (0.0, 0.0),
    },
    "новая запись заказчика\n(13.2 км, 11 271 кадр)": {
        "geometry": (9.21, 25.7),
        "default": (0.04, 0.4),
    },
}
FINAL_COLOR = {"geometry": "tab:gray", "default": "tab:blue"}


def fig_fp_final() -> None:
    """Ложные тревоги итоговых режимов: geometry против default (с классификатором)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    groups = list(FP_FINAL)
    x = np.arange(len(groups))
    w = 0.36
    for i, mode in enumerate(("geometry", "default")):
        for j, (a, fmt) in enumerate(((ax[0], "%.2g%%"), (ax[1], "%.1f"))):
            vals = [FP_FINAL[g][mode][j] for g in groups]
            bars = a.bar(x + (i - 0.5) * w, vals, w, color=FINAL_COLOR[mode], label=mode)
            a.bar_label(bars, labels=[fmt % v for v in vals], fontsize=9)
    ax[0].set_ylabel("кадров с тревогой, %")
    ax[1].set_ylabel("событий тревоги на км")
    ax[0].set_title("Доля кадров с ложной тревогой")
    ax[1].set_title("Частота ложных тревог")
    for a in ax:
        a.set_xticks(x, groups, fontsize=9)
        a.grid(axis="y", alpha=0.3)
        a.set_ylim(0, a.get_ylim()[1] * 1.12)
    ax[0].legend()
    fig.suptitle(
        "Ложные тревоги: geometry (только геометрия) и default (геометрия + классификатор)"
    )
    fig.tight_layout()
    fig.savefig(IMG / "fp_final.png", dpi=110)
    plt.close(fig)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    synth = load_final_synth()
    r4s = load_round4_synth()
    print_tables(synth, r4s)
    fig_pd_default(synth)
    fig_pd_modes(synth)
    fig_points(synth)
    fig_history()
    fig_speed()
    fig_mount()
    fig_real_person()
    fig_fp_final()
    print(f"\nграфики: {IMG}")


if __name__ == "__main__":
    main()
