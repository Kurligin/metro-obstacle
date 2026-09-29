"""Насколько пуст верх габарита на пустых проездах (кандидаты с hh > 0.5 м)."""
import sys, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from bench.data import WORKERS, Bag
from bench.detect import CONFIGS, Detector
from bench.evaluate import EMPTY_BAGS
warnings.filterwarnings("ignore")
cfg = {c.name: c for c in CONFIGS}["A+D+C"]

def run(bag):
    b = Bag(bag); det = Detector(cfg); rows = []
    for k in range(len(b)):
        det.process(b[k].pts)
        H = det.hist[-1]
        up = H["h"] > 0.5
        rows.append((int(up.sum()), np.round(H["s"][up], 1).tolist(), int((~up).sum())))
    return bag, rows

if __name__ == "__main__":
  with ProcessPoolExecutor(min(5, WORKERS)) as ex:
    for bag, rows in ex.map(run, EMPTY_BAGS):
        n = np.array([r[0] for r in rows])
        S = np.concatenate([np.array(r[1]) for r in rows if r[1]]) if n.sum() else np.array([])
        bins = [5, 20, 40, 60, 80, 100, 150]
        fr = [np.mean([any(a <= v < b for v in r[1]) for r in rows]) for a, b in zip(bins[:-1], bins[1:])]
        print(f"{bag:38s} доля кадров с точками в верхней зоне по дальности 5-20/20-40/40-60/60-80/80-100/100-150: " + " ".join(f"{x:.1%}" for x in fr))
        continue
        print(f"{bag:38s} кадров с точками в верхней зоне: {np.mean(n>0):.1%}; ≥3 точек: {np.mean(n>=3):.1%}; медиана точек/кадр при >0: {np.median(n[n>0]) if (n>0).any() else 0}; s p10/50/90: {np.percentile(S,[10,50,90]).round(0) if len(S) else '-'}")
