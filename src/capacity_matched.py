"""Capacity-matched architecture robustness check.

Two pairs are deliberately matched within 0.4% parameters:
  * Graph WaveNet hidden=40 (61,263) vs ST-MRGAT hidden=64 (62,155)
  * Graph WaveNet hidden=64 (147,087) vs ST-MRGAT hidden=100 (147,511)

The output is separate from the primary benchmark so no original result is
overwritten.  It tests whether a conclusion changes merely because one backbone
has more trainable capacity.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, SEEDS
from dataset import STData, load_graphs
from train_eval import DEV, train_neural

PAIRS = (("GraphWaveNet", 40, "ST-MRGAT", 64),
         ("GraphWaveNet", 64, "ST-MRGAT", 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="capacity_matched")
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    d = STData()
    graphs = load_graphs(DEV)
    rows = []
    for left, lh, right, rh in PAIRS:
        pair = f"{left}-h{lh}_vs_{right}-h{rh}"
        for name, hidden in ((left, lh), (right, rh)):
            for seed in seeds:
                m, _ = train_neural(name, d, graphs, seed=seed, hidden=hidden)
                m.pop("_pred", None)
                rows.append(dict(pair=pair, model=name, hidden=hidden, seed=seed,
                                 **m))
                print(f"{pair:43s} {name:14s} seed={seed} "
                      f"MAE={m['MAE']:.4f} params={m['params']}", flush=True)
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["pair", "model", "hidden", "params"], as_index=False).agg(
        n=("seed", "count"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE=("RMSE", "mean"), WMAPE=("WMAPE", "mean"), R2=("R2", "mean"))
    raw.to_csv(os.path.join(OUT_DIR, f"{args.tag}_raw.csv"), index=False)
    agg.to_csv(os.path.join(OUT_DIR, f"{args.tag}_summary.csv"), index=False)
    print("\n" + agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
