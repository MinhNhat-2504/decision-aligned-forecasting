"""Seed-level robustness checks for the decision-centred claims.

The main benchmark stores one prediction tensor per model to keep artefacts small.
That is insufficient for a decision claim: this script retrains the quantile
backbones for every configured seed, applies exactly the same CQR procedure as the
benchmark, and reports seed-level Newsvendor costs and t confidence intervals.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, SEEDS
from dataset import STData, load_graphs
from operations import newsvendor
from train_eval import DEV, train_neural


def mean_ci(values, level=0.95):
    """Student-t CI over independent training initialisations."""
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    if len(x) < 2:
        return mean, np.nan, np.nan, np.nan
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    half = float(stats.t.ppf((1 + level) / 2, len(x) - 1) * se)
    return mean, float(x.std(ddof=1)), mean - half, mean + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0,
                    help="0 uses all configured seeds")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--tag", default="decision_seed")
    ap.add_argument("--models", default="ST-MRGAT,GraphWaveNet-Q")
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds] if args.seeds else SEEDS
    names = [x.strip() for x in args.models.split(",") if x.strip()]
    if not set(names) <= {"ST-MRGAT", "GraphWaveNet-Q"}:
        raise ValueError("This decision check accepts quantile backbones only.")

    d = STData()
    graphs = load_graphs(DEV)
    _, _, y_raw, _ = d.get("test")
    y = y_raw.numpy()[:, :, 0]  # one-hour capacity commitment
    saved, rows = {"y_true": y_raw.numpy()}, []

    for name in names:
        preds = []
        for seed in seeds:
            metric, _ = train_neural(name, d, graphs, seed=seed, hidden=args.hidden)
            pq = metric.pop("_pred")[:, :, 0, :]  # CQR-calibrated quantiles
            preds.append(pq)
            for label, alloc in (("median", pq[..., 1]), ("q0.9", pq[..., -1])):
                r = newsvendor(np.maximum(alloc, 0), y, 0.9)
                r.update(model=name, policy=label, seed=seed,
                         params=metric["params"], forecast_mae=metric["MAE"])
                rows.append(r)
            print(f"{name:16s} seed={seed} MAE={metric['MAE']:.3f} "
                  f"q90-cost={rows[-1]['cost']:.4f}", flush=True)
        saved[name] = np.stack(preds)

    detail = pd.DataFrame(rows)
    summary_rows = []
    for (model, policy), group in detail.groupby(["model", "policy"], sort=False):
        out = {"model": model, "policy": policy, "n_seeds": len(group),
               "params": float(group.params.mean())}
        for col in ("forecast_mae", "cost", "fill_rate", "unmet_pct", "over_pct"):
            mean, sd, lo, hi = mean_ci(group[col])
            out.update({col: mean, f"{col}_sd": sd,
                        f"{col}_ci_low": lo, f"{col}_ci_high": hi})
        summary_rows.append(out)

    # Paired reduction is the decision-level contrast; its unit is a seed.
    for model in names:
        pivot = detail[detail.model == model].pivot(index="seed", columns="policy", values="cost")
        reduction = 100 * (pivot["median"] - pivot["q0.9"]) / pivot["median"]
        mean, sd, lo, hi = mean_ci(reduction)
        summary_rows.append({"model": model, "policy": "q0.9_vs_median_reduction_pct",
                             "n_seeds": len(reduction), "cost": mean, "cost_sd": sd,
                             "cost_ci_low": lo, "cost_ci_high": hi,
                             "params": float(detail[detail.model == model].params.mean())})

    summary = pd.DataFrame(summary_rows)
    detail.to_csv(os.path.join(OUT_DIR, f"{args.tag}_detail.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, f"{args.tag}_summary.csv"), index=False)
    np.savez_compressed(os.path.join(OUT_DIR, f"{args.tag}_preds.npz"), **saved)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
