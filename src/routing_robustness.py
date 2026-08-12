"""Route-aware selection robustness check.

The original fixed-K experiment selected demand hotspots first and routed them
afterwards.  This script adds a stronger, route-aware comparator: greedy cheapest
insertion chooses each next zone by predicted demand per marginal tour kilometre,
then the common NN+2-opt heuristic refines the selected set.  It is still a
single-vehicle proxy (not a full VRP), but directly tests whether conclusions hold
when distance is part of the selection rule.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROC_DIR, OUT_DIR, GRAPH_FILE, HORIZONS, QUANTILES, SEQ_LEN
from dataset import STData
from operations import solve

TOP_K = 15
EPS = 1e-6


def route_length(route, D):
    return float(sum(D[route[i], route[i + 1]] for i in range(len(route) - 1)))


def demand_first(score, D, depot, k):
    selected = np.argsort(-score)[:k].tolist()
    _, length = solve(selected, D, depot)
    return np.asarray(selected, dtype=int), length


def route_aware_greedy(score, D, depot, k):
    """Greedy orienteering-style selection using cheapest tour insertion.

    At each step the candidate maximises score / incremental tour length.  The
    marginal length is evaluated over every edge in the current closed depot tour,
    so it accounts for detours rather than merely radial distance from the depot.
    """
    n = len(score)
    # Route-aware comparison should consume the same information as the fixed-K
    # task (a ranking), rather than favour a model merely for having a larger
    # numerical output scale.  Positive descending ranks also make the
    # distance-only policy well-defined.
    utility = n - np.argsort(np.argsort(score, kind="stable"), kind="stable")
    route, available = [depot, depot], np.ones(n, dtype=bool)
    available[depot] = False
    chosen = []
    for _ in range(k):
        cand = np.flatnonzero(available)
        # Vectorise candidate-by-edge insertion costs; this is materially faster
        # than evaluating Python loops for every test hour and candidate.
        a, b = np.asarray(route[:-1]), np.asarray(route[1:])
        deltas = D[a[:, None], cand] + D[cand[None, :], b[:, None]] - D[a, b][:, None]
        pos = np.argmin(deltas, axis=0)
        delta = np.maximum(deltas[pos, np.arange(len(cand))], EPS)
        ratio = utility[cand] / delta
        j = int(np.argmax(ratio))
        best_node, best_pos = int(cand[j]), int(pos[j])
        route.insert(best_pos + 1, best_node)
        available[best_node] = False
        chosen.append(best_node)
    # Common tour heuristic is applied after selection, so differences are due to
    # selection information rather than a privileged route solver.
    _, length = solve(chosen, D, depot)
    return np.asarray(chosen, dtype=int), length


def main():
    d = STData()
    g = np.load(os.path.join(PROC_DIR, GRAPH_FILE))
    D = g["dist_road"].astype(np.float64)
    bad = D > 1e5
    D[bad] = g["dist_geo"].astype(np.float64)[bad] * 2.0
    np.fill_diagonal(D, 0.0)
    depot = int(np.argmax(d.raw[:d.bounds[1]].sum(0)))

    P = np.load(os.path.join(OUT_DIR, "test_preds.npz"))
    y = P["y_true"][:, :, 0]
    ids = d.idx["test"]
    policies = {
        "HistoricalAvg": np.stack([d.ha_raw[d.how[s + SEQ_LEN + HORIZONS[0] - 1]]
                                    for s in ids]),
        "DistanceOnly": np.tile(-D[depot][None, :], (len(y), 1)),
        "Oracle": y,
    }
    for name in ("ST-MRGAT", "GraphWaveNet", "GraphWaveNet-Q", "GraphSAGE", "MLP"):
        if name not in P.files:
            continue
        arr = P[name]
        policies[name] = (arr[:, :, 0, len(QUANTILES) // 2] if arr.ndim == 4
                          else arr[:, :, 0])

    rows = []
    for policy, score in policies.items():
        for selector, fn in (("demand_first", demand_first), ("route_aware", route_aware_greedy)):
            coverage, length = [], []
            for t in range(len(y)):
                selected, L = fn(score[t], D, depot, TOP_K)
                coverage.append(100 * y[t, selected].sum() / (y[t].sum() + EPS))
                length.append(L)
            rows.append(dict(policy=policy, selector=selector, k=TOP_K,
                             coverage=float(np.mean(coverage)),
                             coverage_sd=float(np.std(coverage, ddof=1)),
                             distance_km=float(np.mean(length)),
                             distance_km_sd=float(np.std(length, ddof=1))))

    out = pd.DataFrame(rows).sort_values(["selector", "coverage"], ascending=[True, False])
    out.to_csv(os.path.join(OUT_DIR, "routing_route_aware.csv"), index=False)
    print(f"depot={d.zones.iloc[depot]['zone_name']} | K={TOP_K} | windows={len(y)}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
