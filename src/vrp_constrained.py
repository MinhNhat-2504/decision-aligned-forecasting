"""Constrained multi-vehicle routing proxy for the locked test forecasts.

This is a transparent deterministic heuristic, not an exact VRP solver: each
vehicle greedily appends high-score zones while respecting a demand capacity and
maximum route distance, then returns to the depot. It adds the operational
constraints missing from the original fixed-K TSP proxy.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROC_DIR, OUT_DIR, GRAPH_FILE, HORIZONS, SEQ_LEN, QUANTILES
from dataset import STData

VEHICLES, CAPACITY_FRAC, MAX_ROUTE_KM = 3, 0.42, 22.0


def build_routes(score, demand, D, depot):
    cap = max(float(demand.sum()) * CAPACITY_FRAC, 1.0)
    routes = [[depot] for _ in range(VEHICLES)]
    loads = np.zeros(VEHICLES)
    lengths = np.zeros(VEHICLES)
    # A constrained dispatch policy only considers the top candidate pool; this
    # keeps the transparent heuristic tractable while retaining more candidates
    # than the original K=15 selection.
    remaining = set(np.argsort(-score)[:60].tolist())
    remaining.discard(depot)
    while remaining:
        best = None
        for z in remaining:
            for v in range(VEHICLES):
                if loads[v] + demand[z] > cap:
                    continue
                inc = D[routes[v][-1], z]
                close = D[z, depot]
                if lengths[v] + inc + close > MAX_ROUTE_KM:
                    continue
                ratio = score[z] / max(inc + close, 1e-6)
                cand = (ratio, z, v, inc + close)
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None:
            break
        _, z, v, added = best
        routes[v].append(z)
        loads[v] += demand[z]
        lengths[v] += added
        remaining.remove(z)
    for v in range(VEHICLES):
        routes[v].append(depot)
    return routes, loads, lengths


def main():
    d = STData()
    g = np.load(os.path.join(PROC_DIR, GRAPH_FILE))
    D = g["dist_road"].astype(float)
    bad = D > 1e5
    D[bad] = g["dist_geo"].astype(float)[bad] * 2.0
    np.fill_diagonal(D, 0.0)
    depot = int(np.argmax(d.raw[:d.bounds[1]].sum(0)))
    P = np.load(os.path.join(OUT_DIR, "test_preds.npz"))
    y = P["y_true"][:, :, 0]
    ids = d.idx["test"]
    policies = {"HistoricalAvg": np.stack([d.ha_raw[d.how[s + SEQ_LEN + HORIZONS[0] - 1]] for s in ids]),
                "Oracle": y}
    for name in ("ST-MRGAT", "GraphWaveNet", "GraphWaveNet-Q"):
        if name in P.files:
            a = P[name]
            policies[name] = a[:, :, 0, len(QUANTILES)//2] if a.ndim == 4 else a[:, :, 0]
    rows = []
    for name, score in policies.items():
        covered, distances, served, used = [], [], [], []
        for t in range(len(y)):
            routes, loads, lengths = build_routes(score[t], y[t], D, depot)
            chosen = [z for route in routes for z in route[1:-1]]
            covered.append(100 * y[t, chosen].sum() / (y[t].sum() + 1e-9))
            distances.append(lengths.sum())
            served.append(len(chosen)); used.append(sum(lengths > 0))
        rows.append(dict(policy=name, vehicles=VEHICLES, capacity_fraction=CAPACITY_FRAC,
                         max_route_km=MAX_ROUTE_KM, coverage=float(np.mean(covered)),
                         distance_km=float(np.mean(distances)), zones_served=float(np.mean(served)),
                         vehicles_used=float(np.mean(used))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "vrp_constrained.csv"), index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
