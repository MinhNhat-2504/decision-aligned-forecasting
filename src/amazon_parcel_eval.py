"""Real parcel-demand and constrained-routing validation on Amazon Last Mile data.

Uses the public 2021 Amazon Last Mile Routing Research Challenge *training* data
(2018).  The script makes a chronological route-date split, derives actual package
count, cubic volume, planned service time, vehicle capacity and time windows, then:
1) evaluates training-only median/q0.9 station-day capacity policies; and
2) evaluates actual driver routes, nearest-neighbour, and a time-window-aware
   greedy policy using route-specific historically realised travel-time estimates.

Observed driver sequences are operational references, not mathematical optima.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import ijson
import numpy as np
import pandas as pd

from operations import newsvendor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "amazon_lastmile"
OUT = ROOT / "outputs"
CACHE = DATA / "parcel_route_summaries.pkl"
TAU, RNG = .9, np.random.default_rng(20260729)


def read_json(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return json.load(f)


def departure_seconds(route):
    return datetime.fromisoformat(f"{route['date_YYYY_MM_DD']} {route['departure_time_utc']}")


def route_split(routes):
    dates = sorted(x["date_YYYY_MM_DD"] for x in routes.values())
    cut = dates[int(.70 * len(dates)) - 1]
    train = {rid for rid, r in routes.items() if r["date_YYYY_MM_DD"] <= cut}
    test = set(routes) - train
    return cut, train, test


def package_summaries(routes, test_ids):
    """Summarise packages; retain stop detail only for holdout routing instances.

    Amazon encodes missing time windows as the non-standard JSON token ``NaN``.
    Python's standard decoder accepts it; strict streaming decoders do not.
    The decoded package object is released before the much larger travel-time file
    is streamed, so the two large objects are never resident together.
    """
    daily, tests = defaultdict(lambda: np.zeros(3, dtype=float)), {}
    with (DATA / "package_data.json").open(encoding="utf-8") as f:
        packages = json.load(f)
    for n, (rid, stops) in enumerate(packages.items(), 1):
        if rid not in routes:
            continue
        route, depart = routes[rid], departure_seconds(routes[rid])
        total = np.zeros(3, dtype=float)  # package count, volume, service
        keep, stop_summary = rid in test_ids, {}
        for sid, packs in stops.items():
            service, volume, ends, starts, count = 0., 0., [], [], 0
            for p in packs.values():
                count += 1
                service += float(p.get("planned_service_time_seconds") or 0.)
                dim = p.get("dimensions") or {}
                try:
                    volume += float(dim["depth_cm"]) * float(dim["height_cm"]) * float(dim["width_cm"])
                except (KeyError, TypeError, ValueError):
                    pass
                tw = p.get("time_window") or {}
                begin, end = tw.get("start_time_utc"), tw.get("end_time_utc")
                if begin and end and str(begin).lower() != "nan" and str(end).lower() != "nan":
                    try:
                        starts.append((datetime.fromisoformat(begin) - depart).total_seconds())
                        ends.append((datetime.fromisoformat(end) - depart).total_seconds())
                    except ValueError:
                        pass
            total += (count, volume, service)
            if keep:
                # Serving begins no earlier than the most restrictive package
                # opening time. Lateness remains package-specific.
                stop_summary[sid] = dict(service=service, earliest=max([0., *starts]), ends=ends,
                                         packages=count, volume=volume)
        key = (route["station_code"], route["date_YYYY_MM_DD"])
        daily[key] += total
        if keep:
            tests[rid] = dict(stops=stop_summary, total_volume=total[1],
                              packages=total[0], service=total[2])
        if n % 1000 == 0:
            print(f"package routes summarized: {n:,}", flush=True)
    del packages
    return dict(daily), tests


def load_or_build_summaries(routes, test_ids):
    if CACHE.exists():
        with CACHE.open("rb") as f:
            print("loading cached package summaries", flush=True)
            return pickle.load(f)
    daily, tests = package_summaries(routes, test_ids)
    with CACHE.open("wb") as f:
        pickle.dump((daily, tests), f, protocol=pickle.HIGHEST_PROTOCOL)
    print("cached package summaries", flush=True)
    return daily, tests


def capacity_eval(routes, daily, cutoff):
    rows = []
    train = {k: v for k, v in daily.items() if k[1] <= cutoff}
    test = {k: v for k, v in daily.items() if k[1] > cutoff}
    by_key, by_station = defaultdict(list), defaultdict(list)
    for (station, date), values in train.items():
        dow = datetime.fromisoformat(date).weekday()
        by_key[(station, dow)].append(values)
        by_station[station].append(values)
    dates, losses = [], {"median": [], "q0.9": []}
    for (station, date), y in sorted(test.items(), key=lambda x: x[0][1]):
        hist = np.asarray(by_key.get((station, datetime.fromisoformat(date).weekday()), by_station[station]))
        if not len(hist):
            continue
        for label, q in (("median", .5), ("q0.9", .9)):
            a = np.quantile(hist, q, axis=0)
            for metric, ix in (("package_count", 0), ("volume_cm3", 1), ("planned_service_seconds", 2)):
                r = newsvendor(np.array([a[ix]]), np.array([y[ix]]), TAU)
                rows.append(dict(policy=label, metric=metric, station=station, date=date,
                                 cost=r["cost"], allocation=a[ix], realised=y[ix]))
        dates.append(date)
    df = pd.DataFrame(rows)
    out = []
    for metric, g in df.groupby("metric"):
        med = g[g.policy == "median"].set_index(["station", "date"]).cost
        q90 = g[g.policy == "q0.9"].set_index(["station", "date"]).cost
        paired = pd.DataFrame({"median": med, "q90": q90}).reset_index()
        day = paired.groupby("date")[["median", "q90"]].mean()
        draw = RNG.integers(0, len(day), size=(2000, len(day)))
        med_day, q90_day = day["median"].to_numpy(), day["q90"].to_numpy()
        red = 100 * (med_day[draw].mean(1) - q90_day[draw].mean(1)) / med_day[draw].mean(1)
        out.append(dict(metric=metric, n_station_days=len(paired), n_test_days=len(day),
                        median_cost=paired["median"].mean(), q90_cost=paired["q90"].mean(),
                        q90_vs_median_reduction_pct=100 * (1 - paired["q90"].mean()/paired["median"].mean()),
                        ci95_low=np.quantile(red, .025), ci95_high=np.quantile(red, .975)))
    return df, pd.DataFrame(out)


def sequence_actual(route, actual):
    return [sid for sid, _ in sorted(actual["actual"].items(), key=lambda x: x[1])]


def travel(tt, a, b):
    try:
        return float(tt[a][b])
    except (KeyError, TypeError):
        return 0.0


def greedy_nn(start, candidates, tt, stops):
    cur, left, order = start, set(candidates), []
    while left:
        nxt = min(left, key=lambda x: travel(tt, cur, x))
        order.append(nxt); left.remove(nxt); cur = nxt
    return [start, *order]


def greedy_tw(start, candidates, tt, stops):
    cur, left, order, now = start, set(candidates), [], 0.
    while left:
        def score(x):
            s = stops[x]; arrival = now + travel(tt, cur, x); begin = max(arrival, s["earliest"])
            deadline = min(s["ends"]) if s["ends"] else np.inf
            # One late minute is heavily penalised; with no deadline this reduces
            # to a nearest-neighbour decision.
            return travel(tt, cur, x) + 20. * max(0., begin - deadline)
        nxt = min(left, key=score); s = stops[nxt]
        now = max(now + travel(tt, cur, nxt), s["earliest"]) + s["service"]
        order.append(nxt); left.remove(nxt); cur = nxt
    return [start, *order]


def simulate(order, tt, stops):
    now = travel_time = service = wait = 0.; late_packages = total_packages = 0
    for a, b in zip(order[:-1], order[1:]):
        leg = travel(tt, a, b); travel_time += leg; now += leg
        s = stops.get(b)
        if s is None:  # station start
            continue
        begin = max(now, s["earliest"]); wait += begin - now; now = begin
        late_packages += sum(now > deadline for deadline in s["ends"])
        total_packages += s["packages"]; service += s["service"]; now += s["service"]
    return dict(travel_seconds=travel_time, service_seconds=service, wait_seconds=wait,
                route_seconds=now, late_packages=late_packages, packages=total_packages)


def routing_eval(routes, actual, info, test_ids):
    path = OUT / "amazon_parcel_routing_detail.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["route_id"] for r in rows}
    completed = len(done)
    print(f"routing checkpoint: {completed:,}/{len(test_ids):,} routes complete", flush=True)
    with (DATA / "travel_times.json").open("rb") as f:
        for n, (rid, tt) in enumerate(ijson.kvitems(f, ""), 1):
            if rid in done or rid not in test_ids or rid not in actual or rid not in info:
                continue
            route, st = routes[rid], info[rid]["stops"]
            observed = sequence_actual(route, actual[rid])
            start = next((s for s in observed if route["stops"][s]["type"] == "Station"), observed[0])
            candidates = [s for s in st if s != start and route["stops"].get(s, {}).get("type") == "Dropoff"]
            if len(candidates) < 2 or not all(s in tt for s in observed):
                continue
            policies = {"Observed driver sequence": observed,
                        "Nearest-neighbour travel time": greedy_nn(start, candidates, tt, st),
                        "Time-window-aware greedy": greedy_tw(start, candidates, tt, st)}
            util = info[rid]["total_volume"] / max(route["executor_capacity_cm3"], 1.)
            for name, order in policies.items():
                r = simulate(order, tt, st)
                r.update(route_id=rid, policy=name, date=route["date_YYYY_MM_DD"],
                         route_score=route["route_score"], capacity_utilization=util,
                         capacity_violation=util > 1.)
                rows.append(r)
            completed += 1
            if completed % 100 == 0:
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"routing routes evaluated: {completed:,}", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    routes, actual = read_json("route_data.json"), read_json("actual_sequences.json")
    cut, _, test_ids = route_split(routes)
    print(f"routes={len(routes)}; chronological train through {cut}; test routes={len(test_ids)}")
    daily, info = load_or_build_summaries(routes, test_ids)
    detail, cap = capacity_eval(routes, daily, cut)
    detail.to_csv(OUT / "amazon_parcel_capacity_detail.csv", index=False)
    cap.to_csv(OUT / "amazon_parcel_capacity_summary.csv", index=False)
    routing = routing_eval(routes, actual, info, test_ids)
    summary = routing.groupby("policy").agg(n_routes=("route_id", "nunique"),
        travel_hours=("travel_seconds", lambda x: x.mean()/3600),
        service_hours=("service_seconds", lambda x: x.mean()/3600),
        route_hours=("route_seconds", lambda x: x.mean()/3600),
        late_package_pct=("late_packages", lambda x: 100*x.sum()/max(1, routing.loc[x.index, "packages"].sum())),
        mean_capacity_utilization=("capacity_utilization", "mean"),
        capacity_violation_pct=("capacity_violation", "mean")).reset_index()
    summary["capacity_violation_pct"] *= 100
    summary.to_csv(OUT / "amazon_parcel_routing_summary.csv", index=False)
    print("\nCapacity:\n", cap.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nRouting:\n", summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
