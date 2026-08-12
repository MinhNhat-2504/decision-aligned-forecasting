"""Small external-city validation using the official Chicago Taxi Trips API.

Downloads a bounded sample through the City of Chicago Socrata endpoint, aggregates
hourly pickup demand by community area, and compares a training-only historical
median/0.9-quantile policy on the later chronological block. This is intentionally
an external validation of the decision mechanism, not a claim that NYC graph
topology transfers unchanged to Chicago.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "chicago_sample.csv"
OUT = ROOT / "outputs" / "external_chicago_results.csv"
URL = "https://data.cityofchicago.org/resource/r2u4-wwk3.json"


def download():
    import requests
    params = {"$limit": 250000,
              "$select": "trip_start_timestamp,pickup_community_area",
              "$where": "pickup_community_area is not null and trip_start_timestamp is not null",
              "$order": "trip_start_timestamp"}
    r = requests.get(URL, params=params, timeout=180)
    r.raise_for_status()
    DATA.write_text(json.dumps(r.json()), encoding="utf-8")
    print(f"Downloaded {len(r.json())} records -> {DATA}")


def main():
    if not DATA.exists():
        download()
    raw = pd.read_json(DATA)
    raw["ts"] = pd.to_datetime(raw["trip_start_timestamp"], errors="coerce")
    raw["zone"] = raw["pickup_community_area"].astype(str)
    raw = raw.dropna(subset=["ts"])
    tab = raw.set_index("ts").groupby("zone").resample("h").size().unstack(0, fill_value=0)
    tab = tab.sort_index().fillna(0).astype(float)
    if len(tab) < 48:
        raise RuntimeError("Chicago sample has too few hourly observations")
    cut = int(len(tab) * 0.7)
    tr, te = tab.iloc[:cut], tab.iloc[cut:]
    # Week-hour empirical functionals estimated only on training observations.
    key = tr.index.dayofweek * 24 + tr.index.hour
    base = tr.groupby(key).agg(["median", lambda x: x.quantile(.9)])
    med, q90 = [], []
    for ts in te.index:
        row = base.loc[ts.dayofweek * 24 + ts.hour]
        med.append(row.xs("median", level=1).values)
        q90.append(row.xs("<lambda_0>", level=1).values)
    y = te.values
    rows = []
    for name, a in (("historical_median", np.asarray(med)), ("historical_q90", np.asarray(q90))):
        tau = .9; short = np.maximum(y - a, 0); over = np.maximum(a - y, 0)
        cost = (tau * short + (1-tau) * over).mean()
        rows.append(dict(dataset="Chicago taxi 2020 sample", policy=name,
                         hours=len(te), zones=te.shape[1], cost=float(cost),
                         fill_rate=float(100 * (1-short.sum()/(y.sum()+1e-9))),
                         unmet_pct=float(100 * short.sum()/(y.sum()+1e-9))))
    out = pd.DataFrame(rows); OUT.parent.mkdir(exist_ok=True); out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__": main()
