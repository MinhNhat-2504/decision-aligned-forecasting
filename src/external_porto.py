"""External-city decision validation on the UCI Porto taxi trajectory data.

The original UCI archive supplies individual taxi trips rather than ready-made
zone demand.  This script bins the pickup coordinate (the first polyline point)
into a fixed 8 x 8 grid, aggregates hourly trips, then evaluates training-only
week-hour median and 0.9-quantile stocking policies chronologically.  It is a
deliberately conservative *external decision* check, not a topology-transfer
claim for the NYC neural graph.
"""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "porto_train.csv.zip"
OUT = ROOT / "outputs" / "external_porto_results.csv"
BOOT_OUT = ROOT / "outputs" / "external_porto_block_bootstrap.csv"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00339/train.csv.zip"


def download():
    import requests
    ARCHIVE.parent.mkdir(exist_ok=True)
    # The UCI archive is large.  Preserve an interrupted partial transfer and
    # request the remaining byte range on the next invocation.
    offset = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(URL, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        resumed = offset > 0 and response.status_code == 206
        print(f"HTTP {response.status_code}; offset={offset / 1e6:.1f} MB; resumed={resumed}", flush=True)
        mode = "ab" if resumed else "wb"
        written = offset if resumed else 0
        with ARCHIVE.open(mode) as f:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    f.write(block)
                    written += len(block)
                    if written and written % (25 * 1024 * 1024) < len(block):
                        print(f"received {written / 1e6:.1f} MB", flush=True)
    if not zipfile.is_zipfile(ARCHIVE):
        raise RuntimeError("Transfer ended before a valid ZIP archive was received; rerun --download to resume")
    print(f"Downloaded {ARCHIVE} ({ARCHIVE.stat().st_size / 1e6:.1f} MB)")


def aggregate(max_rows=None):
    parts, seen = [], 0
    with zipfile.ZipFile(ARCHIVE) as zf:
        member = next(n for n in zf.namelist() if n.endswith("train.csv"))
        with zf.open(member) as raw:
            for chunk_no, chunk in enumerate(pd.read_csv(io.TextIOWrapper(raw, encoding="utf-8"),
                                     usecols=["TIMESTAMP", "MISSING_DATA", "POLYLINE"],
                                     chunksize=100_000), start=1):
                chunk = chunk[~chunk["MISSING_DATA"]].copy()
                # Extract only the pickup coordinate. Vectorised regex is much
                # faster and more auditable than evaluating 1.7M Python literals.
                coord = chunk["POLYLINE"].str.extract(
                    r"^\[\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", expand=True).astype(float)
                chunk["lon"], chunk["lat"] = coord[0], coord[1]
                chunk = chunk[chunk.lon.between(-8.75, -8.50) & chunk.lat.between(41.05, 41.28)].copy()
                x = np.minimum(7, ((chunk.lon + 8.75) / .25 * 8).astype(int))
                y = np.minimum(7, ((chunk.lat - 41.05) / .23 * 8).astype(int))
                chunk["zone"] = "z" + x.astype(str) + "_" + y.astype(str)
                chunk["hour"] = pd.to_datetime(chunk["TIMESTAMP"], unit="s", utc=True).dt.floor("h")
                parts.append(chunk.groupby(["hour", "zone"]).size())
                seen += len(chunk)
                if chunk_no % 5 == 0:
                    print(f"processed {chunk_no * 100_000:,} rows; kept {seen:,}", flush=True)
                if max_rows and seen >= max_rows:
                    break
    long = pd.concat(parts).groupby(level=[0, 1]).sum()
    table = long.unstack("zone", fill_value=0).sort_index()
    return table.asfreq("h", fill_value=0)


def evaluate(table):
    if len(table) < 24 * 14:
        raise RuntimeError("Porto aggregation has fewer than two weeks of usable hours")
    cut = int(len(table) * .7)
    train, test = table.iloc[:cut], table.iloc[cut:]
    key = train.index.dayofweek * 24 + train.index.hour
    med = train.groupby(key).median()
    q90 = train.groupby(key).quantile(.9)
    y = test.to_numpy(float)
    policies = {
        "historical_median": np.asarray([med.loc[t.dayofweek * 24 + t.hour] for t in test.index]),
        "historical_q90": np.asarray([q90.loc[t.dayofweek * 24 + t.hour] for t in test.index]),
    }
    rows, losses, shortages = [], {}, {}
    for name, action in policies.items():
        shortage, overage = np.maximum(y - action, 0), np.maximum(action - y, 0)
        loss = .9 * shortage + .1 * overage
        losses[name], shortages[name] = loss, shortage
        rows.append({"dataset": "Porto taxi trajectories (UCI)", "policy": name,
                     "hours": len(test), "zones": table.shape[1],
                     "q90_cost": float(loss.mean()),
                     "fill_rate_pct": float(100 * (1 - shortage.sum() / (y.sum() + 1e-9))),
                     "unmet_pct": float(100 * shortage.sum() / (y.sum() + 1e-9))})
    result = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))

    # Resample complete calendar days, not correlated zone-hours, for an external
    # decision interval.  The statistic is the reduction in the realised q0.9
    # objective from replacing the median policy with the q0.9 policy.
    day = pd.factorize(test.index.normalize())[0]
    n_day = day.max() + 1
    def daily(a):
        return np.bincount(day, weights=a.sum(1), minlength=n_day) / table.shape[1]
    med, q90 = daily(losses["historical_median"]), daily(losses["historical_q90"])
    rng = np.random.default_rng(20260729)
    draws = rng.integers(0, n_day, size=(2000, n_day))
    reduction = 100 * (med[draws].mean(1) - q90[draws].mean(1)) / med[draws].mean(1)
    boot = pd.DataFrame([{
        "dataset": "Porto taxi trajectories (UCI)", "block": "calendar day",
        "n_days": int(n_day), "draws": len(draws),
        "q90_vs_median_cost_reduction_pct": float(100 * (med.mean() - q90.mean()) / med.mean()),
        "ci95_low": float(np.quantile(reduction, .025)),
        "ci95_high": float(np.quantile(reduction, .975)),
    }])
    boot.to_csv(BOOT_OUT, index=False)
    print(boot.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Optional development cap; do not use for final results.")
    args = ap.parse_args()
    if args.download and (not ARCHIVE.exists() or not zipfile.is_zipfile(ARCHIVE)):
        download()
    if not ARCHIVE.exists() or not zipfile.is_zipfile(ARCHIVE):
        raise FileNotFoundError(f"Missing {ARCHIVE}; rerun with --download")
    evaluate(aggregate(args.max_rows))


if __name__ == "__main__":
    main()
