"""Resumable fetcher for the public Amazon Last Mile challenge training data.

The source is the CC BY-NC 4.0 Amazon/MIT challenge bucket.  We retain only the
four files required for a reproducible parcel-routing evaluation: route and package
attributes, observed stop sequences, and route-specific realised travel times.
"""
import argparse
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "amazon_lastmile"
BASE = "https://amazon-last-mile-challenges.s3.us-west-2.amazonaws.com/almrrc2021/almrrc2021-data-training/model_build_inputs"
FILES = {
    "route_data.json": 78_972_162,
    "package_data.json": 375_437_806,
    "actual_sequences.json": 9_665_078,
    "travel_times.json": 1_817_146_363,
}


def fetch(name, expected):
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / name
    offset = dest.stat().st_size if dest.exists() else 0
    if offset == expected:
        print(f"complete {name}: {offset / 1e6:.1f} MB")
        return
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(f"{BASE}/{name}", headers=headers, stream=True, timeout=(30, 180)) as r:
        r.raise_for_status()
        resume = offset > 0 and r.status_code == 206
        mode, done = ("ab", offset) if resume else ("wb", 0)
        print(f"{name}: HTTP {r.status_code}; starting at {done / 1e6:.1f} MB", flush=True)
        with dest.open(mode) as f:
            for block in r.iter_content(1024 * 1024):
                if block:
                    f.write(block); done += len(block)
                    if done % (100 * 1024 * 1024) < len(block):
                        print(f"  {name}: {done / 1e6:.1f}/{expected / 1e6:.1f} MB", flush=True)
    if dest.stat().st_size != expected:
        raise RuntimeError(f"Incomplete {name}: {dest.stat().st_size} of {expected}; rerun to resume")
    print(f"complete {name}: {expected / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--without-travel", action="store_true", help="Download metadata first only.")
    args = ap.parse_args()
    for name, size in FILES.items():
        if args.without_travel and name == "travel_times.json":
            continue
        fetch(name, size)


if __name__ == "__main__":
    main()
