"""Fetch the long-horizon Cainiao-LaDe parcel delivery and courier GPS data.

LaDe is Apache-2.0 on the official Cainiao-AI Hugging Face repository.  We retain
only the five-city delivery events, their task-event companion, and 20-second
courier GPS trajectories.  The source has no physical vehicle-capacity field, so
downstream analysis must use it for parcel/GPS linkage, never for measured vehicle
capacity.
"""
from pathlib import Path
import argparse
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lade"
BASE = "https://huggingface.co/datasets/Cainiao-AI/LaDe/resolve/main"
FILES = {
    "delivery_five_cities.csv": 135_539_702,
    "data_with_trajectory_20s/delivery_five_cities.pkl.xz": 34_028_796,
    "data_with_trajectory_20s/courier_detailed_trajectory_20s.pkl.xz": 438_656_312,
}
FULL_SHANGHAI = {"delivery/delivery_sh.csv": 228_069_571}


def fetch(name, expected):
    dest = OUT / Path(name).name; dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected:
        print(f"complete {dest.name}: {expected/1e6:.1f} MB")
        return
    offset = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(f"{BASE}/{name}", headers=headers, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        append = offset and r.status_code == 206
        mode, done = ("ab", offset) if append else ("wb", 0)
        print(f"{dest.name}: HTTP {r.status_code}, starting {done/1e6:.1f} MB", flush=True)
        with dest.open(mode) as f:
            for block in r.iter_content(1024 * 1024):
                if block:
                    f.write(block); done += len(block)
                    if done % (100 * 1024 * 1024) < len(block): print(f"  {done/1e6:.1f}/{expected/1e6:.1f} MB", flush=True)
    if dest.stat().st_size != expected:
        raise RuntimeError(f"incomplete {dest}: {dest.stat().st_size} of {expected}; rerun to resume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-full-shanghai", action="store_true", help="Add the full long-horizon Shanghai delivery file.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items(): fetch(name, expected)
    if args.with_full_shanghai:
        for name, expected in FULL_SHANGHAI.items(): fetch(name, expected)
    print("LaDe parcel/GPS download complete")


if __name__ == "__main__": main()
