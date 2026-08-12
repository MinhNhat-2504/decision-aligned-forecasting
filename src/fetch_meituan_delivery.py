"""Fetch the public Meituan/INFORMS operational food-delivery data locally.

This source is the public 2024--25 TSL-Meituan Research Challenge repository.
It is CC BY-NC 4.0 and explicitly requires the acknowledgement ``This research
was supported by data provided by Meituan.''  The files remain local: the licence
does not permit redistributing the data or derivatives.
"""
from pathlib import Path
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "meituan"
BASE = "https://raw.githubusercontent.com/meituan/Meituan-INFORMS-TSL-Research-Challenge/main"
FILES = [
    "all_waybill_info_meituan_0322.csv.zip",
    "courier_wave_info_meituan.csv",
    "dispatch_rider_meituan.csv",
    "dispatch_waybill_meituan.csv",
    "TSL-Meituan challenge_background and data_20240321.pdf",
    "License.txt",
]


def fetch(name):
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 100:
        print(f"complete {name}: {dest.stat().st_size / 1e6:.1f} MB")
        return
    print(f"downloading {name}", flush=True)
    with requests.get(f"{BASE}/{requests.utils.quote(name)}", stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for block in r.iter_content(1024 * 1024):
                if block:
                    f.write(block)
    print(f"saved {dest.stat().st_size / 1e6:.1f} MB")


def main():
    for name in FILES:
        fetch(name)
    archive = OUT / "all_waybill_info_meituan_0322.csv.zip"
    if not zipfile.is_zipfile(archive):
        raise RuntimeError("waybill download is not a zip archive")
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX/")]
        if len(names) != 1:
            raise RuntimeError(f"unexpected archive members: {z.namelist()}")
        name = names[0]; dest = OUT / name
        if not dest.exists() or dest.stat().st_size != z.getinfo(name).file_size:
            print(f"extracting {name}")
            z.extract(name, OUT)
    print("Meituan extraction complete")


if __name__ == "__main__": main()
