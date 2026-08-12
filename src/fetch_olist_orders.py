"""Fetch the public Olist Brazilian E-Commerce parcel-order data.

The source is Kaggle's official Olist release (CC BY-NC-SA 4.0).  The archive
contains real, anonymised purchase timestamps, customer geography, order items,
and product dimensions.  We keep only the five CSVs needed for the chronological
order-arrival-to-capacity study.
"""
from pathlib import Path
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "olist"
URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
NEEDED = {
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_geolocation_dataset.csv",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "brazilian-ecommerce.zip"
    if not archive.exists() or archive.stat().st_size < 1_000_000:
        print("Downloading official Olist archive from Kaggle...", flush=True)
        with requests.get(URL, stream=True, timeout=(30, 300), headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            with archive.open("wb") as f:
                for block in r.iter_content(1024 * 1024):
                    if block:
                        f.write(block)
        print(f"Downloaded {archive.stat().st_size / 1e6:.1f} MB")
    if not zipfile.is_zipfile(archive):
        raise RuntimeError("Kaggle response is not a zip archive; remove it and rerun with Kaggle access.")
    with zipfile.ZipFile(archive) as z:
        available = set(z.namelist())
        missing = NEEDED - available
        if missing:
            raise RuntimeError(f"Archive misses expected files: {sorted(missing)}")
        for name in sorted(NEEDED):
            dest = OUT / name
            if not dest.exists() or dest.stat().st_size != z.getinfo(name).file_size:
                print(f"Extracting {name}")
                with z.open(name) as src, dest.open("wb") as dst:
                    while block := src.read(1024 * 1024):
                        dst.write(block)
    print("Olist extraction complete")


if __name__ == "__main__":
    main()
