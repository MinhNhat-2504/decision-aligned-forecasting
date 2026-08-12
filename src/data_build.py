"""
Xay dung bo du lieu KHONG-THOI GIAN that tu NYC TLC yellow taxi 2016 Q1.

Khac biet co ban so voi ban goc:
  - Nhan y(z,t) = so su kien trong zone z tai GIO t  -> khong con la ham cua dac trung dau vao.
  - Gan diem vao zone bang point-in-polygon (shapely STRtree), khong dung "centroid gan nhat".
  - Centroid tinh bang shapely .centroid, khong lay trung binh so trong chuoi WKT.
  - Ma tran OD chi tinh tren khoang thoi gian TRAIN de tranh ro ri sang val/test.
"""
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import shapely
from shapely import wkt, points, STRtree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ZIP_PATH, ZONES_MEMBER, TRIP_MEMBERS, PROC_DIR,
    T_START, T_END, TRAIN_END,
)

CHUNK = 1_000_000
LON_MIN, LON_MAX = -74.30, -73.65
LAT_MIN, LAT_MAX = 40.45, 40.95


def load_zones(zf):
    """Doc taxi zones, tra ve dataframe + danh sach polygon theo dung thu tu index."""
    zones = pd.read_csv(zf.open(ZONES_MEMBER))
    zones = zones.rename(columns={"LocationID": "zone_id", "zone": "zone_name"})
    geoms = [wkt.loads(g) for g in zones["the_geom"]]

    # Centroid hinh hoc that su (khong phai trung binh cac dinh polygon)
    cents = [g.centroid for g in geoms]
    zones["lon"] = [c.x for c in cents]
    zones["lat"] = [c.y for c in cents]
    # Dien tich xap xi km^2 (quy doi do -> km tai vi do NYC)
    kx = 111.320 * np.cos(np.radians(40.75))
    ky = 110.574
    zones["area_km2"] = [g.area * kx * ky for g in geoms]

    zones = zones[["zone_id", "zone_name", "borough", "lat", "lon", "area_km2"]]
    zones = zones.reset_index(drop=True)
    zones["idx"] = np.arange(len(zones))
    return zones, geoms


def assign_zone(tree, lon, lat, n_zones):
    """Point-in-polygon vector hoa. Tra ve mang zone-index, -1 neu nam ngoai moi zone."""
    valid = (
        np.isfinite(lon) & np.isfinite(lat)
        & (lon > LON_MIN) & (lon < LON_MAX)
        & (lat > LAT_MIN) & (lat < LAT_MAX)
    )
    out = np.full(len(lon), -1, dtype=np.int32)
    if not valid.any():
        return out
    vidx = np.flatnonzero(valid)
    pts = points(lon[vidx], lat[vidx])
    ipt, igeom = tree.query(pts, predicate="within")
    if len(ipt):
        # Neu 1 diem roi vao nhieu polygon (bien chung) -> giu match dau tien
        first = np.ones(len(ipt), dtype=bool)
        first[1:] = ipt[1:] != ipt[:-1]
        out[vidx[ipt[first]]] = igeom[first].astype(np.int32)
    return out


def build():
    t_index = pd.date_range(T_START, T_END, freq="h")
    T, n_hours_train = len(t_index), len(pd.date_range(T_START, TRAIN_END, freq="h"))
    t0_ts = t_index[0]

    zf = zipfile.ZipFile(ZIP_PATH)
    zones, geoms = load_zones(zf)
    N = len(zones)
    tree = STRtree(geoms)
    print(f"Zones: {N} | Hours: {T} ({t_index[0]} -> {t_index[-1]}) | train hours: {n_hours_train}")

    pickup = np.zeros((T, N), dtype=np.float32)
    dropoff = np.zeros((T, N), dtype=np.float32)
    od_train = np.zeros((N, N), dtype=np.float64)

    stats = dict(rows=0, kept=0, out_of_zone=0, out_of_window=0)
    t_start = time.time()

    usecols = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "trip_distance",
    ]

    for member in TRIP_MEMBERS:
        print(f"\n--- {os.path.basename(member)} ---", flush=True)
        reader = pd.read_csv(
            zf.open(member), usecols=usecols, chunksize=CHUNK,
            dtype={c: np.float32 for c in usecols if c != "tpep_pickup_datetime"
                   and c != "tpep_dropoff_datetime"},
        )
        for ci, ch in enumerate(reader):
            stats["rows"] += len(ch)

            # Loc ban ghi vo ly truoc khi tinh toan nang
            ch = ch[(ch["trip_distance"] > 0) & (ch["trip_distance"] < 100)]
            if not len(ch):
                continue

            pu_t = pd.to_datetime(ch["tpep_pickup_datetime"], errors="coerce")
            do_t = pd.to_datetime(ch["tpep_dropoff_datetime"], errors="coerce")

            pu_h = ((pu_t - t0_ts).dt.total_seconds() // 3600).to_numpy()
            do_h = ((do_t - t0_ts).dt.total_seconds() // 3600).to_numpy()

            pz = assign_zone(tree, ch["pickup_longitude"].to_numpy(np.float64),
                             ch["pickup_latitude"].to_numpy(np.float64), N)
            dz = assign_zone(tree, ch["dropoff_longitude"].to_numpy(np.float64),
                             ch["dropoff_latitude"].to_numpy(np.float64), N)

            # --- Dem pickup ---
            m = (pz >= 0) & np.isfinite(pu_h) & (pu_h >= 0) & (pu_h < T)
            stats["out_of_zone"] += int((pz < 0).sum())
            np.add.at(pickup, (pu_h[m].astype(np.int64), pz[m]), 1.0)

            # --- Dem dropoff (proxy nhu cau giao hang den zone) ---
            m2 = (dz >= 0) & np.isfinite(do_h) & (do_h >= 0) & (do_h < T)
            np.add.at(dropoff, (do_h[m2].astype(np.int64), dz[m2]), 1.0)

            # --- OD flow: CHI tren khoang train ---
            m3 = m & m2 & (pu_h < n_hours_train)
            if m3.any():
                np.add.at(od_train, (pz[m3], dz[m3]), 1.0)

            stats["kept"] += int(m.sum())
            if ci % 3 == 0:
                el = time.time() - t_start
                print(f"  chunk {ci:>3} | rows={stats['rows']:>10,} "
                      f"kept={stats['kept']:>10,} | {el:6.1f}s", flush=True)

    el = time.time() - t_start
    print(f"\nDone in {el:.1f}s. rows={stats['rows']:,} kept_pickups={stats['kept']:,} "
          f"out_of_zone={stats['out_of_zone']:,}")

    np.savez_compressed(
        os.path.join(PROC_DIR, "demand_hourly.npz"),
        pickup=pickup, dropoff=dropoff,
        timestamps=t_index.astype("int64").to_numpy(),
        od_train=od_train.astype(np.float32),
    )
    zones.to_csv(os.path.join(PROC_DIR, "zones.csv"), index=False)

    tot = dropoff.sum(0)
    print(f"\nDropoff/zone: mean={tot.mean():.0f} median={np.median(tot):.0f} "
          f"max={tot.max():.0f} zeros={int((tot == 0).sum())}")
    print(f"Hourly total: mean={dropoff.sum(1).mean():.0f} min={dropoff.sum(1).min():.0f}")
    print(f"Saved -> {PROC_DIR}")


if __name__ == "__main__":
    build()
