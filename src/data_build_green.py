"""
Bo du lieu THU HAI: NYC green taxi, 2016 Q1.

Vi sao la mot bo du lieu that su khac chu khong phai bien the:
  - Green taxi (Boro Taxi) bi CAM don khach o Manhattan duoi 96th St va tai san bay.
    Phan bo khong gian cua nhu cau vi vay lech han ve Bronx/Brooklyn/Queens,
    trong khi yellow taxi tap trung o Manhattan.
  - Quy mo nho hon mot bac (~4,3 trieu so voi 34,5 trieu chuyen).
  - Cung 263 vung, cung khung thoi gian, cung do phan giai gio
    -> nen khong gian giu nguyen, chi CO CHE SINH NHU CAU la khac.

Day la phep thu tot: neu ket luan cua bai bao chi dung cho yellow taxi thi no khong
phai ket luan ve phuong phap.

Ban phat hanh lai cua TLC da co san PULocationID/DOLocationID nen khong can
point-in-polygon.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROC_DIR, T_START, T_END, TRAIN_END, K_FLOW

RAW2 = os.path.join(os.path.dirname(PROC_DIR), "data", "raw2")
FILES = [f"green_2016-{m}.parquet" for m in ("01", "02", "03")]


def main():
    zones = pd.read_csv(os.path.join(PROC_DIR, "zones.csv"))
    N = len(zones)
    # LocationID -> chi so hang trong zones.csv (giu dung thu tu voi bo yellow)
    loc2idx = {int(z): i for i, z in enumerate(zones["zone_id"].values)}

    ts = pd.date_range(T_START, T_END, freq="h")
    T = len(ts)
    t0 = ts[0]
    n_tr = len(pd.date_range(T_START, TRAIN_END, freq="h"))

    pickup = np.zeros((T, N), dtype=np.float32)
    dropoff = np.zeros((T, N), dtype=np.float32)
    od_train = np.zeros((N, N), dtype=np.float64)
    tot = kept = 0

    for f in FILES:
        p = os.path.join(RAW2, f)
        df = pd.read_parquet(p, columns=[
            "lpep_pickup_datetime", "lpep_dropoff_datetime",
            "PULocationID", "DOLocationID", "trip_distance"])
        tot += len(df)
        df = df[(df.trip_distance > 0) & (df.trip_distance < 100)]

        pu_h = ((pd.to_datetime(df.lpep_pickup_datetime, errors="coerce") - t0)
                .dt.total_seconds() // 3600).to_numpy()
        do_h = ((pd.to_datetime(df.lpep_dropoff_datetime, errors="coerce") - t0)
                .dt.total_seconds() // 3600).to_numpy()
        pz = np.array([loc2idx.get(int(v), -1) if np.isfinite(v) else -1
                       for v in df.PULocationID.to_numpy(np.float64)])
        dz = np.array([loc2idx.get(int(v), -1) if np.isfinite(v) else -1
                       for v in df.DOLocationID.to_numpy(np.float64)])

        m = (pz >= 0) & np.isfinite(pu_h) & (pu_h >= 0) & (pu_h < T)
        np.add.at(pickup, (pu_h[m].astype(np.int64), pz[m]), 1.0)
        m2 = (dz >= 0) & np.isfinite(do_h) & (do_h >= 0) & (do_h < T)
        np.add.at(dropoff, (do_h[m2].astype(np.int64), dz[m2]), 1.0)
        m3 = m & m2 & (pu_h < n_tr)
        if m3.any():
            np.add.at(od_train, (pz[m3], dz[m3]), 1.0)
        kept += int(m2.sum())
        print(f"  {f}: {len(df):,} chuyen hop le, tich luy dropoff {kept:,}",
              flush=True)

    np.savez_compressed(
        os.path.join(PROC_DIR, "demand_hourly_green.npz"),
        pickup=pickup, dropoff=dropoff,
        timestamps=ts.astype("int64").to_numpy(),
        od_train=od_train.astype(np.float32))

    # --- Do thi: geo/road giu nguyen (cung vung); flow phai tinh lai tu OD green ---
    g = np.load(os.path.join(PROC_DIR, "graphs.npz"))
    W = od_train + od_train.T
    np.fill_diagonal(W, 0)
    s, t = [], []
    for i in range(N):
        for j in np.argsort(-W[i])[:K_FLOW]:
            if W[i, j] > 0:
                s.append(i)
                t.append(int(j))
    pairs = set()
    for a, b in zip(s, t):
        pairs.add((a, b))
        pairs.add((b, a))
    arr = np.array(sorted(pairs), dtype=np.int64).T

    np.savez_compressed(os.path.join(PROC_DIR, "graphs_green.npz"),
                        geo=g["geo"], road=g["road"], flow=arr,
                        dist_geo=g["dist_geo"], dist_road=g["dist_road"])

    d_tot = dropoff.sum(0)
    y_tot = np.load(os.path.join(PROC_DIR, "demand_hourly.npz"))["dropoff"].sum(0)
    print(f"\nTong chuyen doc: {tot:,} | dropoff gan duoc vung: {kept:,}")
    print(f"Dropoff/vung: TB={d_tot.mean():.0f} trung vi={np.median(d_tot):.0f} "
          f"max={d_tot.max():.0f} vung rong={int((d_tot == 0).sum())}")
    print(f"Canh flow (green): {arr.shape[1]} | (yellow): {g['flow'].shape[1]}")

    # So sanh phan bo khong gian giua hai dich vu
    zb = zones.copy()
    zb["green"], zb["yellow"] = d_tot, y_tot
    by = zb.groupby("borough")[["green", "yellow"]].sum()
    by = 100 * by / by.sum()
    print("\nTy trong nhu cau theo borough (%):")
    print(by.to_string(float_format=lambda x: f"{x:6.2f}"))
    corr = np.corrcoef(np.log1p(d_tot), np.log1p(y_tot))[0, 1]
    print(f"\ncorr(log green, log yellow) theo vung = {corr:.3f}")


if __name__ == "__main__":
    main()
