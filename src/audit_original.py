"""
Kiem chung dinh luong cac loi cua pipeline goc (GAT_LastMile_Delivery).

Chay tren chinh cac tep ma ban goc sinh ra (nodes.csv / training_log.csv) de cho thay
con so R2 = 0.8084 trong bao cao khong phan anh nang luc du bao nao.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Thu muc du an GOC (truoc khi lam lai). Dat qua bien moi truong:
ORIG = os.environ.get(
    "GAT_ORIG", r"d:/Project/AI Engineer/Paper1/work/GAT_LastMile_Delivery")


def main():
    n = pd.read_csv(os.path.join(ORIG, "data/processed/nodes.csv"))
    log = pd.read_csv(os.path.join(ORIG, "outputs/training_log.csv"))

    print("=" * 72)
    print("KIEM CHUNG PIPELINE GOC")
    print("=" * 72)

    # --- 1. Nhan la ham xac dinh cua mot dac trung dau vao -------------------
    feats = ["latitude", "longitude", "population_score",
             "historical_trip_count", "centrality_score"]
    y = n["demand"].values
    h = n["historical_trip_count"].values

    print("\n[1] RO RI NHAN (target leakage)")
    print(f"    demand := historical_trip_count / max  -> dung y het: "
          f"{np.allclose(y, h / h.max())}")
    print(f"    corr(demand, historical_trip_count) = {np.corrcoef(y, h)[0, 1]:.10f}")
    print(f"    'historical_trip_count' NAM TRONG feature_cols: "
          f"{'historical_trip_count' in feats}")
    print(f"    'population_score' = log1p(cung bien do): "
          f"{np.allclose(n['population_score'], np.log1p(h))}")

    lr = LinearRegression().fit(h.reshape(-1, 1), y)
    print(f"    -> hoi quy tuyen tinh 1 bien dat R2 = {lr.score(h.reshape(-1, 1), y):.10f}")
    print("    Nghia la bai toan la phep chia da biet truoc, khong phai du bao.")
    print(f"    Mo hinh GAT chi dat R2 = {log['r2'].iloc[-1]:.4f} tren chinh bai toan do.")

    # --- 2. Khong he co tap kiem tra ---------------------------------------
    print("\n[2] KHONG CO TAP TEST")
    print(f"    So node trong do thi          : {len(n)}")
    print("    train.py: compute_metrics(y, predictions) tren TOAN BO node,")
    print("    khong co mask train/val/test -> moi chi so deu la tren tap huan luyen.")
    print(f"    Bao cao goi day la 'tren bo du lieu thu nghiem' (R2={log['r2'].iloc[-1]:.4f}).")

    # --- 3. Dac trung chet --------------------------------------------------
    print("\n[3] DAC TRUNG VO NGHIA")
    print(f"    centrality_score: cac gia tri = {np.unique(n['centrality_score'])} "
          f"(hang so)")
    print("    sau chuan hoa (x-mean)/std -> dung bang 0 o moi node.")
    print("    Bao cao van liet ke 'muc do ket noi' la dac trung dau vao.")

    # --- 4. Chi so bi che giau ---------------------------------------------
    print("\n[4] CHI SO KHONG DUOC BAO CAO")
    print(f"    sMAPE cuoi cung = {log['smape_%'].iloc[-1]:.2f}%  "
          f"(toi da co the la 200%)")
    print("    Chi so nay co trong training_log.csv nhung khong xuat hien trong bao cao.")
    print("    No cho thay sai so tuong doi rat lon o phan lon zone.")

    pred = pd.read_csv(os.path.join(ORIG, "outputs/predictions.csv"))
    neg = (pred["predicted_demand"] < 0).sum()
    print(f"    So zone bi du bao nhu cau AM: {neg}/{len(pred)} "
          f"(min = {pred['predicted_demand'].min():.4f})")

    # --- 5. Phan phoi lech -> R2 bi vai zone chi phoi ----------------------
    print("\n[5] R2 BI CHI PHOI BOI VAI ZONE LON")
    q = np.percentile(h, [50, 90, 99])
    print(f"    historical_trip_count: median={q[0]:.0f} p90={q[1]:.0f} "
          f"p99={q[2]:.0f} max={h.max():.0f}")
    top = np.argsort(-h)[:10]
    print(f"    10/{len(n)} zone lon nhat chiem {100 * h[top].sum() / h.sum():.1f}% "
          f"tong luu luong")
    print("    => R2 tren thang do tuyet doi gan nhu chi do vai zone Manhattan.")

    # --- 6. Mau du lieu -----------------------------------------------------
    print("\n[6] QUY MO MAU")
    print("    build_dataset_real.py: sample_rows=150_000, doc tuan tu roi break")
    print("    -> chi lay ~150k dong DAU TIEN cua yellow_tripdata_2015-01.csv")
    print("    (tuc vai gio dau ngay 01/01/2015 - dung dip Tet duong lich).")
    print("    Trong khi zip chua 4 thang du lieu, ~34,5 trieu chuyen.")
    print("=" * 72)


if __name__ == "__main__":
    main()
