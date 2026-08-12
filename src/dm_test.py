"""
Kiem dinh Diebold-Mariano (1995) + hieu chinh mau nho Harvey-Leybourne-Newbold (1997).

Vi sao can:
  Kiem dinh t ghep cap trong train_eval.py so sanh TRUNG BINH MAE giua cac seed,
  tuc chi co 5 quan sat -> luc thong ke rat yeu, khong the ket luan "khong khac nhau"
  chi vi p > 0.05. DM so sanh tren CHUOI SAI SO theo thoi gian (379 cua so), la
  chuan cua gioi du bao, va xu ly duoc tu tuong quan chuoi bang phuong sai HAC.

Thiet ke:
  - Moi cua so t cho mot sai so trung binh tren 263 vung -> chuoi d_t do dai 379.
    Gop theo khong gian truoc de khong coi 263 vung la quan sat doc lap
    (chung tuong quan manh) - neu khong se thoi phong luc kiem dinh.
  - Phuong sai Newey-West voi do tre tu dong.
  - CHI dung du bao cua seed 0 (da luu). DM so sanh hai CHUOI du bao cu the.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, HORIZONS, QUANTILES


def newey_west_var(d, lag=None):
    """Phuong sai dai han cua trung binh mau, co hieu chinh tu tuong quan."""
    T = len(d)
    if lag is None:
        lag = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    d = d - d.mean()
    g0 = float((d @ d) / T)
    s = g0
    for k in range(1, lag + 1):
        gk = float((d[k:] @ d[:-k]) / T)
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return max(s, 1e-12), lag


def dm_test(e1, e2, h=1, power=1):
    """
    e1, e2: [T] chuoi sai so (da gop theo vung) cua hai mo hinh.
    Tra ve (DM*, p, lag) voi DM* la thong ke da hieu chinh HLN.
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    T = len(d)
    var, lag = newey_west_var(d, lag=max(h - 1, None) if h > 1 else None)
    dm = d.mean() / np.sqrt(var / T)
    # Hieu chinh mau nho Harvey-Leybourne-Newbold
    k = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_star = dm * k
    p = 2 * (1 - stats.t.cdf(abs(dm_star), df=T - 1))
    return float(dm_star), float(p), lag


def main():
    P = np.load(os.path.join(OUT_DIR, "test_preds.npz"))
    y = P["y_true"]                                     # [T, N, H]
    qmed = len(QUANTILES) // 2

    preds = {}
    for nm in P.files:
        if nm == "y_true":
            continue
        a = P[nm]
        preds[nm] = a[..., qmed] if a.ndim == 4 else a  # [T, N, H]

    print("=" * 78)
    print("KIEM DINH DIEBOLD-MARIANO (HLN-corrected), loss = |sai so|")
    print(f"Chuoi thoi gian: {y.shape[0]} cua so | gop tren {y.shape[1]} vung")
    print("=" * 78)

    rows = []
    for hi, h in enumerate(HORIZONS):
        # Sai so trung binh tren cac vung -> chuoi theo thoi gian
        err = {nm: (np.maximum(p[..., hi], 0) - y[..., hi]).mean(axis=1)
               for nm, p in preds.items()}
        # Dung MAE theo cua so (gop |.| truoc khi trung binh) de khop chi so bao cao
        abserr = {nm: np.abs(np.maximum(p[..., hi], 0) - y[..., hi]).mean(axis=1)
                  for nm, p in preds.items()}

        print(f"\n--- Tam du bao t+{h} gio ---")
        base = "ST-MRGAT"
        if base not in abserr:
            continue
        others = [n for n in abserr if n != base]
        for nm in sorted(others, key=lambda n: abserr[n].mean()):
            # d_t = loss(ST-MRGAT) - loss(doi chung); am => ST-MRGAT tot hon
            d = abserr[base] - abserr[nm]
            T = len(d)
            var, lag = newey_west_var(d, None)
            dm = d.mean() / np.sqrt(var / T)
            k = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
            dm_star = dm * k
            p = 2 * (1 - stats.t.cdf(abs(dm_star), df=T - 1))
            better = "ST-MRGAT" if d.mean() < 0 else nm
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  vs {nm:12s} MAE {abserr[base].mean():6.3f} vs "
                  f"{abserr[nm].mean():6.3f} | DM*={dm_star:+7.3f} p={p:.4f} {sig:3s}"
                  f" -> {better}")
            rows.append(dict(horizon=h, baseline=nm, mae_ours=abserr[base].mean(),
                             mae_base=abserr[nm].mean(), dm=dm_star, p=p,
                             nw_lag=lag, better=better))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "dm_test.csv"), index=False)

    print("\n" + "=" * 78)
    print("TONG HOP (so lan ST-MRGAT thang co y nghia p<0.05, tren 3 tam du bao)")
    print("=" * 78)
    for nm in sorted(set(df.baseline)):
        s = df[df.baseline == nm]
        w = int(((s.p < 0.05) & (s.better == "ST-MRGAT")).sum())
        l = int(((s.p < 0.05) & (s.better != "ST-MRGAT")).sum())
        print(f"  vs {nm:12s} thang {w}/3, thua {l}/3, hoa {3 - w - l}/3")
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'dm_test.csv')}")


if __name__ == "__main__":
    main()
