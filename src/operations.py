"""
Danh gia GIA TRI VAN HANH cua du bao — phan quan trong nhat de bien mot bai
"du bao chuoi thoi gian" thanh mot bai logistics.

Hai thi nghiem:

A. PHAN BO NANG LUC KIEU NEWSVENDOR (chinh)
   Moi gio, moi vung phai chot truoc nang luc phuc vu a(z) (so shipper/xe).
     thieu  = max(0, y - a)   -> don khong phuc vu duoc, chi phi c_u
     thua   = max(0, a - y)   -> nguon luc nhan roi,     chi phi c_o
   Ly thuyet newsvendor: muc dat hang toi uu la PHAN VI tau = c_u/(c_u+c_o)
   cua phan phoi nhu cau du bao. Day chinh la thu ma dau ra quantile cung cap,
   con du bao diem thi khong. Do la ly do dau quantile la thanh phan chuc nang
   chu khong phai trang tri.

B. DO DAI TUYEN (phu)
   Voi K vung co uu tien cao nhat, giai TSP bang NN + 2-opt tren ma tran khoang
   cach DUONG BO, bao cao ty le nhu cau thu duoc va quang duong.
   Luu y: ty le nhu cau/km KHONG phai chi so tot vi no thuong cho tuyen ngan
   bat ke phuc vu duoc bao nhieu — nen ta bao cao rieng hai chieu.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROC_DIR, OUT_DIR, SEQ_LEN, HORIZONS, QUANTILES, GRAPH_FILE
from dataset import STData

TOP_K = 15
CRITICAL_RATIOS = [0.5, 0.9]     # c_u / (c_u + c_o)


# ---------------------------------------------------------------- TSP
def tour_length(order, D):
    o = np.asarray(order)
    return float(D[o, np.roll(o, -1)].sum())


def nearest_neighbour(nodes, D, start):
    unv = np.array([n for n in nodes if n != start], dtype=np.int64)
    order = [start]
    while len(unv):
        k = int(np.argmin(D[order[-1], unv]))
        order.append(int(unv[k]))
        unv = np.delete(unv, k)
    return order


def two_opt(order, D, max_pass=60):
    """2-opt danh gia DELTA O(1). Ban goc khong co buoc cai thien nao."""
    t = np.asarray(order, dtype=np.int64)
    n = len(t)
    for _ in range(max_pass):
        improved = False
        for i in range(1, n - 1):
            a, b = t[i - 1], t[i]
            j = np.arange(i + 1, n)
            c, dn = t[j], t[(j + 1) % n]
            delta = D[a, c] + D[b, dn] - D[a, b] - D[c, dn]
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                jj = int(j[k])
                t[i:jj + 1] = t[i:jj + 1][::-1]
                improved = True
        if not improved:
            break
    return t.tolist(), tour_length(t, D)


def solve(selected, D, depot):
    return two_opt(nearest_neighbour([depot] + [z for z in selected if z != depot],
                                     D, depot), D)


# ---------------------------------------------------------------- experiments
def newsvendor(alloc, y, tau):
    """Chi phi ky vong khi chot truoc nang luc `alloc` va nhu cau that la `y`."""
    cu, co = tau, 1.0 - tau
    short = np.maximum(y - alloc, 0.0)
    over = np.maximum(alloc - y, 0.0)
    return dict(
        cost=float((cu * short + co * over).mean()),
        unmet_pct=float(short.sum() / (y.sum() + 1e-9)) * 100,
        over_pct=float(over.sum() / (y.sum() + 1e-9)) * 100,
        fill_rate=float(1 - short.sum() / (y.sum() + 1e-9)) * 100,
    )


def main():
    d = STData()
    g = np.load(os.path.join(PROC_DIR, GRAPH_FILE))
    D = g["dist_road"].astype(np.float64)
    bad = D > 1e5
    D[bad] = g["dist_geo"].astype(np.float64)[bad] * 2.0
    np.fill_diagonal(D, 0.0)

    P = np.load(os.path.join(OUT_DIR, "test_preds.npz"))
    hi = 0
    y = P["y_true"][:, :, hi]
    ids = d.idx["test"]

    # --- Cac du bao diem san co ---
    point = {}
    point["Persistence"] = np.stack([d.raw[s + SEQ_LEN - 1] for s in ids])
    how = d.how
    point["HistoricalAvg"] = np.stack(
        [d.ha_raw[how[s + SEQ_LEN + HORIZONS[hi] - 1]] for s in ids])
    quant = {}
    for nm in P.files:
        if nm == "y_true":
            continue
        a = P[nm]
        if a.ndim == 4:
            point[f"{nm} (q50)"] = a[:, :, hi, len(QUANTILES) // 2]
            quant[nm] = a[:, :, hi, :]
        else:
            point[nm] = a[:, :, hi]

    # ================= A. Newsvendor =================
    print("=" * 78)
    print("A. PHAN BO NANG LUC KIEU NEWSVENDOR (tam t+1 gio, tap test)")
    print("=" * 78)
    rows = []
    for tau in CRITICAL_RATIOS:
        print(f"\n  Ty le toi han tau = {tau}  (c_u:c_o = {tau:.0%}:{1-tau:.0%})"
              f"  -> muc toi uu ly thuyet = phan vi {tau:.0%}")
        cands = {k: v for k, v in point.items()}
        # Chinh sach nhan biet quantile: dung dung phan vi tau neu co
        for nm, q in quant.items():
            if abs(tau - 0.5) < 1e-9:
                cands[f"{nm} [quantile tau]"] = q[..., len(QUANTILES) // 2]
            elif abs(tau - QUANTILES[-1]) < 1e-9:
                cands[f"{nm} [quantile tau]"] = q[..., -1]
        cands["Oracle (nhu cau that)"] = y

        for nm, a in sorted(cands.items(),
                            key=lambda kv: newsvendor(kv[1], y, tau)["cost"]):
            r = newsvendor(np.maximum(a, 0), y, tau)
            r.update(policy=nm, tau=tau)
            rows.append(r)
            print(f"    {nm:34s} cost={r['cost']:7.3f}  "
                  f"fill={r['fill_rate']:6.2f}%  thieu={r['unmet_pct']:5.2f}%  "
                  f"thua={r['over_pct']:6.2f}%")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "newsvendor.csv"), index=False)

    # ================= B. Do dai tuyen =================
    print("\n" + "=" * 78)
    print(f"B. LAP TUYEN CHO K={TOP_K} VUNG UU TIEN (NN + 2-opt, khoang cach duong bo)")
    print("=" * 78)
    depot = int(np.argmax(d.raw[:d.bounds[1]].sum(0)))
    print(f"  Kho = {d.zones.iloc[depot]['zone_name']}\n")

    rt = []
    pol = dict(point)
    pol["Oracle (nhu cau that)"] = y
    pol["DistanceOnly"] = np.tile(-D[depot][None, :], (len(y), 1))
    for nm, sc in pol.items():
        cov, dist = [], []
        for b in range(len(sc)):
            sel = np.argsort(-sc[b])[:TOP_K]
            _, L = solve(sel.tolist(), D, depot)
            cov.append(100 * y[b][sel].sum() / (y[b].sum() + 1e-9))
            dist.append(L)
        r = dict(policy=nm, coverage=float(np.mean(cov)),
                 distance_km=float(np.mean(dist)))
        rt.append(r)
        print(f"  {nm:34s} nhu cau thu duoc={r['coverage']:6.2f}%  "
              f"quang duong={r['distance_km']:6.2f} km")

    df = pd.DataFrame(rt).sort_values("coverage", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, "routing_eval.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
