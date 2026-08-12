"""
DE XUAT A: hieu chuan conformal CO DIEU KIEN THEO NHOM (Mondrian CQR).

Van de bai tu neu ra: CQR toan cuc dat dung muc danh nghia 80% tren TRUNG BINH,
nhung do phu theo TUNG VUNG trai rong 54%-100%. Voi nguoi van hanh, mot khoang chi
dung "trung binh" thi khong the dung de cam ket muc dich vu cho tung vung.

Y tuong: thay vi mot luong noi rong Q duy nhat cho toan bo, chia tap hieu chuan
thanh cac NHOM va tinh Q RIENG cho tung nhom. Do la Mondrian conformal prediction.
Bao dam huu han mau van giu trong TUNG nhom, va do phu tro nen deu hon giua cac nhom.

Ba so do duoc so sanh, tat ca dung CHUNG mot mo hinh da huan luyen (khong huan luyen
lai), nen khac biet quan sat duoc thuan tuy den tu cach hieu chuan:

  global   : mot Q duy nhat            (nhu trong bai hien nay)
  zone     : Q rieng theo TAM PHAN nhu cau cua vung, uoc luong TU TRAIN
  pred     : Q rieng theo TAM PHAN cua chinh gia tri du bao (thich nghi theo mau)

Nhom phai duoc xac dinh bang thong tin CO TAI THOI DIEM DU BAO. Ca hai so do tren
deu thoa: mot dung thong ke train, mot dung dau ra cua mo hinh.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, FIG_DIR, QUANTILES, SEEDS, DATASET
from dataset import STData, load_graphs
from train_eval import train_neural, DEV

ALPHA = 1 - (QUANTILES[-1] - QUANTILES[0])      # 0.2 cho khoang 80%
HI = 0                                           # tam t+1


def conformal_q(scores, alpha=ALPHA):
    """Phan vi conformal co hieu chinh huu han mau."""
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return 0.0
    n = len(s)
    k = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, k, method="higher"))


def scores_of(pq, y):
    """Diem bat tuan CQR: max(lo - y, y - hi)."""
    return np.maximum(pq[..., 0] - y, y - pq[..., -1])


def apply_offset(pq, off):
    out = pq.copy()
    out[..., 0] -= off
    out[..., -1] += off
    return out


def coverage_stats(pq, y, zone_axis=1):
    lo = np.maximum(pq[..., 0], 0.0)
    hi = np.maximum(pq[..., -1], 0.0)
    inside = (y >= lo) & (y <= hi)
    per_zone = inside.mean(axis=0) * 100          # [N]
    return dict(
        picp=float(inside.mean()) * 100,
        mpiw=float((hi - lo).mean()),
        zone_min=float(per_zone.min()),
        zone_max=float(per_zone.max()),
        zone_spread=float(per_zone.max() - per_zone.min()),
        zone_p05=float(np.percentile(per_zone, 5)),
        zone_p95=float(np.percentile(per_zone, 95)),
        # Do lech tuyet doi trung binh so voi muc danh nghia: mot con so duy nhat
        # tom tat "do phu co deu giua cac vung khong"
        zone_mad=float(np.abs(per_zone - (1 - ALPHA) * 100).mean()),
        frac_zones_below_70=float((per_zone < 70).mean()) * 100,
    ), per_zone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", default="ST-MRGAT")
    ap.add_argument("--bins", type=int, default=3)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]

    d = STData()
    graphs = load_graphs(DEV)

    # Nhom theo VUNG: tam phan nhu cau trung binh trong ky TRAIN (chi dung train)
    zone_mean = d.raw[:d.bounds[1]].mean(axis=0)
    zq = np.quantile(zone_mean, np.linspace(0, 1, args.bins + 1)[1:-1])
    zone_grp = np.digitize(zone_mean, zq)                       # [N]

    print("=" * 78)
    print(f"MONDRIAN CQR  [{DATASET}]  model={args.model}  "
          f"{len(seeds)} seed  {args.bins} nhom")
    print("=" * 78)
    print(f"  Nhom theo vung (tam phan nhu cau train): "
          f"{[int((zone_grp == g).sum()) for g in range(args.bins)]} vung")

    rows, per_zone_all = [], {}
    for sd in seeds:
        m, _ = train_neural(args.model, d, graphs, sd)
        pq_test = m.pop("_pred_raw")                 # [B,N,H,Q] CHUA hieu chuan
        pq_cal = m.pop("_cal_pred_raw")
        y_cal = m.pop("_cal_true")
        m.pop("_pred", None)
        _, _, yte, _ = d.get("test")
        y_test = yte.numpy()

        s_cal = scores_of(pq_cal, y_cal)             # [Bc,N,H]

        variants = {}

        # ---- 1. global: mot Q duy nhat ----
        variants["global"] = np.full(pq_test.shape[:3],
                                     conformal_q(s_cal[..., HI]))

        # ---- 2. zone: Q theo tam phan nhu cau cua vung ----
        off_zone = np.zeros(d.N)
        for g in range(args.bins):
            sel = zone_grp == g
            off_zone[sel] = conformal_q(s_cal[:, sel, HI])
        variants["zone"] = np.broadcast_to(
            off_zone[None, :, None], pq_test.shape[:3]).copy()

        # ---- 3. pred: Q theo tam phan cua chinh gia tri du bao ----
        qmed = QUANTILES.index(0.5)
        pred_cal = pq_cal[..., HI, qmed]
        pred_te = pq_test[..., HI, qmed]
        pq_bins = np.quantile(pred_cal, np.linspace(0, 1, args.bins + 1)[1:-1])
        g_cal = np.digitize(pred_cal, pq_bins)
        g_te = np.digitize(pred_te, pq_bins)
        off_pred_te = np.zeros_like(pred_te, dtype=float)
        for g in range(args.bins):
            q = conformal_q(s_cal[..., HI][g_cal == g])
            off_pred_te[g_te == g] = q
        v3 = np.zeros(pq_test.shape[:3])
        v3[..., HI] = off_pred_te
        variants["pred"] = v3

        for name, off in variants.items():
            pq_c = pq_test.copy()
            pq_c[..., HI, 0] -= off[..., HI]
            pq_c[..., HI, -1] += off[..., HI]
            st, per_zone = coverage_stats(pq_c[:, :, HI, :], y_test[..., HI])
            st.update(scheme=name, seed=sd)
            rows.append(st)
            per_zone_all.setdefault(name, []).append(per_zone)
            print(f"  seed={sd} {name:6s} PICP={st['picp']:6.2f}%  "
                  f"MPIW={st['mpiw']:6.2f}  vung {st['zone_min']:5.1f}-"
                  f"{st['zone_max']:5.1f}%  MAD={st['zone_mad']:5.2f}  "
                  f"<70%: {st['frac_zones_below_70']:4.1f}% vung", flush=True)

    df = pd.DataFrame(rows)
    # OUT_DIR da tach theo DATASET nen KHONG them hau to nua,
    # neu khong se thanh outputs_green/..._green.csv
    tag = ""
    df.to_csv(os.path.join(OUT_DIR, f"mondrian_cqr_detail{tag}.csv"), index=False)

    print("\n" + "=" * 78)
    print("TONG HOP (trung binh qua seed)")
    print("=" * 78)
    g = df.groupby("scheme").agg(
        PICP=("picp", "mean"), MPIW=("mpiw", "mean"),
        zone_min=("zone_min", "mean"), zone_max=("zone_max", "mean"),
        zone_spread=("zone_spread", "mean"), zone_MAD=("zone_mad", "mean"),
        below70=("frac_zones_below_70", "mean")).reindex(["global", "zone", "pred"])
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))
    g.to_csv(os.path.join(OUT_DIR, f"mondrian_cqr_summary{tag}.csv"))

    base = g.loc["global"]
    print("\nSo voi CQR toan cuc:")
    for k in ("zone", "pred"):
        r = g.loc[k]
        print(f"  {k:6s} do lech do phu theo vung (MAD): "
              f"{base.zone_MAD:.2f} -> {r.zone_MAD:.2f} "
              f"({100*(r.zone_MAD-base.zone_MAD)/base.zone_MAD:+.1f}%)"
              f" | be rong: {base.MPIW:.2f} -> {r.MPIW:.2f} "
              f"({100*(r.MPIW-base.MPIW)/base.MPIW:+.1f}%)")
    print(f"\nSaved -> {OUT_DIR}")
    plot(per_zone_all)


def plot(per_zone_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = ["#2a78d6", "#eb6834", "#1baf7a"]
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300, "font.size": 9,
        "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False})

    names = [n for n in ("global", "zone", "pred") if n in per_zone_all]
    lbl = {"global": "Global CQR", "zone": "Mondrian by zone demand",
           "pred": "Mondrian by predicted level"}
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i, n in enumerate(names):
        v = np.concatenate(per_zone_all[n])
        ax.hist(v, bins=np.arange(0, 102.5, 2.5), histtype="step", lw=2.0,
                color=C[i], label=lbl[n])
    ax.axvline(80, color=INK2, lw=1.4, ls="--")
    ax.annotate("nominal 80%", (80, ax.get_ylim()[1] * 0.94), xytext=(5, 0),
                textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Per-zone empirical coverage (%)")
    ax.set_ylabel("Zones")
    ax.legend(fontsize=8)
    ax.set_title("Group-conditional calibration narrows the coverage spread",
                 color=INK, fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig_mondrian_cqr.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
