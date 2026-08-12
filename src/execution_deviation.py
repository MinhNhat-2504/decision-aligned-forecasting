"""
Do NHIEU THUC THI cua tai xe that, va doi chieu voi khac biet giua cac chinh sach
chon vung trong phan dinh tuyen cua bai bao.

Vi sao can:
  Toan bo danh gia dinh tuyen o day GIA DINH tuyen duoc lap se duoc di dung nhu vay.
  Neu tai xe that khong di dung thu tu do, moi khac biet nho giua cac chinh sach
  chon vung se bi nhieu thuc thi nuot mat. Do la mot moi de doa THAT doi voi ket
  luan am cua bai bao, va truoc day chua he duoc kiem tra.

Nguon: Konovalenko, Hvattum & Iversen, Mendeley Data 10.17632/kkwgfvmtxn.1 (CC BY 4.0)
  19.647 tuyen, 400 tai xe, 249.231 diem dung, 32 tuan, 2 quoc gia.
  Truong: IndexP/IndexA = thu tu KE HOACH / THUC TE; DistanceP/DistanceA = quang
  duong tu diem truoc theo tung thu tu.

Luu y pham vi: bo nay KHONG co thoi diem tao don, KHONG co suc chua xe, va KHONG co
vet GPS lien tuc (khoang cach tinh bang Haversine giua cac diem dung). No chi dung
de do do lech KE HOACH-THUC THI, khong dung de dong khoang trong ba truc o muc 7.9.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, FIG_DIR

XLSX = os.path.join(
    os.path.dirname(OUT_DIR), "data", "raw3",
    "Last-mile delivery route deviations dataset planned vs. actual routes",
    "routes_performance.xlsx")

RNG = np.random.default_rng(0)


def bootstrap_ci(x, stat=np.mean, n=2000, alpha=0.05):
    """Khoang tin cay bootstrap theo TUYEN (don vi doc lap), khong theo diem dung."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    boot = stat(x[idx], axis=1)
    return float(stat(x)), float(np.quantile(boot, alpha / 2)), float(
        np.quantile(boot, 1 - alpha / 2))


def main():
    d = pd.read_excel(XLSX)
    print("=" * 78)
    print("NHIEU THUC THI: KE HOACH so voi THUC TE (du lieu tai xe that)")
    print("=" * 78)
    print(f"  {d['Route ID'].nunique():,} tuyen | {d['Driver ID'].nunique()} tai xe | "
          f"{len(d):,} diem dung | {d['Week ID'].nunique()} tuan | "
          f"{d.Country.nunique()} quoc gia")

    rows = []
    for rid, g in d.groupby("Route ID", sort=False):
        if len(g) < 3:
            continue
        p = g["IndexP"].to_numpy()
        a = g["IndexA"].to_numpy()
        # Ty le diem dung KHONG duoc phuc vu dung vi tri da lap
        off = float((p != a).mean())
        tau = stats.kendalltau(p, a).statistic
        dp, da = float(g["DistanceP"].sum()), float(g["DistanceA"].sum())
        rows.append(dict(route=rid, n_stop=len(g), off_position=off, kendall_tau=tau,
                         dist_plan=dp, dist_actual=da,
                         dist_ratio=(da / dp) if dp > 0 else np.nan))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT_DIR, "execution_deviation_detail.csv"), index=False)

    # ---------------- 1. Tai xe co di dung thu tu da lap khong? ----------------
    print("\n[1] DO LECH THU TU")
    m, lo, hi = bootstrap_ci(r.off_position)
    print(f"  Ty le diem dung bi phuc vu SAI vi tri da lap: "
          f"{100*m:.1f}% [{100*lo:.1f}, {100*hi:.1f}]")
    m2, lo2, hi2 = bootstrap_ci(r.kendall_tau)
    print(f"  Kendall tau (ke hoach vs thuc te)           : "
          f"{m2:.3f} [{lo2:.3f}, {hi2:.3f}]")
    exact = float((r.off_position == 0).mean())
    print(f"  Ty le tuyen di DUNG HOAN TOAN ke hoach      : {100*exact:.1f}%")

    # ---------------- 2. Nhieu quang duong ----------------
    print("\n[2] QUANG DUONG: THUC TE so voi KE HOACH")
    m3, lo3, hi3 = bootstrap_ci(r.dist_ratio)
    print(f"  Ty so quang duong thuc/ke hoach : {m3:.4f} [{lo3:.4f}, {hi3:.4f}]")
    print(f"  -> tuyen thuc te {'NGAN hon' if m3 < 1 else 'DAI hon'} ke hoach "
          f"{abs(1-m3)*100:.2f}% tinh trung binh")
    # Do lon cua nhieu, khong phai dau cua no: |1 - ty so|
    absdev = (r.dist_ratio - 1.0).abs()
    m4, lo4, hi4 = bootstrap_ci(absdev)
    print(f"  DO LON lech tuyet doi |1 - ty so|: {100*m4:.2f}% "
          f"[{100*lo4:.2f}, {100*hi4:.2f}]")
    for q in (50, 75, 90):
        print(f"    phan vi {q}%: {100*np.nanpercentile(absdev, q):.2f}%")

    # ---------------- 3. Doi chieu voi khac biet giua cac chinh sach ----------------
    print("\n[3] DOI CHIEU VOI PHAN DINH TUYEN CUA BAI BAO")
    rt_path = os.path.join(OUT_DIR, "routing_eval.csv")
    if os.path.exists(rt_path):
        rt = pd.read_csv(rt_path)
        learned = rt[~rt.policy.str.contains("Oracle|DistanceOnly", regex=True)]
        dspread = learned.distance_km.max() - learned.distance_km.min()
        dbase = learned.distance_km.mean()
        cspread = learned.coverage.max() - learned.coverage.min()
        print(f"  Quang duong giua cac chinh sach : "
              f"{learned.distance_km.min():.2f}-{learned.distance_km.max():.2f} km "
              f"(chenh {100*dspread/dbase:.2f}%)")
        print(f"  Nhu cau thu duoc giua cac chinh sach: chenh {cspread:.2f} diem %")
        ratio = (100 * m4) / (100 * dspread / dbase)
        print(f"\n  => Nhieu thuc thi ({100*m4:.2f}%) lon gap "
              f"{ratio:.1f} lan khac biet quang duong giua cac chinh sach "
              f"({100*dspread/dbase:.2f}%).")
        print("     Khac biet do do nam DUOI nguong nhieu ma bat ky trien khai that")
        print("     nao cung phai chiu -> ket luan am ve dinh tuyen khong phai do")
        print("     bo giai cua ta yeu, ma do khac biet that su khong quan sat duoc.")
        summary_extra = dict(policy_distance_spread_pct=100 * dspread / dbase,
                             policy_coverage_spread_pp=cspread,
                             noise_to_signal_ratio=ratio)
    else:
        summary_extra = {}

    # ---------------- 4. Do lech co tang theo do dai tuyen khong? ----------------
    print("\n[4] DO LECH THEO QUY MO TUYEN")
    r["bucket"] = pd.cut(r.n_stop, [0, 5, 10, 15, 20, 100],
                         labels=["<=5", "6-10", "11-15", "16-20", ">20"])
    by = r.groupby("bucket", observed=True).agg(
        tuyen=("route", "size"), lech_vi_tri=("off_position", "mean"),
        kendall=("kendall_tau", "mean"), lech_quang_duong=("dist_ratio", "mean"))
    by["lech_vi_tri"] *= 100
    print(by.to_string(float_format=lambda x: f"{x:.3f}"))

    pd.DataFrame([dict(
        n_routes=len(r), n_drivers=int(d["Driver ID"].nunique()),
        off_position_mean=m, off_position_lo=lo, off_position_hi=hi,
        kendall_tau_mean=m2, exact_plan_followed=exact,
        dist_ratio_mean=m3, dist_ratio_lo=lo3, dist_ratio_hi=hi3,
        abs_dist_dev_mean=m4, abs_dist_dev_lo=lo4, abs_dist_dev_hi=hi4,
        **summary_extra)]).to_csv(
        os.path.join(OUT_DIR, "execution_deviation_summary.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}")
    plot(r)


def plot(r):
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

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.1))

    a1.hist(100 * r.off_position, bins=40, color=C[0], edgecolor="white", lw=0.4)
    a1.axvline(100 * r.off_position.mean(), color=C[1], lw=1.8, ls="--")
    a1.annotate(f"mean {100*r.off_position.mean():.1f}%",
                (100 * r.off_position.mean(), a1.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color=C[1])
    a1.set_xlabel("% of stops served out of planned position")
    a1.set_ylabel("Routes")
    a1.set_title("How far drivers depart from the plan",
                 color=INK, fontsize=9.5, loc="left", pad=8)

    dev = (100 * (r.dist_ratio - 1).abs()).clip(upper=60)
    a2.hist(dev, bins=40, color=C[2], edgecolor="white", lw=0.4)
    a2.axvline(dev.mean(), color=C[1], lw=1.8, ls="--")
    a2.annotate(f"mean {dev.mean():.1f}%",
                (dev.mean(), a2.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", fontsize=8, color=C[1])
    a2.set_xlabel("|deviation of realised vs planned route length| (%, clipped at 60)")
    a2.set_ylabel("Routes")
    a2.set_title("Magnitude of execution noise", color=INK, fontsize=9.5,
                 loc="left", pad=8)

    fig.suptitle("Execution noise across 19,647 real delivery routes",
                 x=0.008, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig_execution_deviation.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
