"""
Sinh hinh cho bai bao. Bang mau da qua kiem dinh CVD (khong dung mau tu bia dat).

Nguyen tac ap dung:
  - Mau danh muc gan theo THU TU CO DINH, khong xoay vong.
  - Moi hinh mot truc y duy nhat (khong bao gio dung 2 thang do).
  - Nhan truc tiep tren cot thay vi bat nguoi doc doi chieu chu giai.
  - Luoi/truc lui ve sau; net du lieu manh.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, FIG_DIR, HORIZONS, QUANTILES

# Bang mau danh muc da validate (light mode, surface #fcfcfb)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "font.size": 9,
    "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, name):
    fig.tight_layout()
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {p}")


# ------------------------------------------------------------------ fig 1
def fig_horizon(raw):
    """MAE theo tam du bao. Cong viec cua du lieu: so sanh do lon giua cac mo hinh."""
    order = ["Persistence", "Ridge", "XGBoost", "HistoricalAvg", "MLP", "LSTM",
             "GCN", "GraphSAGE", "GAT", "ST-MRGAT"]
    order = [m for m in order if m in raw and m != "Persistence"]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))

    ends = {}
    for i, m in enumerate(order):
        v = [np.mean([r[f"h{h}"]["MAE"] for r in raw[m]]) for h in HORIZONS]
        lw, z = (2.4, 5) if m == "ST-MRGAT" else (1.5, 3)
        ax.plot(HORIZONS, v, "-o", color=C[i % len(C)], lw=lw, ms=4.5, zorder=z,
                markeredgecolor="white", markeredgewidth=1.0)
        ends[m] = (v[-1], C[i % len(C)])

    # Tach nhan o mep phai de khong de len nhau
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.052
    items = sorted(ends.items(), key=lambda kv: kv[1][0])
    ys = [kv[1][0] for kv in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    for (m, (yv, col)), yl in zip(items, ys):
        ax.annotate(m, (HORIZONS[-1], yv), xytext=(HORIZONS[-1] + 0.35, yl),
                    textcoords="data", va="center", fontsize=7.8,
                    color=INK if m == "ST-MRGAT" else INK2,
                    fontweight="bold" if m == "ST-MRGAT" else "normal",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8,
                                    shrinkA=2, shrinkB=1, alpha=0.55))

    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE (events per hour per zone)")
    ax.set_xticks(HORIZONS)
    ax.set_xlim(0.6, HORIZONS[-1] + 3.4)
    ax.set_title("Error by forecast horizon, test block 16-31 Mar 2016",
                 color=INK, fontsize=10, loc="left", pad=10)
    _save(fig, "fig1_horizon_mae.png")


# ------------------------------------------------------------------ fig 2
def fig_benchmark(df):
    """Xep hang mo hinh theo MAE, kem thanh sai so giua cac seed."""
    d = df[df.model != "Persistence"].sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    y = np.arange(len(d))
    cols = [C[0] if m == "ST-MRGAT" else MUTED for m in d.model]
    ax.barh(y, d.MAE, xerr=d.MAE_std, height=0.62, color=cols,
            error_kw=dict(ecolor=INK2, lw=1.0, capsize=2.5))
    for yi, (v, s) in enumerate(zip(d.MAE, d.MAE_std)):
        ax.text(v + s + 0.12, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, d.model, fontsize=8.5)
    ax.set_xlabel("Test MAE (lower is better)")
    ax.set_xlim(0, d.MAE.max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_title("Model comparison (mean of 5 seeds +/- SD)",
                 color=INK, fontsize=10, loc="left", pad=10)
    _save(fig, "fig2_benchmark.png")


# ------------------------------------------------------------------ fig 3
def fig_ablation(path):
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    lbl = {"ST-MRGAT": "Full (geo+road+flow, quantile)",
           "  w/o road": "- road relation", "  w/o flow": "- OD-flow relation",
           "  geo only": "geo relation only", "  point loss": "- quantile objective"}
    # Chi giu cac cau hinh ablation: cac baseline (nhat la Persistence ~29)
    # se nen het thang do va lam bien mat chenh lech 7.36-7.61 can xem.
    df = df[df.model.isin(lbl)].copy()
    df["lab"] = df.model.map(lbl)
    df = df.sort_values("MAE", ascending=False)
    base = df[df.model == "ST-MRGAT"].MAE.iloc[0]
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    y = np.arange(len(df))
    cols = [C[0] if m == "ST-MRGAT" else MUTED for m in df.model]
    ax.barh(y, df.MAE, xerr=df.MAE_std, height=0.6, color=cols,
            error_kw=dict(ecolor=INK2, lw=1.0, capsize=2.5))
    for yi, (v, sd, m) in enumerate(zip(df.MAE, df.MAE_std, df.model)):
        d = "" if m == "ST-MRGAT" else f"  (+{100 * (v - base) / base:.1f}%)"
        ax.text(v + sd + 0.006, yi, f"{v:.3f}{d}", va="center", fontsize=8,
                color=INK)
    ax.set_yticks(y, df.lab, fontsize=8.5)
    ax.set_xlabel("Test MAE")
    ax.set_xlim(df.MAE.min() * 0.985, df.MAE.max() * 1.022)
    ax.grid(axis="y", visible=False)
    ax.set_title("Component ablation", color=INK, fontsize=10,
                 loc="left", pad=10)
    _save(fig, "fig3_ablation.png")


# ------------------------------------------------------------------ fig 4
def fig_coverage(raw):
    """Do phu cua moi backbone quantile truoc va sau hieu chuan conformal."""
    if "ST-MRGAT" not in raw or "PICP_raw" not in raw["ST-MRGAT"][0]:
        return
    models = [m for m in ("ST-MRGAT", "GraphWaveNet-Q")
              if m in raw and "PICP_raw" in raw[m][0]]
    nominal = (QUANTILES[-1] - QUANTILES[0]) * 100
    raw_vals = [np.mean([x["PICP_raw"] for x in raw[m]]) for m in models]
    cal_vals = [np.mean([x["PICP"] for x in raw[m]]) for m in models]
    raw_errs = [np.std([x["PICP_raw"] for x in raw[m]]) for m in models]
    cal_errs = [np.std([x["PICP"] for x in raw[m]]) for m in models]
    raw_wid = [np.mean([x["MPIW_raw"] for x in raw[m]]) for m in models]
    cal_wid = [np.mean([x["MPIW"] for x in raw[m]]) for m in models]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(models))
    w = 0.34
    ax.bar(x - w / 2, raw_vals, yerr=raw_errs, width=w, color=MUTED,
           label="Raw quantiles",
           error_kw=dict(ecolor=INK2, lw=1.0, capsize=3))
    ax.bar(x + w / 2, cal_vals, yerr=cal_errs, width=w, color=C[0],
           label="After CQR",
           error_kw=dict(ecolor=INK2, lw=1.0, capsize=3))
    ax.axhline(nominal, color=C[1], lw=1.6, ls="--", zorder=4)
    ax.annotate(f"nominal {nominal:.0f}%", (len(models) - 0.55, nominal),
                xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=8, color=C[1])
    for xi, (rv, cv, rw, cw) in enumerate(zip(raw_vals, cal_vals,
                                               raw_wid, cal_wid)):
        ax.text(xi - w / 2, rv + 2.0, f"{rv:.1f}%", ha="center",
                fontsize=8, color=INK)
        ax.text(xi + w / 2, cv + 2.0, f"{cv:.1f}%", ha="center",
                fontsize=8, color=INK, fontweight="bold")
        ax.text(xi - w / 2, 3, f"W={rw:.1f}", ha="center", fontsize=7,
                color="white")
        ax.text(xi + w / 2, 3, f"W={cw:.1f}", ha="center", fontsize=7,
                color="white")
    ax.set_xticks(x, models, fontsize=8.5)
    ax.set_ylabel("Empirical coverage PICP (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_title("CQR transfers across forecasting backbones",
                 color=INK, fontsize=10,
                 loc="left", pad=10)
    _save(fig, "fig4_coverage.png")


# ------------------------------------------------------------------ fig 5
def fig_newsvendor(path):
    """Chi phi newsvendor tai tau=.9: point vs quantile tren hai backbone."""
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df = df[(df.tau == 0.9) & ~df.policy.str.contains("Oracle")]
    pols = ["HistoricalAvg", "GraphWaveNet", "GraphWaveNet-Q (q50)",
            "GraphWaveNet-Q [quantile tau]", "ST-MRGAT (q50)",
            "ST-MRGAT [quantile tau]"]
    pols = [p for p in pols if p in set(df.policy)]
    lab = {"HistoricalAvg": "Historical\naverage",
           "GraphWaveNet": "Graph WaveNet\npoint",
           "GraphWaveNet-Q (q50)": "Graph WaveNet-Q\nmedian",
           "GraphWaveNet-Q [quantile tau]": "Graph WaveNet-Q\nq=.9",
           "ST-MRGAT (q50)": "ST-MRGAT\nmedian",
           "ST-MRGAT [quantile tau]": "ST-MRGAT\nq=.9"}

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(pols))
    v = [df[df.policy == p].cost.iloc[0] for p in pols]
    cols = [C[0] if "quantile tau" in p else MUTED for p in pols]
    ax.bar(x, v, width=0.66, color=cols, edgecolor="white", linewidth=1.2)
    for xi, vv in zip(x, v):
        ax.text(xi, vv + max(v) * 0.018, f"{vv:.2f}", ha="center",
                fontsize=8, color=INK)
    ax.set_xticks(x, [lab.get(p, p) for p in pols], fontsize=7.5)
    ax.set_ylabel("Newsvendor cost (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Decision-aligned quantiles transfer across backbones (τ=.9)",
                 color=INK, fontsize=10, loc="left", pad=10)
    _save(fig, "fig5_newsvendor.png")


# ------------------------------------------------------------------ fig 6
def fig_timeseries():
    """Chuoi that vs du bao kem khoang, cho vung co nhu cau lon nhat."""
    from dataset import STData
    p = os.path.join(OUT_DIR, "test_preds.npz")
    if not os.path.exists(p):
        return
    P = np.load(p)
    if "ST-MRGAT" not in P.files:
        return
    d = STData()
    yt = P["y_true"][:, :, 0]
    pq = P["ST-MRGAT"][:, :, 0, :]
    n = int(np.argmax(yt.sum(0)))
    k = slice(0, 168)
    t = np.arange(len(yt))[k]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.fill_between(t, np.maximum(pq[k, n, 0], 0), pq[k, n, -1],
                    color=C[0], alpha=0.18, lw=0)
    ax.plot(t, yt[k, n], color=INK2, lw=1.5, label="Actual")
    ax.plot(t, pq[k, n, 1], color=C[0], lw=1.8, label="Forecast (median)")
    ax.set_xlabel("Hours from 16 Mar 2016 (first test week)")
    ax.set_ylabel("Dropoffs per hour")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title(f"Zone {d.zones.iloc[n]['zone_name']} - one-hour-ahead forecast "
                 f"with calibrated 80% interval",
                 color=INK, fontsize=10, loc="left", pad=10)
    _save(fig, "fig6_timeseries.png")


def main():
    print("Sinh hinh:")
    raw_p = os.path.join(OUT_DIR, "benchmark_raw.json")
    if os.path.exists(raw_p):
        raw = json.load(open(raw_p))
        df = pd.read_csv(os.path.join(OUT_DIR, "benchmark.csv"))
        fig_horizon(raw)
        fig_benchmark(df)
        fig_coverage(raw)
        fig_timeseries()
    fig_ablation(os.path.join(OUT_DIR, "benchmark_ablation.csv"))
    fig_newsvendor(os.path.join(OUT_DIR, "newsvendor.csv"))


if __name__ == "__main__":
    main()
