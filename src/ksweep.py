"""
Quet do nhay theo K (so vung phuc vu moi gio) cho bai toan chon vung + lap tuyen.

Ket qua am o K=15 duoc do tai DUNG MOT diem. Cau hoi dung hon la:
   ky nang du bao bat dau co gia tri tu quy mo doi xe nao?
Voi K lon, moi chinh sach deu quet gan het cac diem nong nen ai cung giong ai.
Voi K nho, viec chon dung vung moi thuc su quan trong.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROC_DIR, OUT_DIR, FIG_DIR, SEQ_LEN, HORIZONS, QUANTILES, GRAPH_FILE
from dataset import STData
from operations import solve

K_GRID = [3, 5, 8, 12, 15, 20, 30, 50]


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
    depot = int(np.argmax(d.raw[:d.bounds[1]].sum(0)))

    pol = {}
    pol["Oracle"] = y
    pol["ST-MRGAT"] = P["ST-MRGAT"][:, :, hi, len(QUANTILES) // 2]
    for nm in ("GraphSAGE", "MLP"):
        if nm in P.files:
            pol[nm] = P[nm][:, :, hi]
    pol["HistoricalAvg"] = np.stack(
        [d.ha_raw[d.how[s + SEQ_LEN + HORIZONS[hi] - 1]] for s in ids])
    pol["DistanceOnly"] = np.tile(-D[depot][None, :], (len(y), 1))

    print(f"Kho = {d.zones.iloc[depot]['zone_name']} | "
          f"{len(y)} cua so | K = {K_GRID}")

    rows = []
    for K in K_GRID:
        line = f"  K={K:<3d}"
        for nm, sc in pol.items():
            cov, dist = [], []
            for b in range(len(sc)):
                sel = np.argsort(-sc[b])[:K]
                _, L = solve(sel.tolist(), D, depot)
                cov.append(100 * y[b][sel].sum() / (y[b].sum() + 1e-9))
                dist.append(L)
            rows.append(dict(K=K, policy=nm, coverage=float(np.mean(cov)),
                             distance_km=float(np.mean(dist))))
            line += f" | {nm}={np.mean(cov):5.2f}%"
        print(line, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "ksweep.csv"), index=False)

    # --- Khoang cach toi Oracle: thuoc do "ky nang du bao con lai bao nhieu" ---
    piv = df.pivot(index="K", columns="policy", values="coverage")
    print("\nThieu hut so voi Oracle (diem phan tram):")
    gap = piv.apply(lambda c: piv["Oracle"] - c)
    print(gap.drop(columns="Oracle").to_string(float_format=lambda x: f"{x:6.2f}"))

    plot(piv, gap)


def plot(piv, gap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300, "font.size": 9,
        "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False})

    order = [c for c in ["ST-MRGAT", "GraphSAGE", "MLP", "HistoricalAvg",
                         "DistanceOnly"] if c in gap.columns]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for i, nm in enumerate(order):
        lw = 2.4 if nm == "ST-MRGAT" else 1.5
        ax.plot(gap.index, gap[nm], "-o", color=C[i], lw=lw, ms=4.5,
                markeredgecolor="white", markeredgewidth=1.0,
                label=nm, zorder=5 if nm == "ST-MRGAT" else 3)
    ax.set_xscale("log")
    ax.set_xticks(list(gap.index), [str(k) for k in gap.index])
    ax.set_xlabel("K - zones served per hour (log scale)")
    ax.set_ylabel("Shortfall from oracle (pp)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Forecast skill matters only at small fleet size",
                 color=INK, fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig8_ksweep.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  -> {p}")


if __name__ == "__main__":
    main()
