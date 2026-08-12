"""
Phan tich dien giai: mo hinh DUA VAO quan he nao, o dau, va khi nao?

Day la phan ma mot mo hinh mot-quan-he khong the tra loi. `MRGATLayer` hoc trong so
attention muc quan he theo TUNG NODE, nen ta doc truc tiep xem vung nao phu thuoc
lang gieng dia ly, vung nao phu thuoc lang gieng theo mang duong bo, vung nao phu
thuoc luong OD.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, FIG_DIR, HORIZONS, QUANTILES, BATCH_SIZE
from dataset import STData, load_graphs
from train_eval import train_neural, DEV

RELS = ("geo", "road", "flow")
REL_VI = {"geo": "Geographic", "road": "Road", "flow": "OD-flow"}


def relation_weights(model, d, graphs, split="test"):
    """Trich trong so attention muc quan he: [B, N, R] o lop thu nhat."""
    static = torch.from_numpy(d.static).to(DEV)
    edge_list = [graphs[r] for r in RELS]
    x, _, _, base = d.get(split)
    model.eval()
    ws = []
    with torch.no_grad():
        for i in range(0, len(x), BATCH_SIZE):
            xb = x[i:i + BATCH_SIZE].to(DEV)
            h = model.tcn(xb)
            if model.static is not None:
                h = h + model.static(static).unsqueeze(0)
            _, (_, w) = model.g1(h, edge_list, return_att=True)
            ws.append(w.squeeze(-1).cpu())              # [B, N, R]
    return torch.cat(ws).numpy()


def main():
    d = STData()
    graphs = load_graphs(DEV)
    print("Huan luyen ST-MRGAT (seed 0) de phan tich ...")
    m, model = train_neural("ST-MRGAT", d, graphs, seed=0)
    m.pop("_pred", None)
    print(f"  MAE={m['MAE']:.3f}  R2={m['R2']:.4f}")

    W = relation_weights(model, d, graphs)              # [B, N, 3]
    print(f"  trong so: {W.shape}")

    # --- Trung binh toan cuc ---
    g = W.mean((0, 1))
    print("\nTrong so quan he trung binh toan cuc:")
    for r, v in zip(RELS, g):
        print(f"  {REL_VI[r]:10s} {v:.4f}")

    # --- Theo borough ---
    zn = d.zones.copy()
    per_node = W.mean(0)                                # [N, 3]
    for i, r in enumerate(RELS):
        zn[r] = per_node[:, i]
    by_bor = zn.groupby("borough")[list(RELS)].mean()
    by_bor["n_zones"] = zn.groupby("borough").size()
    print("\nTheo borough:")
    print(by_bor.to_string(float_format=lambda x: f"{x:.4f}"))
    by_bor.to_csv(os.path.join(OUT_DIR, "relation_weights_borough.csv"))

    # --- Theo gio trong ngay ---
    tt = d.target_times("test")[:, 0]
    hod = d.ts[tt].hour.values
    rows = []
    for h in range(24):
        sel = hod == h
        if sel.sum():
            rows.append(dict(hour=h, **{r: float(W[sel].mean((0, 1))[i])
                                        for i, r in enumerate(RELS)}))
    by_hour = pd.DataFrame(rows)
    by_hour.to_csv(os.path.join(OUT_DIR, "relation_weights_hour.csv"), index=False)

    # --- Vung phu thuoc OD-flow nhieu nhat (khong the giai thich bang dia ly) ---
    top = zn.nlargest(10, "flow")[["zone_name", "borough"] + list(RELS)]
    print("\n10 vung dua vao quan he OD-flow nhieu nhat:")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Hinh ---
    plot(by_bor, by_hour)


def plot(by_bor, by_hour):
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

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.2))

    b = by_bor.sort_values("flow")
    y = np.arange(len(b))
    left = np.zeros(len(b))
    for i, r in enumerate(RELS):
        a1.barh(y, b[r], left=left, height=0.62, color=C[i], label=REL_VI[r],
                edgecolor="white", linewidth=1.6)
        for yi, (v, l) in enumerate(zip(b[r], left)):
            if v > 0.12:
                a1.text(l + v / 2, yi, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        left = left + b[r].values
    a1.set_yticks(y, b.index, fontsize=8.5)
    a1.set_xlabel("Relation-level attention weight")
    a1.set_xlim(0, 1)
    a1.grid(axis="y", visible=False)
    a1.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    a1.set_title("By borough", color=INK, fontsize=9.5, loc="left", pad=24)

    for i, r in enumerate(RELS):
        a2.plot(by_hour.hour, by_hour[r], "-", color=C[i], lw=2.0)
        a2.annotate(REL_VI[r], (23, by_hour[r].iloc[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=C[i])
    a2.set_xlabel("Hour of day")
    a2.set_ylabel("Weight")
    a2.set_xlim(0, 28)
    a2.set_xticks(range(0, 24, 4))
    a2.set_title("By hour of day", color=INK, fontsize=9.5, loc="left", pad=24)

    fig.suptitle("Which relation does the model rely on?", x=0.008, ha="left",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig7_relation_attention.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  -> {p}")


if __name__ == "__main__":
    main()
