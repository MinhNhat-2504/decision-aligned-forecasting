"""
DE XUAT B: chon mo hinh theo CHI PHI QUYET DINH thay vi theo HAM MAT MAT.

Lap luan: luan diem trung tam cua bai la mot du bao phai duoc DANH GIA theo ham
quyet dinh ma no phuc vu. Neu chap nhan lap luan do thi viec CHON mo hinh — chon
epoch dung bo — cung khong nen dua tren pinball loss. Day la he qua logic truc tiep
cua chinh bai, va theo hieu biet cua chung toi chua duoc kiem tra trong literature
du bao nhu cau do thi.

Thiet ke:
  Hai nhanh giong nhau hoan toan tru MOT dieu: chi so dung bo tren val_es.
    loss : pinball loss  (nhu moi thi nghiem khac trong bai)
    cost : chi phi newsvendor o phan vi 0.9, tam t+1, trong khong gian so dem
  Moi nhanh chay tren nhieu seed, ca hai backbone quantile, hai bo du lieu.

Doc ket qua the nao:
  - Neu 'cost' giam chi phi test  -> co mot phuong phap moi, don gian, mien phi.
  - Neu khong  -> mot ket qua am nua, van dang bao cao: tieu chi dung bo khong phai
    kenh ma su khong khop ham mat mat/ham quyet dinh di qua.
  Ca hai truong hop deu tra loi mot cau hoi ma bai dang bo ngo.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, QUANTILES, SEEDS, DATASET
from dataset import STData, load_graphs
from train_eval import train_neural, metrics, DEV

TAU = 0.9
HI = 0                      # tam t+1, khop voi quyet dinh trong operations.py
BACKBONES = ["ST-MRGAT", "GraphWaveNet-Q"]


def newsvendor(alloc, y, tau=TAU):
    alloc = np.maximum(alloc, 0.0)
    short = np.maximum(y - alloc, 0.0)
    over = np.maximum(alloc - y, 0.0)
    return dict(
        cost=float((tau * short + (1 - tau) * over).mean()),
        fill_rate=float(1 - short.sum() / (y.sum() + 1e-9)) * 100,
        unmet_pct=float(short.sum() / (y.sum() + 1e-9)) * 100,
        over_pct=float(over.sum() / (y.sum() + 1e-9)) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--models", default=",".join(BACKBONES))
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    backbones = args.models.split(",")

    d = STData()
    graphs = load_graphs(DEV)
    _, _, yte_raw, _ = d.get("test")
    y = yte_raw.numpy()[..., HI]
    qmed = QUANTILES.index(0.5)
    q90 = QUANTILES.index(TAU) if TAU in QUANTILES else len(QUANTILES) - 1

    print("=" * 78)
    print(f"CHON MO HINH: HAM MAT MAT so voi CHI PHI QUYET DINH  [{DATASET}]")
    print(f"  tau={TAU}, tam t+{HI + 1}, {len(seeds)} seed, backbone: {backbones}")
    print("=" * 78)

    rows = []
    for nm in backbones:
        for sel in ("loss", "cost"):
            for sd in seeds:
                m, _ = train_neural(nm, d, graphs, sd, select_by=sel,
                                    select_tau=TAU, select_horizon=HI)
                pq = m.pop("_pred")                      # [B,N,H,Q] da hieu chuan
                r = dict(backbone=nm, select=sel, seed=sd,
                         epochs=m["epochs"], MAE=m["MAE"], PICP=m.get("PICP"))
                r.update({f"q90_{k}": v for k, v in
                          newsvendor(pq[:, :, HI, q90], y).items()})
                r.update({f"med_{k}": v for k, v in
                          newsvendor(pq[:, :, HI, qmed], y).items()})
                rows.append(r)
                print(f"  {nm:16s} select={sel:5s} seed={sd} "
                      f"MAE={r['MAE']:7.3f} q90_cost={r['q90_cost']:7.4f} "
                      f"fill={r['q90_fill_rate']:6.2f}% ep={r['epochs']}",
                      flush=True)

    df = pd.DataFrame(rows)
    # OUT_DIR da tach theo DATASET nen KHONG them hau to nua,
    # neu khong se thanh outputs_green/..._green.csv
    tag = ""
    df.to_csv(os.path.join(OUT_DIR, f"selection_criterion_detail{tag}.csv"),
              index=False)

    # ------------------------------------------------------------------ tong hop
    print("\n" + "=" * 78)
    print("TONG HOP (trung binh qua seed)")
    print("=" * 78)
    g = df.groupby(["backbone", "select"]).agg(
        n=("seed", "size"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
        q90_cost=("q90_cost", "mean"), q90_cost_sd=("q90_cost", "std"),
        fill=("q90_fill_rate", "mean"), PICP=("PICP", "mean"),
        epochs=("epochs", "mean")).reset_index()
    print(g.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------ kiem dinh
    print("\n" + "=" * 78)
    print("KIEM DINH GHEP CAP THEO SEED (cost so voi loss)")
    print("=" * 78)
    out = []
    for nm in backbones:
        a = df[(df.backbone == nm) & (df.select == "cost")].sort_values("seed")
        b = df[(df.backbone == nm) & (df.select == "loss")].sort_values("seed")
        if len(a) != len(b) or len(a) < 2:
            continue
        for metric, better_is_low in (("q90_cost", True), ("MAE", True)):
            t, p = stats.ttest_rel(a[metric].values, b[metric].values)
            delta = a[metric].mean() - b[metric].mean()
            pct = 100 * delta / b[metric].mean()
            verdict = ("cost tot hon" if (delta < 0) == better_is_low
                       else "loss tot hon")
            print(f"  {nm:16s} {metric:9s} cost={a[metric].mean():8.4f} "
                  f"loss={b[metric].mean():8.4f} ({pct:+.2f}%) "
                  f"t={t:+.2f} p={p:.4f}  -> {verdict}"
                  + ("  [co y nghia]" if p < 0.05 else ""))
            out.append(dict(dataset=DATASET, backbone=nm, metric=metric,
                            mean_cost_sel=a[metric].mean(),
                            mean_loss_sel=b[metric].mean(), pct_change=pct,
                            t=t, p=p, n_seed=len(a)))
    pd.DataFrame(out).to_csv(
        os.path.join(OUT_DIR, f"selection_criterion_summary{tag}.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
