"""
Do NGUONG NHIEU KHONG TAT DINH: chay lai CUNG mot cau hinh, CUNG seed, nhieu lan.

Vi sao can:
  Khi so sanh hai bien the huan luyen, chenh lech quan sat duoc chi co y nghia neu
  no lon hon do dao dong cua chinh viec chay lai. Tren GPU, cac kernel giam
  (reduction) khong tat dinh nen cung mot seed van cho ket qua hoi khac nhau.
  Neu khong bao gio do nguong nay, ta co the bao cao mot "hieu ung" nho hon nhieu
  cua chinh he thong.

Script chay R lan cung (mo hinh, seed, select_by) roi bao cao do lech chuan cua
MAE va chi phi quyet dinh. Con so do la NGUONG PHAN GIAI cua moi so sanh khac
trong bai.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, QUANTILES, DATASET
from dataset import STData, load_graphs
from train_eval import train_neural, DEV

TAU, HI = 0.9, 0


def cost(alloc, y, tau=TAU):
    alloc = np.maximum(alloc, 0.0)
    return float((tau * np.maximum(y - alloc, 0.0)
                  + (1 - tau) * np.maximum(alloc - y, 0.0)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--model", default="ST-MRGAT")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = STData()
    graphs = load_graphs(DEV)
    _, _, yte, _ = d.get("test")
    y = yte.numpy()[..., HI]
    q90 = QUANTILES.index(TAU) if TAU in QUANTILES else len(QUANTILES) - 1

    print("=" * 78)
    print(f"NGUONG NHIEU KHONG TAT DINH  [{DATASET}]")
    print(f"  {args.model}, seed={args.seed} co dinh, chay lai {args.repeats} lan")
    print("=" * 78)

    rows = []
    for r in range(args.repeats):
        m, _ = train_neural(args.model, d, graphs, args.seed, select_by="loss")
        pq = m.pop("_pred")
        c = cost(pq[:, :, HI, q90], y)
        rows.append(dict(run=r, MAE=m["MAE"], q90_cost=c, epochs=m["epochs"]))
        print(f"  lan {r}: MAE={m['MAE']:7.4f}  q90_cost={c:7.4f}  "
              f"ep={m['epochs']}", flush=True)

    df = pd.DataFrame(rows)
    # OUT_DIR da tach theo DATASET nen KHONG them hau to nua,
    # neu khong se thanh outputs_green/..._green.csv
    tag = ""
    df.to_csv(os.path.join(OUT_DIR, f"nondeterminism{tag}.csv"), index=False)

    print("\n" + "=" * 78)
    for col in ("MAE", "q90_cost"):
        v = df[col].values
        rng = v.max() - v.min()
        print(f"  {col:9s} trung binh={v.mean():8.4f}  sd={v.std(ddof=1):.4f}  "
              f"bien do={rng:.4f}  ({100*rng/v.mean():.2f}% cua trung binh)")
    print("\n  => Moi chenh lech nho hon bien do nay KHONG the quy cho can thiep")
    print("     dang so sanh; no nam trong nhieu cua chinh viec chay lai.")
    print(f"\nSaved -> {os.path.join(OUT_DIR, f'nondeterminism{tag}.csv')}")


if __name__ == "__main__":
    main()
