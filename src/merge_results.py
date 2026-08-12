"""
Gop ket qua benchmark tu nhieu lan chay vao mot bang duy nhat.

Can den vi lan chay bo sung (them DCRNN / GraphWaveNet) ghi de len tep ket qua cu
thay vi noi them. Script nay gop lai va TU KIEM TRA de khong tron nham hai lan chay
co cau hinh khac nhau.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR
from train_eval import summarize


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    new_raw = os.path.join(OUT_DIR, "benchmark_raw.json")
    bak_raw = os.path.join(OUT_DIR, "benchmark_raw.BACKUP.json")
    if not os.path.exists(bak_raw):
        print("Khong co ban sao luu -> khong can gop.")
        return

    old, new = load(bak_raw), load(new_raw)

    # Kiem tra tinh nhat quan: cac baseline co mat o ca hai lan chay phai khop nhau.
    # Neu lech nghia la hai lan chay khac cau hinh va KHONG duoc gop.
    common = [k for k in old if k in new]
    print("Kiem tra tinh nhat quan tren baseline chung:")
    ok = True
    for k in common:
        a = old[k][0]["MAE"]
        b = new[k][0]["MAE"]
        same = abs(a - b) < 1e-6
        ok &= same
        print(f"  {k:16s} cu={a:8.4f} moi={b:8.4f} {'khop' if same else 'LECH'}")
    if not ok:
        print("\nDUNG LAI: co baseline lech nhau -> hai lan chay khong cung cau hinh.")
        sys.exit(1)

    merged = dict(old)
    added = []
    for k, v in new.items():
        if k not in merged:
            merged[k] = v
            added.append(k)
    print(f"\nThem moi: {added}")
    print(f"Tong cong: {len(merged)} mo hinh")

    with open(new_raw, "w") as f:
        json.dump(merged, f, indent=2)

    df = summarize({k: v for k, v in merged.items()})
    df.to_csv(os.path.join(OUT_DIR, "benchmark.csv"), index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Gop ca du bao tren test de DM test / operations dung duoc day du mo hinh
    bak_p = os.path.join(OUT_DIR, "test_preds.BACKUP.npz")
    new_p = os.path.join(OUT_DIR, "test_preds.npz")
    if os.path.exists(bak_p) and os.path.exists(new_p):
        out = {}
        with np.load(bak_p) as A, np.load(new_p) as B:
            for k in A.files:
                out[k] = A[k]
            for k in B.files:
                if k not in out:
                    out[k] = B[k]
            assert np.allclose(A["y_true"], B["y_true"]), "y_true lech giua hai lan chay"
        np.savez_compressed(new_p, **out)
        print(f"\nDu bao test da gop: {sorted(out.keys())}")


if __name__ == "__main__":
    main()
