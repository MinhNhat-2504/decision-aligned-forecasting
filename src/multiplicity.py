"""
Hieu chinh DA KIEM DINH cho cac ket luan so sanh trong bai.

Vi sao can:
  Bai chay hang chuc kiem dinh (Diebold-Mariano tren 3 tam x 7 doi chung x 2 bo du
  lieu, cong 4 so sanh ablation x 2 bo). Neu bao cao p tho thi mot phan cac ket qua
  "co y nghia" la do so lan thu, khong phai do hieu ung. Reviewer thong ke se hoi.

Hai thu tuc:
  Holm-Bonferroni  -- khong che sai so ho ngo gia (FWER); bao thu, dung cho ket luan
                      khang dinh ("mo hinh A tot hon B").
  Benjamini-Hochberg -- khong che ty le phat hien sai (FDR); thich hop khi coi ca
                      nhom la mot cuoc kham pha.

Ho kiem dinh duoc dinh nghia TRUOC theo tung tuyen bo, khong gop tat ca vao mot ro:
  F1  DM, tung bo du lieu x tung tam du bao   (8 doi chung)  -> "o tam nay, A khac B?"
  F2  DM, tung bo du lieu                     (24 kiem dinh) -> "tren bo nay, A khac B?"
  F3  DM, toan bo                             (48 kiem dinh) -> ket luan chung nhat
  F4  Ablation, tung bo du lieu               (4 so sanh)    -> "thanh phan nao can?"

Bao cao ca ba muc F1-F3 de nguoi doc thay ket luan ben den dau khi mo rong ho.
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def holm(p):
    """Holm-Bonferroni: tra ve p da hieu chinh, giu thu tu dau vao."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def bh(p):
    """Benjamini-Hochberg: tra ve q-value, giu thu tu dau vao."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        running = min(running, m * p[idx] / (i + 1))
        adj[idx] = min(running, 1.0)
    return adj


def load_dm():
    rows = []
    for ds, sub in (("yellow", "outputs"), ("green", "outputs_green")):
        f = os.path.join(BASE, sub, "dm_test.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d["dataset"] = ds
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def load_ablation():
    """t ghep cap cua ablation, tinh lai tu MAE theo tung seed."""
    from scipy import stats
    out = []
    for ds, sub in (("yellow", "outputs"), ("green", "outputs_green")):
        f = os.path.join(BASE, sub, "benchmark_raw_ablation.json")
        if not os.path.exists(f):
            continue
        raw = json.load(io.open(f, encoding="utf-8"))
        if "ST-MRGAT" not in raw:
            continue
        base = [r["MAE"] for r in raw["ST-MRGAT"]]
        for k, v in raw.items():
            if k == "ST-MRGAT" or not k.startswith("  "):
                continue
            other = [r["MAE"] for r in v]
            if len(other) != len(base):
                continue
            t, p = stats.ttest_rel(base, other)
            out.append(dict(dataset=ds, comparison=k.strip(), t=t, p=p,
                            n_seed=len(base)))
    return pd.DataFrame(out)


def report(df, pcol, label, group_cols, tag):
    """In bang so sanh p tho / Holm / BH cho mot dinh nghia ho."""
    df = df.copy()
    if group_cols:
        df[["p_holm", "p_bh"]] = (
            df.groupby(group_cols, group_keys=False)[pcol]
            .apply(lambda g: pd.DataFrame(
                {"p_holm": holm(g.values), "p_bh": bh(g.values)}, index=g.index)))
    else:
        df["p_holm"] = holm(df[pcol].values)
        df["p_bh"] = bh(df[pcol].values)
    n = len(df)
    raw_sig = int((df[pcol] < 0.05).sum())
    holm_sig = int((df.p_holm < 0.05).sum())
    bh_sig = int((df.p_bh < 0.05).sum())
    print(f"\n{label}  (m = {n} kiem dinh"
          + (f", {df.groupby(group_cols).ngroups} ho" if group_cols else "") + ")")
    print(f"  co y nghia p<0.05 : tho {raw_sig}/{n} | Holm {holm_sig}/{n} | "
          f"BH {bh_sig}/{n}")
    lost = df[(df[pcol] < 0.05) & (df.p_holm >= 0.05)]
    if len(lost):
        print(f"  MAT y nghia sau Holm ({len(lost)}):")
        for _, r in lost.iterrows():
            who = r.get("baseline", r.get("comparison", "?"))
            h = f"t+{int(r.horizon)}" if "horizon" in r else ""
            print(f"    {r.dataset:6s} {who:16s} {h:5s} "
                  f"p={r[pcol]:.4f} -> Holm {r.p_holm:.4f}")
    df["family"] = tag
    return df


def main():
    print("=" * 78)
    print("HIEU CHINH DA KIEM DINH")
    print("=" * 78)

    dm = load_dm()
    print(f"\nDM: {len(dm)} kiem dinh ({dm.dataset.nunique()} bo du lieu, "
          f"{dm.horizon.nunique()} tam, {dm.baseline.nunique()} doi chung)")

    f1 = report(dm, "p", "[F1] DM theo (bo du lieu x tam du bao)",
                ["dataset", "horizon"], "F1")
    f2 = report(dm, "p", "[F2] DM theo bo du lieu", ["dataset"], "F2")
    f3 = report(dm, "p", "[F3] DM toan bo", None, "F3")

    ab = load_ablation()
    frames = [f1, f2, f3]
    if len(ab):
        print(f"\nAblation: {len(ab)} so sanh ghep cap")
        f4 = report(ab, "p", "[F4] Ablation theo bo du lieu", ["dataset"], "F4")
        frames.append(f4)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(BASE, "outputs", "multiplicity.csv"), index=False)

    # ---- Ket luan nao con dung sau hieu chinh chat nhat (F3 + Holm) ----
    print("\n" + "=" * 78)
    print("KET LUAN BEN SAU HO CHAT NHAT (F3, Holm, toan bo 48 kiem dinh DM)")
    print("=" * 78)
    surv = f3[f3.p_holm < 0.05].sort_values(["dataset", "horizon"])
    if len(surv) == 0:
        print("  Khong con so sanh DM nao co y nghia.")
    for _, r in surv.iterrows():
        print(f"  {r.dataset:6s} t+{int(r.horizon)} vs {r.baseline:16s} "
              f"p_tho={r.p:.4f} Holm={r.p_holm:.4f} -> thang: {r.better}")
    print(f"\nSaved -> {os.path.join(BASE, 'outputs', 'multiplicity.csv')}")


if __name__ == "__main__":
    main()
