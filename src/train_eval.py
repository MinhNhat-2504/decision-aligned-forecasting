"""
Khung thuc nghiem chuan: huan luyen + danh gia tren TAP TEST tach theo thoi gian.

Nhung diem ban goc thieu ma o day bat buoc phai co:
  - Chia train/val/test; chi so bao cao la tren TEST chua tung duoc nhin thay.
  - Dung bo (early stopping tren VAL), khong lay epoch cuoi.
  - Nhieu seed + do lech chuan + kiem dinh t ghep cap.
  - Day du baseline: Persistence, Historical Average, Ridge, XGBoost, MLP, LSTM,
    GCN, GraphSAGE, GAT, va mo hinh de xuat.

MOI mo hinh hoc (ke ca baseline cay/tuyen tinh) deu duoc cap cung mot nen mua vu
HA(zone, gio-trong-tuan) uoc luong tu train, va du bao PHAN DU tren nen do
-> so sanh cong bang, khong mo hinh nao duoc loi the thong tin.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (OUT_DIR, SEQ_LEN, HORIZONS, QUANTILES, HIDDEN_DIM, N_HEADS,
                    DROPOUT, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, BATCH_SIZE,
                    SEEDS)
from dataset import STData, load_graphs
from models import (GraphForecaster, STMRGAT, MLPForecaster, LSTMForecaster,
                    DCRNN, GraphWaveNet, pinball_loss)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
QMED = QUANTILES.index(0.5)


# ------------------------------------------------------------------ metrics
def metrics(pred, true):
    """pred/true [B, N, H] o KHONG GIAN SO DEM GOC."""
    pred = np.maximum(pred, 0.0)
    out = {}
    for hi, h in enumerate(HORIZONS):
        p, t = pred[..., hi].ravel(), true[..., hi].ravel()
        e = p - t
        ss_res = float((e ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        out[f"h{h}"] = dict(
            MAE=float(np.abs(e).mean()),
            RMSE=float(np.sqrt((e ** 2).mean())),
            WMAPE=float(np.abs(e).sum() / (np.abs(t).sum() + 1e-9)) * 100,
            R2=1 - ss_res / (ss_tot + 1e-9))
    for k in ("MAE", "RMSE", "WMAPE", "R2"):
        out[k] = float(np.mean([out[f"h{h}"][k] for h in HORIZONS]))
    return out


def interval_metrics(pq, true, tag=""):
    lo, hi = np.maximum(pq[..., 0], 0), np.maximum(pq[..., -1], 0)
    return {f"PICP{tag}": float(((true >= lo) & (true <= hi)).mean()) * 100,
            f"MPIW{tag}": float((hi - lo).mean())}


def conformal_offset(pq_val, y_val, alpha):
    """
    Hieu chuan conformal cho hoi quy phan vi (CQR, Romano et al. 2019).

    Hoi quy phan vi thuong THIEU PHU trong thuc te. Ta tinh diem bat tuan tren tap
    hieu chuan (val) roi noi rong khoang mot luong Q duy nhat, nho do khoang dat
    do phu >= 1-alpha voi bao dam huu han mau, khong can gia dinh phan phoi.
    """
    lo, hi = pq_val[..., 0], pq_val[..., -1]
    score = np.maximum(lo - y_val, y_val - hi).ravel()
    n = len(score)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(score, q, method="higher"))


# ------------------------------------------------------------------ baselines
def run_persistence(d):
    _, _, yr, _ = d.get("test")
    last = np.stack([d.raw[s + SEQ_LEN - 1] for s in d.idx["test"]])
    return metrics(np.repeat(last[..., None], len(HORIZONS), 2), yr.numpy())


def run_historical_average(d):
    _, _, yr, base = d.get("test")
    return metrics(d.inverse(base.numpy()), yr.numpy())


def _tabular(d, split):
    """Dac trung phang cho mo hinh co dien: lich su + lich + zone + nen mua vu."""
    x, _, yr, base = d.get(split)
    B, N, L, C = x.shape
    feats = np.concatenate([
        x[..., 0].reshape(B * N, L).numpy(),            # z(t-L..t-1)
        x[..., 1].reshape(B * N, L).numpy(),            # z(t-24-...)
        x[..., 2].reshape(B * N, L).numpy(),            # z(t-168-...)
        x[:, :, -1, 3:].reshape(B * N, C - 3).numpy(),  # dac trung lich
        np.tile(d.static, (B, 1)),                       # dac trung zone
    ], axis=1)
    return feats, yr.numpy().reshape(B * N, -1), base.numpy().reshape(B * N, -1)


def _fit_tabular(d, make_model, use_val=False):
    Xtr, ytr, btr = _tabular(d, "train")
    Xva, yva, bva = _tabular(d, "val")
    Xte, yte, bte = _tabular(d, "test")
    preds = []
    for hi in range(len(HORIZONS)):
        # Du bao PHAN DU tren nen mua vu, trong khong gian log chuan hoa
        ztr = (np.log1p(ytr[:, hi]) - d.mu) / d.sd - btr[:, hi]
        zva = (np.log1p(yva[:, hi]) - d.mu) / d.sd - bva[:, hi]
        m = make_model()
        if use_val:
            m.fit(np.c_[Xtr, btr[:, hi]], ztr,
                  eval_set=[(np.c_[Xva, bva[:, hi]], zva)], verbose=False)
        else:
            m.fit(np.c_[Xtr, btr[:, hi]], ztr)
        preds.append(d.inverse(m.predict(np.c_[Xte, bte[:, hi]]) + bte[:, hi]))
    B = len(d.idx["test"])
    return metrics(np.stack(preds, 1).reshape(B, d.N, -1), yte.reshape(B, d.N, -1))


def run_ridge(d):
    from sklearn.linear_model import Ridge
    return _fit_tabular(d, lambda: Ridge(alpha=1.0))


def run_xgboost(d):
    import xgboost as xgb
    return _fit_tabular(d, lambda: xgb.XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, early_stopping_rounds=40, tree_method="hist",
        device=DEV, verbosity=0), use_val=True)


# ------------------------------------------------------------------ neural
GRAPH_MODELS = {"GCN", "GraphSAGE", "GAT", "DCRNN", "GraphWaveNet",
                "GraphWaveNet-Q"}
# Mo hinh dung DO THI DA QUAN HE (nhan edge_list thay vi 1 edge_index)
MULTIREL = {"ST-MRGAT"}
# Mo hinh xuat PHAN PHOI (dau quantile + pinball loss + hieu chuan conformal).
# Tach khoi MULTIREL de tang quyet dinh gan duoc len backbone bat ky.
QUANTILE_CAPABLE = {"ST-MRGAT", "GraphWaveNet-Q"}


def build_model(name, d, n_rel=3, n_q=None, hidden=HIDDEN_DIM):
    nh = len(HORIZONS)
    nq = n_q if n_q is not None else len(QUANTILES)
    if name == "MLP":
        return MLPForecaster(d.C, SEQ_LEN, d.S, hidden, nh, DROPOUT)
    if name == "LSTM":
        return LSTMForecaster(d.C, d.S, hidden, nh, DROPOUT)
    if name == "DCRNN":
        return DCRNN(d.C, d.S, hidden, nh, dropout=DROPOUT)
    if name == "GraphWaveNet":
        return GraphWaveNet(d.C, d.S, hidden, nh, N=d.N, dropout=DROPOUT)
    if name == "GraphWaveNet-Q":
        return GraphWaveNet(d.C, d.S, hidden, nh, N=d.N, dropout=DROPOUT,
                            n_quantile=nq)
    if name in ("GCN", "GraphSAGE", "GAT"):
        kind = {"GCN": "gcn", "GraphSAGE": "sage", "GAT": "gat"}[name]
        return GraphForecaster(d.C, d.S, hidden, nh, kind, N_HEADS, DROPOUT)
    return STMRGAT(d.C, d.S, hidden, nh, nq, n_rel, N_HEADS, DROPOUT)


def train_neural(name, d, graphs, seed, rels=("geo", "road", "flow"),
                 quantile=True, verbose=False, hidden=HIDDEN_DIM,
                 quantile_grid=None, select_by="loss", select_tau=0.9,
                 select_horizon=0):
    """
    select_by = "loss" : dung bo theo pinball/MSE tren val_es  (mac dinh, nhu cu)
    select_by = "cost" : dung bo theo CHI PHI NEWSVENDOR tren val_es.

    Vi sao co lua chon thu hai: luan diem cua bai la phai DANH GIA theo ham quyet
    dinh. Neu vay thi cung khong nen CHON mo hinh theo ham mat mat. Tieu chi "cost"
    chon epoch toi thieu hoa chi phi quyet dinh o phan vi select_tau va tam
    select_horizon, tinh trong khong gian so dem goc tren tap val_es.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_multirel = name in MULTIREL
    qgrid = list(QUANTILES if quantile_grid is None else quantile_grid)
    is_quant = quantile and name in QUANTILE_CAPABLE
    n_q = len(qgrid) if is_quant else 1
    model = build_model(name, d, len(rels), n_q, hidden=hidden).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)

    static = torch.from_numpy(d.static).to(DEV)
    edge_list = [graphs[r] for r in rels]
    ei = graphs["geo"]

    def fwd(x, base):
        if is_multirel:
            o = model(x, static, edge_list)[0]
        elif name in GRAPH_MODELS:
            o = model(x, static, ei)[0]
        else:
            o = model(x, static)[0]
        # A single decision quantile still needs an explicit quantile axis so
        # pinball_loss can broadcast [B,N,H,1] against the target.
        if is_quant and o.dim() == 3:
            o = o.unsqueeze(-1)
        return o + (base.unsqueeze(-1) if o.dim() == 4 else base)

    xtr, ytr, _, btr = [t.to(DEV) for t in d.get("train")]
    _xva, _yva, _yva_raw, _bva = d.get("val_es")
    xva, yva, bva = _xva.to(DEV), _yva.to(DEV), _bva.to(DEV)
    yva_raw = _yva_raw.numpy()

    mse = nn.MSELoss()
    use_pinball = is_quant
    # Chon epoch theo chi phi quyet dinh chi kha thi khi mo hinh xuat phan phoi
    by_cost = (select_by == "cost") and use_pinball
    if by_cost:
        q_sel = (qgrid.index(select_tau) if select_tau in qgrid
                 else len(qgrid) // 2)
    best, best_state, bad, ep = float("inf"), None, 0, 0

    def val_score(out):
        """Chi so dung bo tren val_es (cang nho cang tot)."""
        if by_cost:
            a = np.maximum(d.inverse(out.cpu().numpy())[..., q_sel], 0.0)
            h = select_horizon
            y = yva_raw[..., h]
            short = np.maximum(y - a[..., h], 0.0)
            over = np.maximum(a[..., h] - y, 0.0)
            return float((select_tau * short
                          + (1.0 - select_tau) * over).mean())
        return (pinball_loss(out, yva, qgrid) if use_pinball
                else mse(out.squeeze(-1) if out.dim() == 4 else out, yva)).item()

    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(xtr), device=DEV)
        for i in range(0, len(xtr), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            out = fwd(xtr[b], btr[b])
            loss = (pinball_loss(out, ytr[b], qgrid) if use_pinball
                    else mse(out.squeeze(-1) if out.dim() == 4 else out, ytr[b]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = val_score(fwd(xva, bva))
        sched.step(vl)

        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if verbose and ep % 10 == 0:
            print(f"    ep{ep:3d} val={vl:.5f} best={best:.5f}")

    model.load_state_dict(best_state)

    @torch.no_grad()
    def predict(split):
        xs, _, yr, bs = d.get(split)
        o = [fwd(xs[i:i + BATCH_SIZE].to(DEV), bs[i:i + BATCH_SIZE].to(DEV)).cpu()
             for i in range(0, len(xs), BATCH_SIZE)]
        return torch.cat(o).numpy(), yr.numpy()

    model.eval()
    out, yte = predict("test")

    if use_pinball:
        pq = d.inverse(out)
        qmed = qgrid.index(0.5) if 0.5 in qgrid else 0
        pred = pq[..., qmed]
        m = metrics(pred, yte)
        m.update(interval_metrics(pq, yte, "_raw"))
        # Hieu chuan conformal tren val_cal - phan VAL khong tham gia early stopping,
        # nen tap hieu chuan doc lap voi viec chon mo hinh (dieu kien cua CQR).
        ov, yv = predict("val_cal")
        qv = d.inverse(ov)
        off = conformal_offset(qv, yv, alpha=1 - (QUANTILES[-1] - QUANTILES[0]))
        pq_cal = pq.copy()
        pq_cal[..., 0] -= off
        pq_cal[..., -1] += off
        m.update(interval_metrics(pq_cal, yte))
        m["conformal_offset"] = off
        m["_pred"] = pq_cal
        # Lo ra phan vi CHUA hieu chuan tren test va tren tap hieu chuan, de cac
        # so do hieu chuan khac (vi du Mondrian theo nhom) so sanh duoc ma khong
        # phai huan luyen lai mo hinh.
        m["_pred_raw"] = pq
        m["_cal_pred_raw"] = qv
        m["_cal_true"] = yv
    else:
        if out.ndim == 4:
            out = out[..., 0]
        pred = d.inverse(out)
        m = metrics(pred, yte)
        m["_pred"] = pred
    m["epochs"] = ep + 1
    m["params"] = sum(p.numel() for p in model.parameters())
    return m, model


# ------------------------------------------------------------------ driver
NEURAL = ["MLP", "LSTM", "GCN", "GraphSAGE", "GAT", "DCRNN", "GraphWaveNet",
          "ST-MRGAT", "GraphWaveNet-Q"]

ABLATIONS = {
    "ST-MRGAT":            dict(rels=("geo", "road", "flow"), quantile=True),
    "  w/o road":          dict(rels=("geo", "flow"), quantile=True),
    "  w/o flow":          dict(rels=("geo", "road"), quantile=True),
    "  geo only":          dict(rels=("geo",), quantile=True),
    "  point loss":        dict(rels=("geo", "road", "flow"), quantile=False),
}


def summarize(results):
    rows = []
    for nm, ms in results.items():
        r = {"model": nm, "n": len(ms)}
        for k in ("MAE", "RMSE", "WMAPE", "R2"):
            v = [x[k] for x in ms]
            r[k], r[k + "_std"] = float(np.mean(v)), float(np.std(v))
        for k in ("PICP", "MPIW", "PICP_raw", "MPIW_raw", "params"):
            if k in ms[0]:
                r[k] = float(np.mean([x[k] for x in ms]))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("MAE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default="all")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--seeds", type=int, default=0,
                    help="so seed (0 = dung SEEDS trong config)")
    ap.add_argument("--append", action="store_true",
                    help="gop vao ket qua da co thay vi ghi de")
    args = ap.parse_args()

    seeds = [0] if args.quick else (SEEDS[:args.seeds] if args.seeds else SEEDS)
    d = STData()
    d.summary()
    graphs = load_graphs(DEV)
    print("edges:", {k: v.shape[1] for k, v in graphs.items()})

    results, saved = defaultdict(list), {}
    t0 = time.time()

    print("\n=== Baseline khong hoc / co dien ===")
    for nm, fn in [("Persistence", run_persistence),
                   ("HistoricalAvg", run_historical_average),
                   ("Ridge", run_ridge), ("XGBoost", run_xgboost)]:
        if args.append and nm in results:
            print(f"  {nm:16s} (giu ket qua cu)")
            continue
        m = fn(d)
        results[nm].append(m)
        print(f"  {nm:16s} MAE={m['MAE']:7.3f} RMSE={m['RMSE']:8.3f} "
              f"WMAPE={m['WMAPE']:6.2f}% R2={m['R2']:.4f}")

    jobs = ([(n, dict(rels=("geo", "road", "flow"), quantile=True)) for n in
             (NEURAL if args.models == "all" else args.models.split(","))])
    if args.ablation:
        jobs = list(ABLATIONS.items())

    print("\n=== Mo hinh neural ===")
    for nm, kw in jobs:
        if args.append and nm in results:
            print(f"  {nm:16s} (giu ket qua cu, {len(results[nm])} seed)")
            continue
        for s in seeds:
            t_run = time.time()
            m, _ = train_neural(nm.strip() if nm.strip() in NEURAL else "ST-MRGAT",
                                d, graphs, s, **kw)
            dt = time.time() - t_run
            p = m.pop("_pred")
            # Cac khoa "_..." chua mang numpy; phai bo truoc khi tong hop va ghi
            # JSON, neu khong ban ket qua se hong.
            for k in [k for k in m if k.startswith("_")]:
                m.pop(k)
            if s == seeds[0]:
                # Lam sach ten khoa: np.savez dung khoa lam ten tep trong zip,
                # ky tu '/' se tao thu muc va lam hong buoc doc lai.
                key = nm.strip().replace("/", "-").replace(" ", "_")
                saved[key] = p
            results[nm].append(m)
            extra = f" PICP={m['PICP']:.1f}%" if "PICP" in m else ""
            print(f"  {nm:16s} seed={s} MAE={m['MAE']:7.3f} RMSE={m['RMSE']:8.3f} "
                  f"WMAPE={m['WMAPE']:6.2f}% R2={m['R2']:.4f} ep={m['epochs']}{extra} "
                  f"[{dt:.0f}s]", flush=True)

    _, _, yte, _ = d.get("test")
    tag = "_ablation" if args.ablation else ""
    np.savez_compressed(os.path.join(OUT_DIR, f"test_preds{tag}.npz"),
                        y_true=yte.numpy(), **saved)

    df = summarize(results)
    df.to_csv(os.path.join(OUT_DIR, f"benchmark{tag}.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"benchmark_raw{tag}.json"), "w") as f:
        json.dump({k: [{a: b for a, b in x.items()} for x in v]
                   for k, v in results.items()}, f, indent=2, default=float)

    print(f"\n=== KET QUA TEST ({time.time() - t0:.0f}s) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Kiem dinh y nghia thong ke: mo hinh de xuat vs baseline manh nhat ---
    if len(seeds) > 1 and "ST-MRGAT" in results:
        ours = [x["MAE"] for x in results["ST-MRGAT"]]
        for nm in results:
            if nm == "ST-MRGAT" or len(results[nm]) != len(ours):
                continue
            other = [x["MAE"] for x in results[nm]]
            t, p = stats.ttest_rel(ours, other)
            print(f"  paired t-test MAE  ST-MRGAT vs {nm:14s} "
                  f"t={t:+.2f} p={p:.4f}")
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
