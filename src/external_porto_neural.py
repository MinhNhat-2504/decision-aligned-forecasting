"""Independent Porto retraining of the Graph WaveNet quantile pipeline.

This is intentionally separate from the NYC ``STData`` pipeline: it constructs a
new 54-cell grid graph, fits all transforms and seasonal profiles using Porto train
hours only, initialises a fresh Graph WaveNet-Q for every seed, and uses a later
Porto validation block for early stopping and CQR.  It therefore tests *model*
transferability by retraining, never by reusing NYC weights or graph topology.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_porto import aggregate
from models import GraphWaveNet, pinball_loss
from operations import newsvendor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "porto_hourly_grid.npz"
OUT = ROOT / "outputs"
Q = [0.1, 0.5, 0.9]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = 12


def load_table():
    """Cache the deterministic hourly grid after the first full archive pass."""
    if DATA.exists():
        z = np.load(DATA, allow_pickle=True)
        return pd.DataFrame(z["values"], index=pd.to_datetime(z["index"]), columns=z["columns"])
    table = aggregate()
    np.savez_compressed(DATA, values=table.to_numpy(np.float32),
                        index=table.index.astype(str).to_numpy(),
                        columns=table.columns.to_numpy())
    return table


def graph_and_static(columns):
    xy = np.array([[int(c.split("_")[0][1:]), int(c.split("_")[1])] for c in columns], dtype=float)
    # Four-neighbour grid where available; add the nearest cell for any boundary
    # isolate.  This is built only from Porto grid geometry.
    dist = np.abs(xy[:, None] - xy[None, :]).sum(-1)
    src, dst = np.where((dist > 0) & (dist <= 1))
    for i in range(len(xy)):
        if not np.any(src == i):
            j = np.argsort(dist[i])[1]
            src, dst = np.append(src, i), np.append(dst, j)
    static = (xy - xy.mean(0)) / (xy.std(0) + 1e-8)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long, device=DEV), torch.tensor(static, dtype=torch.float32, device=DEV)


def make_split(table):
    y = table.to_numpy(np.float32)
    T, N = y.shape
    tr_end, va_end = int(.60 * T), int(.70 * T)  # final 30% is held-out test
    ly = np.log1p(y)
    mu, sd = float(ly[:tr_end].mean()), float(ly[:tr_end].std() + 1e-8)
    z = (ly - mu) / sd
    how = table.index.dayofweek.to_numpy() * 24 + table.index.hour.to_numpy()
    profile = np.zeros((168, N), np.float32); cnt = np.zeros(168, np.float32)
    for t in range(tr_end):
        profile[how[t]] += y[t]; cnt[how[t]] += 1
    profile /= np.maximum(cnt[:, None], 1)
    base = (np.log1p(profile[how]) - mu) / sd
    x = np.stack([z, base], axis=-1)

    def build(lo, hi):
        target = np.arange(max(L, lo), hi)
        xx = np.stack([x[t-L:t] for t in target]).transpose(0, 2, 1, 3)
        return (torch.from_numpy(np.ascontiguousarray(xx)),
                torch.from_numpy(np.ascontiguousarray(z[target, :, None])),
                torch.from_numpy(np.ascontiguousarray(y[target, :, None])),
                torch.from_numpy(np.ascontiguousarray(base[target, :, None])), target)
    # Disjoint early-stopping and calibration regions within validation.
    mid = tr_end + (va_end - tr_end) * 3 // 5
    return dict(train=build(0, tr_end), val_es=build(tr_end, mid),
                val_cal=build(mid, va_end), test=build(va_end, T),
                raw=y, profile=profile, how=how, mu=mu, sd=sd,
                n_nodes=N, test_times=table.index[va_end:])


def inverse(z, mu, sd):
    return np.expm1(np.clip(z, -30, 30) * sd + mu)


def cqr_offset(pq, y):
    score = np.maximum(pq[..., 0] - y, y - pq[..., -1]).ravel()
    level = min(1., np.ceil((len(score) + 1) * .8) / len(score))
    return float(np.quantile(score, level, method="higher"))


def ci(x):
    x = np.asarray(x, float); m = x.mean()
    if len(x) < 2: return m, np.nan, np.nan
    h = stats.t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m-h, m+h


def train_one(d, edge, static, seed, hidden, epochs, patience, batch):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GraphWaveNet(2, static.shape[1], hidden, 1, N=d["n_nodes"], n_quantile=3).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=5)
    tr = [v.to(DEV) if torch.is_tensor(v) else v for v in d["train"][:4]]
    va = [v.to(DEV) if torch.is_tensor(v) else v for v in d["val_es"][:4]]
    def fwd(x, base):
        return model(x, static, edge)[0] + base.unsqueeze(-1)
    best, state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr[0]), device=DEV)
        for i in range(0, len(perm), batch):
            ix = perm[i:i+batch]; opt.zero_grad()
            loss = pinball_loss(fwd(tr[0][ix], tr[3][ix]), tr[1][ix], Q)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
        model.eval()
        with torch.no_grad(): val = pinball_loss(fwd(va[0], va[3]), va[1], Q).item()
        sch.step(val)
        if val < best - 1e-6:
            best, bad = val, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    @torch.no_grad()
    def pred(split):
        x, _, y, b = d[split][:4]; out=[]
        for i in range(0, len(x), batch): out.append(fwd(x[i:i+batch].to(DEV), b[i:i+batch].to(DEV)).cpu())
        return inverse(torch.cat(out).numpy(), d["mu"], d["sd"]), y.numpy()
    pq, y = pred("test"); qv, yv = pred("val_cal")
    off = cqr_offset(qv, yv); pq[..., 0] -= off; pq[..., -1] += off
    q90, q50, truth = pq[:, :, 0, -1], pq[:, :, 0, 1], y[:, :, 0]
    r = newsvendor(np.maximum(q90, 0), truth, .9)
    r.update(seed=seed, model="GraphWaveNet-Q (Porto retrained)", epochs=ep+1,
             params=sum(p.numel() for p in model.parameters()),
             mae=float(np.abs(np.maximum(q50, 0) - truth).mean()),
             cqr_offset=off)
    return r


def historical(d):
    y = d["raw"][int(.70 * len(d["raw"])):]
    times = d["test_times"]
    # Both policies are estimated from the same Porto training sequence only.
    tr = d["raw"][:int(.60 * len(d["raw"]))]
    keys = d["how"][:len(tr)]
    grouped = pd.DataFrame(tr).groupby(keys)
    med, q90 = grouped.median(), grouped.quantile(.9)
    acts = {"Historical median (Porto train)": np.asarray([med.loc[t.dayofweek*24+t.hour] for t in times]),
            "Historical q0.9 (Porto train)": np.asarray([q90.loc[t.dayofweek*24+t.hour] for t in times])}
    rows=[]
    for name, a in acts.items():
        r=newsvendor(a, y, .9); r.update(seed=-1, model=name, epochs=0, params=0, mae=float(np.abs(a-y).mean()), cqr_offset=0.)
        rows.append(r)
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15); ap.add_argument("--batch", type=int, default=64)
    a=ap.parse_args(); OUT.mkdir(exist_ok=True)
    table=load_table(); d=make_split(table); edge, static=graph_and_static(table.columns)
    rows=historical(d)
    for seed in range(a.seeds):
        r=train_one(d, edge, static, seed, a.hidden, a.epochs, a.patience, a.batch); rows.append(r)
        pd.DataFrame(rows).to_csv(OUT/"external_porto_neural_detail.csv", index=False)
        print(f"seed={seed} cost={r['cost']:.4f} MAE={r['mae']:.3f} ep={r['epochs']}", flush=True)
    df=pd.DataFrame(rows); nn=df[df.seed>=0]
    cm,cl,ch=ci(nn.cost); mm,ml,mh=ci(nn.mae)
    summary=pd.DataFrame([{"model":"GraphWaveNet-Q (Porto retrained)","n_seeds":len(nn),"cost":cm,"cost_ci_low":cl,"cost_ci_high":ch,"mae":mm,"mae_ci_low":ml,"mae_ci_high":mh,"params":nn.params.mean()}, *[r for r in rows if r["seed"]<0]])
    summary.to_csv(OUT/"external_porto_neural_summary.csv",index=False)
    print(summary.to_string(index=False, float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__": main()
