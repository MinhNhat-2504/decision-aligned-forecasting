"""Chronological real parcel-order arrival -> regional capacity experiment.

Olist is independent of the Amazon route corpus.  It supplies the missing public
order-arrival stream: fulfilled e-commerce orders are observed at purchase time
and aggregated to customer states on a daily clock.  A fresh Graph WaveNet-Q is
fit only on preceding Olist dates, then its q=.9 order-capacity allocations are
evaluated on a later holdout.  It must not be interpreted as route optimisation.
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
from models import GraphWaveNet, pinball_loss
from operations import newsvendor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "olist"
OUT = ROOT / "outputs"
CACHE = DATA / "olist_daily_state_orders.npz"
Q = [0.1, 0.5, 0.9]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = 14


def load_table():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return (pd.DataFrame(z["values"], index=pd.to_datetime(z["index"]), columns=z["columns"]),
                z["centroids"].astype(np.float32), int(z["n_orders"]), "2017-01-01", str(z["end"]))
    orders = pd.read_csv(DATA / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    customers = pd.read_csv(DATA / "olist_customers_dataset.csv", usecols=["customer_id", "customer_state", "customer_zip_code_prefix"])
    geo = pd.read_csv(DATA / "olist_geolocation_dataset.csv", usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"])
    # The task is fulfilled outbound capacity, so completed orders only; events are
    # nevertheless dated at the purchase-time arrival, never delivery time.
    d = orders.loc[orders.order_status.eq("delivered"), ["order_id", "customer_id", "order_purchase_timestamp"]]
    d = d.merge(customers, on="customer_id", how="inner")
    d["day"] = d.order_purchase_timestamp.dt.normalize()
    # 2016 is an incomplete launch tail.  The fixed 2017 start is declared before
    # splitting and avoids treating a handful of launch orders as a normal regime.
    d = d.loc[(d.day >= "2017-01-01") & (d.day <= "2018-08-29")].copy()
    states = np.sort(d.customer_state.unique())
    dates = pd.date_range("2017-01-01", d.day.max(), freq="D")
    table = (d.groupby(["day", "customer_state"]).order_id.nunique().unstack(fill_value=0)
             .reindex(index=dates, columns=states, fill_value=0).astype(np.float32))
    # Static coordinates come from the public ZIP geolocation release, not outcomes.
    zip_geo = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    cgeo = customers.merge(zip_geo, left_on="customer_zip_code_prefix", right_index=True, how="left")
    cent = cgeo.groupby("customer_state")[["geolocation_lat", "geolocation_lng"]].mean().reindex(states)
    # The few ZIPs absent from geolocation are replaced by the national coordinate
    # mean solely to construct a topology; no target information is used.
    cent = cent.fillna(cent.mean())
    np.savez_compressed(CACHE, values=table.to_numpy(), index=table.index.astype(str).to_numpy(),
                        columns=table.columns.to_numpy(), centroids=cent.to_numpy(np.float32),
                        n_orders=len(d), start="2017-01-01", end=str(d.day.max().date()))
    return table, cent.to_numpy(np.float32), len(d), "2017-01-01", str(d.day.max().date())


def graph_and_static(centroids):
    # Three nearest geographic state neighbours; symmetric directed edges.
    xy = centroids.copy(); xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-8)
    dist = ((xy[:, None] - xy[None, :]) ** 2).sum(-1)
    edges = set()
    for i in range(len(xy)):
        for j in np.argsort(dist[i])[1:4]:
            edges.add((i, int(j))); edges.add((int(j), i))
    src, dst = zip(*sorted(edges))
    return (torch.tensor(np.array([src, dst]), dtype=torch.long, device=DEV),
            torch.tensor(xy, dtype=torch.float32, device=DEV))


def make_split(table):
    y = table.to_numpy(np.float32); T, N = y.shape
    tr_end, va_end = int(.60 * T), int(.70 * T)
    ly = np.log1p(y); mu, sd = float(ly[:tr_end].mean()), float(ly[:tr_end].std() + 1e-8)
    z = (ly - mu) / sd
    dow = table.index.dayofweek.to_numpy()
    profile = np.zeros((7, N), np.float32); cnt = np.zeros(7, np.float32)
    for t in range(tr_end): profile[dow[t]] += y[t]; cnt[dow[t]] += 1
    profile /= np.maximum(cnt[:, None], 1)
    base = (np.log1p(profile[dow]) - mu) / sd
    x = np.stack([z, base], axis=-1)
    def build(lo, hi):
        target = np.arange(max(L, lo), hi)
        xx = np.stack([x[t-L:t] for t in target]).transpose(0, 2, 1, 3)
        return (torch.from_numpy(np.ascontiguousarray(xx)), torch.from_numpy(np.ascontiguousarray(z[target, :, None])),
                torch.from_numpy(np.ascontiguousarray(y[target, :, None])), torch.from_numpy(np.ascontiguousarray(base[target, :, None])), target)
    mid = tr_end + (va_end - tr_end) * 3 // 5
    return dict(train=build(0, tr_end), val_es=build(tr_end, mid), val_cal=build(mid, va_end), test=build(va_end, T),
                raw=y, dow=dow, mu=mu, sd=sd, n_nodes=N, test_times=table.index[va_end:], tr_end=tr_end, va_end=va_end)


def inverse(z, mu, sd): return np.expm1(np.clip(z, -30, 30) * sd + mu)


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
    def fwd(x, base): return model(x, static, edge)[0] + base.unsqueeze(-1)
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
            best, bad = val, 0; state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    @torch.no_grad()
    def pred(split):
        x, _, yy, b = d[split][:4]; out=[]
        for i in range(0, len(x), batch): out.append(fwd(x[i:i+batch].to(DEV), b[i:i+batch].to(DEV)).cpu())
        return inverse(torch.cat(out).numpy(), d["mu"], d["sd"]), yy.numpy()
    pq, truth4 = pred("test"); qv, yv4 = pred("val_cal")
    off = cqr_offset(qv, yv4); pq[..., 0] -= off; pq[..., -1] += off
    q90, q50, truth = np.maximum(pq[:, :, 0, -1], 0), np.maximum(pq[:, :, 0, 1], 0), truth4[:, :, 0]
    r = newsvendor(q90, truth, .9)
    r.update(seed=seed, model="GraphWaveNet-Q (Olist retrained)", epochs=ep + 1, params=sum(p.numel() for p in model.parameters()),
             mae=float(np.abs(q50-truth).mean()), cqr_offset=off)
    return r, q90, truth


def historical(d):
    tr = d["raw"][:d["tr_end"]]; times = d["test_times"]; truth = d["raw"][d["va_end"]:]
    grouped = pd.DataFrame(tr).groupby(d["dow"][:d["tr_end"]])
    med, q90 = grouped.median(), grouped.quantile(.9)
    acts = {"Historical median (Olist train)": np.asarray([med.loc[t.dayofweek] for t in times]),
            "Historical q0.9 (Olist train)": np.asarray([q90.loc[t.dayofweek] for t in times])}
    rows, decisions = [], {}
    for name, a in acts.items():
        r = newsvendor(a, truth, .9); r.update(seed=-1, model=name, epochs=0, params=0, mae=float(np.abs(a-truth).mean()), cqr_offset=0.)
        rows.append(r); decisions[name] = a
    return rows, decisions, truth


def block_saving(q50, q90, truth, rng):
    """Five-day moving-block bootstrap of proportional operational saving."""
    c50 = .1 * np.maximum(q50-truth, 0) + .9 * np.maximum(truth-q50, 0)
    c90 = .1 * np.maximum(q90-truth, 0) + .9 * np.maximum(truth-q90, 0)
    a, b = c50.sum(1), c90.sum(1); T, block = len(a), 5
    values=[]
    for _ in range(2000):
        ix=[]
        while len(ix) < T:
            s=int(rng.integers(0, T-block+1)); ix.extend(range(s,s+block))
        ix=np.asarray(ix[:T]); values.append(100*(1-b[ix].sum()/a[ix].sum()))
    return float(100*(1-b.sum()/a.sum())), float(np.quantile(values,.025)), float(np.quantile(values,.975))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32); ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=18); ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args(); OUT.mkdir(exist_ok=True)
    table, centroids, n_orders, start, end = load_table(); d = make_split(table); edge, static = graph_and_static(centroids)
    rows, decisions, truth = historical(d)
    saving, lo, hi = block_saving(decisions["Historical median (Olist train)"], decisions["Historical q0.9 (Olist train)"], truth, np.random.default_rng(20260729))
    for seed in range(a.seeds):
        r, _, _ = train_one(d, edge, static, seed, a.hidden, a.epochs, a.patience, a.batch); rows.append(r)
        pd.DataFrame(rows).to_csv(OUT / "olist_order_arrival_detail.csv", index=False)
        print(f"seed={seed} cost={r['cost']:.4f} MAE={r['mae']:.3f} ep={r['epochs']}", flush=True)
    df = pd.DataFrame(rows); nn = df[df.seed >= 0]; cm, cl, ch = ci(nn.cost); mm, ml, mh = ci(nn.mae)
    hq_cost = float(df.loc[df.model.eq("Historical q0.9 (Olist train)"), "cost"].iloc[0])
    summary = pd.DataFrame([{
        "model":"GraphWaveNet-Q (Olist retrained)", "n_seeds":len(nn), "cost":cm, "cost_ci_low":cl, "cost_ci_high":ch,
        "mae":mm, "mae_ci_low":ml, "mae_ci_high":mh, "params":nn.params.mean(), "n_states":table.shape[1],
        "n_fulfilled_orders":n_orders, "date_start":start, "date_end":end, "test_start":str(d["test_times"][0].date()),
        "test_end":str(d["test_times"][-1].date()), "historical_q90_vs_median_saving_pct":saving,
        "historical_q90_vs_median_ci_low":lo, "historical_q90_vs_median_ci_high":hi,
        "gwnq_vs_historical_q90_saving_pct":100 * (1 - cm / hq_cost)
    }, *[r for r in rows if r["seed"] < 0]])
    summary.to_csv(OUT / "olist_order_arrival_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__": main()
