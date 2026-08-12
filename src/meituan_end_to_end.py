"""Same-order arrival -> courier wave -> observed delivery case study (Meituan).

The Meituan/INFORMS data uniquely keeps an order ID through creation, dispatch,
courier acceptance, pickup, and customer arrival, and links it to a courier wave.
We use it for a transparent 5-minute, area-level q=.9 arrival-capacity experiment
and report lifecycle/wave linkage separately.  This is *not* a counterfactual
dispatch or route-optimisation claim: the released data records the observed
assignment and event sequence, not a road GPS trace or a randomized policy.
"""
import argparse
import ast
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
DATA = ROOT / "data" / "meituan"
OUT = ROOT / "outputs"
CACHE = DATA / "meituan_5min_area_arrivals.npz"
Q, L, DEV = [0.1, 0.5, 0.9], 12, ("cuda" if torch.cuda.is_available() else "cpu")
START, TRAIN_END, VAL_END, TEST_END = (pd.Timestamp("2022-10-17"), pd.Timestamp("2022-10-21"),
                                        pd.Timestamp("2022-10-22"), pd.Timestamp("2022-10-24"))
TIME_COLS = ["platform_order_time", "order_push_time", "dispatch_time", "grab_time", "fetch_time", "arrive_time"]


def load_orders():
    w = pd.read_csv(DATA / "all_waybill_info_meituan_0322.csv")
    keep = ((w.is_courier_grabbed.eq(1)) & (w[TIME_COLS].gt(0).all(axis=1))).copy()
    a = w.loc[keep].copy()
    # Dataset documentation specifies that one accepted waybill exists per order.
    if a.order_id.duplicated().any():
        raise RuntimeError("expected one accepted waybill per order")
    for c in TIME_COLS:
        a[c + "_dt"] = pd.to_datetime(a[c], unit="s")
    monotone = np.ones(len(a), bool)
    for left, right in zip(TIME_COLS, TIME_COLS[1:]):
        monotone &= a[left].to_numpy() <= a[right].to_numpy()
    a = a.loc[monotone].copy()
    a["arrival_bin"] = a.platform_order_time_dt.dt.floor("5min")
    return w, a


def build_table(a):
    x = a.loc[(a.arrival_bin >= START) & (a.arrival_bin < TEST_END)].copy()
    areas = np.sort(x.da_id.unique())
    bins = pd.date_range(START, TEST_END, freq="5min", inclusive="left")
    table = (x.groupby(["arrival_bin", "da_id"]).order_id.nunique().unstack(fill_value=0)
             .reindex(index=bins, columns=areas, fill_value=0).astype(np.float32))
    cent = x.groupby("da_id")[["sender_lat", "sender_lng"]].mean().reindex(areas).to_numpy(np.float32)
    return table, cent, x


def load_table(a):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return (pd.DataFrame(z["values"], index=pd.to_datetime(z["index"]), columns=z["columns"]),
                z["centroids"].astype(np.float32))
    table, cent, _ = build_table(a)
    np.savez_compressed(CACHE, values=table.to_numpy(), index=table.index.astype(str).to_numpy(),
                        columns=table.columns.to_numpy(), centroids=cent)
    return table, cent


def graph(centroids):
    xy = (centroids - centroids.mean(0)) / (centroids.std(0) + 1e-8)
    dist = ((xy[:, None] - xy[None, :]) ** 2).sum(-1); edges = set()
    for i in range(len(xy)):
        for j in np.argsort(dist[i])[1:4]: edges.update(((i, int(j)), (int(j), i)))
    src, dst = zip(*sorted(edges))
    return (torch.tensor(np.array([src, dst]), dtype=torch.long, device=DEV),
            torch.tensor(xy, dtype=torch.float32, device=DEV))


def split(table):
    y = table.to_numpy(np.float32); T, N = y.shape
    idx = table.index
    tr_end, va_end = int(np.searchsorted(idx.values, TRAIN_END.to_datetime64())), int(np.searchsorted(idx.values, VAL_END.to_datetime64()))
    if not (tr_end > L and va_end > tr_end and T > va_end): raise RuntimeError("unexpected Meituan temporal split")
    ly = np.log1p(y); mu, sd = float(ly[:tr_end].mean()), float(ly[:tr_end].std() + 1e-8); z = (ly-mu)/sd
    tod = (idx.hour * 12 + idx.minute // 5).to_numpy()
    profile = np.zeros((288, N), np.float32); cnt = np.zeros(288, np.float32)
    for t in range(tr_end): profile[tod[t]] += y[t]; cnt[tod[t]] += 1
    profile /= np.maximum(cnt[:, None], 1); base = (np.log1p(profile[tod])-mu)/sd; x = np.stack([z, base], -1)
    def build(lo, hi):
        target = np.arange(max(L, lo), hi)
        xx = np.stack([x[t-L:t] for t in target]).transpose(0,2,1,3)
        return (torch.from_numpy(np.ascontiguousarray(xx)), torch.from_numpy(np.ascontiguousarray(z[target,:,None])),
                torch.from_numpy(np.ascontiguousarray(y[target,:,None])), torch.from_numpy(np.ascontiguousarray(base[target,:,None])), target)
    # First 60% of the validation day is early stopping; final 40% calibrates CQR.
    mid = tr_end + (va_end-tr_end)*3//5
    return dict(train=build(0,tr_end), val_es=build(tr_end,mid), val_cal=build(mid,va_end), test=build(va_end,T),
                raw=y, tod=tod, mu=mu, sd=sd, n_nodes=N, tr_end=tr_end, va_end=va_end, test_times=idx[va_end:])


def inverse(z, mu, sd): return np.expm1(np.clip(z, -30, 30)*sd + mu)


def cqr(pq, y):
    score = np.maximum(pq[...,0]-y, y-pq[...,-1]).ravel()
    return float(np.quantile(score, min(1., np.ceil((len(score)+1)*.8)/len(score)), method="higher"))


def ci(v):
    v=np.asarray(v,float); m=float(v.mean())
    if len(v)<2: return m,np.nan,np.nan
    h=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)); return m,m-h,m+h


def train_one(d, edge, static, seed, hidden, epochs, patience, batch):
    torch.manual_seed(seed); np.random.seed(seed)
    model=GraphWaveNet(2,static.shape[1],hidden,1,N=d["n_nodes"],n_quantile=3).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=5)
    tr=[v.to(DEV) if torch.is_tensor(v) else v for v in d["train"][:4]]; va=[v.to(DEV) if torch.is_tensor(v) else v for v in d["val_es"][:4]]
    def f(x,b): return model(x,static,edge)[0]+b.unsqueeze(-1)
    best,state,bad=float("inf"),None,0
    for ep in range(epochs):
        model.train(); perm=torch.randperm(len(tr[0]),device=DEV)
        for i in range(0,len(perm),batch):
            ix=perm[i:i+batch]; opt.zero_grad(); loss=pinball_loss(f(tr[0][ix],tr[3][ix]),tr[1][ix],Q); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
        model.eval()
        with torch.no_grad(): val=pinball_loss(f(va[0],va[3]),va[1],Q).item()
        sch.step(val)
        if val<best-1e-6: best,bad,state=val,0,{k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(state); model.eval()
    @torch.no_grad()
    def pred(name):
        x,_,y,b=d[name][:4]; out=[]
        for i in range(0,len(x),batch): out.append(f(x[i:i+batch].to(DEV),b[i:i+batch].to(DEV)).cpu())
        return inverse(torch.cat(out).numpy(),d["mu"],d["sd"]),y.numpy()
    pq,truth4=pred("test"); pv,yv=pred("val_cal"); off=cqr(pv,yv); pq[...,0]-=off; pq[...,-1]+=off
    q90,q50,truth=np.maximum(pq[:,:,0,-1],0),np.maximum(pq[:,:,0,1],0),truth4[:,:,0]
    r=newsvendor(q90,truth,.9); r.update(model="GraphWaveNet-Q (Meituan retrained)",seed=seed,epochs=ep+1,params=sum(p.numel() for p in model.parameters()),mae=float(np.abs(q50-truth).mean()),cqr_offset=off)
    return r


def historical(d):
    tr=d["raw"][:d["tr_end"]]; truth=d["raw"][d["va_end"]:]; times=d["test_times"]
    g=pd.DataFrame(tr).groupby(d["tod"][:d["tr_end"]]); med,q90=g.median(),g.quantile(.9)
    rows=[]
    for name, a in {"Historical median (Meituan train)":np.asarray([med.loc[t.hour*12+t.minute//5] for t in times]),
                    "Historical q0.9 (Meituan train)":np.asarray([q90.loc[t.hour*12+t.minute//5] for t in times])}.items():
        r=newsvendor(a,truth,.9); r.update(model=name,seed=-1,epochs=0,params=0,mae=float(np.abs(a-truth).mean()),cqr_offset=0.); rows.append(r)
    return rows


def block_saving(d, rng):
    tr=d["raw"][:d["tr_end"]]; truth=d["raw"][d["va_end"]:]; g=pd.DataFrame(tr).groupby(d["tod"][:d["tr_end"]]); med,q90=g.median(),g.quantile(.9)
    a=np.asarray([med.loc[t.hour*12+t.minute//5] for t in d["test_times"]]); b=np.asarray([q90.loc[t.hour*12+t.minute//5] for t in d["test_times"]])
    ca=(.1*np.maximum(a-truth,0)+.9*np.maximum(truth-a,0)).sum(1); cb=(.1*np.maximum(b-truth,0)+.9*np.maximum(truth-b,0)).sum(1)
    T,block=len(ca),12; vals=[]
    for _ in range(2000):
        ix=[]
        while len(ix)<T: s=int(rng.integers(0,T-block+1)); ix.extend(range(s,s+block))
        ix=np.asarray(ix[:T]); vals.append(100*(1-cb[ix].sum()/ca[ix].sum()))
    return 100*(1-cb.sum()/ca.sum()),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))


def lifecycle(a, raw):
    wave=pd.read_csv(DATA / "courier_wave_info_meituan.csv")
    pairs=set()
    for row in wave.itertuples(index=False):
        try: pairs.update((int(row.courier_id),int(x)) for x in ast.literal_eval(row.order_ids))
        except (ValueError,SyntaxError): pass
    matched=np.fromiter(((int(c),int(o)) in pairs for c,o in zip(a.courier_id,a.order_id)),bool,count=len(a))
    mins=(a.arrive_time-a.platform_order_time)/60
    return {"accepted_orders":len(a),"unique_orders":a.order_id.nunique(),"source_waybills":len(raw),"rejected_waybills":int((raw.is_courier_grabbed==0).sum()),
            "couriers":a.courier_id.nunique(),"areas":a.da_id.nunique(),"waves":len(wave),"wave_link_pct":100*float(matched.mean()),
            "ordered_lifecycle_pct":100.0,"median_create_to_delivery_min":float(mins.median()),"p90_create_to_delivery_min":float(mins.quantile(.9)),
            "late_pct":100*float((a.arrive_time>a.estimate_arrived_time).mean())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=3); ap.add_argument("--hidden",type=int,default=32)
    ap.add_argument("--epochs",type=int,default=100); ap.add_argument("--patience",type=int,default=15); ap.add_argument("--batch",type=int,default=128); args=ap.parse_args(); OUT.mkdir(exist_ok=True)
    raw,a=load_orders(); table,cent=load_table(a); d=split(table); edge,static=graph(cent); rows=historical(d); saving,lo,hi=block_saving(d,np.random.default_rng(20260729))
    for seed in range(args.seeds):
        r=train_one(d,edge,static,seed,args.hidden,args.epochs,args.patience,args.batch); rows.append(r); pd.DataFrame(rows).to_csv(OUT/"meituan_end_to_end_detail.csv",index=False); print(f"seed={seed} cost={r['cost']:.4f} MAE={r['mae']:.3f} ep={r['epochs']}",flush=True)
    df=pd.DataFrame(rows); nn=df[df.seed>=0]; cm,cl,ch=ci(nn.cost); mm,ml,mh=ci(nn.mae); hq=float(df.loc[df.model.eq("Historical q0.9 (Meituan train)"),"cost"].iloc[0])
    s=lifecycle(a,raw); s.update({"model":"GraphWaveNet-Q (Meituan retrained)","n_seeds":len(nn),"cost":cm,"cost_ci_low":cl,"cost_ci_high":ch,"mae":mm,"mae_ci_low":ml,"mae_ci_high":mh,
              "params":nn.params.mean(),"train_start":str(START.date()),"train_end":"2022-10-20","validation":"2022-10-21","test_start":"2022-10-22","test_end":"2022-10-23",
              "historical_q90_vs_median_saving_pct":saving,"historical_q90_vs_median_ci_low":lo,"historical_q90_vs_median_ci_high":hi,"gwnq_vs_historical_q90_saving_pct":100*(1-cm/hq)})
    pd.DataFrame([s,*[r for r in rows if r["seed"]<0]]).to_csv(OUT/"meituan_end_to_end_summary.csv",index=False)
    print(pd.DataFrame([s,*[r for r in rows if r["seed"]<0]]).to_string(index=False,float_format=lambda x:f"{x:.4f}"))


if __name__=="__main__": main()
