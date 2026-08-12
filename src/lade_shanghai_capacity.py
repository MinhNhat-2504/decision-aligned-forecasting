"""Long-horizon LaDe Shanghai parcel task-load capacity experiment.

The full LaDe Shanghai file covers 184 days and records a parcel task's courier
acceptance/delivery event times plus GPS at both events.  It has no physical vehicle
volume/capacity field, therefore the decision unit below is an accepted courier-task
slot, not truck capacity.  This distinction is part of the reported scope.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import GraphWaveNet, pinball_loss
from operations import newsvendor

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'lade'; OUT=ROOT/'outputs'; CACHE=DATA/'lade_shanghai_hourly_region.npz'
Q=[.1,.5,.9]; L=12; DEV='cuda' if torch.cuda.is_available() else 'cpu'
START=pd.Timestamp('2022-05-01'); END=pd.Timestamp('2022-11-01')

def load_table():
    if CACHE.exists():
        z=np.load(CACHE,allow_pickle=True); return pd.DataFrame(z['values'],index=pd.to_datetime(z['index']),columns=z['columns']),z['centroids'].astype(np.float32),int(z['n_tasks'])
    sums={}; counts={}; groups=[]; total=0
    for c in pd.read_csv(DATA/'delivery_sh.csv',chunksize=200000):
        ts=pd.to_datetime('2022-'+c.accept_time,format='%Y-%m-%d %H:%M:%S',errors='coerce')
        ok=ts.notna() & (ts>=START) & (ts<END) & c.delivery_time.notna()
        d=c.loc[ok,['region_id','lng','lat']].copy(); d['hour']=ts.loc[ok].dt.floor('h'); total+=len(d)
        groups.append(d.groupby(['hour','region_id']).size())
        g=d.groupby('region_id')[['lng','lat']].agg(['sum','count'])
        for r,row in g.iterrows():
            sums[r]=np.array([row[('lng','sum')],row[('lat','sum')]])+sums.get(r,np.zeros(2)); counts[r]=int(row[('lng','count')])+counts.get(r,0)
    agg=pd.concat(groups).groupby(level=[0,1]).sum(); regions=np.array(sorted(counts)); idx=pd.date_range(START,END,freq='h',inclusive='left')
    table=agg.unstack(fill_value=0).reindex(index=idx,columns=regions,fill_value=0).astype(np.float32)
    cent=np.array([sums[r]/counts[r] for r in regions],np.float32)
    np.savez_compressed(CACHE,values=table.to_numpy(),index=table.index.astype(str).to_numpy(),columns=table.columns.to_numpy(),centroids=cent,n_tasks=total)
    return table,cent,total

def graph(cent):
    xy=(cent-cent.mean(0))/(cent.std(0)+1e-8); ds=((xy[:,None]-xy[None,:])**2).sum(-1); es=set()
    for i in range(len(xy)):
        for j in np.argsort(ds[i])[1:min(4,len(xy))]: es.update(((i,int(j)),(int(j),i)))
    s,t=zip(*sorted(es)); return torch.tensor(np.array([s,t]),dtype=torch.long,device=DEV),torch.tensor(xy,dtype=torch.float32,device=DEV)

def make_split(table):
    y=table.to_numpy(np.float32); T,N=y.shape; te,ve=int(.6*T),int(.7*T); ly=np.log1p(y); mu,sd=float(ly[:te].mean()),float(ly[:te].std()+1e-8); z=(ly-mu)/sd
    how=table.index.dayofweek.to_numpy()*24+table.index.hour.to_numpy(); profile=np.zeros((168,N),np.float32); cnt=np.zeros(168,np.float32)
    for t in range(te): profile[how[t]]+=y[t];cnt[how[t]]+=1
    profile/=np.maximum(cnt[:,None],1); base=(np.log1p(profile[how])-mu)/sd; x=np.stack([z,base],-1)
    def build(lo,hi):
        ix=np.arange(max(L,lo),hi); xx=np.stack([x[t-L:t] for t in ix]).transpose(0,2,1,3)
        return torch.from_numpy(np.ascontiguousarray(xx)),torch.from_numpy(np.ascontiguousarray(z[ix,:,None])),torch.from_numpy(np.ascontiguousarray(y[ix,:,None])),torch.from_numpy(np.ascontiguousarray(base[ix,:,None])),ix
    mid=te+(ve-te)*3//5
    return dict(train=build(0,te),val_es=build(te,mid),val_cal=build(mid,ve),test=build(ve,T),raw=y,how=how,mu=mu,sd=sd,n_nodes=N,tr_end=te,va_end=ve,test_times=table.index[ve:])

def inv(z,mu,sd): return np.expm1(np.clip(z,-30,30)*sd+mu)
def cqr(p,y):
    s=np.maximum(p[...,0]-y,y-p[...,-1]).ravel(); return float(np.quantile(s,min(1.,np.ceil((len(s)+1)*.8)/len(s)),method='higher'))
def ci(x):
    x=np.asarray(x,float);m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return m,m-h,m+h

def train_one(d,e,st,seed,hidden,epochs,patience,batch):
    torch.manual_seed(seed);np.random.seed(seed); m=GraphWaveNet(2,st.shape[1],hidden,1,N=d['n_nodes'],n_quantile=3).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=5)
    tr=[v.to(DEV) if torch.is_tensor(v) else v for v in d['train'][:4]];va=[v.to(DEV) if torch.is_tensor(v) else v for v in d['val_es'][:4]]
    def f(x,b):return m(x,st,e)[0]+b.unsqueeze(-1)
    best,state,bad=float('inf'),None,0
    for ep in range(epochs):
        m.train();pm=torch.randperm(len(tr[0]),device=DEV)
        for i in range(0,len(pm),batch):
            ix=pm[i:i+batch];opt.zero_grad();loss=pinball_loss(f(tr[0][ix],tr[3][ix]),tr[1][ix],Q);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
        m.eval()
        with torch.no_grad():v=pinball_loss(f(va[0],va[3]),va[1],Q).item()
        sch.step(v)
        if v<best-1e-6:best,bad,state=v,0,{k:x.detach().clone() for k,x in m.state_dict().items()}
        else:
            bad+=1
            if bad>=patience:break
    m.load_state_dict(state);m.eval()
    @torch.no_grad()
    def pred(k):
        x,_,y,b=d[k][:4];o=[]
        for i in range(0,len(x),batch):o.append(f(x[i:i+batch].to(DEV),b[i:i+batch].to(DEV)).cpu())
        return inv(torch.cat(o).numpy(),d['mu'],d['sd']),y.numpy()
    p,y=pred('test');pv,yv=pred('val_cal');off=cqr(pv,yv);p[...,0]-=off;p[...,-1]+=off;q90,q50,truth=np.maximum(p[:,:,0,-1],0),np.maximum(p[:,:,0,1],0),y[:,:,0]
    r=newsvendor(q90,truth,.9);r.update(model='GraphWaveNet-Q (LaDe Shanghai retrained)',seed=seed,epochs=ep+1,params=sum(x.numel() for x in m.parameters()),mae=float(np.abs(q50-truth).mean()),cqr_offset=off);return r

def history(d):
    tr=d['raw'][:d['tr_end']];truth=d['raw'][d['va_end']:];g=pd.DataFrame(tr).groupby(d['how'][:d['tr_end']]);med,q90=g.median(),g.quantile(.9); rows=[]
    for name,a in {'Historical median (LaDe train)':np.asarray([med.loc[t.dayofweek*24+t.hour] for t in d['test_times']]),'Historical q0.9 (LaDe train)':np.asarray([q90.loc[t.dayofweek*24+t.hour] for t in d['test_times']])}.items():
        r=newsvendor(a,truth,.9);r.update(model=name,seed=-1,epochs=0,params=0,mae=float(np.abs(a-truth).mean()),cqr_offset=0);rows.append(r)
    return rows

def saving(d):
    tr=d['raw'][:d['tr_end']];truth=d['raw'][d['va_end']:];g=pd.DataFrame(tr).groupby(d['how'][:d['tr_end']]);med,q90=g.median(),g.quantile(.9);a=np.asarray([med.loc[t.dayofweek*24+t.hour] for t in d['test_times']]);b=np.asarray([q90.loc[t.dayofweek*24+t.hour] for t in d['test_times']]);ca=(.1*np.maximum(a-truth,0)+.9*np.maximum(truth-a,0)).sum(1);cb=(.1*np.maximum(b-truth,0)+.9*np.maximum(truth-b,0)).sum(1);rng=np.random.default_rng(20260730);T=len(ca);vals=[]
    for _ in range(2000):
        ix=[]
        while len(ix)<T:s=int(rng.integers(0,T-24+1));ix.extend(range(s,s+24))
        ix=np.array(ix[:T]);vals.append(100*(1-cb[ix].sum()/ca[ix].sum()))
    return 100*(1-cb.sum()/ca.sum()),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--hidden',type=int,default=32);ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--patience',type=int,default=12);ap.add_argument('--batch',type=int,default=128);a=ap.parse_args();OUT.mkdir(exist_ok=True)
    table,cent,n=load_table();d=make_split(table);e,st=graph(cent);rows=history(d);save,lo,hi=saving(d)
    for seed in range(a.seeds):r=train_one(d,e,st,seed,a.hidden,a.epochs,a.patience,a.batch);rows.append(r);pd.DataFrame(rows).to_csv(OUT/'lade_shanghai_capacity_detail.csv',index=False);print(f'seed={seed} cost={r["cost"]:.4f} MAE={r["mae"]:.3f} ep={r["epochs"]}',flush=True)
    df=pd.DataFrame(rows);nn=df[df.seed>=0];cm,cl,ch=ci(nn.cost);mm,ml,mh=ci(nn.mae);hq=float(df.loc[df.model.eq('Historical q0.9 (LaDe train)'),'cost'].iloc[0]);summary=pd.DataFrame([{'model':'GraphWaveNet-Q (LaDe Shanghai retrained)','n_seeds':len(nn),'cost':cm,'cost_ci_low':cl,'cost_ci_high':ch,'mae':mm,'mae_ci_low':ml,'mae_ci_high':mh,'params':nn.params.mean(),'n_tasks':n,'n_regions':table.shape[1],'date_start':'2022-05-01','date_end':'2022-10-31','test_start':str(d['test_times'][0]),'test_end':str(d['test_times'][-1]),'historical_q90_vs_median_saving_pct':save,'historical_q90_vs_median_ci_low':lo,'historical_q90_vs_median_ci_high':hi,'gwnq_vs_historical_q90_saving_pct':100*(1-cm/hq)},*[r for r in rows if r['seed']<0]]);summary.to_csv(OUT/'lade_shanghai_capacity_summary.csv',index=False);print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
if __name__=='__main__':main()
