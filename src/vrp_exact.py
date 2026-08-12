"""Exact small-candidate capacitated multi-vehicle routing benchmark.

For each selected hour, Held--Karp computes the shortest closed tour for every
subset of ten forecast-selected zones. Dynamic programming then partitions these
zones over three capacity- and route-length-feasible vehicles. This is exact for
the stated ten-candidate proxy, unlike the larger greedy VRP extension.
"""
import os, sys
from functools import lru_cache
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROC_DIR, OUT_DIR, GRAPH_FILE, QUANTILES
from dataset import STData

K,V,MAX_KM,CAP_FRAC=10,3,22.0,.42

def tsp_lengths(D,depot,nodes):
    n=len(nodes); full=1<<n
    dp={(1<<j,j):D[depot,nodes[j]] for j in range(n)}
    out=np.zeros(full)
    for mask in range(1,full):
        for j in range(n):
            if not(mask>>j&1): continue
            prev=mask^(1<<j)
            if prev:
                dp[(mask,j)]=min(dp[(prev,k)]+D[nodes[k],nodes[j]] for k in range(n) if prev>>k&1)
        if mask:
            out[mask]=min(dp[(mask,j)]+D[nodes[j],depot] for j in range(n) if mask>>j&1)
    return out
def solve(D,depot,nodes,demand):
    n=len(nodes); L=tsp_lengths(D,depot,nodes); cap=CAP_FRAC*demand.sum(); full=(1<<n)-1
    feasible=[]
    for m in range(1,1<<n):
        load=sum(demand[nodes[j]] for j in range(n) if m>>j&1)
        if load<=cap and L[m]<=MAX_KM: feasible.append(m)
    @lru_cache(None)
    def f(mask,v):
        if mask==0:return (0.,0.)
        # Unserved candidates are allowed; this is a prize-collecting capacitated
        # routing problem, not an infeasible requirement to serve every zone.
        if v==0:return (0.,0.)
        best=(0.,0.)
        for s in feasible:
            if s&mask!=s:continue
            val,dist=f(mask^s,v-1)
            gain=sum(demand[nodes[j]] for j in range(n) if s>>j&1)
            cand=(gain+val,dist+L[s])
            if cand[0]>best[0] or (cand[0]==best[0] and cand[1]<best[1]):best=cand
        return best
    return f(full,V)
def main():
    d=STData(); g=np.load(os.path.join(PROC_DIR,GRAPH_FILE));D=g['dist_road'].astype(float);bad=D>1e5;D[bad]=g['dist_geo'].astype(float)[bad]*2;np.fill_diagonal(D,0)
    P=np.load(os.path.join(OUT_DIR,'test_preds.npz'));y=P['y_true'][:,:,0];depot=int(np.argmax(d.raw[:d.bounds[1]].sum(0)));
    # evenly spaced 24 hours avoids cherry-picking
    hrs=np.linspace(0,len(y)-1,24,dtype=int); rows=[]
    policies={'HistoricalAvg':np.stack([d.ha_raw[d.how[s+12]] for s in d.idx['test']]),'Oracle':y}
    for name in ('ST-MRGAT','GraphWaveNet-Q'):
        if name in P.files: policies[name]=P[name][:,:,0,len(QUANTILES)//2]
    for name,sc in policies.items():
        cov=[];dist=[]
        for t in hrs:
            nodes=np.argsort(-sc[t])[:K].tolist(); gain,L=solve(D,depot,nodes,y[t]);cov.append(100*gain/(y[t].sum()+1e-9));dist.append(L)
        rows.append(dict(policy=name,hours=len(hrs),candidates=K,vehicles=V,coverage=float(np.mean(cov)),distance_km=float(np.mean(dist))))
    out=pd.DataFrame(rows);out.to_csv(os.path.join(OUT_DIR,'vrp_exact.csv'),index=False);print(out.to_string(index=False,float_format=lambda v:f'{v:.3f}'))
if __name__=='__main__':main()
