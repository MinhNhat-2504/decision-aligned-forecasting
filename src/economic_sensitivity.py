"""Economic and critical-ratio sensitivity of capacity decisions."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, QUANTILES

TAUS = [.60, .70, .80, .90, .95]

def interp_quantile(pq, tau):
    q = np.asarray(QUANTILES)
    if tau <= q[0]: return pq[..., 0]
    if tau >= q[-1]: return pq[..., -1]
    hi = int(np.searchsorted(q, tau)); lo = hi - 1
    w = (tau-q[lo])/(q[hi]-q[lo])
    return (1-w)*pq[..., lo] + w*pq[..., hi]

def main():
    P = np.load(os.path.join(OUT_DIR, 'test_preds.npz'))
    y = P['y_true'][:, :, 0]
    rows=[]
    for tau in TAUS:
        cu, co = tau, 1-tau
        for model in ('ST-MRGAT','GraphWaveNet-Q'):
            if model not in P.files: continue
            a = interp_quantile(P[model][:,:,0,:], tau)
            short, over = np.maximum(y-a,0), np.maximum(a-y,0)
            raw_cost=(cu*short+co*over).mean()
            rows.append(dict(model=model,tau=tau,cost_units=raw_cost,
                             cost_per_1000_demand=1000*raw_cost/(y.mean()+1e-9),
                             fill_rate=100*(1-short.sum()/(y.sum()+1e-9)),
                             capacity_ratio=float(a.sum()/(y.sum()+1e-9))))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,'economic_sensitivity.csv'),index=False)
    print(pd.DataFrame(rows).to_string(index=False,float_format=lambda v:f'{v:.3f}'))
if __name__=='__main__': main()
