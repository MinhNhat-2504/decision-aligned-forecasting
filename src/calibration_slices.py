"""Coverage diagnostics by zone, hour of day, and realised demand regime."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, QUANTILES
from dataset import STData

def report(name,pq,y,times,zones):
    lo,hi=pq[:,:,0,0],pq[:,:,0,-1]
    inside=(y>=lo)&(y<=hi)
    rows=[]
    for z in range(y.shape[1]):
        rows.append(dict(model=name, slice_type='zone', slice_value=str(zones.iloc[z].zone_name),
                         n=len(y), coverage=100*inside[:,z].mean()))
    hour=pd.DatetimeIndex(times).hour
    for h in range(24):
        m=hour==h
        rows.append(dict(model=name,slice_type='hour',slice_value=str(h),n=int(m.sum()),coverage=100*inside[m].mean()))
    total=y.sum(1); cuts=np.quantile(total,[1/3,2/3])
    for label,m in [('low',total<=cuts[0]),('middle',(total>cuts[0])&(total<=cuts[1])),('high',total>cuts[1])]:
        rows.append(dict(model=name,slice_type='regime',slice_value=label,n=int(m.sum()),coverage=100*inside[m].mean()))
    return rows
def main():
    d=STData(); P=np.load(os.path.join(OUT_DIR,'test_preds.npz')); y=P['y_true'][:,:,0]
    times=d.ts[d.target_times('test')[:,0]]
    rows=[]
    for name in ('ST-MRGAT','GraphWaveNet-Q'):
        if name in P.files: rows += report(name,P[name],y,times,d.zones)
    out=pd.DataFrame(rows); out.to_csv(os.path.join(OUT_DIR,'calibration_slices.csv'),index=False)
    print(out.groupby(['model','slice_type']).coverage.agg(['mean','min','max']).round(2))
if __name__=='__main__': main()
