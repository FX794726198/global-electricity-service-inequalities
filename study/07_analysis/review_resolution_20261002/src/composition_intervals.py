"""Keep original contrasts; put regional-resampling uncertainty on the main figure."""
from pathlib import Path
import numpy as np,pandas as pd,json
P=Path(__file__).resolve().parents[1];R=P.parents[1]
rng=np.random.default_rng(20261002);rows=[];audits=[]
b=pd.read_csv(R/'07_analysis/brazil_detailed_cause_extension_20261002/data/duration_gap_decomposition.csv').set_index('component')
for label,key in [('Within class','Within-cause duration'),('Composition','Cause composition'),('Net','Total duration gap')]:
    z=b.loc[key];rows.append(dict(region='Brazil',component=label,value=z.low_minus_high_h,lower=z.lower_h,upper=z.upper_h,unit='record-hours per record',interval='Existing municipality bootstrap'))
data=pd.read_csv(R/'10_manuscript/20261002_global_blackout_inequalities_v14/data/duration_composition_inputs.csv')
reference=pd.read_csv(R/'07_analysis/brazil_detailed_cause_extension_20261002/source_data/Fig05_duration_decomposition.csv').set_index('region')

def calc(l,h):
    lf,lh=l.mean(0);hf,hh=h.mean(0)
    if np.any(lf<=0) or np.any(hf<=0):return None
    ls,hs=lf/lf.sum(),hf/hf.sum();ld,hd=lh/lf,hh/hf
    within=np.sum((ls+hs)*(ld-hd)/2);mix=np.sum((ls-hs)*(ld+hd)/2)
    gap=lh.sum()/lf.sum()-hh.sum()/hf.sum()
    assert np.isclose(within+mix,gap,atol=1e-10)
    return np.array([within,mix,gap])
for region in ['United States','New Zealand']:
    z=data[data.region.eq(region)];arrays=[]
    for q in ['Q1','Q4']:
        a=z[z.quartile.eq(q)].pivot(index='unit',columns='category',values=['frequency','burden_h'])
        arrays.append(np.stack([a.frequency.to_numpy(),a.burden_h.to_numpy()],axis=1))
    l,h=arrays;point=calc(l,h)
    ref=reference.loc[region]
    assert np.allclose(point,[ref.within_category_h,ref.composition_h,ref.gap_low_minus_high_h])
    draws=[]
    for _ in range(4000):
        out=calc(l[rng.integers(0,len(l),len(l))],h[rng.integers(0,len(h),len(h))])
        if out is not None:draws.append(out)
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    for k,label in enumerate(['Within class','Composition','Net']):
        rows.append(dict(region=region,component=label,value=point[k],lower=lo[k],upper=hi[k],unit='hours per customer interruption',interval='Fixed-quartile whole-region percentile bootstrap'))
    audits.append(dict(region=region,q1_n=len(l),q4_n=len(h),requested=4000,admitted=len(draws),excluded_zero_cause_draws=4000-len(draws)))
t=pd.read_csv(R/'07_analysis/cause_conditioned_service_inequality_20261002/data/overall_duration_accounting.csv').query("voltage=='All'").iloc[0]
for label,col in [('Within class','within_cause_h'),('Composition','composition_h'),('Net','duration_gap_h')]:
    rows.append(dict(region='Türkiye',component=label,value=t[col],lower=np.nan,upper=np.nan,unit='hours per customer interruption',interval='Observed annual administrative accounting; no sampling interval'))
pd.DataFrame(rows).to_csv(P/'data/regional_composition_intervals.csv',index=False)
(P/'metadata/composition_intervals.json').write_text(json.dumps(dict(seed=20261002,population='Original source-specific fixed economic quartiles',audits=audits,interpretation='Resampling variability across recorded regions; no causal or global pooled mechanism'),indent=2))
print(pd.DataFrame(rows).round(3).to_string(index=False))
