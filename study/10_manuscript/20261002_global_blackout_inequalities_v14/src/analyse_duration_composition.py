"""Exact symmetric composition decomposition of observed economic duration gaps.

No model or significance selection. Same admitted populations as v07; economic
quartiles are fixed before resampling. Category definitions remain source-native.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]; R=P.parents[1]
V=P; D=P/'data'
summaries=[]; details=[]; inputs=[]

def decompose(region, a, categories, unit, income, quartile=None, scope=''):
    z=a.copy()
    if quartile is None:
        z['quartile']=pd.qcut(z[income].rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
    else: z['quartile']=z[quartile].astype(str)
    fs=[];hs=[]
    for q in ['Q1','Q4']:
        s=z[z.quartile.eq(q)]
        fs.append(np.array([s[fc].mean() for _,fc,_ in categories]))
        hs.append(np.array([s[hc].mean() for _,_,hc in categories]))
        for cat,fc,hc in categories:
            for _,row in s.iterrows():
                inputs.append(dict(region=region,unit=str(row[unit]),quartile=q,category=cat,
                    frequency=float(row[fc]),burden_h=float(row[hc]),income=float(row[income])))
    fl,fh=fs;hl,hh=hs
    assert np.all(fl>0)&np.all(fh>0),region
    sl,sh=fl/fl.sum(),fh/fh.sum();dl,dh=hl/fl,hh/fh
    within=(sl+sh)/2*(dl-dh);mix=(dl+dh)/2*(sl-sh)
    low,high=hl.sum()/fl.sum(),hh.sum()/fh.sum();gap=low-high
    assert np.isclose(within.sum()+mix.sum(),gap,atol=1e-10),region
    summaries.append(dict(region=region,n=len(z),q1_n=int(z.quartile.eq('Q1').sum()),q4_n=int(z.quartile.eq('Q4').sum()),
        scope=scope,duration_low_h=low,duration_high_h=high,gap_low_minus_high_h=gap,
        within_category_h=within.sum(),composition_h=mix.sum(),
        within_pct_of_high=100*within.sum()/high,composition_pct_of_high=100*mix.sum()/high,
        gap_pct_of_high=100*gap/high,identity_error_h=within.sum()+mix.sum()-gap))
    for j,(cat,_,_) in enumerate(categories):
        details.append(dict(region=region,category=cat,frequency_low=fl[j],frequency_high=fh[j],
            burden_low_h=hl[j],burden_high_h=hh[j],duration_low_h=dl[j],duration_high_h=dh[j],
            frequency_share_low=sl[j],frequency_share_high=sh[j],within_category_h=within[j],composition_h=mix[j]))

c=pd.read_parquet(D/'china_common_counties.parquet')
for k in ['planned','unplanned']:
    c[k+'_F']=c[k+'_count']/c['pop'];c[k+'_H']=c[k+'_hours']/c['pop']
decompose('China',c,[(k.title(),k+'_F',k+'_H') for k in ['planned','unplanned']],
    'County','income','income_quartile','All admitted notices; equal county rates')

b=pd.read_parquet(D/'brazil_2021_joint_context.parquet')
decompose('Brazil',b,[(x,'FEC'+s,'DEC'+s) for x,s in [('Ordinary unplanned','IND'),('Planned','IP'),('Emergency','INE'),('Critical day','INC')]],
    'IdeConjUndConsumidoras','gdp_pc_brl','income_quartile','Four mutually exclusive internal interruption classes; equal areas')

for reg,fn in [('United States','us_same49_state_means'),('Europe','europe_customer11_system_means')]:
    z=pd.read_parquet(D/(fn+'.parquet'))
    decompose(reg,z,[(label,'F_'+s,'H_'+s) for label,s in [('Routine / non-major','routine'),('Exceptional / major','exceptional')]],
        'state' if reg=='United States' else 'ISO3','income','quartile' if 'quartile' in z else None,'Paired annual-index components; equal reporting systems')

n=pd.read_parquet(D/'nz_explicit21_total_context.parquet')
parts=pd.read_parquet(D/'nz_explicit21_complete_partition.parquet')
cats=[]
for i,cat in enumerate(['Weather','Human activity','Remaining causes']):
    q=parts[parts.category.eq(cat)][['system','frequency','burden_h']].rename(columns={'frequency':f'F{i}','burden_h':f'H{i}'})
    n=n.merge(q,on='system',validate='one_to_one');cats.append((cat,f'F{i}',f'H{i}'))
assert np.allclose(n.frequency,n[[c[1] for c in cats]].sum(axis=1))
assert np.allclose(n.burden_h,n[[c[2] for c in cats]].sum(axis=1))
decompose('New Zealand',n,cats,'system','income_grouped_median_nzd',None,'All Class C causes, including residual; equal networks')

for name,rows in [('duration_composition_summary',summaries),('duration_composition_by_category',details),('duration_composition_inputs',inputs)]:
    df=pd.DataFrame(rows)
    df.to_csv(D/f'{name}.csv',index=False);df.to_parquet(D/f'{name}.parquet',index=False)
    df.to_csv(V/'source_data'/f'{name}.csv',index=False);df.to_parquet(V/'data'/f'{name}.parquet',index=False)
(P/'metadata/duration_composition_method.json').write_text(json.dumps(dict(
    contrast='Lowest economic quartile minus highest; fixed source populations',
    identity='D_low-D_high = sum[(s_low+s_high)/2*(D_low-D_high)] + sum[(D_low+D_high)/2*(s_low-s_high)]',
    interpretation='Exact observed accounting, not causal restoration-efficiency mediation',
    inference='Descriptive; no resampling or cross-region pooled effect claimed',
    additional_scope='Brazil four internal classes; other regions full admitted class partition'),indent=2))
print(pd.DataFrame(summaries).to_string(index=False))
