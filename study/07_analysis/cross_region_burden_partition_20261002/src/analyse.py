"""Observed annual service-gap accounting in fixed regional populations."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1];ROOT=P.parents[1]
V=ROOT/'10_manuscript/20261002_global_blackout_inequalities_v14/data'
for folder in ['data','metadata','figures','reports']: (P/folder).mkdir(exist_ok=True)
manifest=[]
def read(name):
    f=V/name;manifest.append(dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    return pd.read_csv(f)
a=read('duration_composition_inputs.csv')
b=read('brazil_2021_joint_context.csv')
mun=b.assign(unit=b.IdeConjUndConsumidoras.astype(int).astype(str)).set_index('unit').municipality_code
assert mun.index.is_unique
scopes={
 'Brazil':dict(total_units=814,period='2021',social='municipal GDP per person',population='four internal interruption classes in regulatory service areas',cluster='municipality'),
 'United States':dict(total_units=49,period='2013–2024',social='state household income',population='matched reported state-level customer indices',cluster='state'),
 'Europe':dict(total_units=11,period='2010–2018',social='GDP per person',population='systems with documented customer denominators; native paired years',cluster='system'),
 'New Zealand':dict(total_units=21,period='2023–2025',social='geographically allocated personal income',population='Class C interruption indices in geographical supply regions',cluster='region')}
seed=20261002;n_draws=4000
rng=np.random.default_rng(seed)
records=[];summary=[];draw_records=[];group_records=[];inputs=[];checks=[]
def check(name,condition):
    checks.append(dict(check=name,passed=bool(condition)));assert condition,name
def partition(fl,fh,hl,hh):
    dl=np.divide(hl,fl,out=np.full_like(hl,np.nan),where=fl>0)
    dh=np.divide(hh,fh,out=np.full_like(hh,np.nan),where=fh>0)
    cf=(fl-fh)*(dl+dh)/2;cd=(dl-dh)*(fl+fh)/2
    ft_l=fl.sum(axis=-1);ft_h=fh.sum(axis=-1);ht_l=hl.sum(axis=-1);ht_h=hh.sum(axis=-1)
    dt_l=ht_l/ft_l;dt_h=ht_h/ft_h
    return dict(gap=hl-hh,frequency=cf,duration=cd,total_gap=ht_l-ht_h,
        aggregate_frequency=(ft_l-ft_h)*(dt_l+dt_h)/2,
        aggregate_duration=(dt_l-dt_h)*(ft_l+ft_h)/2,
        frequency_ratio=ft_l/ft_h,duration_ratio=dt_l/dt_h,
        low_hours=ht_l,high_hours=ht_h,low_frequency=ft_l,high_frequency=ft_h,
        low_duration=dt_l,high_duration=dt_h)

for region,meta in scopes.items():
    z=a[a.region.eq(region)].copy();check(region+' unique unit/class',not z.duplicated(['unit','category']).any())
    z['cluster']=z.unit.map(mun).astype('Int64').astype(str) if region=='Brazil' else z.unit
    check(region+' complete social clustering',not z.cluster.isin(['<NA>','nan']).any())
    check(region+' clusters in one economic group',z.groupby('cluster').quartile.nunique().eq(1).all())
    check(region+' no negative observations',z[['frequency','burden_h']].ge(0).all().all())
    categories=sorted(z.category.unique());k=len(categories);arrs=[];boots=[];info={}
    for q in ['Q1','Q4']:
        w=z[z.quartile.eq(q)].pivot(index=['unit','cluster'],columns='category',values=['frequency','burden_h'])
        w=w.reindex(columns=pd.MultiIndex.from_product([['frequency','burden_h'],categories]))
        check(region+' '+q+' complete class partition',w.notna().all().all())
        vals=w.to_numpy();arrs.append(vals.mean(axis=0))
        cls=w.index.get_level_values('cluster').to_numpy();unique=np.unique(cls)
        # Resample complete geographic clusters; every area and class in a selected
        # municipality travels together. The point estimator remains equal-area.
        sums=np.stack([vals[cls==c].sum(axis=0) for c in unique]);counts=np.array([(cls==c).sum() for c in unique])
        idx=rng.integers(0,len(unique),(n_draws,len(unique)))
        boots.append(sums[idx].sum(axis=1)/counts[idx].sum(axis=1)[:,None])
        info['n_'+q]=len(w);info['clusters_'+q]=len(unique)
        for j,cat in enumerate(categories):
            group_records.append(dict(region=region,group=q,category=cat,n_units=len(w),n_clusters=len(unique),frequency=arrs[-1][j],burden_h=arrs[-1][k+j],duration_h=arrs[-1][k+j]/arrs[-1][j]))
    low,high=arrs;lb,hb=boots
    point=partition(low[:k],high[:k],low[k:],high[k:]);draw=partition(lb[:,:k],hb[:,:k],lb[:,k:],hb[:,k:])
    check(region+' category gap exact',np.allclose(point['frequency']+point['duration'],point['gap'],rtol=1e-11,atol=1e-10))
    check(region+' category sums exact',np.isclose(point['gap'].sum(),point['total_gap']))
    check(region+' aggregate gap exact',np.isclose(point['aggregate_frequency']+point['aggregate_duration'],point['total_gap']))
    finite=np.isfinite(draw['frequency']).all(axis=1)&np.isfinite(draw['duration']).all(axis=1)
    check(region+' finite draw partitions close',np.allclose((draw['frequency']+draw['duration'])[finite],draw['gap'][finite],rtol=1e-10,atol=1e-10))
    for j,cat in enumerate(categories):
        for component in ['gap','frequency','duration']:
            lo,hi=np.nanquantile(draw[component][:,j],[.025,.975])
            records.append(dict(region=region,category=cat,component=component,value_h=point[component][j],lower_h=lo,upper_h=hi,defined_draws=int(np.isfinite(draw[component][:,j]).sum()),**info,**meta))
    row=dict(region=region,**info,**meta,undefined_class_partition_draws=int((~finite).sum()))
    for component in ['total_gap','aggregate_frequency','aggregate_duration','frequency_ratio','duration_ratio','low_hours','high_hours','low_frequency','high_frequency','low_duration','high_duration']:
        row[component]=float(point[component]);row[component+'_lower'],row[component+'_upper']=map(float,np.quantile(draw[component],[.025,.975]))
    row['aggregate_frequency_share_pct']=100*point['aggregate_frequency']/point['total_gap']
    # Shares of a net gap are not emphasized when its bootstrap direction is unresolved.
    row['gap_direction_resolved']=row['total_gap_lower']>0 or row['total_gap_upper']<0
    row['class_frequency_sum_h']=float(point['frequency'].sum());row['class_duration_sum_h']=float(point['duration'].sum())
    row['leading_gap_class']=categories[int(np.argmax(point['gap']))]
    row['leading_gap_class_share_pct']=100*point['gap'].max()/point['total_gap']
    summary.append(row)
    for j in range(n_draws):
        draw_records.append(dict(region=region,draw=j,total_gap=draw['total_gap'][j],aggregate_frequency=draw['aggregate_frequency'][j],aggregate_duration=draw['aggregate_duration'][j],class_frequency_sum=draw['frequency'][j].sum(),class_duration_sum=draw['duration'][j].sum()))
    inputs.append(z.assign(period=meta['period'],social=meta['social'],population=meta['population']))

pd.DataFrame(records).to_csv(P/'data/class_gap_components.csv',index=False)
pd.DataFrame(summary).to_csv(P/'data/regional_gap_summary.csv',index=False)
pd.DataFrame(group_records).to_csv(P/'data/class_group_means.csv',index=False)
pd.DataFrame(draw_records).to_parquet(P/'data/bootstrap_draws.parquet',index=False)
pd.concat(inputs,ignore_index=True).to_csv(P/'data/frozen_accounting_inputs.csv',index=False)
(P/'metadata/input_manifest.json').write_text(json.dumps(manifest,indent=2))
(P/'metadata/numerical_validation.json').write_text(json.dumps(dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=seed,draws=n_draws,checks=checks,all_passed=all(x['passed'] for x in checks)),indent=2))
print(pd.DataFrame(summary)[['region','n_Q1','n_Q4','total_gap','total_gap_lower','total_gap_upper','aggregate_frequency_share_pct','leading_gap_class','leading_gap_class_share_pct','undefined_class_partition_draws']].to_string(index=False))
print(pd.DataFrame(records)[['region','category','component','value_h','lower_h','upper_h']].round(3).to_string(index=False))
