"""Within-system cause-duration ratios, including the independent NZ test."""
from pathlib import Path
import pandas as pd,numpy as np,json
from scipy.stats import t
P=Path(__file__).resolve().parents[1];D=P/'data'
def save(z,stem):
    z.to_parquet(D/(stem+'.parquet'),index=False)
    z.to_csv(P/'source_data'/(stem+'.csv'),index=False)
parts=[]
ny=pd.read_parquet(D/'ny_regional_category_year.parquet')
ny=ny[ny.category.isin(['Recorded weather','Operating error'])].rename(columns={'region':'system'})
v=ny.groupby(['system','category'])[['customer_interruptions','customer_hours']].sum().reset_index()
v['duration_h']=v.customer_hours/v.customer_interruptions;v['source']='New York';v['cause']=v.category.map({'Recorded weather':'Weather','Operating error':'Error'})
parts.append(v[['source','system','cause','duration_h','customer_interruptions','customer_hours']].rename(columns={'customer_interruptions':'frequency_sum','customer_hours':'burden_sum'}))
nz=pd.read_parquet(D/'ny_nz_narrow_human_error_year.parquet')
v=nz.groupby(['system','category'])[['frequency','burden_h']].mean().reset_index()
v['duration_h']=v.burden_h/v.frequency;v['source']='New Zealand';v['cause']=v.category.map({'Recorded weather':'Weather','Operating error':'Error'})
parts.append(v[['source','system','cause','duration_h','frequency','burden_h']].rename(columns={'frequency':'frequency_sum','burden_h':'burden_sum'}))
common=pd.read_parquet(D/'ny_common_named_cause_year.parquet')
for source in ['New Zealand','New York']:
    v=common[(common.source==source)&(common.category=='Lightning')].copy()
    if source=='New York':
        den=pd.read_parquet(D/'ny_regional_customer_denominators.parquet').rename(columns={'region':'system'})
        v=v.merge(den[['system','year','customers_served']],on=['system','year'],validate='many_to_one')
        v['frequency']=v.frequency*v.customers_served;v['burden_h']=v.burden_h*v.customers_served
        v=v.groupby('system')[['frequency','burden_h']].sum().reset_index()
    else:v=v.groupby('system')[['frequency','burden_h']].mean().reset_index()
    v['duration_h']=v.burden_h/v.frequency;v['source']=source;v['cause']='Lightning'
    parts.append(v[['source','system','cause','duration_h','frequency','burden_h']].rename(columns={'frequency':'frequency_sum','burden_h':'burden_sum'}))
a=pd.concat(parts,ignore_index=True);a['positive_support']=a.frequency_sum.gt(0)&a.burden_sum.gt(0)&np.isfinite(a.duration_h)
save(a,'cause_persistence_context');ratios=[];summary=[]
for source in ['New Zealand','New York']:
    v=a[a.source.eq(source)].pivot(index='system',columns='cause',values='duration_h')
    for cause in ['Weather','Lightning']:
        valid=v[[cause,'Error']].replace([np.inf,-np.inf],np.nan).dropna();valid=valid[(valid>0).all(axis=1)]
        r=valid[cause]/valid.Error;lr=np.log(r);mean=lr.mean();se=lr.std(ddof=1)/np.sqrt(len(lr));hw=t.ppf(.975,len(lr)-1)*se
        summary.append(dict(source=source,cause=cause,comparator='Error',systems=len(lr),all_systems=len(v),
            geometric_ratio=np.exp(mean),ci_low=np.exp(mean-hw),ci_high=np.exp(mean+hw),number_above_one=int((r>1).sum()),p_value=2*t.sf(abs(mean/se),len(lr)-1)))
        ratios.extend(dict(source=source,cause=cause,system=k,ratio=x) for k,x in r.items())
save(pd.DataFrame(ratios),'cause_persistence_unit_ratios');save(pd.DataFrame(summary),'cause_persistence_summary')
print(pd.DataFrame(summary).to_string(index=False))
