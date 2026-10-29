"""Reproduce the NY comparison and NZ structure diagnostic from frozen inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import analyse_process_contrasts as cp
P=Path(__file__).resolve().parents[1]
D=P/'data'
def save(a,stem):
    a.to_parquet(D/(stem+'.parquet'),index=False)
    a.to_csv(P/'source_data'/(stem+'.csv'),index=False)
defs={'Weather minus operating-error frequency':{'Fw':1,'Fh':-1},
      'Weather minus operating-error duration':{'Hw':1,'Fw':-1,'Hh':-1,'Fh':1},
      'Weather frequency':{'Fw':1},'Weather duration':{'Hw':1,'Fw':-1},
      'Operating-error frequency':{'Fh':1},'Operating-error duration':{'Hh':1,'Fh':-1}}
for region,stem,context,annual,index in [
    ('National Grid New York','ny_regional_income_contrasts','ny_regional_income_context','ny_regional_category_year','region'),
    ('New Zealand narrow human error','ny_nz_narrow_human_error_contrasts','nz_explicit21_cause_context','ny_nz_narrow_human_error_year','system')]:
    c=pd.read_parquet(D/(context+'.parquet'))
    if 'Region' in c: c=c.rename(columns={'Region':'region'})
    c=c.drop_duplicates(index).set_index(index)
    a=pd.read_parquet(D/(annual+'.parquet'))
    v=a.groupby([index,'category'])[['frequency','burden_h']].mean().unstack().loc[c.index]
    ys={'Fw':v['frequency','Recorded weather'],'Hw':v['burden_h','Recorded weather'],
        'Fh':v['frequency','Operating error'],'Hh':v['burden_h','Operating error']}
    cp.out.clear()
    for spec in ['Economic context','Density adjusted']:
        X=pd.DataFrame({'const':1.,'log_income':c.log_income},index=c.index)
        if spec=='Density adjusted':X['log_density']=c.log_density
        cp.contrasts(c,X,ys,{'log_income':defs},dict(region=region,specification=spec),
                     {'log_income':c.log_income.quantile(.75)-c.log_income.quantile(.25)})
    save(pd.DataFrame(cp.out),stem)

z=pd.read_parquet(D/'nz_structure_structure_context.parquet').set_index('system')
v=pd.read_parquet(D/'nz_explicit21_cause_context.parquet').pivot(index='system',columns='category',values=['frequency','burden_h'])
sd={k.replace('operating-error','human').replace('Operating-error','Human'):val for k,val in defs.items()}
cp.out.clear()
for spec,extra in [('Income, density and gust',[]),('Plus preceding underground share',['underground_share_10pp']),
                  ('Secondary full structure',['underground_share_10pp','vegetation_share_10pp','remote_rugged_share_10pp'])]:
    cols=['log_income','log_density','log_gust_days']+extra;c=z.dropna(subset=cols).copy()
    X=pd.DataFrame({'const':1.},index=c.index)
    for col in cols:X[col]=c[col]
    ys={'Fw':v.loc[c.index,('frequency','Weather')],'Hw':v.loc[c.index,('burden_h','Weather')],
        'Fh':v.loc[c.index,('frequency','Human activity')],'Hh':v.loc[c.index,('burden_h','Human activity')]}
    ds={key:1. for key in ['log_income']+extra};ds['log_income']=z.log_income.quantile(.75)-z.log_income.quantile(.25)
    cp.contrasts(c,X,ys,{key:sd for key in ds},dict(region='New Zealand',specification=spec),ds)
save(pd.DataFrame(cp.out),'nz_structure_structure_direct_contrasts')

a=pd.read_parquet(D/'ny_regional_category_year.parquet')
den=pd.read_parquet(D/'ny_regional_customer_denominators.parquet')
den=den[den.year.between(2021,2024)].groupby('region').customers_served.sum()
z=a.groupby(['region','category'])[['interruption_events','customer_interruptions','customer_hours']].sum().reset_index()
z['customer_years']=z.region.map(den);z['event_rate']=1000*z.interruption_events/z.customer_years
z['scope']=z.customer_interruptions/z.interruption_events;z['duration']=z.customer_hours/z.customer_interruptions
z['burden']=z.customer_hours/z.customer_years
valid=z.interruption_events.gt(0)&z.customer_interruptions.gt(0)
assert np.allclose((z.event_rate*z.scope*z.duration/1000)[valid],z.burden[valid])
save(z,'ny_observed_process_stages')
v=z.pivot(index='region',columns='category',values=['event_rate','scope','duration','burden']);rows=[];ratios=[]
for key in ['event_rate','scope','duration','burden']:
    r=v[key,'Recorded weather']/v[key,'Operating error'];lr=np.log(r);mean=lr.mean();se=lr.std(ddof=1)/np.sqrt(len(lr));ci=t.ppf(.975,len(lr)-1)*se
    rows.append(dict(metric=key,regions=8,geometric_ratio=np.exp(mean),ci_low=np.exp(mean-ci),ci_high=np.exp(mean+ci),number_weather_larger=int((r>1).sum())))
    ratios.extend(dict(region=reg,metric=key,ratio=val) for reg,val in r.items())
for name,frame in [('ny_process_stage_summary',pd.DataFrame(rows)),('ny_process_stage_region_ratios',pd.DataFrame(ratios))]:
    save(frame,name);frame.to_csv(D/(name+'.csv'),index=False)
print('Reproduced independent cause comparison, narrower cause definition, measured structure diagnostic and observed stage ratios.')
