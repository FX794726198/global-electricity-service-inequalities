"""Reproduce exposure-adjusted contrasts from frozen, source-audited exposures."""
from pathlib import Path
import pandas as pd
import analyse_process_contrasts as cp
P=Path(__file__).resolve().parents[1]
z=pd.read_parquet(P/'data/nz_network_exposure_context.parquet').set_index('system')
a=pd.read_parquet(P/'data/nz_explicit21_cause_context.parquet')
v=a.pivot(index='system',columns='category',values=['frequency','burden_h']).loc[z.index]
ys={'Fw':v['frequency','Weather'],'Hw':v['burden_h','Weather'],'Fh':v['frequency','Human activity'],'Hh':v['burden_h','Human activity']}
defs={'Weather minus human frequency':{'Fw':1,'Fh':-1},'Weather minus human duration':{'Hw':1,'Fw':-1,'Hh':-1,'Fh':1},'Weather frequency':{'Fw':1},'Weather duration':{'Hw':1,'Fw':-1},'Human frequency':{'Fh':1},'Human duration':{'Hh':1,'Fh':-1}}
delta=z.log_income.quantile(.75)-z.log_income.quantile(.25)
for spec,extra in [('Density adjusted',[]),('Density + gust exposure',['log_gust_days']),('Density + gust + rain exposure',['log_gust_days','log_rain_days'])]:
    X=pd.DataFrame({'const':1.,'log_income':z.log_income,'log_density':z.log_density},index=z.index)
    tests={'log_income':defs};ds={'log_income':delta}
    for k in extra:
        X[k]=z[k];tests[k]=defs;ds[k]=z[k].quantile(.75)-z[k].quantile(.25)
    cp.contrasts(z,X,ys,tests,dict(region='New Zealand',specification=spec),ds)
r=pd.DataFrame(cp.out)
r.to_parquet(P/'data/nz_exposure_adjusted_direct_contrasts.parquet',index=False)
r.to_csv(P/'source_data/nz_exposure_adjusted_direct_contrasts.csv',index=False)
print(r.query("term=='log_income' and contrast=='Weather minus human frequency'").to_string(index=False))
