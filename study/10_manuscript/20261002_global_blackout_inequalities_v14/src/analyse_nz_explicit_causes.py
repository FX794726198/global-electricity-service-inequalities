from analyse_zero_inclusive import *
import analyse_process_contrasts as cp
from pathlib import Path
P=Path(__file__).resolve().parents[1];D=P/'data';R=P.parents[1]
a=pd.read_parquet(R/'07_analysis/non_europe_cause_loss_extension_20260929/data/nzl_cause_operator_year.parquet')
old=pd.read_parquet(R/'07_analysis/inequality_process_linkage_20261001/data/nz_same25_all_context.parquet').set_index('system')
target=['Adverse weather','Lightning','Human error','Third party interference']
x=a[a.system.isin(old.index)&a.year.between(2023,2025)&a.cause.isin(target)].copy();n=x.groupby('system').size();keep=n[n.eq(12)].index
x=x[x.system.isin(keep)];assert x.paired.all() and len(x)==21*3*4
x['category']=x.cause.map(dict(zip(target,['Weather','Weather','Human activity','Human activity'])))
y=x.groupby(['system','year','category']).agg(frequency=('SAIFI','sum'),burden_h=('SAIDI',lambda v:v.sum()/60)).reset_index()
context=old.loc[keep].drop(columns=['frequency','burden_h','category'])
c=y.groupby(['system','category'])[['frequency','burden_h']].mean().reset_index().merge(context,left_on='system',right_index=True,validate='many_to_one')
di=float(context.log_income.quantile(.75)-context.log_income.quantile(.25))
save(c,'nz_explicit21_cause_context');save(y,'nz_explicit21_cause_year')
total=pd.read_parquet(R/'07_analysis/non_europe_cause_loss_extension_20260929/data/nzl_class_c_total_checks.parquet')
total=total[total.system.isin(keep)&total.year.between(2023,2025)];assert len(total)==63
v=total.groupby('system').agg(frequency=('SAIFI_total','mean'),burden_h=('SAIDI_total',lambda v:v.mean()/60)).join(context)
save(v.reset_index(),'nz_explicit21_total_context')
sums=c.groupby('system')[['frequency','burden_h']].sum();res=v[['frequency','burden_h']]-sums;assert (res>=-1e-8).all().all();res=res.reset_index();res['category']='Remaining causes';full=pd.concat([c[['system','category','frequency','burden_h']],res]);save(full,'nz_explicit21_complete_partition')
g=gpd.read_parquet(R/'07_analysis/inequality_process_linkage_20261001/data/nz_same25_map.parquet');g=g[g.system.isin(keep)];save(g,'nz_explicit21_map')
for category,z in c.groupby('category'):
 for spec in ['Economic context','Density adjusted']:
  X=pd.DataFrame({'const':1.,'log_income':z.log_income},index=z.index)
  if spec=='Density adjusted':X['log_density']=z.log_density
  paired(z,X,z.frequency,z.burden_h,dict(region='New Zealand',category=category,specification=spec),{'log_income':di})
save(pd.DataFrame(rows),'nz_explicit21_models')
pv=c.pivot(index='system',columns='category',values=['frequency','burden_h']).loc[context.index]
ys={'Fw':pv['frequency','Weather'],'Hw':pv['burden_h','Weather'],'Fh':pv['frequency','Human activity'],'Hh':pv['burden_h','Human activity']}
defs={'Weather minus human frequency':{'Fw':1,'Fh':-1},'Weather minus human duration':{'Hw':1,'Fw':-1,'Hh':-1,'Fh':1},'Weather frequency minus duration':{'Fw':2,'Hw':-1},'Weather frequency':{'Fw':1},'Weather duration':{'Hw':1,'Fw':-1},'Human frequency':{'Fh':1},'Human duration':{'Hh':1,'Fh':-1}}
for spec in ['Economic context','Density adjusted']:
 X=pd.DataFrame({'const':1.,'log_income':context.log_income},index=context.index)
 if spec=='Density adjusted':X['log_density']=context.log_density
 cp.contrasts(context,X,ys,{'log_income':defs},dict(region='New Zealand',specification=spec),{'log_income':di})
save(pd.DataFrame(cp.out),'nz_explicit21_direct_contrasts')
M.joinpath('nz_explicit_cause_admission.json').write_text(json.dumps(dict(primary_operators=21,operator_years=63,exact_cause_rows=252,missing_cause_exclusions=old.index.difference(keep).tolist(),earlier_25_operator_result='Superseded for primary cause comparisons: source totals reconciling does not establish explicitly observed zeros for omitted individual causes.',remaining='Published Class C total minus four explicitly listed causes; not reclassified natural or anthropogenic.',income_p25=float(np.exp(context.log_income.quantile(.25))),income_p75=float(np.exp(context.log_income.quantile(.75)))),indent=2))
print(pd.DataFrame(cp.out)[['specification','contrast','contrast_pct','contrast_lo_pct','contrast_hi_pct','p']].to_string(index=False))
