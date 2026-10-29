"""Reconstruct new point results from prior native class tables, not the new analyzer."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];R=P.parents[1];V=P/'inputs'
reported=pd.read_csv(P/'data/class_gap_components.csv');summ=pd.read_csv(P/'data/regional_gap_summary.csv').set_index('region');checks=[]
def ck(label,actual,expected):
 ok=bool(np.allclose(actual,expected,rtol=1e-10,atol=1e-9));checks.append(dict(check=label,passed=ok));assert ok,label
frames=[];group_notes={}
b=pd.read_parquet(V/'brazil_2021_joint_context.parquet');municipal=b.drop_duplicates('municipality_code')
cuts=municipal.log_income.quantile([.25,.5,.75]);q=pd.cut(b.log_income,[-np.inf,*cuts,np.inf],labels=['Q1','Q2','Q3','Q4'])
assert (q.astype(str)==b.income_quartile.astype(str)).all()
ck('Brazil complete municipal sample',municipal.municipality_code.nunique(),336)
group_notes['Brazil']={'basis':'GDP quartile cutpoints across the 336 distinct admitted municipalities; all 814 areas inherit the municipality group','cutpoints_brl':list(map(float,np.exp(cuts))),'municipalities_by_group':municipal.assign(q=pd.cut(municipal.log_income,[-np.inf,*cuts,np.inf],labels=['Q1','Q2','Q3','Q4'])).groupby('q',observed=True).size().to_dict(),'area_counts':q.value_counts().to_dict()}
frames.append(('Brazil',b,q,[(c,'FEC'+suffix,'DEC'+suffix) for c,suffix in [('Ordinary unplanned','IND'),('Planned','IP'),('Emergency','INE'),('Critical day','INC')]]))
for name,fn in [('United States','us_same49_state_means'),('Europe','europe_customer11_system_means')]:
 z=pd.read_parquet(V/(fn+'.parquet'));q=z.quartile if 'quartile' in z else pd.qcut(z.income.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
 frames.append((name,z,q,[(label,'F_'+s,'H_'+s) for label,s in [('Routine / non-major','routine'),('Exceptional / major','exceptional')]]))
 group_notes[name]={'basis':'Frozen source-wide economic quartiles; equal reporting units; unit means over admitted years','all_units':len(z),'group_counts':q.value_counts().to_dict()}
n=pd.read_parquet(V/'nz_explicit21_total_context.parquet');parts=pd.read_parquet(V/'nz_explicit21_complete_partition.parquet');cats=[]
for i,cat in enumerate(['Weather','Human activity','Remaining causes']):
 n=n.merge(parts[parts.category.eq(cat)][['system','frequency','burden_h']].rename(columns={'frequency':f'F{i}','burden_h':f'H{i}'}),on='system',validate='one_to_one');cats.append((cat,f'F{i}',f'H{i}'))
q=pd.qcut(n.income_grouped_median_nzd.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4']);frames.append(('New Zealand',n,q,cats));group_notes['New Zealand']={'basis':'Rank quartiles of geographically allocated personal income across 21 admitted supply regions','group_counts':q.value_counts().to_dict()}
for region,z,q,cats in frames:
 lh=[]
 for g in ['Q1','Q4']:
  x=z[q.eq(g)];ck(region+' '+g+' support',len(x),summ.loc[region,'n_'+g]);lh.append(np.array([[x[f].mean(),x[h].mean()] for _,f,h in cats]))
 low,high=lh
 for j,(cat,_,_) in enumerate(cats):
  fl,hl=low[j];fh,hh=high[j];dl=hl/fl;dh=hh/fh
  # Two factor replacement orders provide an independent algebraic implementation.
  cf=((fl*dh-fh*dh)+(fl*dl-fh*dl))/2
  cd=((fh*dl-fh*dh)+(fl*dl-fl*dh))/2
  for component,value in [('gap',hl-hh),('frequency',cf),('duration',cd)]:
   rr=reported[reported.region.eq(region)&reported.category.eq(cat)&reported.component.eq(component)];assert len(rr)==1
   ck(region+' '+cat+' '+component,rr.value_h.iloc[0],value)
 ck(region+' total annual gap',low[:,1].sum()-high[:,1].sum(),summ.loc[region,'total_gap'])
 ck(region+' class totals',reported[reported.region.eq(region)&reported.component.eq('gap')].value_h.sum(),summ.loc[region,'total_gap'])
assert summ.loc['United States','total_gap_lower']<0<summ.loc['United States','total_gap_upper']
checks.append(dict(check='US net gap not labelled direction resolved',passed=not bool(summ.loc['United States','gap_direction_resolved'])))
assert summ.loc['Europe','undefined_class_partition_draws']==144
(P/'metadata/group_definitions.json').write_text(json.dumps(group_notes,indent=2,default=int))
(P/'metadata/independent_reproduction.json').write_text(json.dumps({'checks':checks,'all_passed':all(x['passed'] for x in checks),'point_reconstruction':'Original native class input tables and frozen source-wide economic groups; no import of analyse.py','bootstrap_defined_draws':'Europe exceptional class: 3856 of 4000; zero-frequency draws have undefined duration components and are not imputed'},indent=2))
print('Independent reconstruction passed',len(checks),'checks')
print(json.dumps(group_notes,indent=2,default=int))
