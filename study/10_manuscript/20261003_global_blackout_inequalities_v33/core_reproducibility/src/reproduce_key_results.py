"""Independent numerical reconstruction from supplied inputs; no analysis-module imports."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
P=Path(__file__).resolve().parents[1];I=P/'inputs';O=P/'metadata';O.mkdir(exist_ok=True)
checks=[]
def check(name,value):
 checks.append({'check':name,'passed':bool(value)})
 if not value:raise AssertionError(name)
g=pd.read_csv(I/'country_benchmarks.csv')
check('152 distinct economic benchmarks',len(g)==152 and g.ISO3.is_unique)
# Locate the exact economy and database label in both original indicator responses.
for measure,code in [('saifi','IC.ELC.SAIF.XD.DB1619'),('saidi_h','IC.ELC.SAID.XD.DB1619')]:
 raw=json.loads((I/(code+'.json')).read_text())[1]
 for r in g.itertuples():
  entries=[x for x in raw if x['country']['value']==r.label and int(x['date'])==r.year]
  check('raw '+measure+' '+r.ISO3,len(entries)==1 and np.isclose(entries[0]['value'],getattr(r,measure)))
check('CAIDI reconstructed from the paired indices',np.allclose(g.caidi_h,g.saidi_h/g.saifi))
# Average the frequency contribution along both paths between the group means.
def decompose(d):
 lo=d[d.gdp_quartile=='Q1'];hi=d[d.gdp_quartile=='Q4']
 f0,f1=hi.saifi.mean(),lo.saifi.mean();h0,h1=hi.saidi_h.mean(),lo.saidi_h.mean();d0,d1=h0/f0,h1/f1
 fterm=((f1-f0)*d0+(f1-f0)*d1)/2
 dterm=((d1-d0)*f0+(d1-d0)*f1)/2
 logH=np.log(lo.saidi_h).mean()-np.log(hi.saidi_h).mean()
 logF=np.log(lo.saifi).mean()-np.log(hi.saifi).mean()
 return dict(n=len(d),n_Q1=len(lo),n_Q4=len(hi),annual_gap_h=h1-h0,frequency_component_h=fterm,duration_component_h=dterm,frequency_share=100*fterm/(h1-h0),log_frequency_share=100*logF/logH)
base=decompose(g)
check('absolute components close exactly',np.isclose(base['frequency_component_h']+base['duration_component_h'],base['annual_gap_h']))
j=pd.read_csv(I/'definition_linkage.csv');ad=j.year.eq(2019)&j.minimum_minutes.notna()&j.saifi.sub(j.saifi_published).abs().le(.051)&j.saidi_h.sub(j.saidi_published).abs().le(.051)
check('definition gate reconstructed',np.array_equal(ad,j.definition_admitted))
check('132 definition-matched cases',ad.sum()==132)
variants=[g,g[g.year==2019],j[ad],j[ad&j.minimum_minutes.eq(3)],j[ad&j.minimum_minutes.eq(5)]]
saved=pd.read_csv(I/'benchmark_definition_sensitivity.csv')
calc=pd.DataFrame([decompose(x) for x in variants]);check('all five threshold accounting results reproduced',np.allclose(calc[saved.columns.intersection(calc.columns)].to_numpy(),saved[saved.columns.intersection(calc.columns)].to_numpy()))
# Same respondents and same weights: conditional local grid context in both groups.
h=pd.read_csv(I/'conditional_grid_contrasts.csv');a=h.connection_no-h.connection_high;b=(h.connection_high-h.service_high)-(h.connection_no-h.service_no)
check('23 conditional-grid populations',len(h)==23 and h.admitted.all())
check('every service partition closes',np.allclose(a+b,h.service_gap))
check('both groups share admitted support',h[['n_no','n_high']].ge(30).all().all())
check('all 23 conditional service gaps positive',h.service_gap.gt(0).all())
rng=np.random.default_rng(20261002);v=np.column_stack([a,b,a+b]);bs=v[rng.integers(0,23,size=(4000,23))].mean(axis=1)*100
house={'components_pp':(v.mean(axis=0)*100).tolist(),'lower_pp':np.quantile(bs,.025,axis=0).tolist(),'upper_pp':np.quantile(bs,.975,axis=0).tolist()}
# Cause shares use their native denominators; no country-level pooling of impacts.
d=pd.read_parquet(I/'global_cause_atlas.parquet');pr=pd.read_parquet(I/'global_cause_atlas_profiles.parquet');vals=[]
for pid,z in d.groupby('profile_id'):
 check('native shares close within published rounding '+pid,abs(z.occurrence_share_pct.sum()-100)<=.75 and abs(z.burden_share_pct.sum()-100)<=.75)
 check('signed cause change '+pid,np.allclose(z.share_gap_pp,z.burden_share_pct-z.occurrence_share_pct))
 p=z.occurrence_share_pct.to_numpy()/z.occurrence_share_pct.sum();q=z.burden_share_pct.to_numpy()/z.burden_share_pct.sum()
 # Independent identity for total variation, rather than the plotting implementation.
 tv=(1-np.minimum(p,q).sum())*100;vals.append(tv)
check('fifteen profiles across fourteen countries',len(pr)==15 and pr.country_iso3.nunique()==14)
check('six leader changes',int((pr.leading_occurrence!=pr.leading_burden).sum())==6)
report={'checks':checks,'passed':all(x['passed'] for x in checks),'n_checks':len(checks),'benchmark_accounting':base,'definition_checks':calc.to_dict('records'),'household_partition':house,'cause_displacement_pct_range':[min(vals),max(vals)],'spearman_frequency':float(spearmanr(g.gdp_ppp_pc_at_year,g.saifi).statistic),'spearman_duration':float(spearmanr(g.gdp_ppp_pc_at_year,g.caidi_h).statistic)}
(O/'independent_numerical_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
