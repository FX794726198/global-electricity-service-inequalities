"""Independent raw-record reconstruction of the primary paired population."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];R=P.parents[1]
raw=pd.read_parquet(R/'07_analysis/turkiye_cause_extension_20261002/data/sedas_2025_numeric.parquet')
u=raw[(raw.KesintiniSinifi5BildirimeGore=='Bildirimsiz')&(raw.KesintiniSinifi5SureyeGore5B=='Uzun')].copy()
u['R_CI']=u.EtkilenenKullaniciSayisiKirsalAG+u.EtkilenenKullaniciSayisiKirsalOG
u['U_CI']=u.EtkilenenKullaniciSayisiKentselAG+u.EtkilenenKullaniciSayisiKentselOG
u['R_H']=u.ToplamEtkilenmeSuresiKirsalAG+u.ToplamEtkilenmeSuresiKirsalOG
u['U_H']=u.ToplamEtkilenmeSuresiKentselAG+u.ToplamEtkilenmeSuresiKentselOG
g=u.groupby('KesintininKodu1').agg(R_CI=('R_CI','sum'),U_CI=('U_CI','sum'),R_H=('R_H','sum'),U_H=('U_H','sum'),starts=('KesintininBaslamaBitisTarihi','nunique'))
g=g[(g.R_CI>0)&(g.U_CI>0)&(g.R_H>0)&(g.U_H>0)&(g.starts==1)]
pairs=pd.read_parquet(P/'data/shared_incident_pairs.parquet')
q=pairs[(pairs.voltage_scope=='All')&(pairs.start_count==1)].set_index('KesintininKodu1').sort_index()
g=g.sort_index()
checks=[]
def check(name,condition):
    checks.append(dict(check=name,passed=bool(condition)))
    assert condition,name
check('Raw-source primary cohort agrees exactly',g.index.equals(q.index) and len(g)==2661)
for a,b in [('R_CI','Rural_CI'),('U_CI','Urban_CI'),('R_H','Rural_H'),('U_H','Urban_H')]:
    check('Independent raw sums '+a,np.allclose(g[a],q[b],rtol=1e-12,atol=1e-7))
check('One common onset within each primary incident',(q.start_count==1).all())
est=pd.read_csv(P/'data/shared_incident_estimates.csv')
v=est.query("population=='Shared identifier and onset' and voltage=='All' and province=='All' and scope=='All causes'").iloc[0]
ratio=np.exp(np.log((g.R_H/g.R_CI)/(g.U_H/g.U_CI)).mean())
check('Independent primary ratio',np.isclose(v.ratio,ratio,atol=1e-12))
check('Positive-count summaries have ordered intervals',(est.loc[est.n>0,'lower']<=est.loc[est.n>0,'upper']).all())
account=pd.read_csv(P/'data/shared_incident_accounting.csv')
for key,z in account.groupby(['population','voltage','scope']):
    a=z.set_index('metric').value
    check('Incident composition identity '+str(key),np.isclose(a.gap_h,a.within_incident_h+a.between_incident_allocation_h,atol=1e-11))
ca=pd.read_csv(P/'data/cause_contrast_and_accounting.csv')
check('All cause annual-hour identities',np.allclose(ca.hour_gap,ca.frequency_component_h+ca.duration_component_h,atol=1e-11))
check('All cause ratio identities',np.allclose(ca.frequency_ratio*ca.duration_ratio,ca.rural_hours/ca.urban_hours))
cov=pd.read_csv(P/'data/shared_incident_coverage.csv')
check('Reported matched coverage is bounded',((cov.shared_CI_fraction.between(0,1))&(cov.shared_H_fraction.between(0,1))).all())
check('Primary paired cohorts nested in original support',set(q.index)<=set(u.KesintininKodu1))
out=dict(checks=checks,passed=sum(c['passed'] for c in checks),total=len(checks),
    scientific_limit='Verifies source arithmetic and population handling, not national representativeness or repair causality.')
(P/'metadata/independent_validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:out[k] for k in ['passed','total']}))
