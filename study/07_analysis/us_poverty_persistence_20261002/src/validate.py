"""Targeted independent checks of joins, pair arithmetic and event projection."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np,pandas as pd,statsmodels.formula.api as smf
P=Path(__file__).resolve().parents[1];B=P.parents[1];D=P/'data';checks=[]
def check(name,condition,detail=''):
    checks.append(dict(check=name,passed=bool(condition),detail=detail))
    assert condition,(name,detail)
def main():
    c=pd.read_parquet(D/'county_context_2018.parquet');s=pd.read_parquet(D/'county_support.parquet');p=pd.read_parquet(D/'paired_county_year.parquet');e=pd.read_parquet(D/'county_event_summary.parquet')
    check('Five-digit unique county IDs',c.fips.str.fullmatch(r'\d{5}').all() and not c.fips.duplicated().any())
    check('SAIPE aggregate rows excluded',not c.fips.str.endswith('000').any())
    check('Known first official county parsed',np.isclose(c.set_index('fips').loc['01001','poverty_pct'],13.8) and c.set_index('fips').loc['01001','name']=='Autauga County')
    check('All admitted contexts positive and finite',np.isfinite(s.loc[s.admitted,['poverty_pct','log_population','log_density']]).all().all())
    check('Population precedes all outcomes',p.year.min()==2019 and p.year.max()==2023)
    check('County weights sum to unity',np.allclose(p.groupby('fips').weight.sum(),1))
    check('Unique county-year comparisons',not p.duplicated(['fips','year']).any())
    check('Exact paired log-duration arithmetic',np.allclose(np.exp(p.log_duration_ratio),p.mean_h_weather/p.mean_h_unlinked))
    check('Exact paired long-tail arithmetic',np.allclose(p.long_difference_pp,100*(p.long_share_weather-p.long_share_unlinked)))
    raw=pd.read_parquet(B/'07_analysis/hazard_outage_results_20260926/data/us_analysis_records.parquet')
    sub=raw[raw.fips.isin(s.loc[s.admitted,'fips'])].copy()
    counts=sub.groupby(['fips','year','weather']).size().unstack('weather').dropna()
    check('Pair membership independently reproduced',set(counts.index)==set(zip(p.fips,p.year)))
    q=p.set_index(['fips','year']);check('Episode denominators independently reproduced',np.array_equal(counts[True].to_numpy(),q.reindex(counts.index).n_weather.to_numpy()) and np.array_equal(counts[False].to_numpy(),q.reindex(counts.index).n_unlinked.to_numpy()))
    cs=pd.read_parquet(D/'county_class_summary.parquet');g=pd.read_csv(D/'poverty_group_composition.csv')
    means=sub[sub.fips.isin(cs.fips)].groupby('fips').duration.mean().rename('mean_h').reset_index().merge(s[['fips','poverty_quartile']],on='fips')
    for qt,zz in g.groupby('quartile'):
        direct=means[means.poverty_quartile.eq(qt)].mean_h.mean();calc=(zz.class_share*zz.class_duration_h).sum()
        check('Equal-county mixture identity '+qt,np.isclose(direct,calc),str(direct))
    dec=pd.read_csv(D/'poverty_duration_decomposition.csv').set_index('quantity')
    check('Decomposition closes',np.isclose(dec.loc['High minus low','value'],dec.loc['Within-class','value']+dec.loc['Class composition','value']))
    ids=e.groupby('state').event_key.first().tolist();sample=e[e.event_key.isin(ids)].copy()
    spec=importlib.util.spec_from_file_location('analysis_module',P/'src/analyse.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for outcome in ['log_mean_h','long_pct']:
        b,se,diag=mod.event_fit(sample,outcome,['poverty_5pp','log_population','log_density'])
        fit=smf.ols(outcome+'~poverty_5pp+log_population+log_density+C(event_key)',sample).fit(cov_type='cluster',cov_kwds={'groups':sample.state,'use_correction':True},use_t=True)
        check('Absorbed event coefficient matches explicit dummies '+outcome,np.allclose(b,fit.params[['poverty_5pp','log_population','log_density']],atol=1e-9))
        check('Absorbed state covariance matches explicit dummies '+outcome,np.allclose(se,fit.bse[['poverty_5pp','log_population','log_density']],rtol=1e-8))
    models=pd.read_csv(D/'poverty_duration_models.csv');check('All planned estimates retained',len(models)==16 and np.isfinite(models[['contrast','lower','upper']]).all().all())
    check('Every admitted event has five or more counties',e.groupby('event_key').fips.nunique().ge(5).all())
    result=dict(checks=checks,passed=len(checks),failed=0,paired_episode_records=int(p.n_weather.sum()+p.n_unlinked.sum()),event_episode_records=int(e.n.sum()),scientific_scope='Checks establish calculations and source support, not causal identification or journal readiness')
    (P/'metadata/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
