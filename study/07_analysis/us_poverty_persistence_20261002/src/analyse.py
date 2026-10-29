"""Estimate all specified poverty contrasts; retain null and reversed results."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t
P=Path(__file__).resolve().parents[1];D=P/'data';M=P/'metadata'
def row(label,spec,outcome,b,se,n,counties,states,extra=None):
    critical=t.ppf(.975,states-1);lo=b-critical*se;hi=b+critical*se
    ratio=outcome in ['log_duration_ratio','log_mean_h']
    transform=(lambda v:100*np.expm1(v)) if ratio else (lambda v:v)
    return dict(comparison=label,specification=spec,outcome=outcome,beta=b,se=se,contrast=transform(b),lower=transform(lo),upper=transform(hi),units='Percent' if ratio else 'Percentage points',poverty_contrast_pp=5,n=n,counties=counties,states=states,df=states-1,p_value=2*t.sf(abs(b/se),states-1),**(extra or {}))
def event_fit(q,outcome,covars):
    # Within-event projection equals an explicit event-intercept regression.
    y=q[outcome]-q.groupby('event_key')[outcome].transform('mean')
    x=q[covars]-q.groupby('event_key')[covars].transform('mean')
    X=x.to_numpy();Y=y.to_numpy();bread=np.linalg.inv(X.T@X);b=bread@X.T@Y;resid=Y-X@b
    scores=pd.DataFrame(X*resid[:,None]);scores['state']=q.state.to_numpy();S=scores.groupby('state').sum().to_numpy()
    n=len(q);g=q.state.nunique();full_k=q.event_key.nunique()+len(covars)
    cov=bread@S.T@S@bread*(g/(g-1))*((n-1)/(n-full_k))
    return b,np.sqrt(np.diag(cov)),dict(absorbed_event_intercepts=q.event_key.nunique(),within_r_squared=1-(resid@resid)/(Y@Y),residual_sd=float(np.std(resid)),design_condition=float(np.linalg.cond(X)))
def mixture(g):
    # Input per-state sufficient statistics with optional multiplicities.
    z=g.groupby(['poverty_quartile','weather'],observed=True)[['counties','share_sum','duration_contribution_sum']].sum()
    vals={}
    for qt in ['Q1','Q4']:
        h=z.loc[qt].sort_index();s=h.share_sum.to_numpy()/h.counties.to_numpy();d=h.duration_contribution_sum.to_numpy()/h.share_sum.to_numpy();vals[qt]=(s,d)
    s0,d0=vals['Q1'];s1,d1=vals['Q4'];total=s1@d1-s0@d0
    within=((s1+s0)/2)@(d1-d0);composition=((d1+d0)/2)@(s1-s0)
    assert np.isclose(total,within+composition)
    return np.array([s0@d0,s1@d1,total,within,composition,100*s0[1],100*s1[1],100*(s1[1]-s0[1])])
def main():
    pairs=pd.read_parquet(D/'paired_county_year.parquet');rows=[];diagnostics=[]
    for sample,q0 in [('Primary',pairs),('Five per class',pairs[pairs.five_each])]:
        q=q0.copy();q['weight']=1/q.groupby('fips').fips.transform('size')
        for spec,terms in [('Unadjusted','poverty_5pp'),('State and year','poverty_5pp+C(state)+C(year)'),('Population and density','poverty_5pp+log_population+log_density+C(state)+C(year)')]:
            for outcome in ['log_duration_ratio','long_difference_pp']:
                fit=smf.wls(outcome+'~'+terms,q,weights=q.weight).fit(cov_type='cluster',cov_kwds={'groups':q.state,'use_correction':True},use_t=True)
                rows.append(row('County-year: '+sample,spec,outcome,float(fit.params.poverty_5pp),float(fit.bse.poverty_5pp),len(q),q.fips.nunique(),q.state.nunique(),{'r_squared':float(fit.rsquared)}))
                diagnostics.append(dict(comparison='County-year: '+sample,specification=spec,outcome=outcome,design_rank=int(np.linalg.matrix_rank(fit.model.exog)),columns=fit.model.exog.shape[1],residual_sd=float(np.std(fit.resid)),condition=float(fit.condition_number)))
    e=pd.read_parquet(D/'county_event_summary.parquet')
    for spec,covars in [('Same event',['poverty_5pp']),('Same event, population and density',['poverty_5pp','log_population','log_density'])]:
        for outcome in ['log_mean_h','long_pct']:
            b,se,diag=event_fit(e,outcome,covars)
            rows.append(row('Within published weather event',spec,outcome,float(b[0]),float(se[0]),len(e),e.fips.nunique(),e.state.nunique(),diag))
            diagnostics.append(dict(comparison='Within published weather event',specification=spec,outcome=outcome,**diag))
    results=pd.DataFrame(rows);results.to_csv(D/'poverty_duration_models.csv',index=False)
    # Equal-county descriptive mixture decomposition and state bootstrap.
    cs=pd.read_parquet(D/'county_class_summary.parquet');cs['contribution']=cs.share*cs.mean_h
    s=cs.groupby(['state','poverty_quartile','weather'],observed=True).agg(counties=('fips','size'),share_sum=('share','sum'),duration_contribution_sum=('contribution','sum')).reset_index()
    observed=mixture(s);states=s.state.unique();rng=np.random.default_rng(261002);boots=[]
    for _ in range(999):
        multiplicity=pd.Series(rng.choice(states,len(states),replace=True)).value_counts().rename('m')
        b=s.merge(multiplicity,left_on='state',right_index=True)
        b[['counties','share_sum','duration_contribution_sum']]=b[['counties','share_sum','duration_contribution_sum']].mul(b.m,axis=0)
        boots.append(mixture(b))
    bounds=np.quantile(boots,[.025,.975],axis=0)
    decomp=pd.DataFrame({'quantity':['Low-poverty mean','High-poverty mean','High minus low','Within-class','Class composition','Low-poverty weather share','High-poverty weather share','Weather share gap'],'value':observed,'lower':bounds[0],'upper':bounds[1],'units':['Hours']*5+['Percent','Percent','Percentage points']});decomp.to_csv(D/'poverty_duration_decomposition.csv',index=False)
    pd.DataFrame(boots,columns=decomp.quantity).to_parquet(D/'decomposition_bootstrap.parquet',index=False)
    # Distribution summaries are calculated on counties, not independent episodes.
    county=pd.read_parquet(D/'county_contrasts.parquet');summary=[]
    for quartile,z in county.groupby('poverty_quartile',observed=True):
        summary.append(dict(quartile=str(quartile),counties=len(z),poverty_median=z.poverty_pct.median(),poverty_min=z.poverty_pct.min(),poverty_max=z.poverty_pct.max(),duration_ratio_geomean=np.exp(z.log_duration_ratio.mean()),duration_ratio_median=np.exp(z.log_duration_ratio.median()),long_difference_pp_mean=z.long_difference_pp.mean(),log_duration_sd=z.log_duration_ratio.std()))
    pd.DataFrame(summary).to_csv(D/'county_poverty_group_contrasts.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(M/'model_diagnostics.csv',index=False)
    result=dict(primary_model=results.query("comparison=='County-year: Primary' and specification=='Population and density'").to_dict('records'),event_model=results.query("comparison=='Within published weather event'").to_dict('records'),decomposition=decomp.to_dict('records'),uncertainty='State-clustered covariance with t on states-1; mixture intervals use 999 state bootstrap resamples; intervals condition on published poverty estimates',scope='Reported episode persistence and weather association, not customer frequency or whole-disaster recovery',all_specified_models=len(results))
    (M/'findings.json').write_text(json.dumps(result,indent=2));print(results[['comparison','specification','outcome','contrast','lower','upper','states']].round(3).to_string(index=False));print(decomp.round(3).to_string(index=False))
if __name__=='__main__':main()
