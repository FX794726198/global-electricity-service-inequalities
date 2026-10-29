"""Paired PPML equations, true zeros retained, joint stage covariance."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,geopandas as gpd
import statsmodels.api as sm
from scipy.stats import t

P=Path(__file__).resolve().parents[1];ROOT=P.parents[1];D=P/'data';M=P/'metadata'
rows=[];audits=[];receipts=[]

def read(rel,geo=False):
    p=ROOT/rel
    with p.open('rb') as f:s=hashlib.file_digest(f,'sha256').hexdigest()
    receipts.append(dict(path=rel,sha256=s))
    return gpd.read_parquet(p) if geo else pd.read_parquet(p)

def save(d,n):
    d.to_parquet(D/f'{n}.parquet',index=False);d.drop(columns='geometry',errors='ignore').to_csv(D/f'{n}.csv',index=False)

def meat(scores,g,p):
    codes=pd.factorize(g)[0];ng=codes.max()+1
    agg=np.zeros((ng,scores.shape[1]));np.add.at(agg,codes,scores)
    n=len(scores)
    return (agg.T@agg)*(ng/(ng-1))*((n-1)/(n-p))

def paired(z,X,F,H,meta,contrasts,offset=None,groups=None):
    x=X.to_numpy(float);n,p=x.shape;offset=np.zeros(n) if offset is None else np.asarray(offset,float)
    estimates=[];scores=[];bread=np.zeros((2*p,2*p));fits=[]
    for k,y in enumerate([np.asarray(F,float),np.asarray(H,float)]):
        fit=sm.GLM(y,x,family=sm.families.Poisson(),offset=offset).fit(maxiter=200,tol=1e-10)
        assert fit.converged,meta
        mu=fit.fittedvalues
        bread[k*p:(k+1)*p,k*p:(k+1)*p]=np.linalg.pinv(x.T@(mu[:,None]*x))
        scores.append(x*(y-mu)[:,None]);estimates.extend(fit.params);fits.append(fit)
    scores=np.hstack(scores)
    if groups is None:
        # One independent geographical aggregate per row.
        V=scores.T@scores*n/(n-p);df=n-p
    elif len(groups)==1:
        V=meat(scores,groups[0],p);df=pd.Series(groups[0]).nunique()-1
    else:
        joint=pd.Series(list(zip(groups[0],groups[1]))).astype(str)
        V=meat(scores,groups[0],p)+meat(scores,groups[1],p)-meat(scores,joint,p)
        df=min(pd.Series(groups[0]).nunique(),pd.Series(groups[1]).nunique())-1
    cov=bread@V@bread;co=np.asarray(estimates);crit=t.ppf(.975,df)
    for term,delta in contrasts.items():
        j=X.columns.get_loc(term)
        for outcome,w in [('frequency',(1,0)),('duration_component',(-1,1)),('burden',(0,1)),('frequency_minus_duration',(2,-1))]:
            L=np.zeros(2*p);L[j]=w[0];L[p+j]=w[1];b=L@co;var=L@cov@L
            se=np.sqrt(var) if var>=0 else np.nan;lo=b-crit*se;hi=b+crit*se
            rows.append(dict(**meta,outcome=outcome,term=term,n=n,df=df,beta=b,se=se,lo=lo,hi=hi,p=2*t.sf(abs(b/se),df),
                contrast_log=delta,contrast_pct=100*np.expm1(b*delta),contrast_lo_pct=100*np.expm1(lo*delta),contrast_hi_pct=100*np.expm1(hi*delta),
                F_zeros=int(np.sum(np.asarray(F)==0)),H_zeros=int(np.sum(np.asarray(H)==0)),model='paired PPML; joint sandwich'))
    audits.append(dict(**meta,n=n,p=p,rank=np.linalg.matrix_rank(x),max_score=float(np.max(abs(scores.sum(axis=0)))),
        F_observed_sum=float(np.sum(F)),F_fitted_sum=float(fits[0].fittedvalues.sum()),H_observed_sum=float(np.sum(H)),H_fitted_sum=float(fits[1].fittedvalues.sum()),
        fitted_F_min=float(fits[0].fittedvalues.min()),fitted_H_min=float(fits[1].fittedvalues.min()),both_converged=all(f.converged for f in fits)))

def run_china():
    a=pd.read_parquet(D/'china_common_counties.parquet')
    delta=float(a.log_income.quantile(.75)-a.log_income.quantile(.25))
    for cls,code in [('All notices','all_notices'),('Planned','planned'),('Unplanned','unplanned')]:
        for spec in ['Economic context','Population and hazard exposure']:
            X=pd.DataFrame({'const':1.,'log_income':a.log_income},index=a.index)
            if spec!='Economic context':X['log_population']=a.log_population;X['hazard_fraction']=a.hazard_fraction
            paired(a,X,a[code+'_count'],a[code+'_hours'],dict(region='China',category=cls,specification=spec),{'log_income':delta},offset=a.log_population)

def run_brazil():
    a=pd.read_parquet(D/'brazil_2021_joint_context.parquet')
    m=a.drop_duplicates('municipality_code');di=float(m.log_income.quantile(.75)-m.log_income.quantile(.25));dd=float(a.log_density.quantile(.75)-a.log_density.quantile(.25))
    classes={'Ordinary unplanned':('FECIND','DECIND'),'Planned':('FECIP','DECIP'),
        'Emergency':('FECINE','DECINE'),'Critical day':('FECINC','DECINC'),
        'Internal unplanned':('FEC_internal_unplanned_3strata','DEC_internal_unplanned_3strata')}
    for cls,(fc,hc) in classes.items():
        for spec in ['Economic context','Density adjusted','Within operator','Within operator + density']:
            z=a.copy()
            if spec.startswith('Within'):
                # All-zero fixed-effect groups have fitted means at the boundary;
                # they inform absolute accounting, not within-group log slopes.
                sums=z.groupby('NumCNPJ')[[fc,hc]].transform('sum');z=z[(sums>0).all(axis=1)].copy()
            X=pd.DataFrame({'const':1.,'log_income':z.log_income},index=z.index);contrasts={'log_income':di}
            if 'density' in spec.lower():X['log_density']=z.log_density;contrasts['log_density']=dd
            if spec.startswith('Within'):X=pd.concat([X,pd.get_dummies(z.NumCNPJ.astype(str),prefix='op',drop_first=True,dtype=float)],axis=1)
            paired(z,X,z[fc],z[hc],dict(region='Brazil',category=cls,specification=spec),contrasts,groups=[z.municipality_code,z.NumCNPJ])

if __name__=='__main__':
    run_china();run_brazil()
    save(pd.DataFrame(rows),'zero_inclusive_models');save(pd.DataFrame(audits),'zero_inclusive_diagnostics')
    M.joinpath('zero_inclusive_source_manifest.json').write_text(json.dumps(receipts,indent=2))
    r=pd.DataFrame(rows);print(r[(r.term=='log_income')&r.outcome.isin(['frequency','duration_component'])][['region','category','specification','outcome','n','contrast_pct','contrast_lo_pct','contrast_hi_pct']].to_string(index=False))
