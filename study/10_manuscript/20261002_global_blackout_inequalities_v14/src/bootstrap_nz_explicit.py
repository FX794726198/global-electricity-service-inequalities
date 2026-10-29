"""One paired operator bootstrap for the new, small NZ cause comparison."""
from pathlib import Path
import numpy as np,pandas as pd,json,statsmodels.api as sm
P=Path(__file__).resolve().parents[1];D=P/'data';M=P/'metadata'
d=pd.read_parquet(D/'nz_explicit21_cause_context.parquet');c=d.drop_duplicates('system').set_index('system');z=d.pivot(index='system',columns='category',values=['frequency','burden_h']).loc[c.index]
Y=np.column_stack([z['frequency','Weather'],z['burden_h','Weather'],z['frequency','Human activity'],z['burden_h','Human activity']])
di=c.log_income.quantile(.75)-c.log_income.quantile(.25)
defs={'Weather frequency':[1,0,0,0],'Weather duration':[-1,1,0,0],'Human frequency':[0,0,1,0],'Human duration':[0,0,-1,1],
    'Weather minus human frequency':[1,0,-1,0],'Weather minus human duration':[-1,1,1,-1],'Weather frequency minus duration':[2,-1,0,0]}
rng=np.random.default_rng(202610011);B=2000;indices=rng.integers(0,len(c),(B,len(c)));draws=[];failures=[];summary=[]
for spec in ['Economic context','Density adjusted']:
    X=np.column_stack([np.ones(len(c)),c.log_income-c.log_income.mean()])
    if spec=='Density adjusted':X=np.column_stack([X,c.log_density-c.log_density.mean()])
    def fit(idx):
        slopes=[]
        for k in range(4):
            m=sm.GLM(Y[idx,k],X[idx],family=sm.families.Poisson()).fit(maxiter=100,tol=1e-9)
            if not m.converged or not np.isfinite(m.params).all():raise ValueError('model did not converge')
            slopes.append(m.params[1])
        return np.asarray(slopes)
    point=fit(np.arange(len(c)));betas=[]
    for i,idx in enumerate(indices):
        try:betas.append(fit(idx));draws.append(dict(specification=spec,replicate=i,**dict(zip(['Fw','Hw','Fh','Hh'],betas[-1]))))
        except Exception as e:failures.append(dict(specification=spec,replicate=i,error=str(e)))
    betas=np.array(betas)
    assert len(betas)>=B*.99
    for label,w in defs.items():
        values=betas@np.array(w);lo,hi=np.quantile(values,[.025,.975]);b=point@np.array(w)
        summary.append(dict(specification=spec,contrast=label,n_operators=len(c),successful_bootstraps=len(betas),beta=b,lo=lo,hi=hi,contrast_log=di,
            contrast_pct=100*np.expm1(b*di),contrast_lo_pct=100*np.expm1(lo*di),contrast_hi_pct=100*np.expm1(hi*di)))
    print('Finished',spec,len(betas),flush=True)
pd.DataFrame(draws).to_parquet(D/'nz_explicit21_operator_bootstrap_draws.parquet',index=False)
pd.DataFrame(summary).to_csv(D/'nz_explicit21_operator_bootstrap_summary.csv',index=False)
M.joinpath('nz_explicit21_bootstrap_qa.json').write_text(json.dumps(dict(seed=202610011,replicates=B,units=len(c),failures=failures),indent=2))
print(pd.DataFrame(summary)[['specification','contrast','contrast_pct','contrast_lo_pct','contrast_hi_pct']].to_string(index=False))
