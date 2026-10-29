"""Post-hoc direct contrast of poverty gradients in two regulatory conditions.

Motivated by the initial stage estimates, not represented as a preregistered test.
All four equations use the same 703 areas and their joint cluster covariance.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import block_diag
from scipy.stats import t
P=Path(__file__).resolve().parents[1];D=P/'data'
c=pd.read_parquet(D/'analysis_context.parquet').sort_values('IdeConjunto').reset_index(drop=True)
reg=pd.read_parquet(D/'regulatory_stages.parquet')
ys=[]
for cat in ['Ordinary unplanned','Emergency / critical']:
    a=reg[reg.category.eq(cat)].set_index('IdeConjunto').loc[c.IdeConjunto]
    ys.extend([a.frequency.to_numpy(),a.burden_hours.to_numpy()])
rows=[]
for spec in ['Unadjusted','Density adjusted','State + density']:
    X=pd.DataFrame({'const':np.ones(len(c)),'disadvantage_10pp':c.disadvantage_10pp})
    if spec!='Unadjusted':X['log_density']=c.log_density
    if spec=='State + density':X=pd.concat([X,pd.get_dummies(c.state_code,drop_first=True,dtype=float)],axis=1)
    fits=[];breads=[];scores=[]
    for y in ys:
        fit=sm.GLM(y,X,family=sm.families.Poisson()).fit(maxiter=300,tol=1e-10);assert fit.converged
        mu=np.asarray(fit.fittedvalues);fits.append(fit)
        scores.append(np.asarray(X)*(y-mu)[:,None]);breads.append(np.linalg.inv(np.asarray(X).T@(mu[:,None]*np.asarray(X))))
    S=np.column_stack(scores);B=block_diag(*breads);n,k=X.shape;meats=[]
    for group in [c.municipality_code,c.cnpj,pd.MultiIndex.from_arrays([c.municipality_code,c.cnpj])]:
        codes,vals=pd.factorize(group);sg=pd.DataFrame(S).groupby(codes).sum().to_numpy();G=len(vals)
        meats.append((sg.T@sg)*(G/(G-1))*(n-1)/(n-k))
    cov=B@(meats[0]+meats[1]-meats[2])@B
    pars=np.concatenate([f.params for f in fits]);df=min(c.municipality_code.nunique(),c.cnpj.nunique())-1
    for metric,weights in [('Frequency',[(2*k+1,1),(1,-1)]),
        ('Annual burden',[(3*k+1,1),(k+1,-1)]),
        ('Conditional duration',[(3*k+1,1),(2*k+1,-1),(k+1,-1),(1,1)])]:
        L=np.zeros(4*k)
        for i,w in weights:L[i]=w
        beta=float(L@pars);v=float(L@cov@L);assert v>=0
        se=np.sqrt(v);crit=t.ppf(.975,df)
        rows.append(dict(specification=spec,metric=metric,areas=n,beta_difference=beta,se=se,
            contrast_ratio=np.exp(beta),ratio_lower=np.exp(beta-crit*se),ratio_upper=np.exp(beta+crit*se),
            p_value=2*t.sf(abs(beta/se),df),df=df,
            interpretation='Emergency/critical versus ordinary gradient ratio per +10 percentage points low-income share',
            analysis_status='Post-hoc direct comparison motivated by observed condition-specific gradients'))
out=pd.DataFrame(rows);out.to_csv(D/'condition_gradient_comparison.csv',index=False)
print(out[['specification','metric','contrast_ratio','ratio_lower','ratio_upper','p_value']].round(3).to_string(index=False))
