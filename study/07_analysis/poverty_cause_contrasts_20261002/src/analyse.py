"""Joint, common-geography contrasts between six pre-existing cause gradients."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, itertools, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import block_diag
from scipy.stats import t, f
from statsmodels.stats.multitest import multipletests

P=Path(__file__).resolve().parents[1]
S=P.parent/'social_cause_gradient_20261002'
D=P/'data'; M=P/'metadata'
D.mkdir(parents=True,exist_ok=True); M.mkdir(parents=True,exist_ok=True)
causes=['Wind','Lightning','Vegetation','Flood','Operational error','Planned']
paths=[S/'data/analysis_context.parquet',S/'data/cause_stages.parquet']
context=pd.read_parquet(paths[0]).sort_values('IdeConjunto').reset_index(drop=True)
a=pd.read_parquet(paths[1]); a=a[a.cause.isin(causes)].merge(context[['IdeConjunto','state_code']],on='IdeConjunto',validate='many_to_one')
tot=a.groupby(['state_code','cause'])[['records','record_hours']].sum()
valid=(tot.records.gt(0)&tot.record_hours.gt(0)).unstack('cause').reindex(columns=causes).fillna(False).all(axis=1)
states=valid[valid].index
c=context[context.state_code.isin(states)].reset_index(drop=True)
X=pd.concat([pd.DataFrame({'const':np.ones(len(c)),'poverty_10pp':c.disadvantage_10pp,'log_density':c.log_density}),pd.get_dummies(c.state_code,prefix='state',drop_first=True,dtype=float)],axis=1)
assert np.linalg.matrix_rank(X)==X.shape[1]
n,k=X.shape; j=1
parameters=[]; scores=[]; breads=[]; diagnostics=[]; support=[]
for cause in causes:
    z=c[['IdeConjunto']].merge(a[a.cause.eq(cause)],on='IdeConjunto',validate='one_to_one')
    assert len(z)==len(c)
    support.append(dict(cause=cause,areas=n,positive_areas=int(z.records.gt(0).sum()),records=int(z.records.sum()),record_hours=float(z.record_hours.sum()),customer_years=float(z.customer_years.sum())))
    for metric,col in [('Frequency','record_rate'),('Record-hours burden','record_hours_rate')]:
        y=1000*z[col].to_numpy()
        with warnings.catch_warnings(record=True) as caught:
            fit=sm.GLM(y,X,family=sm.families.Poisson()).fit(maxiter=300,tol=1e-10)
        mu=np.asarray(fit.fittedvalues)
        parameters.extend(fit.params)
        scores.append(np.asarray(X)*(y-mu)[:,None])
        breads.append(np.linalg.inv(np.asarray(X).T@(mu[:,None]*np.asarray(X))))
        score=float(np.max(np.abs(fit.model.score(fit.params)))/max(np.sum(y),1))
        diagnostics.append(dict(cause=cause,metric=metric,converged=bool(fit.converged),score=score,warnings=[str(w.message) for w in caught]))
        assert fit.converged and score<1e-5
parameters=np.asarray(parameters); scores=np.column_stack(scores); bread=block_diag(*breads)
groups=[pd.factorize(c.municipality_code)[0],pd.factorize(c.cnpj)[0],pd.factorize(pd.MultiIndex.from_arrays([c.municipality_code,c.cnpj]))[0]]
meats=[]
for g in groups:
    agg=pd.DataFrame(scores).groupby(g).sum().to_numpy();G=len(agg)
    meats.append((agg.T@agg)*(G/(G-1))*((n-1)/(n-k)))
cov=bread@(meats[0]+meats[1]-meats[2])@bread
df=min(c.municipality_code.nunique(),c.cnpj.nunique())-1
critical=t.ppf(.975,df)
def vector(i,metric):
    v=np.zeros(len(parameters));q=2*i*k+j
    if metric=='Frequency':v[q]=1
    elif metric=='Record-hours burden':v[q+k]=1
    else:v[q+k]=1;v[q]=-1
    return v
def estimate(v):
    beta=float(v@parameters);var=float(v@cov@v)
    assert var>0,var
    se=np.sqrt(var)
    return dict(beta=beta,se=se,change_pct=100*np.expm1(beta),lower_pct=100*np.expm1(beta-critical*se),upper_pct=100*np.expm1(beta+critical*se),p=float(2*t.sf(abs(beta/se),df)))
gradients=[];pairs=[];omnibus=[];checks=[]
for metric in ['Frequency','Record-hours burden','Conditional duration']:
    for i,cause in enumerate(causes):gradients.append(dict(cause=cause,metric=metric,**estimate(vector(i,metric))))
    for i,jj in itertools.combinations(range(len(causes)),2):
        pairs.append(dict(cause_a=causes[i],cause_b=causes[jj],metric=metric,**estimate(vector(i,metric)-vector(jj,metric))))
    L=np.vstack([vector(i,metric)-vector(0,metric) for i in range(1,len(causes))])
    v=L@cov@L.T;minimum=float(np.linalg.eigvalsh(v).min())
    assert minimum>0,(metric,minimum)
    beta=L@parameters; w=float(beta@np.linalg.solve(v,beta));p=float(f.sf(w/len(L),len(L),df))
    omnibus.append(dict(metric=metric,Wald_chisq=w,F=w/len(L),df_numerator=len(L),df_denominator=df,p=p))
    checks.append(dict(metric=metric,minimum_contrast_covariance_eigenvalue=minimum))
pairs=pd.DataFrame(pairs);pairs['p_holm_45']=multipletests(pairs.p,method='holm')[1]
omnibus=pd.DataFrame(omnibus);omnibus['p_holm_3']=multipletests(omnibus.p,method='holm')[1]
for name,z in [('gradients',pd.DataFrame(gradients)),('pairwise_contrasts',pairs),('omnibus',omnibus),('cause_support',pd.DataFrame(support))]:
    z.to_csv(D/(name+'.csv'),index=False);z.to_parquet(D/(name+'.parquet'),index=False)
c.to_parquet(D/'common_context.parquet',index=False)
np.savez_compressed(D/'joint_parameters.npz',parameters=parameters,covariance=cov,columns=np.asarray(X.columns,dtype=str),causes=np.asarray(causes))
meta=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='exploratory_after_existing_results',original_areas=len(context),areas=n,municipalities=int(c.municipality_code.nunique()),states=len(states),reporting_contexts=int(c.cnpj.nunique()),excluded_states=list(map(str,valid[~valid].index)),df=df,scope='same geographical sample for every cause; zeros retained',cluster_sizes=[int(np.unique(g).size) for g in groups],diagnostics=diagnostics,contrast_covariance_checks=checks,inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
(M/'analysis.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({k:v for k,v in meta.items() if k not in ['diagnostics','inputs']},indent=2))
print(omnibus.to_string(index=False))
print(pd.DataFrame(gradients).round(4).to_string(index=False))
print(pairs[pairs.p_holm_45.lt(.05)].round(5).to_string(index=False))
