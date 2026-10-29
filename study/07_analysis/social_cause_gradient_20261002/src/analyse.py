"""Joint stage gradients, declared support and complete exploratory estimates."""
from pathlib import Path
import json, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import block_diag
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

P=Path(__file__).resolve().parents[1]; D=P/'data'; M=P/'metadata'
ctx=pd.read_parquet(D/'analysis_context.parquet').sort_values('IdeConjunto').reset_index(drop=True)
reg=pd.read_parquet(D/'regulatory_stages.parquet')
fine=pd.read_parquet(D/'cause_stages.parquet')
rows=[];diags=[];predictions=[]

def joint_models(yf,yh,X,c):
    fits=[];scores=[];breads=[];messages=[]
    for y in [yf,yh]:
        with warnings.catch_warnings(record=True) as caught:
            fit=sm.GLM(np.asarray(y),X,family=sm.families.Poisson()).fit(maxiter=300,tol=1e-10)
        fits.append(fit);messages.extend(str(w.message) for w in caught)
        mu=np.asarray(fit.fittedvalues)
        scores.append(np.asarray(X)*(np.asarray(y)-mu)[:,None])
        breads.append(np.linalg.inv(np.asarray(X).T@(mu[:,None]*np.asarray(X))))
    score=np.column_stack(scores);bread=block_diag(*breads);n,k=X.shape
    gs=[pd.factorize(c.municipality_code)[0],pd.factorize(c.cnpj)[0],
        pd.factorize(pd.MultiIndex.from_arrays([c.municipality_code,c.cnpj]))[0]]
    meats=[]
    for g in gs:
        agg=pd.DataFrame(score).groupby(g).sum().to_numpy();G=len(agg)
        assert G>1 and n>k
        meats.append(agg.T@agg*(G/(G-1))*((n-1)/(n-k)))
    cov=bread@(meats[0]+meats[1]-meats[2])@bread
    return fits,np.r_[fits[0].params,fits[1].params],cov,messages

def estimate(domain,category,yf,yh,base,predictor,scale,label):
    base=base.reset_index(drop=True);yf=np.asarray(yf);yh=np.asarray(yh)
    assert np.isfinite(yf).all() and np.isfinite(yh).all()
    assert min(yf)>=0 and min(yh)>=0 and np.sum(yf)>0 and np.sum(yh)>0
    for spec in ['Unadjusted','Density adjusted','State + density']:
        use=np.ones(len(base),dtype=bool)
        removed=[]
        if spec=='State + density':
            sums=pd.DataFrame({'state':base.state_code,'f':yf,'h':yh}).groupby('state')[['f','h']].sum()
            removed=sums[(sums.f==0)|(sums.h==0)].index.tolist()
            use=~base.state_code.isin(removed).to_numpy()
        c=base.loc[use].reset_index(drop=True)
        X=pd.DataFrame({'const':np.ones(len(c)),predictor:c[predictor]})
        if spec!='Unadjusted':X['log_density']=c.log_density
        if spec=='State + density':X=pd.concat([X,pd.get_dummies(c.state_code,prefix='state',drop_first=True,dtype=float)],axis=1)
        assert np.linalg.matrix_rank(X)==X.shape[1]
        fits,pars,cov,ws=joint_models(yf[use],yh[use],X,c)
        k=X.shape[1];j=X.columns.get_loc(predictor)
        df=min(c.municipality_code.nunique(),c.cnpj.nunique())-1
        meta=dict(domain=domain,category=category,predictor=predictor,contrast_label=label,
            contrast_scale=scale,specification=spec,areas=len(c),municipalities=c.municipality_code.nunique(),
            reporting_contexts=c.cnpj.nunique(),states=c.state_code.nunique(),df=df,
            zero_frequency_areas=int((yf[use]==0).sum()),removed_all_zero_states=';'.join(removed))
        for metric,weights in [('Frequency',[(j,1)]),('Annual burden',[(k+j,1)]),
            ('Conditional duration',[(k+j,1),(j,-1)]),('Frequency versus duration',[(j,2),(k+j,-1)])]:
            L=np.zeros(2*k)
            for ix,w in weights:L[ix]=w
            beta=float(L@pars);var=float(L@cov@L)
            se=np.sqrt(var) if var>=0 else np.nan;critical=t.ppf(.975,df)
            rows.append(dict(**meta,metric=metric,beta=beta,se=se,
                contrast_pct=100*np.expm1(scale*beta),
                lower_pct=100*np.expm1(scale*(beta-critical*se)),
                upper_pct=100*np.expm1(scale*(beta+critical*se)),
                p_value=2*t.sf(abs(beta/se),df) if se>0 else np.nan,valid_variance=bool(var>=0)))
        diags.append(dict(**meta,parameters=k,design_rank=int(np.linalg.matrix_rank(X)),
            condition_number=float(np.linalg.cond(X)),converged=all(f.converged for f in fits),
            max_score=max(float(np.max(abs(f.model.score(f.params))))/max(float(np.sum(f.model.endog)),1) for f in fits),
            dispersion_frequency=float(fits[0].pearson_chi2/fits[0].df_resid),
            dispersion_burden=float(fits[1].pearson_chi2/fits[1].df_resid),warnings=' | '.join(ws)))
        for i,aid in enumerate(c.IdeConjunto):
            predictions.append(dict(**meta,IdeConjunto=int(aid),observed_frequency=float(yf[use][i]),
                observed_burden=float(yh[use][i]),predicted_frequency=float(fits[0].fittedvalues.iloc[i]),
                predicted_burden=float(fits[1].fittedvalues.iloc[i])))

for category,g in reg.groupby('category'):
    c=ctx.merge(g[['IdeConjunto','frequency','burden_hours']],on='IdeConjunto',validate='one_to_one')
    for predictor,scale,label in [('disadvantage_10pp',1.,'+10 percentage points low-income share'),
                                  ('log_gdp',np.log(2),'Doubling municipal GDP per person'),
                                  ('log_median_income',np.log(2),'Doubling median household income per person')]:
        estimate('Customer indices',category,c.frequency,c.burden_hours,c,predictor,scale,label)

total=fine.groupby('IdeConjunto').agg(record_hours=('record_hours','sum'))
for cause in ['All records','Wind','Lightning','Vegetation','Flood','Operational error','Planned']:
    if cause=='All records':
        g=ctx[['IdeConjunto','records_reference','customer_years']].merge(total,on='IdeConjunto',validate='one_to_one')
        g['record_rate']=g.records_reference/g.customer_years;g['record_hours_rate']=g.record_hours/g.customer_years
    else:g=fine[fine.cause.eq(cause)]
    c=ctx.merge(g[['IdeConjunto','record_rate','record_hours_rate']],on='IdeConjunto',validate='one_to_one')
    estimate('Source records',cause,1000*c.record_rate,1000*c.record_hours_rate,c,
             'disadvantage_10pp',1.,'+10 percentage points low-income share')

res=pd.DataFrame(rows);diag=pd.DataFrame(diags)
res['p_adjusted_bh']=np.nan;res['multiplicity_family']=''
family=(res.domain.eq('Source records')&res.specification.eq('Density adjusted')&~res.category.eq('All records'))
for metrics,label in [(['Frequency','Annual burden','Conditional duration'],'18 named-cause stage contrasts'),
                      (['Frequency versus duration'],'6 named-cause direct stage comparisons')]:
    use=family&res.metric.isin(metrics)&res.p_value.notna()
    res.loc[use,'p_adjusted_bh']=multipletests(res.loc[use,'p_value'],method='fdr_bh')[1]
    res.loc[use,'multiplicity_family']=label
res.to_csv(D/'stage_gradients.csv',index=False);res.to_parquet(D/'stage_gradients.parquet',index=False)
diag.to_csv(D/'model_diagnostics.csv',index=False)
pd.DataFrame(predictions).to_parquet(D/'model_predictions.parquet',index=False)
assert diag.converged.all()
assert res.valid_variance.all()
assert diag.max_score.max()<1e-5
audit=dict(created_utc=datetime.now(timezone.utc).isoformat(),joint_model_pairs=len(diag),
    outcome_contrasts=len(res),all_converged=bool(diag.converged.all()),all_variances_nonnegative=bool(res.valid_variance.all()),
    largest_relative_score=float(diag.max_score.max()),covariance='two-way municipality and reporting-context clustered',
    stage_comparison='beta_frequency - (beta_burden - beta_frequency); joint cross-equation covariance',
    low_income_definition='2022 household monthly income per person <= R$606, including no income',
    population_scope='regional ecological association; not person-linked outage observations')
(M/'model_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
print(res[(res.predictor.eq('disadvantage_10pp'))&(res.specification.eq('Density adjusted'))][
    ['domain','category','metric','contrast_pct','lower_pct','upper_pct','p_adjusted_bh']].round(3).to_string(index=False))
