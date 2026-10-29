"""Cause-resolved annual burden gradients on the audited common Brazilian sample."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

P=Path(__file__).resolve().parents[1]
ROOT=P.parents[1]
D=P/'data'
rng=np.random.default_rng(20261002)
z=pd.read_parquet(D/'annual_reconciliation_2021.parquet')
z=z[z.base_strict_cohort & z.full_12_month_cube].copy().reset_index(drop=True)
causes=pd.read_parquet(D/'cause_annual_2021.parquet')
causes=causes.merge(z[['conjunto','distribuidora','mean_customers']],on=['conjunto','distribuidora'],validate='many_to_one')
wide=causes.pivot(index=['conjunto','distribuidora'],columns='cause_group',values='customer_hours').fillna(0)
wide=wide.div(z.set_index(['conjunto','distribuidora']).mean_customers,axis=0)
z=z.merge(wide.reset_index(),on=['conjunto','distribuidora'],validate='one_to_one')
native_groups=list(wide.columns)
z['All causes']=z[native_groups].sum(axis=1)
assert np.allclose(z['All causes'],z.H_cube_annual)
z['Remaining unplanned or uncertain']=z['All causes']-z['Planned']-z['Environment-associated']
components=['Environment-associated','Planned','Remaining unplanned or uncertain']
outcomes=['All causes']+components
z['log_income']=np.log(z.gdp_pc_brl)
z['density']=z.mean_customers/z.service_area_km2
z['log_density']=np.log(z.density)
municipalities=z.drop_duplicates('municipality_code')
cuts=municipalities.log_income.quantile([.25,.5,.75]).to_numpy()
delta=float(cuts[-1]-cuts[0])
z['income_quartile']=pd.cut(z.log_income,[-np.inf,*cuts,np.inf],labels=['Q1','Q2','Q3','Q4'])
# IBGE municipality-code prefixes determine the state; no company is a display unit.
states={'11':'RO','12':'AC','13':'AM','14':'RR','15':'PA','16':'AP','17':'TO',
        '21':'MA','22':'PI','23':'CE','24':'RN','25':'PB','26':'PE','27':'AL','28':'SE','29':'BA',
        '31':'MG','32':'ES','33':'RJ','35':'SP','41':'PR','42':'SC','43':'RS',
        '50':'MS','51':'MT','52':'GO','53':'DF'}
z['state']=z.municipality_code.astype(str).str[:2].map(states)
assert z.state.notna().all()
z.to_parquet(D/'burden_economic_common_2021.parquet',index=False)
z.to_csv(D/'burden_economic_common_2021.csv',index=False)

results=[];diagnostics=[]
for sample, x in [('Primary common support',z),('Burden agreement within 10%',z[z.burden_concordant_10pct])]:
    for spec in ['Economic context','Density adjusted','Within provider + density']:
        X=pd.DataFrame({'const':1.,'log_income':x.log_income},index=x.index)
        if spec!='Economic context': X['log_density']=x.log_density
        if spec.startswith('Within'):
            X=pd.concat([X,pd.get_dummies(x.distribuidora,prefix='provider',drop_first=True,dtype=float)],axis=1)
        assert np.linalg.matrix_rank(X)==X.shape[1]
        for outcome in outcomes:
            y=x[outcome]
            fit=sm.GLM(y,X,family=sm.families.Poisson()).fit(maxiter=200,tol=1e-9)
            covariance=cov_cluster_2groups(fit,pd.factorize(x.municipality_code)[0],pd.factorize(x.distribuidora)[0])[0]
            df=min(x.municipality_code.nunique(),x.distribuidora.nunique())-1
            j=X.columns.get_loc('log_income')
            variance=float(covariance[j,j]);beta=float(fit.params.log_income)
            se=np.sqrt(variance) if variance>=0 else np.nan
            lo,hi=beta-t.ppf(.975,df)*se,beta+t.ppf(.975,df)*se
            results.append(dict(sample=sample,specification=spec,outcome=outcome,n_areas=len(x),
                municipalities=x.municipality_code.nunique(),provider_clusters=x.distribuidora.nunique(),
                zero_areas=int(y.eq(0).sum()),beta=beta,se=se,lower_beta=lo,upper_beta=hi,
                income_iqr_log=delta,contrast_pct=100*np.expm1(delta*beta),
                lower_pct=100*np.expm1(delta*lo),upper_pct=100*np.expm1(delta*hi),
                p_value=float(2*t.sf(abs(beta/se),df)),converged=bool(fit.converged)))
            diagnostics.append(dict(sample=sample,specification=spec,outcome=outcome,
                design_rank=np.linalg.matrix_rank(X),n_columns=X.shape[1],
                max_score=float(np.max(np.abs(fit.model.score(fit.params)))),
                fitted_sum=float(fit.fittedvalues.sum()),observed_sum=float(y.sum()),
                pearson_dispersion=float(fit.pearson_chi2/fit.df_resid),cluster_variance=variance))

est=pd.DataFrame(results)
est.to_csv(D/'economic_burden_models.csv',index=False)
pd.DataFrame(diagnostics).to_csv(D/'economic_model_diagnostics.csv',index=False)
assert est.converged.all() and est.se.notna().all()

descriptive=[]
for weighting in ['Equal service areas','Customer weighted']:
    for q,x in z.groupby('income_quartile',observed=True):
        weights=np.ones(len(x)) if weighting.startswith('Equal') else x.mean_customers.to_numpy()
        for cause in outcomes+native_groups:
            if cause in components and cause in native_groups: continue
            descriptive.append(dict(weighting=weighting,quartile=str(q),cause=cause,
                hours=np.average(x[cause],weights=weights),areas=len(x),
                municipalities=x.municipality_code.nunique(),mean_customers=x.mean_customers.sum()))
        # Exhaustive main components are retained exactly once.
        for cause in ['Environment-associated','Planned']:
            descriptive.append(dict(weighting=weighting,quartile=str(q),cause=cause,
                hours=np.average(x[cause],weights=weights),areas=len(x),
                municipalities=x.municipality_code.nunique(),mean_customers=x.mean_customers.sum()))
pd.DataFrame(descriptive).to_csv(D/'quartile_burden_components.csv',index=False)

# The GDP rank is fixed before resampling; municipality clusters preserve shared assignment.
q1=z.income_quartile.eq('Q1').to_numpy();q4=z.income_quartile.eq('Q4').to_numpy()
mcodes,uniques=pd.factorize(z.municipality_code)
values=z[outcomes].to_numpy()
boot=[]
for _ in range(2000):
    weights=np.bincount(rng.integers(0,len(uniques),len(uniques)),minlength=len(uniques))[mcodes]
    if weights[q1].sum()==0 or weights[q4].sum()==0: continue
    boot.append(np.average(values[q1],weights=weights[q1],axis=0)-np.average(values[q4],weights=weights[q4],axis=0))
boot=np.asarray(boot)
point=values[q1].mean(axis=0)-values[q4].mean(axis=0)
decomp=[]
for k,outcome in enumerate(outcomes):
    ci=np.quantile(boot[:,k],[.025,.975])
    decomp.append(dict(cause=outcome,low_minus_high_h=float(point[k]),lower_h=ci[0],upper_h=ci[1],
                       fraction_of_gap=point[k]/point[0] if point[0]!=0 else np.nan,
                       bootstrap_unit='municipality',replicates=len(boot)))
assert np.isclose(point[1:].sum(),point[0])
pd.DataFrame(decomp).to_csv(D/'additive_burden_gap.csv',index=False)

region=[]
for state,x in z.groupby('state'):
    w=x.mean_customers
    for cause in outcomes+native_groups:
        region.append(dict(state=state,cause=cause,areas=len(x),municipalities=x.municipality_code.nunique(),
            customers=w.sum(),hours_area_equal=x[cause].mean(),hours_customer_weighted=np.average(x[cause],weights=w)))
pd.DataFrame(region).drop_duplicates(['state','cause']).to_csv(D/'state_burden_components.csv',index=False)

summary=dict(areas=len(z),municipalities=z.municipality_code.nunique(),states=z.state.nunique(),
    providers=z.distribuidora.nunique(),customers=float(z.mean_customers.sum()),
    primary_year=2021,gdp_p25_brl=float(np.exp(cuts[0])),gdp_p75_brl=float(np.exp(cuts[-1])),
    total_customer_hours=float(z.customer_hours.sum()),
    environmental_customer_hours=float((z['Environment-associated']*z.mean_customers).sum()),
    environmental_share=float((z['Environment-associated']*z.mean_customers).sum()/z.customer_hours.sum()),
    primary_mean_burden_h=float(z['All causes'].mean()),
    economic_interpretation='Regional GDP context; not household poverty or a causal effect',
    frequency_duration_status='No cause frequency or sustained-interruption duration model admitted from the aggregate cube')
(P/'metadata/burden_results.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print(est[est['sample'].eq('Primary common support')][['specification','outcome','contrast_pct','lower_pct','upper_pct']].round(2).to_string(index=False))
print(pd.DataFrame(decomp).round(3).to_string(index=False))
