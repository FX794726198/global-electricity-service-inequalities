"""Prespecified exploratory cause, composition and regional economic contrasts."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import block_diag
from scipy.stats import t

P=Path(__file__).resolve().parents[1];D=P/'data';M=P/'metadata'
rng=np.random.default_rng(2026100214)
ctx=pd.read_parquet(D/'economic_cohort.parquet').sort_values('IdeConjunto').reset_index(drop=True)
cells=pd.read_parquet(D/'fine_cause_cells.parquet')
cells=cells[cells.IdeConjunto.isin(ctx.IdeConjunto)].copy()
annual=cells.groupby(['IdeConjunto','cause']).agg(records=('record_count_lower','sum'),
    record_hours=('record_hours','sum')).reset_index()
extra=pd.read_csv(D/'zero_duration_allocation_bounds.csv')
extra=extra[extra.exact_duration_transport&extra.IdeConjunto.isin(ctx.IdeConjunto)]
extra=extra.groupby(['IdeConjunto','cause']).unresolved_count.sum().reset_index(name='extra_zero_upper')
annual=annual.merge(extra,on=['IdeConjunto','cause'],how='left',validate='one_to_one')
annual['extra_zero_upper']=annual.extra_zero_upper.fillna(0)
annual['duration_upper_h']=annual.record_hours/annual.records
annual['duration_lower_h']=annual.record_hours/(annual.records+annual.extra_zero_upper)
annual=annual.merge(ctx[['IdeConjunto','municipality_code','state_code','cnpj','customer_years','gdp_pc_brl','density']],
    on='IdeConjunto',validate='many_to_one')
annual['record_rate']=annual.records/annual.customer_years
annual['record_hours_rate']=annual.record_hours/annual.customer_years
annual.to_parquet(D/'area_cause_stages.parquet',index=False)
annual.to_csv(D/'area_cause_stages.csv',index=False)
native=annual.groupby('cause').agg(records=('records','sum'),record_hours=('record_hours','sum'),
    possible_zero_additions=('extra_zero_upper','sum'),areas=('IdeConjunto','nunique')).reset_index()
native['record_share_pct']=100*native.records/native.records.sum()
native['hour_share_pct']=100*native.record_hours/native.record_hours.sum()
native['duration_upper_h']=native.record_hours/native.records
native['duration_lower_h']=native.record_hours/(native.records+native.possible_zero_additions)
native.to_csv(D/'common_cause_summary.csv',index=False)

ctx['log_income']=np.log(ctx.gdp_pc_brl)
ctx['log_density']=np.log(ctx.density)
quants=ctx.drop_duplicates('municipality_code').log_income.quantile([.25,.5,.75]).to_numpy()
delta=quants[-1]-quants[0]
ctx['income_quartile']=pd.cut(ctx.log_income,[-np.inf,*quants,np.inf],labels=['Q1','Q2','Q3','Q4'])
ctx.to_parquet(D/'model_context.parquet',index=False)
ctx.to_csv(D/'model_context.csv',index=False)
causes=sorted(annual.cause.unique())
ix=pd.MultiIndex.from_product([ctx.IdeConjunto,causes],names=['IdeConjunto','cause'])
full=annual.set_index(['IdeConjunto','cause'])[['records','record_hours','extra_zero_upper']].reindex(ix,fill_value=0).reset_index()
full=full.merge(ctx[['IdeConjunto','customer_years']],on='IdeConjunto',validate='many_to_one')
R=full.pivot(index='IdeConjunto',columns='cause',values='records').loc[ctx.IdeConjunto]
H=full.pivot(index='IdeConjunto',columns='cause',values='record_hours').loc[ctx.IdeConjunto]
upper_extra=full.pivot(index='IdeConjunto',columns='cause',values='extra_zero_upper').loc[ctx.IdeConjunto]
den=ctx.customer_years.to_numpy()
totR=ctx.records_reference.to_numpy()
assert np.all(totR-R.sum(axis=1).to_numpy()>=0)
assert np.allclose(totR-R.sum(axis=1).to_numpy(),ctx.unresolved_count)

def joint_models(y_rate,y_hours,X,xcontext):
    fits=[];scores=[];breads=[]
    for y in [y_rate,y_hours]:
        with warnings.catch_warnings(record=True) as caught:
            fit=sm.GLM(np.asarray(y),X,family=sm.families.Poisson()).fit(maxiter=250,tol=1e-9)
        fits.append(fit)
        mu=np.asarray(fit.fittedvalues)
        scores.append(np.asarray(X)*(np.asarray(y)-mu)[:,None])
        breads.append(np.linalg.inv(np.asarray(X).T@(mu[:,None]*np.asarray(X))))
    score=np.column_stack(scores);bread=block_diag(*breads)
    groups=[pd.factorize(xcontext.municipality_code)[0],pd.factorize(xcontext.cnpj)[0],
            pd.factorize(pd.MultiIndex.from_arrays([xcontext.municipality_code,xcontext.cnpj]))[0]]
    n,k=X.shape;meats=[]
    for g in groups:
        agg=pd.DataFrame(score).groupby(g).sum().to_numpy();G=len(agg)
        meats.append(agg.T@agg*(G/(G-1))*((n-1)/(n-k)))
    covariance=bread@(meats[0]+meats[1]-meats[2])@bread
    return fits,np.r_[fits[0].params,fits[1].params],covariance

rows=[];diagnostics=[]
focus=['All records','Wind','Lightning','Vegetation','Flood','Operational error','Planned']
for cause in focus:
    n=totR if cause=='All records' else R[cause].to_numpy()
    h=H.sum(axis=1).to_numpy() if cause=='All records' else H[cause].to_numpy()
    for spec in ['Economic context','Density adjusted','Within context + density']:
        X=pd.DataFrame({'const':1.,'log_income':ctx.log_income},index=ctx.index)
        if spec!='Economic context':X['log_density']=ctx.log_density
        if spec.startswith('Within'):X=pd.concat([X,pd.get_dummies(ctx.cnpj,prefix='context',drop_first=True,dtype=float)],axis=1)
        # Contexts with no records of a specific cause offer no within-cause rate information.
        # Retain them in the first two unconditional models; exclude all-zero fixed-effect
        # contexts from the within-context model to avoid a nonexistent finite intercept.
        use=np.ones(len(ctx),dtype=bool)
        if spec.startswith('Within'):
            active=pd.Series(n).groupby(ctx.cnpj).sum()
            use=ctx.cnpj.isin(active[active>0].index).to_numpy()
            X=X.loc[use].copy()
            X=X.loc[:,~(X.eq(0).all(axis=0))]
            # Dropping zero contexts may remove the initial reference group.
            inds=[c for c in X if c.startswith('context_')]
            if inds and np.allclose(X[inds].sum(axis=1),1):X=X.drop(columns=inds[0])
        xctx=ctx.loc[use].copy()
        assert np.linalg.matrix_rank(X)==X.shape[1]
        fits,params,cov=joint_models(n[use]/den[use],h[use]/den[use],X,xctx)
        j=X.columns.get_loc('log_income');k=X.shape[1]
        df=min(xctx.municipality_code.nunique(),xctx.cnpj.nunique())-1
        for metric,weights in [('Record frequency',[(j,1)]),('Record-hour burden',[(k+j,1)]),
                               ('Mean record duration',[(k+j,1),(j,-1)])]:
            L=np.zeros(2*k)
            for idx,v in weights:L[idx]=v
            beta=float(L@params);var=float(L@cov@L)
            se=np.sqrt(var) if var>=0 else np.nan
            lo,hi=beta-t.ppf(.975,df)*se,beta+t.ppf(.975,df)*se
            rows.append(dict(cause=cause,specification=spec,metric=metric,areas=len(xctx),
                municipalities=xctx.municipality_code.nunique(),contexts=xctx.cnpj.nunique(),
                positive_cause_areas=int((n[use]>0).sum()),zero_cause_areas=int((n[use]==0).sum()),
                beta=beta,se=se,contrast_pct=100*np.expm1(beta*delta),
                lower_pct=100*np.expm1(lo*delta),upper_pct=100*np.expm1(hi*delta),
                p_value=2*t.sf(abs(beta/se),df) if se>0 else np.nan,
                log_gdp_iqr=delta,df=df,valid_variance=var>=0))
        diagnostics.append(dict(cause=cause,specification=spec,areas=len(xctx),parameters=k,
            converged=all(f.converged for f in fits),design_rank=int(np.linalg.matrix_rank(X)),
            max_relative_score=max(float(np.max(abs(f.model.score(f.params))))/max(float(np.sum(f.model.endog)),1) for f in fits),
            dispersion_rate=float(fits[0].pearson_chi2/fits[0].df_resid),
            dispersion_hours=float(fits[1].pearson_chi2/fits[1].df_resid)))
est=pd.DataFrame(rows);est.to_csv(D/'economic_stage_models.csv',index=False)
diag=pd.DataFrame(diagnostics);diag.to_csv(D/'economic_model_diagnostics.csv',index=False)
assert diag.converged.all()

# Exact symmetric decomposition on six exhaustive common-support cause groups.
fixed=['Wind','Lightning','Vegetation','Operational error','Planned']
R6=R[fixed].copy();H6=H[fixed].copy()
R6['Remaining causes']=totR-R6.sum(axis=1);H6['Remaining causes']=H.sum(axis=1)-H6.sum(axis=1)
r=R6.to_numpy()/den[:,None];h=H6.to_numpy()/den[:,None]
q1=ctx.income_quartile.eq('Q1').to_numpy();q4=ctx.income_quartile.eq('Q4').to_numpy()
def decompose(weights):
    rs=[np.average(r[q],weights=weights[q],axis=0) for q in [q1,q4]]
    hs=[np.average(h[q],weights=weights[q],axis=0) for q in [q1,q4]]
    ss=[a/a.sum() for a in rs];ds=[b/a for a,b in zip(rs,hs)]
    if not all(np.isfinite(v).all() for v in ds):return None
    mix=(ss[0]-ss[1])*(ds[0]+ds[1])/2
    within=(ds[0]-ds[1])*(ss[0]+ss[1])/2
    gap=hs[0].sum()/rs[0].sum()-hs[1].sum()/rs[1].sum()
    assert np.isclose(gap,mix.sum()+within.sum())
    return np.r_[gap,mix.sum(),within.sum(),mix,within]
point=decompose(np.ones(len(ctx)))
mc,munis=pd.factorize(ctx.municipality_code)
draws=[]
for _ in range(2000):
    w=np.bincount(rng.integers(0,len(munis),len(munis)),minlength=len(munis))[mc]
    if min(w[q1].sum(),w[q4].sum())==0:continue
    v=decompose(w)
    if v is not None:draws.append(v)
draws=np.array(draws);lo,hi=np.quantile(draws,[.025,.975],axis=0)
names=['Total duration gap','Cause composition','Within-cause duration']+[f'Composition: {c}' for c in R6]+[f'Within cause: {c}' for c in R6]
decomp=pd.DataFrame({'component':names,'low_minus_high_h':point,'lower_h':lo,'upper_h':hi})
decomp.to_csv(D/'duration_gap_decomposition.csv',index=False)
qs=[]
for q in ['Q1','Q2','Q3','Q4']:
    keep=ctx.income_quartile.eq(q).to_numpy()
    for cause in ['All records']+list(R6):
        rr=totR/den if cause=='All records' else R6[cause].to_numpy()/den
        hh=H.sum(axis=1).to_numpy()/den if cause=='All records' else H6[cause].to_numpy()/den
        qs.append(dict(quartile=q,cause=cause,areas=int(keep.sum()),
            records_per_1000_customer_year=1000*rr[keep].mean(),record_hours_per_1000_customer_year=1000*hh[keep].mean(),
            mean_record_duration_h=hh[keep].mean()/rr[keep].mean()))
pd.DataFrame(qs).to_csv(D/'quartile_source_record_stages.csv',index=False)

# Paired differences across areas, preserving municipality dependence.
paired=[];individual=[]
for a,b in [('Wind','Lightning'),('Wind','Operational error'),('Lightning','Operational error'),
            ('Vegetation','Operational error'),('Flood','Operational error')]:
    da=H[a]/R[a].where(R[a]>0);db=H[b]/R[b].where(R[b]>0)
    mask=(da>0)&(db>0)
    lr=np.log((da/db)[mask]).to_numpy();mx=ctx.loc[mask.to_numpy()].municipality_code
    codes,units=pd.factorize(mx);sim=[]
    for _ in range(2000):
        w=np.bincount(rng.integers(0,len(units),len(units)),minlength=len(units))[codes]
        sim.append(np.exp(np.average(lr,weights=w)))
    l,u=np.quantile(sim,[.025,.975])
    paired.append(dict(numerator=a,denominator=b,areas=len(lr),municipalities=len(units),
        geometric_ratio=np.exp(lr.mean()),lower=l,upper=u,share_ratio_above_one=(lr>0).mean()))
    for cid,v in zip(R.index[mask],np.exp(lr)):
        individual.append(dict(IdeConjunto=cid,numerator=a,denominator=b,ratio=v,
            numerator_records=R.loc[cid,a],denominator_records=R.loc[cid,b]))
pd.DataFrame(paired).to_csv(D/'within_area_duration_contrasts.csv',index=False)
pd.DataFrame(individual).to_csv(D/'within_area_duration_ratios.csv',index=False)
state=annual.groupby(['state_code','cause']).agg(records=('records','sum'),record_hours=('record_hours','sum'),
    areas=('IdeConjunto','nunique')).reset_index()
state['duration_h']=state.record_hours/state.records
state.to_csv(D/'state_cause_duration.csv',index=False)

# Voltage composition is a separate standardized comparison, with support reported.
v=cells.groupby(['IdeConjunto','cause','voltage_group']).agg(records=('record_count_lower','sum'),
    hours=('record_hours','sum')).reset_index()
v=v.merge(ctx[['IdeConjunto','income_quartile','customer_years']],on='IdeConjunto',validate='many_to_one')
v['rate']=v.records/v.customer_years;v['hours_rate']=v.hours/v.customer_years
g=v[v.income_quartile.isin(['Q1','Q4'])].groupby(['income_quartile','cause','voltage_group'],observed=True)[['rate','hours_rate','records']].sum()
pv=g.unstack('income_quartile')
good=(pv['rate','Q1']>0)&(pv['rate','Q4']>0)
d1=pv.loc[good,('hours_rate','Q1')]/pv.loc[good,('rate','Q1')]
d4=pv.loc[good,('hours_rate','Q4')]/pv.loc[good,('rate','Q4')]
ww=pv.loc[good,('rate','Q1')]/q1.sum()+pv.loc[good,('rate','Q4')]/q4.sum();ww=ww/ww.sum()
voltage=dict(common_cause_voltage_cells=int(good.sum()),total_cells=len(good),
    fixed_composition_low_minus_high_h=float(np.sum(ww*(d1-d4))),
    q1_record_support=float(pv.loc[good,('records','Q1')].sum()/pv['records','Q1'].sum()),
    q4_record_support=float(pv.loc[good,('records','Q4')].sum()/pv['records','Q4'].sum()),
    interpretation='Descriptive common-support standardization, not a causal mechanism or a threshold-matched customer duration')
(M/'voltage_composition_diagnostic.json').write_text(json.dumps(voltage,indent=2))
summary=dict(areas=len(ctx),municipalities=ctx.municipality_code.nunique(),states=ctx.state_code.nunique(),
    contexts=ctx.cnpj.nunique(),source_records=int(totR.sum()),record_hours=float(H.to_numpy().sum()),
    records_with_unresolved_zero_cause_allocation=int(ctx.unresolved_count.sum()),
    gdp_p25_brl=float(np.exp(quants[0])),gdp_p75_brl=float(np.exp(quants[-1])),
    bootstrap_draws=len(draws),source_duration_weighting='Record weighted',
    supported_interpretation='Geographic GDP and source-record stages; not household poverty, customer SAIFI, customer-hours or direct recovery')
(M/'analysis_results.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print(est[est.cause.isin(['All records','Wind','Lightning'])][['cause','specification','metric','contrast_pct','lower_pct','upper_pct']].round(2).to_string(index=False))
print(decomp.head(3).round(3).to_string(index=False))
print(pd.DataFrame(paired).round(3).to_string(index=False))
