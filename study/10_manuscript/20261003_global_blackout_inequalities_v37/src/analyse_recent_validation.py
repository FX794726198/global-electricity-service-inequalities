"""Reproduce the specified 2021–2023 country/reporting-area validation.
Run with Python 3.12, numpy, pandas, pyarrow, scipy and statsmodels.
No network access, source mutation, imputation or unverified unit conversion.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import t

V=Path(__file__).resolve().parents[1]
PROJECT=V.parents[1]
OLD=V.parent/'20261003_global_blackout_inequalities_v33'
EXT=OLD/'core_reproducibility/upbeat_extension'
OUT=V/'analysis/recent_validation'; DATA=V/'source_data'
OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(exist_ok=True)
B=40000; SEED=2026100301
rng=np.random.default_rng(SEED)
INPUT=EXT/'inputs/utility_year_all.parquet'
d=pd.read_parquet(INPUT)
r=d[d.year.between(2021,2023)].copy()
r['v37_admitted']=r.admission_reason.eq('paired_eligible_before_minimum_years') & ~r.utility.eq('MEPCO')
r['v37_decision']=np.where(r.utility.eq('MEPCO'),'quarantined_regulator_unit_and_scope_conflict',r.admission_reason)
# Source identity resolution: Codensa is Enel Colombia's predecessor distribution business.
# Keep the contemporaneous 2021 archived value (487 min, rounded in the archive to
# 8.1167 h). The later retrospective 488-min value is a vintage sensitivity, not
# an additional reporting area. Preserve raw labels and exclude only its duplicate row.
r['source_record_id']=r.utility
r['canonical_source_area_id']=r.utility.replace({'CODENSA':'Enel Colombia'})
alias_duplicate=r.utility.eq('Enel Colombia') & r.year.eq(2021)
assert alias_duplicate.sum()==1
r.loc[alias_duplicate,'v37_admitted']=False
r.loc[alias_duplicate,'v37_decision']='excluded_duplicate_predecessor_scope_retrospective_2021_vintage'
r['identity_resolution']=np.where(r.utility.eq('CODENSA'),'predecessor_CODENSA_to_Enel_Colombia',np.where(alias_duplicate,'retrospective_duplicate_2021','unchanged'))
r.rename(columns={'burden':'annual_hours','utility':'source_area_id','utility_display':'source_area_name'}).to_csv(DATA/'recent_source_row_admission.csv',index=False)
e=r[r.v37_admitted].copy().rename(columns={'burden':'annual_hours','utility':'source_area_id','utility_display':'source_area_name'})
e['source_area_id']=e.canonical_source_area_id
assert not e.duplicated(['source_area_id','year']).any()
assert e[['frequency','annual_hours','gdp']].gt(0).all().all()
assert e.quality_admitted.all() and (~e.cross_snapshot_duration_flag).all()
assert np.allclose(e.annual_hours,e.frequency*e.duration)
# Reporting type D/GTD etc retains distribution metrics; never transmission indices.
assert len(e)==95 and e.country_iso3.nunique()==31 and e.source_area_id.nunique()==44
cols=['source_area_id','source_record_id','source_area_name','identity_resolution','year','country_iso3','country','region','utility_type','frequency','annual_hours','duration','gdp','income_at_year']
e[cols].assign(frequency_unit='interruptions/customer-year',annual_hours_unit='hours/customer-year',duration_unit='hours/interruption',minimum_duration='unknown',planned_inclusion='unknown',load_shedding_inclusion='unknown',major_event_inclusion='unknown').to_csv(DATA/'recent_admitted_source_years.csv',index=False)

def country_year(z):
    # F and H use identical rows; duration is derived after aggregating.
    assert z.groupby(['country_iso3','year']).gdp.nunique().eq(1).all()
    c=z.groupby(['country_iso3','country','region','year'],as_index=False).agg(frequency=('frequency','mean'),annual_hours=('annual_hours','mean'),gdp=('gdp','first'),source_area_count=('source_area_id','nunique'),source_area_ids=('source_area_id',lambda q:';'.join(sorted(q))))
    c['duration']=c.annual_hours/c.frequency
    return c

def quartiles(g):
    g=g.sort_values(['gdp','country_iso3']).copy()
    g['quartile']=pd.qcut(np.arange(len(g)),4,labels=['Q1','Q2','Q3','Q4']).astype(str)
    return g

cy=country_year(e)
annual=pd.concat([quartiles(g) for y,g in cy.groupby('year')],ignore_index=True)
annual.to_csv(DATA/'recent_country_year_available.csv',index=False)
common_ids=cy.groupby('country_iso3').year.nunique().loc[lambda x:x.eq(3)].index
common=cy[cy.country_iso3.isin(common_ids)].copy()
common_groups=quartiles(common.groupby('country_iso3',as_index=False).gdp.mean())[['country_iso3','gdp','quartile']].rename(columns={'gdp':'group_gdp_mean_2021_2023'})
common=common.merge(common_groups,on='country_iso3',validate='many_to_one')
common.to_csv(DATA/'recent_common_country_years.csv',index=False)
strict_ids=e.groupby('source_area_id').year.nunique().loc[lambda x:x.eq(3)].index
strict_raw=e[e.source_area_id.isin(strict_ids)].copy()
strict=country_year(strict_raw)
strict_groups=quartiles(strict.groupby('country_iso3',as_index=False).gdp.mean())[['country_iso3','gdp','quartile']].rename(columns={'gdp':'group_gdp_mean_2021_2023'})
strict=strict.merge(strict_groups,on='country_iso3',validate='many_to_one')
assert strict.country_iso3.nunique()==15 and strict_raw.source_area_id.nunique()==21
strict.to_csv(DATA/'recent_strict_fixed_country_years.csv',index=False)

METRICS=['gap_h','frequency_h','duration_h','frequency_minus_duration_h','low_F','high_F','low_H','high_H','low_D','high_D']
def components(fl,hl,fr,hr):
    dl=hl/fl; dr=hr/fr
    cf=(fl-fr)*(dl+dr)/2; cd=(dl-dr)*(fl+fr)/2
    return np.stack([hl-hr,cf,cd,cf-cd,fl,fr,hl,hr,dl,dr],axis=-1)
summary=[]; groups=[]; draws=[]
def analyse(z,label,year,indices=None):
    l=z[z.quartile.eq('Q1')].sort_values('country_iso3'); u=z[z.quartile.eq('Q4')].sort_values('country_iso3')
    point=components(l.frequency.mean(),l.annual_hours.mean(),u.frequency.mean(),u.annual_hours.mean())
    if indices is None:
        il=rng.integers(len(l),size=(B,len(l))); iu=rng.integers(len(u),size=(B,len(u)))
    else: il,iu=indices
    lf=l.frequency.to_numpy()[il].mean(1); lh=l.annual_hours.to_numpy()[il].mean(1)
    uf=u.frequency.to_numpy()[iu].mean(1); uh=u.annual_hours.to_numpy()[iu].mean(1)
    a=components(lf,lh,uf,uh)
    assert np.allclose(a[:,0],a[:,1]+a[:,2],rtol=1e-10,atol=1e-10)
    rec={'analysis':label,'year':year,'countries':len(z),'low_countries':len(l),'high_countries':len(u),'low_country_iso3':';'.join(l.country_iso3),'high_country_iso3':';'.join(u.country_iso3)}
    for j,k in enumerate(METRICS):
        rec[k]=float(point[j]);rec[k+'_low'],rec[k+'_high']=np.quantile(a[:,j],[.025,.975])
    rec['positive_gap_draw_fraction']=np.mean(a[:,0]>0)
    rec['frequency_greater_than_duration_draw_fraction']=np.mean(a[:,3]>0)
    rec['frequency_share_pct']=100*point[1]/point[0] if point[0]!=0 else np.nan
    rec['frequency_share_interval_reportable']=rec['gap_h_low']>0
    rec['frequency_share_pct_low']=np.nan;rec['frequency_share_pct_high']=np.nan
    if rec['frequency_share_interval_reportable']:
        # Preserve negative denominator draws rather than condition on positive gaps.
        share=np.divide(100*a[:,1],a[:,0],out=np.full(B,np.nan),where=a[:,0]!=0)
        rec['frequency_share_pct_low'],rec['frequency_share_pct_high']=np.nanquantile(share,[.025,.975])
    summary.append(rec)
    for q,g in z.groupby('quartile'):
        groups.append({'analysis':label,'year':year,'quartile':q,'countries':len(g),'country_iso3':';'.join(sorted(g.country_iso3)),'frequency':g.frequency.mean(),'annual_hours':g.annual_hours.mean(),'duration':g.annual_hours.mean()/g.frequency.mean(),'gdp_min':g.gdp.min(),'gdp_max':g.gdp.max()})
    dt=pd.DataFrame(a,columns=METRICS);dt['analysis']=label;dt['year']=year;dt['draw']=np.arange(B);draws.append(dt)
    return a

for year,z in annual.groupby('year'): analyse(z,'available_countries_yearly_groups',int(year))
# Deliver main-panel source first; later code adds sensitivity rows to this same file.
pd.DataFrame(summary).to_csv(DATA/'recent_yearly_gap_components.csv',index=False)

changes=[]
for label,panel in [('common_countries_fixed_groups',common),('strict_fixed15_sources_fixed_groups',strict)]:
    q=panel[panel.year.eq(2021)]
    nl=q.quartile.eq('Q1').sum();nu=q.quartile.eq('Q4').sum()
    idx=(rng.integers(nl,size=(B,nl)),rng.integers(nu,size=(B,nu)))
    joint={}
    for year,z in panel.groupby('year'):joint[int(year)]=analyse(z,label,int(year),idx)
    da=joint[2023]-joint[2021]
    aa=pd.DataFrame(summary)
    p1=aa[(aa.analysis==label)&(aa.year==2021)].iloc[0];p3=aa[(aa.analysis==label)&(aa.year==2023)].iloc[0]
    for j,k in enumerate(METRICS):
        lo,hi=np.quantile(da[:,j],[.025,.975]); changes.append({'analysis':label,'contrast':'2023 minus 2021','quantity':k,'estimate':p3[k]-p1[k],'ci_low':lo,'ci_high':hi,'resampling':'same country trajectory draw reused in all years'})
    mean=panel.groupby(['country_iso3','country','region','quartile'],as_index=False)[['frequency','annual_hours','gdp']].mean()
    mean['duration']=mean.annual_hours/mean.frequency
    analyse(mean,label+'_three_year_mean',0)
# Original unbalanced estimator first averages within reporting source then within country.
m=e.groupby(['source_area_id','country_iso3','country','region'],as_index=False)[['frequency','annual_hours','gdp']].mean()
u=m.groupby(['country_iso3','country','region'],as_index=False)[['frequency','annual_hours','gdp']].mean()
u=quartiles(u);u['duration']=u.annual_hours/u.frequency
analyse(u,'original_unbalanced31_reporting_source_means',0)
u.to_csv(DATA/'recent_unbalanced31_original_estimator.csv',index=False)
# Same country-year weights as new yearly estimator, summarized over observed years per country.
uc=quartiles(cy.groupby(['country_iso3','country','region'],as_index=False)[['frequency','annual_hours','gdp']].mean())
uc['duration']=uc.annual_hours/uc.frequency
analyse(uc,'unbalanced31_country_year_means',0)
res=pd.DataFrame(summary)
res.to_csv(DATA/'recent_yearly_gap_components.csv',index=False)
pd.DataFrame(groups).to_csv(DATA/'recent_quartile_group_means.csv',index=False)
pd.DataFrame(changes).to_csv(DATA/'recent_fixed_cohort_changes.csv',index=False)
pd.concat(draws,ignore_index=True).to_parquet(OUT/'country_bootstrap_draws.parquet',index=False)

models=[];diagnostics=[]
def model(z,label,year,pooled=False):
    z=z.copy()
    for x in ['frequency','annual_hours','duration','gdp']:z['log_'+x]=np.log(z[x])
    z['log_frequency_minus_duration']=z.log_frequency-z.log_duration
    for outcome in ['frequency','duration','annual_hours','frequency_minus_duration']:
        if pooled:
            z['country_weight']=1/z.groupby('country_iso3').year.transform('size')
            fit=smf.wls(f'log_{outcome} ~ log_gdp + C(year)',data=z,weights=z.country_weight).fit(cov_type='cluster',cov_kwds={'groups':z.country_iso3,'use_correction':True},use_t=True)
            df=z.country_iso3.nunique()-1;cov='country-clustered HC1; country-equal total weights'
        else:
            fit=smf.ols(f'log_{outcome} ~ log_gdp',data=z).fit(cov_type='HC3',use_t=True)
            df=len(z)-2;cov='HC3; equal country weights'
        beta=fit.params['log_gdp'];se=fit.bse['log_gdp'];critical=t.ppf(.975,df)
        lo,hi=beta+np.array([-1,1])*critical*se
        models.append({'analysis':label,'year':year,'outcome':outcome,'countries':z.country_iso3.nunique(),'observations':len(z),'beta':beta,'se':se,'beta_low':lo,'beta_high':hi,'doubling_pct':100*np.expm1(beta*np.log(2)),'doubling_low':100*np.expm1(lo*np.log(2)),'doubling_high':100*np.expm1(hi*np.log(2)),'p_value':2*t.sf(abs(beta/se),df),'df':df,'covariance':cov,'r_squared':fit.rsquared})
        if not pooled:
            infl=fit.get_influence()
            diagnostics.append({'analysis':label,'year':year,'outcome':outcome,'max_leverage':float(infl.hat_matrix_diag.max()),'max_absolute_studentized_residual':float(np.abs(infl.resid_studentized_external).max()),'design_condition_number':float(fit.condition_number),'n_countries':len(z),'no_value_based_exclusion':True})
for year,z in annual.groupby('year'):model(z,'available_countries_yearly_groups',int(year))
for year,z in strict.groupby('year'):model(z,'strict_fixed15_sources_fixed_groups',int(year))
model(cy,'available_country_years_pooled',0,True)
model(strict,'strict_fixed15_country_years_pooled',0,True)
pd.DataFrame(models).to_csv(DATA/'recent_country_economic_gradients.csv',index=False)
pd.DataFrame(diagnostics).to_csv(OUT/'model_diagnostics.csv',index=False)

# Reproduce previously released point estimates independently; bootstrap draws differ by design.
prior=pd.read_csv(EXT/'data/absolute_gap_replication.csv')
pairs=[('strict_fixed15_sources_fixed_groups_three_year_mean',0,'Balanced 2021–2023 · equal country'),('original_unbalanced31_reporting_source_means',0,'Unbalanced 2021–2023 · equal country')]+[('strict_fixed15_sources_fixed_groups',y,f'{y} · fixed balanced cohort') for y in [2021,2022,2023]]
checks=[]
for new,yr,old in pairs:
    rr=res[(res.analysis==new)&(res.year==yr)].iloc[0]; pp=prior[prior.comparison==old].iloc[0]
    for k in ['gap_h','frequency_h','duration_h']:
        err=abs(rr[k]-pp[k]);checks.append({'analysis':new,'year':yr,'quantity':k,'v37':rr[k],'prior':pp[k],'absolute_difference':err});assert err<1e-9
pd.DataFrame(checks).to_csv(OUT/'prior_point_reproduction.csv',index=False)
files=[INPUT,EXT/'inputs/panel_audit.json',EXT/'metadata/mepco_regulator_unit_flag.json',EXT/'inputs/new_duration_source_flags.csv',EXT/'data/absolute_gap_replication.csv',OLD/'core_reproducibility/recent_geographic_extension/extract.py']
manifest=[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files]
(OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
audit={'seed':SEED,'bootstrap_draws':B,'admitted_source_years':len(e),'canonical_source_areas':e.source_area_id.nunique(),'identity_duplicate_rows_excluded':int(alias_duplicate.sum()),'identity_resolution':'CODENSA2021 contemporary 487-minute vintage retained under canonical Enel Colombia identity; retrospective 488-minute duplicate excluded','admitted_countries_total':e.country_iso3.nunique(),'available_country_year_counts':{str(y):g.country_iso3.nunique() for y,g in annual.groupby('year')},'common_countries':common.country_iso3.nunique(),'strict_fixed_countries':strict.country_iso3.nunique(),'strict_fixed_sources':strict_raw.source_area_id.nunique(),'source_year_exclusion_counts':r.v37_decision.value_counts().to_dict(),'point_reproduction_max_abs_error':max(x['absolute_difference'] for x in checks),'country_year_H_equals_FD':bool(np.allclose(cy.annual_hours,cy.frequency*cy.duration)),'bootstrap_components_sum_to_gap':True,'national_customer_estimates':False,'source_inclusion_rules_globally_harmonized':False,'source_extensions_pooled':False,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}
(OUT/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
print(res[res.analysis.eq('available_countries_yearly_groups')][['year','countries','low_countries','high_countries','gap_h','gap_h_low','gap_h_high','frequency_h','frequency_h_low','frequency_h_high','duration_h','duration_h_low','duration_h_high','frequency_minus_duration_h','frequency_minus_duration_h_low','frequency_minus_duration_h_high']].to_string(index=False))
print(pd.DataFrame(models).query("analysis=='available_countries_yearly_groups'")[['year','outcome','doubling_pct','doubling_low','doubling_high']].to_string(index=False))
print(json.dumps(audit,indent=2))
