"""Global-first analytical release; no extrapolated national observations."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,duckdb,patsy,statsmodels.api as sm
from scipy.stats import t,spearmanr

P=Path(__file__).resolve().parents[1];R=P.parents[1]
inputs=[]
def read(path):
    q=R/path;inputs.append(dict(path=path,sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
    return pd.read_parquet(q) if q.suffix=='.parquet' else pd.read_csv(q)
def save(d,n):
    d.to_parquet(P/'data'/(n+'.parquet'),index=False)
    d.to_csv(P/'source_data'/(n+'.csv'),index=False)

g=read('10_manuscript/20261001_blackout_inequalities_v08/data/global_paired.parquet')
g=g.rename(columns={'gdp_ppp_pc_at_year':'gdp','saifi':'frequency','caidi_h':'duration','saidi_h':'burden'})
assert len(g)==152 and g.ISO3.is_unique
assert np.allclose(g.frequency*g.duration,g.burden)
g['gdp_quartile']=pd.qcut(g.gdp,4,labels=['Q1','Q2','Q3','Q4'])
g['log_gdp']=np.log(g.gdp)
save(g,'global_benchmark_context')

# Paired log models retain covariance by fitting each contrast directly.
old=read('07_analysis/economic_hazard_strengthening_20260928/components/global_baseline/data/firms_all_source_pairs.parquet')
save(old,'historical_conditional_firm_context')
models=[]
for source,z in [('Business-city benchmark',g),('Conditional enterprise survey',old)]:
    z=z[(z[['frequency','duration','gdp']]>0).all(axis=1)].copy()
    z['log_gdp']=np.log(z.gdp);z['year_c']=(z.year-z.year.mean())/10
    for spec in ['Income only','Continent adjusted']:
        formula='1 + log_gdp'
        if spec=='Continent adjusted':
            formula+=' + C(continent)'
            if source=='Conditional enterprise survey':formula+=' + year_c'
        X=patsy.dmatrix(formula,z,return_type='dataframe',eval_env=-1)
        assert np.linalg.matrix_rank(X)==X.shape[1]
        ys={'Frequency':np.log(z.frequency),'Duration':np.log(z.duration),
            'Frequency minus duration':np.log(z.frequency)-np.log(z.duration)}
        if 'burden' in z:ys['Annual burden']=np.log(z.burden)
        for outcome,y in ys.items():
            m=sm.OLS(y,X).fit(cov_type='HC3',use_t=True)
            b=float(m.params.log_gdp);lo,hi=m.conf_int().loc['log_gdp']
            models.append(dict(source=source,specification=spec,outcome=outcome,n=len(z),
                coefficient=b,ci_low=lo,ci_high=hi,p_value=m.pvalues.log_gdp,
                response_pct=100*np.expm1(b*np.log(2)),response_low=100*np.expm1(lo*np.log(2)),
                response_high=100*np.expm1(hi*np.log(2)),gdp_min=z.gdp.min(),gdp_max=z.gdp.max()))
save(pd.DataFrame(models),'global_paired_income_models')

# Continent-specific gaps: same-country paired resampling, fixed GDP groups.
rng=np.random.default_rng(20261002);cont=[];elig=[]
for region,z in [('All economies',g)]+list(g.groupby('continent')):
    z=z.copy();z['q']=pd.qcut(z.gdp.rank(method='first'),4,labels=False)
    lo=z[z.q.eq(0)];hi=z[z.q.eq(3)];admit=min(len(lo),len(hi))>=5
    elig.append(dict(region=region,economies=len(z),n_low=len(lo),n_high=len(hi),admitted=admit))
    if not admit:continue
    cols=['frequency','duration','burden'];a=np.log(lo[cols].to_numpy());b=np.log(hi[cols].to_numpy())
    gap=a.mean(0)-b.mean(0)
    draws=a[rng.integers(0,len(a),(4000,len(a)))].mean(1)-b[rng.integers(0,len(b),(4000,len(b)))].mean(1)
    for j,k in enumerate(cols):
        lower,upper=np.quantile(np.exp(draws[:,j]),[.025,.975])
        cont.append(dict(region=region,metric=k,n=len(z),n_low=len(lo),n_high=len(hi),ratio=np.exp(gap[j]),
                         ci_low=lower,ci_high=upper,log_gap=gap[j],frequency_log_share=gap[0]/gap[2]))
save(pd.DataFrame(cont),'global_continent_decomposition');save(pd.DataFrame(elig),'global_continent_admission')

# Newer enterprise data. No unconditional median is used for recovery inference.
db=R/'08_database_expansion/20260927_v13/global_electricity_v13.duckdb'
con=duckdb.connect(str(db),read_only=True)
survey=con.sql("SELECT * FROM wbes_country_survey_history WHERE indicator IN ('in16','bready_in2','bready_in9','bready_in3_median','in12')").df()
con.close();save(survey,'enterprise_native_selected_indicators')
inputs.append(dict(path=str(db.relative_to(R)),bytes=db.stat().st_size,
    role='Read-only database query; extracted rows frozen and hashed separately',
    query="SELECT * FROM wbes_country_survey_history WHERE indicator IN ('in16','bready_in2','bready_in9','bready_in3_median','in12')",
    extracted_path='data/enterprise_native_selected_indicators.parquet',
    extracted_sha256=hashlib.sha256((P/'data/enterprise_native_selected_indicators.parquet').read_bytes()).hexdigest()))
context=read('07_analysis/hazard_cause_country_panel_20260929/data/country_year_electricity_context.parquet')
valid=survey[survey.screening_eligible & survey.value.notna() & survey.value.ge(0)].copy()
assert not valid.duplicated(['survey_wave_id','indicator']).any()
wide=valid.pivot(index=['survey_wave_id','country_iso3','year'],columns='indicator',values='value').reset_index()
wide=wide.rename(columns={'country_iso3':'ISO3','in16':'prevalence_pct','bready_in2':'monthly_frequency',
    'bready_in9':'generator_pct','bready_in3_median':'unconditional_median_duration_h','in12':'constraint_pct'})
wide=wide.merge(context[['ISO3','year','gdp','name','continent']],on=['ISO3','year'],how='left',validate='many_to_one')
paired=wide.dropna(subset=['prevalence_pct','monthly_frequency','generator_pct']).sort_values('year').groupby('ISO3').tail(1).copy()
dates=valid[['survey_wave_id','fieldwork_date_native']].drop_duplicates()
paired=paired.merge(dates,on='survey_wave_id',validate='many_to_one')
static=context.drop_duplicates('ISO3').set_index('ISO3')
paired['continent']=paired.continent.fillna(paired.ISO3.map(static.continent)).fillna('Unassigned')
paired['name']=paired.name.fillna(paired.ISO3.map(static['name'])).fillna(paired.ISO3)
for iso,name,continent in [('ALB','Albania','Europe'),('FSM','Micronesia (Federated States of)','Oceania')]:
    paired.loc[paired.ISO3.eq(iso),['name','continent']]=[name,continent]
paired['duration_use']='Unconditional median including zero-outage firms; not used as conditional restoration duration'
save(wide,'enterprise_all_wave_profiles');save(paired,'enterprise_latest_profiles')
des=[]
for region,z in [('All economies',paired)]+list(paired.groupby('continent')):
    for metric in ['prevalence_pct','monthly_frequency','generator_pct']:
        des.append(dict(region=str(region),metric=metric,n=len(z),median=z[metric].median(),q25=z[metric].quantile(.25),q75=z[metric].quantile(.75)))
save(pd.DataFrame(des),'enterprise_geographic_summary')
ass=[]
for metric in ['prevalence_pct','monthly_frequency','generator_pct','constraint_pct']:
    z=paired.dropna(subset=['gdp',metric]);rho,pv=spearmanr(np.log(z.gdp),z[metric])
    ass.append(dict(metric=metric,n=len(z),rho=rho,p_value=pv))
save(pd.DataFrame(ass),'enterprise_rank_associations')

# Native cause profiles. This is an atlas of reported systems, not national imputation.
native=read('07_analysis/project_database_review_20260930/data/within_system_cause_profiles.parquet')
keep=native.profile_id.str.startswith('NORDIC_') | native.profile_id.isin(['RWA_REG','SEN_SENELEC','FIN_DISTRIBUTION'])
atlas=native[keep][['profile_id','country_iso3','system','period','cause_native','occurrence_share_pct','burden_share_pct','occurrence_unit','burden_unit','profile_method','source_path']].copy()

jam=read('07_analysis/country_channel_expansion2_20260930/data/jamaica_cause_year_2018_2022.parquet')
jam=jam[jam.year.between(2018,2021)].copy()
assert jam.groupby('year').cause_native.nunique().eq(9).all()
for col in ['saifi_customer','saidi_min_customer']:
    jam[col+'_share']=100*jam[col]/jam.groupby('year')[col].transform('sum')
j=jam.groupby('cause_native')[['saifi_customer_share','saidi_min_customer_share']].mean().reset_index()
j=j.rename(columns={'saifi_customer_share':'occurrence_share_pct','saidi_min_customer_share':'burden_share_pct'})
j=j.assign(profile_id='JAM_JPS',country_iso3='JAM',system='Jamaica · JPS',period='2018–2021',occurrence_unit='customer interruption share',burden_unit='customer duration share',profile_method='equal-year mean; stable nine-cause taxonomy',source_path='07_analysis/country_channel_expansion2_20260930/data/jamaica_cause_year_2018_2022.parquet')

ny=read('10_manuscript/20261001_blackout_inequalities_v08/data/ny_regional_category_year.parquet')
n=ny.groupby('category')[['customer_interruptions','customer_hours']].sum().reset_index().rename(columns={'category':'cause_native'})
# All disjoint categories are retained; no broad parent row is added to its children.
n['occurrence_share_pct']=100*n.customer_interruptions/n.customer_interruptions.sum()
n['burden_share_pct']=100*n.customer_hours/n.customer_hours.sum()
n=n.assign(profile_id='USA_NY_QUALIFIED',country_iso3='USA',system='New York · National Grid',period='2021–2024',occurrence_unit='customer interruption share',burden_unit='customer-hour share',profile_method='pooled native customer counts across eight regions',source_path='10_manuscript/20261001_blackout_inequalities_v08/data/ny_regional_category_year.parquet')
nz=read('10_manuscript/20261001_blackout_inequalities_v08/data/nz_explicit21_complete_partition.parquet')
z=nz.groupby('category')[['frequency','burden_h']].mean().reset_index().rename(columns={'category':'cause_native'})
z['occurrence_share_pct']=100*z.frequency/z.frequency.sum();z['burden_share_pct']=100*z.burden_h/z.burden_h.sum()
z=z.assign(profile_id='NZL_QUALIFIED21',country_iso3='NZL',system='New Zealand · 21 networks',period='2023–2025',occurrence_unit='equal-network customer-frequency share',burden_unit='equal-network customer-duration share',profile_method='shares of equal-network mean annual indices; full remaining class retained',source_path='10_manuscript/20261001_blackout_inequalities_v08/data/nz_explicit21_complete_partition.parquet')
atlas=pd.concat([atlas,j[atlas.columns],n[atlas.columns],z[atlas.columns]],ignore_index=True)

groups={
'Weather':['Lightning','Lightning / thunderstorm','Wind / storm','Snow / ice','Other weather','Recorded weather','Weather'],
'Other environment':['Other environmental','Animals','Vegetation','Wild Life','Contamination','Tree contacts','Tree contact'],
'Human / operation':['Operation and maintenance','Public Error','Operating error','Human activity','Emergency Works','switching_or_works'],
'Equipment / electrical':['Technical equipment','Equipment','Customer equipment','Technical / operational','Equipment Failure','Earth Fault','Overcurrent','Overvoltage','Undervoltage','Overload','Under Frequency','overload_or_low_voltage'],
'Supply constraint':['Power Supply','Emergency Load Shedding','production_shortage','large_customer_curtailment'],
'Planned':['Planned','Planned Works','Prearranged'],
'Other / unresolved':['External influence','Other','Unknown','Other / mixed','Remaining','Remaining causes','Mixed accidents','incident_network_or_generation_unspecified']}
lookup={v:k for k,vals in groups.items() for v in vals}
atlas['cause_group']=atlas.cause_native.map(lookup)
if atlas.cause_group.isna().any():raise ValueError('Unclassified literal causes: '+str(atlas.loc[atlas.cause_group.isna(),'cause_native'].unique()))
save(atlas,'global_cause_native_crosswalk')
ag=atlas.groupby(['profile_id','country_iso3','system','period','cause_group','occurrence_unit','burden_unit','profile_method'],dropna=False)[['occurrence_share_pct','burden_share_pct']].sum().reset_index()
ag['share_gap_pp']=ag.burden_share_pct-ag.occurrence_share_pct
save(ag,'global_cause_atlas')
summary=[]
for pid,a in ag.groupby('profile_id'):
    topf=a[a.occurrence_share_pct.eq(a.occurrence_share_pct.max())].cause_group.tolist()
    toph=a[a.burden_share_pct.eq(a.burden_share_pct.max())].cause_group.tolist()
    weather=a[a.cause_group.eq('Weather')]
    summary.append(dict(profile_id=pid,country_iso3=a.country_iso3.iloc[0],system=a.system.iloc[0],period=a.period.iloc[0],
        occurrence_sum=a.occurrence_share_pct.sum(),burden_sum=a.burden_share_pct.sum(),leading_occurrence='; '.join(topf),leading_burden='; '.join(toph),
        weather_frequency_share=weather.occurrence_share_pct.sum() if len(weather) else np.nan,
        weather_burden_share=weather.burden_share_pct.sum() if len(weather) else np.nan,
        physical_weather_reported=bool(len(weather)),classification_note='Absent physical-weather category is unresolved, not zero natural-hazard contribution'))
save(pd.DataFrame(summary),'global_cause_atlas_profiles')
summary=dict(global_benchmark_economies=len(g),enterprise_latest_economies=len(paired),enterprise_year_min=int(paired.year.min()),enterprise_year_max=int(paired.year.max()),
    enterprise_gdp_paired=int(paired.gdp.notna().sum()),historical_conditional_economies=len(old),historical_positive_economies=int(old.eligible_log.sum()),
    cause_countries=int(atlas.country_iso3.nunique()),cause_profiles=int(atlas.profile_id.nunique()),cause_groups=groups,
    global_and_enterprise_union=len(set(g.ISO3)|set(paired.ISO3)),global_and_enterprise_intersection=len(set(g.ISO3)&set(paired.ISO3)))
(P/'metadata/global_first_summary.json').write_text(json.dumps(summary,indent=2))
(P/'metadata/global_first_inputs.json').write_text(json.dumps(inputs,indent=2))
print(json.dumps(summary,indent=2));print(pd.DataFrame(models).query("specification=='Continent adjusted'")[['source','outcome','n','response_pct','response_low','response_high']].to_string(index=False))
