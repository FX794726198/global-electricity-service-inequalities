"""Admission, reporting-threshold and influence checks for historical benchmarks."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
import patsy,statsmodels.api as sm

P=Path(__file__).resolve().parents[1];R=P.parents[1]
input_path=R/'07_analysis/global_absolute_gap_20261002/source_data/country_benchmarks.csv'
g=pd.read_csv(input_path)
assert len(g)==152 and g.ISO3.is_unique
rawdir=R/'01_raw/global/worldbank'
cities={'Beijing','Delhi','Jakarta','Kano','Karachi','Lahore','Los Angeles','Mexico City','Monterrey','Moscow','Mumbai','New York','Osaka','Rio de Janeiro','Sao Paulo','Shanghai','St. Petersburg','Surabaya','Tokyo'}
records=[];lookup={}
for m,code in [('saifi','IC.ELC.SAIF.XD.DB1619'),('saidi_h','IC.ELC.SAID.XD.DB1619')]:
    for x in json.loads((rawdir/(code+'.json')).read_text())[1]:
        name=x['country']['value'];iso=x['countryiso3code'] or ('TWN' if x['country']['id']=='TW' else '')
        lookup[name]=iso
        if name in cities or not iso:continue
        records.append(dict(ISO3=iso,label=name,year=int(x['date']),metric=m,value=x['value']))
raw=pd.DataFrame(records).pivot(index=['ISO3','label','year'],columns='metric',values='value').reset_index()
latest=raw.dropna(subset=['saifi','saidi_h'],how='all').sort_values('year').groupby('ISO3').tail(1)
assert len(latest)==156 and not latest.ISO3.duplicated().any()
latest['admitted']=latest.ISO3.isin(g.ISO3)
latest['reason']=np.where(latest.admitted,'Paired positive benchmark and GDP','GDP unmatched in retained archive')
latest.to_csv(P/'data/benchmark_admission.csv',index=False)
none=raw.groupby('ISO3')[['saifi','saidi_h']].count().sum(axis=1).eq(0)
missing=raw[raw.ISO3.isin(none[none].index)].drop_duplicates('ISO3')[['ISO3','label']]
missing.to_csv(P/'data/economies_without_benchmark.csv',index=False)
raw[raw[['saifi','saidi_h']].eq(0).any(axis=1)].to_csv(P/'data/historical_zero_rows.csv',index=False)
assert latest[['saifi','saidi_h']].gt(0).all().all()
check=latest.merge(g[['ISO3','year','saifi','saidi_h']],on=['ISO3','year'],suffixes=('_raw','_analytical'),validate='one_to_one')
assert len(check)==152
assert np.allclose(check.saifi_raw,check.saifi_analytical) and np.allclose(check.saidi_h_raw,check.saidi_h_analytical)

alias={'Iran':'IRN','Bosnia Herzegovina':'BIH',"Cote d'Ivoire":'CIV','Yemen, Republic of':'YEM',
       'Korea, Republic of':'KOR','Congo, Republic of':'COG','Micronesia, Federated States of':'FSM',
       'Congo, Democratic Republic of':'COD','Venezuela, Republica Bolivariana de':'VEN',
       'Egypt, Arab Republic of':'EGY','São Tomé and Príncipe':'STP','Hong Kong SAR, China':'HKG'}
d=pd.read_csv(P/'data/profile_definition_rows.csv');d['ISO3']=d.country.map(lookup|alias)
assert d.ISO3.notna().all()
d.to_csv(P/'source_data/profile_definition_rows_with_iso.csv',index=False)
counts=d.groupby('ISO3').size()
one=d[d.ISO3.map(counts).eq(1)].copy()
joined=g.merge(one,on='ISO3',how='left',suffixes=('','_profile'),validate='one_to_one')
joined['numeric_match']=((joined.saifi-joined.saifi_published).abs().le(.051)&
                         (joined.saidi_h-joined.saidi_published).abs().le(.051))
joined['definition_admitted']=joined.year.eq(2019)&joined.numeric_match&joined.minimum_minutes.notna()
joined.to_csv(P/'data/definition_linkage.csv',index=False)

def point(z):
    l=z[z.gdp_quartile.eq('Q1')];h=z[z.gdp_quartile.eq('Q4')]
    def vals(z):
        f=z.saifi.mean();v=z.saidi_h.mean();return f,v,v/f
    lf,lh,ld=vals(l);hf,hh,hd=vals(h)
    f=(lf-hf)*(ld+hd)/2;dur=(ld-hd)*(lf+hf)/2;gap=lh-hh
    dl=np.log(l[['saifi','caidi_h','saidi_h']]).mean()-np.log(h[['saifi','caidi_h','saidi_h']]).mean()
    assert np.isclose(f+dur,gap)
    return dict(n=len(z),n_Q1=len(l),n_Q4=len(h),frequency_share=100*f/gap,
        frequency_component_h=f,duration_component_h=dur,annual_gap_h=gap,
        log_frequency_share=100*dl.saifi/dl.saidi_h,
        frequency_ratio=np.exp(dl.saifi),duration_ratio=np.exp(dl.caidi_h))

def slopes(z):
    z=z[z.groupby('continent').ISO3.transform('size').ge(2)].copy();z['log_gdp']=np.log(z.gdp_ppp_pc_at_year)
    X=patsy.dmatrix('1 + log_gdp + C(continent)',z,return_type='dataframe',eval_env=-1)
    rows=[]
    for label,y in [('Frequency',np.log(z.saifi)),('Duration',np.log(z.caidi_h)),
                    ('Paired coefficient difference',np.log(z.saifi)-np.log(z.caidi_h))]:
        m=sm.OLS(y,X).fit(cov_type='HC3',use_t=True);b=m.params.log_gdp;lo,hi=m.conf_int().loc['log_gdp']
        assert np.isfinite([b,lo,hi]).all()
        rows.append(dict(outcome=label,model_n=len(z),coefficient=b,lower=lo,upper=hi,
            change_pct=100*np.expm1(b*np.log(2)),change_low=100*np.expm1(lo*np.log(2)),change_high=100*np.expm1(hi*np.log(2))))
    return rows

variants=[('Original 152',g),('2019 database label',g[g.year.eq(2019)]),
          ('Matched definition, any numeric threshold',joined[joined.definition_admitted]),
          ('Matched definition, 3 minutes',joined[joined.definition_admitted&joined.minimum_minutes.eq(3)]),
          ('Matched definition, 5 minutes',joined[joined.definition_admitted&joined.minimum_minutes.eq(5)])]
rows=[];models=[]
for name,z in variants:
    if min(z.gdp_quartile.value_counts().reindex(['Q1','Q4'],fill_value=0))<3:continue
    rows.append(dict(comparison=name,**point(z)))
    for x in slopes(z):models.append(dict(comparison=name,n=len(z),**x))
pd.DataFrame(rows).to_csv(P/'data/benchmark_definition_sensitivity.csv',index=False)
pd.DataFrame(models).to_csv(P/'data/benchmark_definition_models.csv',index=False)

lo=[]
for x in g.itertuples():lo.append(dict(deleted_ISO3=x.ISO3,deleted_name=x.label,**point(g[~g.ISO3.eq(x.ISO3)])))
loo=pd.DataFrame(lo);loo.to_csv(P/'data/single_economy_influence.csv',index=False)
ranked=g.sort_values('saidi_h',ascending=False);exclude=ranked.head(5).ISO3.tolist()
stress=dict(comparison='Remove five largest observed annual burdens',excluded=exclude,
            **point(g[~g.ISO3.isin(exclude)]))

# Reproduce the original pair bootstrap in its original row order and seed.
orig=pd.read_csv(R/'07_analysis/global_recovery_map_20261001/source_data/paired_frequency_gdp.csv')
assert set(orig.ISO3)==set(g.ISO3)
orig['caidi_h']=orig.saidi_h/orig.saifi
rng=np.random.default_rng(20261001);draws=[]
for _ in range(4000):
    z=orig.iloc[rng.integers(0,len(orig),len(orig))]
    rf=spearmanr(z.gdp_ppp_pc_at_year,z.saifi).statistic
    rd=spearmanr(z.gdp_ppp_pc_at_year,z.caidi_h).statistic
    draws.append([rf,rd,rd-rf])
rho=[spearmanr(orig.gdp_ppp_pc_at_year,orig[x]).statistic for x in ['saifi','caidi_h']]
pd.DataFrame(draws,columns=['frequency_rho','duration_rho','difference']).to_csv(P/'data/paired_spearman_bootstrap.csv',index=False)
meta=dict(n_latest=len(latest),n_admitted=len(g),n_no_nonmissing_pair=len(missing),
    latest_zero_count=int(latest[['saifi','saidi_h']].eq(0).any(axis=1).sum()),
    latest_incomplete_count=int(latest[['saifi','saidi_h']].isna().any(axis=1).sum()),
    database_label_counts=g.year.value_counts().sort_index().to_dict(),
    nonadmitted=latest[~latest.admitted][['ISO3','label','year']].to_dict('records'),
    rho_difference=rho[1]-rho[0],rho_difference_ci=np.quantile(np.array(draws)[:,2],[.025,.975]).tolist(),
    single_deletion_range={k:[loo[k].min(),loo[k].max()] for k in ['frequency_share','log_frequency_share','annual_gap_h']},
    largest_five_stress=stress,bootstrap_seed=20261001,bootstrap_replicates=4000,
    definition_gate='2019 database label; one reported city profile; numeric match at displayed rounding; explicit minimum threshold',
    restriction_limit='Minimum-threshold restriction does not verify planned-outage, load-shedding or major-event inclusion.',
    original_gdp_groups_preserved=True,zero_inclusion_changes_nothing=True,
    input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())
(P/'metadata/benchmark_audit.json').write_text(json.dumps(meta,indent=2))
print(pd.DataFrame(rows).round(3).to_string(index=False));print(json.dumps(meta,indent=2))
