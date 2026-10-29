"""Prepare official resident-income context and unchanged outage support."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P = Path(__file__).resolve().parents[1]
D = P / 'data'; M = P / 'metadata'
B = P.parent / 'brazil_detailed_cause_extension_20261002' / 'data'

def numeric(value):
    if value == '-': return 0.
    if value in ('..', '...', 'X'): return np.nan
    return float(value)

rows = []
for f in sorted((P/'raw').glob('ibge_income_bands_*.json')):
    for var in json.loads(f.read_text()):
        assert var['id'] == '13604'
        for result in var['resultados']:
            cl = next(c for c in result['classificacoes'] if str(c['id']) == '386')
            assert len(cl['categoria']) == 1
            cat, label = next(iter(cl['categoria'].items()))
            for s in result['series']:
                raw = s['serie']['2022']
                rows.append(dict(municipality_code=str(s['localidade']['id']),
                    municipality=s['localidade']['nome'], category=cat,
                    category_label=label, persons=numeric(raw), native_value=raw))
bands = pd.DataFrame(rows)
assert not bands.duplicated(['municipality_code','category']).any()
assert bands.persons.notna().all() and bands.persons.ge(0).all()
assert len(bands) == 5570*12
wide = bands.pivot(index='municipality_code',columns='category',values='persons')
wide['band_sum'] = wide.drop(columns='9680').sum(axis=1)
wide['rounding_difference'] = wide.band_sum-wide['9680']
assert wide.rounding_difference.abs().le(3).all()
context = bands[['municipality_code','municipality']].drop_duplicates().set_index('municipality_code')
context['income_universe_persons'] = wide['9680']
context['low_income_persons'] = wide[['9681','9682','9692']].sum(axis=1)
context['very_low_income_persons'] = wide[['9681','9692']].sum(axis=1)
context['low_income_share_pct'] = 100*context.low_income_persons/context.income_universe_persons
context['very_low_income_share_pct'] = 100*context.very_low_income_persons/context.income_universe_persons
context['rounding_difference_persons'] = wide.rounding_difference
labels = {'13604':'income_universe_check','13431':'mean_household_income_pc_brl',
          '13534':'median_household_income_pc_brl'}
for var in json.loads((P/'raw/ibge_income_mean_median.json').read_text()):
    vals = {str(s['localidade']['id']):numeric(s['serie']['2022'])
            for result in var['resultados'] for s in result['series']}
    assert len(vals) == 5570
    context[labels[var['id']]] = pd.Series(vals)
assert context.income_universe_check.eq(context.income_universe_persons).all()
context['income_year'] = 2022
context['minimum_wage_brl'] = 1212.
context['low_income_threshold_brl'] = 606.
context['very_low_income_threshold_brl'] = 303.
context = context.reset_index()
bands.to_parquet(D/'ibge_income_bands_2022.parquet',index=False)
context.to_parquet(D/'ibge_municipal_income_2022.parquet',index=False)
context.to_csv(D/'ibge_municipal_income_2022.csv',index=False)

cohort = pd.read_parquet(B/'economic_cohort.parquet').sort_values('IdeConjunto').reset_index(drop=True)
assert len(cohort) == 703 and cohort.municipality_code.nunique() == 275
cohort = cohort.merge(context,on='municipality_code',how='left',validate='many_to_one',indicator=True)
assert cohort['_merge'].eq('both').all()
cohort = cohort.drop(columns='_merge')
cohort['disadvantage_10pp'] = cohort.low_income_share_pct/10
cohort['log_density'] = np.log(cohort.density)
cohort['log_gdp'] = np.log(cohort.gdp_pc_brl)
cohort['log_median_income'] = np.log(cohort.median_household_income_pc_brl)
municipalities = cohort.drop_duplicates('municipality_code')
cutpoints = municipalities.low_income_share_pct.quantile([.25,.5,.75]).to_list()
cohort['disadvantage_quartile'] = pd.cut(cohort.low_income_share_pct,
    [-np.inf,*cutpoints,np.inf],labels=['Q1','Q2','Q3','Q4'])
cohort.to_parquet(D/'analysis_context.parquet',index=False)
cohort.to_csv(D/'analysis_context.csv',index=False)

monthly = pd.read_parquet(B/'native_monthly_support.parquet')
monthly = monthly[monthly.IdeConjunto.isin(cohort.IdeConjunto)].copy()
assert not monthly.duplicated(['IdeConjunto','AnoIndice','NumPeriodoIndice']).any()
assert monthly.groupby('IdeConjunto').size().eq(24).all()
strata = ['XP','XN','IP','IND','INE','INC','INO','IPC','XPC','XNC']
categories = {'Ordinary unplanned':['IND'], 'Emergency / critical':['INE','INC'],
              'Internal unplanned':['IND','INE','INC'], 'All ten strata':strata}
regulatory = []; support = []
for category, parts in categories.items():
    fcols=['FEC'+x for x in parts]; hcols=['DEC'+x for x in parts]
    for aid,x in monthly.groupby('IdeConjunto'):
        ok = x[fcols+hcols].notna().all(axis=1)
        support.append(dict(IdeConjunto=aid,category=category,valid_months=int(ok.sum()),
                            common_24_months=bool(ok.all())))
        if not ok.all(): continue
        n = float((x[fcols].sum(axis=1)*x.NumCon).sum())
        h = float((x[hcols].sum(axis=1)*x.NumCon).sum())
        cy = float(x.NumCon.sum()/12)
        regulatory.append(dict(IdeConjunto=aid,category=category,customer_interruptions=n,
            customer_hours=h,customer_years=cy,frequency=n/cy,burden_hours=h/cy,
            conditional_duration_h=h/n if n>0 else np.nan))
regulatory = pd.DataFrame(regulatory)
assert regulatory[['customer_interruptions','customer_hours','customer_years']].ge(0).all().all()
assert not ((regulatory.customer_interruptions == 0)&(regulatory.customer_hours > 0)).any()
regulatory.to_parquet(D/'regulatory_stages.parquet',index=False)
regulatory.to_csv(D/'regulatory_stages.csv',index=False)
pd.DataFrame(support).to_csv(D/'regulatory_support.csv',index=False)

annual = pd.read_parquet(B/'area_cause_stages.parquet')
causes = sorted(annual.cause.unique())
ix = pd.MultiIndex.from_product([cohort.IdeConjunto,causes],names=['IdeConjunto','cause'])
fine = annual.set_index(['IdeConjunto','cause'])[['records','record_hours','extra_zero_upper']].reindex(ix,fill_value=0).reset_index()
fine = fine.merge(cohort[['IdeConjunto','customer_years']],on='IdeConjunto',validate='many_to_one')
fine['record_rate'] = fine.records/fine.customer_years
fine['record_hours_rate'] = fine.record_hours/fine.customer_years
fine['conditional_duration_h'] = fine.record_hours/fine.records.where(fine.records>0)
fine.to_parquet(D/'cause_stages.parquet',index=False)

summary=[]
for group,x in cohort.groupby('disadvantage_quartile',observed=True):
    ms=x.drop_duplicates('municipality_code')
    row=dict(group=str(group),areas=len(x),municipalities=len(ms),states=x.state_code.nunique(),
        contexts=x.cnpj.nunique(),customer_years=x.customer_years.sum(),
        income_universe_persons=ms.income_universe_persons.sum(),
        low_income_persons=ms.low_income_persons.sum())
    for col in ['low_income_share_pct','median_household_income_pc_brl','gdp_pc_brl','density']:
        v=x[col];row.update({col+'_'+stat:float(val) for stat,val in
            [('mean',v.mean()),('sd',v.std()),('median',v.median()),('min',v.min()),('max',v.max())]})
    summary.append(row)
pd.DataFrame(summary).to_csv(D/'group_support_descriptives.csv',index=False)
des=[]
for domain,frame,cat,cols in [('Regulatory',regulatory,'category',['frequency','burden_hours','conditional_duration_h']),
                            ('Source records',fine,'cause',['record_rate','record_hours_rate','conditional_duration_h'])]:
    tmp=frame.merge(cohort[['IdeConjunto','disadvantage_quartile']],on='IdeConjunto',validate='many_to_one')
    for (category,q),g in tmp.groupby([cat,'disadvantage_quartile'],observed=True):
        for col in cols:
            v=g[col]
            des.append(dict(domain=domain,category=category,quartile=str(q),outcome=col,
                n=len(v),missing=int(v.isna().sum()),zeros=int(v.eq(0).sum()),mean=v.mean(),
                sd=v.std(),median=v.median(),p25=v.quantile(.25),p75=v.quantile(.75),max=v.max()))
pd.DataFrame(des).to_csv(D/'outcome_descriptives.csv',index=False)
checks=dict(created_utc=datetime.now(timezone.utc).isoformat(),census_municipalities=len(context),
    category_rows=len(bands),category_rounding_error_max_people=float(wide.rounding_difference.abs().max()),
    area_count=len(cohort),municipality_count=cohort.municipality_code.nunique(),
    state_count=cohort.state_code.nunique(),reporting_contexts=cohort.cnpj.nunique(),
    missing_income_link=0,quartile_cutpoints_pct=cutpoints,
    low_income_range_pct=[float(cohort.low_income_share_pct.min()),float(cohort.low_income_share_pct.max())],
    regulatory_samples=regulatory.groupby('category').size().to_dict(),
    gdp_median_income_spearman=float(spearmanr(municipalities.gdp_pc_brl,municipalities.median_household_income_pc_brl).statistic),
    preliminary_census_sample_estimates=True,household_outage_assignment=False,
    inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((P/'raw').glob('ibge_income*.json'))})
(M/'preparation_audit.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in checks.items() if k!='inputs'},indent=2))
