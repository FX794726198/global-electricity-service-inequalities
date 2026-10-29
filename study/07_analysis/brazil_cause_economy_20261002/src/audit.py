"""Audit downloaded aggregate cubes against independently acquired continuity indices.

Does not execute publisher code or infer individual records from aggregate tables.
"""
from pathlib import Path
import hashlib
import json
import duckdb
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
ROOT = P.parents[1]
D, M = P / 'data', P / 'metadata'
B = ROOT / '07_analysis/economic_hazard_strengthening_20260928/components/brazil_panel/data'
SOURCE = P / 'raw/Dados/ouro/mart_mensal.parquet'
C = duckdb.connect()
C.execute('SET threads=2')
C.execute("SET memory_limit='2GB'")
C.execute(f"CREATE VIEW m AS SELECT * FROM read_parquet('{SOURCE}')")
audit = {'source_role': 'third-party derived monthly cube; individual records not acquired',
         'economic_models_fitted': False,
         'quality_definition_url': 'https://www.gov.br/aneel/pt-br/assuntos/distribuicao/qualidade-do-fornecimento-de-energia-eletrica'}

def save(z, name):
    z.to_csv(D / f'{name}.csv', index=False)
    z.to_parquet(D / f'{name}.parquet', index=False)

def qry(sql, name):
    z = C.sql(sql).df()
    save(z, name)
    return z

def ratio_summary(z, raw, reference, label):
    x = z.loc[z[raw].gt(0) & z[reference].gt(0), [raw, reference]]
    ratio = x[raw] / x[reference]
    return dict(sample=label, raw=raw, reference=reference, n=len(x),
                median_ratio=ratio.median(), p10=ratio.quantile(.1), p90=ratio.quantile(.9),
                within_5pct=ratio.between(.95, 1.05).mean(),
                within_10pct=ratio.between(.9, 1.1).mean(),
                pearson=x.corr().iloc[0, 1], spearman=x.corr(method='spearman').iloc[0, 1])

quality = qry("""SELECT count(*) AS cells, sum(interrupcoes) AS represented_records,
    min(mes) AS start_month, max(mes) AS end_month,
    count(distinct conjunto) AS service_codes, count(distinct distribuidora) AS providers,
    count(*) FILTER(WHERE consumidores IS NULL OR consumidor_minuto IS NULL) AS null_sum_cells,
    count(*) FILTER(WHERE consumidores < 0 OR duracao_total_min < 0 OR consumidor_minuto < 0) AS negative_sum_cells,
    count(*) FILTER(WHERE consumidores = 0) AS zero_customer_cells,
    count(*) FILTER(WHERE programada IS NULL) AS unknown_planned_status_cells,
    count(*) FILTER(WHERE interrupcoes <= 0) AS nonpositive_count_cells
    FROM m""", 'cube_quality')
audit['cube'] = json.loads(quality.to_json(orient='records', date_format='iso'))[0]
audit['duplicate_keys'] = int(C.sql("""SELECT count(*) FROM (
    SELECT conjunto, mes, causa, programada, causa_climatica, distribuidora, count(*) AS n
    FROM m GROUP BY ALL HAVING n>1)""").fetchone()[0])
assert audit['duplicate_keys'] == 0

qry("""SELECT year(mes) AS year, count(*) AS cells, sum(interrupcoes) AS represented_records,
    count(distinct conjunto) AS service_codes, count(distinct distribuidora) AS providers,
    count(distinct mes) AS n_months, sum(consumidores) AS customer_interruptions,
    sum(consumidor_minuto)/60 AS customer_hours FROM m GROUP BY ALL ORDER BY year""", 'year_coverage')
qry("""SELECT causa, programada, causa_climatica, sum(interrupcoes) AS represented_records,
    sum(consumidores) AS customer_interruptions, sum(consumidor_minuto)/60 AS customer_hours
    FROM m GROUP BY ALL ORDER BY represented_records DESC""", 'native_cause_totals')
qry("""SELECT year(mes) AS year, distribuidora, month(mes) AS month,
    count(distinct conjunto) AS service_codes, sum(interrupcoes) AS represented_records
    FROM m GROUP BY ALL ORDER BY year, distribuidora, month""", 'provider_month_coverage')
qry("""SELECT year(mes) AS year,
    sum(interrupcoes) FILTER(WHERE duracao_total_min/nullif(interrupcoes,0)>4320) AS records_in_long_mean_cells,
    sum(consumidor_minuto) FILTER(WHERE consumidor_minuto/nullif(consumidores,0)>4320)/60 AS hours_in_long_mean_cells,
    sum(consumidor_minuto)/60 AS all_customer_hours
    FROM m GROUP BY ALL ORDER BY year""", 'extreme_aggregate_means')

hist = pd.read_parquet(P/'raw/Dados/ouro/mart_duracao.parquet')
save(hist.groupby(['ano', 'causa_climatica', 'faixa_min'], observed=True).interrupcoes.sum().reset_index(),
     'record_duration_bins')
audit['histogram_represented_records'] = int(hist.interrupcoes.sum())
audit['histogram_difference_from_monthly'] = int(audit['cube']['represented_records']) - int(hist.interrupcoes.sum())
audit['histogram_note'] = 'Histogram year is source competency year; monthly year is onset year. Bins cannot isolate the regulatory >=3-minute threshold.'
del hist

z = C.sql('SELECT * FROM m WHERE year(mes)=2021').df()
z['month'] = z.mes.dt.month
z['customer_hours'] = z.consumidor_minuto/60
z['record_hours'] = z.duracao_total_min/60
causes = {
    'MEIO AMBIENTE': 'Other environmental',
    'PROPRIAS DO SISTEMA': 'System and equipment', 'PROPRIA DO SISTEMA': 'System and equipment',
    'TERCEIROS': 'Third party', 'TERCEIRO': 'Third party',
    'FALHA OPERACIONAL': 'Operational fault', 'FALHA DE CONEXAO': 'Connection fault',
}
z['cause_group'] = z.causa.map(causes).fillna('Unresolved or other')
z.loc[z.causa_climatica, 'cause_group'] = 'Environment-associated'
z.loc[z.programada.eq(True), 'cause_group'] = 'Planned'
cross = z.groupby(['causa','programada','causa_climatica','cause_group'], observed=True,dropna=False)[['interrupcoes','consumidores','customer_hours']].sum().reset_index()
assert cross.interrupcoes.sum()==z.interrupcoes.sum()
assert np.isclose(cross.customer_hours.sum(),z.customer_hours.sum())
save(cross, 'cause_crosswalk_2021')

month = pd.read_parquet(B/'national_monthly_panel.parquet')
month = month[month.AnoIndice.eq(2021)].copy()
month = month.rename(columns={'IdeConjUndConsumidoras':'conjunto','SigAgente':'distribuidora','NumPeriodoIndice':'month'})
assert not month.duplicated(['conjunto','distribuidora','month']).any()
month['F_index_all'] = month[['FEC'+s for s in ['XP','XN','IP','IND','INE','INC','INO','IPC','XPC','XNC']]].sum(axis=1, min_count=10)
month['H_index_all'] = month[['DEC'+s for s in ['XP','XN','IP','IND','INE','INC','INO','IPC','XPC','XNC']]].sum(axis=1, min_count=10)
zm = z.merge(month[['conjunto','distribuidora','month','NumCon']], on=['conjunto','distribuidora','month'], how='left', validate='many_to_one')
zm['F_cell'] = zm.consumidores/zm.NumCon.where(zm.NumCon.gt(0))
zm['H_cell'] = zm.customer_hours/zm.NumCon.where(zm.NumCon.gt(0))
monthly = z.groupby(['conjunto','distribuidora','month'], observed=True).agg(
    represented_records=('interrupcoes','sum'), customer_interruptions=('consumidores','sum'),
    customer_hours=('customer_hours','sum')).reset_index()
monthly = month.merge(monthly,on=['conjunto','distribuidora','month'],how='outer',indicator=True,validate='one_to_one')
# Absence is not set to zero when the raw reporting universe is unverified.
monthly['F_cube'] = monthly.customer_interruptions/monthly.NumCon.where(monthly.NumCon.gt(0))
monthly['H_cube'] = monthly.customer_hours/monthly.NumCon.where(monthly.NumCon.gt(0))
save(monthly[['conjunto','distribuidora','month','NumCon','F_index_all','H_index_all',
              'represented_records','customer_interruptions','customer_hours','F_cube','H_cube','_merge']],
     'monthly_reconciliation_2021')

annual = z.groupby(['conjunto','distribuidora'], observed=True).agg(
    represented_records=('interrupcoes','sum'), customer_interruptions=('consumidores','sum'),
    customer_hours=('customer_hours','sum'), record_hours=('record_hours','sum'),
    observed_months=('month','nunique')).reset_index()
official = pd.read_parquet(B/'economic_analysis_2021_all_contexts.parquet').rename(
    columns={'IdeConjUndConsumidoras':'conjunto','SigAgente':'distribuidora'})
joined = annual.merge(official,on=['conjunto','distribuidora'],how='left',indicator=True,validate='one_to_one')
joined['F_cube_annual'] = joined.customer_interruptions/joined.mean_customers
joined['H_cube_annual'] = joined.customer_hours/joined.mean_customers
joined['D_cube_annual'] = joined.customer_hours/joined.customer_interruptions
joined['frequency_ratio'] = joined.F_cube_annual/joined.FEC_all_10_strata
joined['burden_ratio'] = joined.H_cube_annual/joined.DEC_all_10_strata
joined['base_strict_cohort'] = (joined.strict_spatial_assignment.fillna(False).astype(bool)
    & joined.complete_common_12.fillna(False).astype(bool) & joined.gdp_pc_brl.gt(0)
    & joined.mean_customers.gt(0) & joined.service_area_km2.gt(0))
joined['full_12_month_cube'] = joined.observed_months.eq(12)
joined['burden_concordant_10pct'] = joined.burden_ratio.between(.9,1.1)
joined['frequency_concordant_10pct'] = joined.frequency_ratio.between(.9,1.1)
save(joined, 'annual_reconciliation_2021')

summaries=[]
for label, sample in [('All exact agent-code matches',joined),
                      ('Existing strict geographic cohort',joined[joined.base_strict_cohort]),
                      ('Strict cohort and 12 cube months',joined[joined.base_strict_cohort & joined.full_12_month_cube])]:
    for raw, reference in [('F_cube_annual','FEC_all_10_strata'),('H_cube_annual','DEC_all_10_strata')]:
        summaries.append(ratio_summary(sample,raw,reference,label))
save(pd.DataFrame(summaries),'reconciliation_summary')

cause = z.groupby(['conjunto','distribuidora','cause_group'],observed=True).agg(
    represented_records=('interrupcoes','sum'),customer_interruptions=('consumidores','sum'),
    customer_hours=('customer_hours','sum'),record_hours=('record_hours','sum')).reset_index()
save(cause,'cause_annual_2021')
match = joined[joined._merge.eq('both')]
save(match.groupby('distribuidora').agg(areas=('conjunto','size'),
    frequency_ratio_median=('frequency_ratio','median'), burden_ratio_median=('burden_ratio','median'),
    full_month_fraction=('full_12_month_cube','mean')).reset_index(),'provider_reconciliation')
audit['exact_matches_2021'] = len(match)
audit['strict_matches_2021'] = int(joined.base_strict_cohort.sum())
audit['strict_12_months_2021'] = int((joined.base_strict_cohort & joined.full_12_month_cube).sum())
audit['strict_12_burden_concordant_2021'] = int((joined.base_strict_cohort & joined.full_12_month_cube & joined.burden_concordant_10pct).sum())
audit['admission'] = {
    'annual_cause_frequency': 'Not admitted as regulatory FEC: retained cube lacks the >=3-minute filter; considerable frequency mismatch.',
    'mean_duration': 'Not admitted as a sustained-interruption mean: its customer-interruption denominator has the same unresolved threshold.',
    'cause_customer_hours': 'Eligible for explicitly aggregate, source-reconciled analysis; economic model requires reporting support and concordance sensitivity.',
    'natural_versus_human': 'Not identified. Environment-associated flag pools six detail labels; unknown flag=false is not anthropogenic.',
    'fine_hazards': 'Not available from acquired monthly cube.',
    'customer_recovery_T90': 'Not available from these aggregates.'}
audit['files'] = []
for path in [SOURCE,B/'national_monthly_panel.parquet',B/'economic_analysis_2021_all_contexts.parquet']:
    with path.open('rb') as handle:
        sha = hashlib.file_digest(handle,'sha256').hexdigest()
    audit['files'].append(dict(path=str(path.relative_to(ROOT)),sha256=sha,bytes=path.stat().st_size))
(M/'source_reconciliation.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
print(json.dumps({k:audit[k] for k in ['cube','strict_matches_2021','strict_12_months_2021','strict_12_burden_concordant_2021','histogram_difference_from_monthly']},indent=2))
print(pd.DataFrame(summaries).round(4).to_string(index=False))
