"""Reconcile fine-cause aggregates and retain distinctions among native count units."""
from pathlib import Path
import hashlib
import json
import re
import duckdb
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
ROOT=P.parents[1]
D=P/'data'; M=P/'metadata'; RAW=P/'raw'
z=pd.read_parquet(RAW/'fato_causa_mensal_final.parquet')
c=pd.read_parquet(RAW/'dim_causa_interrupcao_final.parquet')
assert c.CausaKey.is_unique

def classify(row):
    text=' | '.join(str(row[k]) for k in ['Origem','Programacao','GrupoCausa','CausaDetalhada'])
    patterns=[('Wind',r'\bVENTO\b|\bVENDAVAL\b'),('Lightning',r'DESCARGA ATMOSFERICA'),
              ('Vegetation',r'ARVORE OU VEGETACAO'),('Flood',r'INUNDACAO'),
              ('Erosion',r'EROSAO'),('Fire',r'QUEIMA|INCENDIO|INCENCIO|INCENCIA'),
              ('Operational error',r'FALHA OPERACIONAL|ERRO DE OPERACAO|ERRO OPERACIONAL|SERVICO MAL EXECUTADO'),
              ('Explicit third-party activity',r'ABALROAMENTO|VANDALISMO|ROUBO|\bPIPA\b|LIGACAO CLANDESTINA|EMPRESA.{0,15}SERVICO'),
              ('Other environment',r'MEIO AMBIENTE'),
              ('Equipment or electrical',r'PROPRIA[S]? (DO )?SISTEMA|FALHA DE MATERIAL OU EQUIPAMENTO'),
              ('Supply constraint',r'ALIVIO DE CARGA|RACIONAMENTO|INTERRUPCAO EMERGENCIAL DO SUPRIMENTO|FALTA TENSAO DA SUPRIDORA'),
              ('Other third-party',r'TERCEIRO')]
    for label,pattern in patterns:
        if re.search(pattern,text):return label
    return 'Other or unresolved'

c['unplanned_cause']=c.apply(classify,axis=1)
c.to_csv(D/'native_cause_crosswalk.csv',index=False)
z=z.merge(c,on='CausaKey',how='left',validate='many_to_one')
z['cause']=z.unplanned_cause.fillna('Other or unresolved')
z.loc[z.TipoInterrupcao.eq('Programada'),'cause']='Planned'
assert set(z.TipoInterrupcao)=={'Programada','Não Programada'}
z['year']=z.DataKey//10000
z['month']=z.DataKey//100%100
z['record_hours']=z.DuracaoTotalHoras
inverted=z.record_hours/z.DuracaoMediaHoras.where(z.DuracaoMediaHoras.gt(0))
assert np.nanmax(abs(inverted-inverted.round()))<1e-6
assert (inverted.dropna().round()>=z.loc[inverted.notna(),'QtdInterrupcoes']).all()
z['zero_total_cell']=z.record_hours.eq(0)
assert (z.loc[z.zero_total_cell,'DuracaoMediaHoras']==0).all()
z['record_count_lower']=inverted.fillna(z.QtdInterrupcoes).round().astype('int64')
assert z.record_hours.ge(0).all() and z.record_count_lower.ge(1).all()
z['voltage_group']=np.select([z.NivelTensao.le(0),z.NivelTensao.lt(1000),z.NivelTensao.lt(69000)],
    ['Unspecified','Below 1 kV','1 to below 69 kV'],default='69 kV or above')

old=ROOT/'07_analysis/brazil_cause_economy_20261002/raw/Dados/ouro/mart_mensal.parquet'
con=duckdb.connect();con.execute('SET threads=2')
con.register('z',z)
monthly=con.sql(f"""WITH a AS (SELECT DataKey,IdeConjunto,sum(QtdInterrupcoes) AS identifiers,
    sum(record_count_lower) AS records_lower,sum(record_hours) AS hours,
    sum(QtdInterrupcoes) FILTER(WHERE zero_total_cell) AS zero_identifiers
    FROM z GROUP BY ALL), b AS (
    SELECT CAST(strftime(mes,'%Y%m%d') AS INTEGER) AS DataKey,conjunto AS IdeConjunto,
    sum(interrupcoes) AS records_reference,sum(duracao_total_min)/60 AS hours_reference,
    count(distinct distribuidora) AS reference_agents
    FROM read_parquet('{old}') WHERE year(mes) IN(2024,2025) GROUP BY ALL)
    SELECT * FROM a FULL JOIN b USING(DataKey,IdeConjunto)""").df()
assert monthly[['records_lower','records_reference','hours','hours_reference']].notna().all().all()
monthly['unresolved_count']=monthly.records_reference-monthly.records_lower
monthly['hour_difference']=monthly.hours-monthly.hours_reference
monthly['exact_duration_transport']=abs(monthly.hour_difference)<1e-6
assert monthly.unresolved_count.ge(0).all()
assert monthly.reference_agents.eq(1).all()
monthly.to_parquet(D/'monthly_reconciliation.parquet',index=False)
monthly.to_csv(D/'monthly_reconciliation.csv',index=False)
z=z.merge(monthly[['DataKey','IdeConjunto','unresolved_count','exact_duration_transport']],
    on=['DataKey','IdeConjunto'],validate='many_to_one')
# Do not allocate unreconstructed zero-duration records to a preferred cause.
zero_options=z[z.zero_total_cell].groupby(['DataKey','IdeConjunto','cause']).size().reset_index(name='zero_cells')
zero_options=zero_options.merge(monthly[['DataKey','IdeConjunto','unresolved_count','exact_duration_transport']],
    on=['DataKey','IdeConjunto'],validate='many_to_one')
zero_options.to_csv(D/'zero_duration_allocation_bounds.csv',index=False)

area=z.groupby('IdeConjunto').agg(months=('DataKey','nunique'),record_count_lower=('record_count_lower','sum'),
    record_hours=('record_hours','sum'),all_duration_transport=('exact_duration_transport','all')).reset_index()
area=area.merge(monthly.groupby('IdeConjunto').agg(unresolved_count=('unresolved_count','sum')).reset_index(),
    on='IdeConjunto',validate='one_to_one')
area.to_csv(D/'area_source_support.csv',index=False)

cols=['DataKey','year','month','IdeConjunto','DscConjunto','TipoInterrupcao','MotivoCodigo','CausaKey',
      'NivelTensao','voltage_group','cause','QtdInterrupcoes','record_count_lower','record_hours',
      'DuracaoMediaHoras','MaiorInterrupcaoHoras','zero_total_cell','exact_duration_transport']
z[cols].to_parquet(D/'fine_cause_cells.parquet',index=False)
native=z.groupby(['year','cause'],observed=True).agg(cells=('DataKey','size'),
    native_identifiers=('QtdInterrupcoes','sum'),records_lower=('record_count_lower','sum'),
    record_hours=('record_hours','sum')).reset_index()
native['duration_upper_h']=native.record_hours/native.records_lower
native.to_csv(D/'all_source_cause_totals.csv',index=False)
manifest=dict(cells=len(z),years=sorted(z.year.unique().tolist()),native_identifier_sum=int(z.QtdInterrupcoes.sum()),
    reconstructed_record_lower_sum=int(z.record_count_lower.sum()),reference_record_sum=int(monthly.records_reference.sum()),
    unreconstructed_records=int(monthly.unresolved_count.sum()),unreconstructed_fraction=float(monthly.unresolved_count.sum()/monthly.records_reference.sum()),
    zero_total_cells=int(z.zero_total_cell.sum()),missing_native_cause_cells=int(z.CausaKey.isna().sum()),
    all_cause_area_months=len(monthly),duration_transports_exact=int(monthly.exact_duration_transport.sum()),
    nonmatching_duration_cells=json.loads(monthly[~monthly.exact_duration_transport].to_json(orient='records')),
    native_duplicate_display_keys=int(z.duplicated(['DataKey','IdeConjunto','TipoInterrupcao','MotivoCodigo','CausaKey','NivelTensao']).sum()),
    cause_groups=sorted(z.cause.unique()),
    semantics={'frequency':'Source record rate, not regulatory customer frequency',
      'duration':'Record weighted; no customer weighting or T90',
      'loss':'Cumulative record-hours, not customer-hours or energy loss',
      'independent_transport':'Same underlying ANEEL observations; not an independent statistical replication',
      'published_code':'Preserved publisher script does not generate the final released schema verbatim; reconciliation uses actual released numerical columns',
      'economic_results_fitted':False})
manifest['inputs']=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,
    sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in [RAW/'fato_causa_mensal_final.parquet',RAW/'dim_causa_interrupcao_final.parquet',old]]
(M/'source_validation.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in manifest.items() if k not in ['inputs','cause_groups','semantics']},indent=2))
