"""Audited source-native country pairs; no pooled regression or implicit harmonization."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
P=Path(__file__).parent; V=P.parents[1]; R=P/'raw'; O=V/'source_data'
receipts=json.loads((P/'acquisition_receipts.json').read_text())
for x in receipts:
    if x['filename']=='brazil_2024_management_report.pdf':
        old=R/x['filename']; new=R/'brazil_2024_management_report.blocked.html'
        if old.exists():old.rename(new)
        x['filename']=new.name;x['content_verified']='403 HTML; not a PDF or numerical evidence'
    elif x['filename']=='brazil_government_2024_release.html':x['content_verified']='Election-period temporary restriction page; HTTP 200 is not successful article access'
    else:x['content_verified']='Expected document content inspected'
(P/'acquisition_receipts.json').write_text(json.dumps(receipts,indent=2))
lookup={x['filename']:x for x in receipts}
rows=[]
for year,f,h,scope,file,where in [
 (2023,1.64,128/60,'long_interruptions_gt3min','norway_2023_system_operation.pdf','Printed p.19; PDF physical page35; Figure14 and adjacent paragraph'),
 (2025,3.7,213/60,'all_long_and_short_interruptions','norway_statistics_2026.html','Main text: 3.7 total = 2.0 long + 1.7 short; total duration 3h33min')]:
    rows.append(dict(iso3='NOR',country='Norway',year=year,geographic_scope='National network end users',frequency_per_customer_year=f,service_loss_hours_per_customer_year=h,derived_mean_duration_hours=h/f,frequency_definition='Average interruptions per end user',duration_definition='Average total interruption time per end user; hours',inclusion_scope=scope,planned_included=True,unplanned_included=True,major_events='No major-event exclusion identified in national published total; large events retained',scope_comparable_across_extracted_years=False,source_native_pair_verified=True,admit_to_pooled_cross_country_decomposition=False,source_url=lookup[file]['url'],source_file=str((R/file).relative_to(V)),source_location=where,access_date='2026-10-02',precision_note='Rounded narrative values; derived duration is not an independently observed restoration measure'))
pd.DataFrame(rows).to_csv(O/'recent_geographic_v31_verified_pairs.csv',index=False)
f=pd.read_excel(R/'finland_national_statistics.xlsx',sheet_name='Statistics',header=None)
fi=[]
for r in range(3,len(f)):
    h=float(f.iloc[r,31]); c=float(f.iloc[r,32]); d=float(f.iloc[r,36]);
    fi.append(dict(iso3='FIN',year=int(f.iloc[r,0]),customers=int(f.iloc[r,26]),published_interruption_time_h=h,published_interruption_count=c,published_caidi=d,derived_time_over_count=h/c,derived_count_over_time=c/h,published_caidi_matches_inverse=np.isclose(d,c/h),eligible_as_customer_weighted_pair=False,reason='2024 national values reproduce energy-weighted medium-voltage statistics; count includes auto-reclosures; CAIDI field equals inverse identity',source_url=lookup['finland_national_statistics.xlsx']['url'],sheet='Statistics',time_cell=f'AF{r+1}',count_cell=f'AG{r+1}',caidi_cell=f'AK{r+1}',access_date='2026-10-02'))
pd.DataFrame(fi).to_csv(O/'recent_geographic_v31_finland_quarantined_national.csv',index=False)
t=pd.read_excel(R/'finland_technical_2024.xlsx',sheet_name='Sähkön jakeluverkko',header=None)
w=pd.to_numeric(t.iloc[2:,3],errors='coerce')
audit=[]
for inds,name,pub in [([95,97],'annual interruption time; unplanned + planned MV',float(f.iloc[-1,31])),([96,98,99,100],'annual interruption count; unplanned + planned + delayed and fast auto-reclosures MV',float(f.iloc[-1,32]))]:
    components=[]
    for i in inds:
        z=pd.to_numeric(t.iloc[2:,i],errors='coerce');value=(z*w).sum()/w.sum();components.append(value)
        audit.append(dict(year=2024,quantity=name,technical_column_index_zero_based=i,technical_column_header=t.iloc[0,i],aggregation='Distributed-energy weighted mean of published energy-weighted indicators',weight_column='(2.1) distributed energy GWh',component_value=value,published_national_value=pub))
    assert np.isclose(sum(components),pub,rtol=1e-10), (sum(components),pub)
pd.DataFrame(audit).to_csv(O/'recent_geographic_v31_finland_identity_audit.csv',index=False)
screen=[
 dict(country='Norway',iso3='NOR',period='2023;2025 separately',source='NVE/RME',scope='National end users',decision='Two source-native paired annual rows admitted; not pooled because duration scope changes',known_catalog_id='no_nve_reliability',prior_state='metadata_only',new_contribution='New numerical extraction from previously catalogued source; latest published2025',gap_filled='Recent national numerical coverage; not an independent global income-gradient replication',url=lookup['norway_statistics_2026.html']['url']),
 dict(country='Brazil',iso3='BRA',period='2023–2024 candidate',source='ANEEL; government EBC release',scope='National distribution service indices',decision='No numerical row admitted; full article restricted and report403; definitions confirmed',known_catalog_id='Brazil ANEEL existing data family',prior_state='Existing ANEEL geographic class and cause data already used',new_contribution='Current national source screening and explicit access audit only',gap_filled='None; not a new independent source',url=lookup['brazil_continuity_definitions.html']['url']),
 dict(country='Finland',iso3='FIN',period='2016–2024 workbook;2024 technical reconstruction',source='Energiavirasto',scope='National distribution statistical summary',decision='Quarantined as customer-weighted reliability pair; energy weighting and auto-reclosure scope verified',known_catalog_id='fi_regulator_reliability',prior_state='2021–2024 technical files already downloaded',new_contribution='National2016–2024 workbook archived; exact2024 weighting/identity audit',gap_filled='None for compatible SAIFI/SAIDI; useful source-native energy-weighted context',url=lookup['finland_regulator_publications.html']['url'])]
pd.DataFrame(screen).assign(access_date='2026-10-02').to_csv(O/'recent_geographic_v31_source_screen.csv',index=False)
print('Verified source-native pairs:2 Norway. Quarantined Finnish rows:9. Brazil admitted:0. All identity assertions pass.')
