"""Outcome-year public geometry, pre-event municipal GDP and common reporting support."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd

P=Path(__file__).resolve().parents[1];ROOT=P.parents[1]
D=P/'data';M=P/'metadata';GP=P/'geometry_2024'
B=ROOT/'07_analysis/economic_hazard_strengthening_20260928/components/brazil_panel/data'
receipts=json.loads((GP/'metadata/all_extraction_manifest.json').read_text())
records=[r for r in receipts if r['status']=='verified_native_2024_CONJ']
parts=[gpd.read_parquet(r['geometry_path']).to_crs(4326) for r in records]
geo=gpd.GeoDataFrame(pd.concat(parts,ignore_index=True),crs=4326)
geo['IdeConjunto']=pd.to_numeric(geo.COD_ID,errors='coerce')
good=geo.IdeConjunto.notna()&geo.geometry.notna()&~geo.geometry.is_empty&geo.geometry.is_valid
bad=geo[~good].drop(columns='geometry')
bad.to_csv(D/'geometry_invalid.csv',index=False)
geo=geo[good].copy();geo['IdeConjunto']=geo.IdeConjunto.astype(int)
dups=geo.IdeConjunto.duplicated(keep=False)
geo[dups].drop(columns='geometry').to_csv(D/'geometry_duplicate_ids.csv',index=False)
geo=geo[~dups].copy()
geo.to_parquet(D/'native_service_areas_2024.geoparquet',index=False)

# Native legal identity and observed monthly customer denominators.
native=pd.read_parquet(B/'national_monthly_panel.parquet')
native=native[native.AnoIndice.isin([2024,2025])].copy()
native['IdeConjunto']=native.IdeConjUndConsumidoras.astype(int)
native['DataKey']=native.AnoIndice*10000+native.NumPeriodoIndice*100+1
def cnpj(x):return pd.to_numeric(x,errors='coerce').astype('Int64').astype('string').str.zfill(14)
native['cnpj']=cnpj(native.NumCNPJ)
dim=pd.read_parquet(P/'raw/dim_conjunto.parquet')
dim['release_cnpj']=cnpj(dim.NumCNPJ)
assert dim.IdeConjunto.is_unique
native=native.merge(dim[['IdeConjunto','release_cnpj']],on='IdeConjunto',how='left',validate='many_to_one')
native['identity_match']=native.cnpj.eq(native.release_cnpj).fillna(False)
strata=['XP','XN','IP','IND','INE','INC','INO','IPC','XPC','XNC']
native['all_index_frequency']=native[['FEC'+s for s in strata]].sum(axis=1,min_count=10)
native['all_index_hours']=native[['DEC'+s for s in strata]].sum(axis=1,min_count=10)
native['native_zero']=native.all_index_frequency.eq(0)&native.all_index_hours.eq(0)
monthly=pd.read_parquet(D/'monthly_reconciliation.parquet')
native=native.merge(monthly,on=['IdeConjunto','DataKey'],how='left',validate='one_to_one')
native['source_observed']=native.records_reference.notna()
native['verified_zero_month']=~native.source_observed&native.native_zero
native['supported_month']=native.source_observed|native.verified_zero_month
native['duration_transport_ok']=native.exact_duration_transport.fillna(False).astype(bool)|native.verified_zero_month
native['customer_valid']=native.NumCon.gt(0)
native.to_parquet(D/'native_monthly_support.parquet',index=False)
support=native.groupby('IdeConjunto').agg(months=('DataKey','nunique'),
    supported_months=('supported_month','sum'),source_observed_months=('source_observed','sum'),
    verified_zero_months=('verified_zero_month','sum'),all_identity_match=('identity_match','all'),
    cnpj_count=('cnpj','nunique'),cnpj=('cnpj','first'),SigAgente=('SigAgente','first'),
    all_customer_valid=('customer_valid','all'),all_duration_transport=('duration_transport_ok','all'),
    mean_customers=('NumCon','mean'),sum_month_customers=('NumCon','sum'),
    unresolved_count=('unresolved_count','sum'),records_reference=('records_reference','sum')).reset_index()
support['customer_years']=support.sum_month_customers/12
support['common_24_months']=(support.months.eq(24)&support.supported_months.eq(24)&support.source_observed_months.gt(0)
    &support.all_identity_match&support.cnpj_count.eq(1)&support.all_customer_valid&support.all_duration_transport)
support.to_csv(D/'area_reporting_support.csv',index=False)

# Apply the established spatial rule to contemporary footprints, before modelling.
area=geo.merge(support,on='IdeConjunto',validate='one_to_one')
area=area[area.common_24_months].copy().to_crs(5880)
area['area_m2']=area.geometry.area
mun=gpd.read_parquet(B/'municipalities_brazil_2021.geoparquet').to_crs(5880)
econ=pd.read_csv(B/'municipal_economy_2021.csv',dtype={'municipality_code':str})
mun=mun.merge(econ,on='municipality_code',validate='one_to_one')
mun['municipality_area_m2']=mun.geometry.area
cross=gpd.overlay(area[['IdeConjunto','area_m2','geometry']],
    mun[['municipality_code','municipality_area_m2','gdp_pc_brl','population','gdp_1000_brl','geometry']],
    how='intersection',keep_geom_type=False)
cross['intersection_m2']=cross.geometry.area
cross=cross[cross.intersection_m2>1].copy()
cross['fraction_area']=cross.intersection_m2/cross.area_m2
cross['fraction_municipality']=cross.intersection_m2/cross.municipality_area_m2
cross.drop(columns='geometry').to_parquet(D/'municipality_crosswalk_2024_2021.parquet',index=False)
context=[]
for cid,x in cross.groupby('IdeConjunto'):
    dom=x.loc[x.intersection_m2.idxmax()];coverage=x.fraction_area.sum()
    pop=(x.population*x.fraction_municipality).sum()
    gdp=(x.gdp_1000_brl*1000*x.fraction_municipality).sum()
    context.append(dict(IdeConjunto=cid,municipality_code=str(dom.municipality_code),
        gdp_pc_brl=dom.gdp_pc_brl,dominant_area_fraction=dom.fraction_area,municipal_coverage=coverage,
        strict_spatial_assignment=bool(dom.fraction_area>=.9 and .95<=coverage<=1.05),
        broad_gdp_pc_uniform_population=gdp/pop if pop>0 else np.nan,
        service_area_km2=dom.area_m2/1e6,economic_year=2021,geometry_year=2024))
context=pd.DataFrame(context)
area=area.merge(context,on='IdeConjunto',validate='one_to_one')
area['state_code']=area.municipality_code.str[:2]
area['density']=area.mean_customers/area.service_area_km2
area.to_crs(4326).to_parquet(D/'source_geographic_cohort.geoparquet',index=False)
primary=area[area.strict_spatial_assignment&area.gdp_pc_brl.gt(0)&area.density.gt(0)].copy()
primary.to_crs(4326).to_parquet(D/'economic_geographic_cohort.geoparquet',index=False)
primary.drop(columns='geometry').to_parquet(D/'economic_cohort.parquet',index=False)
primary.drop(columns='geometry').to_csv(D/'economic_cohort.csv',index=False)
geoident=area.groupby('DIST').agg(areas=('IdeConjunto','size'),legal_identities=('cnpj','nunique')).reset_index()
geoident.to_csv(D/'geometry_legal_identity_check.csv',index=False)
qa=dict(archives_attempted=len(receipts),archives_verified=len(records),
    official_geometry_areas=len(geo),invalid_geometry=len(bad),duplicate_geometry_rows=int(dups.sum()),
    source_reporting_areas=int(support.common_24_months.sum()),
    geographic_common_areas=len(area),economic_areas=len(primary),municipalities=primary.municipality_code.nunique(),
    states=primary.state_code.nunique(),reporting_contexts=primary.cnpj.nunique(),
    customer_years=float(primary.customer_years.sum()),verified_zero_months=int(primary.verified_zero_months.sum()),
    positive_reporting_months=int(primary.source_observed_months.sum()),
    boundary_reference='Official 2024-12-31; constant boundaries throughout 2024–2025 not asserted',
    economic_context='2021 municipal GDP on its original geography overlaid with 2024 service areas',
    statistical_models_fitted=False)
(M/'geographic_validation.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False))
print(json.dumps(qa,indent=2),flush=True)
