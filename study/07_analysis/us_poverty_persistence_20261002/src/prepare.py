"""Build fixed county/year and event comparison populations before regression."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd,geopandas as gpd
P=Path(__file__).resolve().parents[1];B=P.parents[1];D=P/'data';M=P/'metadata'
inputs=[]
def read(path):
    p=B/path;inputs.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return pd.read_parquet(p)
def save(name,a):a.to_parquet(D/(name+'.parquet'),index=False);a.to_csv(D/(name+'.csv'),index=False)
def main():
    rows=[]
    for line in (P/'raw/saipe_est18all.txt').read_text().splitlines():
        state_code=line[:2].strip().zfill(2);county_code=line[3:6].strip().zfill(3)
        if county_code=='000':continue
        def val(lo,hi):return pd.to_numeric(line[lo-1:hi].strip(),errors='coerce')
        rows.append(dict(fips=state_code+county_code,poverty_pct=val(35,38),poverty_lower90=val(40,43),poverty_upper90=val(45,48),median_income_usd=val(134,139),name=line[193:238].strip(),state=line[239:241]))
    c=pd.DataFrame(rows);assert not c.fips.duplicated().any() and c.fips.str.fullmatch(r'\d{5}').all() and len(c)==3142
    pop_path=B/'07_analysis/economic_hazard_tests_20260928/components/usa/raw/co-est2019-alldata.csv'
    inputs.append(dict(path=str(pop_path),sha256=hashlib.sha256(pop_path.read_bytes()).hexdigest()))
    pop=pd.read_csv(pop_path,encoding='latin1',usecols=['SUMLEV','STATE','COUNTY','POPESTIMATE2018']);pop=pop[pop.SUMLEV.eq(50)].copy()
    pop['fips']=pop.STATE.astype(str).str.zfill(2)+pop.COUNTY.astype(str).str.zfill(3)
    c=c.merge(pop[['fips','POPESTIMATE2018']].rename(columns={'POPESTIMATE2018':'population_2018'}),on='fips',how='left',validate='one_to_one')
    shp=B/'07_analysis/hazard_outage_results_20260926/raw/zenodo_22651795/cb_2018_us_county_500k.shp'
    geo=gpd.read_file(shp)[['GEOID','ALAND','geometry']].rename(columns={'GEOID':'fips'})
    for path in sorted(shp.parent.glob(shp.stem+'.*')):inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    c=c.merge(geo[['fips','ALAND']],on='fips',how='left',validate='one_to_one');c['land_km2']=c.ALAND/1e6
    c['density']=c.population_2018/c.land_km2;c['log_population']=np.log(c.population_2018);c['log_density']=np.log(c.density);c['poverty_5pp']=c.poverty_pct/5
    save('county_context_2018',c)
    a=read('07_analysis/hazard_outage_results_20260926/data/us_analysis_records.parquet')
    assert len(a)==867102 and not a.record_id.duplicated().any() and a.duration.gt(0).all()
    support=a.groupby('fips').agg(records=('record_id','size'),years=('year','nunique'),weather_records=('weather','sum')).reset_index().merge(c,on='fips',how='left',validate='one_to_one')
    support['eligible_geography']=support.years.eq(5)&support.records.ge(20)
    support['complete_context']=support[['poverty_pct','log_population','log_density']].notna().all(axis=1)
    support['admitted']=support.eligible_geography&support.complete_context
    qctx=support.loc[support.admitted,['fips','poverty_pct']].copy()
    qctx['poverty_quartile']=pd.qcut(qctx.poverty_pct,4,labels=['Q1','Q2','Q3','Q4'])
    support=support.merge(qctx[['fips','poverty_quartile']],on='fips',how='left',validate='one_to_one')
    save('county_support',support)
    a=a.merge(support[['fips','admitted','poverty_quartile','poverty_pct','poverty_5pp','state','log_population','log_density']],on='fips',validate='many_to_one')
    a=a[a.admitted].copy();a['long']=a.duration.gt(24)
    stats=a.groupby(['fips','year','weather'],observed=True).agg(n=('record_id','size'),mean_h=('duration','mean'),median_h=('duration','median'),p95_h=('duration',lambda v:v.quantile(.95)),long_share=('long','mean')).reset_index()
    pairs=stats[stats.weather].merge(stats[~stats.weather],on=['fips','year'],suffixes=('_weather','_unlinked'),validate='one_to_one')
    pairs['log_duration_ratio']=np.log(pairs.mean_h_weather/pairs.mean_h_unlinked)
    pairs['long_difference_pp']=100*(pairs.long_share_weather-pairs.long_share_unlinked)
    pairs['five_each']=pairs.n_weather.ge(5)&pairs.n_unlinked.ge(5)
    pairs=pairs.merge(support.drop(columns=['records','years','weather_records']),on='fips',validate='many_to_one')
    pairs['weight']=1/pairs.groupby('fips').fips.transform('size')
    save('county_year_class_summary',stats);save('paired_county_year',pairs)
    county=pairs.groupby('fips').agg(log_duration_ratio=('log_duration_ratio','mean'),long_difference_pp=('long_difference_pp','mean'),paired_years=('year','nunique'),weather_n=('n_weather','sum'),unlinked_n=('n_unlinked','sum')).reset_index().merge(support,on='fips',validate='one_to_one')
    save('county_contrasts',county)
    geo.merge(county,on='fips',how='inner',validate='one_to_one').to_parquet(D/'county_contrasts.geoparquet',index=False)
    # Full-period common-county composition. Both classes required for decomposition.
    cs=a.groupby(['fips','weather'],observed=True).agg(n=('record_id','size'),mean_h=('duration','mean'),long_share=('long','mean')).reset_index()
    cs['share']=cs.n/cs.groupby('fips').n.transform('sum')
    complete=cs.groupby('fips').weather.nunique();cs=cs[cs.fips.isin(complete[complete.eq(2)].index)].copy()
    cs=cs.merge(support[['fips','state','poverty_quartile','poverty_pct']],on='fips',validate='many_to_one')
    # Group mixing weights are average county class shares. Class means are
    # weighted by those shares, so their product preserves equal-county means.
    group=[]
    for (quartile,weather),z in cs.groupby(['poverty_quartile','weather'],observed=True):
        group.append(dict(quartile=str(quartile),weather=bool(weather),counties=z.fips.nunique(),class_share=z.share.mean(),class_duration_h=np.average(z.mean_h,weights=z.share),class_long_share=np.average(z.long_share,weights=z.share)))
    save('county_class_summary',cs);save('poverty_group_composition',pd.DataFrame(group))
    # Validate event identifiers before any event-based estimates.
    links=read('07_analysis/hazard_outage_results_20260926/data/us_unique_weather_links.parquet')
    links['event_key']=links.source_year.astype(str)+'|'+links.state.astype(str)+'|'+links.event_id.astype(str)
    consistency=links.groupby('event_key').agg(starts=('linked_event_start','nunique'),ends=('linked_event_end','nunique'),types=('event_type','nunique'),n=('record_id','size')).reset_index()
    consistency['consistent']=consistency[['starts','ends','types']].eq(1).all(axis=1)
    save('event_identifier_audit',consistency)
    link_unique=links[['record_id','event_key']].drop_duplicates()
    multiplicity=link_unique.groupby('record_id').event_key.nunique().rename('event_count')
    selected=link_unique.merge(multiplicity,on='record_id');selected=selected[selected.event_count.eq(1)&selected.event_key.isin(consistency.loc[consistency.consistent,'event_key'])]
    es=a.merge(selected[['record_id','event_key']],on='record_id',how='inner',validate='one_to_one')
    ec=es.groupby(['fips','event_key'],observed=True).agg(n=('record_id','size'),mean_h=('duration','mean'),long_share=('long','mean')).reset_index()
    esupport=ec.groupby('event_key').agg(counties=('fips','nunique'),records=('n','sum')).reset_index();esupport['admitted']=esupport.counties.ge(5)
    ec=ec[ec.event_key.isin(esupport.loc[esupport.admitted,'event_key'])].merge(support,on='fips',validate='many_to_one')
    ec['log_mean_h']=np.log(ec.mean_h);ec['long_pct']=100*ec.long_share;ec['weight']=1.
    save('event_support',esupport);save('county_event_summary',ec)
    summary=dict(original_records=867102,original_counties=int(support.fips.nunique()),fixed_geographic_counties=int(support.eligible_geography.sum()),missing_context_eligible=int((support.eligible_geography&~support.complete_context).sum()),admitted_records=len(a),admitted_counties=a.fips.nunique(),paired_county_years=len(pairs),paired_counties=pairs.fips.nunique(),paired_states=pairs.state.nunique(),five_each_pairs=int(pairs.five_each.sum()),five_each_counties=pairs.loc[pairs.five_each,'fips'].nunique(),events=len(consistency),inconsistent_events=int((~consistency.consistent).sum()),unique_event_records=len(es),admitted_events=ec.event_key.nunique(),county_events=len(ec),event_counties=ec.fips.nunique(),event_states=ec.state.nunique(),poverty_quartile_edges=np.quantile(qctx.poverty_pct,[0,.25,.5,.75,1]).tolist())
    (M/'support_summary.json').write_text(json.dumps(summary,indent=2));(M/'input_manifest.json').write_text(json.dumps(inputs,indent=2))
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(group).round(3).to_string(index=False))
if __name__=='__main__':main()
