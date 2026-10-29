"""Source-defined grid context and exact service-state inequality accounting."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
import pyreadstat

P = Path(__file__).resolve().parents[1]
ROOT = P.parents[1]
BASE = ROOT / '07_analysis/household_blackout_inequality_20261002'
spec = importlib.util.spec_from_file_location('parent', BASE/'src/analyse_afrobarometer.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
STATES = ['unconnected_limited_grid', 'unconnected_grid_present',
          'unconnected_grid_unknown', 'connected_unreliable', 'connected_reliable']

def weighted(g, col):
    return float(np.average(g[col].to_numpy(float), weights=g.weight.to_numpy(float)))

def effective(g):
    w = g.weight.to_numpy(float)
    return float(w.sum()**2/(w@w)) if len(w) else 0.

def summary(df, cols):
    return parent.summarize(df, cols)

def partition(d, poverty_col):
    rows = []
    for (iso, group), sub in d.dropna(subset=[poverty_col, 'state']).groupby(['iso3', poverty_col]):
        row = dict(iso3=iso, country=sub.country.iloc[0], group=int(group), n=len(sub),
                   effective_n=effective(sub), weight_sum=float(sub.weight.sum()))
        row.update({k:weighted(sub,k) for k in STATES})
        row['sum_states'] = sum(row[k] for k in STATES)
        rows.append(row)
    tab = pd.DataFrame(rows)
    contrasts = []
    for iso, g in tab.groupby('iso3'):
        gg = g.set_index('group')
        if not {0,3}.issubset(gg.index):
            continue
        low, high = gg.loc[0],gg.loc[3]
        row=dict(iso3=iso,country=low.country,n_no=int(low.n),n_high=int(high.n),
                 reliable_no=low.connected_reliable,reliable_high=high.connected_reliable,
                 service_gap=low.connected_reliable-high.connected_reliable)
        for k in STATES[:-1]:row[k+'_gap']=high[k]-low[k]
        row['identity_error']=row['service_gap']-sum(row[k+'_gap'] for k in STATES[:-1])
        contrasts.append(row)
    return tab,pd.DataFrame(contrasts)

def main():
    for s in ['data','metadata','reports','source_data']:(P/s).mkdir(parents=True,exist_ok=True)
    src=BASE/'raw/afrobarometer/afrobarometer_r9.sav'
    old=pd.read_parquet(BASE/'data/afrobarometer_analysis_sample.parquet')
    native,meta=pyreadstat.read_sav(str(src),usecols=['COUNTRY','RESPNO','EA_SVC_A','Q92A','Q92B','withinwt_hh'])
    native=native.loc[native.withinwt_hh.gt(0)].reset_index(drop=True)
    assert len(native)==len(old)
    assert np.array_equal(native.COUNTRY.map(parent.COUNTRY_ISO),old.iso3)
    assert np.allclose(native.withinwt_hh,old.weight)
    assert np.allclose(native.Q92A.where(native.Q92A.isin([0,1])),old.connection,equal_nan=True)
    d=old.copy()
    d['respondent_id']=native.RESPNO
    d['grid_code_native']=native.EA_SVC_A
    d['grid_context']=native.EA_SVC_A.where(native.EA_SVC_A.isin([0,1]))
    conditions=[d.state.eq(0)&d.grid_context.eq(0),d.state.eq(0)&d.grid_context.eq(1),
                d.state.eq(0)&d.grid_context.isna(),d.state.eq(1),d.state.eq(2)]
    for k,mask in zip(STATES,conditions):d[k]=mask.astype(float).where(d.state.notna())
    assert np.allclose(d.loc[d.state.notna(),STATES].sum(axis=1),1)
    d.to_parquet(P/'data/respondent_service_states.parquet',index=False)
    primary=pd.read_csv(BASE/'data/afrobarometer_country_contrasts.csv')
    admitted=set(primary.loc[primary.admitted,'iso3'])
    tab,co=partition(d,'poverty')
    co['admitted_parent']=co.iso3.isin(admitted)
    tab.to_csv(P/'data/country_group_states.csv',index=False)
    co.to_csv(P/'data/country_state_contrasts.csv',index=False)
    matched=co.merge(primary[['iso3','service_gap']],on='iso3',suffixes=('','_parent'),validate='one_to_one')
    assert np.allclose(matched.service_gap,matched.service_gap_parent,atol=1e-12)
    assert co.identity_error.abs().max()<1e-12
    selected=co[co.admitted_parent].copy()
    cols=['service_gap']+[k+'_gap' for k in STATES[:-1]]
    result={'primary':summary(selected,cols),'fixed_countries':sorted(admitted)}
    # Conditional observed contrasts: local grid broadly accessible, not matched neighbourhoods.
    cond=[];conditional_groups=[]
    for iso in sorted(admitted):
        g=d[d.iso3.eq(iso)&d.grid_context.eq(1)&d.state.notna()]
        z={}
        for group in [0,3]:
            q=g[g.poverty.eq(group)]
            z[group]={'n':len(q),'regions':int(q.loc[q.REGION.gt(0),'REGION'].nunique()),
                      'effective_n':effective(q),'connection':weighted(q,'connection') if len(q) else np.nan,
                      'service':weighted(q,'service') if len(q) else np.nan}
            conditional_groups.append(dict(iso3=iso,group=group,**z[group]))
        admit=all(z[x]['n']>=30 and z[x]['regions']>=2 for x in [0,3])
        cond.append(dict(iso3=iso,country=d.loc[d.iso3.eq(iso),'country'].iloc[0],admitted=admit,
                         n_no=z[0]['n'],n_high=z[3]['n'],regions_no=z[0]['regions'],regions_high=z[3]['regions'],
                         connection_no=z[0]['connection'],connection_high=z[3]['connection'],
                         service_no=z[0]['service'],service_high=z[3]['service'],
                         connection_gap=z[0]['connection']-z[3]['connection'],
                         service_gap=z[0]['service']-z[3]['service']))
    cond=pd.DataFrame(cond)
    cond.to_csv(P/'data/conditional_grid_contrasts.csv',index=False)
    pd.DataFrame(conditional_groups).to_csv(P/'data/conditional_group_support.csv',index=False)
    cc=cond[cond.admitted]
    result['conditional_grid']=summary(cc,['connection_no','connection_high','connection_gap','service_no','service_high','service_gap'])
    result['conditional_countries']=cc.iso3.tolist()
    result['same_country_unconditional']=summary(primary[primary.iso3.isin(cc.iso3)],['connection_gap','service_gap'])
    # Preserve the original four-item construct as one declared secondary comparison.
    tab4,co4=partition(d,'poverty4')
    old4=pd.read_csv(BASE/'data/afrobarometer_four_item_contrasts.csv')
    fouradmit=set(old4.loc[old4.admitted,'iso3']) & admitted
    co4['admitted_common']=co4.iso3.isin(fouradmit)
    tab4.to_csv(P/'data/four_item_group_states.csv',index=False)
    co4.to_csv(P/'data/four_item_contrasts.csv',index=False)
    result['construct_common_countries']=sorted(fouradmit)
    result['construct_four_item']=summary(co4[co4.admitted_common],cols)
    result['construct_five_item_common']=summary(co[co.iso3.isin(fouradmit)],cols)
    # Native grid/connection combinations, including locally uncommon grid access.
    cross=(d.assign(grid_native=d.grid_code_native.fillna(-999),conn=d.connection.fillna(-999))
            .groupby(['iso3','grid_native','conn'],dropna=False).agg(n=('weight','size'),weight_sum=('weight','sum')).reset_index())
    cross.to_csv(P/'data/grid_connection_crossclassification.csv',index=False)
    support=d.groupby('iso3').agg(n=('weight','size'),valid_state=('state','count'),
                                 grid_known=('grid_context','count'),poverty_known=('poverty','count'))
    support.to_csv(P/'data/all_country_support.csv')
    (P/'data/summary.json').write_text(json.dumps(result,indent=2))
    co[co.admitted_parent].to_csv(P/'source_data/service_gap_components.csv',index=False)
    tab[tab.iso3.isin(admitted)].to_csv(P/'source_data/service_state_groups.csv',index=False)
    cond.to_csv(P/'source_data/conditional_grid_contrasts.csv',index=False)
    audit={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'source_rows':len(d),
           'fixed_countries':len(admitted),'parent_gap_max_error':float((matched.service_gap-matched.service_gap_parent).abs().max()),
           'state_probability_max_error':float((tab.sum_states-1).abs().max()),
           'gap_identity_max_error':float(co.identity_error.abs().max()),
           'grid_native_value_labels':meta.variable_value_labels['EA_SVC_A'],
           'psu_identifier':'not exposed; not reconstructed',
           'interval_scope':'paired resampling of observed countries; not within-country design-based intervals',
           'numerical_checks_passed':True}
    (P/'metadata/validation.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
