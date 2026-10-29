"""Country-resolved electricity service inequalities; see frozen analysis plan."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import pyreadstat

P = Path(__file__).resolve().parents[1]
SEED, B = 20261002, 4000
COUNTRY_ISO = dict(zip(range(2, 41), 'AGO BEN BWA BFA CPV CMR COG CIV SWZ ETH GAB GMB GHA GIN KEN LSO LBR MDG MWI MLI MRT MUS MAR MOZ NAM NER NGA STP SEN SYC SLE ZAF SDN TZA TGO TUN UGA ZMB ZWE'.split()))
COLS = ['COUNTRY', 'REGION', 'URBRUR_COND', 'Q92A', 'Q92B', 'LivedPoverty', 'LivedPoverty_CAT', 'withinwt_hh', 'DATEINTR'] + ['Q6'+x for x in 'ABCDE']


def mean(g, col):
    x, w = g[col].to_numpy(float), g.weight.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def neff(g):
    w = g.weight.to_numpy(float)
    return float(w.sum()**2 / (w @ w)) if len(w) else 0.


def cells(df, poverty_col):
    out = []
    for (iso, group), g in df.dropna(subset=[poverty_col]).groupby(['iso3', poverty_col]):
        dec = g[g.state.notna()]
        connected = dec[dec.connection == 1]
        reg = connected.REGION.where(connected.REGION > 0)
        out.append(dict(iso3=iso, country=g.country.iloc[0], group=int(group), n=len(g),
                        n_connection=int(g.connection.notna().sum()), n_decomposition=len(dec),
                        n_connected=len(connected), connected_regions=int(reg.nunique()),
                        weight_sum=float(g.weight.sum()), effective_n=neff(g),
                        effective_n_connected=neff(connected),
                        connection_all=mean(g, 'connection'), connection=mean(dec, 'connection'),
                        reliable_connected=mean(connected, 'reliable'),
                        unreliable_connected=1-mean(connected, 'reliable'),
                        reliable_service=mean(dec, 'service'),
                        no_grid=mean(dec.assign(no_grid=(dec.state == 0).astype(float)), 'no_grid'),
                        unreliable_service=mean(dec.assign(unreliable_service=(dec.state == 1).astype(float)), 'unreliable_service')))
    return pd.DataFrame(out)


def contrasts(tab):
    out = []
    for iso, g in tab.groupby('iso3'):
        gg = g.set_index('group')
        if not {0, 3}.issubset(gg.index):
            continue
        low, high = gg.loc[0], gg.loc[3]  # low vs high deprivation
        admit = min(low.n_connected, high.n_connected) >= 30 and min(low.connected_regions, high.connected_regions) >= 2
        row = dict(iso3=iso, country=low.country, admitted=bool(admit),
                   n_no=int(low.n_connected), n_high=int(high.n_connected),
                   n_all_no=int(low.n_decomposition), n_all_high=int(high.n_decomposition),
                   neff_no=low.effective_n_connected, neff_high=high.effective_n_connected,
                   connection_no=low.connection, connection_high=high.connection,
                   reliable_no=low.reliable_connected, reliable_high=high.reliable_connected,
                   service_no=low.reliable_service, service_high=high.reliable_service,
                   conditional_gap=low.reliable_connected-high.reliable_connected,
                   connection_gap=low.connection-high.connection,
                   service_gap=low.reliable_service-high.reliable_service,
                   connection_component=(low.reliable_connected+high.reliable_connected)/2*(low.connection-high.connection),
                   reliability_component=(low.connection+high.connection)/2*(low.reliable_connected-high.reliable_connected))
        row['identity_error'] = row['service_gap']-row['connection_component']-row['reliability_component']
        out.append(row)
    return pd.DataFrame(out)


def summarize(tab, columns):
    rng = np.random.default_rng(SEED)
    a = tab[columns].to_numpy(float)
    assert np.isfinite(a).all()
    idx = rng.integers(0, len(a), size=(B, len(a)))
    bs = a[idx].mean(axis=1)
    result = {'countries':len(a), 'estimand':'equal-country average; country-bootstrap interval',
              'bootstrap_replicates':B, 'seed':SEED}
    for i, c in enumerate(columns):
        result[c] = {'estimate':float(a[:,i].mean()),'lo':float(np.quantile(bs[:,i], .025)),
                     'hi':float(np.quantile(bs[:,i], .975)), 'median':float(np.median(a[:,i]))}
    if {'service_gap','connection_component','reliability_component'}.issubset(columns):
        ii={c:columns.index(c) for c in columns}
        frac=bs[:,ii['reliability_component']]/bs[:,ii['service_gap']]
        result['reliability_fraction_of_gap']={'estimate':float(a[:,ii['reliability_component']].mean()/a[:,ii['service_gap']].mean()),
                                             'lo':float(np.quantile(frac,.025)), 'hi':float(np.quantile(frac,.975))}
    return result


def main():
    f = P/'raw/afrobarometer/afrobarometer_r9.sav'
    d, m = pyreadstat.read_sav(str(f), usecols=COLS)
    audit = {'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(), 'raw_respondents':len(d),
             'country_count':int(d.COUNTRY.nunique()), 'weight':'withinwt_hh',
             'fieldwork_min':str(d.DATEINTR.min()), 'fieldwork_max':str(d.DATEINTR.max()),
             'PSU_identifier':'not exposed in the public merged file; not reconstructed',
             'Q92A_counts':d.Q92A.value_counts(dropna=False).to_dict(),
             'Q92B_counts':d.Q92B.value_counts(dropna=False).to_dict(),
             'invalid_weight':int((~(d.withinwt_hh > 0)).sum())}
    d = d[d.withinwt_hh > 0].copy()
    d['weight']=d.withinwt_hh
    d['iso3']=d.COUNTRY.map(COUNTRY_ISO)
    d['country']=d.COUNTRY.map(m.variable_value_labels['COUNTRY'])
    d['poverty']=d.LivedPoverty_CAT.where(d.LivedPoverty_CAT.isin([0,1,2,3]))
    d['settlement']=d.URBRUR_COND.where(d.URBRUR_COND.isin([1,2]))
    d['connection']=d.Q92A.where(d.Q92A.isin([0,1]))
    d['reliable']=(d.Q92B >= 4).astype(float).where(d.Q92B.isin([1,2,3,4,5]) & (d.connection == 1))
    d['state']=np.nan
    d.loc[d.connection == 0, 'state']=0
    d.loc[(d.connection == 1) & (d.reliable == 0), 'state']=1
    d.loc[(d.connection == 1) & (d.reliable == 1), 'state']=2
    d['service']=(d.state == 2).astype(float).where(d.state.notna())
    items=d[['Q6'+x for x in 'ABCDE']].where(lambda x:x.isin(range(5)))
    full5=items.notna().all(axis=1)
    lpi=items.mean(axis=1).where(full5)
    cat=np.select([lpi.eq(0), lpi.gt(0)&lpi.le(1), lpi.gt(1)&lpi.le(2), lpi.gt(2)&lpi.le(4)],[0,1,2,3],default=np.nan)
    comparable=full5 & d.poverty.notna()
    audit['lpi_category_disagreements_complete_items']=int((d.loc[comparable,'poverty'] != cat[comparable]).sum())
    four=items[['Q6A','Q6B','Q6C','Q6E']]
    x=four.mean(axis=1).where(four.notna().all(axis=1))
    d['poverty4']=np.select([x.eq(0),x.gt(0)&x.le(1),x.gt(1)&x.le(2),x.gt(2)&x.le(4)], [0,1,2,3], default=np.nan)
    audit.update({'poverty_missing':int(d.poverty.isna().sum()), 'connection_missing':int(d.connection.isna().sum()),
                  'connected_reliability_missing':int(((d.connection == 1)&d.reliable.isna()).sum()),
                  'nonconnected_with_valid_reliability_response':int(((d.connection == 0)&d.Q92B.isin(range(1,6))).sum()),
                  'decomposition_n':int(d.state.notna().sum()), 'connected_reliability_n':int(d.reliable.notna().sum())})
    d[['iso3','country','REGION','settlement','poverty','poverty4','weight','connection','reliable','state','service']].to_parquet(P/'data/afrobarometer_analysis_sample.parquet',index=False)
    tab=cells(d,'poverty');tab.to_csv(P/'data/afrobarometer_country_groups.csv',index=False)
    co=contrasts(tab);co.to_csv(P/'data/afrobarometer_country_contrasts.csv',index=False)
    main=co[co.admitted]
    assert main.identity_error.abs().max()<1e-12
    cols=['connection_no','connection_high','reliable_no','reliable_high','service_no','service_high','conditional_gap','connection_gap','service_gap','connection_component','reliability_component']
    out={'primary':summarize(main,cols), 'positive_conditional_gaps':int((main.conditional_gap>0).sum()),
         'positive_service_gaps':int((main.service_gap>0).sum()), 'admitted_iso3':main.iso3.tolist(),
         'excluded_primary_iso3':co.loc[~co.admitted,'iso3'].tolist()}
    fourco=contrasts(cells(d,'poverty4'));fourco.to_csv(P/'data/afrobarometer_four_item_contrasts.csv',index=False)
    common=main.merge(fourco[fourco.admitted],on='iso3',suffixes=('_five','_four'))
    out['construct_common_countries']=len(common)
    out['construct_five_item']=summarize(common,['conditional_gap_five','service_gap_five'])
    out['construct_four_item']=summarize(common,['conditional_gap_four','service_gap_four'])
    st=[]
    for iso in main.iso3:
        dc=d[(d.iso3 == iso)&d.reliable.notna()&d.poverty.notna()&d.settlement.notna()]
        rates={}
        counts={}
        for (g,s),sub in dc.groupby(['poverty','settlement']):
            rates[g,s]=mean(sub,'reliable');counts[g,s]=len(sub)
        if min(counts.get((g,s),0) for g in [0,3] for s in [1,2]) < 20:
            continue
        urban=float(dc.loc[dc.settlement == 1,'weight'].sum()/dc.weight.sum())
        gap=urban*(rates[0,1]-rates[3,1])+(1-urban)*(rates[0,2]-rates[3,2])
        st.append(dict(iso3=iso,country=dc.country.iloc[0],urban_weight=urban,
                       total_gap=float(main.loc[main.iso3 == iso,'conditional_gap'].iloc[0]),standardized_gap=gap,
                       urban_gap=rates[0,1]-rates[3,1],rural_gap=rates[0,2]-rates[3,2],min_cell_n=min(counts[g,s] for g in [0,3] for s in [1,2])))
    st=pd.DataFrame(st);st.to_csv(P/'data/afrobarometer_settlement_contrasts.csv',index=False)
    out['settlement']=summarize(st,['total_gap','standardized_gap','urban_gap','rural_gap'])
    out['settlement_iso3']=st.iso3.tolist()
    audit['identity_error_max']=float(main.identity_error.abs().max())
    audit['all_weights_finite']=bool(np.isfinite(d.weight).all())
    audit['all_iso3_mapped']=bool(d.iso3.notna().all())
    (P/'metadata/afrobarometer_analysis_audit.json').write_text(json.dumps(audit,indent=2,default=str))
    (P/'data/afrobarometer_summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
