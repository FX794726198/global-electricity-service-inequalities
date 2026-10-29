"""Cause accounting and paired restoration-stage contrasts; no global pooling."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
R = P.parents[1]
T = R / '07_analysis/turkiye_cause_extension_20261002'
SEED = 20261002
INPUTS = [T/'data/unannounced_customer_stages_2025.parquet',
          T/'data/sedas_2025_numeric.parquet',
          T/'data/cause_settlement_metrics_2025.csv',
          T/'data/published_annual_settlement_metrics.csv']


def save(name, frame):
    frame.to_csv(P/'data'/f'{name}.csv', index=False)
    frame.to_parquet(P/'data'/f'{name}.parquet', index=False)


def boot_ratio(z):
    """Resample complete onset dates, not stages, for an equal-incident estimand."""
    if len(z) == 0:
        return dict(n=0, dates=0, ratio=np.nan, lower=np.nan, upper=np.nan)
    q = z.groupby('onset_date').agg(log_sum=('log_ratio', 'sum'), n=('log_ratio', 'size'))
    k = len(q)
    if k < 2:
        lo = hi = np.nan
    else:
        w = np.random.default_rng(SEED).multinomial(k, np.full(k, 1/k), 2000)
        sims = np.exp((w @ q.log_sum.to_numpy()) / (w @ q.n.to_numpy()))
        lo, hi = np.quantile(sims, [.025, .975])
    return dict(n=len(z), dates=k, ratio=float(np.exp(z.log_ratio.mean())),
                lower=float(lo), upper=float(hi),
                paired_difference_h=float((z.Rural_duration-z.Urban_duration).mean()),
                within_one_minute_share=float(((z.Rural_duration-z.Urban_duration).abs() <= 1/60).mean()),
                multistage_incidents=int((z.stage_count > 1).sum()),
                Rural_CI=float(z.Rural_CI.sum()), Urban_CI=float(z.Urban_CI.sum()),
                customer_weighted_ratio=float((z.Rural_H.sum()/z.Rural_CI.sum()) /
                                              (z.Urban_H.sum()/z.Urban_CI.sum())))


def incident_accounting(z):
    """Exact partition of customer-weighted means on one shared population."""
    if len(z)==0: return []
    q=z.copy()
    q['rural_weight_urban_duration']=q.Rural_CI*q.Urban_duration
    q['urban_weight_rural_duration']=q.Urban_CI*q.Rural_duration
    cols=['Rural_CI','Urban_CI','Rural_H','Urban_H',
          'rural_weight_urban_duration','urban_weight_rural_duration']
    day=q.groupby('onset_date')[cols].sum()
    def calc(t):
        cr,cu,hr,hu,xru,xur=t.T
        dr=hr/cr;du=hu/cu
        within=.5*((hr-xru)/cr+(xur-hu)/cu)
        allocation=dr-du-within
        return np.stack([dr,du,dr/du,dr-du,within,allocation,
                         100*within/du,100*allocation/du],axis=-1)
    values=calc(day.sum().to_numpy()[None,:])[0]
    if len(day)>=2:
        n=len(day)
        w=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),2000)
        sims=calc(w@day.to_numpy())
        lo,hi=np.quantile(sims,[.025,.975],axis=0)
    else:lo=hi=np.full(len(values),np.nan)
    names=['rural_mean_h','urban_mean_h','mean_ratio','gap_h','within_incident_h',
           'between_incident_allocation_h','within_pct_urban','allocation_pct_urban']
    assert np.isclose(values[4]+values[5],values[3])
    return [dict(metric=k,value=v,lower=a,upper=b,n=len(q),dates=len(day))
            for k,v,a,b in zip(names,values,lo,hi)]


def main():
    l = pd.read_parquet(INPUTS[0])
    raw = pd.read_parquet(INPUTS[1])
    c = pd.read_csv(INPUTS[2])
    annual = pd.read_csv(INPUTS[3])
    weather = ['Ice loading', 'Wind / lightning']
    assert not raw.duplicated(['KesintininKodu1', 'Kademe2']).any()
    assert (l.groupby('KesintininKodu1').cause_group.nunique() == 1).all()
    assert (l.groupby('KesintininKodu1').Il3A.nunique() == 1).all()
    assert ((l.customer_interruptions >= 0) & (l.reported_customer_hours >= 0)).all()
    assert not ((l.customer_interruptions == 0) & (l.reported_customer_hours > 0)).any()

    parts=[]; summary=[]
    for voltage in ['All','LV','MV']:
        a=c[c.voltage.eq(voltage)].pivot(index='cause_group',columns='settlement')
        fl=a['frequency_contribution']['Rural']; fu=a['frequency_contribution']['Urban']
        hl=a['annual_hours_contribution']['Rural']; hu=a['annual_hours_contribution']['Urban']
        assert ((fl>0)&(fu>0)).all()
        dl=hl/fl; du=hu/fu
        sl=fl/fl.sum(); su=fu/fu.sum()
        cf=(fl-fu)*(dl+du)/2; cd=(dl-du)*(fl+fu)/2
        within=(sl+su)*(dl-du)/2; mix=(sl-su)*(dl+du)/2
        for cause in a.index:
            parts.append(dict(voltage=voltage,cause=cause,
                rural_frequency=fl[cause],urban_frequency=fu[cause],
                rural_hours=hl[cause],urban_hours=hu[cause],
                rural_duration=dl[cause],urban_duration=du[cause],
                frequency_ratio=fl[cause]/fu[cause],duration_ratio=dl[cause]/du[cause],
                rural_share=sl[cause],urban_share=su[cause],
                hour_gap=hl[cause]-hu[cause],frequency_component_h=cf[cause],duration_component_h=cd[cause],
                within_duration_h=within[cause],composition_duration_h=mix[cause]))
        dr=hl.sum()/fl.sum(); du_all=hu.sum()/fu.sum()
        summary.append(dict(voltage=voltage,rural_frequency=fl.sum(),urban_frequency=fu.sum(),
            rural_hours=hl.sum(),urban_hours=hu.sum(),rural_duration=dr,urban_duration=du_all,
            duration_ratio=dr/du_all,duration_gap_h=dr-du_all,
            within_cause_h=within.sum(),composition_h=mix.sum(),
            within_cause_pct_urban=100*within.sum()/du_all,
            composition_pct_urban=100*mix.sum()/du_all,
            net_pct_urban=100*(dr-du_all)/du_all,
            weather_duration_ratio=(hl[weather].sum()/fl[weather].sum())/(hu[weather].sum()/fu[weather].sum()),
            other_duration_ratio=(hl.drop(weather).sum()/fl.drop(weather).sum())/(hu.drop(weather).sum()/fu.drop(weather).sum()),
            weather_frequency_ratio=fl[weather].sum()/fu[weather].sum(),
            weather_hour_gap=hl[weather].sum()-hu[weather].sum(),
            weather_frequency_component_h=cf[weather].sum(),weather_duration_component_h=cd[weather].sum()))
        assert np.allclose(cf+cd,hl-hu,atol=1e-10)
        assert np.isclose(within.sum()+mix.sum(),dr-du_all,atol=1e-10)
    parts=pd.DataFrame(parts); summary=pd.DataFrame(summary)
    save('cause_contrast_and_accounting',parts);save('overall_duration_accounting',summary)

    stages=l.drop_duplicates(['KesintininKodu1','Kademe2'])
    meta=stages.groupby('KesintininKodu1').agg(province=('Il3A','first'),district=('Ilce3B','first'),
        cause=('cause_group','first'),onset=('KesintininBaslamaBitisTarihi','min'),
        start_count=('KesintininBaslamaBitisTarihi','nunique'),stage_count=('Kademe2','size'))
    meta['onset_date']=meta.onset.dt.strftime('%Y-%m-%d')
    checks=raw[['KesintininKodu1','Kademe2','timestamp_duration_h']]
    l=l.merge(checks,on=['KesintininKodu1','Kademe2'],validate='many_to_one')
    l['timestamp_customer_hours']=l.customer_interruptions*l.timestamp_duration_h
    all_pairs=[]; estimates=[]; provenance=[]; incident_parts=[]; coverage=[]
    for voltage in ['All','LV']:
        z=l if voltage=='All' else l[l.voltage.eq(voltage)]
        wide=z.groupby(['KesintininKodu1','settlement']).agg(CI=('customer_interruptions','sum'),
            H=('reported_customer_hours','sum'),product_H=('product_customer_hours','sum'),
            timestamp_H=('timestamp_customer_hours','sum')).unstack('settlement')
        wide.columns=[f'{s}_{metric}' for metric,s in wide.columns]
        q=wide.join(meta)
        q=q[(q.Rural_CI>0)&(q.Urban_CI>0)&(q.Rural_H>0)&(q.Urban_H>0)].copy()
        for group in ['Rural','Urban']:
            q[f'{group}_duration']=q[f'{group}_H']/q[f'{group}_CI']
        q['log_ratio']=np.log(q.Rural_duration/q.Urban_duration)
        q['product_log_ratio']=np.log((q.Rural_product_H/q.Rural_CI)/(q.Urban_product_H/q.Urban_CI))
        q['timestamp_log_ratio']=np.log((q.Rural_timestamp_H/q.Rural_CI)/(q.Urban_timestamp_H/q.Urban_CI))
        q['voltage_scope']=voltage
        all_pairs.append(q.reset_index())
        for pop,sub in [('Shared identifier',q),('Shared identifier and onset',q[q.start_count.eq(1)]),
                        ('Shared onset, multistage only',q[q.start_count.eq(1)&q.stage_count.gt(1)])]:
            specs=[('All causes',sub),('Weather',sub[sub.cause.isin(weather)]),
                   ('Other causes',sub[~sub.cause.isin(weather)])]
            specs += [(cause,sub[sub.cause.eq(cause)]) for cause in sorted(meta.cause.unique())]
            for scope,frame in specs:
                estimates.append(dict(population=pop,voltage=voltage,scope=scope,province='All',**boot_ratio(frame)))
                if scope in ['All causes','Weather','Other causes']:
                    incident_parts.extend(dict(population=pop,voltage=voltage,scope=scope,**row)
                                          for row in incident_accounting(frame))
                    source=z if scope=='All causes' else z[z.cause_group.isin(weather)==(scope=='Weather')]
                    totals=source.groupby('settlement')[['customer_interruptions','reported_customer_hours']].sum()
                    for group in ['Urban','Rural']:
                        coverage.append(dict(population=pop,voltage=voltage,scope=scope,settlement=group,
                            shared_CI=frame[f'{group}_CI'].sum(),all_CI=totals.loc[group,'customer_interruptions'],
                            shared_CI_fraction=frame[f'{group}_CI'].sum()/totals.loc[group,'customer_interruptions'],
                            shared_H=frame[f'{group}_H'].sum(),all_H=totals.loc[group,'reported_customer_hours'],
                            shared_H_fraction=frame[f'{group}_H'].sum()/totals.loc[group,'reported_customer_hours']))
            for province,frame in sub.groupby('province'):
                for scope,f in [('All causes',frame),('Weather',frame[frame.cause.isin(weather)])]:
                    estimates.append(dict(population=pop,voltage=voltage,scope=scope,province=province,**boot_ratio(f)))
            provenance.append(dict(population=pop,voltage=voltage,n=len(sub),
                reported_equal_incident_ratio=float(np.exp(sub.log_ratio.mean())),
                stage_product_equal_incident_ratio=float(np.exp(sub.product_log_ratio.mean())),
                timestamp_equal_incident_ratio=float(np.exp(sub.timestamp_log_ratio.mean())),
                n_product_different_1pct=int((abs(sub.log_ratio-sub.product_log_ratio)>np.log(1.01)).sum())))
    save('shared_incident_pairs',pd.concat(all_pairs,ignore_index=True))
    es=pd.DataFrame(estimates);save('shared_incident_estimates',es)
    save('shared_incident_measurement_check',pd.DataFrame(provenance))
    save('shared_incident_accounting',pd.DataFrame(incident_parts))
    save('shared_incident_coverage',pd.DataFrame(coverage))
    # Archive all positive-population support rather than only matched incidents.
    pop=l[l.customer_interruptions>0].groupby(['Il3A','cause_group','settlement']).agg(incidents=('KesintininKodu1','nunique'),
        CI=('customer_interruptions','sum'),H=('reported_customer_hours','sum')).reset_index()
    save('province_cause_group_contributions',pop)
    for f in INPUTS:
        assert f.exists()
    (P/'metadata/input_manifest.json').write_text(json.dumps([dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in INPUTS],indent=2))
    validation=dict(raw_stage_uniqueness=True,cause_constant_within_id=True,province_constant_within_id=True,
        zero_customer_positive_hour_cells=0,annual_cause_hour_partition=True,duration_composition_identity=True,
        bootstrap_unit='calendar onset date, all incidents resampled together',draws=2000,seed=SEED,
        primary_selection='Positive Rural and Urban CI and H, and one common start timestamp across all admitted stages')
    (P/'metadata/validation.json').write_text(json.dumps(validation,indent=2))
    print(summary.to_string(index=False))
    print(es.query("population=='Shared identifier and onset' and voltage=='All' and (scope in ['All causes','Weather','Other causes'])").to_string(index=False))
    print(pd.DataFrame(provenance).to_string(index=False))
    print(pd.DataFrame(incident_parts).query("population=='Shared identifier and onset' and voltage=='All' and scope=='Weather'").to_string(index=False))


if __name__=='__main__': main()
