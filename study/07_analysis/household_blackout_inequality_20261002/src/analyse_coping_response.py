"""Questionnaire-grounded resource and response comparisons within households."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyreadstat
from analyse_kenya_ethiopia import load_country

P = Path(__file__).resolve().parents[1]
OUT = P / 'data/coping_response'
OUT.mkdir(exist_ok=True)
B, SEED = 2000, 20261002
METRICS = ['appliance_no_backup', 'lighting_no_backup', 'assistance_wait',
           'assistance_provider', 'no_repair_channel', 'response_over24h']


def binary(s, codes, yes):
    return s.isin(yes).astype(float).where(s.isin(codes))


def load(country):
    if country == 'cambodia':
        fn = P/'raw/mtf/cambodia-multi-tier-framework-mtf-survey-2018/extracted/Main Dataset.dta'
        cols = ['Province', 'District', 'Commune', 'Village', 'Locality', 'pw', 'HHtot',
                'A18', 'C2', 'C30', 'C31', 'C32a', 'C32', 'C33', 'C34']
        d, meta = pyreadstat.read_dta(fn, usecols=cols)
        d = d.apply(pd.to_numeric, errors='coerce')
        d = d[(d.pw>0)&(d.HHtot>0)&(d.A18>=0)&~d.A18.isin([888,999])].copy()
        d['w'], d['rank_value'], d['settlement'] = d.pw, d.A18/d.HHtot, d.Locality
        d['connection'] = binary(d.C2,[0,1],[1])
        d['block'] = d[['Province','District','Commune','Village']].astype(int).astype(str).agg('_'.join,axis=1)
        d['stratum'] = d[['Province','Locality']].astype(int).astype(str).agg('_'.join,axis=1)
        appliance_codes, lighting_codes, multiplier = [1,2,3,4,5,6,111,555], list(range(1,14))+[111,555], 24
        d['no_repair_channel'] = binary(d.C32a,[0,1],[0])
        files = [fn]
    else:
        d, prior_audit = load_country(country)
        d = d[(d.w>0)&(d['size']>0)&d.food_pc.notna()&d.settlement.isin([1,2])].copy()
        d['rank_value'] = d.food_pc
        files = [P/f['path'] for f in prior_audit['source_files']]
        if country == 'ethiopia':
            fn = P/'raw/mtf/ethiopia-multi-tier-framework-mtf-survey-2018/extracted/Section C.dta'
            s, meta = pyreadstat.read_dta(fn, usecols=['HHID','C30','C31','C32','C33','C34'], encoding='latin1')
            assert s.HHID.is_unique
            d = d.merge(s,on='HHID',how='left',validate='one_to_one')
            appliance_codes, lighting_codes, multiplier = list(range(1,7))+[111,555], list(range(1,12))+[111,555], 24
            d['no_repair_channel'] = binary(d.C32, list(range(1,8))+[555], [6])
        else:
            for i in range(30,35):
                d[f'C{i}'] = d[f'c_c_{i}']
            appliance_codes, lighting_codes, multiplier = [1,2,3,4,111,555], list(range(1,8))+[111,555], 1
            d['no_repair_channel'] = binary(d.C32, list(range(1,7))+[555], [5])
    for col in ['C30','C31','C32','C33','C34']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    z = d.sort_values('rank_value'); cum = z.w.cumsum()/z.w.sum()
    cuts = [float(z.iloc[np.searchsorted(cum,p)].rank_value) for p in [.2,.4,.6,.8]]
    d['quintile'] = np.searchsorted(cuts,d.rank_value,side='left')+1
    d['appliance_no_backup'] = binary(d.C31,appliance_codes,[111])
    d['lighting_no_backup'] = binary(d.C30,lighting_codes,[111])
    d['assistance_wait'] = binary(d.C33,[1,2,3],[3])
    d['assistance_provider'] = binary(d.C33,[1,2,3],[1])
    d['response_hours'] = (d.C34*multiplier).where((d.C34>=0)&~d.C34.isin([888,999])&d.C33.isin([1,2])&(d.connection==1))
    d['response_over24h'] = d.response_hours.gt(24).astype(float).where(d.response_hours.notna())
    d[METRICS] = d[METRICS].where(d.connection.eq(1), axis=0)
    assert d.settlement.isin([1,2]).all()
    audit = dict(country=country,eligible_n=len(d),connected_n=int(d.connection.eq(1).sum()),
                 fixed_rank_cutpoints=cuts,response_hour_multiplier=multiplier,
                 sources=[dict(path=str(f.relative_to(P)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],
                 appliance_codes={str(k):int(v) for k,v in d.loc[d.connection==1,'C31'].value_counts(dropna=False).items()},
                 repair_codes={str(k):int(v) for k,v in d.loc[d.connection==1,'C32'].value_counts(dropna=False).items()},
                 response_range_hours=[float(d.response_hours.min()),float(d.response_hours.max())])
    return d,audit


def wmedian(v,w):
    v=np.asarray(v,dtype=float); w=np.asarray(w,dtype=float)
    good=np.isfinite(v)&np.isfinite(w)&(w>0)
    if not good.any(): return np.nan
    v,w=np.asarray(v[good]),np.asarray(w[good]); ix=np.argsort(v);v,w=v[ix],w[ix]
    return float(v[np.searchsorted(np.cumsum(w),w.sum()/2)])


def run(country):
    d,audit=load(country)
    info=d.groupby('block').stratum.first()
    assert d.groupby('block').stratum.nunique().max()==1
    bi=pd.Categorical(d.block,categories=info.index).codes
    cell=((d.quintile-1)*2+d.settlement-1).astype(int).to_numpy()
    arr=np.zeros((len(info),10,len(METRICS),2))
    for j,m in enumerate(METRICS):
        good=d[m].notna();w=d.w.to_numpy()
        np.add.at(arr[:,:,j,0],(bi,cell),w*good)
        np.add.at(arr[:,:,j,1],(bi,cell),w*d[m].fillna(0))
    t=arr.sum(0).reshape(5,2,len(METRICS),2)
    rng=np.random.default_rng(SEED);reps=np.zeros((B,len(info)),dtype=np.int16)
    for s in info.unique():
        ix=np.flatnonzero(info.to_numpy()==s);draw=rng.choice(ix,(B,len(ix)),replace=True)
        for b in range(B): np.add.at(reps[b],draw[b],1)
    draws=(reps@arr.reshape(len(info),-1)).reshape(B,5,2,len(METRICS),2)
    with np.errstate(divide='ignore',invalid='ignore'):
        p=t.sum(axis=1)[...,1]/t.sum(axis=1)[...,0]
        boots=draws.sum(axis=2)[...,1]/draws.sum(axis=2)[...,0]
        ps=t[...,1]/t[...,0]; bs=draws[...,1]/draws[...,0]
    uw=d.loc[d.connection.eq(1)&d.settlement.eq(1),'w'].sum()/d.loc[d.connection.eq(1),'w'].sum()
    rows=[];contrasts=[];cells=[]
    for j,m in enumerate(METRICS):
        for q in range(1,6):
            sub=d[d.quintile.eq(q)&d.connection.eq(1)]; v=sub[sub[m].notna()]
            rows.append(dict(country=country,metric=m,quintile=q,n_connected=len(sub),n_valid=len(v),
                             blocks=v.block.nunique(),weighted_response_coverage=v.w.sum()/sub.w.sum(),
                             estimate=p[q-1,j],lo=np.nanquantile(boots[:,q-1,j],.025),hi=np.nanquantile(boots[:,q-1,j],.975),
                             median_response_hours=wmedian(v.response_hours,v.w) if m=='response_over24h' else np.nan))
            for s in [1,2]:
                a=v[v.settlement.eq(s)]
                cells.append(dict(country=country,metric=m,quintile=q,settlement=s,n=len(a),blocks=a.block.nunique(),estimate=ps[q-1,s-1,j]))
        ext=pd.DataFrame(rows).query('country==@country and metric==@m and quintile in [1,5]')
        eligible=bool(m!='response_over24h' or (ext.n_valid.min()>=30 and ext.blocks.min()>=10))
        dd=boots[:,0,j]-boots[:,4,j]
        contrasts.append(dict(country=country,metric=m,contrast='observed Q1-Q5',admitted=eligible,
                              difference_pp=100*(p[0,j]-p[4,j]),lo_pp=100*np.nanquantile(dd,.025),hi_pp=100*np.nanquantile(dd,.975),
                              min_extreme_n=int(ext.n_valid.min()),min_extreme_blocks=int(ext.blocks.min())))
        ss=pd.DataFrame(cells).query('country==@country and metric==@m and quintile in [1,5]')
        stdok=ss.n.min()>=30
        if stdok:
            sd=uw*(bs[:,0,0,j]-bs[:,4,0,j])+(1-uw)*(bs[:,0,1,j]-bs[:,4,1,j])
            point=uw*(ps[0,0,j]-ps[4,0,j])+(1-uw)*(ps[0,1,j]-ps[4,1,j])
            contrasts.append(dict(country=country,metric=m,contrast='settlement standardized Q1-Q5',admitted=eligible,
                                  difference_pp=100*point,lo_pp=100*np.nanquantile(sd,.025),hi_pp=100*np.nanquantile(sd,.975),
                                  min_extreme_n=int(ss.n.min()),min_extreme_blocks=int(ss.blocks.min())))
    audit.update(bootstrap_replicates=B,seed=SEED,blocks=len(info),strata=int(info.nunique()),
                 singleton_strata=int((info.value_counts()==1).sum()),common_urban_weight=float(uw))
    a=pd.DataFrame(rows);c=pd.DataFrame(contrasts);s=pd.DataFrame(cells)
    assert a.estimate.dropna().between(0,1).all()
    assert (a.n_valid<=a.n_connected).all()
    for name,tab in [('quintiles',a),('contrasts',c),('settlement',s)]:tab.to_csv(OUT/f'{country}_{name}.csv',index=False)
    (OUT/f'{country}_audit.json').write_text(json.dumps(audit,indent=2))
    print(country,c.to_string(index=False))
    return a,c,s


if __name__=='__main__':
    results=[run(c) for c in ['cambodia','ethiopia','kenya']]
    for i,n in enumerate(['quintiles','contrasts','settlement']):
        pd.concat([r[i] for r in results],ignore_index=True).to_csv(OUT/f'all_{n}.csv',index=False)
