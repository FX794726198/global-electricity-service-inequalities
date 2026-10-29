"""Native reason-coded household connection-gap accounting, all quintiles retained."""
from pathlib import Path
import sys
import hashlib
import json
import numpy as np
import pandas as pd
import pyreadstat

P=Path(__file__).resolve().parents[1]
BASE=P.parents[1]/'07_analysis/household_blackout_inequality_20261002'
sys.path.insert(0,str(BASE/'src'))
from analyse_kenya_ethiopia import load_country
B,SEED=2000,20261002
NAMES=['connected','distance','cost','administrative','landlord','satisfied','unreliable','other','unknown']

def load(country):
    if country=='cambodia':
        f=BASE/'raw/mtf/cambodia-multi-tier-framework-mtf-survey-2018/extracted/Main Dataset.dta'
        cols=['Province','District','Commune','Village','Locality','pw','HHtot','A18','C2','C3','C4']
        d,m=pyreadstat.read_dta(f,usecols=cols)
        d=d.apply(pd.to_numeric,errors='coerce')
        d=d[(d.pw>0)&(d.HHtot>0)&(d.A18>=0)&~d.A18.isin([888,999])].copy()
        d['w'],d['rank_value'],d['settlement']=d.pw,d.A18/d.HHtot,d.Locality
        d['connection']=d.C2.eq(1).astype(float).where(d.C2.isin([0,1]))
        d['block']=d[['Province','District','Commune','Village']].astype(int).astype(str).agg('_'.join,axis=1)
        d['stratum']=d[['Province','Locality']].astype(int).astype(str).agg('_'.join,axis=1)
        labels=m.variable_value_labels['C4'];d['reason']=d.C4
        files=[f]
    else:
        d,a=load_country(country)
        d=d[(d.w>0)&(d['size']>0)&d.food_pc.notna()&d.settlement.isin([1,2])].copy()
        d['rank_value']=d.food_pc
        files=[BASE/x['path'] for x in a['source_files']]
        if country=='ethiopia':
            f=BASE/'raw/mtf/ethiopia-multi-tier-framework-mtf-survey-2018/extracted/Section C.dta'
            s,m=pyreadstat.read_dta(f,usecols=['HHID','C3','C4'],encoding='latin1')
            d=d.merge(s,on='HHID',how='left',validate='one_to_one');labels=m.variable_value_labels['C4'];d['reason']=d.C4
        else:
            f=files[0];_,m=pyreadstat.read_dta(f,metadataonly=True,encoding='latin1')
            labels=m.variable_value_labels['c_c_4'];d['reason']=d.c_c_4
    z=d.sort_values('rank_value');cum=z.w.cumsum()/z.w.sum()
    cut=[float(z.iloc[np.searchsorted(cum,q)].rank_value) for q in [.2,.4,.6,.8]]
    d['quintile']=np.searchsorted(cut,d.rank_value,side='left')+1
    previous=pd.read_parquet(BASE/f'data/{country}_analysis_sample.parquet')
    assert len(previous)==len(d)
    assert np.array_equal(previous.quintile,d.quintile)
    oldconn=previous.connected if country=='cambodia' else previous.connection
    assert np.allclose(oldconn,d.connection,equal_nan=True)
    codegroups={1:'distance',2:'cost',3:'cost',4:'satisfied',5:'landlord',6:'unreliable',7:'administrative',8:'administrative',9:'administrative'}
    if country=='cambodia':codegroups[10]='cost'
    for code in labels:
        if code>0 and code not in codegroups:codegroups[code]='other'
    d['barrier']=d.reason.map(codegroups).fillna('unknown')
    d.loc[d.connection.eq(1),'barrier']='connected'
    d.loc[d.connection.isna(),'barrier']=pd.NA
    audit={'country':country,'rows':len(d),'cutpoints':cut,'native_labels':labels,
           'codegroups':codegroups,'missing_connection':int(d.connection.isna().sum()),
           'native_unconnected_reason_counts':{str(k):int(v) for k,v in d.loc[d.connection.eq(0),'reason'].value_counts(dropna=False).items()},
           'connected_with_nonmissing_reason':int((d.connection.eq(1)&d.reason.notna()).sum()),
           'sources':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]}
    return d,audit

def run(country):
    d,audit=load(country)
    d=d[d.connection.notna()].copy()
    info=d.groupby('block').stratum.first()
    assert d.groupby('block').stratum.nunique().max()==1
    bi=pd.Categorical(d.block,categories=info.index).codes
    qi=d.quintile.to_numpy(int)-1
    si=pd.Categorical(d.barrier,categories=NAMES).codes
    assert (si>=0).all()
    arr=np.zeros((len(info),5,len(NAMES)))
    np.add.at(arr,(bi,qi,si),d.w)
    n=arr.sum(0);rate=n/n.sum(axis=1,keepdims=True)
    rng=np.random.default_rng(SEED);mult=np.zeros((B,len(info)),dtype=np.int16)
    for s in info.unique():
        ix=np.flatnonzero(info.to_numpy()==s);draw=rng.choice(ix,(B,len(ix)),replace=True)
        for b in range(B):np.add.at(mult[b],draw[b],1)
    draws=(mult@arr.reshape(len(info),-1)).reshape(B,5,len(NAMES))
    boots=draws/draws.sum(axis=2,keepdims=True)
    unconn=rate[:,1:]/rate[:,1:].sum(axis=1,keepdims=True)
    bu=boots[:,:,1:]/boots[:,:,1:].sum(axis=2,keepdims=True)
    rows=[];contrasts=[]
    for q in range(1,6):
        sub=d[d.quintile.eq(q)]
        for j,name in enumerate(NAMES):
            rows.append(dict(country=country,quintile=q,state=name,n_households=len(sub),blocks=sub.block.nunique(),
                             n_unconnected=int(sub.connection.eq(0).sum()),n_state=int(sub.barrier.eq(name).sum()),
                             prevalence=rate[q-1,j],lo=np.quantile(boots[:,q-1,j],.025),hi=np.quantile(boots[:,q-1,j],.975),
                             share_among_unconnected=unconn[q-1,j-1] if j else np.nan,
                             conditional_lo=np.quantile(bu[:,q-1,j-1],.025) if j else np.nan,
                             conditional_hi=np.quantile(bu[:,q-1,j-1],.975) if j else np.nan))
    for j,name in enumerate(NAMES):
        delta=boots[:,0,j]-boots[:,4,j]
        contrasts.append(dict(country=country,state=name,contrast='Q1 minus Q5',
                              difference_pp=100*(rate[0,j]-rate[4,j]),lo_pp=100*np.quantile(delta,.025),hi_pp=100*np.quantile(delta,.975)))
    error=float((rate[0,1:]-rate[4,1:]).sum()-(rate[4,0]-rate[0,0]))
    assert abs(error)<1e-12
    assert np.max(np.abs(rate.sum(1)-1))<1e-12
    assert np.isfinite(boots).all()
    audit.update(blocks=len(info),strata=info.nunique(),bootstrap_replicates=B,seed=SEED,identity_error=error,
                 singleton_strata=int(info.value_counts().eq(1).sum()),numerical_checks_passed=True)
    d[['w','rank_value','quintile','connection','reason','barrier','block','stratum']].to_parquet(P/f'data/barriers_{country}_sample.parquet',index=False)
    pd.DataFrame(rows).to_csv(P/f'data/barriers_{country}_quintiles.csv',index=False)
    pd.DataFrame(contrasts).to_csv(P/f'data/barriers_{country}_contrasts.csv',index=False)
    (P/f'metadata/barriers_{country}_audit.json').write_text(json.dumps(audit,indent=2,default=int))
    return pd.DataFrame(rows),pd.DataFrame(contrasts)

if __name__=='__main__':
    out=[run(c) for c in ['cambodia','ethiopia','kenya']]
    for i,n in enumerate(['quintiles','contrasts']):
        a=pd.concat([x[i] for x in out],ignore_index=True)
        a.to_csv(P/f'data/barriers_all_{n}.csv',index=False)
        a.to_csv(P/f'source_data/barriers_all_{n}.csv',index=False)
        if i:print(a.to_string(index=False))
