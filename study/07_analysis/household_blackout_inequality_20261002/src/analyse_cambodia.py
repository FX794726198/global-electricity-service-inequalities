"""Weighted household analysis with paired frequency and accumulated duration."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import pyreadstat

P=Path(__file__).resolve().parents[1]
B=2000
SEED=20261002


def metrics(t):
    # Sufficient-statistic columns: all weight, connected weight, pair weight,
    # weighted F, weighted H, F-valid weight, weighted F-valid, backup weight/yes.
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.stack([t[...,1]/t[...,0],t[...,3]/t[...,2],t[...,4]/t[...,2],
                         t[...,4]/t[...,3],t[...,6]/t[...,5],t[...,8]/t[...,7]],axis=-1)


def main():
    f=P/'raw/mtf/cambodia-multi-tier-framework-mtf-survey-2018/extracted/Main Dataset.dta'
    cols=['Household_ID','Province','District','Commune','Village','Locality','date','pw','HHtot','A18','C2','C28b','C29b','C31']
    raw,m=pyreadstat.read_dta(str(f),usecols=cols)
    raw=raw.apply(pd.to_numeric,errors='coerce')
    d=raw[(raw.pw>0)&(raw.HHtot>0)&(raw.A18>=0)&~raw.A18.isin([888,999])].copy()
    d['income_pc']=d.A18/d.HHtot
    z=d.sort_values('income_pc');cum=z.pw.cumsum()/z.pw.sum()
    cuts=[float(z.iloc[np.searchsorted(cum,p)].income_pc) for p in [.2,.4,.6,.8]]
    assert len(set(cuts))==4
    d['quintile']=np.searchsorted(cuts,d.income_pc,side='left')+1
    d['connected']=d.C2.eq(1).astype(float).where(d.C2.isin([0,1]))
    d['F']=d.C28b.where((d.C28b>=0)&~d.C28b.isin([888,999])&(d.connected==1))
    minutes=d.C29b.where((d.C29b>=0)&~d.C29b.isin([888,999])&(d.connected==1))
    structural=(d.F==0)&d.C29b.isna()
    minutes.loc[structural]=0
    d['H']=minutes/60
    incompatible=((d.H>168)|((d.F==0)&(d.H>0)))
    d.loc[incompatible,'H']=np.nan
    d['paired']=d.F.notna()&d.H.notna()
    d['backup']=d.C31.isin([1,2,3,4,5,6]).astype(float).where(d.C31.isin([1,2,3,4,5,6,111])&(d.connected==1))
    d['block']=d[['Province','District','Commune','Village']].astype(int).astype(str).agg('_'.join,axis=1)
    d['stratum']=d[['Province','Locality']].astype(int).astype(str).agg('_'.join,axis=1)
    assert d.groupby('block').stratum.nunique().max()==1
    blockinfo=d.groupby('block').stratum.first()
    d['bi']=pd.Categorical(d.block,categories=blockinfo.index).codes
    # 5 quintiles × 2 settlement groups, so the same array supplies every contrast.
    d['cell']=(d.quintile-1)*2+(d.Locality-1).astype(int)
    arr=np.zeros((len(blockinfo),10,9))
    weights=d.pw.to_numpy();pair=d.paired.to_numpy();con=d.connected.to_numpy()
    values=np.column_stack([weights*np.isfinite(con),weights*np.nan_to_num(con),weights*pair,
                            weights*np.nan_to_num(d.F)*pair,weights*np.nan_to_num(d.H)*pair,
                            weights*d.F.notna(),weights*np.nan_to_num(d.F),
                            weights*d.backup.notna(),weights*np.nan_to_num(d.backup)])
    for j in range(9):np.add.at(arr[:,:,j],(d.bi,d.cell),values[:,j])
    overall=arr.sum(axis=0).reshape(5,2,9)
    point=metrics(overall.sum(axis=1))
    rng=np.random.default_rng(SEED)
    reps=np.zeros((B,len(blockinfo)),dtype=np.int16)
    for s in blockinfo.unique():
        ix=np.flatnonzero(blockinfo.to_numpy()==s)
        sampled=rng.choice(ix,(B,len(ix)),replace=True)
        for i in range(B):np.add.at(reps[i],sampled[i],1)
    draws=np.einsum('bk,kcj->bcj',reps,arr).reshape(B,5,2,9)
    bm=metrics(draws.sum(axis=2))
    names=['connection','frequency_week','hours_week','duration_hours','frequency_week_all_valid','backup_available']
    rows=[]
    for q in range(5):
        sub=d[d.quintile==q+1]
        row=dict(quintile=q+1,n=len(sub),n_connected=int(sub.connected.eq(1).sum()),n_paired=int(sub.paired.sum()),
                 villages=int(sub.block.nunique()),weight_share=float(sub.pw.sum()/d.pw.sum()),
                 income_min=float(sub.income_pc.min()),income_max=float(sub.income_pc.max()))
        for j,name in enumerate(names):
            row[name]=float(point[q,j]);row[name+'_lo']=float(np.nanquantile(bm[:,q,j],.025));row[name+'_hi']=float(np.nanquantile(bm[:,q,j],.975))
        rows.append(row)
    tab=pd.DataFrame(rows);tab.to_csv(P/'data/cambodia_income_quintiles.csv',index=False)
    contrasts={}
    for j,name in enumerate(names):
        rd=bm[:,0,j]/bm[:,4,j]
        dd=bm[:,0,j]-bm[:,4,j]
        contrasts[name]={'ratio':float(point[0,j]/point[4,j]),'ratio_lo':float(np.nanquantile(rd,.025)),
                         'ratio_hi':float(np.nanquantile(rd,.975)),'difference':float(point[0,j]-point[4,j]),
                         'difference_lo':float(np.nanquantile(dd,.025)),'difference_hi':float(np.nanquantile(dd,.975))}
    cell_n=d[d.paired].groupby(['quintile','Locality']).size().unstack(fill_value=0).reindex(index=range(1,6),columns=[1,2],fill_value=0)
    psett=metrics(overall)
    st=[]
    for q in range(5):
        for s in range(2):
            row={'quintile':q+1,'settlement':['Urban','Rural'][s],'n_paired':int(cell_n.loc[q+1,s+1])}
            row.update(dict(zip(names,psett[q,s].tolist())));st.append(row)
    pd.DataFrame(st).to_csv(P/'data/cambodia_settlement_groups.csv',index=False)
    standard={'admitted':bool(cell_n.loc[[1,5],[1,2]].min().min()>=30),'min_extreme_cell_n':int(cell_n.loc[[1,5],[1,2]].min().min())}
    if standard['admitted']:
        uw=float(d.loc[(d.connected==1)&(d.Locality==1),'pw'].sum()/d.loc[d.connected==1,'pw'].sum())
        pp=uw*psett[:,0,:]+(1-uw)*psett[:,1,:]
        pp[:,3]=pp[:,2]/pp[:,1]
        bb=metrics(draws)
        std=uw*bb[:,:,0,:]+(1-uw)*bb[:,:,1,:]
        std[:,:,3]=std[:,:,2]/std[:,:,1]
        standard['urban_weight']=uw
        for j,name in enumerate(names[1:4],1):
            rr=std[:,0,j]/std[:,4,j]
            standard[name]={'ratio':float(pp[0,j]/pp[4,j]),'lo':float(np.nanquantile(rr,.025)),'hi':float(np.nanquantile(rr,.975))}
    summary={'source':'World Bank MTF, Cambodia 2017','cutpoints_monthly_income_per_person_riel':cuts,
             'primary_contrast':'bottom/top household income-per-member quintile; all households define ranks',
             'households_with_weights':int(raw.pw.gt(0).sum()),'income_eligible_n':len(d),'paired_n':int(d.paired.sum()),
             'contrasts':contrasts,'settlement_standardization':standard,'bootstrap_replicates':B,'seed':SEED}
    audit={'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'raw_n':len(raw),'unique_id':int(raw.Household_ID.nunique()),
           'positive_weight_n':int(raw.pw.gt(0).sum()),'income_eligible_n':len(d),'weighted_villages':len(blockinfo),
           'bootstrap_strata':int(blockinfo.nunique()),'singleton_strata':int((blockinfo.value_counts()==1).sum()),
           'structural_zero_duration':int(structural.sum()),'invalid_hours_or_zeroF_positiveH':int(incompatible.sum()),
           'positive_F_zero_H':int(((d.F>0)&(d.H==0)).sum()),'positive_F_missing_H':int(((d.F>0)&d.H.isna()).sum()),
           'frequency_valid_n':int(d.F.notna().sum()),'paired_n':int(d.paired.sum()),
           'duration_raw_unit':'minutes per week, field questionnaire pp.20–21',
           'identity_error_max':float(np.abs(point[:,2]-point[:,1]*point[:,3]).max()),
           'earliest_interview':str(pd.Timestamp('1960-01-01')+pd.Timedelta(days=float(raw.date.min()))),
           'latest_interview':str(pd.Timestamp('1960-01-01')+pd.Timedelta(days=float(raw.date.max())))}
    (P/'metadata/cambodia_analysis_audit.json').write_text(json.dumps(audit,indent=2))
    (P/'data/cambodia_summary.json').write_text(json.dumps(summary,indent=2))
    d.drop(columns=['Household_ID']).to_parquet(P/'data/cambodia_analysis_sample.parquet',index=False)
    print(tab[['quintile','n_connected','n_paired','connection','frequency_week','hours_week','duration_hours','backup_available']].to_string(index=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
