"""Brazil resident-poverty priority accounting on a fixed 703-area cohort.

Run with Python, pandas and numpy. Input CSV snapshots preserve the native
monthly indices and the previously validated geographical/income join.
This is area-level descriptive accounting, not individual poverty exposure.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
A = P / 'analysis/poverty_priority'
I = A / 'inputs'
S = P / 'source_data'
STRATA = ['XP', 'XN', 'IP', 'IND', 'INE', 'INC', 'INO', 'IPC', 'XPC', 'XNC']
Q = ['Q1', 'Q2', 'Q3', 'Q4']
CLASSES = ['Ordinary unplanned', 'Emergency / critical', 'Covered internal unplanned']


def ci(x):
    x = np.asarray(x, float)
    good = np.isfinite(x)
    return (*np.quantile(x[good], [.025, .975]), int(good.sum())) if good.any() else (np.nan, np.nan, 0)


def ratio(h, f):
    return np.divide(h, f, out=np.full_like(np.asarray(h, float), np.nan), where=np.asarray(f) > 0)


def csv(df, name):
    df.to_csv(S / f'poverty_priority_{name}.csv', index=False)


def main(draws=4000, seed=20261003):
    A.mkdir(parents=True, exist_ok=True)
    S.mkdir(parents=True, exist_ok=True)
    context = pd.read_csv(I/'context_703.csv').sort_values('IdeConjunto').reset_index(drop=True)
    monthly = pd.read_csv(I/'native_monthly_703.csv')
    old = pd.read_csv(I/'previous_regulatory_stages.csv')
    assert len(context) == 703 and context.IdeConjunto.is_unique
    assert context.municipality_code.nunique() == 275
    assert context.common_24_months.all() and context.strict_spatial_assignment.all()
    assert set(monthly.IdeConjunto) == set(context.IdeConjunto)
    assert not monthly.duplicated(['IdeConjunto','AnoIndice','NumPeriodoIndice']).any()
    assert monthly.groupby('IdeConjunto').size().eq(24).all()
    assert set(monthly.AnoIndice) == {2024, 2025}
    assert monthly.NumCon.gt(0).all()
    municipalities = context.drop_duplicates('municipality_code')
    cuts = municipalities.low_income_share_pct.quantile([.25,.5,.75]).to_numpy()
    check_q = pd.cut(context.low_income_share_pct, [-np.inf,*cuts,np.inf], labels=Q).astype(str)
    assert check_q.eq(context.disadvantage_quartile).all()

    # Preserve incomplete strata explicitly. No absence is recoded as zero.
    native = []
    for aid, group in monthly.groupby('IdeConjunto', sort=True):
        cy = float(group.NumCon.sum()/12)
        for st in STRATA:
            valid = group[[f'FEC{st}',f'DEC{st}']].notna().all(axis=1)
            complete = bool(valid.all())
            f = float((group[f'FEC{st}']*group.NumCon).sum()/cy) if complete else np.nan
            h = float((group[f'DEC{st}']*group.NumCon).sum()/cy) if complete else np.nan
            native.append(dict(IdeConjunto=aid,stratum=st,valid_months=int(valid.sum()),
                complete_24_months=complete,customer_years=cy,F=f,H=h,
                observation_status='reported' if complete else 'not completely reported'))
    native = pd.DataFrame(native)
    csv(native, 'native_stratum_support')
    support = native.groupby('stratum').agg(complete_areas=('complete_24_months','sum'),
        available_months=('valid_months','sum')).reset_index()
    support['cohort_areas'] = len(context)
    csv(support, 'native_stratum_coverage')
    wideF = native.pivot(index='IdeConjunto', columns='stratum', values='F').loc[context.IdeConjunto]
    wideH = native.pivot(index='IdeConjunto', columns='stratum', values='H').loc[context.IdeConjunto]
    blocks = []
    for name, parts in zip(CLASSES, [['IND'],['INE','INC'],['IND','INE','INC']]):
        frame = context[['IdeConjunto','municipality_code','state_code','disadvantage_quartile','low_income_share_pct','customer_years']].copy()
        frame['class'] = name
        frame['F'] = wideF[parts].sum(axis=1, min_count=len(parts)).to_numpy()
        frame['H'] = wideH[parts].sum(axis=1, min_count=len(parts)).to_numpy()
        frame['D'] = ratio(frame.H.to_numpy(), frame.F.to_numpy())
        assert frame[['F','H']].notna().all().all()
        blocks.append(frame)
    regulatory = pd.concat(blocks,ignore_index=True)
    # Existing table calls the sum "Internal unplanned".
    comp = regulatory.assign(category=regulatory['class'].replace({'Covered internal unplanned':'Internal unplanned'})).merge(
        old, on=['IdeConjunto','category'], suffixes=('','_old'), validate='one_to_one')
    errors = dict(F=float(np.abs(comp.F-comp.frequency).max()),H=float(np.abs(comp.H-comp.burden_hours).max()))
    assert max(errors.values()) < 1e-9
    assert np.allclose(regulatory[regulatory['class'].eq(CLASSES[-1])].F,
        blocks[0].F.to_numpy()+blocks[1].F.to_numpy())
    assert np.allclose(blocks[2].H, blocks[0].H+blocks[1].H)
    csv(regulatory,'area_metrics')
    mapframe = context.copy()
    for key,b in zip(['ordinary','emergency','covered_internal_unplanned'],blocks):
        for metric in ['F','H','D']:mapframe[f'{key}_{metric}'] = b[metric].to_numpy()
    csv(mapframe,'area_context')

    # Fixed municipal ranks. One cluster draw is reused for every outcome,
    # class and weighting scheme, preserving all cross-class covariance.
    rng = np.random.default_rng(seed)
    cluster_draws = {}
    cluster_ids = {}
    for q in Q:
        ids = np.sort(context.loc[context.disadvantage_quartile.eq(q),'municipality_code'].unique())
        cluster_ids[q] = ids
        cluster_draws[q] = rng.multinomial(len(ids), np.ones(len(ids))/len(ids), size=draws)

    group_rows, priority_rows, gap_rows, covariance_rows = [], [], [], []
    draws_out = {}
    max_identity_error = 0.

    def analyse(frame, classes, domain, scope, f_unit, h_unit):
        nonlocal max_identity_error
        tensor = np.stack([frame.pivot(index='IdeConjunto',columns='class',values=m).loc[context.IdeConjunto,classes].to_numpy()
                           for m in ['F','H']], axis=-1)
        assert np.isfinite(tensor).all() and (tensor >= 0).all()
        for weighting in ['area_equal','municipality_equal']:
            pts, boots, denoms, bootdenoms = [], [], [], []
            for q in Q:
                idx = np.flatnonzero(context.disadvantage_quartile.eq(q))
                members = [idx[context.iloc[idx].municipality_code.eq(mid).to_numpy()] for mid in cluster_ids[q]]
                if weighting == 'area_equal':
                    sums = np.stack([tensor[v].sum(axis=0) for v in members])
                    weights = np.array([len(v) for v in members],float)
                else:
                    sums = np.stack([tensor[v].mean(axis=0) for v in members])
                    weights = np.ones(len(members))
                denoms.append(weights.sum())
                bootden = cluster_draws[q]@weights
                bootdenoms.append(bootden)
                pts.append(sums.sum(axis=0)/weights.sum())
                boots.append(np.einsum('bi,iko->bko',cluster_draws[q],sums)/bootden[:,None,None])
            pts = np.stack(pts)
            boots = np.stack(boots, axis=1)
            denoms = np.asarray(denoms)
            bootdenoms = np.stack(bootdenoms,axis=1)
            overall = np.einsum('q,qko->ko',denoms,pts)/denoms.sum()
            overall_b = np.einsum('bq,bqko->bko',bootdenoms,boots)/bootdenoms.sum(axis=1)[:,None,None]
            values = np.concatenate([pts,overall[None]],axis=0)
            reps = np.concatenate([boots,overall_b[:,None]],axis=1)
            for iq,label in enumerate(Q+['All admitted areas']):
                denomH = values[iq,-1,1]
                for ic,cl in enumerate(classes):
                    f,h = values[iq,ic]
                    metrics = {'F':(f,reps[:,iq,ic,0]),'H':(h,reps[:,iq,ic,1]),
                        'D':(ratio(h,f),ratio(reps[:,iq,ic,1],reps[:,iq,ic,0])),
                        'H_share_pct':(100*h/denomH,100*reps[:,iq,ic,1]/reps[:,iq,-1,1])}
                    row=dict(domain=domain,scope=scope,weighting=weighting,group=label,**{'class':cl},F_unit=f_unit,H_unit=h_unit,D_unit='hours per interruption' if domain=='customer' else 'hours per reconstructed record')
                    for name,(point,rep) in metrics.items():
                        lo,hi,n=ci(rep);row.update({name:float(point),name+'_ci_low':lo,name+'_ci_high':hi,name+'_valid_draws':n})
                    group_rows.append(row)

            base = (pts[0]+pts[3])/2
            base_b = (boots[:,0]+boots[:,3])/2
            delta = pts[3]-pts[0]
            delta_b = boots[:,3]-boots[:,0]
            for ic,cl in enumerate(classes):
                loF,loH = pts[0,ic]; hiF,hiH = pts[3,ic]
                loD,hiD = ratio(loH,loF),ratio(hiH,hiF)
                bLoF,bLoH=boots[:,0,ic,0],boots[:,0,ic,1]
                bHiF,bHiH=boots[:,3,ic,0],boots[:,3,ic,1]
                bLoD,bHiD=ratio(bLoH,bLoF),ratio(bHiH,bHiF)
                fterm=(hiF-loF)*(hiD+loD)/2
                dterm=(hiD-loD)*(hiF+loF)/2
                bf=(bHiF-bLoF)*(bHiD+bLoD)/2
                bd=(bHiD-bLoD)*(bHiF+bLoF)/2
                if np.isfinite(fterm+dterm):max_identity_error=max(max_identity_error,abs(fterm+dterm-(hiH-loH)))
                row=dict(domain=domain,scope=scope,weighting=weighting,**{'class':cl},contrast='Q4 minus Q1',F_unit=f_unit,H_unit=h_unit,
                    baseline_definition='Arithmetic mean of Q1 and Q4 means; groups equally weighted',
                    Q1_F=loF,Q1_H=loH,Q1_D=float(loD),Q4_F=hiF,Q4_H=hiH,Q4_D=float(hiD))
                metrics={'baseline_F':(base[ic,0],base_b[:,ic,0]),'baseline_H':(base[ic,1],base_b[:,ic,1]),
                    'baseline_F_share_pct':(100*base[ic,0]/base[-1,0],100*base_b[:,ic,0]/base_b[:,-1,0]),
                    'baseline_H_share_pct':(100*base[ic,1]/base[-1,1],100*base_b[:,ic,1]/base_b[:,-1,1]),
                    'overall_H':(overall[ic,1],overall_b[:,ic,1]),
                    'overall_H_share_pct':(100*overall[ic,1]/overall[-1,1],100*overall_b[:,ic,1]/overall_b[:,-1,1]),
                    'gap_F':(delta[ic,0],delta_b[:,ic,0]),'gap_H':(delta[ic,1],delta_b[:,ic,1]),
                    'gap_D':(hiD-loD,bHiD-bLoD),
                    'frequency_component_H':(fterm,bf),'duration_component_H':(dterm,bd),
                    'frequency_minus_duration_component_H':(fterm-dterm,bf-bd)}
                for name,(point,rep) in metrics.items():
                    low,high,n=ci(rep);row.update({name:float(point),name+'_ci_low':low,name+'_ci_high':high,name+'_valid_draws':n})
                priority_rows.append(row)
                for key in ['gap_F','gap_H','gap_D','frequency_component_H','duration_component_H','frequency_minus_duration_component_H']:
                    val,rep=metrics[key];low,high,n=ci(rep)
                    gap_rows.append(dict(domain=domain,scope=scope,weighting=weighting,**{'class':cl},outcome=key,estimate=float(val),ci_low=low,ci_high=high,valid_draws=n))
            assert np.allclose(delta[:-1,1].sum(),delta[-1,1],rtol=1e-12,atol=1e-10)
            assert np.allclose(delta_b[:,:-1,1].sum(axis=1),delta_b[:,-1,1],rtol=1e-12,atol=1e-9)
            cov=np.cov(delta_b[:,:,1],rowvar=False,ddof=1)
            for ix,c1 in enumerate(classes):
                for iy,c2 in enumerate(classes):
                    covariance_rows.append(dict(domain=domain,weighting=weighting,class_1=c1,class_2=c2,gap_H_covariance=cov[ix,iy],draws=draws))
            draws_out[f'{domain}_{weighting}_Q1_Q4_F_H']=boots[:,[0,3]]
            draws_out[f'{domain}_{weighting}_class_gap_H']=delta_b[:,:,1]
            draws_out[f'{domain}_{weighting}_classes']=np.asarray(classes)

    analyse(regulatory, CLASSES, 'customer', 'IND + INE + INC only; external/planned/other strata outside this denominator',
            'customer interruptions per customer-year','hours per customer-year')

    fine=pd.read_csv(I/'fine_cause_stages_703.csv')
    causes=sorted(fine.cause.unique())
    assert len(fine)==703*len(causes)
    assert not fine.duplicated(['IdeConjunto','cause']).any()
    assert set(fine.IdeConjunto)==set(context.IdeConjunto)
    # Counts are the reconstructed lower bound, not distinct identifiers or
    # customer interruptions. The exact 76-record global ambiguity is retained.
    fine['class']=fine.cause
    fine['F']=fine.record_rate
    fine['H']=fine.record_hours_rate
    total=fine.groupby('IdeConjunto')[['F','H']].sum().reset_index()
    total['class']='All reconstructed records'
    fine_full=pd.concat([fine[['IdeConjunto','class','F','H']],total],ignore_index=True)
    fine_full=fine_full.merge(context[['IdeConjunto','municipality_code','disadvantage_quartile']],on='IdeConjunto',validate='many_to_one')
    analyse(fine_full,causes+['All reconstructed records'],'record',
            'Fourteen source-native cause groups; reconstructed records, not customer interruptions',
            'reconstructed records per customer-year','record-hours per customer-year')
    csv(fine,'fine_area_records')
    csv(pd.DataFrame(group_rows),'group_summary')
    csv(pd.DataFrame(priority_rows),'priority_summary')
    csv(pd.DataFrame(gap_rows),'gap_components')
    csv(pd.DataFrame(covariance_rows),'class_gap_covariance')
    np.savez_compressed(A/'joint_bootstrap_draws.npz',**draws_out)
    # Parallel-profile schema: "low" denotes the disadvantaged social group,
    # so the Brazilian poverty Q4 is mapped to low, never to poverty Q1.
    priority=pd.DataFrame(priority_rows)
    aligned=[]
    pairwise=[]
    for weighting in ['area_equal','municipality_equal']:
        z=priority[priority.domain.eq('customer') & priority.weighting.eq(weighting)].set_index('class').loc[CLASSES]
        b=draws_out[f'customer_{weighting}_Q1_Q4_F_H']
        bbase=(b[:,0]+b[:,1])/2
        bgap=b[:,1,:,1]-b[:,0,:,1]
        ranks={key:vals.rank(ascending=False,method='min') for key,vals in {
            'F':z.baseline_F.iloc[:2],'H':z.baseline_H.iloc[:2],'gap':z.gap_H.iloc[:2]}.items()}
        winners={'F':np.argmax(bbase[:,:2,0],axis=1),'H':np.argmax(bbase[:,:2,1],axis=1),'gap':np.argmax(bgap[:,:2],axis=1)}
        for idx,cl in enumerate(CLASSES):
            r=z.loc[cl]
            row=dict(profile='Brazil resident-income-share quartiles',country='Brazil',period='2024–2025',
                weighting=weighting,scope=r.scope,category=cl,category_type='regulatory interruption class',rank_eligible=idx<2,
                disadvantaged_group='Q4 municipal low-income share',advantaged_group='Q1 municipal low-income share',
                F_low=r.Q4_F,F_high=r.Q1_F,H_low=r.Q4_H,H_high=r.Q1_H,
                F_balanced_mean=r.baseline_F,H_balanced_mean=r.baseline_H,H_gap=r.gap_H,
                F_share_pct=r.baseline_F_share_pct,H_share_pct=r.baseline_H_share_pct,
                signed_gap_share_pct=np.nan,F_lower=r.baseline_F_ci_low,F_upper=r.baseline_F_ci_high,
                H_lower=r.baseline_H_ci_low,H_upper=r.baseline_H_ci_high,
                gap_lower=r.gap_H_ci_low,gap_upper=r.gap_H_ci_high)
            for key in ['F','H','gap']:
                row[key+'_rank']=ranks[key].iloc[idx] if idx<2 else np.nan
                row[key+'_top_rank_support']=float(np.mean(winners[key]==idx)) if idx<2 else np.nan
            aligned.append(row)
        for name,point,rep in [
            ('baseline_H_ordinary_minus_emergency',z.baseline_H.iloc[0]-z.baseline_H.iloc[1],bbase[:,0,1]-bbase[:,1,1]),
            ('gap_H_ordinary_minus_emergency',z.gap_H.iloc[0]-z.gap_H.iloc[1],bgap[:,0]-bgap[:,1])]:
            low,high,n=ci(rep)
            pairwise.append(dict(weighting=weighting,contrast=name,estimate=point,ci_low=low,ci_high=high,valid_draws=n,
                unit='hours per customer-year',scope='IND + INE + INC only'))
    csv(pd.DataFrame(aligned),'aligned_class_estimates')
    csv(pd.DataFrame(pairwise),'class_priority_contrasts')
    descriptive=[]
    for (cl,q),x in regulatory.groupby(['class','disadvantage_quartile']):
        for m in ['F','H','D']:
            y=x[m]
            descriptive.append(dict(category=cl,quartile=q,metric=m,n=len(y),missing=int(y.isna().sum()),
                zeros=int(y.eq(0).sum()),mean=y.mean(),sd=y.std(),median=y.median(),p25=y.quantile(.25),p75=y.quantile(.75),min=y.min(),max=y.max()))
    csv(pd.DataFrame(descriptive),'outcome_descriptives')
    support_rows=[]
    for q,x in context.groupby('disadvantage_quartile',sort=True):
        support_rows.append(dict(group=q,areas=len(x),municipalities=x.municipality_code.nunique(),
            states=x.state_code.nunique(),reporting_contexts=x.cnpj.nunique(),customer_years=x.customer_years.sum(),
            low_income_share_min_pct=x.low_income_share_pct.min(),low_income_share_max_pct=x.low_income_share_pct.max(),
            served_customer_density_median=x.density.median(),area_counts_max_per_municipality=x.groupby('municipality_code').size().max()))
    csv(pd.DataFrame(support_rows),'group_support')
    audit=dict(cohort_areas=703,municipalities=275,states=int(context.state_code.nunique()),months_per_area=24,
        years=[2024,2025],customer_years=float(context.customer_years.sum()),quartile_cutpoints_pct=cuts.tolist(),
        bootstrap_draws=draws,seed=seed,bootstrap_unit='Municipality; fixed poverty-quartile strata; whole class vectors retained',
        weighting=['area_equal','municipality_equal'],source_reproduction_max_absolute_error=errors,
        decomposition_identity_max_error=max_identity_error,
        all_area_F_H_nonnegative=bool(regulatory[['F','H']].ge(0).all().all()),
        undefined_area_D=int(regulatory.D.isna().sum()),
        primary_classes_complete=True,external_XPC_XNC_complete_areas=100,external_unknown_areas=603,
        absent_external_categories_imputed_zero=False,household_outage_assignment=False,
        fine_lower_records=int(fine.records.sum()),fine_reference_records=int(context.records_reference.sum()),
        fine_unallocated_zero_duration_records=int(context.unresolved_count.sum()),
        fine_cause_upper_bounds_not_jointly_additive=float(fine.extra_zero_upper.sum()),
        fine_record_hours=float(fine.record_hours.sum()),
        fine_zero_count_interpretation='76 records are missing from lower-bound cause reconstruction. Cause-specific allocation upper bounds overlap and must not be summed (their sum is 89). This does not affect record-hour totals.',
        python=sys.version,executable=sys.executable,pandas=pd.__version__,numpy=np.__version__,platform=platform.platform())
    assert audit['fine_reference_records']-audit['fine_lower_records']==76
    assert max_identity_error<1e-8
    manifest=json.loads((A/'input_manifest.json').read_text())
    for item in manifest['snapshot_files']:
        assert hashlib.sha256((P/item['path']).read_bytes()).hexdigest()==item['sha256']
    audit['input_hashes_verified']=True
    audit['status']='pass'
    (A/'self_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    p=pd.DataFrame(priority_rows)
    print(p[p.domain.eq('customer')][['weighting','class','Q1_F','Q1_H','Q1_D','Q4_F','Q4_H','Q4_D','baseline_H_share_pct','gap_H','gap_H_ci_low','gap_H_ci_high','frequency_component_H','duration_component_H']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--draws',type=int,default=4000)
    parser.add_argument('--seed',type=int,default=20261003)
    args=parser.parse_args()
    main(args.draws,args.seed)
