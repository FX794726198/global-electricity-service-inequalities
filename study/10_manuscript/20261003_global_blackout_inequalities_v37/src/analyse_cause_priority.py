"""Paired cause/class priorities: frequency, annual hours, and social hour gaps.

Uses only the frozen v33 observations. Run with Python + pandas/numpy/scipy/pyarrow.
No source file is modified. Definitions are written into summary.md and the manifest.
"""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
SRC = PROJECT / '10_manuscript/20261003_global_blackout_inequalities_v33'
EXT = SRC / 'core_reproducibility/class_gap_extension'
OUT = ROOT / 'analysis/cause_priority'
SD = ROOT / 'source_data'
SEED = 202610037
N = 10000
manifest = []
checks = []

def read(path):
    path = Path(path)
    manifest.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)

def check(name, condition, detail=''):
    checks.append({'check': name, 'passed': bool(condition), 'detail': detail})
    if not condition:
        raise AssertionError((name, detail))

def add_units(df, region, unit, group, cluster, weight, mappings, period, social,
              category_type):
    rows = []
    for category, f, h in mappings:
        z = pd.DataFrame({'profile': region, 'country': region, 'unit': df[unit].astype(str),
                         'group': df[group].astype(str), 'cluster': df[cluster].astype(str),
                         'customers': pd.to_numeric(df[weight]), 'category': category,
                         'F': pd.to_numeric(df[f]), 'H': pd.to_numeric(df[h]),
                         'rank_eligible': True, 'period': period, 'social': social,
                         'category_type': category_type})
        rows.append(z)
    return pd.concat(rows, ignore_index=True)

def prepare():
    frames = []
    b = read(EXT / 'inputs/brazil_2021_joint_context.parquet')
    frames.append(add_units(b, 'Brazil', 'IdeConjUndConsumidoras', 'income_quartile',
        'municipality_code', 'mean_customers', [(c, 'FEC'+s, 'DEC'+s) for c,s in
        [('Ordinary unplanned','IND'),('Planned','IP'),('Emergency','INE'),('Critical day','INC')]],
        '2021', 'Municipal GDP per person', 'Administrative interruption classes'))
    u = read(EXT / 'inputs/us_same49_state_means.parquet')
    frames.append(add_units(u, 'United States', 'state', 'quartile', 'state', 'customers',
        [('Routine / non-major','F_routine','H_routine'),
         ('Exceptional / major','F_exceptional','H_exceptional')],
        '2013–2024', 'State household income', 'Major-event administrative classes'))
    n = read(EXT / 'inputs/nz_explicit21_total_context.parquet')
    n['quartile'] = pd.qcut(n.income_grouped_median_nzd.rank(method='first'),4,
                           labels=['Q1','Q2','Q3','Q4']).astype(str)
    n = read(EXT / 'inputs/nz_explicit21_complete_partition.parquet').merge(
        n[['system','quartile','customers']], on='system', validate='many_to_one')
    z = pd.DataFrame({'profile':'New Zealand','country':'New Zealand','unit':n.system,
        'cluster':n.system,'group':n.quartile,'customers':n.customers,'category':n.category,
        'F':n.frequency,'H':n.burden_h,'rank_eligible':True,'period':'2023–2025',
        'social':'Geographically allocated personal income','category_type':'Reported cause groups'})
    frames.append(z)
    e = read(EXT / 'data/frozen_accounting_inputs.csv')
    assembled = pd.concat(frames, ignore_index=True)
    for region in ['Brazil','New Zealand','United States']:
        a = assembled[(assembled.profile==region)&assembled.group.isin(['Q1','Q4'])]
        old = e[e.region==region].copy();old['unit']=old.unit.astype(str)
        m = a.merge(old, on=['unit','category'], validate='one_to_one')
        check(region+' exact frozen extreme-group membership',len(m)==len(a)==len(old))
        check(region+' unchanged frozen group labels',(m.group==m.quartile).all())
        check(region+' unchanged frozen F/H',np.allclose(m.F,m.frequency) and
              np.allclose(m.H,m.burden_h))
    t = read(SRC / 'source_data/turkiye_multiregion_cause_settlement_metrics_2025.csv')
    annual = read(SRC / 'core_reproducibility/turkiye_extension/data/annual_settlement_indices_2025.csv')
    key = read(SRC / 'source_data/turkiye_multiregion_key_regional_results_2025.csv')
    native_gap = read(SRC / 'source_data/turkiye_multiregion_cause_contributions_to_gap_2025.csv')
    regions = key.set_index('provider').region.to_dict()
    t = t[t.voltage=='All'].copy()
    for provider, df in t.groupby('provider',sort=True):
        profile = 'Türkiye: '+regions[provider]
        rows = []
        aa = annual[(annual.provider==provider)&(annual.voltage=='All')&
                    (annual.notification=='Unannounced')].set_index('settlement')
        for _, r in df.iterrows():
            rows.append({'profile':profile,'country':'Türkiye','unit':r.settlement,
                'cluster':r.settlement,'group':{'Rural':'Q1','Urban':'Q4','Suburban':'Q2'}[r.settlement],
                'customers':r.users,'category':r.cause_group,'F':r.initial_stage_rate,
                'H':r.hours_per_customer,'rank_eligible':True,'period':'2025',
                'social':'Rural versus urban settlement','category_type':'Reported cause groups'})
        for settlement,g in df.groupby('settlement'):
            r = aa.loc[settlement]
            rows.append({'profile':profile,'country':'Türkiye','unit':settlement,'cluster':settlement,
                'group':{'Rural':'Q1','Urban':'Q4','Suburban':'Q2'}[settlement],
                'customers':r.users,'category':'Reconciliation residual',
                'F':float(r.frequency-g.initial_stage_rate.sum()),
                'H':float(r.hours-g.hours_per_customer.sum()),'rank_eligible':False,
                'period':'2025','social':'Rural versus urban settlement',
                'category_type':'Accounting residual; not a cause'})
        z = pd.DataFrame(rows)
        for settlement in ['Rural','Urban','Suburban']:
            zz = z[z.unit==settlement]
            check(profile+' '+settlement+' annual total F/H closure',
                  np.isclose(zz.F.sum(),aa.loc[settlement,'frequency']) and
                  np.isclose(zz.H.sum(),aa.loc[settlement,'hours']))
        zz = z.pivot(index='category',columns='unit',values='H')
        ng = native_gap[(native_gap.provider==provider)&(native_gap.voltage=='All')].set_index('cause_group')
        check(profile+' existing cause-gap reconstruction',
              np.allclose((zz.Rural-zz.Urban).reindex(ng.index),ng.gap_h,atol=1e-9))
        frames.append(z)
    return pd.concat(frames, ignore_index=True)

def weighted_mean(z, categories, weights='equal_unit'):
    p = z.pivot(index=['unit','cluster','customers'],columns='category',values=['F','H'])
    p = p.reindex(columns=pd.MultiIndex.from_product([['F','H'],categories]))
    check('complete category partition '+z.profile.iloc[0]+' '+','.join(sorted(z.group.unique())),
          p.notna().all().all())
    w = np.ones(len(p)) if weights=='equal_unit' else p.index.get_level_values('customers').to_numpy()
    return np.average(p.to_numpy(),axis=0,weights=w),p

def bootstrap_group(p, rng, n=N):
    vals=p.to_numpy(); clusters=p.index.get_level_values('cluster').to_numpy()
    unique=np.unique(clusters)
    sums=np.stack([vals[clusters==c].sum(axis=0) for c in unique])
    counts=np.array([(clusters==c).sum() for c in unique])
    inds=rng.integers(0,len(unique),size=(n,len(unique)))
    return sums[inds].sum(axis=1)/counts[inds].sum(axis=1)[:,None]

def rank_summary(a,b):
    ra=rankdata(-a,method='average');rb=rankdata(-b,method='average')
    return {'kendall_tau':float(kendalltau(a,b).statistic),
            'mean_absolute_rank_shift':float(np.mean(np.abs(ra-rb)))}

def run():
    OUT.mkdir(parents=True,exist_ok=True);SD.mkdir(exist_ok=True)
    inputs=prepare(); inputs.to_csv(SD/'priority_paired_inputs.csv',index=False)
    rng=np.random.default_rng(SEED)
    results=[]; profiles=[]; pairwise=[]; sens=[]; draws_out=[]
    for profile,z in inputs.groupby('profile',sort=True):
        cats=sorted(z.category.unique()); k=len(cats)
        eligible=np.array([c!='Reconciliation residual' for c in cats]);ec=np.array(cats)[eligible]
        low,pl=weighted_mean(z[z.group=='Q1'],cats)
        high,ph=weighted_mean(z[z.group=='Q4'],cats)
        point={'F':(low[:k]+high[:k])/2,'H':(low[k:]+high[k:])/2,
               'gap':low[k:]-high[k:]}
        point['gap_excess_vs_hours_mix'] = point['gap'] - point['H']/point['H'].sum()*point['gap'].sum()
        domestic=z.country.iloc[0]!='Türkiye'
        if domestic:
            bl=bootstrap_group(pl,rng);bh=bootstrap_group(ph,rng)
            draw={'F':(bl[:,:k]+bh[:,:k])/2,'H':(bl[:,k:]+bh[:,k:])/2,
                  'gap':bl[:,k:]-bh[:,k:]}
            draw['gap_excess_vs_hours_mix'] = draw['gap'] - (
                draw['H']/draw['H'].sum(axis=1)[:,None]*draw['gap'].sum(axis=1)[:,None])
        else:
            draw=None
        row={'profile':profile,'country':z.country.iloc[0],'period':z.period.iloc[0],
             'social':z.social.iloc[0],'category_type':z.category_type.iloc[0],
             'n_all_units':z.unit.nunique(),'n_low_units':len(pl),'n_high_units':len(ph),
             'n_low_clusters':pl.index.get_level_values('cluster').nunique(),
             'n_high_clusters':ph.index.get_level_values('cluster').nunique(),
             'total_F_low':float(low[:k].sum()),'total_F_high':float(high[:k].sum()),
             'total_H_low':float(low[k:].sum()),'total_H_high':float(high[k:].sum()),
             'total_H_gap':float(point['gap'].sum()),'uncertainty':
             'Geographic cluster bootstrap within fixed economic groups' if domestic else
             'Administrative annual population; no sampling confidence interval'}
        net=point['gap'].sum()
        if domestic:
            gl,gh=np.quantile(draw['gap'].sum(axis=1),[.025,.975]);resolved=gl>0 or gh<0
        else:
            gl=gh=np.nan;resolved=True
        row.update(total_H_gap_lower=gl,total_H_gap_upper=gh,gap_direction_resolved=bool(resolved))
        ranks={m:rankdata(-point[m][eligible],method='average') for m in ['F','H','gap']}
        for m in ['F','H','gap']:
            v=point[m][eligible];order=np.argsort(-v,kind='stable')
            leader=str(ec[order[0]]);runner=str(ec[order[1]])
            row[m+'_leader']=leader;row[m+'_runner_up']=runner
            row[m+'_leader_margin']=float(v[order[0]]-v[order[1]])
            if domestic:
                dd=draw[m][:,eligible];winner=dd.argmax(axis=1)
                row[m+'_leader_bootstrap_support']=float(np.mean(winner==order[0]))
                margin=dd[:,order[0]]-dd[:,order[1]]
                row[m+'_leader_margin_lower'],row[m+'_leader_margin_upper']=np.quantile(margin,[.025,.975])
        for a,b in [('F','H'),('H','gap'),('F','gap')]:
            row[a+'_'+b+'_leader_changes']=row[a+'_leader']!=row[b+'_leader']
            row.update({a+'_'+b+'_'+n:v for n,v in rank_summary(point[a][eligible],point[b][eligible]).items()})
            if domestic:
                row[a+'_'+b+'_bootstrap_leader_changes']=float(np.mean(
                    draw[a][:,eligible].argmax(axis=1)!=draw[b][:,eligible].argmax(axis=1)))
        for j,c in enumerate(cats):
            q={'profile':profile,'country':z.country.iloc[0],'period':z.period.iloc[0],
               'category':c,'category_type':z.category_type.iloc[0] if eligible[j] else 'Accounting residual; not a cause',
               'rank_eligible':bool(eligible[j]),'F_low':low[j],'F_high':high[j],
               'H_low':low[k+j],'H_high':high[k+j],'F_balanced_mean':point['F'][j],
               'H_balanced_mean':point['H'][j],'H_gap':point['gap'][j],
               'F_share_pct':100*point['F'][j]/point['F'].sum(),
               'H_share_pct':100*point['H'][j]/point['H'].sum(),
               'gap_excess_vs_hours_mix_h':point['gap_excess_vs_hours_mix'][j],
               'signed_gap_share_pct':100*point['gap'][j]/net if resolved and net!=0 else np.nan}
            q['gap_prominence_minus_hours_share_pp'] = q['signed_gap_share_pct']-q['H_share_pct']
            q['hours_minus_frequency_share_pp'] = q['H_share_pct']-q['F_share_pct']
            if domestic:
                q['gap_excess_lower'],q['gap_excess_upper']=np.quantile(
                    draw['gap_excess_vs_hours_mix'][:,j],[.025,.975])
                shift = 100*(draw['H'][:,j]/draw['H'].sum(axis=1)-
                             draw['F'][:,j]/draw['F'].sum(axis=1))
                q['hours_minus_frequency_share_lower'],q['hours_minus_frequency_share_upper']=np.quantile(shift,[.025,.975])
            if eligible[j]:
                j2=int(np.where(ec==c)[0][0])
                for m in ['F','H','gap']:q[m+'_rank']=ranks[m][j2]
            for m in ['F','H','gap']:
                if domestic:
                    q[m+'_lower'],q[m+'_upper']=np.quantile(draw[m][:,j],[.025,.975])
                    if eligible[j]:
                        q[m+'_top_rank_support']=float(np.mean(draw[m][:,eligible].argmax(axis=1)==j2))
            results.append(q)
        for a,b in [('F','H'),('H','gap'),('F','gap')]:
            for i,c1 in enumerate(ec):
                for j,c2 in enumerate(ec):
                    if j<=i:continue
                    ia=cats.index(c1);ib=cats.index(c2)
                    q={'profile':profile,'metric_1':a,'metric_2':b,'category_1':c1,'category_2':c2,
                       'contrast_1':point[a][ia]-point[a][ib],'contrast_2':point[b][ia]-point[b][ib]}
                    q['point_order_reversal']=q['contrast_1']*q['contrast_2']<0
                    if domestic:
                        x=draw[a][:,ia]-draw[a][:,ib];y=draw[b][:,ia]-draw[b][:,ib]
                        q['bootstrap_order_reversal_support']=float(np.mean(x*y<0))
                        for label,arr in [('contrast_1',x),('contrast_2',y)]:
                            q[label+'_lower'],q[label+'_upper']=np.quantile(arr,[.025,.975])
                    pairwise.append(q)
        # Explicitly separate alternative overall weights from the fixed primary gap.
        # Only the priority of the overall F/H vector changes; no new social gap is implied.
        for population,zz in [('contrast_groups',z[z.group.isin(['Q1','Q4'])]),('all_admitted_groups',z)]:
            for weight in ['equal_unit','customers']:
                mm,_=weighted_mean(zz,cats,weight)
                for metric,vals in [('F',mm[:k]),('H',mm[k:])]:
                    vv=vals[eligible]; leader=str(ec[np.argmax(vv)])
                    for j,c in enumerate(cats):
                        sens.append({'profile':profile,'population':population,'overall_weight':weight,
                            'metric':metric,'category':c,'mean_index':vals[j],
                            'share_pct':100*vals[j]/vals.sum(),'leader':leader,
                            'gap_leader':row['gap_leader'],'leader_differs_from_primary_gap':leader!=row['gap_leader'],
                            'interpretation':'Alternative overall mean indices; primary social gap remains equal-unit within group'})
        if domestic:
            for d in range(N):
                dd={'profile':profile,'draw':d,'total_gap':draw['gap'][d].sum()}
                for m in ['F','H','gap']:
                    dd[m+'_leader']=str(ec[np.argmax(draw[m][d,eligible])])
                draws_out.append(dd)
        profiles.append(row)
        check(profile+' exact signed category-gap accounting',np.isclose(point['gap'].sum(),low[k:].sum()-high[k:].sum()))
        check(profile+' excess contributions sum to zero',np.isclose(point['gap_excess_vs_hours_mix'].sum(),0,atol=1e-10))
        check(profile+' real causes/classes nonnegative',z.loc[z.rank_eligible,['F','H']].ge(0).all().all())
    outputs={'class_estimates':pd.DataFrame(results),'profile_summary':pd.DataFrame(profiles),
             'pairwise_ranking':pd.DataFrame(pairwise),'overall_weight_sensitivity':pd.DataFrame(sens)}
    for name,df in outputs.items():df.to_csv(SD/f'priority_{name}.csv',index=False)
    pd.DataFrame(draws_out).to_parquet(OUT/'bootstrap_leaders.parquet',index=False)
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'validation.json').write_text(json.dumps({'seed':SEED,'draws':N,'checks':checks,
        'all_passed':all(c['passed'] for c in checks)},indent=2))
    write_report(outputs)
    print(outputs['profile_summary'][['profile','F_leader','H_leader','gap_leader','H_gap_leader_changes',
                                    'total_H_gap','total_H_gap_lower','total_H_gap_upper']].to_string(index=False))

def write_report(outputs):
    s=outputs['profile_summary'];c=outputs['class_estimates'];w=outputs['overall_weight_sensitivity']
    lines=['# Service-loss priorities and social-gap priorities\n',
      '## Question and study population\n',
      'Within the same observation population and period, does the category responsible for the largest average interruption frequency or average annual interruption hours also contribute the largest signed low-minus-high social-group difference in annual interruption hours? The analysis comprises four countries: Brazil, New Zealand, the United States and Türkiye. Türkiye contributes four geographically distinct regional comparisons; these are not four independent country replications. No pooled global percentage of reversals is estimated.\n',
      'The economic comparisons retain the previously fixed lowest and highest quartiles: 138 and 283 Brazilian regulatory areas in 84 municipalities per group (2021); six and five New Zealand supply regions (2023–2025); and 13 and 12 US reporting states (2013–2024). Türkiye uses rural and urban registered accounts in four regional administrative populations in 2025. The full admitted populations contain 814 Brazilian areas, 21 New Zealand regions and 49 US states; middle groups enter only explicitly labelled alternative overall-priority calculations.\n',
      '## Definitions and methods\n',
      'For category k, F_gk denotes mean annual customer-interruption frequency and H_gk denotes mean annual interruption hours per customer within group g. The primary overall benchmark gives the two contrasted social groups equal weight: Fbar_k = (F_low,k + F_high,k)/2 and Hbar_k = (H_low,k + H_high,k)/2. The signed inequality contribution is G_k = H_low,k − H_high,k. The category contributions sum exactly to the total mean annual-hour gap. Ranking G_k uses its signed value, not its absolute magnitude: a negative contribution offsets the low-group disadvantage. These are average customer-level annual indices, not literal total customer-hour losses or economic damages.\n',
      'Brazilian groups use equal area weights, US groups equal state weights, and New Zealand groups equal geographical supply-region weights. Separate sensitivity tables retain the same paired populations while comparing pooled equal-unit and customer-weighted overall priorities; a further explicitly labelled analysis includes all admitted economic/settlement groups. Customer weights are available annual or period-mean account denominators, so period means are not literal sums of all lost customer-hours over the study period. The fixed primary gap is not silently reweighted.\n',
      'Brazilian ordinary unplanned, planned, emergency and critical-day classes are administrative interruption categories. US routine/non-major and exceptional/major-event classes are administrative definitions, not natural-hazard labels. New Zealand uses reported weather, human-activity and remaining-cause groups. Turkish causes retain the frozen initiating-description crosswalk, including unresolved fire and other causes; equipment/electrical conditions are not automatically labelled anthropogenic. Southern Marmara does not distinguish weather initiation, which is not recoded as zero weather risk.\n',
      'For the three economic comparisons, 10,000 paired bootstrap replicates resample complete geographical clusters separately within fixed groups (seed 202610037): municipalities in Brazil, states in the US and supply regions in New Zealand. All categories travel together. Percentile intervals describe geographical sampling/composition variation; top-rank support and order-reversal support are fractions of these resamples, not causal probabilities. Turkish administrative annual totals receive no fabricated sampling intervals. Their published annual indices remain the denominator-consistent total. Differences between summed source cause records and the published indices are shown as a signed reconciliation residual, excluded from cause rankings but retained in all accounting sums. First-stage customer counts are used only for Turkish sources where that rule was verified; the Bolu–Sakarya export uses its documented customer-interruption accounting.\n',
      'Cause/class shares divide a category index by the corresponding summed index. Signed gap shares are reported only when the bootstrap interval for the total gap excludes zero, or for the Turkish administrative population contrast. A category share of the gap may be negative or exceed 100% when other categories offset it. Kendall rank correlation and mean absolute rank shift summarize the complete order, with no claim that categories have equivalent meaning across countries.\n',
      'To quantify changes in prominence even when the top category does not change, an additional hour-scale contrast is E_k = G_k − (Hbar_k / sum_j Hbar_j) × sum_j G_j. This compares the observed category gap with the gap implied by allocating the total gap according to the balanced overall hour composition. E_k sums to zero across categories; positive values identify a category contributing more to the observed gap than its overall hour share. It is a descriptive compositional contrast, not a counterfactual intervention effect. Its paired-bootstrap interval avoids dividing by a potentially small or sign-uncertain net gap. A point-only percentage-point comparison of gap share and hour share is supplied only when the net gap direction is resolved.\n',
      '## Main findings\n',
      'Service-wide and distributional priorities can differ, but the analysis does not support a universal reversal. In Ankara–western Black Sea and Bolu–Sakarya, equipment/electrical conditions account for the largest balanced mean annual-hour component, whereas recorded weather makes the largest contribution to the rural–urban annual-hour gap. These two reversals persist when overall priorities are customer-weighted and when suburban accounts are included. In Bolu–Sakarya, weather represents 9.1% of the balanced mean frequency and 22.9% of mean annual hours, but 56.9% of the rural–urban annual-hour gap. The other two Turkish regional comparisons retain operations/work as the leader under the balanced-group definition; their customer-weighted overall hour leader is instead equipment/electrical conditions. Therefore, the weighting population must accompany any claim about a leading overall cause.\n',
      'Changed prominence also occurs without a changed leader. In Brazil, ordinary unplanned interruptions contribute 71.8% of balanced mean annual hours and 86.8% of the low-versus-high municipal-GDP hour gap. Their gap exceeds the allocation implied by the average-hour composition by 1.52 h (95% municipality-bootstrap interval 0.38–2.95 h). New Zealand weather rises from 18.9% of frequency to 29.1% of mean hours and 36.7% of the income-group gap, but its excess gap contribution relative to the overall hour composition is uncertain. The remaining-cause group stays first in the point estimate; its gap-ranking support is 85.1%.\n',
      'The US comparison separates consequence weighting from evidence of inequality. Major-event categories account for 22.9% of balanced mean frequency but 62.5% of mean annual hours. However, the low-minus-high state-income annual-hour gap remains uncertain, and the major-event category is the gap leader in only 70.6% of bootstrap replicates. Consequently, the strong frequency-to-hour shift does not establish a corresponding economic disparity. Together these comparisons show that occurrence, average lost service and unequal lost service answer different questions; the category connecting them depends on the region and on the population represented.\n',
      '## Detailed results\n']
    for _,r in s.iterrows():
        lines.append(f"### {r['profile']}\n")
        lines.append(f"The frequency leader is **{r.F_leader}**, the annual-hour leader **{r.H_leader}**, and the largest signed social-gap contribution **{r.gap_leader}**. The primary annual-hour gap is {r.total_H_gap:.3f} h/customer-year"+
                     (f" (95% geographical bootstrap interval {r.total_H_gap_lower:.3f}–{r.total_H_gap_upper:.3f})" if pd.notna(r.total_H_gap_lower) else ' (administrative population estimate)')+'.\n')
        cc=c[(c.profile==r['profile'])&c.rank_eligible].sort_values('gap_rank')
        lines.append('| Cause/class | Frequency share (%) | Annual-hour share (%) | Signed hour-gap contribution | Gap rank |\n|---|---:|---:|---:|---:|')
        for _,q in cc.iterrows():
            ci=f" [{q.gap_lower:.3f}, {q.gap_upper:.3f}]" if 'gap_lower' in q and pd.notna(q.gap_lower) else ''
            lines.append(f'| {q.category} | {q.F_share_pct:.2f} | {q.H_share_pct:.2f} | {q.H_gap:.3f}{ci} | {q.gap_rank:.0f} |')
        lines.append('')
        if pd.notna(r.get('H_leader_bootstrap_support')):
            lines.append(f"Bootstrap support for the point leaders is {100*r.F_leader_bootstrap_support:.1f}% (frequency), {100*r.H_leader_bootstrap_support:.1f}% (annual hours) and {100*r.gap_leader_bootstrap_support:.1f}% (signed gap). Annual-hour and gap leaders differ in {100*r.H_gap_bootstrap_leader_changes:.1f}% of resamples. The annual-hour leader versus its point runner-up differs by {r.H_leader_margin:.3f} h [{r.H_leader_margin_lower:.3f}, {r.H_leader_margin_upper:.3f}]; the gap leader versus its runner-up differs by {r.gap_leader_margin:.3f} h [{r.gap_leader_margin_lower:.3f}, {r.gap_leader_margin_upper:.3f}].\n")
        lines.append('Prominence differences on the hour scale (observed class gap minus allocation by average-hour composition): '+ '; '.join(
            f"{q.category}: {q.gap_excess_vs_hours_mix_h:.3f} h"+
            (f" [{q.gap_excess_lower:.3f}, {q.gap_excess_upper:.3f}]" if pd.notna(q.get('gap_excess_lower')) else '')
            for _,q in cc.iterrows())+'.\n')
        ss=w[(w.profile==r['profile'])&(w.metric=='H')].drop_duplicates(['population','overall_weight'])
        lines.append('Alternative overall annual-hour leaders: '+ '; '.join(f'{q.population}, {q.overall_weight}: {q.leader}' for _,q in ss.iterrows())+'.\n')
        residual=c[(c.profile==r['profile'])&~c.rank_eligible]
        if len(residual):
            q=residual.iloc[0]
            lines.append(f"The retained reconciliation residual contributes {q.H_gap:.6f} h to the rural–urban annual-hour gap (rural residual {q.H_low:.6f} h; urban {q.H_high:.6f} h).\n")
    lines.extend(['## Interpretation\n',
      'The new comparison distinguishes system-wide service priorities from distributional priorities without assuming that their leading categories must differ. Agreement of leaders is a substantive result; a point reversal with weak ranking support is not treated as established. The US net economic contrast remains uncertain. The four Turkish regional results provide within-country heterogeneity, not four additional national observations. Category-specific findings should be stated with their source definitions; administrative exceptional-event classes do not measure the share attributable to natural hazards.\n',
      '## Reproduction and provenance\n',
      'Run `python src/analyse_cause_priority.py` from the v37 release. Frozen source paths and SHA-256 checksums are in `analysis/cause_priority/input_manifest.json`. No raw observations, social group definitions, periods or cause crosswalks were changed. `source_data/priority_paired_inputs.csv` records every admitted unit/category and customer denominator; `priority_class_estimates.csv` gives indices, shares, signed gap contributions and intervals; `priority_profile_summary.csv` gives leaders and rank support; `priority_pairwise_ranking.csv` gives all pairwise order changes; `priority_overall_weight_sensitivity.csv` separates alternative overall weighting. `bootstrap_leaders.parquet` preserves every replicate leader and net gap. `validation.json` records pairing, unchanged historical F/H values and exact accounting checks.\n',
      '### Exact input files\n'])
    lines.extend('- `'+m['path']+'`' for m in manifest)
    (OUT/'summary.md').write_text('\n'.join(lines))

if __name__=='__main__':
    run()
