"""Questionnaire-verified household F–H–D estimates, separately by country.

See KENYA_ETHIOPIA_ANALYSIS_PLAN.md, specified before group comparisons.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import pyreadstat

P = Path(__file__).resolve().parents[1]
B = 2000
SEED = 20261002
NAMES = ['connection', 'frequency_week', 'hours_week', 'duration_hours', 'frequency_week_all_valid']


def read(path):
    return pyreadstat.read_dta(path, encoding='latin1')[0]


def numeric(s):
    return pd.to_numeric(s, errors='coerce')


def clean(s, unknown=(888,)):
    z = numeric(s)
    return z.where((z >= 0) & ~z.isin(unknown))


def geographic_key(d, cols):
    return d[cols].apply(numeric).astype('Int64').astype(str).agg('_'.join, axis=1)


def load_country(country):
    root = P / 'raw/mtf'
    audit = {}
    if country == 'kenya':
        base = root / 'kenya-multi-tier-framework-mtf-survey/extracted/RAW DATA'
        paths = [base / 'MTF_HH_Core_Survey_Final_Data_trimmed-2.dta', base / 'weight.dta']
        main, weights = [read(f) for f in paths]
        assert main.PARENT_KEY.is_unique and weights.PARENT_KEY.is_unique
        d = main.merge(weights[['PARENT_KEY', 'pw_final']], on='PARENT_KEY', how='left', validate='one_to_one')
        audit['unmatched_weights'] = int(d.pw_final.isna().sum())
        d['w'] = d.pw_final
        d['size'] = d.a_a_1
        food = pd.concat([clean(d[f'l_l_{i}']) for i in range(1, 11)], axis=1)
        d['food_complete'] = food.notna().all(axis=1)
        d['food_pc'] = food.sum(axis=1, min_count=10) / d['size']
        d['connection'] = d.c_c_2.eq(1).astype(float).where(d.c_c_2.isin([1, 2]))
        d['raw_F'], d['raw_hours'], d['raw_minutes'] = d.c_c_28b, d.c_c_29b_hrs, d.c_c_29b_min
        d['settlement'] = d.locality_ur
        # Province names and cluster codes such as '1027_2' are strings.
        # Numeric coercion would merge unrelated clusters, so preserve them.
        assert d[['prov', 'cty', 'cluuq']].notna().all().all()
        d['block'] = d[['prov', 'cty', 'cluuq']].astype(str).agg('|'.join, axis=1)
        d['stratum'] = d.prov.astype(str)
        audit['interview_dates'] = [str(d.subdate.min()), str(d.subdate.max())]
        audit['currency'] = 'Kenyan shillings per person per week'
        audit['raw_cluster_codes_spanning_provinces'] = int((d.groupby('cluuq').prov.nunique() > 1).sum())
    else:
        base = root / 'ethiopia-multi-tier-framework-mtf-survey-2018'
        paths = [base / 'extracted/Identification.dta', base / 'weights_ethiopia.dta',
                 base / 'extracted/Section C.dta', base / 'extracted/Section L_consumption.dta']
        ident, weights, section, foodlong = [read(f) for f in paths]
        for tab in [ident, weights, section]:
            assert tab.HHID.is_unique
        d = ident.merge(weights, on='HHID', how='left', validate='one_to_one')
        d = d.merge(section[['HHID', 'C2', 'C28_tm', 'C29_t_h', 'C29_t_m']], on='HHID', how='left', validate='one_to_one')
        foodlong = foodlong[foodlong.Consumption_item.between(1, 10)].copy()
        assert not foodlong.duplicated(['HHID', 'Consumption_item']).any()
        # Ethiopia labels negative 888, rather than positive 888, as nonresponse.
        foodlong['value'] = clean(foodlong.L1, unknown=())
        food = foodlong.pivot(index='HHID', columns='Consumption_item', values='value').reindex(columns=range(1, 11))
        agg = pd.DataFrame({'food_complete': food.notna().all(axis=1), 'food_total': food.sum(axis=1, min_count=10)})
        d = d.merge(agg, on='HHID', how='left', validate='one_to_one')
        d['w'], d['size'] = d.hh_weights, d.A0
        d['food_pc'] = d.food_total / d['size']
        d['connection'] = d.C2.eq(1).astype(float).where(d.C2.isin([1, 2]))
        d['raw_F'], d['raw_hours'], d['raw_minutes'] = d.C28_tm, d.C29_t_h, d.C29_t_m
        d['settlement'] = d.HI_6
        d['block'] = geographic_key(d, ['HI_1', 'HI_2', 'HI_3', 'HI_4', 'HI_5'])
        d['stratum'] = d.HI_1.astype(str)
        audit['unmatched_weights'] = int(d.w.isna().sum())
        audit['interview_year'] = 2017
        audit['interview_date_note'] = 'Year verified from country report; raw day/month strings retained in source.'
        audit['currency'] = 'Ethiopian birr per person per week'
    audit['source_files'] = [{'path': str(f.relative_to(P)), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in paths]
    audit['raw_n'] = len(d)
    audit['positive_weight_n'] = int(d.w.gt(0).sum())
    audit['complete_food_n'] = int((d.food_complete & d.w.gt(0)).sum())
    audit['food_complete_weight_fraction'] = float(d.loc[d.food_complete & d.w.gt(0), 'w'].sum() / d.loc[d.w.gt(0), 'w'].sum())
    return d, audit


def metrics(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.stack([x[..., 1] / x[..., 0], x[..., 3] / x[..., 2], x[..., 4] / x[..., 2],
                         x[..., 4] / x[..., 3], x[..., 6] / x[..., 5]], axis=-1)


def estimate(d, duration_col, country, variant):
    d = d.copy()
    d['H'] = d[duration_col]
    d['paired'] = d.F.notna() & d.H.notna()
    info = d.groupby('block').stratum.first()
    assert d.groupby('block').stratum.nunique().max() == 1
    d['bi'] = pd.Categorical(d.block, categories=info.index).codes
    d['cell'] = ((d.quintile - 1) * 2 + d.settlement - 1).astype(int)
    arr = np.zeros((len(info), 10, 7))
    w = d.w.to_numpy(); pair = d.paired.to_numpy()
    vals = np.column_stack([w*d.connection.notna(), w*d.connection.fillna(0), w*pair,
                           w*d.F.fillna(0)*pair, w*d.H.fillna(0)*pair, w*d.F.notna(), w*d.F.fillna(0)])
    for j in range(7):
        np.add.at(arr[:, :, j], (d.bi, d.cell), vals[:, j])
    totals = arr.sum(axis=0).reshape(5, 2, 7)
    point = metrics(totals.sum(axis=1))
    rng = np.random.default_rng(SEED)
    rep = np.zeros((B, len(info)), dtype=np.int16)
    for s in info.unique():
        ix = np.flatnonzero(info.to_numpy() == s)
        sampled = rng.choice(ix, (B, len(ix)), replace=True)
        for i in range(B):
            np.add.at(rep[i], sampled[i], 1)
    draws = np.einsum('bk,kcj->bcj', rep, arr).reshape(B, 5, 2, 7)
    bm = metrics(draws.sum(axis=2))
    rows = []
    for q in range(5):
        sub = d[d.quintile == q+1]
        row = dict(country=country, variant=variant, quintile=q+1, n=len(sub),
                   n_connected=int(sub.connection.eq(1).sum()), n_paired=int(sub.paired.sum()),
                   n_frequency_valid=int(sub.F.notna().sum()), blocks=int(sub.block.nunique()),
                   weight_share=float(sub.w.sum()/d.w.sum()), food_min=float(sub.food_pc.min()), food_max=float(sub.food_pc.max()))
        for j, name in enumerate(NAMES):
            row[name] = float(point[q, j])
            row[name+'_lo'], row[name+'_hi'] = np.nanquantile(bm[:, q, j], [.025, .975]).tolist()
        rows.append(row)
    tab = pd.DataFrame(rows)
    contrasts = {}
    for j, name in enumerate(NAMES):
        ratios = bm[:, 0, j] / bm[:, 4, j]
        diffs = bm[:, 0, j] - bm[:, 4, j]
        contrasts[name] = dict(ratio=float(point[0,j]/point[4,j]), ratio_lo=float(np.nanquantile(ratios,.025)),
                               ratio_hi=float(np.nanquantile(ratios,.975)), difference=float(point[0,j]-point[4,j]),
                               difference_lo=float(np.nanquantile(diffs,.025)), difference_hi=float(np.nanquantile(diffs,.975)))
    cells = d[d.paired].groupby(['quintile','settlement']).size().unstack(fill_value=0).reindex(index=range(1,6), columns=[1,2], fill_value=0)
    standard = {'admitted': bool(cells.loc[[1,5]].min().min()>=30), 'min_extreme_cell_n': int(cells.loc[[1,5]].min().min())}
    pm = metrics(totals)
    settlement = []
    for q in range(5):
        for s in range(2):
            row = dict(country=country,variant=variant,quintile=q+1,settlement=['Urban','Rural'][s],n_paired=int(cells.loc[q+1,s+1]))
            row.update(dict(zip(NAMES, pm[q,s].tolist())))
            settlement.append(row)
    if standard['admitted']:
        uw = float(d.loc[(d.connection==1)&(d.settlement==1),'w'].sum()/d.loc[d.connection==1,'w'].sum())
        point_std = uw*pm[:,0] + (1-uw)*pm[:,1]
        point_std[:,3] = point_std[:,2] / point_std[:,1]
        sb = metrics(draws)
        sb = uw*sb[:,:,0] + (1-uw)*sb[:,:,1]
        sb[:,:,3] = sb[:,:,2] / sb[:,:,1]
        standard['urban_weight'] = uw
        for j,name in enumerate(NAMES[1:4],1):
            rr = sb[:,0,j]/sb[:,4,j]
            standard[name] = dict(ratio=float(point_std[0,j]/point_std[4,j]),lo=float(np.nanquantile(rr,.025)),hi=float(np.nanquantile(rr,.975)))
    summary = dict(country=country,variant=variant,eligible_n=len(d),paired_n=int(d.paired.sum()),
                   contrasts=contrasts,settlement_standardization=standard,bootstrap_replicates=B,seed=SEED,
                   blocks=len(info),bootstrap_strata=int(info.nunique()),singleton_strata=int((info.value_counts()==1).sum()),
                   identity_error_max=float(np.abs(point[:,2]-point[:,1]*point[:,3]).max()))
    stem = f'{country}_{variant}'
    tab.to_csv(P/f'data/{stem}_quintiles.csv',index=False)
    pd.DataFrame(settlement).to_csv(P/f'data/{stem}_settlement.csv',index=False)
    (P/f'data/{stem}_summary.json').write_text(json.dumps(summary,indent=2))
    return tab, summary


def main(country):
    raw, audit = load_country(country)
    d = raw[(raw.w>0)&(raw['size']>0)&raw.food_pc.notna()&raw.settlement.isin([1,2])].copy()
    z = d.sort_values('food_pc'); cum = z.w.cumsum()/z.w.sum()
    cuts = [float(z.iloc[np.searchsorted(cum,p)].food_pc) for p in [.2,.4,.6,.8]]
    assert len(set(cuts))==4
    d['quintile'] = np.searchsorted(cuts,d.food_pc,side='left')+1
    d['F'] = clean(d.raw_F).where(d.connection==1)
    h = clean(d.raw_hours).where(d.connection==1)
    m = clean(d.raw_minutes).where(d.connection==1)
    m = m.where(m.between(0,59))
    d['H_complete'] = h+m/60
    structural = (d.F==0)&d.raw_hours.isna()&d.raw_minutes.isna()
    d.loc[structural,'H_complete'] = 0
    known = (h.notna() | m.notna()) & (h.notna() | d.raw_hours.isna()) & (m.notna() | d.raw_minutes.isna())
    d['H_component_zero'] = (h.fillna(0)+m.fillna(0)/60).where(known)
    d.loc[structural,'H_component_zero'] = 0
    for v in ['H_complete','H_component_zero']:
        bad=(d[v]>168)|((d.F==0)&(d[v]>0))
        audit[v+'_inconsistent'] = int(bad.sum())
        d.loc[bad,v] = np.nan
        audit[v+'_positiveF_missingH'] = int(((d.F>0)&d[v].isna()).sum())
        audit[v+'_positiveF_zeroH'] = int(((d.F>0)&(d[v]==0)).sum())
    audit.update(eligible_n=len(d),cutpoints_weekly_food_value_per_person=cuts,structural_zero_n=int(structural.sum()),
                 frequency_valid_n=int(d.F.notna().sum()),duration_definition='Total interrupted hours plus minutes per typical week; D=H/F',
                 ranking='Complete first ten food-consumption categories per household member; no official poverty designation')
    summaries={}
    for variant,col in [('complete','H_complete'),('component_zero','H_component_zero')]:
        tab,summary=estimate(d,col,country,variant)
        summaries[variant]=summary
        print(country,variant)
        print(tab[['quintile','n','n_connected','n_paired','connection','frequency_week','hours_week','duration_hours']].to_string(index=False))
        print(json.dumps(summary,indent=2))
    (P/f'metadata/{country}_analysis_audit.json').write_text(json.dumps(audit,indent=2))
    d[['w','size','food_pc','quintile','connection','F','H_complete','H_component_zero','settlement','block','stratum']].to_parquet(P/f'data/{country}_analysis_sample.parquet',index=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('country',choices=['kenya','ethiopia']);args=ap.parse_args();main(args.country)
