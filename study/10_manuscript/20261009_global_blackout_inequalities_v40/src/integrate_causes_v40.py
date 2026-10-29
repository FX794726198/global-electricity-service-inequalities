from pathlib import Path
import pandas as pd
import numpy as np
import json

P=Path(__file__).resolve().parents[1]
ROOT=P.parents[1]
OLD=ROOT/'08_database_expansion/20261005_fig3_cause_expansion/processed'
S=P/'source_data'
cats=['Weather','Other environment','Equipment / electrical','Human / operation','Supply constraint','Planned','Other specified','Other / unresolved']
s=pd.read_csv(OLD/'fig3_expanded_profile_summary.csv').set_index('profile_id')
q=pd.read_csv(OLD/'fig3_expanded_paired_shares.csv')
old28=pd.read_csv(S/'profiles_original.csv').set_index('profile_id')
for pid in s.index.intersection(old28.index):
    for col in s.columns.intersection(old28.columns):
        if pd.isna(s.loc[pid,col]) and pd.notna(old28.loc[pid,col]):s.loc[pid,col]=old28.loc[pid,col]
s['paired']=True
s['scope_flag']='paired source population'
eu=pd.read_csv(P/'research/europe_americas/revised_paired_shares.csv')
q=pd.concat([q[~q.profile_id.isin(eu.profile_id)],eu],ignore_index=True)
m=pd.read_csv(P/'research/europe_americas/revised_profile_metadata.csv').set_index('profile_id')
for pid,r in m.iterrows():
    for col,val in r.items():
        if pd.notna(val):s.loc[pid,col]=val
a=pd.read_csv(P/'research/asia/asia_profiles_v40_full_shares.csv')
exclude=['IND_BRPL_UNPLANNED_2025_12_ENS','BTN_DISTRIBUTION_EXTERNAL_2023']
a=a[~a.profile_id.isin(exclude)].copy()
a['source_address']=a.source_url
q=pd.concat([q[~q.profile_id.isin(a.profile_id)],a],ignore_index=True)
positions={'IND_BRPL_UNPLANNED_2025_12':(77.10,28.50,'India · Delhi south/west†'),
 'THA_EGAT_FORCED_2024':(101,15,'Thailand · transmission*'),
 'MYS_PENINSULAR_UNSCHEDULED_2020':(102,4,'Malaysia · Peninsula*'),
 'MYS_SARAWAK_CAUSES_2020':(112,2,'Malaysia · Sarawak*'),
 'MYS_SABAH_UNSCHEDULED_2020':(116.5,5.5,'Malaysia · Sabah*')}
for pid,g in a.groupby('profile_id',sort=False):
    r=g.iloc[0]
    if pid not in s.index:
        lon,lat,name=positions[pid]
        s.loc[pid,'display_name']=name;s.loc[pid,'longitude']=lon;s.loc[pid,'latitude']=lat
    for col in ['country_iso3','period','occurrence_unit','burden_unit','source_address','system','source_locator']:
        s.loc[pid,col]=r[col]
    s.loc[pid,'occurrence_total']=r.occurrence_denominator;s.loc[pid,'burden_total']=r.burden_denominator
    s.loc[pid,'paired']=r.paired_status=='paired'
    s.loc[pid,'scope_flag']=r.paired_status
    s.loc[pid,'source_notes']=r.note
    if pid in positions:
        s.loc[pid,'scope']=r.system;s.loc[pid,'aggregation']='Native cause totals divided by the same disclosed population total'
        s.loc[pid,'geometry_role']='Source reporting-context locator; no national coverage inferred'
s.loc['GBR_SSEN2','occurrence_unit']='published incident records'
s.loc['GBR_SSEN2','burden_unit']='customer-hours interrupted'
s.loc['GBR_SSEN2','source_address']='https://data.ssen.co.uk/'
s.loc['JPN_TOHOKU_RESTORED_GROUPS_20261005','display_name']='Japan · Tohoku snapshot†'
s.loc['IND_TPDDL_2024_UNPLANNED_ROWS','display_name']='India · Delhi north/west'
s.loc['CHE_ELCOM91_2024','display_name']='Switzerland · distribution'
s.loc['GBR_SSEN2','display_name']='UK · two service regions'
s.loc['AUS_AER_NEM_CAUSES_FY2023_2024','display_name']='Australia · eastern networks'
# Preserve absent classifications as unreported, rather than assigning an observed zero.
q=q.set_index(['profile_id','cause_group']).reindex(pd.MultiIndex.from_product([s.index,cats],names=['profile_id','cause_group'])).reset_index()
q['category_reported']=q.category_reported.fillna(False).map(lambda z:str(z).lower()=='true')
for col in ['occurrence_share_pct','burden_share_pct']:q[col]=pd.to_numeric(q[col],errors='coerce')
q.loc[~q.profile_id.map(s.paired).astype(bool),'burden_share_pct']=np.nan
q['occurrence_share_normalized_pct']=100*q.occurrence_share_pct/q.groupby('profile_id').occurrence_share_pct.transform('sum')
q['burden_share_normalized_pct']=100*q.burden_share_pct/q.groupby('profile_id').burden_share_pct.transform('sum').replace(0,np.nan)
q['share_gap_pp']=q.burden_share_pct-q.occurrence_share_pct
q['normalized_difference_pp']=q.burden_share_normalized_pct-q.occurrence_share_normalized_pct
stats=[]
for pid,g in q.groupby('profile_id',sort=False):
    assert np.isclose(g.occurrence_share_pct.sum(),100,atol=.03),(pid,g.occurrence_share_pct.sum())
    pair=bool(s.loc[pid,'paired'])
    lo=g.loc[g.occurrence_share_pct.idxmax(),'cause_group']
    if pair:
        assert np.isclose(g.burden_share_pct.sum(),100,atol=.03),(pid,g.burden_share_pct.sum())
        lb=g.loc[g.burden_share_pct.idxmax(),'cause_group'];shift=float((g.burden_share_normalized_pct.fillna(0)-g.occurrence_share_normalized_pct.fillna(0)).abs().sum()/2)
    else:lb=None;shift=None
    for k,v in {'leading_occurrence':lo,'leading_burden':lb,'leader_changed':pair and lo!=lb,'composition_shift_pct':shift}.items():s.loc[pid,k]=v
    residual=g[g.cause_group.eq('Other / unresolved')].iloc[0]
    s.loc[pid,'other_unresolved_occurrence_share_pct']=residual.occurrence_share_pct
    s.loc[pid,'other_unresolved_burden_share_pct']=residual.burden_share_pct
    kind='unpaired' if not pair else ('unchanged' if lo==lb else ('residual_involved' if {lo,lb}&{'Other specified','Other / unresolved'} else 'substantive_to_substantive'))
    stats.append({'profile_id':pid,'paired':pair,'leading_occurrence':lo,'leading_consequence':lb,'changed':pair and lo!=lb,'change_kind':kind,'composition_shift_pp':shift})
st=pd.DataFrame(stats);st.to_csv(P/'analysis/fig2_leader_audit.csv',index=False)
s.to_csv(S/'fig2_profiles.csv');q.to_csv(S/'fig2_shares.csv',index=False)
short=['IND_BRPL_UNPLANNED_2025_12','JPN_TOHOKU_RESTORED_GROUPS_20261005']
summary={'profiles_total':len(s),'paired_profiles':int(s.paired.sum()),'occurrence_only_profiles':int((~s.paired.astype(bool)).sum()),'countries':int(s[s.country_iso3.ne('MULTI')].country_iso3.nunique()),'paired_countries':int(s[s.paired.astype(bool)&s.country_iso3.ne('MULTI')].country_iso3.nunique()),'leader_changes':int(st.changed.sum()),'change_kinds':st.change_kind.value_counts().to_dict(),'paired_excluding_short_snapshots':int(st[~st.profile_id.isin(short)].paired.sum()),'changes_excluding_short_snapshots':int(st[~st.profile_id.isin(short)].changed.sum()),'retained_base_countries':30,'new_main_countries':['Thailand','Malaysia'],'restored_countries':['Australia','Japan','Germany','Switzerland','Portugal','Italy','Peru']}
(P/'analysis/fig2_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
