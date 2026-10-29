"""Read-only source audit; writes only this v40 research directory."""
from pathlib import Path
import pandas as pd, numpy as np, json, hashlib
ROOT=next(p for p in Path(__file__).resolve().parents if p.name == 'study')
OUT=ROOT/'10_manuscript/20261009_global_blackout_inequalities_v40/research/europe_americas'
B=ROOT/'08_database_expansion/20261005_fig3_cause_expansion'
V=ROOT/'10_manuscript/20261003_global_blackout_inequalities_v33'
CATS=['Weather','Other environment','Equipment / electrical','Human / operation','Supply constraint','Planned','Other specified','Other / unresolved']
old=pd.read_csv(V/'source_data/cause_atlas_v33_expanded_paired_shares.csv')
branch=pd.read_csv(B/'processed/fig3_expanded_paired_shares.csv')
summary=pd.read_csv(B/'processed/fig3_expanded_profile_summary.csv').set_index('profile_id')
inputs={}; native=[]; frames=[]; meta=[]
def read(p):
 p=Path(p);inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def emit(pid,q,oc,bu,method,locator,period=None,notes='',url=None):
 s=summary.loc[pid];q=q.copy();q['profile_id']=pid
 q['source_address']=q['source_address'] if 'source_address' in q else (url or s.source_address)
 q['source_locator']=q['source_locator'] if 'source_locator' in q else locator
 q['period']=period or s.period;q['occurrence_unit']=s.occurrence_unit;q['burden_unit']=s.burden_unit
 q['mapping_confidence']=q.get('mapping_confidence','high: source-native label retained; broad class is explicit documented crosswalk')
 q['mapping_boundary']=q.get('mapping_boundary',notes)
 q['native_occurrence_value']=q[oc];q['native_burden_value']=q[bu]
 native.append(q)
 agg=q.groupby('cause_group')[[oc,bu]].sum().reindex(CATS)
 total0=q[oc].sum();total1=q[bu].sum()
 frame=pd.DataFrame({'profile_id':pid,'country_iso3':s.country_iso3,'system':s.display_name,'period':period or s.period,'cause_group':CATS,'occurrence_unit':s.occurrence_unit,'burden_unit':s.burden_unit,'profile_method':method,'source_address':url or s.source_address,'source_locator':locator,'source_notes':notes})
 frame['category_reported']=frame.cause_group.isin(q.cause_group)
 frame['occurrence_value']=agg[oc].fillna(0).to_numpy();frame['burden_value']=agg[bu].fillna(0).to_numpy()
 frame['occurrence_total']=total0;frame['burden_total']=total1
 frame['occurrence_share_pct']=100*frame.occurrence_value/total0;frame['burden_share_pct']=100*frame.burden_value/total1
 frame['occurrence_share_normalized_pct']=frame.occurrence_share_pct;frame['burden_share_normalized_pct']=frame.burden_share_pct
 frame['share_gap_pp']=frame.burden_share_pct-frame.occurrence_share_pct;frame['normalized_difference_pp']=frame.share_gap_pp
 frame['native_categories']=frame.cause_group.map(q.groupby('cause_group').native_label.agg(lambda x:' | '.join(sorted(set(x.fillna('[missing native text]').astype(str))))))
 frames.append(frame)
 meta.append(dict(profile_id=pid,occurrence_total=total0,burden_total=total1,occurrence_unit=s.occurrence_unit,burden_unit=s.burden_unit,aggregation=method,period=period or s.period,source_native_rows=len(q),source_address=url or s.source_address,locator=locator,scope=s.scope,notes=notes))
# UK: exact archived strict primary-cause cohort; retain no-fault and transient statuses unresolved.
u=read(B/'uk_ssen/processed/ssen_2016_2025_native_cause_totals.csv').rename(columns={'cause_native':'native_label'})
u['old_cause_group']=u.cause_group
u['source_address']=' | '.join(read(B/'uk_ssen/source_manifest.csv').source_address.dropna().tolist())
specified=['Mechanical Shock or Vibration','Fire not due to Faults','Load Current above Previous Assessment','Failure of infeed from Adjacent DNO','Operational or Safety Restriction','Failure of Supply from Generating Company or the National Grid Company','Local PESC Generation Failure (Isolated System)','PESC Equipment Affected by National Grid Company Personnel or Equipment','PESC Equipment affected by Private Generator or Authorised Electricity Operator (other than National Grid Company)']
u.loc[u.native_label.isin(specified),'cause_group']='Other specified'
assert u.records.sum()==267861 and np.isclose(u.customer_hours.sum(),47529128.38333333)
emit('GBR_SSEN2',u,'records','customer_hours','pooled same 267861 source rows; strict primary CAUSE/CAUSE_DESCRIPTION only','Four frozen20260922 SSEN CSVs: CAUSE/CAUSE_DESCRIPTION, CAUSE_CODE, customer minutes lost; methodology p2','2016–2025 (incident calendar years)','Other specified is a recorded origin/event/condition with unresolved physical initiator. Transient/no-fault/unknown/unclassified/missing remain Other / unresolved. Primary cause differs from old any-weather primary/contributor screen; cohort and customer-hours identical.')
# New Zealand: reuse source-verified three extra COMPLETE causes; do not assign incomplete cells to zero.
n=read(B/'audit/nzl_same21_paired_shares_candidate.csv');n=n[n.category_reported].copy();n['native_label']=n.cause_group.map({'Weather':'Adverse weather | Lightning','Human / operation':'Human error | Third party interference','Other environment':'Vegetation | Wildlife','Equipment / electrical':'Defective equipment','Other / unresolved':'Incomplete Adverse environment / Cause unknown / Other cause plus signed source-total closure'})
n['old_cause_group']=n.cause_group
emit('NZL_QUALIFIED21',n,'occurrence_value','burden_value','shares of equal-network three-year mean indices; same21networks /63network-years','ComCom full CSV; network All; section10(ii) Class C Interruptions and Duration by Cause; category Cause; SAIFI /SAIDI','2023–2025','Only three newly mapped causes have all63paired network-year cells: Defective equipment, Vegetation, Wildlife. Missing cause labels not imputed. Native cause table has583rows, all source-verified; incomplete labels remain in residual. Small signed rounding closure retained.',url='https://www.comcom.govt.nz/assets/file/0034/366298/EDB_ID_Full_2026-06-01-csv.gz')
# Nordic national transmission profiles; percentages are equal-year means then close rounding to100.
a=read(ROOT/'07_analysis/hazard_cause_country_panel_20260929/data/nordic_paired_annual_2018_2021.csv')
nmap={'External influence':'Other specified','Lightning':'Weather','Operation and maintenance':'Human / operation','Other':'Other / unresolved','Other environmental':'Other environment','Technical equipment':'Equipment / electrical','Unknown':'Other / unresolved'}
for iso,g in a.groupby('ISO3'):
 q=g.groupby('cause')[['disturbance_share','ENS_share']].mean().reset_index().rename(columns={'cause':'native_label'});q['cause_group']=q.native_label.map(nmap);q['old_cause_group']=q.cause_group.where(q.native_label.ne('External influence'),'Other / unresolved')
 assert q.cause_group.notna().all()
 emit('NORDIC_'+iso,q,'disturbance_share','ENS_share','equal-year mean published cause percentages; normalize native rounded sum to100','Annual Nordic/Baltic2018–2021 cause-share tables;2021 Table1.1 printedp4 (PDFp15); Table3.4 PDFp23 disturbance; Table4.5 PDFp28 ENS','2018–2021','External influence includes human and animal/fire categories; retained as Other specified. Other/Unknown stay unresolved. Other environmental is native mixed environmental class (may include weather); no hidden weather fraction inferred.',url='https://eepublicdownloads.entsoe.eu/clean-documents/SOC%20documents/Nordic/2022/2021_Nordic_and_Baltic_Grid_Disturbance_Statistics_FOR_PUBLISHING.pdf')
# NERC sustained cause counts/hours, preserving exactly the same circuit/period denominator.
r=read(V/'reports/us_cause_inputs_v33/NERC_North_America_paired_sustaining_cause_profile_2021_2025.csv').rename(columns={'cause':'native_label'})
m={'Contamination':'Other environment','Environmental':'Other environment','Fire':'Other specified','ForeignInterference':'Other specified','HumanError':'Human / operation','Lightning':'Weather','Other':'Other / unresolved','PowerSystemCondition':'Other specified','Unknown':'Other / unresolved','Vandalism,Terrorism,orMaliciousActs':'Human / operation','Vegetation':'Other environment','Weather,excludinglightning':'Weather'}
r['cause_group']=r.native_label.map(m);r.loc[r.native_label.str.startswith('Failed'),'cause_group']='Equipment / electrical';assert r.cause_group.notna().all();r['old_cause_group']=r.cause_group.replace({'Other specified':'Other / unresolved'})
emit('NERC_NORTH_AMERICA',r,'sustaining_count','sustaining_hours','pooled2021–2025 same sustainedAC200plus causes for counts and circuit hours','TADS Dashboard Supporting Data; Sustaining Cause category; definitions2026DRI AppendixA','2021–2025','Fire, ForeignInterference and PowerSystemCondition have meaningful native definitions but no single broad physical initiator. Other specified retains them; Other and Unknown remain unresolved. Not a national US customer-loss profile.')
# Additional 20261005 branch profiles: source-native cells and complete matched denominators.
e=read(B/'other_countries/europe/processed/europe_native_cause_observations.csv');e=e[e.primary_selected & e.country_iso3.isin(['CHE','PRT','DEU'])].copy();e=e.rename(columns={'native_cause':'native_label'});e['old_cause_group']=e.cause_group
changes={'Naturereignisse','Fremdeinwirkung','Incêndios','ungeplant: Einwirkung Dritter','ungeplant: Höhere Gewalt','ungeplant: Rückwirkstörung'}
e.loc[e.native_label.isin(changes),'cause_group']='Other specified'
e['source_locator']=e.apply(lambda x:('PDF p'+str(x.source_pdf_page)) if pd.notna(x.source_pdf_page) else ('XLSX '+str(x.source_sheet)),axis=1)
for iso,pid in [('CHE','CHE_ELCOM91_2024'),('PRT','PRT_REN_2023_2024'),('DEU','DEU_BNETZA_LV_2024')]:
 q=e[e.country_iso3.eq(iso)].copy()
 emit(pid,q,'native_occurrence','native_loss',str(summary.loc[pid,'aggregation']),'; '.join(q.source_locator.unique()),notes='Same admitted20261005branch source population. Other specified records mixed native event/origin/condition; physical initiator not inferred. Genericother and German network-responsibility (includes no discernible cause) remain unresolved.')
# Australia: third-party origin is retained, administrative excluded/network business classes remain unresolved.
z=read(B/'other_countries/australia/processed/aer_native_cause_indices_2009_2024.csv');z=z[z.period.eq('2023–24')].rename(columns={'cause_native':'native_label'});z['old_cause_group']=z.cause_group;z.loc[z.native_label.eq('Third party'),'cause_group']='Other specified';z['source_locator']=z.source_saifi+'; '+z.source_saidi
emit('AUS_AER_NEM_CAUSES_FY2023_2024',z,'saifi','saidi_minutes','same13 NEMnetworks FY2023–24 cause-specific SAIFI/SAIDI components; explicitplanned; exacttotal closure','AER StateoftheEnergyMarket2025 Networks XLSX: Figures3.34/3.35 plus planned3.32/3.33','FY2023–2024','Other specified retains third-party origin. Networkbusiness/STPISexcluded are administrative classes with unresolved physical cause, kept unresolved; do not infer weather from exclusion status. Not allAustralia.')
res=pd.concat(frames,ignore_index=True);assert np.allclose(res.groupby('profile_id')[['occurrence_share_pct','burden_share_pct']].sum(),100,atol=1e-10)
res.to_csv(OUT/'revised_paired_shares.csv',index=False)
pd.concat(native,ignore_index=True).to_csv(OUT/'native_mapping_audit.csv',index=False)
mt=pd.DataFrame(meta);mt.to_csv(OUT/'revised_profile_metadata.csv',index=False)
checks=[]
for pid,g in res.groupby('profile_id'):
 rec={'profile_id':pid,'after_other_unresolved_occurrence_pct':g.loc[g.cause_group.eq('Other / unresolved'),'occurrence_share_pct'].sum(),'after_other_unresolved_burden_pct':g.loc[g.cause_group.eq('Other / unresolved'),'burden_share_pct'].sum(),'after_other_specified_occurrence_pct':g.loc[g.cause_group.eq('Other specified'),'occurrence_share_pct'].sum(),'after_other_specified_burden_pct':g.loc[g.cause_group.eq('Other specified'),'burden_share_pct'].sum(),'after_leading_occurrence':g.loc[g.occurrence_share_pct.idxmax(),'cause_group'],'after_leading_burden':g.loc[g.burden_share_pct.idxmax(),'cause_group'],'after_composition_shift_pct':abs(g.share_gap_pp).sum()/2}
 for label,tab in [('v33',old),('oct05branch',branch)]:
  x=tab[tab.profile_id.eq(pid)];o=x[x.cause_group.eq('Other / unresolved')];rec[label+'_other_unresolved_occurrence_pct']=o.occurrence_share_pct.sum() if len(o) else np.nan;rec[label+'_other_unresolved_burden_pct']=o.burden_share_pct.sum() if len(o) else np.nan
 checks.append(rec)
pd.DataFrame(checks).to_csv(OUT/'before_after_unresolved.csv',index=False)
(OUT/'validation.json').write_text(json.dumps({'profiles':res.profile_id.nunique(),'rows':len(res),'allshares_sum100':True,'zeros_mean_no_native_allocation_not_absence_of_cause':True,'native_labels_preserved':True,'input_sha256':inputs},indent=2))
print(pd.DataFrame(checks).round(4).to_string(index=False))
