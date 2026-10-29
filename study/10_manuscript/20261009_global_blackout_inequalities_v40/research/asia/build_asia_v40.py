"""Conservative native-cause recovery and newly acquired official Asian profiles.

Run with the bundled Python runtime. Never modifies historical sources or documents.
"""
from pathlib import Path
import re, json, hashlib
import pandas as pd
from pypdf import PdfReader

OUT=Path(__file__).resolve().parent
V40=OUT.parents[1]
ROOT=V40.parents[1]
RAW=V40/'raw_sources/asia'
V33=ROOT/'10_manuscript/20261003_global_blackout_inequalities_v33'
GROUPS=['Weather','Other environment','Equipment / electrical','Human / operation','Supply constraint','Planned','Other specified','Other / unresolved']
profiles=[]; native=[]; checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def add_profile(pid,iso,system,period,df,count,burden,cu,bu,url,locator,note,status='paired'):
    totalc=float(df[count].sum());totalb=float(df[burden].sum()) if burden else None
    for g in GROUPS:
        sub=df[df.cause_group.eq(g)]
        c=float(sub[count].sum());b=float(sub[burden].sum()) if burden else None
        profiles.append(dict(profile_id=pid,country_iso3=iso,system=system,period=period,cause_group=g,native_occurrence=c,native_burden=b,occurrence_share_pct=c/totalc*100,burden_share_pct=b/totalb*100 if totalb else None,category_reported=len(sub)>0,occurrence_unit=cu,burden_unit=bu,occurrence_denominator=totalc,burden_denominator=totalb,paired_status=status,source_url=url,source_locator=locator,note=note))
    for _,x in df.iterrows():
        native.append(dict(profile_id=pid,country_iso3=iso,system=system,period=period,native_cause=x.cause_native,cause_group=x.cause_group,occurrence_value=x[count],burden_value=x[burden] if burden else None,occurrence_unit=cu,burden_unit=bu,occurrence_denominator=totalc,burden_denominator=totalb,paired_status=status,source_url=url,source_locator=x.get('source_locator',locator),note=note))
    checks.append(dict(profile_id=pid,occurrence_total=totalc,burden_total=totalb,native_rows=len(df),status=status))

def classify_bgd(row):
    s=str(row.cause_native).lower()
    if row.eligible_hazard_paired:return 'Weather','retained_v33_explicit_weather'
    if re.search(r'(bird|monkey|tree|nest).{0,30}(remov|clean|cut)|breaking bird',s):return 'Human / operation','specified_biotic_removal_work'
    # Require direct contact/interference phrases; never infer from an asset name.
    if re.search(r'(tree|branch).{0,35}(fall|fell|touch|contact)|(?:fall|fell).{0,25}(tree|branch)|bird.{0,15}(hit|contact|nest)|snake|monkey',s):return 'Other environment','explicit_environmental_contact'
    if re.search(r'(crane|vehicle|truck).{0,40}(touch|hit|damage)|(?:touch|hit|damage).{0,40}(crane|vehicle|truck)|theft|stolen|sabotage',s):return 'Human / operation','explicit_external_interference'
    # Physical damage/condition is recorded; no unrecorded initiating hazard is assigned.
    equipment=r'\b(ct|pt|cvt|cb|gcb|ds|la)\b|circuit.breaker|transformer|conductor|connector|insulator|bushing|cable|busbar|isolator|arrester|jumper|terminal'
    damage=r'burst|blast|burnt|burned|broken|break.down|torn|ruptur|punctur|oil.leak|hydraulic.{0,15}(fail|problem)|red.?hot|sparking|overheat|faulty|defect|damag'
    if re.search(equipment,s) and re.search(damage,s):return 'Equipment / electrical','explicit_component_damage_or_condition'
    # Maintenance/repair describes a known action, not necessarily the initiating cause.
    if re.search(r'maint|repair|replac|cleaning|oil.top|oil.fill|testing|test work|erection|re.route|re.routing|shifting|re.string|installation|shutdown for|shut down for',s):return 'Human / operation','specified_work_in_unscheduled_cohort'
    if re.search(r'load.management|control overload',s):return 'Human / operation','explicit_load_management_action'
    # An indicated protection operation alone does not resolve the initiating condition.
    if re.search(r'relay|protection|prot\b|protn|pront|tripp|\btrip\b',s):return 'Other / unresolved','trip_protection_indicator_only'
    if re.search(r'transient.fault',s):return 'Other / unresolved','transient_without_specific_initiator'
    if re.search(r'over.?voltage|under.?voltage|over.?current|under.?frequency|over.?load|short.?circuit|feeder.fault|bus.?bar.fault|bus.fault|cable.fault',s):return 'Equipment / electrical','specified_electrical_condition_without_protection_only_signal'
    return 'Other / unresolved','unknown_or_trip_protection_indicator_only'

bgdpath=ROOT/'07_analysis/cause_loss_reaudit_20260930/data/bangladesh_asset_outage_records_v2.csv'
b=pd.read_csv(bgdpath,low_memory=False)
b=b[b.eligible_paired & b.report_month.str[:4].isin(['2024','2025'])].copy()
assert len(b)==5687
b[['cause_group','mapping_rule']]=pd.DataFrame([classify_bgd(x) for _,x in b.iterrows()],index=b.index)
# October-5 reviewed classifications are the baseline. Only improve its unresolved
# records using explicit newly checked native descriptions; no conservative regression.
prior=pd.read_csv(ROOT/'08_database_expansion/20261005_fig3_cause_expansion/processed/bangladesh_native_records.csv',low_memory=False)
assert set(prior.record_id)==set(b.record_id)
prior=prior.set_index('record_id')
for idx,row in b.iterrows():
    old=prior.loc[row.record_id]
    if old.cause_group!='Other / unresolved':
        b.loc[idx,['cause_group','mapping_rule']]=[old.cause_group,'retained_oct5: '+old.mapping_rule]
b.to_csv(OUT/'bgd_2024_2025_recoded_records.csv',index=False)
comparison=b[['record_id','cause_native','cause_group','mapping_rule','reported_asset_outage_hours']].merge(prior.reset_index()[['record_id','cause_group','mapping_rule']],on='record_id',suffixes=('_v40','_oct5'),validate='1:1')
comparison.to_csv(OUT/'bgd_oct5_vs_v40_mapping_comparison.csv',index=False)
bn=b.groupby(['cause_native','cause_group','mapping_rule'],dropna=False).agg(count=('record_id','size'),hours=('reported_asset_outage_hours','sum'),ens_mwh=('source_reported_unserved_energy_mwh','sum'),source_locator=('record_id',lambda x:';'.join(x))).reset_index()
bn.to_csv(OUT/'bgd_native_mapping_v40.csv',index=False)
add_profile('BGD_PGCB_2024_2025','BGD','Bangladesh transmission ledger','2024–2025',bn,'count','hours','qualified equipment records','asset-hours','https://erp.powergrid.gov.bd/web/','Native record IDs resolve to workbook/sheet/row in bgd_2024_2025_recoded_records.csv','Identical v33 paired cohort; explicit initiating weather retained. Component damage and named actions recovered without inventing initiating causes. Relay operation/trip alone remains unresolved.')

# Reproduce new BRPL PDF extraction, preserving source-designated planned status.
url='https://www.derc.gov.in/sites/default/files/Website_Report_December_2025.pdf'
r=PdfReader(RAW/'ind_brpl_dec2025.pdf');records=[];fail=[]
start=re.compile(r'(?m)^\s*(\d+)\s+(\d{2}-\d{2}-\d{4})\s*')
pat=re.compile(r'^(.*?)\s*(\d+)\s+(.*?)\s+(Unplanned Outage|Planned Outage)\s+(\d{2}-\d{2}-\d{4})\s+(\d{1,2}:\d{2})\s+(\d{2}-\d{2}-\d{4})\s+(\d{1,2}:\d{2})\s+(\d+:\d{2})\s+([0-9.]+)\s+(.*)$',re.S)
for page,pg in enumerate(r.pages,1):
    s=pg.extract_text() or ''
    # PDF p9 joins the last two table cells into BOLT259; visually checked source row.
    if page==9:s=s.replace('BOLT259 05-12-2025','BOLT\n259 05-12-2025')
    hits=list(start.finditer(s))
    for k,h in enumerate(hits):
        chunk=s[h.end():hits[k+1].start() if k+1<len(hits) else len(s)]
        chunk=re.split(r'\n(?:From To|Name of Distribution Licensee:|Date Time Date Time)',chunk)[0].strip();m=pat.match(chunk)
        if not m:fail.append((page,h[1],chunk));continue
        loc,customers,equipment,planned,sd,st,ed,et,dur,ens,cause=m.groups();hh,mm=map(int,dur.split(':'))
        records.append(dict(source_page=page,serial=int(h[1]),date=h[2],location=' '.join(loc.split()),customers=int(customers),equipment=equipment,planned=planned=='Planned Outage',start=sd+' '+st,end=ed+' '+et,duration_h=hh+mm/60,ens_mwh=float(ens)*1000,cause_native=' '.join(cause.split())))
d=pd.DataFrame(records)
assert not fail
assert d.serial.nunique()==len(d)==2530
assert set(range(1,2531))==set(d.serial)
d['elapsed_h']=(pd.to_datetime(d.end,dayfirst=True)-pd.to_datetime(d.start,dayfirst=True)).dt.total_seconds()/3600
assert (abs(d.duration_h-d.elapsed_h)<1e-9).all()
def brpl(row):
    s=row.cause_native
    if row.planned:return 'Planned'
    if any(x in s for x in ['OUTSIDE AGENCY','UNKNOWN VEHICLE']):return 'Human / operation'
    if any(x in s for x in ['TREE TOUCHING','BRANCH FALLEN','BIRDAGE','MONKEY']):return 'Other environment'
    if 'TRANSIENT FAULT' in s:return 'Other / unresolved'
    if any(x in s for x in ['CABLE FAULT','TRANSFORMER FAULTY','FAULT IN','HOT SPOT','DAMAGED WIRE','SYSTEM FAULT','HVDS FAULT']):return 'Equipment / electrical'
    # Following the reviewed Bangladesh baseline: explicit work is operational
    # activity, not operator error, planned status, or proof of initiating mechanism.
    if any(x in s for x in ['REPAIR','REPLAC','REFILL','REFILLING','LAYING','CONNECTION','CONNETION','TRIMMING','MAINTENANCE','CHECK RMU','TERMINATION','RELAY CHANGE']):return 'Human / operation'
    return 'Other specified'
d['cause_group']=d.apply(brpl,axis=1);d.to_csv(OUT/'ind_brpl_dec2025_records.csv',index=False)
sub=d[~d.planned].copy();assert len(sub)==265
dn=sub.groupby(['cause_native','cause_group']).agg(count=('serial','size'),hours=('duration_h','sum'),ens_mwh=('ens_mwh','sum'),source_locator=('source_page',lambda x:'PDF pages '+','.join(map(str,sorted(set(x)))))).reset_index()
dn.to_csv(OUT/'ind_brpl_native_mapping_v40.csv',index=False)
note='All 265 source-designated unplanned rows, December 2025 BRPL service population; separate from TPDDL. Count and record-hours use identical rows. Explicit repair/maintenance is Human / operation as recorded activity, not operator error or inferred initiation; source planned flag is retained. MU converted to MWh ×1000. Not independent incidents or customer-hours.'
add_profile('IND_BRPL_UNPLANNED_2025_12','IND','Delhi BRPL service area','2025-12',dn,'count','hours','interruption table rows','sum of record-duration hours',url,'PDF pp1–85; native serials and page in record CSV',note)
add_profile('IND_BRPL_UNPLANNED_2025_12_ENS','IND','Delhi BRPL service area','2025-12',dn,'count','ens_mwh','interruption table rows','reported MWh not supplied',url,'PDF pp1–85; native serials and page in record CSV',note+' Alternative burden; do not treat as independent sample.')

# Existing primary profiles: all eight groups emitted, no missing denominator hidden.
old=pd.read_csv(V33/'reports/asia_cause_profiles_v33.csv')
cw=pd.read_csv(V33/'reports/asia_native_cause_crosswalk_v33.csv')
for pid in ['BTN_TRANSMISSION_2023','PHL_CAGAYAN_DE_ORO_2015_2022']:
    q=old[old.profile_id.eq(pid)].iloc[0];n=cw[cw.profile_id.eq(pid)].copy();n['cause_group']=n.broad_cause
    if pid=='BTN_TRANSMISSION_2023':n.loc[n.cause_native.eq('Transient Faults'),'cause_group']='Other / unresolved'
    add_profile(pid,q.iso3,q.geography,q.period,n,'count_value','loss_value',q.count_unit,q.loss_unit,q.source_url,q.source_path,q.scope+' Transient faults without a specific initiator remain Other / unresolved under the shared cross-source rule.' if pid=='BTN_TRANSMISSION_2023' else q.scope)

# Use the October-5 annual TPDDL base, not the stale v33 two-month population.
OCT5=ROOT/'08_database_expansion/20261005_fig3_cause_expansion'
n=pd.read_csv(OCT5/'india/processed/tpddl_2024_native_cause_totals.csv')
n=n[n.administrative_outage_type_native.eq('Unplanned Outage')].copy()
n['cause_group']=n.cause_native.map({'EODB COMPLIANCE':'Other specified','FORCE MAJEURE/EXTERNAL FACTORS':'Other / unresolved','INTERRUPTION DUE TO FAULT':'Equipment / electrical'})
assert n.cause_group.notna().all()
add_profile('IND_TPDDL_2024_UNPLANNED_ROWS','IND','Delhi TPDDL service area','2024',n,'records','record_hours','interruption table rows','sum of record-duration hours','https://www.derc.gov.in/tpddl','October-5 package; twelve monthly reports in india/raw and source_manifest.csv','Annual 24,096-row same population. EODB compliance is specified administrative/work context; generic fault is specified condition. Neither resolves initiating cause; external factors remain unresolved.')

j=pd.read_csv(OCT5/'processed/japan_tohoku_native_crosswalk.csv')
j.loc[j.cause_native.str.contains('火災',na=False),'cause_group']='Other specified'
add_profile('JPN_TOHOKU_RESTORED_GROUPS_20261005','JPN','Tohoku restored locality groups','2026-09-05–2026-10-05',j,'records','hours','published restored locality groups','published group interval-hours','https://nw.tohoku-epco.co.jp/teideninfo/','October-5 native archive and source-record URLs','Frozen 31-day snapshot, 153 restored locality groups; no national annual inference. Explicit nearby fire is Other specified because initiation is not given.')

# A new, explicitly limited distribution external-factor population. Rounded chart shares only.
q=pd.DataFrame([('Weather','Weather',52,51),('Third Party','Human / operation',16,14),('Grid/Source Failure','Supply constraint',6,4),('Natural Calamities','Weather',4,2),('Animals','Other environment',4,2),('RoW (trees from outside corridor)','Other environment',18,27)],columns=['cause_native','cause_group','count_share','duration_share'])
add_profile('BTN_DISTRIBUTION_EXTERNAL_2023','BTN','19 ESD distribution external-factor outages only','2023',q,'count_share','duration_share','reported occurrence percentage','reported duration percentage','https://era.gov.bt/wp-content/uploads/2024/08/ERA-Report-on-Power-Outage.pdf','PDF p12 (printed p10), Figure 6','Within external-factor class only: source denominators 1,587 interruptions and 4,143 h 49 min. Rounded integer percentages are retained, not converted to fabricated exact counts/hours. Not a replacement for transmission external factors.','paired_rounded_shares_restricted_scope')

q=pd.DataFrame([('Animals','Other environment',14),('Malfunctions of substation equipment','Equipment / electrical',13),('Human errors','Human / operation',7),('Unidentifiable reasons','Other / unresolved',3),('Outsiders','Human / operation',2),('Other causes','Other / unresolved',6)],columns=['cause_native','cause_group','count'])
add_profile('THA_EGAT_FORCED_2024','THA','EGAT transmission system','2024',q,'count',None,'forced outages','not cause-resolved','https://www.egat.co.th/home/en/wp-content/uploads/2025/06/EGAT-AR_EN_2024_250612.pdf','PDF p18; printed pp34–35, Transmission System','45 forced outages; 3 line and 42 substation. No cause-specific duration/energy; annual SAIDI and one highlighted incident cannot be assigned to these causes.','occurrence_only')

q=pd.DataFrame([('Faulty equipment','Equipment / electrical',16.92),('Transient','Other / unresolved',37.70),('Damaged by third party','Human / operation',6.38),('Animal','Other environment',10.06),('Natural Disaster','Weather',19.48),('Others','Other / unresolved',9.46)],columns=['cause_native','cause_group','share'])
add_profile('MYS_SARAWAK_CAUSES_2020','MYS','Sarawak distribution','2020',q,'share',None,'reported occurrence percentage','not cause-resolved','https://meih.st.gov.my/documents/10620/bb01e5b6-181d-4cda-9438-72403a2a656f','PDF p49; printed p47, Chart 32','Published two-decimal cause percentages sum to 100. No cause-specific duration; do not invent counts from rounded shares. Chart 31 reports 3,804 unscheduled interruptions, but Chart 32 does not explicitly repeat that denominator.','occurrence_only_rounded_shares')

# Official Malaysian exhaustive annual interruption-type tables, inspected as rendered pages.
q=pd.DataFrame([('Transformer','Equipment / electrical',17),('Auto Reclose','Other specified',0),('Flood','Weather',2),('Feeder pillar','Equipment / electrical',2404),('Fuse','Equipment / electrical',2390),('Animal','Other environment',756),('Insulating Piercing Connectors (IPC)','Equipment / electrical',17441),('Jumper','Equipment / electrical',1),('Cable','Equipment / electrical',4037),('Conductor','Equipment / electrical',13520),('Fuse Box','Equipment / electrical',8383),('Others','Other / unresolved',13),('Null','Other / unresolved',236),('Link','Equipment / electrical',33),('Termination','Equipment / electrical',206),('Insulator','Equipment / electrical',1),('Switchgear','Equipment / electrical',10),('Third Party','Human / operation',2741),('Tree','Other environment',5612),('Relay','Equipment / electrical',1),('Storm','Weather',78),('Joint','Equipment / electrical',4209),('Pole','Equipment / electrical',1511),('Transient','Other / unresolved',6),('Vandalism','Human / operation',193)],columns=['cause_native','cause_group','count'])
assert q['count'].sum()==63801
add_profile('MYS_PENINSULAR_UNSCHEDULED_2020','MYS','Peninsular Malaysia distribution excluding high voltage','2020',q,'count',None,'unscheduled interruptions excluding high voltage','not cause-resolved','https://meih.st.gov.my/documents/10620/bb01e5b6-181d-4cda-9438-72403a2a656f','PDF p30; printed p28, Table 5, 2020 column','All 25 interruption-type entries sum to the explicit 63,801 table total. High-voltage interruptions excluded: broader Table 4 totals 63,809. Component labels describe reported interruption types or locations; no initiating hazard is imputed. No cause-specific duration.','occurrence_only')
q=pd.DataFrame([('Flood','Weather',0),('Overload','Equipment / electrical',1918),('Animal','Other environment',2372),('Poor Contact','Equipment / electrical',2383),('Bad Weather (Wind, Storm, Lightning)','Weather',2028),('Caused by Other Parties (Hit, Treachery, Theft and Illegal Connection)','Human / operation',2052),('Cable','Equipment / electrical',962),('Fire','Other specified',58),('Touched with Guard Wire/Sagging','Equipment / electrical',0),('Faulty Equipment','Equipment / electrical',0),('Quality of Material','Equipment / electrical',311),('Others (Unavailable Data, Shut Down, No Supply)','Other / unresolved',1333),('Old/Decayed','Equipment / electrical',1214),('Substation','Equipment / electrical',0),('Tree','Other environment',4044),('Landslide','Other environment',0),('Unknown','Other / unresolved',0),('Transient','Other / unresolved',0)],columns=['cause_native','cause_group','count'])
assert q['count'].sum()==18675
add_profile('MYS_SABAH_UNSCHEDULED_2020','MYS','Sabah distribution','2020',q,'count',None,'unscheduled interruptions','not cause-resolved','https://meih.st.gov.my/documents/10620/bb01e5b6-181d-4cda-9438-72403a2a656f','PDF p44; printed p42, Table 17, 2020 column','All 18 interruption-type entries sum to the explicit 18,675 table total. Mixed Others category remains unresolved; fire origin is unspecified. No cause-specific duration.','occurrence_only')

pd.DataFrame(profiles).to_csv(OUT/'asia_profiles_v40_full_shares.csv',index=False)
pd.DataFrame(native).to_csv(OUT/'asia_qualified_native_profiles_v40.csv',index=False)
(OUT/'asia_profile_checks_v40.json').write_text(json.dumps(checks,indent=2))
for pid,g in pd.DataFrame(profiles).groupby('profile_id'):
    assert abs(g.occurrence_share_pct.sum()-100)<1e-8
    if g.burden_share_pct.notna().any():assert abs(g.burden_share_pct.sum()-100)<1e-8
print(pd.DataFrame(profiles)[['profile_id','cause_group','native_occurrence','native_burden','occurrence_share_pct','burden_share_pct']].to_string(index=False))
