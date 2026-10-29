"""Independent planned/unplanned density comparison, fixed before coefficient inspection."""
from analyse_zero_inclusive import *
import analyse_process_contrasts as cp

src='07_analysis/non_europe_cause_loss_extension_20260929/raw/nzl_reliability_from_existing_full.parquet'
a=read(src)
for col in a.select_dtypes('object'):a[col]=a[col].str.strip()
context=pd.read_parquet(D/'nz_explicit21_total_context.parquet').set_index('system')
labels={'Class B (planned interruptions on the network)':'Planned',
        'Class C (unplanned interruptions on the network)':'Unplanned'}
s=a[a.network.eq('All')&a.edb.isin(context.index)&a.disc_yr.between(2023,2025)&
    a.section.eq('10(i): Interruptions')&a.category.eq('SAIFI and SAIDI by class')&a.description.isin(labels)].copy()
assert not s.duplicated(['edb','disc_yr','description','sub_category']).any()
s['interruption_class']=s.description.map(labels)
v=s.pivot(index=['edb','disc_yr','interruption_class'],columns='sub_category',values='value').reset_index()
assert len(v)==21*3*2 and v[['SAIFI','SAIDI']].notna().all().all()
v=v.rename(columns={'edb':'system','disc_yr':'year','SAIFI':'frequency'});v['burden_h']=v.SAIDI/60
save(v,'nz_explicit21_planned_unplanned_year')
m=v.groupby(['system','interruption_class'])[['frequency','burden_h']].mean().reset_index().merge(context.drop(columns=['frequency','burden_h']),on='system',validate='many_to_one')
save(m,'nz_explicit21_planned_unplanned_context')
pv=m.pivot(index='system',columns='interruption_class',values=['frequency','burden_h']).loc[context.index]
ys={'Fp':pv['frequency','Planned'],'Hp':pv['burden_h','Planned'],
    'Fu':pv['frequency','Unplanned'],'Hu':pv['burden_h','Unplanned']}
direct_c=pd.read_parquet(D/'nz_explicit21_total_context.parquet').set_index('system').loc[context.index]
assert np.allclose(ys['Fu'],direct_c.frequency) and np.allclose(ys['Hu'],direct_c.burden_h)
X=pd.DataFrame({'const':1.,'log_income':context.log_income,'log_density':context.log_density},index=context.index)
defs={'Planned frequency':{'Fp':1},'Unplanned frequency':{'Fu':1},
      'Planned duration':{'Hp':1,'Fp':-1},'Unplanned duration':{'Hu':1,'Fu':-1},
      'Planned minus unplanned frequency':{'Fp':1,'Fu':-1},
      'Planned minus unplanned duration':{'Hp':1,'Fp':-1,'Hu':-1,'Fu':1}}
delta={k:float(context[k].quantile(.75)-context[k].quantile(.25)) for k in ['log_income','log_density']}
cp.contrasts(context,X,ys,{k:defs for k in delta},dict(region='New Zealand',specification='Planned/unplanned: economic context and density'),delta)
save(pd.DataFrame(cp.out),'nz_planned_unplanned_direct_contrasts')
M.joinpath('nz_planned_unplanned_source.json').write_text(json.dumps(dict(inputs=receipts,operators=21,years=[2023,2024,2025],year_class_pairs=len(v),
    definition_source='Commerce Commission 2015 consolidated determination, definitions p.10: Class B network-initiated planned interruptions; Class C unplanned interruptions originating on network',
    decision='Independent extension; population fixed from primary cause comparison before coefficient inspection'),indent=2))
print(pd.DataFrame(cp.out)[['term','contrast','contrast_pct','contrast_lo_pct','contrast_hi_pct','p']].to_string(index=False))
