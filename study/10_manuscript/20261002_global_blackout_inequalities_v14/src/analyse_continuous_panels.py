"""Same-population routine/exceptional comparisons for US states and Europe."""
from analyse_zero_inclusive import *

def run_us():
    a=read('07_analysis/economic_hazard_strengthening_20260928/components/usa_panel/data/state_year_panel.parquet')
    a=a[a.panel.eq('stable_2013_2024')].copy()
    a['log_income']=np.log(a.median_household_income_usd);a['income_between']=a.groupby('state').log_income.transform('mean');a['income_within']=a.log_income-a.income_between
    state=a.drop_duplicates('state');delta=float(state.income_between.quantile(.75)-state.income_between.quantile(.25))
    X=pd.DataFrame({'const':1.,'income_between':a.income_between,'income_within':a.income_within},index=a.index)
    X=pd.concat([X,pd.get_dummies(a.year,prefix='year',drop_first=True,dtype=float)],axis=1)
    for cls,code in [('All interruptions','all'),('Non-major days','nonmed'),('Major-event days','med')]:
        paired(a,X,a[f'saifi_{code}'],a[f'saidi_h_{code}'],dict(region='United States',category=cls,specification='49 states; balanced utilities; year effects'),{'income_between':delta,'income_within':np.log(1.1)},groups=[a.state])
    # The 49-state mean is a descriptive cross-system contrast, not US customer total.
    av=a.groupby('state').agg(state_fips=('state_fips','first'),state_name=('state_name','first'),income=('median_household_income_usd','mean'),
        F_routine=('saifi_nonmed','mean'),H_routine=('saidi_h_nonmed','mean'),F_exceptional=('saifi_med','mean'),H_exceptional=('saidi_h_med','mean'),customers=('customers','mean')).reset_index()
    av['D_routine']=av.H_routine/av.F_routine;av['D_exceptional']=av.H_exceptional/av.F_exceptional.replace(0,np.nan);av['exceptional_share']=av.H_exceptional/(av.H_routine+av.H_exceptional)
    av['quartile']=pd.qcut(np.log(av.income),4,labels=['Q1','Q2','Q3','Q4'])
    save(av,'us_same49_state_means');save(a,'us_same49_year_panel')
    per=[]
    for (year,q),z in a.merge(av[['state','quartile']],on='state').groupby(['year','quartile'],observed=True):
        for cls,code in [('Non-major days','nonmed'),('Major-event days','med')]:
            f=z[f'saifi_{code}'].mean();h=z[f'saidi_h_{code}'].mean()
            per.append(dict(year=year,quartile=str(q),category=cls,states=z.state.nunique(),frequency=f,burden_h=h,duration_h=h/f))
    save(pd.DataFrame(per),'us_same49_quartile_year_components')

def run_europe():
    a=read('07_analysis/economic_hazard_strengthening_20260928/components/europe_stress/data/system_year_input.parquet')
    a=a[a.native_scope_consistent].copy()
    # Equal system weighting uses each system's admitted-year mean. The full year
    # panel remains available for the source-defined period and scope audit.
    for scope,z in [('Native matching scope',a),('Documented customer indices',a[a.documented_customer])]:
        av=z.groupby('ISO3').agg(label=('label','first'),years=('year','nunique'),first=('year','min'),last=('year','max'),
            income=('gdp','mean'),F_routine=('routine_saifi','mean'),H_routine=('routine_saidi_h','mean'),
            F_exceptional=('exceptional_saifi','mean'),H_exceptional=('exceptional_saidi_h','mean')).reset_index()
        av['log_income']=np.log(av.income);av['F_all']=av.F_routine+av.F_exceptional;av['H_all']=av.H_routine+av.H_exceptional
        X=pd.DataFrame({'const':1.,'log_income':av.log_income},index=av.index);delta=float(av.log_income.quantile(.75)-av.log_income.quantile(.25))
        for cls,code in [('All interruptions','all'),('Routine','routine'),('Exceptional','exceptional')]:
            paired(av,X,av[f'F_{code}'],av[f'H_{code}'],dict(region='Europe',category=cls,specification=scope),{'log_income':delta})
        av['D_routine']=av.H_routine/av.F_routine;av['D_exceptional']=av.H_exceptional/av.F_exceptional.replace(0,np.nan)
        av['exceptional_share']=av.H_exceptional/av.H_all
        save(av,'europe_'+('same21' if scope.startswith('Native') else 'customer11')+'_system_means')
    save(a,'europe_same21_year_panel')

if __name__=='__main__':
    run_us();run_europe()
    save(pd.DataFrame(rows),'continuous_panel_models');save(pd.DataFrame(audits),'continuous_panel_diagnostics')
    M.joinpath('continuous_panel_manifest.json').write_text(json.dumps(receipts,indent=2))
    r=pd.DataFrame(rows);print(r[r.outcome.isin(['frequency','duration_component'])][['region','category','specification','outcome','term','n','contrast_pct','contrast_lo_pct','contrast_hi_pct']].to_string(index=False))
