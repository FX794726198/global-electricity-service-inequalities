"""Household service inequality and direct connection barriers, source-backed panels."""
from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parents[1];R=P.parents[1]
V=R/'10_manuscript/20261002_global_blackout_inequalities_v14'
sys.path.insert(0,str(V/'src'))
import figure_style as fs
fs.P=P
from figure_style import *
for s in ['figures','metadata']:(P/s).mkdir(exist_ok=True)
BASE=R/'07_analysis/household_blackout_inequality_20261002'
W=gpd.read_parquet(R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
W=W[W.ISO3.ne('ATA')]
gap=pd.read_csv(P/'data/conditional_grid_contrasts.csv')
states=pd.read_csv(P/'data/country_state_contrasts.csv').query('admitted_parent')
bc=pd.read_csv(P/'data/barriers_all_contrasts.csv')
SHORT={'São Tomé and Príncipe':'São Tomé & Príncipe'}
f=page(210)
cm=LinearSegmentedColormap.from_list('reliability_gap',['#f9f1ef','#d899a1','#a82d43'])
norm=plt.Normalize(0,40)
a=ax(f,12,10,81,82)
z=W.merge(gap[['iso3','service_gap']],left_on='ISO3',right_on='iso3',how='left',validate='many_to_one')
z['gap_pp']=z.service_gap*100
z.plot(ax=a,color='#e7e9e7',edgecolor='#b7bec0',lw=.35,aspect='auto')
z[z.gap_pp.notna()].plot(column='gap_pp',ax=a,cmap=cm,norm=norm,edgecolor='white',lw=.38,aspect='auto')
for iso,x,y in [('CPV',-23.7,15.1),('STP',6.6,.3),('SYC',55.5,-4.7)]:
    v=float(gap.loc[gap.iso3.eq(iso),'service_gap'].iloc[0])*100
    a.scatter(x,y,s=25,facecolor=cm(norm(v)),edgecolor=INK,lw=.55,zorder=4)
a.set(xlim=(-28,60),ylim=(-37,39));a.set_aspect('auto')
geographic_frame(a,4326,longitudes=[0,30,60],latitudes=[-30,0,30],fontsize=9.2)
title(f,a,'a','Inequality where the grid reaches',offset=9)
cb(f,cm,norm,[18,100,67,2.5],[0,10,20,30,40],['0','10','20','30','40'],
   'Reliable-service gap (percentage points)')

b=ax(f,139,10,65,82)
order=states.sort_values('service_gap',ascending=False).reset_index(drop=True)
keys=['unconnected_limited_grid_gap','unconnected_grid_present_gap','connected_unreliable_gap','unconnected_grid_unknown_gap']
cols=[BLUE,GOLD,RED,GREY]
for i,row in order.iterrows():
    pos=neg=0.
    for key,col in zip(keys,cols):
        val=100*row[key];left=pos if val>=0 else neg
        b.barh(i,val,left=left,color=col,height=.71,alpha=.92)
        if val>=0:pos+=val
        else:neg+=val
    b.scatter(100*row.service_gap,i,s=8,color=INK,zorder=4)
b.set_yticks(range(len(order)),[SHORT.get(x,x) for x in order.country],fontsize=9)
b.tick_params(axis='y',length=0);b.set(xlim=(-20,62),ylim=(len(order)-.45,-.65))
b.set_xticks([-20,0,20,40,60]);b.set_xlabel('Reliable-service gap (pp)',fontsize=9.8)
b.axvline(0,color=GREY,lw=.65);clean(b)
title(f,b,'b','Where service gaps occur',offset=40)
legend(f,[Patch(facecolor=c) for c in cols],
       ['Limited local grid','Grid present, unconnected','Connected, unreliable','Local grid unknown'],
       158,111,2,8.7)

c=ax(f,29,135,65,50)
countries=['Cambodia','Ethiopia','Kenya']
for i,country in enumerate(countries):
    dat=bc[bc.country.eq(country.lower())].set_index('state')
    values=[dat.loc[k,'difference_pp'] for k in ['distance','cost','administrative']]
    values.append(dat.loc[['landlord','satisfied','unreliable','other','unknown'],'difference_pp'].sum())
    pos=neg=0.
    for v,col in zip(values,[BLUE,GOLD,TEAL,GREY]):
        left=pos if v>=0 else neg
        c.barh(i,v,left=left,height=.55,color=col,alpha=.92)
        if v>=9:c.text(left+v/2,i,f'{v:.0f}',ha='center',va='center',fontsize=9,color='white' if col in [BLUE,TEAL] else INK)
        if v>=0:pos+=v
        else:neg+=v
    rr=dat.loc['connected'];net=-rr.difference_pp
    c.errorbar(net,i-.36,xerr=[[net+rr.hi_pp],[-rr.lo_pp-net]],fmt='D',color=INK,mec='white',mew=.5,ms=4.4,lw=.8,capsize=2,zorder=5)
c.axvline(0,color=GREY,lw=.65);c.set(xlim=(-2,66),ylim=(2.65,-.6),xlabel='Connection gap (percentage points)')
c.set_yticks(range(3),countries,fontsize=9.5);c.set_xticks([0,20,40,60]);clean(c)
title(f,c,'c','Reported barriers to connection',offset=24)
legend(f,[Patch(facecolor=x) for x in [BLUE,GOLD,TEAL,GREY]],
       ['Distance / unavailable','Cost','Application / administration','Other / unknown'],56,202,2,8.7)

d=ax(f,133,135,71,50)
for i,country in enumerate(countries):
    fn='cambodia_summary.json' if country=='Cambodia' else country.lower()+'_complete_summary.json'
    s=json.loads((BASE/'data'/fn).read_text())
    if i%2==0:d.axhspan(i-.43,i+.43,color='#f0f2f2',zorder=0)
    for j,(key,col) in enumerate([('frequency_week',BLUE),('duration_hours',TEAL),('hours_week',RED)]):
        row=s['contrasts'][key];y=i+(j-1)*.22
        d.plot([row['ratio_lo'],row['ratio_hi']],[y,y],color=col,lw=1.3)
        d.scatter(row['ratio'],y,s=27,color=col,edgecolor='white',lw=.5,zorder=3)
d.axvline(1,color=GREY,lw=.8,ls='--');d.set(xscale='log',xlim=(.125,16),ylim=(2.6,-.6),xlabel='Lowest / highest quintile')
d.set_yticks(range(3),countries);d.tick_params(axis='y',length=0);logticks(d,'x',[.25,1,4,16]);clean(d)
title(f,d,'d','Interruption after connection',offset=28)
legend(f,[Line2D([],[],marker='o',color=x,lw=1.3,ms=4) for x in [BLUE,TEAL,RED]],
       ['Frequency','Duration','Hours'],162,202,3,9.0)
save(f,'Fig04_household_service_inequality')

# Full quintile profiles, without a sign-based selection of reported barriers.
q=pd.read_csv(P/'data/barriers_all_quintiles.csv')
f=page(140)
barcols=[BLUE,GOLD,TEAL,GREY]
for i,country in enumerate(countries):
    a=ax(f,14+i*66,12,56,40)
    z=q[q.country.eq(country.lower())]
    for state,col in zip(['distance','cost','administrative'],barcols):
        zz=z[z.state.eq(state)].sort_values('quintile')
        a.fill_between(zz.quintile,zz.lo*100,zz.hi*100,color=col,alpha=.12,lw=0)
        a.plot(zz.quintile,zz.prevalence*100,'o-',color=col,ms=3.4,lw=1.2)
    others=z[z.state.isin(['landlord','satisfied','unreliable','other','unknown'])].groupby('quintile').prevalence.sum()
    a.plot(others.index,others*100,'o-',color=GREY,ms=3.2,lw=1.1)
    a.set(xlim=(.7,5.3),ylim=(0,75),xlabel='Living-standard quintile')
    a.set_xticks(range(1,6),['Q1','Q2','Q3','Q4','Q5']);a.set_yticks([0,25,50,75]);clean(a)
    if i==0:a.set_ylabel('All households (%)')
    title(f,a,chr(97+i),country,offset=10)
    a=ax(f,14+i*66,81,56,40)
    sums=[]
    for state in ['distance','cost','administrative']:
        sums.append(z[z.state.eq(state)].sort_values('quintile').share_among_unconnected.to_numpy()*100)
    sums.append(z[z.state.isin(['landlord','satisfied','unreliable','other','unknown'])].groupby('quintile').share_among_unconnected.sum().to_numpy()*100)
    bottom=np.zeros(5)
    for vals,col in zip(sums,barcols):
        a.bar(np.arange(1,6),vals,bottom=bottom,color=col,width=.8,alpha=.92);bottom+=vals
    assert np.allclose(bottom,100)
    a.set(xlim=(.5,5.5),ylim=(0,100),xlabel='Living-standard quintile')
    a.set_xticks(range(1,6),['Q1','Q2','Q3','Q4','Q5']);a.set_yticks([0,25,50,75,100]);clean(a)
    if i==0:a.set_ylabel('Unconnected households (%)')
    title(f,a,chr(100+i),'',offset=0)
legend(f,[Patch(facecolor=x) for x in barcols],['Distance / unavailable','Cost','Application / administration','Other / unknown'],109,72,4,9)
save(f,'SuppFig16_connection_barriers')
inputs=[P/'data/conditional_grid_contrasts.csv',P/'data/country_state_contrasts.csv',P/'data/barriers_all_contrasts.csv',P/'data/barriers_all_quintiles.csv']
inputs += [BASE/'data'/x for x in ['cambodia_summary.json','ethiopia_complete_summary.json','kenya_complete_summary.json']]
inputs += [R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet',V/'src/figure_style.py',V/'src/map_coordinates.py']
(P/'metadata/figure_sources.json').write_text(json.dumps([{'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in inputs],indent=2))
