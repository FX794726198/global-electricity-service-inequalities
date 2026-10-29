"""Global-to-group and cause-to-burden composites; all panels have declared populations."""
from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parents[1];R=P.parents[1]
sys.path.insert(0,str(R/'10_manuscript/20261002_global_blackout_inequalities_v14/src'))
import figure_style as fs
fs.P=P
from figure_style import *
W=gpd.read_parquet(R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
W=W[W.ISO3.ne('ATA')].copy()

def base(a):
 W.plot(ax=a,color='#E8EAE9',edgecolor='#BBC2C4',lw=.27,aspect='auto')
 a.set(xlim=(-180,180),ylim=(-58,82));a.set_aspect('auto')
 geographic_frame(a,4326,longitudes=[-120,-60,0,60,120],latitudes=[-30,0,30,60],fontsize=9.5)

# The global panel measures access; both household panels use exactly the same 23-country population.
raw=json.loads((R/'01_raw/global/worldbank/EG.ELC.ACCS.ZS.json').read_text())[1]
e=pd.DataFrame([{'ISO3':x['countryiso3code'],'country':x['country']['value'],'year':int(x['date']),'access_pct':x['value']} for x in raw if x['date']=='2023' and x['value'] is not None])
e=e[e.ISO3.isin(W.ISO3)].copy();e['without_access_pct']=100-e.access_pct
assert not e.ISO3.duplicated().any() and e.without_access_pct.between(0,100).all()
g=pd.read_csv(R/'07_analysis/grid_reach_inequality_20261002/data/conditional_grid_contrasts.csv')
g['unreliable_connected_gap']=(g.connection_high-g.service_high)-(g.connection_no-g.service_no)
assert np.allclose(g.service_gap,g.connection_gap+g.unreliable_connected_gap)
assert len(g)==23 and g.service_gap.gt(0).all()
e.to_csv(P/'source_data/Fig03_global_access_2023.csv',index=False)
g.to_csv(P/'source_data/Fig03_same_population_service_components.csv',index=False)
# Equal-country resampling preserves the two correlated service components.
rng=np.random.default_rng(20261002);v=g[['connection_gap','unreliable_connected_gap','service_gap']].to_numpy()
boot=v[rng.integers(0,len(v),size=(4000,len(v)))].mean(axis=1)*100
res={k:{'estimate_pp':float(v[:,j].mean()*100),'lower_pp':float(np.quantile(boot[:,j],.025)),'upper_pp':float(np.quantile(boot[:,j],.975))} for j,k in enumerate(['connection','connected_unreliable','net'])}
(P/'metadata/household_same_population_components.json').write_text(json.dumps({'country_count':len(g),'bootstrap':'4000 whole-country paired draws; seed 20261002; not full household survey design','results':res,'global_access_mapped':len(e)},indent=2))
f=page(206)
cm=LinearSegmentedColormap.from_list('access',['#F3F5F0','#CBDCC7','#7FB5A9','#397E88','#154B63']);norm=BoundaryNorm([0,1,5,10,25,50,75,100],cm.N)
a=ax(f,13,11,188,62);base(a)
z=W.merge(e,on='ISO3',how='left',validate='many_to_one');z[z.access_pct.notna()].plot(column='without_access_pct',ax=a,cmap=cm,norm=norm,edgecolor='#ADB7B9',lw=.24,aspect='auto')
a.set(xlim=(-180,180),ylim=(-58,82));a.set_aspect('auto')
a.add_patch(Rectangle((-28,-37),88,76,fill=False,edgecolor=RED,lw=.75,zorder=5))
title(f,a,'a','Electricity access leaves a global service divide · 2023',offset=9)
cb(f,cm,norm,[48,80,122,2.5],[0,1,5,10,25,50,75,100],['0','1','5','10','25','50','75','100'],'Population without electricity access (%)')
cm2=LinearSegmentedColormap.from_list('gap',['#FBEDE5','#E2A184','#BE5360','#782B4B']);norm2=plt.Normalize(0,40)
b=ax(f,13,109,78,75);W.plot(ax=b,color='#E8EAE9',edgecolor='#BBC2C4',lw=.27,aspect='auto')
z=W.merge(g,left_on='ISO3',right_on='iso3',how='left',validate='many_to_one');z['gap_pp']=z.service_gap*100
z[z.gap_pp.notna()].plot(column='gap_pp',ax=b,cmap=cm2,norm=norm2,edgecolor='white',lw=.4,aspect='auto')
for iso,x,y in [('CPV',-23.7,15.1),('STP',6.6,.3),('SYC',55.5,-4.7)]:
 b.scatter(x,y,s=27,facecolor=cm2(norm2(g.loc[g.iso3.eq(iso),'service_gap'].iloc[0]*100)),edgecolor=INK,lw=.5,zorder=4)
b.set(xlim=(-28,60),ylim=(-37,39));b.set_aspect('auto')
geographic_frame(b,4326,longitudes=[0,30,60],latitudes=[-30,0,30],fontsize=9.5)
title(f,b,'b','Gaps where the grid is accessible',offset=9)
cb(f,cm2,norm2,[18,192,65,2.5],[0,10,20,30,40],['0','10','20','30','40'],'Reliable-service gap (pp)')
c=ax(f,139,109,63,75);order=g.sort_values('service_gap',ascending=False).reset_index(drop=True)
for i,r in order.iterrows():
 pos=neg=0.
 for key,col in [('connection_gap',BLUE),('unreliable_connected_gap',RED)]:
  val=r[key]*100;c.barh(i,val,left=pos if val>=0 else neg,height=.73,color=col)
  if val>=0:pos+=val
  else:neg+=val
 c.scatter(r.service_gap*100,i,s=11,color=INK,zorder=4,edgecolor='white',lw=.3)
c.set_yticks(range(len(order)),order.country.replace({'São Tomé and Príncipe':'São Tomé & Príncipe'}),fontsize=9.1)
c.tick_params(axis='y',length=0);c.set(xlim=(-17,43),ylim=(len(order)-.4,-.6),xlabel='Contribution to service gap (pp)');c.set_xticks([-10,0,10,20,30,40]);c.axvline(0,color=GREY,lw=.6);clean(c)
title(f,c,'c','Connection and continuity · same groups',offset=40)
legend(f,[Patch(color=BLUE),Patch(color=RED),Line2D([],[],marker='o',lw=0,color=INK,ms=3)],['Unconnected','Connected, unreliable','Net gap'],159,197,2,9.1)
save(f,'Fig03_access_and_reliable_service')

# Each country profile is compared internally, in its own native occurrence/burden units.
V=R/'10_manuscript/20261002_global_blackout_inequalities_v15/data'
d=pd.read_parquet(V/'global_cause_atlas.parquet');pr=pd.read_parquet(V/'global_cause_atlas_profiles.parquet').set_index('profile_id')
CAUSES=['Weather','Other environment','Human / operation','Equipment / electrical','Supply constraint','Planned','Other / unresolved']
order=['SEN_SENELEC','RWA_REG','JAM_JPS','BRA_AGGREGATE689','USA_NY_QUALIFIED','NZL_QUALIFIED21','FIN_DISTRIBUTION','NORDIC_DNK','NORDIC_EST','NORDIC_FIN','NORDIC_ISL','NORDIC_LVA','NORDIC_LTU','NORDIC_NOR','NORDIC_SWE']
labels=['Senegal · 2018–23','Rwanda · 2019/20–22/23','Jamaica · 2018–21','Brazil · 2021','New York · 2021–24','New Zealand · 2023–25','Finland, distribution · 2010–19','Denmark · 2018–21','Estonia · 2018–21','Finland, transmission · 2018–21','Iceland · 2018–21','Latvia · 2018–21','Lithuania · 2018–21','Norway · 2018–21','Sweden · 2018–21']
rows=[]
for pid in order:
 z=d[d.profile_id.eq(pid)]
 assert abs(z.occurrence_share_pct.sum()-100)<=.75 and abs(z.burden_share_pct.sum()-100)<=.75
 delta=z.burden_share_pct/z.burden_share_pct.sum()*100-z.occurrence_share_pct/z.occurrence_share_pct.sum()*100
 rows.append({'profile_id':pid,'ISO3':pr.loc[pid,'country_iso3'],'displacement_pct':abs(delta).sum()/2,'leader_changes':pr.loc[pid,'leading_occurrence']!=pr.loc[pid,'leading_burden'],'burden_unit':z.burden_unit.iloc[0]})
tv=pd.DataFrame(rows).set_index('profile_id');tv.to_csv(P/'source_data/Fig04_profile_displacement.csv')
d.to_csv(P/'source_data/Fig04_cause_share_changes.csv',index=False)
locs={'NORDIC_DNK':(10,56),'NORDIC_EST':(26,59),'NORDIC_FIN':(27,66),'NORDIC_ISL':(-19,65),'NORDIC_LVA':(24,57),'NORDIC_LTU':(24,54.5),'NORDIC_NOR':(8,63),'NORDIC_SWE':(16,62),'FIN_DISTRIBUTION':(31,63),'USA_NY_QUALIFIED':(-75,43),'JAM_JPS':(-77,18),'SEN_SENELEC':(-14,14),'RWA_REG':(30,-2),'NZL_QUALIFIED21':(173,-41),'BRA_AGGREGATE689':(-49,-18)}
f=page(213);a=ax(f,13,11,188,59);base(a)
cm=LinearSegmentedColormap.from_list('tv',['#F4EADB','#D89B68','#B44A54','#6E2948']);norm=plt.Normalize(0,70)
for pid,(x,y) in locs.items():
 if pid.startswith('NORDIC') or pid=='FIN_DISTRIBUTION':continue
 a.scatter(x,y,s=75,c=[tv.loc[pid,'displacement_pct']],cmap=cm,norm=norm,edgecolor=INK,lw=.6,zorder=4)
 a.annotate(tv.loc[pid,'ISO3'],(x,y),xytext=(-6,-8) if pid.startswith('NZL') else (5,-7),textcoords='offset points',ha='right' if pid.startswith('NZL') else 'left',fontsize=9.5)
a.add_patch(Rectangle((-28,50),64,24,fill=False,edgecolor=INK,lw=.7))
inset=ax(f,129,15,70,29);W.plot(ax=inset,color='#F2F4F2',edgecolor='#B3BCBF',lw=.3,aspect='auto');inset.set(xlim=(-28,39),ylim=(50,73));inset.set_aspect('auto');inset.set_xticks([]);inset.set_yticks([])
for s in inset.spines.values():s.set_color(GREY)
geographic_frame(inset,4326,longitudes=[-20,0,20],latitudes=[55,65],fontsize=8.8,label_bottom=False,label_left=False)
offsets={'NORDIC_DNK':(-7,-10),'NORDIC_EST':(6,1),'NORDIC_FIN':(-18,6),'NORDIC_ISL':(5,1),'NORDIC_LVA':(5,-1),'NORDIC_LTU':(5,-4),'NORDIC_NOR':(-19,3),'NORDIC_SWE':(-14,-8),'FIN_DISTRIBUTION':(4,0)}
for pid,(dx,dy) in offsets.items():
 x,y=locs[pid];inset.scatter(x,y,s=51,c=[tv.loc[pid,'displacement_pct']],cmap=cm,norm=norm,edgecolor=INK,lw=.5,zorder=4)
 inset.annotate('FIN-D' if pid=='FIN_DISTRIBUTION' else tv.loc[pid,'ISO3'],(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=9,va='center')
title(f,a,'a','Counting interruptions and weighing losses change cause priorities',offset=9)
cb(f,cm,norm,[45,77,126,2.5],[0,20,40,60,70],['0','20','40','60','70'],'Change in cause composition (%)')
b=ax(f,63,108,138,83)
mat=d.pivot(index='profile_id',columns='cause_group',values='share_gap_pp').reindex(index=order,columns=CAUSES)
cm2=LinearSegmentedColormap.from_list('delta',['#235D80','#8CB8CC','#F8F6F2','#D78A8B','#912E4C']);cm2.set_bad('#E8EAE9');norm2=TwoSlopeNorm(vmin=-50,vcenter=0,vmax=50)
b.imshow(np.ma.masked_invalid(mat),aspect='auto',cmap=cm2,norm=norm2,interpolation='none',origin='upper')
for i in range(len(order)):
 for j in range(7):
  val=mat.iloc[i,j]
  if pd.isna(val):b.text(j,i,'—',ha='center',va='center',fontsize=9,color='#949DA0')
  else:b.text(j,i,f'{int(np.rint(val)):+d}' if int(np.rint(val))!=0 else '0',ha='center',va='center',fontsize=9.6,color='white' if abs(val)>27 else INK)
b.set_yticks(range(15),labels,fontsize=9.1);b.set_xticks(range(7),['Weather','Other\nenvironment','Human /\noperation','Equipment /\nelectrical','Supply\nconstraint','Planned','Other /\nunresolved'],fontsize=9.1);b.tick_params(axis='both',length=0);b.xaxis.tick_top();b.tick_params(axis='x',pad=5)
b.set_xticks(np.arange(-.5,7),minor=True);b.set_yticks(np.arange(-.5,15),minor=True);b.grid(which='minor',color='white',linewidth=.7);b.tick_params(which='minor',length=0)
for sp in b.spines.values():sp.set_visible(False)
title(f,b,'b','Which causes gain or lose importance?',offset=55)
# Move panel title above its two-line column labels.
for tx in f.texts[-2:]:tx.set_y(1-93/f._wh[1])
cb(f,cm2,norm2,[75,198,114,2.5],[-50,-25,0,25,50],['−50','−25','0','25','50'],'Burden share minus occurrence share (pp)',extend='both')
# Missing classes and native units are defined in the caption, leaving a compact lower edge.
save(f,'Fig04_causes_reshape_loss')
(P/'metadata/story_figure_analysis.json').write_text(json.dumps({'profiles':len(tv),'countries':tv.ISO3.nunique(),'leaders_changed':int(tv.leader_changes.sum()),'composition_displacement_range':list(map(float,[tv.displacement_pct.min(),tv.displacement_pct.max()])),'design':'Fig3 access context -> conditional within-country gap -> exact same-population partition. Fig4 total profile displacement -> signed cause contributions; no pooled duration interpretation.'},indent=2))
