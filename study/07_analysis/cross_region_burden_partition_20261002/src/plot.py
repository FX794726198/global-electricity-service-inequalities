"""Global cause priorities and their contributions to observed economic service gaps."""
from pathlib import Path
import sys,json,shutil
P=Path(__file__).resolve().parents[1];R=P.parents[1]
sys.path.insert(0,str(R/'10_manuscript/20261002_global_blackout_inequalities_v14/src'))
import figure_style as fs
fs.P=P
from figure_style import *
W=gpd.read_parquet(R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
W=W[W.ISO3.ne('ATA')].copy()
OLD=R/'10_manuscript/20261002_global_blackout_inequalities_v28'
tv=pd.read_csv(OLD/'source_data/Fig04_profile_displacement.csv').set_index('profile_id')
d=pd.read_csv(P/'data/class_gap_components.csv');s=pd.read_csv(P/'data/regional_gap_summary.csv').set_index('region')
for ext in ['png','pdf','svg']:
 shutil.copy2(OLD/'figures'/f'Fig04_causes_reshape_loss.{ext}',P/'figures'/f'SuppFig23_global_cause_composition.{ext}')
f=page(190)
a=ax(f,13,10,194,59)
W.plot(ax=a,color='#E8EAE9',edgecolor='#BBC2C4',lw=.27,aspect='auto')
a.set(xlim=(-180,180),ylim=(-58,82));a.set_aspect('auto')
geographic_frame(a,4326,longitudes=[-120,-60,0,60,120],latitudes=[-30,0,30,60],fontsize=10)
cm=LinearSegmentedColormap.from_list('tv',['#F4EADB','#D89B68','#B44A54','#6E2948']);norm=plt.Normalize(0,70)
locs={'NORDIC_DNK':(10,56),'NORDIC_EST':(26,59),'NORDIC_FIN':(27,66),'NORDIC_ISL':(-19,65),'NORDIC_LVA':(24,57),'NORDIC_LTU':(24,54.5),'NORDIC_NOR':(8,63),'NORDIC_SWE':(16,62),'FIN_DISTRIBUTION':(31,63),'USA_NY_QUALIFIED':(-75,43),'JAM_JPS':(-77,18),'SEN_SENELEC':(-14,14),'RWA_REG':(30,-2),'NZL_QUALIFIED21':(173,-41),'BRA_AGGREGATE689':(-49,-18)}
for pid,(x,y) in locs.items():
 if pid.startswith('NORDIC') or pid=='FIN_DISTRIBUTION':continue
 a.scatter(x,y,s=76,c=[tv.loc[pid,'displacement_pct']],cmap=cm,norm=norm,edgecolor=INK,lw=.6,zorder=4)
 label='NY' if pid=='USA_NY_QUALIFIED' else tv.loc[pid,'ISO3']
 a.annotate(label,(x,y),xytext=(-5,-8) if pid.startswith('NZL') else (5,-7),textcoords='offset points',ha='right' if pid.startswith('NZL') else 'left',fontsize=10)
a.add_patch(Rectangle((-28,50),64,24,fill=False,edgecolor=INK,lw=.7))
inset=ax(f,135,13,70,28)
W.plot(ax=inset,color='#F2F4F2',edgecolor='#B3BCBF',lw=.3,aspect='auto');inset.set(xlim=(-28,39),ylim=(50,73));inset.set_aspect('auto');inset.set_xticks([]);inset.set_yticks([])
for sp in inset.spines.values():sp.set_color(GREY)
geographic_frame(inset,4326,longitudes=[-20,0,20],latitudes=[55,65],fontsize=9.2,label_bottom=False,label_left=False)
offsets={'NORDIC_DNK':(-7,-10),'NORDIC_EST':(6,1),'NORDIC_FIN':(-18,6),'NORDIC_ISL':(5,1),'NORDIC_LVA':(5,-1),'NORDIC_LTU':(5,-4),'NORDIC_NOR':(-19,3),'NORDIC_SWE':(-14,-8),'FIN_DISTRIBUTION':(4,0)}
for pid,(dx,dy) in offsets.items():
 x,y=locs[pid];inset.scatter(x,y,s=52,c=[tv.loc[pid,'displacement_pct']],cmap=cm,norm=norm,edgecolor=INK,lw=.5,zorder=4)
 inset.annotate('FIN-D' if pid=='FIN_DISTRIBUTION' else tv.loc[pid,'ISO3'],(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=9.5,va='center')
title(f,a,'a','Losses change the importance of failure causes',offset=9)
cb(f,cm,norm,[50,76,123,2.5],[0,20,40,60,70],['0','20','40','60','70'],'Change in cause composition (%)')
legend(f,[Patch(color=BLUE),Patch(color=RED),Line2D([],[],color=INK,marker='o',lw=.8,ms=3.5)],['Frequency component','Duration component','Net gap and 95% interval'],109,96,3,9.7)
settings=[
 ('Brazil','b','Brazil · 2021',13,60,(-3,19),[0,5,10,15],['Ordinary unplanned','Emergency','Critical day','Planned']),
 ('New Zealand','c','New Zealand · 2023–25',81,60,(-2,10),[0,3,6,9],['Weather','Human activity','Remaining causes']),
 ('United States','d','United States · 2013–24',149,58,(-1.5,6),[-1,0,2,4,6],['Routine / non-major','Exceptional / major'])]
for region,letter,label,x,width,xlim,ticks,categories in settings:
 h=60;a2=ax(f,x,112,width,h)
 cats=['All classes']+categories
 # Fixed vertical slots keep the three regional displays directly comparable.
 for j,cat in enumerate(cats):
  y=4-j
  if j%2==0:a2.axhspan(y-.45,y+.53,color='#F0F2F2',zorder=0)
  if j==0:
   rr=s.loc[region];val,lo,hi=rr.total_gap,rr.total_gap_lower,rr.total_gap_upper
  else:
   z=d[d.region.eq(region)&d.category.eq(cat)].set_index('component')
   pos=neg=0.
   for comp,col in [('frequency',BLUE),('duration',RED)]:
    v=z.loc[comp,'value_h'];a2.barh(y-.06,v,left=pos if v>=0 else neg,height=.21,color=col,zorder=2)
    if v>=0:pos+=v
    else:neg+=v
   val,lo,hi=z.loc['gap',['value_h','lower_h','upper_h']]
  a2.plot([lo,hi],[y-.06,y-.06],color=INK,lw=.75,zorder=4)
  a2.scatter([val],[y-.06],s=14,c=INK,edgecolor='white',lw=.3,zorder=5)
  a2.text(.025,(y+.23+.5)/5,{'Routine / non-major':'Routine','Exceptional / major':'Major event'}.get(cat,cat),transform=a2.transAxes,fontsize=9.6,ha='left',va='center',fontweight='bold' if j==0 else 'normal',zorder=6)
 a2.axvline(0,color='#8C9599',ls=(0,(2,2)),lw=.65,zorder=1)
 a2.set(xlim=xlim,ylim=(-.5,4.5));a2.set_xticks(ticks);a2.set_yticks([]);clean(a2);a2.spines['left'].set_visible(False)
 a2.tick_params(axis='x',labelsize=9.6)
 title(f,a2,letter,label,offset=8)
 # Compact sample annotation occupies the deliberately unused class slot.
 if len(cats)<5:
  a2.text(.98,.035,f'{int(s.loc[region,"n_Q1"])} lower / {int(s.loc[region,"n_Q4"])} higher',transform=a2.transAxes,fontsize=9.4,ha='right',color='#57656C')
text(f,108,184,'Annual service-hour gap: lower − higher economic group (h/customer/year)',ha='center',fontsize=10)
save(f,'Fig04_causes_reshape_loss')
(P/'metadata/figure_source_lineage.json').write_text(json.dumps({'global_panel':'v28 source_data/Fig04_profile_displacement.csv; 15 source-native profiles in 14 countries','regional_panels':'data/class_gap_components.csv and regional_gap_summary.csv','intervals':'4000 complete spatial-cluster bootstrap draws in frozen economic groups','bottom_units':'annual customer interruption hours; zero-inclusive class means','not_identical_samples':'Global map New York is eight reporting regions; bottom US uses 49-state annual customer indices. Global Brazil profile uses 689 areas; bottom regulatory comparison uses 814 areas. These are independent compatible-within-source populations.','approved_Fig1_changed':False},indent=2))
