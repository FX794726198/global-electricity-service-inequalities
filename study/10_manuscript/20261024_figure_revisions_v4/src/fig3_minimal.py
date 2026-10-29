"""Fig. 3 minimal revision (v5): original 183 x 183 mm layout (a-h). Panel h now compares, for the two Turkish
regions where the leading cause differs, each leading cause's share of balanced mean annual hours with its share of the
rural-urban gap (same populations, all reported categories in each denominator). Map label A uses the full region name.

Country-internal service-hour accounting, with geography tied to frozen populations.

Contract: annual interruption hours -> economic or settlement gap -> recorded classes.
The figure compares within-source differences, not nationally representative levels.
Python is the established plotting backend. Source geometry is copied without change.
"""
from pathlib import Path
import json, hashlib, sys
_runtime_root=Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.text import Text
from pyproj import Geod
from scipy.optimize import brentq
REV=Path(__file__).resolve().parents[1]
Q=REV.parent/'20261003_global_blackout_inequalities_v37'
P=Q.parent/'20261003_global_blackout_inequalities_v33'
S=P/'source_data'; C=P/'core_reproducibility'; OUT=REV/'minimal_v5'; NEW=Q/'source_data'; ADIR=REV/'minimal_v5'
ADIR.mkdir(parents=True,exist_ok=True)
V=C/'class_gap_extension/inputs'
BLUE='#376F9F'; RED='#B23B4D'; INK='#283238'; TEAL='#32938C'; GREY='#9EA9AE'; WEATHER='#D2963C'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','text.color':INK,
 'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
manifest=[];checks=[]
def ck(n,c):
 checks.append({'check':n,'passed':bool(c)}); assert c,n
def used(p):
 manifest.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return p
w=gpd.read_parquet(used(C/'inputs/world_geometry.geoparquet')).to_crs(4326)
w.geometry=w.geometry.simplify(.018,preserve_topology=True)
bp=pd.read_csv(used(NEW/'poverty_priority_priority_summary.csv'))
bp=bp[(bp.domain=='customer')&(bp.weighting=='area_equal')].copy()
br=bp[bp['class']=='Covered internal unplanned'].iloc[0]
bc=pd.read_csv(used(NEW/'poverty_priority_area_context.csv'))
bgeo=gpd.read_parquet(used(Q.parent.parent/'07_analysis/brazil_detailed_cause_extension_20261002/data/economic_geographic_cohort.geoparquet'))
bg=bgeo[['IdeConjunto','geometry']].merge(bc[['IdeConjunto','covered_internal_unplanned_H','covered_internal_unplanned_F','disadvantage_quartile']],on='IdeConjunto',validate='one_to_one').rename(columns={'covered_internal_unplanned_H':'map_hours','covered_internal_unplanned_F':'map_frequency'})
n=pd.read_parquet(used(V/'nz_explicit21_total_context.parquet'));n['map_hours']=n.burden_h;n['map_frequency']=n.frequency
ng=gpd.read_parquet(used(S/'domestic_v31_nz_geometry.parquet')).merge(n,on='system',validate='one_to_one')
u=pd.read_parquet(used(V/'us_same49_state_means.parquet'));u['map_hours']=u.H_routine+u.H_exceptional;u['map_frequency']=u.F_routine+u.F_exceptional
ug=gpd.read_parquet(used(S/'domestic_v31_us_geometry.parquet')).merge(u,on='state',validate='one_to_one')
for name,g in [('Brazil',bg),('New Zealand',ng),('United States',ug)]:
 ck(name+' finite nonnegative map hours',g.map_hours.ge(0).all() and g.map_hours.notna().all())
 ck(name+' matched map count',len(g)=={'Brazil':703,'New Zealand':21,'United States':49}[name])
tg=gpd.read_parquet(used(C/'turkiye_extension/data/turkiye_province_boundaries.parquet'))
ts=gpd.read_parquet(used(C/'turkiye_extension/data/study_province_boundaries.parquet'))
t=pd.read_csv(used(S/'turkiye_multiregion_key_regional_results_2025.csv')).set_index('provider')
order=['baskent','sedas','uludag','toroslar']; t=t.loc[order]
summary=pd.read_csv(used(S/'class_gap_regional_gap_summary.csv')).set_index('region')
for newcol,oldcol in [('gap_H','total_gap'),('frequency_component_H','aggregate_frequency'),('duration_component_H','aggregate_duration')]:
 summary.loc['Brazil',oldcol]=br[newcol]
 summary.loc['Brazil',oldcol+'_lower']=br[newcol+'_ci_low']
 summary.loc['Brazil',oldcol+'_upper']=br[newcol+'_ci_high']
summary.loc['Brazil','total_units']=703
summary.loc['Brazil','period']='2024–2025'
summary.loc['Brazil','n_Q1']=(bc.disadvantage_quartile=='Q4').sum()
summary.loc['Brazil','n_Q4']=(bc.disadvantage_quartile=='Q1').sum()
summary.loc['Brazil','low_hours']=br.Q4_H
summary.loc['Brazil','high_hours']=br.Q1_H
pc=pd.read_csv(used(NEW/'priority_class_estimates.csv'))
classes=pc[pc.profile.isin(['New Zealand','United States'])][['profile','category','H_gap','gap_lower','gap_upper']].rename(columns={'profile':'region','H_gap':'value_h','gap_lower':'lower_h','gap_upper':'upper_h'})
bcl=bp[bp['class'].isin(['Ordinary unplanned','Emergency / critical'])][['class','gap_H','gap_H_ci_low','gap_H_ci_high']].rename(columns={'class':'category','gap_H':'value_h','gap_H_ci_low':'lower_h','gap_H_ci_high':'upper_h'})
bcl['region']='Brazil';classes=pd.concat([bcl,classes],ignore_index=True)
base=pd.read_csv(used(S/'class_gap_frozen_accounting_inputs.csv'))
boot=pd.read_parquet(used(C/'class_gap_extension/data/bootstrap_draws.parquet'))
contrasts=[]
for country in ['Brazil','New Zealand','United States']:
 if country=='Brazil':
  contrasts.append(dict(country=country,frequency_minus_duration_h=float(br.frequency_minus_duration_component_H),lower_h=float(br.frequency_minus_duration_component_H_ci_low),upper_h=float(br.frequency_minus_duration_component_H_ci_high)))
 else:
  dr=boot[boot.region.eq(country)];v=dr.aggregate_frequency-dr.aggregate_duration
  lo,hi=np.quantile(v,[.025,.975]);r=summary.loc[country]
  contrasts.append(dict(country=country,frequency_minus_duration_h=float(r.aggregate_frequency-r.aggregate_duration),lower_h=float(lo),upper_h=float(hi)))
for name,g,key in [('New Zealand',ng,'system'),('United States',ug,'state')]:
 x=g.copy();x['unit']=x[key].astype(str)
 y=base[base.region.eq(name)].groupby(['unit','quartile'],as_index=False).agg(hours=('burden_h','sum'),frequency=('frequency','sum'))
 z=y.merge(x[['unit','map_hours','map_frequency']],on='unit',validate='one_to_one')
 ck(name+' map uses same accounting scope',np.allclose(z.hours,z.map_hours) and np.allclose(z.frequency,z.map_frequency))
 rr=summary.loc[name]
 for qt,expected in [('Q1','low_hours'),('Q4','high_hours')]:ck(name+' '+qt+' mapped hours mean',np.isclose(z.loc[z.quartile.eq(qt),'map_hours'].mean(),rr[expected]))
for qt,expected in [('Q1','Q1_H'),('Q4','Q4_H')]:ck('Brazil '+qt+' mapped hour mean',np.isclose(bg.loc[bg.disadvantage_quartile.eq(qt),'map_hours'].mean(),br[expected]))
for name in ['Brazil','New Zealand','United States']:
 rr=summary.loc[name]
 ck(name+' aggregate gap closes',np.isclose(rr.aggregate_frequency+rr.aggregate_duration,rr.total_gap))
 ck(name+' class gap closes',np.isclose(classes[classes.region.eq(name)].value_h.sum(),rr.total_gap))
ck('Turkiye four regional partitions close',np.allclose(t.frequency_component_h+t.duration_component_h,t.hour_gap))
ck('Turkiye map matches region',np.allclose(ts.regional_hour_gap,ts.provider.map(t.hour_gap)))
rows=[]
for name in ['Brazil','New Zealand','United States']:
 r=summary.loc[name]
 for term,label in [('total_gap','Total'),('aggregate_frequency','Frequency'),('aggregate_duration','Duration')]:
  rows.append(dict(country=name,comparison='Highest minus lowest municipal poverty quartile' if name=='Brazil' else 'Lowest minus highest regional income quartile',component=label,hours=r[term],lower=r[term+'_lower'],upper=r[term+'_upper'],period=r.period,n_units=int(r.total_units)))
for key,r in t.iterrows():
 for term,label in [('hour_gap','Total'),('frequency_component_h','Frequency'),('duration_component_h','Duration')]:
  rows.append(dict(country='Türkiye',comparison=r.region+' rural minus urban',component=label,hours=r[term],lower=np.nan,upper=np.nan,period=2025,n_units=np.nan))
# Dense, parallel layout. Geographic panels remain actual measured fields.
f=plt.figure(figsize=(183/25.4,183/25.4))
xs=[.055,.291,.527,.763]; width=.212
map_y=.715;map_h=.235
axs=[]; map_axes=[]
def panel(box,letter,title=None):
 a=f.add_axes(box);a.text(-.10,1.045 if title else 1.06,letter,transform=a.transAxes,fontsize=10,fontweight='bold',ha='left',va='bottom' if title else 'top')
 if title:a.set_title(title,loc='left',pad=5,fontsize=9)
 axs.append(a);return a
stat_headers=[]; country_labels=[]
def grouped_header(box,letter,title):
 header_y=box[1]+box[3]+.012
 f.text(box[0]-.10*width,header_y,letter,fontsize=10,fontweight='bold',ha='left',va='baseline')
 f.text(box[0],header_y,title,fontsize=8.5,ha='left',va='baseline')
 stat_headers.append({'panel':letter,'title':title,'letter_position':[box[0]-.10*width,header_y],'title_position':[box[0],header_y],'axes_box':box,'grouped_mini_axes':3})
def country_mini_panel(box,country,group):
 a=f.add_axes(box)
 tx=a.text(.98,.035,country,transform=a.transAxes,ha='right',va='bottom',fontsize=8,color=INK,zorder=10)
 country_labels.append({'panel':group,'country':country,'axes_position':[.98,.035],'axes_box':box})
 axs.append(a);return a
def statistic_panel(box,letter,title):
 a=f.add_axes(box)
 header_y=box[1]+box[3]+.012
 letter_text=f.text(box[0]-.10*box[2],header_y,letter,fontsize=10,fontweight='bold',ha='left',va='baseline')
 title_text=f.text(box[0],header_y,title,fontsize=8.5,ha='left',va='baseline')
 stat_headers.append({'panel':letter,'title':title,'letter_position':[box[0]-.10*box[2],header_y],'title_position':[box[0],header_y],'axes_box':box})
 axs.append(a);return a
bounds=[0,.5,1,2,5,10,20,50,100]
cm=ListedColormap(['#EFF5F3','#DAEAE4','#B9D6CC','#8FBBAD','#619D90','#377E73','#1E5B56','#113E3C'])
norm=BoundaryNorm(bounds,cm.N)
extents=[(-76,-32,-38.5,8.5),(164,181,-48,-34),(-126,-66,23,51),(25,45,35,44)]
ticks=[([-70,-50],[0,-20]),([170,175],[-35,-45]),([-120,-90],[30,45]),([30,40],[36,42])]
def frame(a,lim,xt,yt):
 a.set_xlim(lim[:2]);a.set_ylim(lim[2:]);a.set_aspect(1/np.cos(np.deg2rad((lim[2]+lim[3])/2)),adjustable='box',anchor='N');a.set_xticks(xt);a.set_yticks(yt)
 a.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{abs(x):g}°'+('W' if x<0 else 'E' if x>0 else '')))
 a.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{abs(x):g}°'+('S' if x<0 else 'N' if x>0 else '')))
 a.tick_params(length=2,pad=1.5,labelsize=8,top=True,right=True)
 a.grid(ls=(0,(2,3)),color='#B7C1C5',lw=.35,alpha=.7)
 for sp in a.spines.values():sp.set_visible(True);sp.set_color('#889599');sp.set_linewidth(.6)
scale_metadata=[]
geod=Geod(ellps='WGS84')
def scale_bar(a,lon,lat,km,dy,label_offset,country):
 right=brentq(lambda end:geod.inv(lon,lat,end,lat)[2]-km*1000,lon+1e-5,lon+35)
 # Line and end caps are confined to a checked empty part of each map.
 a.plot([lon,right],[lat,lat],color=INK,lw=1.25,solid_capstyle='butt',zorder=20)
 a.plot([lon,lon],[lat-dy,lat+dy],color=INK,lw=.8,zorder=20)
 a.plot([right,right],[lat-dy,lat+dy],color=INK,lw=.8,zorder=20)
 a.text((lon+right)/2,lat+label_offset,f'{km:g} km',ha='center',va='bottom',fontsize=7.3,color=INK,zorder=20)
 distance=geod.inv(lon,lat,right,lat)[2]/1000
 assert abs(distance-km)<1e-6
 scale_metadata.append({'map':country,'left_lon':lon,'right_lon':right,'latitude':lat,'km':km,'geodesic_distance_km':distance,'calibration':'WGS84 inverse geodesic between equal-latitude endpoints'})
mapinfo=[('Brazil',bg),('New Zealand',ng),('United States',ug)]
for j,(title,g) in enumerate(mapinfo):
 map_left=[.055,.310,.560][j];map_width=[.196,.196,.185][j]
 a=panel([map_left,map_y,map_width,map_h],chr(97+j),title)
 map_axes.append((title,a))
 w.cx[extents[j][0]:extents[j][1],extents[j][2]:extents[j][3]].plot(ax=a,color='#E9ECEB',edgecolor='#BAC2C3',lw=.32,aspect='auto')
 drawg=g.copy();drawg.geometry=drawg.geometry.simplify(.008,preserve_topology=True)
 drawg.plot(column='map_hours',ax=a,cmap=cm,norm=norm,edgecolor='white',lw=.13,aspect='auto')
 frame(a,extents[j],*ticks[j])
 scale_configs=[(-47,-35,500,.25,.6), (165,-36.3,250,.12,.32),(-121.5,25,1000,.20,.45)]
 scale_bar(a,*scale_configs[j],title)
 if j==2:
  for k,(state,lim) in enumerate([('AK',(-170,-130,46,72))]):
   ia=f.add_axes([.572,.700,.161,.082]);map_axes.append(('Alaska',ia));z=drawg[drawg.state.eq(state)]
   z.plot(column='map_hours',ax=ia,cmap=cm,norm=norm,edgecolor='white',lw=.25,aspect='auto')
   frame(ia,lim,[-150] if k==0 else [-160,-155],[60] if k==0 else [20]);ia.tick_params(labelsize=8);ia.set_title('Alaska' if k==0 else 'Hawaii',fontsize=8,pad=3)
   scale_bar(ia,-153,47.5,1000,.25,.25,'Alaska')
a=panel([.800,map_y,.175,map_h],'d','Türkiye')
map_axes.append(('Türkiye',a))
tg.plot(ax=a,color='#E9ECEB',edgecolor='#BEC4C5',lw=.25,aspect='auto')
tcm=LinearSegmentedColormap.from_list('rural',['#F4DFCF','#CC8978','#A74B61','#672B48']);tnorm=Normalize(0,80)
ts.plot(column='regional_hour_gap',ax=a,cmap=tcm,norm=tnorm,edgecolor='white',lw=.27,aspect='auto');frame(a,extents[3],*ticks[3])
scale_bar(a,38.5,35.55,500,.10,.18,'Türkiye')
for label,x,y in [('A',33.3,40.0),('B',30.75,40.85),('C',28.6,39.4),('D',36.3,37.1)]:
 a.text(x,y,label,ha='center',va='center',fontsize=8,weight='bold',color=INK,bbox=dict(fc='white',ec='none',pad=.4))
for yy,lab in zip([.794,.770,.746,.722],['A  Ankara–W. Black Sea','B  Bolu–Sakarya','C  Southern Marmara','D  E. Mediterranean']):
 f.text(xs[3]+.006,yy,lab,fontsize=7.6)
cax=f.add_axes([.165,.658,.47,.010]);cb=f.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation='horizontal',ticks=[.5,2,10,50],extend='max');cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'));cb.set_label('Annual interruption hours per customer',labelpad=2);cb.outline.set_linewidth(.5);cb.ax.tick_params(labelsize=8,length=2,pad=1.3)
cax=f.add_axes([xs[3],.658,width,.010]);cb=f.colorbar(plt.cm.ScalarMappable(norm=tnorm,cmap=tcm),cax=cax,orientation='horizontal',ticks=[0,40,80]);cb.set_label('Rural excess hours',labelpad=2);cb.outline.set_linewidth(.5);cb.ax.tick_params(labelsize=8,length=2,pad=1.3)
# Aggregate accounting, equal scales across economic contrasts.
for j,name in enumerate(['Brazil','New Zealand','United States']):
 a=country_mini_panel([xs[j],.415,width,.147],name,'e')
 r=summary.loc[name]
 for i,(term,col) in enumerate([('total_gap',INK),('aggregate_frequency',BLUE),('aggregate_duration',RED)]):
  y=2-i;val=r[term];lo=r[term+'_lower'];hi=r[term+'_upper']
  a.plot([lo,hi],[y,y],color=col,lw=1.25,zorder=2)
  a.scatter(val,y,s=20,color=col,marker='D' if i==0 else 'o',zorder=3)
 a.axvline(0,color='#8A959A',ls=(0,(3,3)),lw=.65,zorder=1)
 a.set(xlim=(-25,35) if name=='Brazil' else (-3,20),ylim=(-.55,2.55),xticks=[-20,0,20] if name=='Brazil' else [0,10,20],yticks=[])

 a.set_xlabel('Excess hours',labelpad=2)
 a.tick_params(length=2,pad=2)
 a.spines['left'].set_visible(False)
grouped_header([xs[0],.415,xs[2]+width-xs[0],.147],'e','Higher poverty (Brazil); lower income (NZ/US)')
a=statistic_panel([xs[3],.415,width,.147],'f','Rural–urban gap')
for i,(key,r) in enumerate(t.iterrows()):
 a.barh(3-i,r.frequency_component_h,color=BLUE,height=.53)
 a.barh(3-i,r.duration_component_h,left=r.frequency_component_h,color=RED,height=.53)
 a.plot(r.hour_gap,3-i,'D',ms=3,color=INK)
a.set(xlim=(0,83),ylim=(-.55,3.55),xticks=[0,40,80],yticks=[3,2,1,0],yticklabels=['A','B','C','D'],xlabel='Excess hours')
a.tick_params(axis='y',length=0,pad=2);a.tick_params(axis='x',length=2,pad=2);a.spines['left'].set_visible(False)
# Panel f remains a separate settlement comparison; e spans three economic comparisons.
# Named classes of exactly the same economic hour gaps. All CIs retained.
configs=[('Brazil',['Ordinary unplanned','Emergency / critical'],['Ordinary unplanned','Emergency / critical']),('New Zealand',['Weather','Human activity','Remaining causes'],['Weather','Human activity','Remaining']),('United States',['Routine / non-major','Exceptional / major'],['Other days','Major-event days'])]
for j,(name,ca,la) in enumerate(configs):
 a=country_mini_panel([xs[j],.097,width,.193],name,'g')
 z=classes[classes.region.eq(name)].set_index('category').loc[ca]
 for k,(_,r) in enumerate(z.iterrows()):
  y=len(z)-1-k
  a.plot([r.lower_h,r.upper_h],[y,y],color=INK,lw=1.05,zorder=2)
  a.scatter(r.value_h,y,s=19,color=INK,zorder=3)
  a.text(.015,y+.17,la[k],transform=a.get_yaxis_transform(),fontsize=8,va='bottom')
 a.axvline(0,color='#8A959A',ls=(0,(3,3)),lw=.65)
 a.set(xlim=(-15,25) if name=='Brazil' else (-3,20),ylim=(-.5,len(z)-.35),xticks=[-10,0,20] if name=='Brazil' else [0,10,20],yticks=[],xlabel='Excess hours')
 a.tick_params(length=2,pad=2);a.spines['left'].set_visible(False)
grouped_header([xs[0],.097,xs[2]+width-xs[0],.193],'g','Recorded classes contributing to the same gap')
# h: for the two regions where the leading cause differs, compare each leading cause's share of balanced mean
# annual hours with its share of the rural-urban gap (same 2025 populations; all categories in each denominator).
a=statistic_panel([xs[3],.097,width,.193],'h','Mean hours vs rural gap')
EQUIP='#8B75A2'
priority_plotted=[];priority_annotations=[];priority_markers=[]
ypos={('A','Equipment / electrical conditions'):3.25,('A','Weather'):2.45,('B','Equipment / electrical conditions'):1.05,('B','Weather'):0.25}
for reg,region in [('A','Ankara–western Black Sea'),('B','Bolu–Sakarya')]:
 z=pc[pc.profile.eq('Türkiye: '+region)].set_index('category')
 ck(region+' balanced hours identity',np.allclose(z.H_balanced_mean,(z.H_low+z.H_high)/2))
 ck(region+' same-population gap identity',np.allclose(z.H_gap,z.H_low-z.H_high))
 ck(region+' all-category mean shares close',np.isclose(z.H_share_pct.sum(),100))
 ck(region+' all-category gap shares close',np.isclose(z.signed_gap_share_pct.sum(),100))
 ranked=z.drop(index=[c for c in z.index if c=='Reconciliation residual'])
 ck(region+' equipment leads balanced mean',ranked.H_balanced_mean.idxmax()=='Equipment / electrical conditions')
 ck(region+' weather leads rural-urban gap',ranked.H_gap.idxmax()=='Weather')
 ck(region+' gap matches retained regional result',np.isclose(z.H_gap.sum(),t.loc['baskent' if region.startswith('Ankara') else 'sedas','hour_gap']))
 for cat,col in [('Equipment / electrical conditions',EQUIP),('Weather',WEATHER)]:
  r=z.loc[cat];y=ypos[(reg,cat)]
  a.plot([r.H_share_pct,r.signed_gap_share_pct],[y,y],color=col,lw=1.4,alpha=.5,zorder=1,solid_capstyle='butt')
  a.scatter(r.H_share_pct,y,s=17,marker='s',facecolor='white',edgecolor=col,lw=1.0,zorder=3)
  a.scatter(r.signed_gap_share_pct,y,s=21,marker='D',color=col,zorder=3)
  priority_plotted.append({'panel':'h','region':region,'category':cat,'balanced_mean_hours':r.H_balanced_mean,'rural_hours':r.H_low,'urban_hours':r.H_high,'mean_share_pct':r.H_share_pct,'gap_hours':r.H_gap,'gap_share_pct':r.signed_gap_share_pct})
a.set(xlim=(0,65),ylim=(-.45,3.8),xticks=[0,20,40,60],yticks=[2.85,0.65],yticklabels=['A','B'],xlabel='Share (%)')
a.tick_params(axis='y',length=0,pad=2);a.tick_params(axis='x',length=2,pad=2);a.spines['left'].set_visible(False)
f.legend([Line2D([],[],color=INK,marker='D',lw=1,ms=3),Line2D([],[],color=BLUE,marker='o',lw=1,ms=3),Line2D([],[],color=RED,marker='o',lw=1,ms=3)],['Total','Frequency','Duration'],ncol=3,loc='center',bbox_to_anchor=(.40,.351),fontsize=8,columnspacing=1.1,handlelength=1.2,handletextpad=.35)
f.legend([Patch(fc=EQUIP),Patch(fc=WEATHER),Line2D([],[],marker='s',ls='',mfc='white',mec=INK,mew=.9,ms=4),Line2D([],[],marker='D',ls='',color=INK,ms=3.6)],
 ['Equipment','Weather','Share of mean hours','Share of rural gap'],ncol=2,loc='upper center',bbox_to_anchor=(.87,.050),fontsize=7.3,columnspacing=.7,handlelength=.9,handletextpad=.3,labelspacing=.3)
for _ax in f.axes:
 if _ax.get_xlabel() in ('Longitude','Latitude') or 'degree' in _ax.get_xlabel():_ax.set_xlabel('')
 if 'degree' in _ax.get_ylabel() or _ax.get_ylabel() in ('Longitude','Latitude'):_ax.set_ylabel('')
# Numeric sidecar and layout audit.
f.canvas.draw();ren=f.canvas.get_renderer();texts=[]
numeric_marker_overlaps=[]
for annotation in priority_annotations:
 bb=annotation.get_window_extent(ren)
 for ax,mx,my,size in priority_markers:
  px,py=ax.transData.transform((mx,my));radius=(np.sqrt(size)/2+.5)*f.dpi/72
  dx=min(bb.x1,px+radius)-max(bb.x0,px-radius)
  dy=min(bb.y1,py+radius)-max(bb.y0,py-radius)
  if dx>0 and dy>0:numeric_marker_overlaps.append({'text':annotation.get_text(),'marker_x':mx,'marker_y':my,'overlap_px':[dx,dy]})
ck('priority numeric annotations do not overlap any priority marker',not numeric_marker_overlaps)
map_tick_intrusions=[]
for source,ax_source in map_axes:
 for tick in list(ax_source.get_xticklabels())+list(ax_source.get_yticklabels()):
  if not tick.get_visible() or not tick.get_text(): continue
  tb=tick.get_window_extent(ren)
  for target,ax_target in map_axes:
   if ax_target is ax_source: continue
   mb=ax_target.get_window_extent(ren)
   dx=min(tb.x1,mb.x1)-max(tb.x0,mb.x0);dy=min(tb.y1,mb.y1)-max(tb.y0,mb.y0)
   if dx>0 and dy>0:map_tick_intrusions.append({'from_map':source,'tick':tick.get_text(),'into_map':target,'overlap_px':[dx,dy]})
assert not map_tick_intrusions, map_tick_intrusions

for tx in f.findobj(Text):
 if tx.get_visible() and tx.get_text() and tx.get_window_extent(ren).width>0:
  bb=tx.get_window_extent(ren);texts.append({'text':tx.get_text(),'font_size_pt':tx.get_fontsize(),'bbox_px':list(bb.extents)})
overlaps=[]
for i,aa in enumerate(texts):
 for bb in texts[i+1:]:
  A=aa['bbox_px'];B=bb['bbox_px'];dx=min(A[2],B[2])-max(A[0],B[0]);dy=min(A[3],B[3])-max(A[1],B[1])
  if dx>.5 and dy>.5:overlaps.append({'a':aa['text'],'b':bb['text'],'overlap_px':[dx,dy]})
for ext in ['png','pdf','svg']:f.savefig(OUT/('Fig3_minimal.'+ext),dpi=600,bbox_inches='tight',pad_inches=.02)
f.savefig(ADIR/'preview.png',dpi=190,bbox_inches='tight',pad_inches=.02)
plt.close(f)

caption=''
meta={'figure':'Fig3_minimal','design':'Original 183 x 183 mm layout; map label A uses the full region name; panel h compares leading-cause shares of mean hours and rural gap for regions A and B.', 'caption':caption,'checks':checks,'inputs':manifest,'text':texts,'minimum_font_pt':min(x['font_size_pt'] for x in texts),'text_overlaps':overlaps,'statistical_panel_headers':stat_headers,'country_labels':country_labels,'scale_bars':scale_metadata,'map_tick_intrusions':map_tick_intrusions,'direct_frequency_duration_contrasts':contrasts,'priority_plotted':priority_plotted,'numeric_marker_overlaps':numeric_marker_overlaps,'results':pd.DataFrame(rows).replace({np.nan:None}).to_dict('records')}
(ADIR/'qa.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
(ADIR/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
pd.DataFrame(rows).to_csv(ADIR/'plotted_aggregate_components.csv',index=False)
classes.to_csv(ADIR/'plotted_class_gaps.csv',index=False)
pd.DataFrame(priority_plotted).to_csv(ADIR/'plotted_priority_values.csv',index=False)
(ADIR/'fig3_minimal_automated_checks.json').write_text(json.dumps({'checks':checks,'text_overlaps':overlaps,'map_tick_intrusions':map_tick_intrusions,'minimum_font_pt':meta['minimum_font_pt'],'numeric_marker_overlaps':numeric_marker_overlaps},indent=2))
print(json.dumps({'figure':'Fig3_minimal','checks':len(checks),'all_passed':all(x['passed'] for x in checks),'caption_words':len(caption.split()),'minimum_font_pt':meta['minimum_font_pt'],'map_tick_intrusions':map_tick_intrusions,'text_overlaps':overlaps},indent=2))
