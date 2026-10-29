"""Paired category priorities: frequency, annual hours and social-group hour gaps.

Established Python/matplotlib backend. No new statistical estimation is performed.
All numerical displays are read from frozen v37 CSVs; native source categories and
population denominators are retained. Maps represent observed regional values.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.text import Text
from pyproj import Geod
from scipy.optimize import brentq

Q=Path(__file__).resolve().parents[1]; BASE=Q.parent.parent
OLD=Q.parent/'20261003_global_blackout_inequalities_v33'
S=Q/'source_data'; F=Q/'figures'; A=Q/'analysis/fig4_design'
F.mkdir(exist_ok=True);A.mkdir(parents=True,exist_ok=True)
BLUE='#376F9F'; RED='#B23B4D'; INK='#283238'; GREY='#B6C0C4'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','text.color':INK,
 'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
manifest=[]; checks=[]; plotted=[]
def used(p):
 manifest.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return p
def ck(name,b):
 checks.append({'check':name,'passed':bool(b)});assert b,name

pc=pd.read_csv(used(S/'priority_class_estimates.csv'))
ps=pd.read_csv(used(S/'priority_profile_summary.csv'))
bp=pd.read_csv(used(S/'poverty_priority_priority_summary.csv'))
bp=bp[(bp.domain=='customer')&(bp.weighting=='area_equal')].copy()
ck('Brazil customer primary priority rows',len(bp)==3)
bc=pd.read_csv(used(S/'poverty_priority_area_context.csv'))
bgeo=gpd.read_parquet(used(BASE/'07_analysis/brazil_detailed_cause_extension_20261002/data/economic_geographic_cohort.geoparquet'))
bg=bgeo[['IdeConjunto','geometry']].merge(bc[['IdeConjunto','covered_internal_unplanned_H']],on='IdeConjunto',validate='one_to_one').rename(columns={'covered_internal_unplanned_H':'map_hours'})
ck('Brazil703 matched values',len(bg)==703 and bg.map_hours.notna().all())
V=OLD/'core_reproducibility/class_gap_extension/inputs'
n=pd.read_parquet(used(V/'nz_explicit21_total_context.parquet'))
ng=gpd.read_parquet(used(OLD/'source_data/domestic_v31_nz_geometry.parquet')).merge(n[['system','burden_h']],on='system',validate='one_to_one').rename(columns={'burden_h':'map_hours'})
u=pd.read_parquet(used(V/'us_same49_state_means.parquet'));u['map_hours']=u.H_routine+u.H_exceptional
ug=gpd.read_parquet(used(OLD/'source_data/domestic_v31_us_geometry.parquet')).merge(u[['state','map_hours']],on='state',validate='one_to_one')
w=gpd.read_parquet(used(OLD/'core_reproducibility/inputs/world_geometry.geoparquet')).to_crs(4326)
w.geometry=w.geometry.simplify(.025,preserve_topology=True)
tg=gpd.read_parquet(used(OLD/'core_reproducibility/turkiye_extension/data/turkiye_province_boundaries.parquet'))
ts=gpd.read_parquet(used(OLD/'core_reproducibility/turkiye_extension/data/study_province_boundaries.parquet'))
t=pd.read_csv(used(OLD/'source_data/turkiye_multiregion_key_regional_results_2025.csv')).set_index('provider')
ck('NZ21 and US49 map observations',len(ng)==21 and len(ug)==49)
ck('Observed map values finite and nonnegative',all(g.map_hours.notna().all() and g.map_hours.ge(0).all() for g in [bg,ng,ug]))
ck('Türkiye map values match regional gap',np.allclose(ts.regional_hour_gap,ts.provider.map(t.hour_gap)))

bplot=bp[bp['class'].isin(['Ordinary unplanned','Emergency / critical'])].rename(columns={'class':'category','baseline_F_share_pct':'F_share_pct','baseline_H_share_pct':'H_share_pct','gap_H':'H_gap','gap_H_ci_low':'gap_lower','gap_H_ci_high':'gap_upper'})
ck('Brazil shares include exactly covered categories',np.isclose(bplot.F_share_pct.sum(),100) and np.isclose(bplot.H_share_pct.sum(),100))
ck('Brazil class gaps close',np.isclose(bplot.H_gap.sum(),bp.loc[bp['class']=='Covered internal unplanned','gap_H'].iloc[0]))
for name in ['New Zealand','United States']:
 z=pc[pc.profile.eq(name)];ck(name+' category shares close',np.isclose(z.F_share_pct.sum(),100) and np.isclose(z.H_share_pct.sum(),100));ck(name+' gaps close',np.isclose(z.H_gap.sum(),ps.loc[ps.profile.eq(name),'total_H_gap'].iloc[0]))

fig=plt.figure(figsize=(183/25.4,202/25.4));mapaxes=[];texts_checked=[]
def plabel(a,letter,title):
 a.set_title(title,loc='left',pad=5,fontsize=9)
 a.text(-.12,1.045,letter,transform=a.transAxes,fontsize=10,weight='bold',va='bottom')

def frame(a,lim,xt,yt):
 a.set_xlim(lim[:2]);a.set_ylim(lim[2:]);a.set_aspect(1/np.cos(np.deg2rad((lim[2]+lim[3])/2)),adjustable='box',anchor='N')
 a.set_xticks(xt);a.set_yticks(yt)
 a.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{abs(x):g}°'+('W' if x<0 else 'E' if x>0 else '')))
 a.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{abs(x):g}°'+('S' if x<0 else 'N' if x>0 else '')))
 a.tick_params(length=2,pad=1.3,labelsize=7.8,top=True,right=True)
 a.grid(ls=(0,(2,3)),color='#B7C1C5',lw=.35,alpha=.7)
 for sp in a.spines.values():sp.set_visible(True);sp.set_color('#889599');sp.set_linewidth(.55)
geod=Geod(ellps='WGS84');scale_metadata=[]
def scalebar(a,lon,lat,km,dy=.15,offset=.3):
 right=brentq(lambda x:geod.inv(lon,lat,x,lat)[2]-km*1000,lon+1e-5,lon+35)
 a.plot([lon,right],[lat,lat],color=INK,lw=1.1,zorder=10)
 a.plot([lon,lon],[lat-dy,lat+dy],color=INK,lw=.6,zorder=10);a.plot([right,right],[lat-dy,lat+dy],color=INK,lw=.6,zorder=10)
 a.text((lon+right)/2,lat+offset,f'{km:g} km',ha='center',va='bottom',fontsize=7,zorder=10)
 scale_metadata.append({'left_lon':lon,'right_lon':right,'latitude':lat,'distance_km':km})
cm=ListedColormap(['#EFF5F3','#DAEAE4','#B9D6CC','#8FBBAD','#619D90','#377E73','#1E5B56','#113E3C']);norm=BoundaryNorm([0,.5,1,2,5,10,20,50,100],cm.N)
configs=[('Brazil',bg,[.060,.764,.205,.210],(-76,-32,-38.5,8.5),[-70,-50],[0,-20]),('New Zealand',ng,[.321,.764,.177,.210],(164,181,-48,-34),[170,175],[-35,-45]),('United States',ug,[.574,.764,.195,.210],(-126,-66,23,51),[-120,-90],[30,45])]
for j,(name,g,box,lim,xt,yt) in enumerate(configs):
 ax=fig.add_axes(box);plabel(ax,chr(97+j),name);mapaxes.append((name,ax))
 w.cx[lim[0]:lim[1],lim[2]:lim[3]].plot(ax=ax,color='#E9ECEB',edgecolor='#BAC2C3',lw=.26,aspect='auto')
 zz=g.copy();zz.geometry=zz.geometry.simplify(.012,preserve_topology=True);zz.plot(column='map_hours',ax=ax,cmap=cm,norm=norm,edgecolor='white',lw=.12,aspect='auto')
 frame(ax,lim,xt,yt)
 if j==0:scalebar(ax,-47,-35,500,.2,.6)
 if j==1:scalebar(ax,165,-36.3,250,.09,.25)
 if j==2:
  scalebar(ax,-121.5,25,1000,.15,.4)
  aa=fig.add_axes([.610,.752,.140,.077]);ug[ug.state.eq('AK')].plot(column='map_hours',ax=aa,cmap=cm,norm=norm,edgecolor='white',lw=.2,aspect='auto');frame(aa,(-170,-130,46,72),[-150],[60]);aa.set_title('Alaska',fontsize=7.8,pad=3);mapaxes.append(('Alaska',aa))
  scalebar(aa,-153,48,1000,.2,.2)
ax=fig.add_axes([.828,.764,.157,.210]);plabel(ax,'d','Türkiye');mapaxes.append(('Türkiye',ax))
tg.plot(ax=ax,color='#E9ECEB',edgecolor='#BFC6C8',lw=.22,aspect='auto')
tcm=LinearSegmentedColormap.from_list('rural',['#F6E6D9','#D59A84','#A95365','#632D49']);tnorm=Normalize(0,80)
ts.plot(column='regional_hour_gap',ax=ax,cmap=tcm,norm=tnorm,edgecolor='white',lw=.25,aspect='auto');frame(ax,(25,45,35,44),[30,40],[36,42]);scalebar(ax,38.5,35.6,500,.1,.2)
for label,x,y in [('A',33.3,40.0),('B',30.75,40.85),('C',28.6,39.4),('D',36.3,37.1)]:ax.text(x,y,label,ha='center',va='center',fontsize=8,weight='bold',bbox=dict(fc='white',ec='none',pad=.25))
for yy,lab in zip([.858,.835,.812,.789],['A  Ankara–Black Sea','B  Bolu–Sakarya','C  S. Marmara','D  E. Mediterranean']):fig.text(.820,yy,lab,fontsize=7.3)
cax=fig.add_axes([.145,.716,.535,.009]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation='horizontal',ticks=[.5,2,10,50],extend='max');cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'));cb.set_label('Annual interruption hours per customer',labelpad=2);cb.ax.tick_params(length=2,pad=1.5);cb.outline.set_linewidth(.5)
cax=fig.add_axes([.818,.716,.167,.009]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=tnorm,cmap=tcm),cax=cax,orientation='horizontal',ticks=[0,40,80]);cb.set_label('Rural − urban hours',labelpad=2);cb.ax.tick_params(length=2,pad=1.5);cb.outline.set_linewidth(.5)
fig.legend([Line2D([],[],color=BLUE,marker='o',lw=0,ms=4),Line2D([],[],color=RED,marker='s',lw=0,ms=4),Line2D([],[],color=INK,marker='D',lw=1,ms=3)],['Frequency share','Annual-hour share','Social-group hour gap'],ncol=3,loc='center',bbox_to_anchor=(.51,.650),handlelength=1.2,handletextpad=.35,columnspacing=1.2,fontsize=8)

# Each named panel comprises aligned share and hour-gap axes. Every plotted
# category uses the complete admitted category denominator, never subset-renormalized.
def comparison(box,letter,title,data,categories,labels,gaplim,gapticks,tag='',selected=False):
 x,y,width,height=box
 fig.text(x,y+height+.022,letter,fontsize=10,weight='bold',ha='left')
 fig.text(x+.025,y+height+.022,title,fontsize=8.7,ha='left')
 ashr=fig.add_axes([x+.018,y,width*.52,height]);agap=fig.add_axes([x+width*.66,y,width*.33,height])
 n=len(categories); ys=np.arange(n-1,-1,-1,dtype=float)
 for i,(cat,lab) in enumerate(zip(categories,labels)):
  rr=data[data.category.eq(cat)]
  if len(rr)==0:
   ck(title+' missing class is weather only',cat=='Weather')
   ashr.text(.02,ys[i]-.12,'Not separately reported',transform=ashr.get_yaxis_transform(),fontsize=7,color='#7C888E')
   row={'panel':letter,'profile':title,'category':cat,'status':'not separately reported'}
  else:
   ck(title+' category unique '+cat,len(rr)==1);r=rr.iloc[0]
   ashr.plot([r.F_share_pct,r.H_share_pct],[ys[i],ys[i]],color='#9EAAAF',lw=1.4,zorder=1)
   ashr.scatter(r.F_share_pct,ys[i],s=16,color=BLUE,zorder=3)
   ashr.scatter(r.H_share_pct,ys[i],s=16,marker='s',color=RED,zorder=3)
   lo=float(r.gap_lower);hi=float(r.gap_upper)
   if np.isfinite(lo) and np.isfinite(hi):agap.plot([lo,hi],[ys[i],ys[i]],color=INK,lw=1.2,zorder=2)
   agap.scatter(r.H_gap,ys[i],s=16,marker='D',color=INK,zorder=3)
   row={'panel':letter,'profile':title,'category':cat,'frequency_share_pct':float(r.F_share_pct),'annual_hour_share_pct':float(r.H_share_pct),'signed_gap_h':float(r.H_gap),'gap_lower':lo if np.isfinite(lo) else None,'gap_upper':hi if np.isfinite(hi) else None,'selected_category_view':selected}
  plotted.append(row)
  ashr.text(.015,ys[i]+.22,lab,transform=ashr.get_yaxis_transform(),fontsize=7.7,va='bottom')
 for a in [ashr,agap]:
  a.set_ylim(-.37,n-.35);a.set_yticks([]);a.spines['left'].set_visible(False);a.tick_params(length=2,pad=2)
 ashr.set(xlim=(-3,103),xticks=[0,50,100],xlabel='Share (%)');ashr.xaxis.labelpad=2
 agap.set(xlim=gaplim,xticks=gapticks,xlabel='Gap (h)');agap.xaxis.labelpad=2
 agap.axvline(0,color='#939EA4',ls=(0,(2,3)),lw=.6,zorder=1)
 return ashr,agap

comparison([.035,.453,.286,.144],'e','Brazil · 2024–2025',bplot,['Ordinary unplanned','Emergency / critical'],['Ordinary unplanned','Emergency / critical'],(-12,22),[-10,0,20])
comparison([.365,.453,.286,.144],'f','New Zealand · 2023–2025',pc[pc.profile.eq('New Zealand')],['Weather','Human activity','Remaining causes'],['Weather','Human activity','Remaining causes'],(-.5,6.5),[0,3,6])
comparison([.695,.453,.286,.144],'g','United States · 2013–2024',pc[pc.profile.eq('United States')],['Routine / non-major','Exceptional / major'],['Other days','Major-event days'],(-1.5,4.5),[0,2,4])
# All four Turkish reporting populations are displayed; a common three-category
# selection plus environmental contact aids readability without pooling categories.
fig.text(.035,.398,'Türkiye · 2025',fontsize=9,weight='bold')
fig.text(.300,.398,'Rural − urban hours; selected recorded categories',fontsize=8)
cats=['Weather','Equipment / electrical conditions','Operations / work','Other environmental contact']
labels=['Weather','Equipment / electrical','Operations / work','Other environment']
ha,hg=comparison([.035,.214,.452,.125],'h','A  Ankara–western Black Sea',pc[pc.profile.eq('Türkiye: Ankara–western Black Sea')],cats,labels,(-2,13),[0,5,10],selected=True)
ia,ig=comparison([.535,.214,.452,.125],'i','B  Bolu–Sakarya',pc[pc.profile.eq('Türkiye: Bolu–Sakarya')],cats,labels,(-2,13),[0,5,10],selected=True)
comparison([.035,.020,.452,.120],'j','C  Southern Marmara',pc[pc.profile.eq('Türkiye: Southern Marmara')],cats,labels,(-2,40),[0,20,40],selected=True)
comparison([.535,.020,.452,.120],'k','D  Eastern Mediterranean',pc[pc.profile.eq('Türkiye: Eastern Mediterranean')],cats,labels,(-2,40),[0,20,40],selected=True)

for ax in [ha,hg,ia,ig]: ax.set_xlabel('')
fig.canvas.draw();ren=fig.canvas.get_renderer();texts=[]
for tx in fig.findobj(Text):
 if not tx.get_visible() or not tx.get_text():continue
 b=tx.get_window_extent(ren)
 if b.width>0:texts.append({'text':tx.get_text(),'font_pt':tx.get_fontsize(),'bbox':list(b.extents)})
over=[]
for i,a in enumerate(texts):
 for b in texts[i+1:]:
  aa=a['bbox'];bb=b['bbox'];dx=min(aa[2],bb[2])-max(aa[0],bb[0]);dy=min(aa[3],bb[3])-max(aa[1],bb[1])
  if dx>.5 and dy>.5:over.append({'a':a['text'],'b':b['text'],'pixels':[dx,dy]})
for ext in ['png','pdf','svg']:fig.savefig(F/f'SuppFig20_matched_priority_comparison.{ext}',dpi=600,bbox_inches='tight',pad_inches=.025)
fig.savefig(F/'SuppFig20_matched_priority_comparison_preview.png',dpi=190,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
json.dump({'checks':checks,'text_overlaps':over,'texts':texts,'scale_bars':scale_metadata},open(A/'qa.json','w'),indent=2)
json.dump(manifest,open(A/'input_manifest.json','w'),indent=2)
pd.DataFrame(plotted).to_csv(A/'plotted_values.csv',index=False)
print('Figure exported. Text overlap candidates:',len(over));print(json.dumps(over,indent=2))
