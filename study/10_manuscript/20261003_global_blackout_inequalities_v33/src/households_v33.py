"""Same-population geography and composition of reliable-service inequality."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'core_reproducibility/src'))
from map_coordinates import geographic_frame
BLUE='#376F9F'; RED='#B23B4D'; INK='#283238'; GREY='#899397'; GREEN='#C9DBCF'
plt.rcParams.update({'font.family':'Arial','font.size':11,'axes.labelsize':11,
 'axes.linewidth':.7,'xtick.labelsize':10,'ytick.labelsize':10,
 'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.labelpad':4,'xtick.major.pad':3,'ytick.major.pad':3})
src=P/'source_data/Fig03_same_population_service_components.csv'
d=pd.read_csv(src)
assert len(d)==23 and not d.iso3.duplicated().any()
d['unreliable_connected_gap']=(d.connection_high-d.service_high)-(d.connection_no-d.service_no)
assert np.allclose(d.service_gap,d.connection_gap+d.unreliable_connected_gap)
# Frozen numerical sidecars are read-only in this layout revision.
keys=['connection_gap','unreliable_connected_gap','service_gap']
vals=d[keys].to_numpy()*100
rng=np.random.default_rng(20261002)
boot=vals[rng.integers(0,len(d),size=(4000,len(d)))].mean(1)
summary=pd.DataFrame({'component':keys,'estimate_pp':vals.mean(0),'lower_pp':np.quantile(boot,.025,axis=0),'upper_pp':np.quantile(boot,.975,axis=0)})
# Frozen numerical sidecars are read-only in this layout revision.
states=[]
for suffix,label in [('no','No lived poverty'),('high','High lived poverty')]:
 states.append(dict(group=label,unconnected=100*(1-d['connection_'+suffix]).mean(),
  connected_unreliable=100*(d['connection_'+suffix]-d['service_'+suffix]).mean(),
  connected_reliable=100*d['service_'+suffix].mean()))
st=pd.DataFrame(states);assert np.allclose(st.iloc[:,1:].sum(axis=1),100)
# Frozen numerical sidecars are read-only in this layout revision.
W=gpd.read_parquet(P/'core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
W=W[W.ISO3.ne('ATA')]
z=W.merge(d,left_on='ISO3',right_on='iso3',how='left',validate='many_to_one')
z['gap_pp']=z.service_gap*100

fig=plt.figure(figsize=(9.2,8.0),dpi=160)
def panel(rect,letter,title,title_x=None):
 a=fig.add_axes(rect)
 label_x=rect[0] if title_x is None else title_x
 fig.text(label_x-.035,rect[1]+rect[3]+.014,letter,weight='bold',fontsize=13,va='bottom')
 fig.text(label_x,rect[1]+rect[3]+.014,title,fontsize=11.6,va='bottom')
 return a
def clean(a):
 a.spines[['top','right']].set_visible(False);a.tick_params(length=3,width=.7)

# A single observed outcome links the map and country profiles.
a=panel([.062,.406,.427,.547],'a','Reliable-service disadvantage')
W.plot(ax=a,color='#EBEEEC',edgecolor='#B9C0BD',linewidth=.4,aspect='auto')
cm=LinearSegmentedColormap.from_list('povertygap',['#F8EAE4','#DDB1A0','#C36871','#7C294A']);norm=Normalize(0,40)
z[z.gap_pp.notna()].plot(ax=a,column='gap_pp',cmap=cm,norm=norm,edgecolor='white',linewidth=.5,aspect='auto')
for iso,x,y in [('CPV',-23.7,15.1),('STP',6.6,.3),('SYC',55.5,-4.7)]:
 row=d[d.iso3.eq(iso)].iloc[0]
 a.scatter(x,y,s=38,c=[row.service_gap*100],cmap=cm,norm=norm,edgecolor=INK,linewidth=.6,zorder=5)
a.set(xlim=(-27,60),ylim=(-36,39));a.set_aspect('auto')
geographic_frame(a,4326,longitudes=[-20,0,20,40,60],latitudes=[-30,-15,0,15,30],fontsize=9.5)
cax=fig.add_axes([.092,.360,.370,.014])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation='horizontal',ticks=[0,10,20,30,40])
cb.ax.set_xlabel('Reliable-service gap (pp)',fontsize=10.5,labelpad=3)

order=d.sort_values('service_gap',ascending=False).reset_index(drop=True)
b=panel([.702,.406,.281,.547],'b','Reliable service')
for i,r in order.iterrows():
 b.plot([r.service_high*100,r.service_no*100],[i,i],color='#C6CDCB',lw=1.2,zorder=1)
 b.scatter(r.service_high*100,i,s=27,color=RED,zorder=3,edgecolors='white',linewidths=.35)
 b.scatter(r.service_no*100,i,s=27,color=BLUE,zorder=3,edgecolors='white',linewidths=.35)
labels=order.country.replace({'São Tomé and Príncipe':'São Tomé & Príncipe',"Côte d'Ivoire":"Côte d’Ivoire"})
b.set_yticks(range(len(d)),labels,fontsize=10.2);b.tick_params(axis='y',length=0,pad=6)
b.set(xlim=(-2,102),ylim=(len(d)-.35,-.75),xlabel='Respondents (%)');b.set_xticks([0,25,50,75,100]);clean(b)
b.grid(axis='x',color='#E8EBE9',lw=.45,zorder=0)
fig.legend([Line2D([],[],marker='o',color=BLUE,ls='',ms=5),Line2D([],[],marker='o',color=RED,ls='',ms=5)],
 ['No lived poverty','High lived poverty'],loc='upper left',bbox_to_anchor=(.702,.342),bbox_transform=fig.transFigure,fontsize=9.3,ncol=2,borderaxespad=0,columnspacing=.55,handlelength=.75,handletextpad=.3)

c=panel([.165,.055,.324,.202],'c','Service states',title_x=.062)
for i,r in st.iterrows():
 left=0
 for key,color in [('unconnected',BLUE),('connected_unreliable',RED),('connected_reliable',GREEN)]:
  value=r[key];c.barh(i,value,left=left,color=color,height=.51,edgecolor='white',linewidth=.55)
  if value>=7:
   c.text(left+value/2,i,f'{value:.1f}',ha='center',va='center',fontsize=10.2,color='white' if color!=GREEN else INK)
  left+=value
c.set_yticks([0,1],['No lived poverty','High lived poverty']);c.set(xlim=(0,100),ylim=(1.6,-1.45),xlabel='Respondents (%)');c.set_xticks([0,50,100]);clean(c);c.tick_params(axis='y',length=0)
c.legend([Patch(color=BLUE),Patch(color=RED),Patch(color=GREEN)],['Unconnected','Connected,\nunreliable','Reliable'],
 loc='upper center',bbox_to_anchor=(.5,1.0),ncol=3,fontsize=8.8,columnspacing=.8,handlelength=.8,handletextpad=.35,borderaxespad=.2)

e=panel([.635,.055,.348,.202],'d','Sources of the service gap')
colors=[BLUE,RED,INK]
rg=np.random.default_rng(183)
for j,key in enumerate(keys):
 e.scatter(j+rg.uniform(-.16,.16,len(d)),d[key]*100,s=16,color=colors[j],alpha=.4,edgecolor='none',zorder=2)
 r=summary.iloc[j]
 e.errorbar(j,r.estimate_pp,yerr=[[r.estimate_pp-r.lower_pp],[r.upper_pp-r.estimate_pp]],fmt='o',color=INK,
  markerfacecolor='white',markersize=5.5,markeredgewidth=1.2,lw=2.2,capsize=3,zorder=4)
e.axhline(0,color=GREY,lw=.7,ls=(0,(3,3)))
e.set(ylim=(-20,44),xlim=(-.48,2.48),ylabel='Difference (pp)')
e.set_yticks([-20,0,20,40]);e.set_xticks(range(3),['Unconnected','Connected,\nunreliable','Total'],fontsize=10);clean(e)

stem='Fig05_household_reliable_service'
for artist in fig.findobj(Text):
 artist.set_fontsize(artist.get_fontsize()*1.12)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
outside=[t.get_text() for t,bx in texts if bx.x0<-.5 or bx.y0<-.5 or bx.x1>fig.bbox.width+.5 or bx.y1>fig.bbox.height+.5]
overlap=[]
for i,(t,bx) in enumerate(texts):
 for u,by in texts[i+1:]:
  if min(bx.x1,by.x1)-max(bx.x0,by.x0)>2 and min(bx.y1,by.y1)-max(bx.y0,by.y0)>2:
   overlap.append([t.get_text(),u.get_text()])
for ext in ['png','pdf','svg']:fig.savefig(P/'figures'/f'{stem}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.025)
fig.savefig(P/'qa'/f'{stem}_preview.png',dpi=180,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
cap='Fig. 5. Poverty-related differences in reliable electricity persist within areas reached by the grid. (a) Difference in the percentage of respondents reporting reliable grid service between groups with no and high lived poverty in 23 African countries surveyed during 2021–2023. All panels use the same country-specific populations in enumeration areas where interviewers reported that most houses could access an electricity grid; grey countries have no admitted estimate. Small islands are represented by coloured points. (b) Survey-weighted reliable-service percentages in the two poverty groups, ordered by their difference. Reliability means supply most or all of the time. (c) Equal-country mean prevalence of three exhaustive service states in the same populations. A household connection and dependable supply are distinct outcomes. (d) Country-specific contributions to the reliable-service gap (small points) and equal-country mean differences (open points), with 95% percentile intervals from 4,000 paired whole-country bootstrap resamples. The unconnected and connected-but-unreliable components sum to the total gap in each country; negative components offset the difference. These intervals describe variation across admitted countries rather than full survey-design uncertainty. Lived poverty measures reported shortages of basic necessities and is not a monetary poverty threshold. pp, percentage points. Source: Afrobarometer Round 9.'
meta={'figure':5,'stem':stem,'caption':cap,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
 'n_countries':len(d),'summary':summary.to_dict('records'),'all_service_gaps_positive':bool(d.service_gap.gt(0).all()),
 'identity_max_error':float(abs(d.service_gap-d.connection_gap-d.unreliable_connected_gap).max()),
 'outside_canvas':outside,'text_overlap_candidates':overlap,'minimum_font_pt':min(t.get_fontsize() for t,bx in texts),
 'bootstrap':'4000 whole-country paired draws, seed 20261002; equal countries; same as existing exact partition'}
(P/'metadata/fig05_v33.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
