"""Compact scientific figure from audited observations, with editable vector exports."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

P=Path(__file__).resolve().parents[1];ROOT=P.parents[1]
D=P/'data';(P/'figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':9.6,'axes.labelsize':9.6,
    'xtick.labelsize':9.1,'ytick.labelsize':9.1,'axes.linewidth':.65,
    'axes.labelpad':3,'xtick.major.pad':2,'ytick.major.pad':2,
    'legend.frameon':False,'legend.fontsize':9.1,'pdf.fonttype':42,'svg.fonttype':'none'})
ENV='#278C88';OTHER='#64778A';PLANNED='#D3A34C';INK='#283238'
z=pd.read_parquet(D/'burden_economic_common_2021.parquet')
geo=ROOT/'07_analysis/economic_hazard_strengthening_20260928/components/brazil_panel/data'
g=gpd.read_parquet(geo/'service_area_economic_context_2021.geoparquet')[['IdeConjUndConsumidoras','geometry']].to_crs(4326)
g=g.rename(columns={'IdeConjUndConsumidoras':'conjunto'}).merge(
    z[['conjunto','Environment-associated','All causes']],on='conjunto',validate='one_to_one')
assert len(g)==len(z)
g['share']=100*g['Environment-associated']/g['All causes']
mun=gpd.read_parquet(geo/'municipalities_brazil_2021.geoparquet').to_crs(4326)
mun['state']=mun.municipality_code.astype(str).str[:2]
# Repair only the cartographic background before union; analysis areas are unchanged.
mun.geometry=mun.geometry.make_valid()
states=mun.dissolve(by='state')
W,H=190,185
fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=160)
def ax(x,top,w,h):return fig.add_axes([x/W,(H-top-h)/H,w/W,h/H])
def txt(x,y,s,**kw):return fig.text(x/W,1-y/H,s,**kw)
def title(a,letter,s,dx=7):
    b=a.get_position();txt(b.x0*W-dx,(1-b.y1)*H-2,letter,fontweight='bold',fontsize=11.7,va='bottom')
    txt(b.x0*W,(1-b.y1)*H-2,s,fontsize=10.1,va='bottom')
def clean(a):
    a.spines[['top','right']].set_visible(False);a.tick_params(length=3,width=.65)

a=ax(12,9,81,82)
states.plot(ax=a,facecolor='#ECEEEE',edgecolor='white',linewidth=.35)
cm=LinearSegmentedColormap.from_list('environment',['#edf5f3','#add1c8','#5daba1','#197c7c','#074b55'])
g.plot(ax=a,column='share',cmap=cm,norm=Normalize(0,100),edgecolor='none',zorder=3)
states.boundary.plot(ax=a,color='#adb5b8',linewidth=.3,zorder=4)
a.set(xlim=(-74,-34),ylim=(-34,6));a.set_aspect('equal',adjustable='box')
a.set_xticks([-70,-50,-35],['70°W','50°W','35°W'])
a.set_yticks([-30,-15,0],['30°S','15°S','0°'])
a.grid(color='#c3cbcd',lw=.45,ls=(0,(3,3)),zorder=0)
a.tick_params(direction='out',length=3,width=.6)
title(a,'a','Environmental share of lost service')
ca=ax(24,95,61,2.8)
cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,100),cmap=cm),cax=ca,
               orientation='horizontal',ticks=[0,25,50,75,100])
cb.ax.tick_params(labelsize=9,length=2);cb.set_label('Customer-hours (%)',fontsize=9.6,labelpad=2)

b=ax(110,9,76,64)
q=pd.read_csv(D/'quartile_burden_components.csv')
q=q[q.weighting.eq('Equal service areas')].pivot(index='quartile',columns='cause',values='hours')
bottom=np.zeros(4)
comps=['Environment-associated','Planned','Remaining unplanned or uncertain']
colors=[ENV,PLANNED,OTHER]
for cause,col in zip(comps,colors):
    b.bar(np.arange(4),q.loc[['Q1','Q2','Q3','Q4'],cause],bottom=bottom,width=.65,
          color=col,edgecolor='white',linewidth=.4)
    bottom+=q.loc[['Q1','Q2','Q3','Q4'],cause].to_numpy()
b.set_xticks(range(4),['Q1','Q2','Q3','Q4'])
b.set(xlabel='Regional GDP quartile',ylabel='Hours per customer-year',ylim=(0,19.8))
b.set_yticks([0,5,10,15]);clean(b);title(b,'b','Absolute burden by economic group')
for i,h in enumerate(bottom):b.text(i,h+.4,f'{h:.1f}',ha='center',va='bottom',fontsize=9)
b.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,['Environmental','Planned','Other / uncertain'])],
    loc='upper left',bbox_to_anchor=(-.03,-.21),ncol=1,handlelength=1.25,
    labelspacing=.35,columnspacing=.8,borderpad=0)

c=ax(48,127,137,43)
models=pd.read_csv(D/'economic_burden_models.csv')
models=models[models['sample'].eq('Primary common support')]
specs=['Economic context','Density adjusted','Within provider + density']
scols=['#355F8D','#B64458','#248F88']
outcomes=['All causes','Environment-associated','Planned','Remaining unplanned or uncertain']
labels=['All causes','Environmental','Planned','Other / uncertain']
for k,(spec,col,marker) in enumerate(zip(specs,scols,['o','s','D'])):
    r=models[models.specification.eq(spec)].set_index('outcome').loc[outcomes]
    yy=np.arange(4)+(k-1)*.20
    c.errorbar(r.contrast_pct,yy,xerr=np.array([r.contrast_pct-r.lower_pct,r.upper_pct-r.contrast_pct]),
         fmt=marker,color=col,ms=4.4,lw=1.25,capsize=2,mec='white',mew=.45,zorder=3)
c.set_yticks(range(4),labels);c.set(ylim=(3.55,-.55),xlim=(-57,43),xlabel='GDP interquartile contrast in annual burden (%)')
c.set_xticks([-50,-25,0,25]);c.axvline(0,color='#7E898E',lw=.7,ls=(0,(3,3)),zorder=1)
clean(c)
txt(6,111,'c',fontweight='bold',fontsize=11.7,va='bottom')
txt(12,111,'Economic contrasts across cause components',fontsize=10.1,va='bottom')
fig.legend(handles=[Line2D([0],[0],marker=m,color=col,lw=1.15,ms=4.5,label=l) for col,m,l in
    zip(scols,['o','s','D'],['Unadjusted','+ Density','+ Density and service context'])],
    loc='center',bbox_to_anchor=(.595,1-119/H),ncol=3,handlelength=1.25,
    columnspacing=.9,handletextpad=.45,borderpad=0)

fig.canvas.draw();renderer=fig.canvas.get_renderer()
items=[]
for text in fig.findobj(Text):
    if text.get_visible() and text.get_text().strip():
        box=text.get_window_extent(renderer)
        if box.width>.1 and box.height>.1:items.append((text,box))
collisions=[]
for i,(t,bb) in enumerate(items):
    for u,cc in items[i+1:]:
        dx=min(bb.x1,cc.x1)-max(bb.x0,cc.x0);dy=min(bb.y1,cc.y1)-max(bb.y0,cc.y0)
        if dx>1.6 and dy>1.6:collisions.append([t.get_text(),u.get_text()])
outside=[t.get_text() for t,bx in items if bx.x0<0 or bx.y0<0 or bx.x1>fig.bbox.width or bx.y1>fig.bbox.height]
qa=dict(minimum_font_pt=min(t.get_fontsize() for t,bx in items),
    text_overlap_candidates=collisions,outside_canvas=outside,mapped_areas=len(g))
stem='SuppFig11_brazil_cause_economic_context'
for ext in ['png','pdf','svg']:fig.savefig(P/'figures'/f'{stem}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.025)
fig.savefig(P/'figures'/f'{stem}_preview.png',dpi=180,bbox_inches='tight',pad_inches=.025)
(P/'metadata/figure_layout.json').write_text(json.dumps(qa,indent=2))
plt.close(fig);print(json.dumps(qa,indent=2))
