"""The global income gap in accumulated outage burden and its exact components."""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figure_style import P,R,BLUE,RED,TEAL,INK,GREY,page,ax,text,title,clean,logticks,cb,legend,save
from map_coordinates import geographic_frame

d=pd.read_parquet(P/'data/global_inequality_synthesis.parquet')
q=pd.read_csv(P/'source_data/Fig07_quartile_metrics.csv')
de=pd.read_csv(P/'source_data/Fig07_burden_decomposition.csv').set_index('metric')
qcolors={'Q1':BLUE,'Q2':'#82A4BA','Q3':'#C78F8C','Q4':RED}
f=page(210)
a=ax(f,14,10,188,69)
w=gpd.read_parquet(R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
w=w[w.ISO3!='ATA'].merge(d[['ISO3','saidi_h']],on='ISO3',how='left',validate='many_to_one')
cm=LinearSegmentedColormap.from_list('annual_burden',['#f8f4eb','#efce9a','#d79772','#b35463','#742e4d'])
norm=LogNorm(.1,1000)
w.plot(column='saidi_h',ax=a,cmap=cm,norm=norm,edgecolor='#aeb6b9',lw=.25,aspect='auto',missing_kwds={'color':'#e6e8e7'})
a.set(xlim=(-180,180),ylim=(-58,82));a.set_aspect('auto')
geographic_frame(a,4326,longitudes=[-120,-60,0,60,120],latitudes=[-30,0,30,60],fontsize=9.2)
title(f,a,'a','Accumulated service inequality · 152 observed economies')
cb(f,cm,norm,[45,87,126,2.2],[.1,1,10,100,1000],['0.1','1','10','100','1,000'],
   'SAIDI (hours per customer-year)',extend='both')

b=ax(f,34,109,66,44)
dc=pd.read_csv(P/'source_data/duration_composition_summary.csv').set_index('region')
order=['China','United States','Europe','Brazil','New Zealand'];mix_colour='#CB973D'
for j,reg in enumerate(order):
    z=dc.loc[reg];pos=neg=0.
    for key,col in [('within_pct_of_high',BLUE),('composition_pct_of_high',mix_colour)]:
        value=z[key];left=pos if value>=0 else neg
        b.barh(j,value,left=left,color=col,height=.55,lw=0)
        if value>=0:pos+=value
        else:neg+=value
    b.scatter(z.gap_pct_of_high,j,marker='D',s=19,color=INK,edgecolor='white',lw=.5,zorder=4)
b.axvline(0,color='#92999c',ls=':',lw=.8,zorder=0)
b.set(xlim=(-40,82),ylim=(4.6,-.6))
b.set_yticks(range(5),['China','United\nStates','Europe','Brazil','New\nZealand'],fontsize=9.3)
b.set_xticks([-40,0,40,80]);b.set_xlabel('Low–high duration gap\n(% of high-quartile duration)',fontsize=9.5)
clean(b);title(f,b,'b','Composition can mask duration gaps',offset=29)

c=ax(f,132,109,69,44)
for metric,color,label in [('saifi',BLUE,'Frequency'),('caidi_h',TEAL,'Duration'),('saidi_h',RED,'Annual burden')]:
    z=q[q.metric==metric].set_index('quartile').reindex(['Q1','Q2','Q3','Q4'])
    xs=np.arange(4);ys=z.relative_to_q4.to_numpy();lo=z.relative_ci_low.to_numpy();hi=z.relative_ci_high.to_numpy()
    c.plot(xs,ys,marker='o',ms=3.8,color=color,lw=1.6)
    c.fill_between(xs,lo,hi,color=color,alpha=.10,lw=0)
c.set(yscale='log',ylim=(.65,150),xlim=(-.15,3.15));c.axhline(1,color='#9fa8ad',lw=.7,ls=':')
c.set_xticks(range(4),['Q1','Q2','Q3','Q4']);logticks(c,'y',[1,10,100])
c.set_xlabel('GDP/person quartile · lower → higher');c.set_ylabel('Relative to highest GDP quartile')
clean(c);title(f,c,'c','Frequency carries the economic gap')
legend(f,[Patch(color=BLUE),Patch(color=mix_colour),Line2D([],[],marker='D',ms=4,color=INK,lw=0)],
       ['Within class','Composition','Net gap'],54,171,3,9.0)
legend(f,[Line2D([],[],color=col,lw=1.8) for col in [BLUE,TEAL,RED]],
       ['Frequency','Duration','Annual burden'],163,166,3,9.0)

dd=ax(f,29,182,80,16)
for j,(metric,col,lab) in enumerate([('saifi',BLUE,'Frequency'),('caidi_h',TEAL,'Duration'),('saidi_h',RED,'Annual burden')]):
    z=de.loc[metric];dd.errorbar(z.ratio,j,xerr=[[z.ratio-z.ci_low],[z.ci_high-z.ratio]],fmt='o',color=col,ms=4.5,capsize=2,lw=1.3)
    dd.text(1.015,j,(f'{z.ratio:.2f}×' if metric=='caidi_h' else f'{z.ratio:.1f}×'),transform=dd.get_yaxis_transform(),color=col,va='center',fontsize=9.4)
dd.set(xscale='log',xlim=(.75,180),ylim=(2.6,-.6));logticks(dd,'x',[1,10,100]);dd.axvline(1,color='#aaa',ls=':',lw=.7)
dd.set_yticks(range(3),['Frequency','Duration','Annual burden'],fontsize=9);dd.set_xlabel('Lowest / highest GDP quartile');clean(dd)
title(f,dd,'d','Decomposing the global burden gap',offset=22)
share=de.loc['saifi','log_gap']/de.loc['saidi_h','log_gap']
e=ax(f,142,189,57,7)
e.barh(0,share,height=.7,color=BLUE);e.barh(0,1-share,left=share,height=.7,color=TEAL)
e.set(xlim=(0,1),ylim=(-.7,.7));e.axis('off')
text(f,171,185,f'{100*share:.1f}%',fontsize=15,color=BLUE,fontweight='bold',ha='center')
text(f,171,201,'of the log burden gap\nis the frequency component',fontsize=9.0,ha='center',va='top')
save(f,'Fig07_global_inequality_decomposition')
