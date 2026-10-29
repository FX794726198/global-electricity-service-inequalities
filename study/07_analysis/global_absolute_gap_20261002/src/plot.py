"""Rebuild the global synthesis using the established Python figure workflow."""
from pathlib import Path
import sys, hashlib, json, shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D

P=Path(__file__).resolve().parents[1];ROOT=P.parents[1]
OLD=ROOT/'10_manuscript/20261002_global_blackout_inequalities_v15'
sys.path.insert(0,str(OLD/'src'))
import figure_style as fs
fs.P=P;fs.R=ROOT
from figure_style import BLUE,TEAL,RED,INK,GREY,page,ax,text,title,clean,logticks,cb,legend,save
from map_coordinates import geographic_frame

d=pd.read_csv(P/'source_data/country_benchmarks.csv')
q=pd.read_csv(OLD/'source_data/Fig07_quartile_metrics.csv')
m=pd.read_csv(OLD/'source_data/global_paired_income_models.csv')
ar=pd.read_csv(P/'data/arithmetic_group_summary.csv')
de=pd.read_csv(P/'data/absolute_decomposition.csv').set_index('metric')
f=page(223)
a=ax(f,13,9,190,64)
w=gpd.read_parquet(ROOT/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
w=w[w.ISO3.ne('ATA')].merge(d[['ISO3','saidi_h']],on='ISO3',how='left',validate='many_to_one')
cm=LinearSegmentedColormap.from_list('annual_burden',['#f8f4eb','#efce9a','#d79772','#b35463','#742e4d'])
norm=LogNorm(.1,1000)
w.plot(column='saidi_h',ax=a,cmap=cm,norm=norm,edgecolor='#aeb6b9',lw=.22,aspect='auto',missing_kwds={'color':'#e6e8e7'})
a.set(xlim=(-180,180),ylim=(-58,82));a.set_aspect('auto')
geographic_frame(a,4326,longitudes=[-120,-60,0,60,120],latitudes=[-30,0,30,60],fontsize=9.2)
title(f,a,'a','Annual interruption burden · 152 economy benchmarks')
cb(f,cm,norm,[50,81,116,2.1],[.1,1,10,100,1000],['0.1','1','10','100','1,000'],
   'Hours without supply per customer-year',extend='both')

b=ax(f,19,103,76,43)
for metric,color in [('saifi',BLUE),('caidi_h',TEAL),('saidi_h',RED)]:
    z=q[q.metric.eq(metric)].set_index('quartile').reindex(['Q1','Q2','Q3','Q4'])
    xs=np.arange(4);ys=z.relative_to_q4.to_numpy()
    b.plot(xs,ys,marker='o',ms=3.6,color=color,lw=1.5)
    b.fill_between(xs,z.relative_ci_low,z.relative_ci_high,color=color,alpha=.09,lw=0)
b.set(yscale='log',ylim=(.7,150),xlim=(-.16,3.16));b.axhline(1,color='#9fa8ad',lw=.65,ls=':')
b.set_xticks(range(4),['Q1','Q2','Q3','Q4']);logticks(b,'y',[1,10,100])
b.set_xlabel('GDP/person quartile · lower → higher',fontsize=9.5)
b.set_ylabel('Geometric mean relative to Q4',fontsize=9.5)
clean(b);title(f,b,'b','Relative economic differences',offset=12)

c=ax(f,149,103,54,43)
for i,(source,label) in enumerate([('Business-city benchmark','Service\nbenchmarks\n(n = 152)'),('Conditional enterprise survey','Enterprises\n(n = 140)')]):
    c.axhspan(i-.36,i+.36,color='#f2f4f4',lw=0,zorder=0)
    for offset,outcome,color in [(-.14,'Frequency',BLUE),(.14,'Duration',TEAL)]:
        z=m[m.source.eq(source)&m.specification.eq('Continent adjusted')&m.outcome.eq(outcome)].iloc[0]
        c.errorbar(z.response_pct,i+offset,xerr=[[z.response_pct-z.response_low],[z.response_high-z.response_pct]],fmt='o',color=color,ms=4,capsize=2,lw=1.3)
c.axvline(0,color='#989fa3',ls=':',lw=.8)
c.set(xlim=(-76,18),ylim=(1.45,-.45),xticks=[-60,-30,0]);c.set_yticks([0,1],['Service benchmarks\n(n = 152)','Enterprises\n(n = 140)'],fontsize=9)
c.set_xlabel('Change per GDP doubling (%)',fontsize=9.4)
clean(c);title(f,c,'c','Independent economic gradients',offset=38)

legend(f,[Line2D([],[],color=cc,lw=1.6) for cc in [BLUE,TEAL,RED]],
       ['Frequency','Duration','Annual burden'],106,161,3,9.5)

dd=ax(f,19,178,76,34)
rng=np.random.default_rng(20261002)
for j,quartile in enumerate(['Q1','Q2','Q3','Q4']):
    z=d[d.gdp_quartile.eq(quartile)]
    dd.scatter(j+rng.uniform(-.18,.18,len(z)),z.saidi_h,s=7,c='#c4b1b6',alpha=.75,edgecolors='none',zorder=2)
z=ar[ar.metric.eq('annual_hours')].set_index('quartile').reindex(['Q1','Q2','Q3','Q4'])
dd.errorbar(np.arange(4),z.estimate,yerr=[z.estimate-z.ci_low,z.ci_high-z.estimate],fmt='D-',ms=4.3,lw=1.3,color=RED,capsize=2.5,zorder=4)
dd.set(yscale='log',ylim=(.008,3500),xlim=(-.4,3.4));logticks(dd,'y',[.01,1,100,1000])
dd.set_xticks(range(4),['Q1','Q2','Q3','Q4']);dd.set_xlabel('GDP/person quartile · lower → higher',fontsize=9.5)
dd.set_ylabel('Annual hours per customer',fontsize=9.5);clean(dd)
title(f,dd,'d','Absolute burden and its distribution',offset=17)

e=ax(f,132,178,71,34)
values=[de.loc['frequency_component_h','estimate'],de.loc['duration_component_h','estimate'],de.loc['annual_gap_h','estimate']]
bottom=[0,values[0],0];colors=[BLUE,TEAL,RED]
for j,(v,base,col) in enumerate(zip(values,bottom,colors)):
    e.bar(j,v,bottom=base,width=.56,color=col,lw=0,zorder=2)
    e.text(j,base+v+3,f'{v:.1f}' if j!=1 else f'+{v:.1f}',ha='center',va='bottom',fontsize=9.6,color=col)
e.plot([.28,.72],[values[0]]*2,color='#9aa2a5',lw=.7)
e.plot([1.28,1.72],[values[2]]*2,color='#9aa2a5',lw=.7)
e.set(xlim=(-.5,2.5),ylim=(0,160),yticks=[0,50,100,150])
e.set_xticks([0,1,2],['Frequency','Duration','Total gap'],fontsize=9.0)
e.set_ylabel('Q1 − Q4 annual hours',fontsize=9.5);clean(e)
title(f,e,'e','Frequency accounts for 72% of the gap',offset=19)

for file in ['arithmetic_group_summary.csv','absolute_decomposition.csv']:
    shutil.copy2(P/'data'/file,P/'source_data'/file)
q.to_csv(P/'source_data/relative_quartile_metrics.csv',index=False)
m.to_csv(P/'source_data/paired_income_models.csv',index=False)
save(f,'Fig06_global_burden_and_inequality')
manifest={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'source_data':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'source_data').glob('*.csv') if not p.name.startswith('._')},
          'backend':'Python matplotlib; established manuscript workflow',
          'main_figure':6,'dimensions_mm':[210,223]}
(P/'metadata/figure_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
