"""Keep explicit source-unit diagnostics outside the country-centred main figures."""
from pathlib import Path
import pandas as pd
from matplotlib.lines import Line2D
import figure_style as fs
from figure_style import BLUE,TEAL,RED,page,ax,title,clean,legend,save
P=Path(__file__).resolve().parents[1];fs.P=P
m=pd.read_csv(P/'inputs/global_paired_income_models.csv')
u=pd.read_csv(P/'upbeat_extension/data/paired_economic_gradients.csv')
s=pd.read_csv(P/'upbeat_extension/data/absolute_gap_replication.csv')
f=page(96)
legend(f,[Line2D([],[],marker='o',color=BLUE,lw=1.3,ms=4),Line2D([],[],marker='s',color=TEAL,lw=1.3,ms=4)],['Frequency','Mean interruption duration'],106,5,2,10)
c=ax(f,53,25,48,54)
rows=[('Benchmarks','Service benchmarks\n152 economies','Continent adjusted'),('Enterprises','Enterprise surveys\n140 economies','Continent adjusted'),('Recent','Recent utilities\n21 in 15 countries','GDP only'),('Recent','Recent utilities\n+ region adjustment','Region adjusted')]
for i,(typ,label,spec) in enumerate(rows):
 if i%2==0:c.axhspan(i-.45,i+.45,color='#F0F3F3',lw=0)
 for off,out,col,mk in [(-.13,'Frequency',BLUE,'o'),(.13,'Duration',TEAL,'s')]:
  if typ=='Recent':z=u[u.specification.eq(spec)&u.outcome.eq(out.lower())].iloc[0]
  else:
   source='Business-city benchmark' if typ=='Benchmarks' else 'Conditional enterprise survey'
   z=m[m.source.eq(source)&m.specification.eq(spec)&m.outcome.eq(out)].iloc[0]
  c.errorbar(z.response_pct,i+off,xerr=[[z.response_pct-z.response_low],[z.response_high-z.response_pct]],fmt=mk,color=col,ms=4,capsize=2,lw=1.2)
c.axvline(0,color='#929B9F',ls=':',lw=.8)
c.set(ylim=(3.55,-.55),xlim=(-82,48),xticks=[-60,-30,0,30],xlabel='Change per GDP doubling (%)')
c.set_yticks(range(4),[r[1] for r in rows],fontsize=9.3);c.tick_params(axis='y',length=0);clean(c)
title(f,c,'a','Economic associations across sources',offset=46)
e=ax(f,150,25,53,54)
z=s[s.comparison.eq('Balanced 2021–2023 · equal country')].iloc[0]
for i,(v,l,h,col) in enumerate([(z.frequency_h,z.frequency_h_low,z.frequency_h_high,BLUE),(z.duration_h,z.duration_h_low,z.duration_h_high,TEAL),(z.gap_h,z.gap_h_low,z.gap_h_high,RED)]):
 e.barh(i,v,height=.45,color=col,alpha=.9)
 e.errorbar(v,i,xerr=[[v-l],[h-v]],fmt='o',mfc='white',mec=col,ecolor=col,ms=3.8,capsize=2,lw=1)
 e.text(v+2,i-.26,f'{v:.1f}',color=col,fontsize=10,va='bottom')
e.axvline(0,color='#929B9F',ls=':',lw=.8)
e.set(ylim=(2.55,-.65),xlim=(-30,80),xticks=[-20,0,40,80],xlabel='Lower − upper group hours')
e.set_yticks([0,1,2],['Frequency','Duration','Net gap'],fontsize=9.6)
clean(e);title(f,e,'b','Recent hour gap is unresolved',offset=28)
save(f,'SuppFig23_recent_cross_source_checks')
