"""Definition-support map and retained economic contrasts, without extrapolation."""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parents[1];R=P.parents[1]
sys.path.insert(0,str(R/'10_manuscript/20261002_global_blackout_inequalities_v14/src'))
import figure_style as fs
fs.P=P
from figure_style import *

contract=dict(conclusion='The stronger frequency gradient persists within matched reporting-threshold subsets.',
    panels={'a':'Verified minimum-duration definition geography','b':'Paired GDP gradients within definition subsets','c':'Frequency contribution to two burden-gap estimands'},
    backend='Python/matplotlib',size_mm=[210,178],exports=['PNG 600 dpi','PDF','SVG','PNG preview'],
    uncertainty='HC3 Student t intervals in b; c describes fixed observed group accounting.')
(P/'metadata/definition_figure_contract.json').write_text(json.dumps(contract,indent=2))
d=pd.read_csv(P/'data/definition_linkage.csv');ad=d[d.definition_admitted].copy()
W=gpd.read_parquet(R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
def label(x):
    if x==0:return '0'
    if x<=1:return '≤1'
    if x<3:return '2'
    if x==3:return '3'
    if x<=5:return '4–5'
    return '>5'
ad['threshold']=ad.minimum_minutes.map(label)
classes=['0','≤1','2','3','4–5','>5'];cols=['#e8efed','#bcd8d2','#92bfb6','#609e9b','#3e7785','#244b68']
f=page(178);a=ax(f,14,11,151,68)
W.plot(ax=a,color='#E3E5E5',edgecolor='#AEB5B8',lw=.23)
for key,col in zip(classes,cols):
    z=W[W.ISO3.isin(ad.loc[ad.threshold.eq(key),'ISO3'])]
    if len(z):z.plot(ax=a,color=col,edgecolor='#AEB5B8',lw=.23)
a.set(xlim=(-180,180),ylim=(-59,84));a.set_aspect('equal',adjustable='box')
geographic_frame(a,4326,longitudes=[-120,-60,0,60,120],latitudes=[-30,0,30,60],fontsize=9.5)
title(f,a,'a','Matched reporting thresholds across 132 benchmarks',offset=9)
text(f,171,17,'Minimum outage',fontsize=10,va='top');text(f,171,22,'time (minutes)',fontsize=10,va='top')
for i,(key,col) in enumerate(zip(classes,cols)):
    y=33+i*6
    f.patches.append(Rectangle((171/210,1-(y+3)/178),7/210,3/178,transform=f.transFigure,facecolor=col,edgecolor='none'))
    text(f,181,y+1.5,key,fontsize=10,va='center')

m=pd.read_csv(P/'data/benchmark_definition_models.csv');s=pd.read_csv(P/'data/benchmark_definition_sensitivity.csv')
names=s.comparison.tolist();labels=['Original 152','2019 label','Matched reports','3 min threshold','5 min threshold']
a=ax(f,47,101,61,62)
for i,name in enumerate(names):
    if i%2==0:a.axhspan(i-.48,i+.48,color='#F1F3F3',lw=0)
    for off,out,col in [(-.13,'Frequency',BLUE),(.13,'Duration',TEAL)]:
        z=m[(m.comparison==name)&(m.outcome==out)].iloc[0]
        a.errorbar(z.change_pct,i+off,xerr=[[z.change_pct-z.change_low],[z.change_high-z.change_pct]],
            fmt='o',color=col,ms=4.3,lw=1.1,capsize=2,mec='white',mew=.4)
a.axvline(0,color=GREY,ls='--',lw=.8)
a.set(xlim=(-85,25),ylim=(4.7,-.7),xlabel='Change per GDP doubling (%)')
a.set_xticks([-80,-40,0,20]);a.set_yticks(range(5),labels,fontsize=10)
title(f,a,'b','Economic gradients',offset=42);clean(a)
legend(f,[Line2D([],[],marker='o',color=BLUE,ms=4),Line2D([],[],marker='o',color=TEAL,ms=4)],['Frequency','Duration'],78,93,2,9.5)

a=ax(f,140,101,64,62)
for i,z in enumerate(s.itertuples()):
    if i%2==0:a.axhspan(i-.48,i+.48,color='#F1F3F3',lw=0)
    a.plot([z.frequency_share,z.log_frequency_share],[i,i],color='#bec6c9',lw=1.1)
    a.scatter(z.frequency_share,i,s=27,c=BLUE,marker='s',edgecolors='white',lw=.4,zorder=3)
    a.scatter(z.log_frequency_share,i,s=27,c=RED,edgecolors='white',lw=.4,zorder=3)
a.axvline(50,color=GREY,ls='--',lw=.8)
a.set(xlim=(45,100),ylim=(4.7,-.7),xlabel='Frequency contribution (%)')
a.set_yticks(range(5),[f'{r.n_Q1} / {r.n_Q4}' for r in s.itertuples()],fontsize=10)
a.set_xticks([50,75,100]);a.set_ylabel('Q1 / Q4 observations',fontsize=9.5)
title(f,a,'c','Relative and absolute gaps',offset=18);clean(a)
legend(f,[Line2D([],[],marker='s',color=BLUE,lw=0,ms=4),Line2D([],[],marker='o',color=RED,lw=0,ms=4)],['Absolute hours','Log gap'],169,93,2,9.3)
save(f,'SuppFig19_benchmark_definition_support')
