"""Observed cause differences, shared incidents and customer allocation."""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parents[1];R=P.parents[1]
sys.path.insert(0,str(R/'10_manuscript/20261002_global_blackout_inequalities_v14/src'))
import figure_style as fs
fs.P=P
from figure_style import *

f=page(230)
parts=pd.read_csv(P/'data/cause_contrast_and_accounting.csv').query("voltage=='All'").set_index('cause')
summary=pd.read_csv(P/'data/overall_duration_accounting.csv').query("voltage=='All'").iloc[0]
pairs=pd.read_parquet(P/'data/shared_incident_pairs.parquet')
pairs=pairs[(pairs.voltage_scope=='All')&(pairs.start_count==1)]
weather=pairs[pairs.cause.isin(['Ice loading','Wind / lightning'])].copy()
est=pd.read_csv(P/'data/shared_incident_estimates.csv')
inc=pd.read_csv(P/'data/shared_incident_accounting.csv')
inc=inc[(inc.population=='Shared identifier and onset')&(inc.voltage=='All')]

# Annual administrative cause totals. No artificial error bars.
a=ax(f,48,14,59,76)
order=['Ice loading','Wind / lightning','Vegetation contact','Equipment / network fault',
       'Overload','Operations / safety','Animals','Third-party / customer damage',
       'Fire, unspecified origin','Temporary / unspecified fault']
labels=['Ice loading','Wind / lightning','Vegetation','Equipment / network',
        'Overload','Operations / safety','Animals','Third-party damage',
        'Fire, unspecified','Temporary / unknown']
fr=list(parts.loc[order,'frequency_ratio'])+[summary.rural_frequency/summary.urban_frequency]
dr=list(parts.loc[order,'duration_ratio'])+[summary.duration_ratio]
for j in range(11):
    if j%2==0:a.axhspan(j-.45,j+.45,color='#f0f2f2',lw=0,zorder=0)
a.scatter(fr,np.arange(11)-.12,s=20,c=BLUE,edgecolor='white',lw=.4,zorder=3)
a.scatter(dr,np.arange(11)+.12,s=20,c=RED,marker='s',edgecolor='white',lw=.4,zorder=3)
a.axvline(1,color=GREY,ls='--',lw=.8)
a.set(xscale='log',xlim=(.38,16),ylim=(10.65,-.6),xlabel='Rural / urban ratio')
a.set_xticks([.5,1,2,4,8,16],['0.5','1','2','4','8','16']);a.xaxis.set_minor_locator(NullLocator())
a.set_yticks(range(11),labels+['All causes'],fontsize=10)
clean(a);title(f,a,'a','Annual cause contrasts',offset=43)
legend(f,[Line2D([],[],marker='o',color=BLUE,lw=0,ms=5),Line2D([],[],marker='s',color=RED,lw=0,ms=5)],
       ['Frequency','Duration'],77,104,2,10)

# Raw paired evidence, restricted to the pre-specified same-onset population.
a=ax(f,139,14,64,76)
for cause,col,label in [('Wind / lightning',TEAL,'Wind / lightning'),('Ice loading',GOLD,'Ice loading')]:
    w=weather[weather.cause.eq(cause)]
    a.scatter(w.Urban_duration,w.Rural_duration,s=21,color=col,alpha=.67,
              edgecolor='white',lw=.35,label=label,zorder=3)
a.plot([.025,120],[.025,120],color=GREY,ls='--',lw=.9,zorder=1)
a.set(xscale='log',yscale='log',xlim=(.025,120),ylim=(.025,120),
      xlabel='Urban mean duration (h)',ylabel='Rural mean duration (h)')
for axis in ['x','y']:logticks(a,axis,[.1,1,10,100])
assert weather[['Urban_duration','Rural_duration']].ge(.025).all().all()
assert weather[['Urban_duration','Rural_duration']].le(120).all().all()
a.text(.04,.96,'254 shared weather incidents',transform=a.transAxes,fontsize=9.3,va='top')
clean(a);title(f,a,'b','Same interruption and onset',offset=9)
legend(f,[Line2D([],[],marker='o',color=TEAL,lw=0,ms=5),Line2D([],[],marker='o',color=GOLD,lw=0,ms=5)],
       ['Wind / lightning','Ice loading'],170,104,2,9.3)

# Both groups on the same horizontal reference: which incidents bear their CI.
a=ax(f,25,126,73,46)
weather['reference_h']=(weather.Rural_duration+weather.Urban_duration)/2
w=weather.sort_values('reference_h')
for group,col in [('Urban',BLUE),('Rural',RED)]:
    y=100*w[f'{group}_CI'].cumsum()/w[f'{group}_CI'].sum()
    a.step(w.reference_h,y,where='post',lw=1.7,color=col,label=group)
a.set(xscale='log',xlim=(.025,120),ylim=(0,101),xlabel='Shared-incident reference duration (h)',ylabel='Customer interruptions (%)')
assert w.reference_h.between(.025,120).all()
logticks(a,'x',[.1,1,10,100]);a.set_yticks([0,50,100])
a.legend(loc='lower right',fontsize=9.5,handlelength=1.7)
clean(a);title(f,a,'c','Different weights on long incidents',offset=20)

# The primary paired result, all four provinces, and source resolution check.
a=ax(f,143,126,60,46)
primary=est[(est.population=='Shared identifier and onset')&(est.voltage=='All')&(est.scope=='Weather')]
names=['All','BOLU','DÜZCE','KOCAELİ','SAKARYA']
shown=[]
for j,name in enumerate(names):
    z=primary[primary.province.eq(name)].iloc[0]
    a.errorbar(z.ratio,j,xerr=[[z.ratio-z.lower],[z.upper-z.ratio]],fmt='o',
        color=INK if j==0 else TEAL,ms=4.5,capsize=2,lw=1.15,mec='white',mew=.35)
    label={'All':'All provinces','BOLU':'Bolu','DÜZCE':'Düzce','KOCAELİ':'Kocaeli','SAKARYA':'Sakarya'}[name]
    shown.append(label+f' ({z.n})')
z=est[(est.population=='Shared onset, multistage only')&(est.voltage=='All')&
      (est.scope=='Weather')&(est.province=='All')].iloc[0]
a.errorbar(z.ratio,5,xerr=[[z.ratio-z.lower],[z.upper-z.ratio]],fmt='s',color=GOLD,
           ms=4.7,capsize=2,lw=1.15,mec='white',mew=.35)
shown.append(f'Multiple stages ({z.n})')
a.axvline(1,color=GREY,ls='--',lw=.8)
a.set(xlim=(.85,1.1),ylim=(5.65,-.65),xlabel='Paired rural / urban duration ratio')
a.set_xticks([.9,1,1.1],['0.9','1.0','1.1']);a.set_yticks(range(6),shown,fontsize=9.2)
clean(a);title(f,a,'d','Within shared weather incidents',offset=32)

# Exact observed accounting over the unchanged matched incident population.
a=ax(f,47,195,156,23)
for j,scope in enumerate(['All causes','Weather','Other causes']):
    q=inc[inc.scope.eq(scope)].set_index('metric').value
    pos=neg=0
    for metric,col in [('within_incident_h',BLUE),('between_incident_allocation_h',GOLD)]:
        v=q[metric];left=pos if v>=0 else neg
        a.barh(j,v,left=left,color=col,height=.56)
        if v>=0:pos+=v
        else:neg+=v
    a.scatter(q.gap_h,j,marker='D',s=24,color=INK,edgecolor='white',lw=.5,zorder=4)
a.axvline(0,color=GREY,ls=':',lw=.8)
a.set(xlim=(-.17,1.02),ylim=(2.65,-.65),xlabel='Observed rural − urban mean-duration gap (h)')
a.set_xticks([0,.25,.5,.75,1],['0','0.25','0.50','0.75','1.0'])
a.set_yticks(range(3),['All causes','Weather','Other causes'],fontsize=10)
clean(a);title(f,a,'e','Customer allocation and within-incident differences',offset=42)
legend(f,[Patch(color=BLUE),Patch(color=GOLD),Line2D([],[],marker='D',color=INK,lw=0,ms=4)],
       ['Within incident','Allocation across incidents','Net gap'],135,184,3,9.2)

(P/'metadata/figure_contract.json').write_text(json.dumps(dict(
    conclusion='Similar aggregate mean durations coexist with cause-specific differences and unequal allocation of customer interruptions across shared incidents.',
    backend='Python/matplotlib',size_mm=[210,230],primary_population='2025 unannounced sustained interruptions in four Turkish provinces',
    panels={'a':'Complete annual cause contrast','b':'254 shared-onset weather incident pairs','c':'Customer-interruption allocation on common incident-duration axis','d':'Paired contrasts and resolution check','e':'Exact observed group-duration gap accounting'},
    uncertainty='Panel d: onset-date cluster percentile bootstrap, 2000 draws. Panel a and e are observed administrative accounting; date-resampling variability of e is retained in the source table and SI.',
    exports=['600dpi PNG','editable SVG','PDF','PNG preview']),indent=2))
save(f,'SuppFig18_shared_incident_inequality')
