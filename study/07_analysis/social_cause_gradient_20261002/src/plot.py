"""Regional social gradients, with source-resolved denominators and all specifications."""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parents[1];R=P.parents[1]
V=R/'10_manuscript/20261002_global_blackout_inequalities_v14'
sys.path.insert(0,str(V/'src'))
import figure_style as fs
fs.P=P
from figure_style import *
for n in ['figures','source_data']:(P/n).mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10.5,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10})
c=pd.read_parquet(P/'data/analysis_context.parquet')
reg=pd.read_parquet(P/'data/regulatory_stages.parquet')
m=pd.read_csv(P/'data/stage_gradients.csv')
m=m[m.predictor.eq('disadvantage_10pp')]
g=gpd.read_parquet(R/'07_analysis/brazil_detailed_cause_extension_20261002/data/economic_geographic_cohort.geoparquet')
g=g.merge(c[['IdeConjunto','low_income_share_pct']],on='IdeConjunto',validate='one_to_one').to_crs(5880)
w=gpd.read_parquet(R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet')
bra=w[w.ISO3.eq('BRA')].to_crs(5880)
f=page(221)
a=ax(f,15,10,71,58)
bra.plot(ax=a,color='#eceeed',edgecolor='#aeb7bc',lw=.45)
cm=LinearSegmentedColormap.from_list('resident_poverty',['#F1EAF4','#B7A0CC','#734F90','#3C2158'])
norm=plt.Normalize(0,85)
g.plot(column='low_income_share_pct',ax=a,cmap=cm,norm=norm,edgecolor='none',zorder=2)
pt=g.geometry.representative_point();a.scatter(pt.x,pt.y,c=g.low_income_share_pct,cmap=cm,norm=norm,s=5,
    edgecolor='white',lw=.15,zorder=3)
extent(a,bra.total_bounds,71/58,1.03)
geographic_frame(a,5880,longitudes=[-65,-45],latitudes=[-30,-10],fontsize=9.6)
title(f,a,'a','Resident disadvantage',offset=10)
cb(f,cm,norm,[21,76,63,2.5],[0,20,40,60,80],['0','20','40','60','80'],'Low-income residents (%)')

b=ax(f,116,10,88,58)
z=c.merge(reg[reg.category.eq('Internal unplanned')][['IdeConjunto','frequency']],on='IdeConjunto')
denscm=LinearSegmentedColormap.from_list('density',['#CEE3DD','#659C99','#214F68'])
dn=LogNorm(.1,10000)
b.scatter(z.low_income_share_pct,z.frequency,c=z.density,cmap=denscm,norm=dn,s=13,alpha=.72,lw=.2,edgecolor='white')
b.set(xlim=(0,85),yscale='log',ylim=(.3,45),xlabel='Low-income residents (%)',ylabel='Interruptions / customer-year')
b.set_xticks([0,20,40,60,80]);logticks(b,'y',[.5,1,5,10,40]);clean(b)
title(f,b,'b','Internal unplanned interruptions',offset=10)
cb(f,denscm,dn,[131,84,58,2.4],[.1,10,1000],['0.1','10','1,000'],'Customers per km²',extend='both')

specs=['Unadjusted','Density adjusted','State + density']
colors=[GREY,BLUE,RED];markers=['o','D','s'];offsets=[-.24,0,.24]
legend(f,[Line2D([],[],color=co,marker=ma,ms=4,lw=1) for co,ma in zip(colors,markers)],
       ['Observed','+ Density','+ State and density'],106,103,3,10)

def forest(a,domain,categories,metric,labels,left=True,xlim=None):
    for j,cat in enumerate(categories):
        if j%2==0:a.axhspan(j-.48,j+.48,color='#F1F3F3',zorder=0)
        for sp,col,ma,dy in zip(specs,colors,markers,offsets):
            row=m[(m.domain==domain)&(m.category==cat)&(m.metric==metric)&(m.specification==sp)].iloc[0]
            a.errorbar(row.contrast_pct,j+dy,xerr=[[row.contrast_pct-row.lower_pct],[row.upper_pct-row.contrast_pct]],
                fmt=ma,color=col,ms=3.7,lw=1,capsize=1.7,mec='white',mew=.35,zorder=3)
    a.axvline(0,color='#859297',ls=':',lw=.8,zorder=1)
    a.set(ylim=(len(categories)-.5,-.5),xlim=xlim)
    a.set_yticks(range(len(categories)),labels if left else ['']*len(categories));clean(a)
    a.set_xlabel('Change per 10-point rise (%)')

cats=['Internal unplanned','Ordinary unplanned','Emergency / critical']
labels=['All internal\nunplanned','Ordinary','Emergency /\ncritical days']
d=ax(f,45,114,67,31);forest(d,'Customer indices',cats,'Frequency',labels,True,(-28,48))
d.set_xticks([-20,0,20,40]);title(f,d,'c','Customer frequency',offset=40)
e=ax(f,137,114,67,31);forest(e,'Customer indices',cats,'Conditional duration',labels,False,(-28,48))
e.set_xticks([-20,0,20,40]);title(f,e,'d','Customer interruption duration',offset=9)

cats=['Wind','Lightning','Vegetation','Flood','Operational error','Planned']
labels=['Wind','Lightning','Vegetation','Flood','Operating error','Planned']
ff=ax(f,45,167,67,39);forest(ff,'Source records',cats,'Frequency',labels,True,(-80,170))
ff.set_xticks([-50,0,50,100,150]);title(f,ff,'e','Source-record rate',offset=40)
gg=ax(f,137,167,67,39);forest(gg,'Source records',cats,'Conditional duration',labels,False,(-30,60))
gg.set_xticks([-20,0,20,40,60]);title(f,gg,'f','Source-record duration',offset=9)
m.to_csv(P/'source_data/SuppFig13_social_cause_gradients.csv',index=False)
z.drop(columns=[x for x in z if x not in ['IdeConjunto','municipality_code','low_income_share_pct','frequency','density']]).to_csv(
    P/'source_data/SuppFig13_regional_observations.csv',index=False)
g[['IdeConjunto','municipality_code','low_income_share_pct','geometry']].to_parquet(P/'data/social_gradient_map.geoparquet',index=False)
save(f,'SuppFig13_resident_disadvantage_and_interruption')
