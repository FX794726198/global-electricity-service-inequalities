"""Source-resolved regional persistence figures using the established Python style."""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parents[1]
ROOT=P.parents[1]
V=ROOT/'10_manuscript/20261002_global_blackout_inequalities_v14'
sys.path.insert(0,str(V/'src'))
import figure_style as fs
fs.P=P
from figure_style import *
from matplotlib.colors import FuncNorm

for sub in ['figures','source_data']: (P/sub).mkdir(exist_ok=True)
W=gpd.read_parquet(ROOT/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet')
RAT=pd.read_csv(P/'data/within_area_duration_ratios.csv')
EST=pd.read_csv(P/'data/within_area_duration_contrasts.csv')
G=gpd.read_parquet(P/'data/economic_geographic_cohort.geoparquet')

def national_base(a,iso,crs):
    z=W[W.ISO3.eq(iso)].to_crs(crs)
    z.plot(ax=a,color='#eceeed',edgecolor='#aeb7bc',lw=.45)
    return z

def main():
    f=page(152)
    cm=LinearSegmentedColormap.from_list('paired_ratio',['#AE7139','#E8C892','#F6F6F2','#8FB5CB','#285986'])
    norm=LogNorm(.0625,16)
    b=RAT[(RAT.numerator=='Lightning') & (RAT.denominator=='Operational error')]
    bg=G.merge(b,on='IdeConjunto',validate='one_to_one').to_crs(5880)
    a=ax(f,13,10,65,69);base=national_base(a,'BRA',5880)
    bg.plot(column='ratio',ax=a,cmap=cm,norm=norm,edgecolor='none',zorder=2)
    # Representative points improve visibility of small sampled polygons.
    points=bg.geometry.representative_point()
    a.scatter(points.x,points.y,c=bg.ratio,cmap=cm,norm=norm,s=5,lw=.12,edgecolor='white',zorder=3)
    extent(a,base.total_bounds,65/69,1.01)
    geographic_frame(a,5880,longitudes=[-65,-45],latitudes=[-30,-15,0],fontsize=9)
    title(f,a,'a','Brazil · 593 areas',offset=9)

    previous=pd.read_parquet(V/'data/cause_persistence_unit_ratios.parquet')
    previous=previous[previous.cause.eq('Lightning')].copy()
    ny=previous[previous.source.eq('New York')]
    nyg=gpd.read_parquet(V/'data/ny_ng_electric_service_regions.parquet').to_crs(5070)
    nyg=nyg.merge(ny[['system','ratio']],left_on='Region',right_on='system',validate='one_to_one')
    aa=ax(f,97,10,61,43)
    us=gpd.read_parquet(V/'data/us_same49_map.parquet').to_crs(5070)
    us[us.state.eq('NY')].plot(ax=aa,color='#eceeed',edgecolor='#aeb7bc',lw=.45)
    nyg.plot(column='ratio',ax=aa,cmap=cm,norm=norm,edgecolor='white',lw=.4,zorder=2)
    extent(aa,us[us.state.eq('NY')].total_bounds,61/43,1.03)
    geographic_frame(aa,5070,longitudes=[-79,-75],latitudes=[42,44],fontsize=9)
    title(f,aa,'b','New York · 8 regions',offset=9)

    nz=previous[previous.source.eq('New Zealand')]
    nzg=gpd.read_parquet(V/'data/nz_explicit21_map.parquet').to_crs(2193)
    nzg=nzg.merge(nz[['system','ratio']],on='system',validate='one_to_one')
    aaa=ax(f,173,10,32,69);national_base(aaa,'NZL',2193)
    nzg.plot(column='ratio',ax=aaa,cmap=cm,norm=norm,edgecolor='white',lw=.35,zorder=2)
    extent(aaa,nzg.total_bounds,32/69,1.07)
    geographic_frame(aaa,2193,longitudes=[170,176],latitudes=[-45,-40],fontsize=9)
    title(f,aaa,'c','New Zealand',offset=8)
    cb(f,cm,norm,[100,66,55,2.4],[.0625,1,16],['1/16','1','16'],
       'Lightning / operating-error\nmean duration',extend='both')

    summ=pd.read_parquet(V/'data/cause_persistence_summary.parquet')
    br=EST.query("numerator=='Lightning' & denominator=='Operational error'").iloc[0]
    r=pd.concat([previous.assign(unit=previous.system)[['source','unit','ratio']],
        pd.DataFrame({'source':'Brazil','unit':b.IdeConjunto.astype(str),'ratio':b.ratio})],ignore_index=True)
    sumrows=[]
    for country in ['Brazil','New York','New Zealand']:
        if country=='Brazil':lo,hi,point,n=br.lower,br.upper,br.geometric_ratio,int(br.areas)
        else:
            s=summ[(summ.source==country)&(summ.cause=='Lightning')].iloc[0]
            lo,hi,point,n=s.ci_low,s.ci_high,s.geometric_ratio,int(s.systems)
        sumrows.append(dict(source=country,ratio=point,lower=lo,upper=hi,areas=n))
    sd=pd.DataFrame(sumrows)
    d=ax(f,39,103,61,32)
    for j,(country,col) in enumerate(zip(sd.source,[TEAL,RED,BLUE])):
        rr=r[r.source.eq(country)].sort_values('unit')
        yy=np.random.default_rng(2026+j).uniform(-.23,.23,len(rr))+j
        d.scatter(rr.ratio,yy,s=7 if len(rr)>100 else 15,color=col,alpha=.38,lw=0,zorder=2)
        z=sd.iloc[j]
        d.errorbar(z.ratio,j,xerr=[[z.ratio-z.lower],[z.upper-z.ratio]],fmt='D',mfc='white',mec=INK,
                   color=INK,ms=4.6,lw=1.2,capsize=2,zorder=4)
    d.axvline(1,color=GREY,ls=':',lw=.85)
    d.set(xscale='log',xlim=(.025,150),ylim=(2.6,-.6))
    d.set_xticks([.1,1,10,100],['0.1','1','10','100']);d.xaxis.set_minor_locator(NullLocator())
    d.set_yticks(range(3),['Brazil','New York','New Zealand'],fontsize=9.5)
    d.set_xlabel('Within-region duration ratio');clean(d)
    title(f,d,'d','Lightning persists longer on average',offset=34)

    de=pd.read_csv(V/'source_data/duration_composition_summary.csv')
    de=de[de.region.isin(['United States','New Zealand'])].copy()
    gap=pd.read_csv(P/'data/duration_gap_decomposition.csv').set_index('component')
    q=pd.read_csv(P/'data/quartile_source_record_stages.csv')
    high=q[(q.cause=='All records')&(q.quartile=='Q4')].mean_record_duration_h.iloc[0]
    de=pd.concat([de,pd.DataFrame([{'region':'Brazil','within_pct_of_high':100*gap.loc['Within-cause duration','low_minus_high_h']/high,
        'composition_pct_of_high':100*gap.loc['Cause composition','low_minus_high_h']/high,
        'gap_pct_of_high':100*gap.loc['Total duration gap','low_minus_high_h']/high}])],ignore_index=True)
    de=de.set_index('region').loc[['Brazil','United States','New Zealand']].reset_index()
    e=ax(f,146,103,59,32)
    for j,z in de.iterrows():
        neg=pos=0
        for key,col in [('within_pct_of_high',BLUE),('composition_pct_of_high',GOLD)]:
            value=z[key];left=pos if value>=0 else neg
            e.barh(j,value,left=left,color=col,height=.58)
            if value>=0:pos+=value
            else:neg+=value
        e.scatter(z.gap_pct_of_high,j,s=20,marker='D',color=INK,edgecolor='white',lw=.4,zorder=3)
    e.axvline(0,color=GREY,ls=':',lw=.85);e.set(xlim=(-11,30),ylim=(2.6,-.6))
    e.set_xticks([-10,0,10,20,30]);e.set_yticks(range(3),['Brazil','United States','New Zealand'],fontsize=9.5)
    e.set_xlabel('Low–high duration gap\n(% of high-group duration)');clean(e)
    title(f,e,'e','Cause mix can offset or amplify gaps',offset=32)
    legend(f,[Patch(color=BLUE),Patch(color=GOLD),Line2D([],[],marker='D',color=INK,lw=0,ms=4)],
           ['Within class','Composition','Net'],161,96,3,9)
    for name,z in [('Fig05_common_cause_ratios',r),('Fig05_common_cause_summary',sd),('Fig05_duration_decomposition',de)]:
        z.to_csv(P/'source_data'/f'{name}.csv',index=False)
    for name,z,key in [('Brazil',bg,'IdeConjunto'),('New_York',nyg,'system'),('New_Zealand',nzg,'system')]:
        z[[key,'ratio','geometry']].to_parquet(P/'data'/f'figure_{name}_ratio_map.geoparquet',index=False)
    save(f,'Fig05_regional_interpretation')

def supplement():
    d=pd.read_csv(P/'data/common_cause_summary.csv')
    order=['Equipment or electrical','Planned','Other third-party','Vegetation','Wind','Other or unresolved',
           'Other environment','Explicit third-party activity','Fire','Lightning','Operational error','Flood','Erosion','Supply constraint']
    d=d.set_index('cause').loc[order]
    f=page(226)
    a=ax(f,52,10,69,92)
    for j,z in enumerate(d.itertuples()):
        a.plot([z.record_share_pct,z.hour_share_pct],[j,j],lw=1.25,color='#b2b9bb')
        a.scatter(z.record_share_pct,j,s=22,color=BLUE,zorder=3)
        a.scatter(z.hour_share_pct,j,s=22,color=RED,zorder=3)
    labels=['Equipment / electrical','Planned','Other third-party','Vegetation','Wind','Other / unresolved',
            'Other environment','Explicit third-party','Fire','Lightning','Operating error','Flood','Erosion','Supply constraint']
    a.set(xlim=(-1,62),ylim=(13.6,-.6));a.set_xticks([0,20,40,60]);a.set_yticks(range(14),labels,fontsize=9.6)
    a.set_xlabel('Share of records or record-hours (%)');clean(a)
    title(f,a,'a','Occurrence and accumulated duration',offset=46)
    legend(f,[Line2D([],[],color=c,marker='o',lw=0) for c in [BLUE,RED]],['Records','Record-hours'],85,115,2,9.8)
    b=ax(f,147,10,57,92)
    b.scatter(d.duration_upper_h,np.arange(14),s=29,color=TEAL,zorder=3)
    b.set(xscale='log',xlim=(1,120),ylim=(13.6,-.6));b.set_xticks([1,10,100],['1','10','100']);b.xaxis.set_minor_locator(NullLocator())
    b.set_yticks(range(14),['']*14);b.set_xlabel('Mean record duration (h)');clean(b)
    title(f,b,'b','Cause-specific persistence',offset=13)
    for j,val in enumerate(d.duration_upper_h):b.annotate(f'{val:.1f}',(val,j),xytext=(5,0),textcoords='offset points',va='center',fontsize=9.1)

    q=pd.read_csv(P/'data/quartile_source_record_stages.csv');q=q[q.cause.eq('All records')].set_index('quartile')
    c=ax(f,19,128,74,31)
    for metric,col,label in [('records_per_1000_customer_year',BLUE,'Frequency'),('mean_record_duration_h',TEAL,'Duration')]:
        y=q.loc[['Q1','Q2','Q3','Q4'],metric]/q.loc['Q4',metric]
        c.plot(range(4),y,color=col,lw=1.7,marker='o',ms=4,label=label)
    c.axhline(1,color=GREY,ls=':',lw=.7);c.set(ylim=(.6,2),xlim=(-.2,3.2));c.set_xticks(range(4),['Q1','Q2','Q3','Q4'])
    c.set_ylabel('Relative to Q4');c.set_xlabel('Municipal GDP quartile');clean(c)
    title(f,c,'c','Economic contrasts by outcome',offset=12)
    c.legend(loc='upper right',fontsize=9.5,ncol=1)
    m=pd.read_csv(P/'data/economic_stage_models.csv');m=m[m.cause.eq('All records')]
    e=ax(f,145,128,60,31)
    specs=['Economic context','Density adjusted','Within context + density']
    for j,sp in enumerate(specs):
        for dy,metric,col in [(-.16,'Record frequency',BLUE),(.16,'Mean record duration',TEAL)]:
            z=m[(m.specification==sp)&(m.metric==metric)].iloc[0]
            e.errorbar(z.contrast_pct,j+dy,xerr=[[z.contrast_pct-z.lower_pct],[z.upper_pct-z.contrast_pct]],
                       fmt='o',color=col,ms=4,capsize=2,lw=1.2)
    e.set(xlim=(-48,58),ylim=(2.55,-.55));e.set_xticks([-40,0,40]);e.set_yticks(range(3),['GDP only','+ Density','+ Context'],fontsize=9.5)
    e.axvline(0,color=GREY,ls=':',lw=.8);e.set_xlabel('GDP-IQR response (%)');clean(e)
    title(f,e,'d','Adjustment changes the rate gradient',offset=31)
    g=ax(f,60,186,104,26)
    rr=EST.copy()
    for j,z in rr.iterrows():
        g.errorbar(z.geometric_ratio,j,xerr=[[z.geometric_ratio-z.lower],[z.upper-z.geometric_ratio]],
                   fmt='o',color=TEAL,ms=4,capsize=2,lw=1.2)
        g.text(1.12,j,str(int(z.areas)),transform=g.get_yaxis_transform(),va='center',fontsize=9.3)
    g.axvline(1,color=GREY,ls=':',lw=.8)
    g.set(xscale='log',xlim=(.7,2.6),ylim=(4.65,-.65))
    g.set_xticks([.75,1,1.5,2.5],['0.75','1','1.5','2.5']);g.xaxis.set_minor_locator(NullLocator())
    g.set_yticks(range(5),['Wind / lightning','Wind / operating error','Lightning / operating error',
                          'Vegetation / operating error','Flood / operating error'],fontsize=9.2)
    g.set_xlabel('Within-area geometric duration ratio');clean(g)
    title(f,g,'e','Paired causes within the same area',offset=54)
    text(f,176,181,'Areas',fontsize=9.4)
    rr.to_csv(P/'source_data/SuppFig12_paired_cause_ratios.csv',index=False)
    d.reset_index().to_csv(P/'source_data/SuppFig12_cause_profiles.csv',index=False)
    q.reset_index().to_csv(P/'source_data/SuppFig12_economic_quartiles.csv',index=False)
    m.to_csv(P/'source_data/SuppFig12_economic_adjustment.csv',index=False)
    save(f,'SuppFig12_brazil_fine_cause_persistence')

if __name__=='__main__':
    for task in (sys.argv[1:] or ['main','supplement']):globals()[task]()
