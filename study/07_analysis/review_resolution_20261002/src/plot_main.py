"""Global observation footprint, regional zooms and matched interpretation.

Established regional values are retained; direct Turkish settlement groups add a fourth within-source decomposition.
The global map adds observed frequency-duration rank differences in the fixed
152-benchmark population. Boxes link this result to cause-resolved zooms, whose
different metric is explicitly labelled and separately keyed.
"""
from pathlib import Path
import sys, json, hashlib

P = Path(__file__).resolve().parents[1]
R = P.parents[1]
V = R / '10_manuscript/20261002_global_blackout_inequalities_v14'
sys.path.insert(0, str(V / 'src'))
import figure_style as fs
fs.P = P
from figure_style import *

B = R / '07_analysis/brazil_detailed_cause_extension_20261002'
J = R / '07_analysis/poverty_cause_contrasts_20261002'
wp = R / '10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet'
W = gpd.read_parquet(wp).to_crs(4326)
inputs = [wp]
plt.rcParams.update({'font.size':10.5, 'xtick.labelsize':10, 'ytick.labelsize':10,
                     'axes.labelsize':10.5})
f = page(250)
cm = LinearSegmentedColormap.from_list('paired_ratio',
    ['#AE7139', '#E8C892', '#F6F6F2', '#8FB5CB', '#285986'])
norm = LogNorm(1/16, 16)
regions = [
    dict(name='Brazil', iso='BRA', crs=5880, rect=[14,83,60,40],
         lon=[-65,-45], lat=[-30,-15,0], label='Brazil · 593 areas', letter='b',
         year='2024–2025', colour=TEAL, box=[-75,-35,-33,7], labelxy=(-85,-7)),
    dict(name='New_York', iso='USA', crs=5070, rect=[89,83,59,40],
         lon=[-79,-75], lat=[42,44], label='New York · 8 regions', letter='c',
         year='2021–2024', colour=RED, box=[-81,-71,40,46], labelxy=(-89,52)),
    dict(name='New_Zealand', iso='NZL', crs=2193, rect=[165,83,39,40],
         lon=[170,176], lat=[-45,-40], label='New Zealand', letter='d',
         year='2023–2025', colour=BLUE, box=[164,179,-48,-33], labelxy=(149,-43)),
]
for r in regions:
    path = B / 'data' / ('figure_' + r['name'] + '_ratio_map.geoparquet')
    inputs.append(path)
    r['g'] = gpd.read_parquet(path).to_crs(4326)

# Global row: the geographical mismatch between frequency and duration ranks.
gp=R/'07_analysis/global_to_local_figure_design_20261002/source_data/Fig05_global_rank_geography.csv'
global_data=pd.read_csv(gp);inputs.append(gp)
global_cm=LinearSegmentedColormap.from_list('rank_discordance',
    ['#654183','#C3B3D1','#F5F4F0','#B2D4C4','#267C66'])
global_norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=100)
a = ax(f,14,9,146,60)
world_rank=W.merge(global_data[['ISO3','frequency_minus_duration_rank_pp']],on='ISO3',how='left')
world_rank.plot(ax=a,column='frequency_minus_duration_rank_pp',cmap=global_cm,norm=global_norm,
                edgecolor='#AAB3B6',linewidth=.22,
                missing_kwds=dict(color='#DFE3E3',edgecolor='#AAB3B6',linewidth=.22))
for r in regions:
    x0,x1,y0,y1 = r['box']
    a.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False,
                         edgecolor=r['colour'], linewidth=1.15, zorder=7))
    a.text(*r['labelxy'], r['letter'], color=r['colour'], fontsize=12,
           weight='bold', ha='center', va='center', zorder=9,
           bbox=dict(facecolor='white', edgecolor='none', pad=.25, alpha=.9))
a.set(xlim=(-180,180), ylim=(-59,84)); a.set_aspect('equal',adjustable='box')
frame = geographic_frame(a,4326,longitudes=[-120,-60,0,60,120],
                         latitudes=[-30,0,30,60],fontsize=10)
a.text(-173,-39,'152 benchmarks\nRank ρ = 0.05',
       ha='left',va='top',fontsize=9.7,linespacing=1.25,zorder=9,
       bbox=dict(facecolor='white',edgecolor='none',pad=.5,alpha=.92))
title(f,a,'a','Frequency and duration reveal different geographies',offset=9)
text(f,166,12,'a: Frequency − duration',fontsize=9.7,va='top')
text(f,166,17,'rank (percentile points)',fontsize=9.7,va='top')
c=f.colorbar(plt.cm.ScalarMappable(norm=global_norm,cmap=global_cm),
             cax=ax(f,169,25,33,2.5),orientation='horizontal',ticks=[-75,0,75])
c.ax.set_xticklabels(['−75','0','75']);c.ax.minorticks_off()
text(f,166,42,'b–d: Lightning / error',fontsize=9.7,va='top')
text(f,166,47,'mean-duration ratio',fontsize=9.7,va='top')
c=f.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),
             cax=ax(f,169,57,33,2.5),orientation='horizontal',ticks=[1/16,1,16],extend='both')
c.ax.set_xticklabels(['1/16','1','16']);c.ax.minorticks_off()

# Regional row: projected geographic frames, and the same coloured outline
# used by its global locator. No interpolation outside source footprints.
frames={}
for r in regions:
    g=r['g'].to_crs(r['crs']); a=ax(f,*r['rect'])
    if r['name']=='New_York':
        basepath=V/'data/us_same49_map.parquet';inputs.append(basepath)
        base=gpd.read_parquet(basepath);base=base[base.state.eq('NY')].to_crs(r['crs'])
    else:base=W[W.ISO3.eq(r['iso'])].to_crs(r['crs'])
    base.plot(ax=a,color='#eceeed',edgecolor='#aeb7bc',lw=.4)
    g.plot(column='ratio',ax=a,cmap=cm,norm=norm,
           edgecolor='white' if r['name']!='Brazil' else 'none',lw=.3,zorder=2)
    if r['name']=='Brazil':
        pt=g.geometry.representative_point()
        a.scatter(pt.x,pt.y,c=g.ratio,cmap=cm,norm=norm,s=3.7,
                  lw=.08,edgecolor='white',zorder=3)
    extent(a,base.total_bounds if r['name']!='New_Zealand' else g.total_bounds,
           r['rect'][2]/r['rect'][3],1.02)
    frames[r['name']]=geographic_frame(a,r['crs'],longitudes=r['lon'],
                                      latitudes=r['lat'],fontsize=10)
    for spine in a.spines.values():
        spine.set_color(r['colour']);spine.set_linewidth(.9)
    title(f,a,r['letter'],r['label'],offset=9)
    b=a.get_position()
    text(f,(b.x0+b.x1)*105,134,r['year'],ha='center',fontsize=10)

# Poverty gradients: identical analytical support for all six causes.
grpath=J/'data/gradients.csv';gr=pd.read_csv(grpath);inputs.append(grpath)
order=['Wind','Lightning','Vegetation','Flood','Operational error','Planned']
labels=['Wind','Lightning','Vegetation','Flood','Operating error','Planned']
text(f,39,141,'Brazil · per 10 percentage points of low-income residents',
     fontsize=10.4,va='center')
for rect,metric,col,letter,heading in [
    ([39,152,68,36],'Frequency',BLUE,'e','Record rate'),
    ([137,152,67,36],'Conditional duration',RED,'f','Mean record duration')]:
    a=ax(f,*rect);z=gr[gr.metric.eq(metric)].set_index('cause').loc[order]
    for j,row in enumerate(z.itertuples()):
        if j%2==0:a.axhspan(j-.5,j+.5,color='#f1f3f3',zorder=0)
        a.errorbar(row.change_pct,j,
            xerr=[[row.change_pct-row.lower_pct],[row.upper_pct-row.change_pct]],
            fmt='o',color=col,mec='white',mew=.4,ms=4.7,lw=1.2,capsize=2,zorder=3)
    a.axvline(0,color=GREY,ls='--',lw=.8)
    a.set(xlim=(-45,80),ylim=(5.6,-.6));a.set_xticks([-40,0,40,80])
    a.set_xlabel('Change (%)');a.set_yticks(range(6),labels if letter=='e' else ['']*6,fontsize=10)
    clean(a);title(f,a,letter,heading,offset=34 if letter=='e' else 9)

# Across-source cause contrast: never pooled into a global coefficient.
rp=B/'source_data/Fig05_common_cause_ratios.csv'
sp=B/'source_data/Fig05_common_cause_summary.csv'
rat=pd.read_csv(rp);sd=pd.read_csv(sp);inputs.extend([rp,sp])
a=ax(f,39,213,67,23)
for j,(country,col) in enumerate(zip(['Brazil','New York','New Zealand'],[TEAL,RED,BLUE])):
    rr=rat[rat.source.eq(country)].sort_values('unit')
    yy=np.random.default_rng(2026+j).uniform(-.23,.23,len(rr))+j
    a.scatter(rr.ratio,yy,s=6 if len(rr)>100 else 13,color=col,alpha=.38,lw=0,zorder=2)
    z=sd[sd.source.eq(country)].iloc[0]
    a.errorbar(z.ratio,j,xerr=[[z.ratio-z.lower],[z.upper-z.ratio]],fmt='D',
               mfc='white',mec=INK,color=INK,ms=4.5,lw=1.2,capsize=2,zorder=4)
a.axvline(1,color=GREY,ls=':',lw=.85)
a.set(xscale='log',xlim=(.025,150),ylim=(2.6,-.6))
a.set_xticks([.1,1,10,100],['0.1','1','10','100']);a.xaxis.set_minor_locator(NullLocator())
a.set_yticks(range(3),['Brazil','New York','New Zealand'],fontsize=10)
a.set_xlabel('Paired duration ratio');clean(a)
title(f,a,'g','Lightning versus operating error',offset=34)

dp=P/'data/regional_composition_intervals.csv'
de=pd.read_csv(dp);inputs.append(dp)
a=ax(f,153,213,51,23)
for j,reg in enumerate(['Brazil','United States','New Zealand']):
    for off,key,col,marker in [(-.22,'Within class',BLUE,'o'),(0,'Composition',GOLD,'o'),(.22,'Net',INK,'D')]:
        row=de[(de.region==reg)&(de.component==key)].iloc[0]
        a.errorbar(row.value,j+off,xerr=[[row.value-row.lower],[row.upper-row.value]],
                   fmt=marker,color=col,ms=3.2,lw=.85,capsize=1.6,mec='white',mew=.3,zorder=3)
a.axvline(0,color=GREY,ls=':',lw=.8)
a.set(xlim=(-1.6,1.8),ylim=(2.65,-.65));a.set_xticks([-1,0,1])
a.set_yticks(range(3),['Brazil','United States','New Zealand'],fontsize=10)
a.set_xlabel('Low − high group duration (h)');clean(a)
title(f,a,'h','Duration gap accounting',offset=31)
legend(f,[Line2D([],[],marker='o',color=BLUE,ms=4),Line2D([],[],marker='o',color=GOLD,ms=4),Line2D([],[],marker='D',color=INK,lw=0,ms=4)],
       ['Within class','Composition','Net'],154,202,3,9.7)

for name,z in [('Fig05_common_cause_ratios',rat),('Fig05_common_cause_summary',sd),
               ('Fig05_duration_decomposition',de),('Fig05_poverty_gradients',gr)]:
    z.to_csv(P/'source_data'/(name+'.csv'),index=False)
(P/'metadata/figure_sources.json').write_text(json.dumps([
    {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dict.fromkeys(inputs)],indent=2))
(P/'metadata/geographic_frames.json').write_text(json.dumps(dict(global_map=frame,zooms=frames),indent=2))
(P/'metadata/layout_contract.json').write_text(json.dumps(dict(
    layout=['global observed frequency-duration ranking difference','three cause-specific regional zooms','social gradients','cause contrast and economic composition'],
    changed_regional_estimates=False,new_regional_uncertainty=True,new_turkiye_decomposition=False,new_descriptive_global_rank_comparison=False,
    global_extrapolation=False,region_locator_colours={'Brazil':TEAL,'New York':RED,'New Zealand':BLUE},
    scope_note='The global map is a benchmark rank comparison. Regional maps use cause-specific duration ratios with a separate scale. Panel h uses separately defined economic samples, including 49 US states; all three components display regional-resampling intervals. Turkish direct settlement groups are reported in Supplementary Fig. 18.'),indent=2))
save(f,'Fig05_regional_interpretation')
