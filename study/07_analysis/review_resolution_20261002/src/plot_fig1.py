"""Three-row extension of the existing global demand/reliability figure.

Python/matplotlib; observed annual demand and source reliability benchmarks.
Demand and GDP are equal-year arithmetic means over a complete common period.
CAIDI uses the same positive-SAIFI benchmark as the frequency row.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullLocator
from matplotlib.text import Text
from scipy.stats import spearmanr

P = Path(__file__).resolve().parents[1]
ROOT = P.parents[1]
OLD = ROOT / '07_analysis/multiscale_grid_results_20260926'
for sub in ['figures', 'source_data', 'metadata']:
    (P / sub).mkdir(exist_ok=True, parents=True)
STEM = 'Fig01_global_demand_frequency_duration'
START_YEAR, END_YEAR = 2000, 2024
N_YEARS = END_YEAR - START_YEAR + 1
PERIOD = f'{START_YEAR}–{END_YEAR}'
paths = {
    'demand': ROOT / '10_manuscript/20260930_v01/data/demand_all_context.parquet',
    'reliability': OLD / 'data/saifi_gdp_latest.parquet',
    'geometry': ROOT / '10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet'}
contract = dict(
    backend='Python / matplotlib', archetype='quantitative grid with maps and joint distributions',
    purpose='Compare multiyear demand, interruption frequency and mean interruption duration with economic development.',
    evidence=f'Mean annual per-capita demand in {PERIOD}; complete {N_YEARS}-year GDP mean for demand pairs. Latest source-native city/utility reliability benchmark in 2014–2019, paired with GDP in its own year.',
    panels={'a': 'Demand map', 'b': 'Demand–GDP joint counts', 'c': 'Frequency map',
            'd': 'Frequency–GDP joint counts', 'e': 'Mean interruption-duration map', 'f': 'Duration–GDP joint counts'},
    restoration_definition='CAIDI = SAIDI hours / positive SAIFI, in hours per customer interruption; not event T90 or complete national restoration.',
    exports=['PNG 600 dpi', 'PDF with embedded text', 'SVG with editable text', 'PNG preview', 'CSV inputs and bins'],
    figure_width_mm=210, missing_data='Grey countries have insufficient data for the indicated outcome and period; no imputation or modelled values',
    layout='Reduce the map-to-joint-axis gap from 25.2 to 19.95 mm; no footer annotation; retain degree frames and original palettes.',
    reuse='Replace only the demand snapshot with complete-period means; retain frequency and CAIDI values, GDP pairs, binning and palettes.')
(P / 'metadata/figure_contract.json').write_text(json.dumps(contract, indent=2))
inputs = [dict(role=k, path=str(v.relative_to(ROOT)), bytes=v.stat().st_size,
               sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k, v in paths.items()]
(P / 'metadata/input_manifest.json').write_text(json.dumps(inputs, indent=2))

annual = pd.read_parquet(paths['demand'])
annual = annual.loc[annual.year.between(START_YEAR, END_YEAR)].copy()
assert not annual.duplicated(['ISO3', 'year']).any()
annual['valid_demand'] = (annual.demand_twh.gt(0) & annual.population_at_year.gt(0)
                          & np.isfinite(annual.demand_twh) & np.isfinite(annual.population_at_year))
annual['valid_gdp'] = annual.gdp.gt(0) & np.isfinite(annual.gdp)
annual['demand_mwh_pc'] = (annual.demand_twh * 1e6 / annual.population_at_year).where(annual.valid_demand)
annual['valid_gdp_value'] = annual.gdp.where(annual.valid_gdp)
coverage = annual.groupby('ISO3', as_index=False).agg(
    name=('name', 'first'), continent=('continent', 'first'),
    observed_years=('year', 'nunique'), demand_years=('valid_demand', 'sum'), gdp_years=('valid_gdp', 'sum'))
coverage['period_start'], coverage['period_end'] = START_YEAR, END_YEAR
coverage['demand_map_included'] = coverage.demand_years.eq(N_YEARS)
coverage['demand_gdp_pair_included'] = coverage.demand_map_included & coverage.gdp_years.eq(N_YEARS)
full_years = set(range(START_YEAR, END_YEAR + 1))
for ok, field in [('valid_demand', 'missing_demand_years'), ('valid_gdp', 'missing_gdp_years')]:
    observed = annual.loc[annual[ok]].groupby('ISO3').year.apply(set).to_dict()
    coverage[field] = coverage.ISO3.map(lambda iso: ';'.join(map(str, sorted(full_years - observed.get(iso, set())))))
means = annual.groupby('ISO3', as_index=False).agg(
    demand_mwh_pc=('demand_mwh_pc', 'mean'), mean_gdp_ppp_pc_constant_2021=('valid_gdp_value', 'mean'))
demand = coverage.loc[coverage.demand_map_included].merge(means, on='ISO3', validate='one_to_one')
demand.loc[~demand.demand_gdp_pair_included, 'mean_gdp_ppp_pc_constant_2021'] = np.nan
assert demand.demand_years.eq(N_YEARS).all()
assert demand.loc[demand.mean_gdp_ppp_pc_constant_2021.notna(), 'gdp_years'].eq(N_YEARS).all()
annual['demand_map_included'] = annual.ISO3.isin(demand.ISO3)
annual['demand_gdp_pair_included'] = annual.ISO3.isin(demand.loc[demand.demand_gdp_pair_included, 'ISO3'])
annual.to_csv(P / f'source_data/annual_demand_gdp_{START_YEAR}_{END_YEAR}.csv', index=False)
coverage.to_csv(P / 'source_data/demand_period_coverage.csv', index=False)
rel = pd.read_parquet(paths['reliability'])
assert not rel.ISO3.duplicated().any() and not demand.ISO3.duplicated().any()
valid = rel.saifi.gt(0) & rel.saidi_h.ge(0) & np.isfinite(rel.saidi_h)
rel['recovery_h'] = np.where(valid, rel.saidi_h / rel.saifi, np.nan)
assert np.allclose(rel.loc[valid, 'recovery_h'], rel.loc[valid, 'caidi_h'])
assert rel.year.between(2014, 2019).all()
demand.to_csv(P / f'source_data/demand_multiyear_{START_YEAR}_{END_YEAR}.csv', index=False)
rel.to_csv(P / 'source_data/reliability_frequency_recovery_latest.csv', index=False)
world = gpd.read_parquet(paths['geometry']).to_crs(4326)
world = world[world.ISO3.ne('ATA')].copy()
world.geometry = world.geometry.simplify(.035, preserve_topology=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 9, 'axes.labelsize': 9.2,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.linewidth': .65, 'axes.spines.top': True, 'axes.spines.right': True,
    'axes.labelpad': 3, 'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
BLUE, RED, TEAL = '#28649a', '#bd5040', '#187f7e'
FW, FH = 210 / 25.4, 9.01
fig = plt.figure(figsize=(FW, FH), dpi=150)
def axes(rect):
    x, y, w, h = rect
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])
def panel(ax, letter, title, pad=.08):
    b = ax.get_position()
    fig.text(b.x0-.035, b.y1+pad/FH, letter, fontsize=12, fontweight='bold', ha='left', va='bottom')
    fig.text(b.x0, b.y1+pad/FH, title, fontsize=9.6, ha='left', va='bottom')
def lon(v, pos=None): return '0°' if v==0 else f'{abs(v):g}°'+('W' if v<0 else 'E')
def lat(v, pos=None): return '0°' if v==0 else f'{abs(v):g}°'+('S' if v<0 else 'N')

stats, geometry_audit = {}, []
specs = [
    dict(code='demand', letters=('a','b'), data=demand, y='demand_mwh_pc', x='mean_gdp_ppp_pc_constant_2021',
         color=BLUE, title=f'Mean electricity demand · {PERIOD}', label='Demand (MWh/person)', ylabel='Demand (MWh/person)',
         edges=[0,.5,1,2,5,10,20,80], colors=['#edf3f8','#d2e3ee','#afd0e2','#82b4d1','#558fb9','#326998','#174673'],
         ticks=[.5,1,2,5,10,20], yedges=np.geomspace(.01,100,7), yticks=[.1,1,10,100]),
    dict(code='frequency', letters=('c','d'), data=rel, y='saifi', x='gdp_ppp_pc_at_year',
         color=RED, title='Interruption frequency · 2014–2019', label='SAIFI (interruptions/year)', ylabel='SAIFI (per year)',
         edges=[0,.5,1,3,10,30,100,2000], colors=['#f9eee8','#f0d3bf','#e8ad91','#d8896d','#c76250','#ab3a38','#7d1e2c'],
         ticks=[.5,1,3,10,30,100], yedges=np.geomspace(.008,2000,7), yticks=[.01,1,100]),
    dict(code='recovery', letters=('e','f'), data=rel, y='recovery_h', x='gdp_ppp_pc_at_year',
         color=TEAL, title='Mean interruption duration · 2014–2019', label='CAIDI (h/interruption)', ylabel='CAIDI (h)',
         edges=[0,.5,1,2,4,8,24,100], colors=['#eff7f4','#d2e9e2','#a7d4c8','#72b9ab','#409a8f','#247b75','#125550'],
         ticks=[.5,1,2,4,8,24], yedges=np.geomspace(.1,100,7), yticks=[.1,1,10,100])]
for row, s in enumerate(specs):
    bottom = 6.72 - row*2.84
    q, val, gx = s['data'], s['y'], s['x']
    cmap = ListedColormap(s['colors']); norm = BoundaryNorm(s['edges'], cmap.N)
    mp = axes([.07*FW, bottom, .545*FW, 1.845])
    w = world.merge(q[['ISO3',val]], on='ISO3', how='left', validate='many_to_one')
    w.plot(column=val, ax=mp, cmap=cmap, norm=norm, linewidth=.27, edgecolor='#a7aaad',
           missing_kwds={'color':'#e8e8e8'}, aspect='auto')
    mp.set(xlim=(-180,180), ylim=(-60,80))
    mp.set_xticks([-120,-60,0,60,120]); mp.set_yticks([-30,0,30,60])
    mp.xaxis.set_major_formatter(FuncFormatter(lon)); mp.yaxis.set_major_formatter(FuncFormatter(lat))
    mp.grid(lw=.45, color='#cecece', ls='--', alpha=.8)
    panel(mp, s['letters'][0], s['title'])
    bar_ax = axes([.15*FW,bottom-.36,.385*FW,.074])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap), cax=bar_ax,
                      orientation='horizontal', ticks=s['ticks'], spacing='uniform')
    bar.ax.set_xticklabels([f'{x:g}' for x in s['ticks']]); bar.ax.minorticks_off(); bar.set_label(s['label'])
    geometry_audit.extend(dict(panel=s['letters'][0],ISO3=r.ISO3,value=float(getattr(r,val)),geometry_present=r.ISO3 in set(world.ISO3))
                          for r in q[['ISO3',val]].dropna().itertuples(index=False))

    x0, mainw, mainh = .710*FW, .17*FW, 1.468
    ja = axes([x0,bottom,mainw,mainh])
    top = axes([x0,bottom+mainh+.14,mainw,.313])
    side = axes([x0+mainw+.012*FW,bottom,.048*FW,mainh])
    paired = q[q[gx].gt(0)&q[val].gt(0)&np.isfinite(q[gx])&np.isfinite(q[val])].copy()
    xe, ye = np.geomspace(500,200000,7), s['yedges']
    z,_,_ = np.histogram2d(paired[gx],paired[val],bins=[xe,ye]); assert z.sum()==len(paired)
    cm = LinearSegmentedColormap.from_list('joint',['#f9fafb','#b5c8d6'])
    ja.pcolormesh(xe,ye,np.ma.masked_where(z.T==0,z.T),cmap=cm,vmin=0,vmax=max(z.max(),1),edgecolors='white',lw=.7)
    for i,j in zip(*np.where(z>0)):
        ja.text(np.sqrt(xe[i]*xe[i+1]),np.sqrt(ye[j]*ye[j+1]),str(int(z[i,j])),ha='center',va='center',fontsize=8.3,color='#243840')
    ja.set_xscale('log'); ja.set_yscale('log'); ja.set(xlim=(xe[0],xe[-1]),ylim=(ye[0],ye[-1]),xlabel='GDP/person (PPP $)',ylabel=s['ylabel'])
    ja.set_xticks([1000,10000,100000],['1k','10k','100k']); ja.set_yticks(s['yticks'],[f'{v:g}' for v in s['yticks']]); ja.minorticks_off()
    xx, yy = np.sqrt(xe[:-1]*xe[1:]), np.sqrt(ye[:-1]*ye[1:])
    top.plot(xx,100*z.sum(1)/len(paired),color=s['color'],lw=1.35)
    top.fill_between(xx,0,100*z.sum(1)/len(paired),color=s['color'],alpha=.11,lw=0)
    top.set_xscale('log'); top.set(xlim=ja.get_xlim(),ylim=(0,55)); top.set_xticks([]);top.set_yticks([0,50]);top.set_ylabel('%',rotation=0,labelpad=6);top.minorticks_off()
    side.plot(100*z.sum(0)/len(paired),yy,color=s['color'],lw=1.35)
    side.fill_betweenx(yy,0,100*z.sum(0)/len(paired),color=s['color'],alpha=.11,lw=0)
    side_max = max(55, float(np.ceil((100*z.sum(0)/len(paired)).max()/25)*25))
    side.set_yscale('log');side.set(ylim=ja.get_ylim(),xlim=(0,side_max));side.set_yticks([]);side.set_xticks([0,50] if side_max==55 else [0,side_max]);side.set_xlabel('%');side.minorticks_off()
    panel(top,s['letters'][1],f'Joint counts · n={len(paired)}',pad=.06)
    cells = pd.DataFrame([dict(x_lower=xe[i],x_upper=xe[i+1],y_lower=ye[j],y_upper=ye[j+1],countries=int(z[i,j]),share_pct=100*z[i,j]/len(paired)) for i in range(6) for j in range(6)])
    cells.to_csv(P/'source_data'/f'joint_{s["code"]}_gdp_cells.csv',index=False)
    paired.to_csv(P/'source_data'/f'paired_{s["code"]}_gdp.csv',index=False)
    if s['code']=='frequency':
        original=pd.read_csv(OLD/'source_data/joint_saifi_gdp_cells.csv')
        assert np.allclose(original[['x_lower','x_upper','y_lower','y_upper','countries']],cells[['x_lower','x_upper','y_lower','y_upper','countries']])
    stats[s['code']]=dict(observed_economies=int(q[val].notna().sum()),paired_economies=len(paired),
                          spearman_rho=float(spearmanr(paired[gx],paired[val]).statistic),
                          minimum=float(q[val].min()),median=float(q[val].median()),maximum=float(q[val].max()),
                          joint_cell_sum=int(z.sum()),marginal_x_sum=float(100*z.sum()/len(paired)),marginal_y_sum=float(100*z.sum()/len(paired)))
    if s['code']=='demand':
        stats['demand'].update(period_start=START_YEAR, period_end=END_YEAR, years_per_economy=N_YEARS,
                              estimator='Arithmetic mean of annual per-capita demand; equal year weights.',
                              gdp_alignment='Arithmetic mean of GDP per capita over the same complete period.',
                              imputation=False)

fig.canvas.draw(); renderer=fig.canvas.get_renderer(); items=[]
for t in fig.findobj(Text):
    if not t.get_visible() or not t.get_text().strip():continue
    b=t.get_window_extent(renderer)
    if b.width>.1 and b.height>.1:items.append((t,b))
over=[]
for i,(a,aa) in enumerate(items):
    for b,bb in items[i+1:]:
        dx=min(aa.x1,bb.x1)-max(aa.x0,bb.x0);dy=min(aa.y1,bb.y1)-max(aa.y0,bb.y0)
        if dx>1.5 and dy>1.5:over.append(dict(first=a.get_text(),second=b.get_text(),overlap_px=[round(dx,2),round(dy,2)]))
out=[]
for t,b in items:
    if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:out.append(t.get_text())
qa=dict(text_items=len(items),minimum_font_pt=min(t.get_fontsize() for t,_ in items),
        overlap_candidates=over,outside_initial_canvas=out,visual_review_required=True,
        map_to_joint_axes_gap_mm=(.710-.615)*210, previous_gap_mm=(.735-.615)*210,
        footer_annotation=False)
(P/'metadata/layout_qa.json').write_text(json.dumps(qa,indent=2))
for ext in ['png','pdf','svg']:
    fig.savefig(P/'figures'/f'{STEM}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.035)
fig.savefig(P/'figures'/f'{STEM}_preview.png',dpi=170,bbox_inches='tight',pad_inches=.035)
plt.close(fig)
pd.DataFrame(geometry_audit).to_csv(P/'source_data/geometry_join_audit.csv',index=False)
(P/'metadata/results.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(dict(results=stats,layout=qa),indent=2))
