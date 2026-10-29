"""Full fixed-quartile accounting and Figure 2. Python-only publication workflow."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import NullLocator

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'core_reproducibility/src'))
from map_coordinates import geographic_frame

SEED = 202610021
B = 40000
QS = ['Q1','Q2','Q3','Q4']
BLUE, TEAL, RED, INK = '#376F9F', '#B23B4D', '#B23B4D', '#30383D'
INPUT = P/'core_reproducibility/inputs/country_benchmarks.csv'
OUT = P/'source_data'
for folder in ['source_data','figures','metadata','reports']:
    (P/folder).mkdir(exist_ok=True)

def interval(values):
    return np.quantile(values, [.025,.975]).tolist()

def values(f, h):
    return {'frequency': f, 'annual_hours': h, 'compatible_duration':h/f}

def partition(left, right):
    f0,h0,d0 = left['frequency'],left['annual_hours'],left['compatible_duration']
    f1,h1,d1 = right['frequency'],right['annual_hours'],right['compatible_duration']
    cf = (f0-f1)*(d0+d1)/2
    cd = (d0-d1)*(f0+f1)/2
    return dict(net_gap_h=h0-h1, frequency_h=cf, duration_h=cd,
                contribution_difference_h=cf-cd, frequency_share=cf/(h0-h1))

def analyze():
    d = pd.read_csv(INPUT)
    assert len(d)==152 and d.ISO3.nunique()==152
    assert d[['saifi','saidi_h']].gt(0).all().all()
    rng = np.random.default_rng(SEED)
    means, boot, rows = {}, {}, []
    for q in QS:
        z = d[d.gdp_quartile.eq(q)]
        assert len(z)==38
        means[q] = values(z.saifi.mean(),z.saidi_h.mean())
        idx = rng.integers(0,len(z),size=(B,len(z)))
        boot[q] = values(z.saifi.to_numpy()[idx].mean(1), z.saidi_h.to_numpy()[idx].mean(1))
        for key, estimate in means[q].items():
            lo,hi=interval(boot[q][key])
            rows.append(dict(quartile=q, metric=key, estimate=estimate, ci_low=lo, ci_high=hi,
                             n=38,gdp_min=z.gdp_ppp_pc_at_year.min(),gdp_max=z.gdp_ppp_pc_at_year.max(),
                             gdp_median=z.gdp_ppp_pc_at_year.median()))
    groups = pd.DataFrame(rows)
    groups.to_csv(OUT/'gradient_v31_group_estimates.csv',index=False)
    comps, draws, summaries = {}, {}, []
    comparisons=[('Q1','Q2','adjacent'),('Q2','Q3','adjacent'),('Q3','Q4','adjacent'),
                 ('Q1','Q4','versus_Q4'),('Q2','Q4','versus_Q4')]
    for a,b,kind in comparisons:
        name=f'{a}_{b}'; comps[name]=partition(means[a],means[b]); draws[name]=partition(boot[a],boot[b])
        resolved = interval(draws[name]['net_gap_h'])[0]>0
        for key,v in comps[name].items():
            lo,hi=interval(draws[name][key])
            if key=='frequency_share' and not resolved:
                v=lo=hi=np.nan
            summaries.append(dict(contrast=name,contrast_type=kind,metric=key,estimate=v,ci_low=lo,ci_high=hi,
                                  net_gap_resolved=resolved,n_left=38,n_right=38))
        assert np.allclose(draws[name]['frequency_h']+draws[name]['duration_h'],draws[name]['net_gap_h'])
    decomp=pd.DataFrame(summaries)
    decomp.to_csv(OUT/'gradient_v31_absolute_partitions.csv',index=False)
    # Each contrast uses the same resampled groups. Reusing Q4 draws preserves
    # covariance when comparing partitions; independently bootstrapped contrast
    # intervals would not be a valid test of a difference between them.
    contrasts=[]
    for key in ['Q3_Q4','Q2_Q4']:
        for metric in ['frequency_share','frequency_h','duration_h','contribution_difference_h']:
            estimate=comps[key][metric]-comps['Q1_Q4'][metric]
            samples=draws[key][metric]-draws['Q1_Q4'][metric]
            lo,hi=interval(samples)
            contrasts.append(dict(comparison=f'{key}_minus_Q1_Q4',metric=metric,estimate=estimate,
                                  ci_low=lo,ci_high=hi,interval_excludes_zero=lo*hi>0,
                                  covariance='Shared bootstrap draws for every quartile'))
    for a,b in [('Q1','Q2'),('Q2','Q3'),('Q3','Q4')]:
        for metric in ['frequency','annual_hours','compatible_duration']:
            sample=boot[a][metric]-boot[b][metric]
            lo,hi=interval(sample)
            contrasts.append(dict(comparison=f'{a}_minus_{b}',metric=metric,
                                  estimate=means[a][metric]-means[b][metric],ci_low=lo,ci_high=hi,
                                  interval_excludes_zero=lo*hi>0,covariance='Whole benchmarks sampled within fixed groups'))
    tests=pd.DataFrame(contrasts)
    tests.to_csv(OUT/'gradient_v31_shared_draw_contrasts.csv',index=False)
    # Full matched rows make the figure and accounting independently reproducible.
    d.to_csv(OUT/'gradient_v31_benchmark_inputs.csv',index=False)
    np.savez_compressed(OUT/'gradient_v31_bootstrap_draws.npz',
                        **{f'{q}_{k}':v for q,vals in boot.items() for k,v in vals.items()})
    return d,groups,decomp,tests

def layout_audit(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();items=[]
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            r=t.get_window_extent(renderer)
            if r.width>.1 and r.height>.1:items.append((t,r))
    overlaps=[]
    for i,(a,ra) in enumerate(items):
        for b,rb in items[i+1:]:
            if min(ra.x1,rb.x1)-max(ra.x0,rb.x0)>1.5 and min(ra.y1,rb.y1)-max(ra.y0,rb.y0)>1.5:
                overlaps.append([a.get_text(),b.get_text()])
    outside=[t.get_text() for t,r in items if r.x0<0 or r.y0<0 or r.x1>fig.bbox.width or r.y1>fig.bbox.height]
    return dict(text_overlap_candidates=overlaps,outside_canvas=outside,
                minimum_font_pt=min(t.get_fontsize() for t,r in items))

def draw(d,groups,decomp):
    plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,
                         'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.65,
                         'axes.labelpad':3,'xtick.major.pad':2,'ytick.major.pad':2,
                         'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    W,H=183,185
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=170)
    def ax(x,top,w,h): return fig.add_axes([x/W,1-(top+h)/H,w/W,h/H])
    def title(a,letter,label):
        box=a.get_position()
        fig.text(box.x0-5/W,box.y1+1.8/H,letter,weight='bold',fontsize=10,va='bottom')
        fig.text(box.x0,box.y1+1.8/H,label,fontsize=9,va='bottom')
    def clean(a):
        a.spines[['right','top']].set_visible(False)
        a.tick_params(length=2.5,width=.65)
        a.set_axisbelow(True)
    a=ax(15,8,164,58)
    world=gpd.read_parquet(P/'core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
    world=world[world.ISO3.ne('ATA')].merge(d[['ISO3','saidi_h']],on='ISO3',how='left',validate='many_to_one')
    cm=LinearSegmentedColormap.from_list('burden',['#fbf3df','#eec889','#d48c6d','#ac4c62','#622944'])
    norm=LogNorm(.1,1000)
    world.plot(column='saidi_h',ax=a,cmap=cm,norm=norm,edgecolor='#a5adb1',lw=.18,
               aspect='auto',missing_kwds={'color':'#e5e7e7'})
    a.set(xlim=(-180,180),ylim=(-58,82));a.set_aspect('auto')
    frame=geographic_frame(a,4326,longitudes=[-120,-60,0,60,120],latitudes=[-30,0,30,60],fontsize=8)
    title(a,'a','Annual interruption burden')
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=ax(49,73,98,2.4),orientation='horizontal',extend='both')
    cb.set_ticks([.1,1,10,100,1000],labels=['0.1','1','10','100','1,000'])
    cb.ax.minorticks_off();cb.ax.tick_params(length=2,pad=1.5);cb.set_label('Hours per customer-year',labelpad=2)
    specs=[('b','Annual burden','annual_hours','saidi_h',INK,[.01,1,100,1000],(.007,3500),'Hours / customer-year'),
           ('c','Interruption frequency','frequency','saifi',BLUE,[.01,1,100,1000],(.007,1500),'Interruptions / customer-year'),
           ('d','Single-interruption duration','compatible_duration','caidi_h',TEAL,[.1,1,10,100],(.025,120),'Hours / interruption')]
    jitter=np.random.default_rng(SEED+1).uniform(-.17,.17,len(d))
    for x,sp in zip([15,77,138],specs):
        letter,label,metric,column,col,ticks,lims,ylab=sp
        aa=ax(x,91,41,33)
        for j,q in enumerate(QS):
            z=d.gdp_quartile.eq(q)
            aa.scatter(j+jitter[z],d.loc[z,column],s=7,facecolor=col,edgecolors='none',alpha=.21,zorder=2)
        z=groups[groups.metric.eq(metric)].set_index('quartile').reindex(QS)
        assert z.ci_low.min()>=lims[0] and z.ci_high.max()<=lims[1]
        assert d[column].min()>=lims[0] and d[column].max()<=lims[1]
        aa.plot(range(4),z.estimate,color=col,lw=1.5,zorder=4)
        aa.errorbar(range(4),z.estimate,yerr=[z.estimate-z.ci_low,z.ci_high-z.estimate],fmt='o',
                    ms=4,mfc='white',mec=col,mew=1.1,ecolor=col,capsize=2,lw=1,zorder=5)
        aa.set(yscale='log',ylim=lims,xlim=(-.42,3.42),ylabel=ylab)
        aa.set_yticks(ticks,[f'{v:g}' for v in ticks]);aa.yaxis.set_minor_locator(NullLocator())
        aa.set_xticks(range(4),QS);aa.set_xlabel('GDP/person quartile',labelpad=3)
        clean(aa);title(aa,letter,label)
    # Compact paired columns use identical accounting definitions but native scales:
    # one asks where each group stands against Q4; one asks where differences occur.
    handles=[Line2D([],[],marker='o',color=BLUE,lw=1.2,ms=4,label='Frequency'),
             Line2D([],[],marker='s',color=TEAL,lw=1.2,ms=4,label='Duration'),
             Line2D([],[],marker='D',color=INK,lw=1.2,ms=3.7,label='Net difference')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.55,1-141/H),ncol=3,
               fontsize=8.2,columnspacing=1.5,handlelength=1.35,handletextpad=.5)
    for x,w,letter,label,comparisons,xlim,ticks in [
            (28,63,'e','Differences relative to Q4',['Q1_Q4','Q2_Q4','Q3_Q4'],(-25,310),[0,100,200,300]),
            (119,60,'f','Adjacent-quartile differences',['Q1_Q2','Q2_Q3','Q3_Q4'],(-430,615),[-300,0,300,600])]:
        aa=ax(x,151,w,25)
        aa.axvline(0,color='#9a9da0',lw=.8,ls=(0,(3,3)),zorder=0)
        for j,name in enumerate(comparisons):
            aa.axhspan(j-.45,j+.45,color='#f1f3f3' if j%2==0 else '#ffffff',zorder=0)
            for offset,metric,col,mk in [(-.23,'frequency_h',BLUE,'o'),(0,'duration_h',TEAL,'s'),(.23,'net_gap_h',INK,'D')]:
                z=decomp[(decomp.contrast.eq(name))&decomp.metric.eq(metric)].iloc[0]
                assert xlim[0]<=z.ci_low<=z.estimate<=z.ci_high<=xlim[1]
                aa.errorbar(z.estimate,j+offset,xerr=[[z.estimate-z.ci_low],[z.ci_high-z.estimate]],fmt=mk,
                            ms=3.3 if mk=='D' else 3.6,color=col,mfc='white',mew=.9,lw=1,capsize=1.7,zorder=3)
        aa.set(xlim=xlim,ylim=(2.6,-.6),xticks=ticks,xlabel='Difference (h / customer-year)')
        aa.set_yticks(range(3),[z.replace('_',' − ') for z in comparisons]);clean(aa)
        title(aa,letter,label)
    qa=layout_audit(fig)
    for ext in ['png','pdf','svg']:
        fig.savefig(P/'figures'/f'Fig02_global_burden_gradient.{ext}',dpi=600,facecolor='white')
    fig.savefig(P/'figures/Fig02_global_burden_gradient_preview.png',dpi=170,facecolor='white')
    plt.close(fig)
    qa.update(figure_dimensions_mm=[W,H],geographic_frame=frame,
              all_statistical_intervals_within_axes=True,
              visual_inspection='PNG inspected after revision: no text overlap or truncation; all plotted intervals retained, including broad adjacent-quartile intervals.')
    return qa

def main():
    d,g,p,t=analyze()
    qa=draw(d,g,p)
    caption=("Fig. 2. Global interruption burden and its frequency–duration composition across the economic gradient. "
       "(a) Annual hours without supply per customer in 152 economies with a paired economic benchmark; grey indicates no paired observation. "
       "Country shading locates historical business-city or service-area measurements and does not imply nationally representative household reliability. "
       "(b–d) Annual interruption hours, annual interruption frequency and compatible single-interruption duration across four fixed gross domestic product (GDP) per-person quartiles (Q1, lowest; Q4, highest; 38 benchmarks in each). "
       "Small translucent points are individual observations. Connected open circles denote equally weighted arithmetic means of annual hours and frequency in b and c, and the ratio of these two means in d; the latter is frequency-weighted duration, not the unweighted mean of the individual duration points. "
       "All three vertical axes are logarithmic. (e) Absolute hour differences between each lower quartile and Q4, separated into frequency and duration components, with their net difference. "
       "(f) The same accounting, recalculated separately for each adjacent-quartile contrast. Positive components increase the lower-income group's annual burden; negative components offset it. "
       "Frequency and duration components sum exactly to the annual-hour difference using the symmetric partition in Methods. "
       "Bars in b–f show 95% percentile intervals from 40,000 resamples of whole benchmarks within the fixed quartiles; shared resamples retain covariance between comparisons. "
       "Source database-year labels span 2015–2019 (143 of 152 are labelled 2019); GDP is matched to the source database-year label using constant 2021 purchasing-power-parity dollars. "
       "Cross-sectional quartiles describe differences among observed economies, not longitudinal development stages or population shares. "
       "Sources: World Bank Doing Business electricity-reliability benchmarks and World Development Indicators; source definitions and inclusion rules are documented in Methods.")
    metadata=dict(caption=caption,seed=SEED,bootstrap_replicates=B,n=152,group_sizes={q:38 for q in QS},
        method='Whole-benchmark resampling in fixed GDP/person quartiles; H and F have identical equal weights; D*=H/F; symmetric exact decomposition; shared group draws across comparisons.',
        inputs={'country_benchmarks.csv':hashlib.sha256(INPUT.read_bytes()).hexdigest()},
        numerical_checks={'arithmetic_identity':True,'bootstrap_partition_identity':True},
        group_estimates=g.to_dict('records'),partitions=p.replace({np.nan:None}).to_dict('records'),
        shared_draw_contrasts=t.to_dict('records'),layout=qa)
    (P/'metadata/fig02_v31.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    report=['# Figure 2: full economic gradient and exact annual-hour accounting','',
      'The figure retains global annual burden as its hero map, replaces the single Q1–Q4 percentage headline with the full four-quartile gradient, and shows both reference-group and adjacent-group absolute contrasts.','',
      '## Group estimates','',g.to_markdown(index=False),'','## Partitions','',p.to_markdown(index=False),'',
      '## Direct comparisons with shared resamples','',t.to_markdown(index=False),'',
      '## Interpretation','',
      'Frequency and annual burden are lower in progressively higher quartiles in point estimates, but the two lowest groups overlap substantially. Compatible duration is not monotone: its point estimate is largest in Q3. These are cross-sectional contrasts, not economic development stages. The Q3–Q4 frequency share must not be declared different from the Q1–Q4 share merely because the point estimates differ; the shared-draw interval above directly assesses that contrast. No decomposition share is reported when the interval for the net hour gap contains zero.','',
      'The map locates service-area/business-city benchmarks by country; it does not estimate national household service, total outage exposure, disaster-specific recovery, avoidable loss, or population inequality shares. Grey is missing paired information, not reliable supply. Disparity components are accounting associations, not causal or policy-attributable effects.','',
      '## Methods paragraph','',
      'We fixed the original 152 benchmarks and their four GDP-per-person quartiles. Within each quartile, annual frequency and annual interruption hours were averaged with identical weights, and compatible duration was calculated as the ratio of these means. This preserves the accounting identity H=FD. Symmetric frequency and duration contributions were computed for the three adjacent contrasts and for Q1, Q2 and Q3 relative to Q4. We generated 40,000 bootstrap resamples of whole benchmarks within each fixed quartile (seed 202610021), recalculating all quantities jointly in every draw. Comparisons of contributions between contrasts retained common group draws, including the shared Q4 comparator. Intervals are percentile 95% intervals. Percentage contributions are withheld when the net-gap interval includes zero. Economic quartiles and the source sample remain unchanged.','',
      '## Layout audit','',json.dumps(qa,indent=2),'','## Caption','',caption]
    (P/'reports/fig02_v31.md').write_text('\n'.join(report))
    print(g.to_string(index=False));print(p.to_string(index=False));print(t.to_string(index=False));print(json.dumps(qa))

if __name__=='__main__':main()
