"""Global electricity-service inequality: preserved maps plus matched GDP accounting.

Frozen estimates are plotted without refitting or altering observations. Python-only.
"""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullLocator
from matplotlib.lines import Line2D
from matplotlib.text import Text

P=Path(__file__).resolve().parents[1]
BASE=P.parent
V33=BASE/'20261003_global_blackout_inequalities_v33'
V37=BASE/'20261003_global_blackout_inequalities_v37'
STEM='Fig01_global_economic_service_inequality'
W,H=198,212
BLUE,DURATION,INK='#376F9F','#BB8735','#30383D'
SEED=202610021

def freeze_inputs():
    sources={
        'demand':V33/'source_data/review_demand_multiyear_2000_2024.csv',
        'reliability':V33/'source_data/review_reliability_frequency_recovery_latest.csv',
        'benchmarks':V33/'source_data/gradient_v31_benchmark_inputs.csv',
        'groups':V33/'source_data/gradient_v31_group_estimates.csv',
        'partitions':V33/'source_data/gradient_v31_absolute_partitions.csv',
        'recent':V37/'source_data/recent_yearly_gap_components.csv',
        'geometry':V33/'core_reproducibility/inputs/world_geometry.geoparquet',
        **{f'joint_{k}':V33/f'source_data/review_joint_{k}_gdp_cells.csv' for k in ['demand','frequency','recovery']}
    }
    manifest=[];paths={}
    for key,src in sources.items():
        dst=P/'source_data'/f'merged_fig1_{key}{src.suffix}'
        shutil.copyfile(src,dst);paths[key]=dst
        manifest.append(dict(role=key,source=str(src),frozen=str(dst.relative_to(P)),sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size))
    (P/'metadata/merged_fig1_input_manifest.json').write_text(json.dumps(manifest,indent=2))
    return paths

def lon(v,pos=None):return '0°' if v==0 else f'{abs(v):g}°'+('W' if v<0 else 'E')
def lat(v,pos=None):return '0°' if v==0 else f'{abs(v):g}°'+('S' if v<0 else 'N')

def main():
    for folder in ['figures','source_data','metadata','analysis']:(P/folder).mkdir(exist_ok=True,parents=True)
    paths=freeze_inputs()
    demand=pd.read_csv(paths['demand']);rel=pd.read_csv(paths['reliability'])
    d=pd.read_csv(paths['benchmarks']);groups=pd.read_csv(paths['groups']);decomp=pd.read_csv(paths['partitions'])
    recent=pd.read_csv(paths['recent']).query("analysis == 'available_countries_yearly_groups'").set_index('year')
    world=gpd.read_parquet(paths['geometry']).to_crs(4326)
    world=world[world.ISO3.ne('ATA')].copy();world.geometry=world.geometry.simplify(.035,preserve_topology=True)
    assert len(d)==152 and d.ISO3.nunique()==152
    plt.rcParams.update({'font.family':'Arial','font.size':8.8,'axes.labelsize':8.6,'xtick.labelsize':8.4,'ytick.labelsize':8.4,
        'axes.linewidth':.65,'axes.labelpad':2.8,'xtick.major.pad':2,'ytick.major.pad':2,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=170)
    axes_meta=[];geometry=[];checks=[]
    def ax(x,top,w,h,name):
        aa=fig.add_axes([x/W,1-(top+h)/H,w/W,h/H]);axes_meta.append(dict(name=name,x_mm=x,top_mm=top,width_mm=w,height_mm=h))
        return aa
    def title(aa,letter,label,offset=1.6):
        b=aa.get_position();letter_offset=offset+(3.65 if '\n' in label else 0);fig.text(b.x0-4.2/W,b.y1+letter_offset/H,letter,fontsize=10.2,weight='bold',va='bottom')
        # Panel title lines omitted in the submitted figure; keep panel letters.
        pass
    def clean(aa):aa.spines[['top','right']].set_visible(False);aa.set_axisbelow(True)
    specs=[
      dict(code='demand',letters=('a','b'),data=demand,y='demand_mwh_pc',x='mean_gdp_ppp_pc_constant_2021',
        color='#3C8279',title='Mean electricity demand · 2000–2024',label='Demand (MWh/person)',ylabel='Demand (MWh/person)',
        edges=[0,.5,1,2,5,10,20,80],colors=['#f0f7f5','#d9ebe5','#b5d6cb','#8cbcaf','#629e91','#3c8279','#20574f'],
        ticks=[.5,1,2,5,10,20],yedges=np.geomspace(.01,100,7),yticks=[.1,1,10,100]),
      dict(code='frequency',letters=('d','e'),data=rel,y='saifi',x='gdp_ppp_pc_at_year',
        color='#376F9F',title='Interruption frequency · 2014–2019',label='Interruption frequency (per customer-year)',ylabel='Interruption frequency\n(per customer-year)',
        edges=[0,.5,1,3,10,30,100,2000],colors=['#eef4f9','#d8e7f1','#b6d1e4','#88b4d2','#6095be','#376f9f','#17486d'],
        ticks=[.5,1,3,10,30,100],yedges=np.geomspace(.008,2000,7),yticks=[.01,1,100]),
      dict(code='recovery',letters=('g','h'),data=rel,y='recovery_h',x='gdp_ppp_pc_at_year',
        color='#BB8735',title='Mean interruption duration · 2014–2019',label='Mean duration (hours/interruption)',ylabel='Mean duration\n(hours/interruption)',
        edges=[0,.5,1,2,4,8,24,100],colors=['#fcf7ed','#f4e7c9','#e8d09b','#d6b568','#bb8735','#926621','#644819'],
        ticks=[.5,1,2,4,8,24],yedges=np.geomspace(.1,100,7),yticks=[.1,1,10,100])]
    qs=['Q1','Q2','Q3','Q4'];jitter=np.random.default_rng(SEED+1).uniform(-.17,.17,len(d))
    group_specs=[('c','Annual interruption\nhours','annual_hours','saidi_h',INK,[.01,1,100,1000],(.007,3500),'Hours / customer-year'),
                 ('f','Interruption\nfrequency','frequency','saifi',BLUE,[.01,1,100,1000],(.007,1500),'Interruptions / customer-year'),
                 ('i','Mean interruption\nduration','compatible_duration','caidi_h',DURATION,[.1,1,10,100],(.025,120),'Hours / interruption')]
    for row,(s,gs) in enumerate(zip(specs,group_specs)):
        top0=10+row*53
        q,val,gx=s['data'],s['y'],s['x']
        cmap=ListedColormap(s['colors']);norm=BoundaryNorm(s['edges'],cmap.N)
        mp=ax(11.5,top0,80,31.5,s['letters'][0])
        ww=world.merge(q[['ISO3',val]],on='ISO3',how='left',validate='many_to_one')
        ww.plot(column=val,ax=mp,cmap=cmap,norm=norm,linewidth=.22,edgecolor='#a7aaad',missing_kwds={'color':'#e8e8e8'},aspect='auto')
        mp.set(xlim=(-180,180),ylim=(-60,80));mp.set_aspect('auto')
        mp.set_xticks([-120,-60,0,60,120]);mp.set_yticks([-30,0,30,60]);mp.xaxis.set_major_formatter(FuncFormatter(lon));mp.yaxis.set_major_formatter(FuncFormatter(lat));mp.grid(lw=.38,color='#cecece',ls='--',alpha=.8)
        title(mp,s['letters'][0],s['title'])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=ax(22.5,top0+37.3,59,1.65,s['letters'][0]+'_colorbar'),orientation='horizontal',ticks=s['ticks'],spacing='uniform')
        cb.ax.set_xticklabels([f'{x:g}' for x in s['ticks']]);cb.ax.minorticks_off();cb.ax.tick_params(length=2,pad=1.4);cb.set_label(s['label'],labelpad=2.1)
        geometry.extend(dict(panel=s['letters'][0],ISO3=r.ISO3,value=float(getattr(r,val)),geometry_present=r.ISO3 in set(world.ISO3)) for r in q[['ISO3',val]].dropna().itertuples(index=False))
        ja=ax(108,top0+6.5,29,25,s['letters'][1]);marg_top=ax(108,top0-1.6,29,4.5,s['letters'][1]+'_top');side=ax(139.0,top0+6.5,7.5,25,s['letters'][1]+'_side')
        paired=q[q[gx].gt(0)&q[val].gt(0)&np.isfinite(q[gx])&np.isfinite(q[val])].copy()
        xe,ye=np.geomspace(500,200000,7),s['yedges'];z,_,_=np.histogram2d(paired[gx],paired[val],bins=[xe,ye]);assert z.sum()==len(paired)
        frozen=pd.read_csv(paths['joint_'+s['code']]);assert np.array_equal(z.flatten(),frozen.countries.to_numpy())
        cm=LinearSegmentedColormap.from_list('joint',['#f7f8f8','#adb5bb'])
        ja.pcolormesh(xe,ye,np.ma.masked_where(z.T==0,z.T),cmap=cm,vmin=0,vmax=max(z.max(),1),edgecolors='white',lw=.6)
        for i,j in zip(*np.where(z>0)):ja.text(np.sqrt(xe[i]*xe[i+1]),np.sqrt(ye[j]*ye[j+1]),str(int(z[i,j])),ha='center',va='center',fontsize=8.3,color='#243840')
        ja.set_xscale('log');ja.set_yscale('log');ja.set(xlim=(xe[0],xe[-1]),ylim=(ye[0],ye[-1]),xlabel='GDP/person (PPP $)',ylabel=s['ylabel'])
        ja.set_xticks([1000,10000,100000],['1k','10k','100k']);ja.set_yticks(s['yticks'],[f'{v:g}' for v in s['yticks']]);ja.minorticks_off();ja.xaxis.label.set_fontsize(8.3)
        xx,yy=np.sqrt(xe[:-1]*xe[1:]),np.sqrt(ye[:-1]*ye[1:]);marg_top.plot(xx,100*z.sum(1)/len(paired),color=s['color'],lw=1.05);marg_top.fill_between(xx,0,100*z.sum(1)/len(paired),color=s['color'],alpha=.11,lw=0)
        marg_top.set_xscale('log');marg_top.set(xlim=ja.get_xlim(),ylim=(0,55));marg_top.set_xticks([]);marg_top.set_yticks([0,50]);marg_top.set_ylabel('%',rotation=0,labelpad=5.2);marg_top.minorticks_off()
        side.plot(100*z.sum(0)/len(paired),yy,color=s['color'],lw=1.05);side.fill_betweenx(yy,0,100*z.sum(0)/len(paired),color=s['color'],alpha=.11,lw=0)
        side_max=max(55,float(np.ceil((100*z.sum(0)/len(paired)).max()/25)*25));side.set_yscale('log');side.set(ylim=ja.get_ylim(),xlim=(0,side_max));side.set_yticks([]);side.set_xticks([0,50] if side_max==55 else [0,side_max]);side.set_xlabel('%');side.minorticks_off()
        title(marg_top,s['letters'][1],f'Joint counts · n = {len(paired)}',offset=1.7)
        if s['code'] in ['frequency','recovery']:
            rho=paired[gx].rank().corr(paired[val].rank())
            assert len(paired)==152
            b=marg_top.get_position()
            fig.text(b.x0+0.007,b.y1+1.7/H,f'Spearman ρ = {rho:.2f}'.replace('-', '−'),fontsize=8.3,color=INK,va='bottom')
        checks.append(dict(panel=s['letters'][1],observations=len(paired),joint_cells_match_exactly=True,marginal_sum_pct=float(z.sum()/len(paired)*100)))
        letter,label,metric,column,col,ticks,lims,ylab=gs
        aa=ax(165,top0+3.3,30.5,28.2,letter)
        for j,qname in enumerate(qs):
            qq=d.gdp_quartile.eq(qname);aa.scatter(j+jitter[qq],d.loc[qq,column],s=6,facecolor=col,edgecolors='none',alpha=.21,zorder=2)
        gg=groups[groups.metric.eq(metric)].set_index('quartile').reindex(qs)
        assert gg.ci_low.min()>=lims[0] and gg.ci_high.max()<=lims[1];assert d[column].min()>=lims[0] and d[column].max()<=lims[1]
        aa.plot(range(4),gg.estimate,color=col,lw=1.25,zorder=4);aa.errorbar(range(4),gg.estimate,yerr=[gg.estimate-gg.ci_low,gg.ci_high-gg.estimate],fmt='o',ms=3.2,mfc='white',mec=col,mew=1,ecolor=col,capsize=1.8,lw=.9,zorder=5)
        aa.set(yscale='log',ylim=lims,xlim=(-.42,3.42),ylabel=ylab);aa.set_yticks(ticks,[f'{v:g}' for v in ticks]);aa.yaxis.set_minor_locator(NullLocator());aa.set_xticks(range(4),qs);aa.set_xlabel('GDP/person quartile',labelpad=3);aa.xaxis.label.set_fontsize(8.2)
        clean(aa);title(aa,letter,label);checks.append(dict(panel=letter,all_observations=152,group_size=38,groups=4,all_intervals_and_points_visible=True,estimates='Frozen source estimates; no reanalysis'))
    handles=[Line2D([],[],marker='o',color=BLUE,lw=1.1,ms=3.6,label='Frequency component'),Line2D([],[],marker='s',color=DURATION,lw=1.1,ms=3.6,label='Duration component'),Line2D([],[],marker='D',color=INK,lw=1.1,ms=3.4,label='Annual-hour gap')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.56,1-167.2/H),ncol=3,fontsize=8.8,columnspacing=1.8,handlelength=1.5,handletextpad=.5)
    for x,w,letter,label,comparisons,xlim,ticks in [(23,66,'j','Historical differences relative to Q4',['Q1_Q4','Q2_Q4','Q3_Q4'],(-25,310),[0,100,200,300]),(129,66.5,'k','Recent country comparisons',[2021,2022,2023],(-25,90),[0,40,80])]:
        aa=ax(x,176,w,26,letter);aa.axvline(0,color='#9a9da0',lw=.8,ls=(0,(3,3)),zorder=0)
        for j,name in enumerate(comparisons):
            aa.axhspan(j-.45,j+.45,color='#f1f3f3' if j%2==0 else '#ffffff',zorder=0)
            for offset,metric,col,mk in [(-.23,'frequency_h',BLUE,'o'),(0,'duration_h',DURATION,'s'),(.23,'net_gap_h',INK,'D')]:
                if letter=='k':
                    rr=recent.loc[name];key='gap_h' if metric=='net_gap_h' else metric;zz=pd.Series(dict(estimate=rr[key],ci_low=rr[key+'_low'],ci_high=rr[key+'_high']))
                else:zz=decomp[(decomp.contrast.eq(name))&decomp.metric.eq(metric)].iloc[0]
                assert xlim[0]<=zz.ci_low<=zz.estimate<=zz.ci_high<=xlim[1]
                aa.errorbar(zz.estimate,j+offset,xerr=[[zz.estimate-zz.ci_low],[zz.ci_high-zz.estimate]],fmt=mk,ms=3.3 if mk=='D' else 3.6,color=col,mfc='white',mew=.9,lw=.95,capsize=1.7,zorder=3)
        aa.set(xlim=xlim,ylim=(2.6,-.6),xticks=ticks,xlabel='Difference (h / customer-year)')
        aa.set_yticks(range(3),[f'{yr} (n = {int(recent.loc[yr,"countries"])})' for yr in comparisons] if letter=='k' else [z.replace('_',' − ') for z in comparisons]);clean(aa);title(aa,letter,label)
        checks.append(dict(panel=letter,comparisons=3,components=3,all_intervals_visible=True,estimates='Frozen source estimates; no reanalysis'))
    fig.canvas.draw();renderer=fig.canvas.get_renderer();items=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():continue
        b=t.get_window_extent(renderer)
        if b.width>.1 and b.height>.1:items.append((t,b))
    overlaps=[]
    for i,(ta,ba) in enumerate(items):
        for tb,bb in items[i+1:]:
            dx=min(ba.x1,bb.x1)-max(ba.x0,bb.x0);dy=min(ba.y1,bb.y1)-max(ba.y0,bb.y0)
            if dx>1.5 and dy>1.5:overlaps.append(dict(first=ta.get_text(),second=tb.get_text(),overlap_px=[round(dx,2),round(dy,2)]))
    outside=[t.get_text() for t,b in items if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height]
    qa=dict(figure_dimensions_mm=[W,H],minimum_font_pt=min(t.get_fontsize() for t,b in items),minimum_font_at_180mm=min(t.get_fontsize() for t,b in items)*180/W,overlap_candidates=overlaps,outside_canvas=outside,axes=axes_meta,statistical_checks=checks,visual_review='pending')
    for ext in ['png','pdf','svg']:fig.savefig(P/'figures'/f'{STEM}.{ext}',dpi=600,facecolor='white')
    fig.savefig(P/'figures'/f'{STEM}_preview.png',dpi=180,facecolor='white');plt.close(fig)
    pd.DataFrame(geometry).to_csv(P/'source_data/merged_fig1_geometry_join_audit.csv',index=False)
    (P/'metadata/merged_fig1_layout_qa.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps({k:v for k,v in qa.items() if k!='axes'},indent=2))

if __name__=='__main__':main()
