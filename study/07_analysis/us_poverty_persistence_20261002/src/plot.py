"""Compact county geography and poverty contrasts; all panels source-backed."""
from pathlib import Path
import json
import numpy as np,pandas as pd,geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm,LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
P=Path(__file__).resolve().parents[1];B=P.parents[1];D=P/'data';F=P/'figures'
BLUE='#376a9f';RED='#b73d50';GOLD='#c69237';GREY='#9caaaf'
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.titlesize':10.7,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def label(fig,letter,x,y,title):
    fig.text(x-.028,y,letter,weight='bold',fontsize=12,va='bottom');fig.text(x,y,title,fontsize=10.7,va='bottom')
def clean(ax):ax.spines[['top','right']].set_visible(False)
def main():
    g=gpd.read_parquet(D/'county_contrasts.geoparquet').to_crs(4326)
    g['duration_ratio']=np.exp(g.log_duration_ratio)
    world=gpd.read_parquet(D/'plot_world_context.geoparquet')
    state=gpd.read_parquet(D/'plot_state_boundaries.geoparquet')
    fig=plt.figure(figsize=(8.4,6.9))
    cm_p=LinearSegmentedColormap.from_list('poverty',['#eff4f2','#a9d0c7','#4b9c92','#124e61'])
    cm_r=LinearSegmentedColormap.from_list('ratio',['#386895','#a9c5d3','#f7f7f3','#e6b29b','#a33849'])
    for letter,x,col,title,cm,norm,ticks,labels in [
        ('a',.07,'poverty_pct','County poverty',cm_p,Normalize(0,35),[0,10,20,30],None),
        ('b',.565,'log_duration_ratio','Weather-associated persistence',cm_r,TwoSlopeNorm(0,-np.log(8),np.log(8)),np.log([.125,.5,1,2,8]),['⅛','½','1','2','8'])]:
        ax=fig.add_axes([x,.59,.405,.335])
        world.plot(ax=ax,facecolor='#eeeeee',edgecolor='#bfc4c6',linewidth=.45,zorder=0)
        g.plot(column=col,ax=ax,cmap=cm,norm=norm,edgecolor='white',linewidth=.055,aspect=None,zorder=2)
        state.boundary.plot(ax=ax,edgecolor='#9aadb0',linewidth=.25,zorder=3,aspect=None)
        ax.set(xlim=(-126,-66),ylim=(24,50),xticks=[-120,-100,-80],yticks=[30,40,50])
        ax.set_aspect(1.26)
        label(fig,letter,x,ax.get_position().y1+.016,title)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{abs(v):g}°W'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}°N'))
        ax.grid(ls=(0,(2,3)),color='#c0c6c8',lw=.4,zorder=1)
        for spine in ax.spines.values():spine.set_color('#697579')
        ca=fig.add_axes([x+.025,.546,.355,.013]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=ca,orientation='horizontal',ticks=ticks,extend='max' if letter=='a' else 'both')
        if labels:cb.ax.set_xticklabels(labels)
        cb.ax.tick_params(labelsize=8.7,pad=1);cb.set_label('Residents below poverty threshold (%)' if letter=='a' else 'Weather / unlinked mean duration',labelpad=3,size=9);cb.outline.set_linewidth(.6)
    # c: two absolute quantities on parallel aligned axes; no dual y-axis.
    label(fig,'c',.075,.434,'Interruption mixture and duration')
    groups=pd.read_csv(D/'poverty_group_composition.csv')
    duration=fig.add_axes([.14,.13,.205,.255]);share=fig.add_axes([.40,.13,.075,.255])
    for i,qt in enumerate(['Q1','Q2','Q3','Q4']):
        z=groups[groups.quartile.eq(qt)].set_index('weather')
        lo=float(z.loc[False,'class_duration_h']);we=float(z.loc[True,'class_duration_h']);sh=float(z.loc[True,'class_share'])*100
        duration.plot([lo,we],[i,i],color='#b4bcc0',lw=1.5,zorder=1)
        duration.scatter([lo,we],[i,i],s=28,c=[BLUE,RED],edgecolor='white',linewidth=.5,zorder=3)
        share.barh(i,sh,color='#a7b6c0',height=.35)
        share.text(sh+.5,i,f'{sh:.1f}',va='center',fontsize=9)
    duration.set(xlim=(0,8),xticks=[0,2,4,6,8],yticks=range(4),yticklabels=['Q1: lowest','Q2','Q3','Q4: highest'],ylim=(3.6,-.6),xlabel='Mean duration (h)')
    duration.set_ylabel('County poverty group');clean(duration)
    share.set(xlim=(0,19),xticks=[0,10],yticks=[],ylim=(3.6,-.6),xlabel='Share (%)');clean(share)
    duration.legend(handles=[Line2D([],[],marker='o',color=RED,ls='',label='Weather link'),Line2D([],[],marker='o',color=BLUE,ls='',label='Unlinked')],frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(-.33,1.04),columnspacing=.8,handletextpad=.35)
    share.set_title('Weather',pad=8,fontsize=9.3)
    # d: findings from the two predeclared comparison populations.
    label(fig,'d',.565,.434,'Poverty and conditional persistence')
    models=pd.read_csv(D/'poverty_duration_models.csv');ax=fig.add_axes([.745,.13,.225,.255])
    selected=[('County-year: Primary','Unadjusted','log_duration_ratio','County–year\nUnadjusted',BLUE),('County-year: Primary','Population and density','log_duration_ratio','County–year\nAdjusted',BLUE),('Within published weather event','Same event','log_mean_h','Same event\nUnadjusted',RED),('Within published weather event','Same event, population and density','log_mean_h','Same event\nAdjusted',RED)]
    ylabels=[]
    for i,(comp,spec,outcome,lab,color) in enumerate(selected):
        r=models[(models.comparison.eq(comp))&(models.specification.eq(spec))&(models.outcome.eq(outcome))].iloc[0]
        if i%2==1:ax.axhspan(i-.48,i+.48,color='#f1f3f3',zorder=0)
        ax.errorbar(r.contrast,i,xerr=[[r.contrast-r.lower],[r.upper-r.contrast]],fmt='o',color=color,ms=4.6,lw=1.4,capsize=2,zorder=3)
        ylabels.append(lab)
    ax.axvline(0,color='#929a9c',ls=(0,(3,3)),lw=.8)
    ax.set(xlim=(-15,5),xticks=[-15,-10,-5,0,5],yticks=range(4),yticklabels=ylabels,ylim=(3.6,-.6),xlabel='Duration contrast (%)\nper +5 pp county poverty')
    clean(ax);ax.tick_params(axis='y',length=0,pad=6)
    # Draw and check full-figure text extents. Visual panel review remains necessary.
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for txt in fig.findobj(matplotlib.text.Text):
        if not txt.get_visible() or not txt.get_text():continue
        box=txt.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:outside.append(txt.get_text())
    assert not outside,outside
    stem='US_poverty_weather_persistence'
    for ext in ['png','pdf','svg']:fig.savefig(F/(stem+'.'+ext),dpi=400,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)
    (P/'metadata/figure_export.json').write_text(json.dumps(dict(figure=stem,map_counties=len(g),formats=['png','pdf','svg'],outside_canvas_text=outside,figure_size_inches=[8.4,6.9],source_tables=['county_contrasts.geoparquet','poverty_group_composition.csv','poverty_duration_models.csv']),indent=2))
    print('Exported',stem,flush=True)
if __name__=='__main__':main()
