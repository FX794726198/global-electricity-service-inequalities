from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

P=Path(__file__).resolve().parents[1];BASE=P.parents[1]
OUT=P/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
 'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
COL={'Urban':'#3974A2','Suburban':'#26938B','Rural':'#B63A4C'}
FREQ='#3974A2';DUR='#26938B'
monthly=pd.read_csv(P/'data/published_monthly_settlement_metrics.csv')
monthly=monthly.query("voltage=='All' and notification=='Unannounced' and settlement!='All'").copy()
monthly['date']=pd.to_datetime(dict(year=monthly.year,month=monthly.month,day=1))
annual=pd.read_csv(P/'data/published_annual_settlement_metrics.csv')
gap=pd.read_csv(P/'data/rural_urban_gap_partition.csv').query("voltage=='All' and notification=='Unannounced'")
cause=pd.read_csv(P/'data/cause_contributions_to_rural_urban_gap_2025.csv').query("voltage=='All'").copy()
reg=pd.read_csv(P/'data/province_cause_burdens_2025.csv')
reg['weather']=reg.cause_group.isin(['Wind / lightning','Ice loading'])
shares=reg[reg.weather].groupby('Il3A').customer_hours.sum()/reg.groupby('Il3A').customer_hours.sum()*100
g=gpd.read_parquet(P/'data/study_province_boundaries.parquet')
names={'Bolu':'BOLU','Düzce':'DÜZCE','Kocaeli':'KOCAELİ','Sakarya':'SAKARYA'}
g['weather_share']=g.name.map(names).map(shares)
g[['name','weather_share']].to_csv(P/'data/figure_province_weather_shares.csv',index=False)
world=gpd.read_parquet(P/'data/plot_world_context.geoparquet')

fig=plt.figure(figsize=(183/25.4,207/25.4))
axs=[]
def panel(rect,letter,title):
    ax=fig.add_axes(rect);axs.append(ax)
    ax.set_title(title,loc='left',pad=7)
    ax.text(-.105,1.075,letter,transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
    return ax

# Regional map: its colour measures a burden fraction, not data availability.
ax=panel([.065,.745,.47,.213],'a','Weather share of lost service · 2025')
world.plot(ax=ax,color='#F0F1F0',edgecolor='#B4B9BC',linewidth=.4)
cmap=LinearSegmentedColormap.from_list('weather',['#F3EBD6','#D7A786','#9C3C55'])
norm=Normalize(0,30)
g.plot(ax=ax,column='weather_share',cmap=cmap,norm=norm,edgecolor='white',linewidth=.8)
positions={'Kocaeli':(29.84,40.87),'Sakarya':(30.66,40.62),'Düzce':(31.20,40.98),'Bolu':(31.87,40.46)}
for _,r in g.iterrows():
    x,y=positions[r['name']]
    ax.text(x,y,r['name'],ha='center',va='center',fontsize=8,
            color='white' if r.weather_share>20 else '#25282A')
ax.set_xlim(29.10,32.70);ax.set_ylim(40.00,41.40)
ax.set_xticks([29.5,30.5,31.5,32.5]);ax.set_yticks([40,40.5,41])
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}°E'))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}°N'))
ax.grid(color='#BFC4C5',ls=(0,(2,3)),lw=.4,alpha=.6)
for s in ax.spines.values():s.set_visible(True)
ax.text(.02,.97,'Northwestern Türkiye',transform=ax.transAxes,va='top',fontsize=8)
cax=fig.add_axes([.105,.698,.386,.012])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[0,10,20,30])
cb.set_label('Customer-hours linked to weather (%)',labelpad=3);cb.outline.set_linewidth(.5)

ax=panel([.677,.745,.295,.213],'b','Rural / urban contrast')
for i,(var,label,color) in enumerate([('frequency_ratio','Frequency',FREQ),('duration_ratio','Duration',DUR),('burden_ratio','Annual hours','#B63A4C')]):
    r=gap.set_index('year')[var]
    ax.plot([r[2024],r[2025]],[2-i+.09,2-i-.09],color=color,lw=.9)
    ax.scatter(r[2024],2-i+.09,s=29,facecolor='white',edgecolor=color,zorder=3)
    ax.scatter(r[2025],2-i-.09,s=29,color=color,zorder=3)
ax.axvline(1,color='#A7AFB3',ls='--',lw=.7)
ax.set_yticks([2,1,0],['Frequency','Duration','Annual hours']);ax.set_ylim(-.45,2.45)
ax.set_xlim(.9,2.4);ax.set_xticks([1,1.5,2]);ax.set_xlabel('Ratio')
fig.legend([Line2D([],[],ls='',marker='o',mfc='white',mec='#414C53'),Line2D([],[],ls='',marker='o',color='#414C53')],
 ['2024','2025'],loc='center',bbox_to_anchor=(.815,.695),ncol=2,fontsize=8,handletextpad=.35,columnspacing=1.1)

for rect,letter,var,title,ylabel in [
 ([.09,.459,.402,.174],'c','frequency','Interruption frequency','Interruptions / customer / month'),
 ([.587,.459,.39,.174],'d','conditional_duration_h','Duration per interruption','Hours / customer interruption')]:
    ax=panel(rect,letter,title)
    for group,z in monthly.groupby('settlement'):
        z=z.sort_values('date');ax.plot(z.date,z[var],color=COL[group],lw=1.1)
    ax.set_ylabel(ylabel,labelpad=4)
    ticks=pd.to_datetime(['2024-01-01','2024-07-01','2025-01-01','2025-07-01'])
    ax.set_xticks(ticks,['Jan\n2024','Jul','Jan\n2025','Jul'])
    ax.axvline(pd.Timestamp('2025-01-01'),color='#C6CBCD',ls=':',lw=.6)
    ax.set_xlim(pd.Timestamp('2024-01-01'),pd.Timestamp('2025-12-25'));ax.set_ylim(bottom=0)
fig.legend([Line2D([],[],color=COL[k],lw=1.4) for k in COL],list(COL),loc='center',bbox_to_anchor=(.53,.385),ncol=3,fontsize=8,handlelength=1.8,columnspacing=1.5)

ax=panel([.225,.07,.272,.248],'e','Sources of the rural excess · 2025')
display={'Equipment / network fault':'Equipment / network','Third-party / customer damage':'Third-party damage',
 'Temporary / unspecified fault':'Temporary / unknown','Fire, unspecified origin':'Fire (unspecified)',
 'Reconciliation residual':'Residual'}
z=cause.sort_values('rural_urban_hour_gap',ascending=False)
for i,(_,r) in enumerate(z.iterrows()):
    color='#B63A4C' if r.cause_group in ['Ice loading','Wind / lightning'] else '#89979C'
    ax.barh(i,r.rural_urban_hour_gap,color=color,height=.63)
ax.set_yticks(range(len(z)),[display.get(c,c) for c in z.cause_group],fontsize=8)
ax.invert_yaxis();ax.set_xlim(-.22,7.8);ax.set_xticks([0,2,4,6]);ax.set_xlabel('Additional hours / customer-year')
ax.axvline(0,color='#889396',lw=.5)
ax.text(.985,.60,'Weather\n57% of gap',transform=ax.transAxes,va='top',ha='right',fontsize=8,color='#A32F46')
ax.text(-.595,1.075,'e',transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
# Replace the default letter, which would sit within the cause labels.
ax.texts[0].set_visible(False)

ax=panel([.627,.13,.345,.188],'f','Frequency explains most of the gap')
for i,(_,r) in enumerate(gap.sort_values('year').iterrows()):
    ax.barh(i,r.frequency_component_h,color=FREQ,height=.40)
    ax.barh(i,r.duration_component_h,left=r.frequency_component_h,color=DUR,height=.40)
    ax.text(r.frequency_component_h/2,i,f'{r.frequency_component_h:.1f}',ha='center',va='center',color='white',fontsize=8)
    ax.text(r.hour_gap+.35,i,f'{r.hour_gap:.1f}',ha='left',va='center',fontsize=8)
    ax.text(0,i+.30,f'Frequency share: {r.frequency_component_share:.0%}',fontsize=8,color=FREQ,va='center')
ax.set_yticks([0,1],['2024','2025']);ax.set_ylim(1.65,-.55);ax.set_xlim(0,28)
ax.set_xticks([0,10,20]);ax.set_xlabel('Rural − urban annual hours')
fig.legend([Patch(facecolor=FREQ),Patch(facecolor=DUR)],['Frequency','Duration'],loc='center',bbox_to_anchor=(.8,.063),ncol=2,fontsize=8,columnspacing=.9,handlelength=1)

fig.canvas.draw()
renderer=fig.canvas.get_renderer();bounds=[]
for text in fig.findobj(matplotlib.text.Text):
    if text.get_visible() and text.get_text():
        b=text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        if b.width>0 and b.height>0:bounds.append(dict(text=text.get_text(),x0=b.x0,y0=b.y0,x1=b.x1,y1=b.y1))
(P/'metadata/figure_text_bounds.json').write_text(json.dumps(bounds,indent=2))
stem='SuppFig17_turkiye_settlement_inequality'
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'{stem}.{ext}',dpi=450,bbox_inches='tight',pad_inches=.025)
print('Saved',stem)
print('Mapped weather shares:',g[['name','weather_share']].to_dict('records'))
