from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import geopandas as gpd
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
 'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
BLUE='#356CA2';TEAL='#2B948C';RED='#B24959';GRAY='#D9DEE1';INK='#28363C'
order=['baskent','sedas','uludag','toroslar']
labels=['A  Ankara–Black Sea','B  Bolu–Sakarya','C  Southern Marmara','D  E. Mediterranean']
data=pd.read_csv(P/'data/key_regional_results_2025.csv').set_index('provider').loc[order]
g=gpd.read_parquet(P/'data/turkiye_province_boundaries.parquet')
selected=gpd.read_parquet(P/'data/study_province_boundaries.parquet')
f=plt.figure(figsize=(183/25.4,149/25.4))
def panel(box,letter,title,title_y=None):
 a=f.add_axes(box);pos=a.get_position()
 y=pos.y1+.014 if title_y is None else title_y
 f.text(pos.x0,y,title,fontsize=10,va='bottom')
 f.text(.013 if letter=='a' else pos.x0-.019,y,letter,weight='bold',fontsize=11,va='bottom')
 return a
a=panel([.075,.612,.895,.340],'a','Rural excess interruption hours · Türkiye, 2025')
g.plot(ax=a,color='#E9EBE9',edgecolor='#C0C5C5',lw=.35,aspect='auto')
cm=LinearSegmentedColormap.from_list('gap',['#F3EAD7','#DCA777','#B55662','#732D4D']);norm=Normalize(0,80)
selected.plot(column='regional_hour_gap',ax=a,cmap=cm,norm=norm,edgecolor='white',lw=.45,aspect='auto')
# Country outlines and graticules frame the actual geography; each fill uses
# the regional reporting estimate, not an invented provincial inequality.
a.set(xlim=(25.6,45),ylim=(35.6,42.4));a.set_aspect('auto')
a.set_xticks([28,32,36,40,44]);a.set_yticks([36,39,42])
a.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}°E'))
a.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}°N'))
a.tick_params(top=True,right=True,direction='out',length=3)
a.grid(color='#BFC6C8',lw=.35,ls=(0,(2,3)),alpha=.7)
for s in a.spines.values():s.set_visible(True);s.set_color('#778489')
for label,x,y,tx,ty in [('A',33.3,40.6,35.0,41.9),('B',30.9,40.8,29.4,42.03),('C',28.3,39.8,27.1,37.6),('D',36.4,37.1,39.2,37.0)]:
 a.annotate(label,(x,y),xytext=(tx,ty),fontsize=10,weight='bold',ha='center',va='center',
            bbox=dict(boxstyle='circle,pad=.18',fc='white',ec='none'),arrowprops=dict(arrowstyle='-',lw=.65,color=INK))
cax=f.add_axes([.28,.539,.46,.014]);cb=f.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation='horizontal',ticks=[0,20,40,60,80]);cb.set_label('Rural minus urban hours per customer-year',labelpad=3);cb.outline.set_linewidth(.5)

b=panel([.305,.067,.197,.252],'b','Group contrast',title_y=.414)
for i,(_,r) in enumerate(data.iterrows()):
 b.plot([r.duration_ratio,r.frequency_ratio],[i,i],color='#C1C9CD',lw=1.1,zorder=1)
 b.scatter(r.frequency_ratio,i,s=26,color=BLUE,zorder=3)
 b.scatter(r.duration_ratio,i,s=27,color=TEAL,marker='s',zorder=3)
b.axvline(1,color='#8E999F',ls='--',lw=.6)
b.set(ylim=(3.5,-.5),xlim=(.85,5.3),xlabel='Rural / urban');b.set_xticks([1,3,5]);b.set_yticks(range(4),labels,fontsize=8.7);b.tick_params(axis='y',length=0)
f.legend([Line2D([],[],marker='o',color=BLUE,lw=0,ms=4),Line2D([],[],marker='s',color=TEAL,lw=0,ms=4)],['Frequency','Duration'],loc='center',bbox_to_anchor=(.52,.367),ncol=2,fontsize=9,handletextpad=.35,columnspacing=1.1)

c=panel([.535,.067,.197,.252],'c','Gap partition',title_y=.414)
for i,(_,r) in enumerate(data.iterrows()):
 c.barh(i,r.frequency_component_h,color=BLUE,height=.58)
 c.barh(i,r.duration_component_h,left=r.frequency_component_h,color=TEAL,height=.58)
 c.text(r.hour_gap+1,i,f'{r.frequency_share_pct:.0f}%',va='center',fontsize=9,color=INK)
c.set(xlim=(0,107),ylim=(3.5,-.5),xlabel='Excess hours');c.set_xticks([0,40,80]);c.set_yticks(range(4),['']*4);c.tick_params(axis='y',length=0)

d=panel([.766,.067,.204,.252],'d','Weather share',title_y=.414)
sub=data[data.weather_gap_h.notna()]
for provider,r in sub.iterrows():
 i=order.index(provider)
 d.barh(i,r.weather_gap_h,color=RED,height=.58)
 d.barh(i,r.hour_gap-r.weather_gap_h,left=r.weather_gap_h,color=GRAY,height=.58)
 d.text(r.hour_gap+1,i,f'{r.weather_share_of_gap_pct:.0f}%',va='center',fontsize=9,color=RED)
d.text(3,2,'Not separately\ncoded',color='#7F898E',fontsize=8.4,va='center')
d.set(xlim=(0,107),ylim=(3.5,-.5),xlabel='Excess hours');d.set_xticks([0,40,80]);d.set_yticks(range(4),['']*4);d.tick_params(axis='y',length=0)
f.legend([Patch(color=RED),Patch(color=GRAY)],['Weather','Remaining'],loc='center left',bbox_to_anchor=(.766,.368),ncol=1,fontsize=8.7,handlelength=1.0,labelspacing=.15,handletextpad=.4,borderaxespad=0)
for a in [b,c,d]:
 for yy in [.5,1.5,2.5]:a.axhline(yy,color='#E4E8EA',lw=.4,zorder=0)

for extension in ['png','pdf','svg']:
 f.savefig(P/'figures'/('SuppFig22_turkiye_multiregion_validation.'+extension),dpi=600,bbox_inches='tight',pad_inches=.025)
f.savefig(P/'figures/SuppFig22_turkiye_multiregion_validation_preview.png',dpi=190,bbox_inches='tight',pad_inches=.025)
f.canvas.draw()
text_boxes=[]
from matplotlib.text import Text
for t in f.findobj(Text):
 if t.get_visible() and t.get_text():
  bbox=t.get_window_extent(f.canvas.get_renderer())
  text_boxes.append(dict(text=t.get_text(),font_size_pt=t.get_fontsize(),bbox=list(bbox.extents)))
overlaps=[]
for i,a in enumerate(text_boxes):
 for b in text_boxes[i+1:]:
  A=a['bbox'];B=b['bbox'];dx=min(A[2],B[2])-max(A[0],B[0]);dy=min(A[3],B[3])-max(A[1],B[1])
  if dx>.3 and dy>.3:overlaps.append(dict(a=a['text'],b=b['text'],overlap_px=[dx,dy]))
(P/'metadata/figure_render.json').write_text(json.dumps(dict(figure='SuppFig22_turkiye_multiregion_validation',width_mm=183,height_mm=149,minimum_font_pt=min(x['font_size_pt'] for x in text_boxes),text=text_boxes,text_overlaps=overlaps),indent=2))
assert not overlaps,overlaps
plt.close(f)
