"""Regional visual arguments with spatial, distributional and temporal evidence."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm,LogNorm,TwoSlopeNorm,LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle
from matplotlib.text import Text
from matplotlib.ticker import ScalarFormatter,FuncFormatter,NullLocator
from map_coordinates import geographic_frame

P=Path(__file__).resolve().parents[1];R=P.parents[1]
BLUE='#376F9F';RED='#B23B4D';TEAL='#248F88';GOLD='#CB973D';GREY='#8C9599';INK='#283238'
PALS=['#245963','#65A69E','#C17A63','#9182A1','#B23B4D','#CFBC98','#D7DADB']
plt.rcParams.update({'font.family':'Arial','font.size':10.5,'axes.labelsize':10.3,'axes.titlesize':10.6,
 'xtick.labelsize':9.5,'ytick.labelsize':9.5,'legend.fontsize':9.3,'legend.frameon':False,
 'axes.linewidth':.65,'axes.labelpad':3,'xtick.major.pad':2,'ytick.major.pad':2,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def dat(n):return pd.read_parquet(P/'data'/f'{n}.parquet')
def csv(n):return pd.read_csv(P/'source_data'/f'{n}.csv')
def page(h):
 f=plt.figure(figsize=(210/25.4,h/25.4),dpi=160);f._wh=(210,h);return f
def ax(f,x,top,w,h):
 W,H=f._wh;return f.add_axes([x/W,(H-top-h)/H,w/W,h/H])
def text(f,x,y,s,**kw):
 W,H=f._wh;return f.text(x/W,1-y/H,s,**kw)
def title(f,a,letter,label,offset=7):
 W,H=f._wh;b=a.get_position();text(f,b.x0*W-offset,(1-b.y1)*H-2,letter,fontweight='bold',fontsize=12,va='bottom')
 text(f,b.x0*W,(1-b.y1)*H-2,label,fontsize=10.6,va='bottom')
def clean(a,grid=False):
 a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
 if grid:a.grid(axis='y',color='#dfe3e5',lw=.45,zorder=0)
 a.tick_params(length=3,width=.65)
def logticks(a,axis,values):
 setter=a.set_xticks if axis=='x' else a.set_yticks
 setter(values,[f'{v:g}' for v in values]);(a.xaxis if axis=='x' else a.yaxis).set_minor_locator(NullLocator())
def cb(f,cm,norm,rect,ticks,labels,label,extend='neither'):
 c=f.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=ax(f,*rect),orientation='horizontal',ticks=ticks,extend=extend)
 c.ax.set_xticklabels(labels);c.ax.minorticks_off();c.set_label(label,fontsize=10);return c
def extent(a,bounds,ratio,pad=1.04):
 x0,y0,x1,y1=bounds;cx=(x0+x1)/2;cy=(y0+y1)/2;dx=(x1-x0)*pad;dy=(y1-y0)*pad
 if dx/dy>ratio:dy=dx/ratio
 else:dx=dy*ratio
 a.set(xlim=(cx-dx/2,cx+dx/2),ylim=(cy-dy/2,cy+dy/2));a.set_aspect('equal',adjustable='box')
def legend(f,handles,labels,x,y,ncol,fontsize=9.3):
 W,H=f._wh;return f.legend(handles,labels,loc='center',bbox_to_anchor=(x/W,1-y/H),ncol=ncol,fontsize=fontsize,
 handlelength=1.25,columnspacing=1.0,handletextpad=.5,borderaxespad=0)
def save(f,stem):
 f.canvas.draw();renderer=f.canvas.get_renderer();items=[]
 for t in f.findobj(Text):
  if t.get_visible() and t.get_text().strip():
   b=t.get_window_extent(renderer)
   if b.width>.1 and b.height>.1:items.append((t,b))
 collisions=[]
 for i,(t,b) in enumerate(items):
  for u,c in items[i+1:]:
   dx=min(b.x1,c.x1)-max(b.x0,c.x0);dy=min(b.y1,c.y1)-max(b.y0,c.y0)
   if dx>1.6 and dy>1.6:collisions.append([t.get_text(),u.get_text()])
 out=[t.get_text() for t,b in items if b.x0<0 or b.y0<0 or b.x1>f.bbox.width or b.y1>f.bbox.height]
 qa={'figure':stem,'minimum_font_pt':min(t.get_fontsize() for t,b in items),'text_overlap_candidates':collisions,'outside_canvas':out}
 (P/'metadata'/f'{stem}_layout.json').write_text(json.dumps(qa,indent=2))
 for ext in ['png','pdf','svg']:f.savefig(P/'figures'/f'{stem}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.025)
 f.savefig(P/'figures'/f'{stem}_preview.png',dpi=170,bbox_inches='tight',pad_inches=.025)
 plt.close(f);print(json.dumps(qa))
