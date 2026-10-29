"""Compact native Python supplementary validation figure; exact exported source rows."""
from pathlib import Path
import json,hashlib
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parents[1]
O=P/'figures';O.mkdir(exist_ok=True)
A=P/'analysis/recent_validation';D=P/'source_data'
m=pd.read_csv(D/'recent_country_economic_gradients.csv')
r=pd.read_csv(D/'recent_yearly_gap_components.csv')
m=m[m.analysis.eq('available_countries_yearly_groups') & m.outcome.isin(['frequency','duration'])]
r=r[r.analysis.isin(['available_countries_yearly_groups','strict_fixed15_sources_fixed_groups']) & r.year.gt(0)]
m.to_csv(D/'recent_figS18_panel_a.csv',index=False)
r.to_csv(D/'recent_figS18_panel_b.csv',index=False)
contract={'claim':'Across three recent yearly country samples, economic gradients are stronger for frequency than for compatible mean duration; holding the reporting sample fixed increases uncertainty in the component contrast.','archetype':'quantitative grid','backend':'Python/matplotlib','width_mm':183,'height_mm':83,'font_pt':8.5,'panels':{'a':'Paired country-level GDP-doubling frequency and mean-duration associations with same-row HC3 intervals; n changes by year.','b':'Difference between frequency- and duration-associated annual-hour components, comparing available-country and strictly fixed country/source samples.'},'statistics':'a: HC3 Student t 95% pointwise intervals (N-2 degrees of freedom); b: 40,000 within-quartile country-bootstrap percentile intervals. Fixed-country samples reuse whole-country draws across years. No multiplicity-corrected significance annotations.','exports':['PNG 600 dpi','PDF embedded TrueType','SVG editable text','PNG preview 180 dpi'],'risks':'Observations are country-level averages over reporting areas, not national household estimates. Samples overlap through time. Definitions remain source-native.'}
(A/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9.2,'xtick.labelsize':8.2,'ytick.labelsize':8.5,'legend.fontsize':8,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':3,'ytick.major.size':0,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white'})
BLUE='#35689d';RED='#b13c4e';GREY='#626b72'
fig,axes=plt.subplots(1,2,figsize=(183/25.4,83/25.4),gridspec_kw={'width_ratios':[1,1]})
fig.subplots_adjust(left=.177,right=.98,bottom=.21,top=.76,wspace=.30)
for ax in axes:
 ax.spines[['top','right','left']].set_visible(False)
 ax.axvline(0,color='#93999d',lw=.8,ls=(0,(3,3)),zorder=0)
 ax.set_ylim(-.48,2.48)
 ax.set_yticks([2,1,0])
 ax.axhspan(.5,1.5,color='#f3f4f5',zorder=-1)
for yi,y in enumerate([2021,2022,2023]):
 yp=2-yi
 for outcome,color,offset,marker in [('frequency',BLUE,.12,'o'),('duration',RED,-.12,'s')]:
  x=m[(m.year==y)&(m.outcome==outcome)].iloc[0]
  axes[0].errorbar(x.doubling_pct,yp+offset,xerr=[[x.doubling_pct-x.doubling_low],[x.doubling_high-x.doubling_pct]],fmt=marker,ms=4.2,mfc=color,mec='white',mew=.45,color=color,lw=1.1,capsize=2.3,capthick=.8,zorder=3)
 for label,color,offset,marker,face in [('available_countries_yearly_groups',BLUE,.12,'o',BLUE),('strict_fixed15_sources_fixed_groups',GREY,-.12,'s','white')]:
  x=r[(r.year==y)&(r.analysis==label)].iloc[0]
  val=x.frequency_minus_duration_h;lo=x.frequency_minus_duration_h_low;hi=x.frequency_minus_duration_h_high
  axes[1].errorbar(val,yp+offset,xerr=[[val-lo],[hi-val]],fmt=marker,ms=4.2,mfc=face,mec=color,mew=.85,color=color,lw=1.1,capsize=2.3,capthick=.8,zorder=3)
axes[0].set_yticklabels(['2021 · 29 countries','2022 · 23 countries','2023 · 18 countries'])
axes[1].set_yticklabels(['2021','2022','2023'])
axes[0].set_xlim(-80,40);axes[0].set_xticks([-80,-40,0,40])
axes[1].set_xlim(-15,105);axes[1].set_xticks([0,40,80])
axes[0].set_xlabel('Change per GDP doubling (%)',labelpad=5)
axes[1].set_xlabel('Frequency minus duration contribution\n(h/customer-year)',labelpad=5)
axes[0].set_title('Annual economic associations',loc='left',pad=29)
axes[1].set_title('Reporting-sample comparison',loc='left',pad=29)
axes[0].legend(handles=[Line2D([],[],color=BLUE,marker='o',lw=1.1,ms=4,label='Frequency'),Line2D([],[],color=RED,marker='s',lw=1.1,ms=4,label='Mean duration')],loc='lower left',bbox_to_anchor=(-.02,1.02),ncol=2,columnspacing=1.15,handlelength=1.5,handletextpad=.5,borderaxespad=0)
axes[1].legend(handles=[Line2D([],[],color=BLUE,marker='o',lw=1.1,ms=4,label='Available each year'),Line2D([],[],color=GREY,marker='s',markerfacecolor='white',lw=1.1,ms=4,label='Fixed 15 countries')],loc='lower left',bbox_to_anchor=(-.02,1.015),ncol=1,labelspacing=.3,handlelength=1.5,handletextpad=.5,borderaxespad=0)
fig.text(.025,.95,'a',fontweight='bold',fontsize=10.5,va='top')
fig.text(.535,.95,'b',fontweight='bold',fontsize=10.5,va='top')
fig.canvas.draw()
# Explicit clipping check over text artists in the exported canvas.
renderer=fig.canvas.get_renderer(); fw,fh=fig.canvas.get_width_height();outside=[]
for item in fig.findobj(mpl.text.Text):
 if not item.get_visible() or not item.get_text():continue
 box=item.get_window_extent(renderer)
 if box.x0<-1 or box.y0<-1 or box.x1>fw+1 or box.y1>fh+1:outside.append(item.get_text())
assert not outside,outside
stem=O/'SuppFig18_recent_year_validation'
for ext in ['pdf','svg']:fig.savefig(stem.with_suffix('.'+ext))
fig.savefig(stem.with_suffix('.png'),dpi=600)
fig.savefig(O/'SuppFig18_recent_year_validation_preview.png',dpi=180)
caption=("Fig. S18. Recent country-level comparisons of economic gradients and frequency–duration contributions. "
"(a) Percentage differences per doubling of gross domestic product (GDP) per capita at purchasing-power parity, estimated separately in 2021, 2022 and 2023 from 29, 23 and 18 countries. Within each country-year, frequency and annual interruption hours are averaged equally over the same admitted reporting areas; compatible mean duration is their ratio. Points show logarithmic-model estimates and bars show heteroskedasticity-consistent HC3 Student t 95% confidence intervals with N−2 degrees of freedom, where N is the number of countries. Frequency and duration models use identical observations. "
"(b) Frequency-associated minus duration-associated components of the lowest-minus-highest GDP-quartile difference in annual interruption hours. Blue circles use all admitted countries in each year and corresponding-year quartiles; open squares hold the same 21 reporting areas in 15 countries and their three-year GDP groups fixed. Positive values indicate a larger frequency component. Bars show 95% percentile intervals from 40,000 country-bootstrap draws within fixed groups; fixed-cohort draws resample whole country trajectories. Main Fig. 2 reports the full yearly hour partition. These observations summarize selected reporting areas within countries and do not estimate nationwide household reliability. Source-native interruption definitions are retained; h denotes hours.")
metadata={'figure':'Fig. S18','file_stem':stem.name,'caption':caption,'methods':'See analysis/recent_validation/ANALYSIS_PLAN.md and summary.md. No p-value threshold symbols or independent-year claim.','source_data':['recent_figS18_panel_a.csv','recent_figS18_panel_b.csv'],'figure_contract':contract,'qa':{'canvas_clipping_check':'pass','font_pt_min':8,'render_preview':'SuppFig18_recent_year_validation_preview.png','visual_review':'pending'}}
(A/'figure_caption.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(stem)
