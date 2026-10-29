"""Main-text figures: one admitted population per regional argument."""
from figure_style import *
import statsmodels.api as sm
from scipy.stats import t
from matplotlib.patches import Rectangle

WORLD=R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet'
QCOL=[BLUE,'#8CAFB9','#CB9590',RED]

def data(n):return pd.read_parquet(P/'data'/f'{n}.parquet')
def source(a,n):a.drop(columns='geometry',errors='ignore').to_csv(P/'source_data'/f'{n}.csv',index=False)
def err(a,row,y,col,open=False):
 a.errorbar(row.contrast_pct,y,xerr=[[row.contrast_pct-row.contrast_lo_pct],[row.contrast_hi_pct-row.contrast_pct]],fmt='o',color=col,mfc='white' if open else col,mec='white' if not open else col,mew=.6,ms=4.6,capsize=2,lw=1.2,zorder=4)
def response(a,z,F,H,income,color,kind='frequency',normalize=False):
 X=sm.add_constant(np.log(z[income]));x=np.geomspace(z[income].min(),z[income].max(),150)
 f=sm.GLM(z[F],X,family=sm.families.Poisson()).fit();h=sm.GLM(z[H],X,family=sm.families.Poisson()).fit();px=sm.add_constant(np.log(x));yf=np.exp(px@f.params);yh=np.exp(px@h.params)
 xx=X.to_numpy();n,p=xx.shape;bread=np.zeros((4,4));scores=[]
 for i,fit in enumerate([f,h]):
  mu=np.asarray(fit.fittedvalues);bread[i*2:i*2+2,i*2:i*2+2]=np.linalg.pinv(xx.T@(mu[:,None]*xx));scores.append(xx*np.asarray(fit.resid_response)[:,None])
 score=np.hstack(scores);cov=bread@(score.T@score*n/(n-p))@bread
 L=np.column_stack([px,np.zeros_like(px)]) if kind=='frequency' else np.column_stack([-px,px])
 se=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',L,cov,L)));yy=yf if kind=='frequency' else yh/yf;critical=t.ppf(.975,n-p)
 a.fill_between(x/1000,yy*np.exp(-critical*se),yy*np.exp(critical*se),color=color,alpha=.13,lw=0)
 a.plot(x/1000,yy,color=color,lw=1.7)
 y=z[F] if kind=='frequency' else z[H]/z[F].replace(0,np.nan)
 a.scatter(z[income]/1000,y,s=25,facecolor='white',edgecolor=color,lw=.85,zorder=4)
 return pd.DataFrame({'income':x,'frequency':yf,'duration':yh/yf})
def map_base(a,crs):
 w=gpd.read_parquet(WORLD)
 if crs==5880:
  from shapely.geometry import box
  w=w.to_crs(4326);w.geometry=w.geometry.intersection(box(-85,-57,-32,15));w=w[~w.geometry.is_empty].copy()
 w=w.to_crs(crs);w.plot(ax=a,color='#ECEEEC',edgecolor='#B9C0C2',lw=.35);return w

def china():
 f=page(190);heat=data('china_income_month_heatmap');months=sorted(heat.month.unique());hm=LinearSegmentedColormap.from_list('rate',['#F5F5ED','#B5D9CE','#4E9F9B','#173F58']);norm=LogNorm(.001,.04)
 for x,key,letter,label in [(17,'all_rate','a','All interruption notices'),(120,'unplanned_rate','b','Unplanned notices')]:
  a=ax(f,x,10,82,45);z=heat.pivot(index='income_bin',columns='month',values=key).reindex(columns=months)
  a.imshow(z,origin='lower',aspect='auto',cmap=hm,norm=norm,interpolation='nearest');a.set_yticks([0,4.5,9.5,14.5,19],['0','25','50','75','100']);a.set_xticks([1,7,13,19],['Jan 2020','Jul','Jan 2021','Jul']);a.tick_params(length=0);a.set_ylabel('County income percentile');title(f,a,letter,label,offset=10)
 cb(f,hm,norm,[54,64,110,2.2],[.001,.003,.01,.03],['0.001','0.003','0.01','0.03'],'Notices per 10,000 residents per day',extend='both')
 q=data('china_absolute_class_income').query("hazard_context=='All days'");source(q,'Fig02_quartile_absolute_metrics')
 for x,key,letter,label,yl in [(18,'frequency_per10000_fullwindow','c','Frequency hides opposing gradients','Notices per\n10,000 residents'),(122,'hours_per10000_fullwindow','d','Accumulated notice-hours','Notice-hours per\n10,000 residents')]:
  a=ax(f,x,94,77,33)
  for cat,col,ls in [('All notices',INK,'-'),('Planned',BLUE,'-'),('Unplanned',RED,'-')]:
   z=q[q.category.eq(cat)].sort_values('quartile');a.plot(range(4),z[key],color=col,lw=1.5,marker='o',ms=3.5)
  a.set_xticks(range(4),['Q1','Q2','Q3','Q4']);a.set_ylim(bottom=0);a.set_xlabel('Income quartile');a.set_ylabel(yl,fontsize=9.5);clean(a);title(f,a,letter,label,offset=11)
 legend(f,[Line2D([],[],color=k,marker='o',ms=4) for k in [INK,BLUE,RED]],['All notices','Planned','Unplanned'],111,85,3,9.6)
 a=ax(f,47,154,62,21);b=ax(f,139,154,62,21);co=data('direct_process_slope_contrasts');co=co[(co.region=='China')&(co.specification=='County-equal rates + population and hazard')]
 for j,(cat,col) in enumerate([('All',INK),('Planned',BLUE),('Unplanned',RED)]):
  for aa,term in [(a,'frequency'),(b,'duration')]:
   r=co[co.contrast.eq(f'{cat} {term}')].iloc[0];err(aa,r,j,col)
 for aa,letter,label in [(a,'e','Frequency response'),(b,'f','Duration component')]:
  aa.axvline(0,color='#91999D',lw=.8,ls='--');aa.set_ylim(2.6,-.6);aa.set_xlim(-38,25);aa.set_xticks([-30,0,20]);aa.set_yticks(range(3),['All notices','Planned','Unplanned'] if aa==a else []);aa.set_xlabel('Response across income IQR (%)');clean(aa);title(f,aa,letter,label,offset=37 if aa==a else 10)
 save(f,'Fig02_China_interruption_composition')

def usa():
 f=page(170);g=gpd.read_parquet(P/'data/us_same49_map.parquet');sel=g[~g.state.isin(['AK','HI'])].to_crs(5070)
 cm1=LinearSegmentedColormap.from_list('freq',['#eef5f6','#a8ccdb','#548cae','#18496f']);cm2=LinearSegmentedColormap.from_list('hours',['#fff1d1','#e5b777','#c5675d','#7f294b'])
 for x,key,cm,letter,label,bins in [(15,'F_routine',cm1,'a','Non-major-day frequency',[0,.5,1,1.5,2,3]),(119,'H_exceptional',cm2,'b','Major-event service burden',[0,1,2,4,8,16])]:
  a=ax(f,x,10,82,48);map_base(a,5070);norm=BoundaryNorm(bins,cm.N);sel.plot(column=key,ax=a,cmap=cm,norm=norm,edgecolor='white',lw=.4);extent(a,sel.total_bounds,82/48,1.03);geographic_frame(a,5070,longitudes=[-120,-100,-80],latitudes=[30,40],fontsize=9);title(f,a,letter,label,offset=10)
  cb(f,cm,norm,[x+8,66,66,2.2],bins,[str(t) for t in bins],'Interruptions / customer-year' if key=='F_routine' else 'Hours / customer-year',extend='max')
 a=ax(f,17,106,78,47);d=data('us_same49_year_panel');yr=d.groupby('year')[['saidi_h_nonmed','saidi_h_med']].mean();a.fill_between(yr.index,yr.saidi_h_nonmed,yr.sum(axis=1),color=RED,alpha=.2,lw=0);a.plot(yr.index,yr.saidi_h_nonmed,color=BLUE,lw=1.8,marker='o',ms=3);a.plot(yr.index,yr.sum(axis=1),color=RED,lw=1.8,marker='o',ms=3);a.set_xticks([2013,2016,2019,2022,2024],['2013','16','19','22','24']);a.set_ylim(0,15);a.set_yticks([0,5,10,15]);a.set_ylabel('Hours per customer-year');a.set_xlabel('Year');clean(a);title(f,a,'c','Burden in a fixed 49-state panel',offset=10);source(yr.reset_index(),'Fig03_annual_burden')
 legend(f,[Line2D([],[],color=BLUE),Line2D([],[],color=RED)],['Non-major days','Including major days'],63,94,2,9)
 b=ax(f,150,106,51,47);m=data('continuous_panel_models');m=m[(m.region=='United States')&(m.term=='income_between')]
 for j,(cat,lbl) in enumerate([('Non-major days','Non-major'),('Major-event days','Major event')]):
  b.axhspan(j*2-.7,j*2+.7,color='#f2f3f3',zorder=0)
  for dy,out,col in [(-.2,'frequency',BLUE),(.2,'duration_component',TEAL)]:err(b,m[(m.category==cat)&(m.outcome==out)].iloc[0],j*2+dy,col)
 b.axvline(0,color='#91999d',lw=.8,ls='--');b.set_ylim(3,-1);b.set_xlim(-45,30);b.set_xticks([-40,-20,0,20]);b.set_yticks([0,2],['Non-major\ndays','Major-event\ndays']);b.set_xlabel('Income-IQR response (%)');clean(b);title(f,b,'d','Income and interruption stage',offset=32)
 legend(f,[Line2D([],[],marker='o',color=BLUE,lw=0),Line2D([],[],marker='o',color=TEAL,lw=0)],['Frequency','Duration'],166,94,2,9.3)
 save(f,'Fig03_US_routine_and_major_events')

def europe():
 f=page(183);d=data('europe_main11_system_means');yr=data('europe_main11_year_panel');a=ax(f,15,10,96,67);w=gpd.read_parquet(WORLD).to_crs(3035);g=w.merge(d[['ISO3','H_all']],on='ISO3',how='left',validate='many_to_one');cm=LinearSegmentedColormap.from_list('eu',['#e9f0ee','#b9d6d0','#78b0a9','#3c8482','#164653']);norm=LogNorm(.15,20)
 g.plot(column='H_all',ax=a,cmap=cm,norm=norm,edgecolor='white',lw=.4,missing_kwds={'color':'#eaece9'});a.set_xlim(2420000,6520000);a.set_ylim(1360000,4230000);a.set_aspect('equal');geographic_frame(a,3035,longitudes=[0,15,30],latitudes=[35,45,55],fontsize=9.3);title(f,a,'a','Annual service burden',offset=10)
 b=ax(f,150,11,51,65);order=d.sort_values('income');mat=yr.pivot(index='ISO3',columns='year',values='saidi_h_included').reindex(index=order.ISO3,columns=range(2010,2019));b.imshow(mat,cmap=cm,norm=norm,aspect='auto');b.set_facecolor('#e9eae8');b.set_yticks(range(len(order)),order.label.replace({'Czech Republic':'Czechia','Great Britain':'Britain'}),fontsize=9.2);b.set_xticks([0,4,8],['2010','14','18']);b.tick_params(length=0);title(f,b,'b','Systems ordered by GDP',offset=38)
 cb(f,cm,norm,[43,88,128,2.2],[.2,.5,1,2,5,10,20],['0.2','0.5','1','2','5','10','20'],'Hours per customer-year',extend='both');source(yr,'Fig04_same11_annual_panel')
 legend(f,[Line2D([],[],color=BLUE),Line2D([],[],color=RED)],['Routine','Exceptional'],111,112,2,10)
 for xx,kind,letter,label in [(19,'frequency','c','Economic differences in recurrence'),(124,'duration','d','A weaker duration gradient')]:
  aa=ax(f,xx,125,76,42)
  for cls,col in [('routine',BLUE),('exceptional',RED)]:
   tab=response(aa,d,'F_'+cls,'H_'+cls,'income',col,kind);source(tab,f'Fig04_{cls}_{kind}_curves')
  aa.set_xscale('log');aa.set_yscale('log');aa.set_xlim(11,88);logticks(aa,'x',[15,30,60]);aa.set_xlabel('GDP/person (thousand PPP $)')
  if kind=='frequency':aa.set_ylim(.002,15);logticks(aa,'y',[.01,.1,1,10]);aa.set_ylabel('Interruptions per\ncustomer-year')
  else:aa.set_ylim(.1,35);logticks(aa,'y',[.2,1,5,20]);aa.set_ylabel('Hours per interruption')
  clean(aa);title(f,aa,letter,label,offset=9)
 save(f,'Fig04_Europe_recurrence_and_duration')

def brazil():
 f=page(184);d=data('brazil_2021_joint_context');g=gpd.read_parquet(P/'data/brazil_same814_map.parquet').to_crs(5880);a=ax(f,13,9,103,75);w=map_base(a,5880);cm=LinearSegmentedColormap.from_list('dens',['#f1f1eb','#b9c9b9','#6b9c95','#1c6075','#123953']);norm=LogNorm(1,10000)
 g.plot(column='customer_density',ax=a,cmap=cm,norm=norm,edgecolor='none');c=g.geometry.representative_point();a.scatter(c.x,c.y,c=g.customer_density,cmap=cm,norm=norm,s=4,edgecolor='white',lw=.2,zorder=3);br=w[w.ISO3=='BRA'];extent(a,br.total_bounds,103/75,1.025);geographic_frame(a,5880,longitudes=[-70,-55,-40],latitudes=[-30,-15,0],fontsize=9.2);title(f,a,'a','Supply geography · 814 service areas',offset=8)
 cb(f,cm,norm,[25,92,80,2.2],[1,10,100,1000,10000],['1','10','100','1k','10k'],'Customers per km²',extend='both')
 b=ax(f,146,13,55,60);z=b.hexbin(d.gdp_pc_brl/1000,d.customer_density,xscale='log',yscale='log',gridsize=(20,18),mincnt=1,bins='log',cmap='Blues',linewidths=.25,edgecolors='white');b.set(xlim=(5,500),ylim=(.1,30000));logticks(b,'x',[10,30,100,300]);logticks(b,'y',[1,100,10000]);b.set_xlabel('GDP/person (thousand BRL)');b.set_ylabel('Customers per km²');clean(b);title(f,b,'b','Economic and spatial context',offset=31)
 cbar=f.colorbar(z,cax=ax(f,147,88,53,2.2),orientation='horizontal',ticks=[1,3,10,30]);cbar.ax.set_xticklabels(['1','3','10','30']);cbar.ax.minorticks_off();cbar.set_label('Service areas per hexagon',fontsize=9.5)
 m=data('zero_inclusive_models');m=m[m.region=='Brazil'];specs=['Economic context','Density adjusted','Within operator','Within operator + density']
 a3=ax(f,20,127,79,42)
 for out,col in [('frequency',BLUE),('duration_component',TEAL)]:
  z=m[(m.category=='Ordinary unplanned')&(m.term=='log_income')&(m.outcome==out)].set_index('specification').reindex(specs)
  a3.plot(range(4),z.contrast_pct,color=col,lw=1.3)
  a3.errorbar(range(4),z.contrast_pct,yerr=[z.contrast_pct-z.contrast_lo_pct,z.contrast_hi_pct-z.contrast_pct],fmt='o',ms=4,color=col,capsize=2,lw=1)
 a3.axhline(0,color='#91999d',ls='--',lw=.75);a3.set_ylim(-50,25);a3.set_yticks([-40,-20,0,20]);a3.set_xticks(range(4),['Income\nonly','+ density','Within\noperator','Operator\n+ density'],fontsize=8.8);a3.set_ylabel('Income-IQR response (%)');clean(a3);title(f,a3,'c','The economic gradient changes',offset=11)
 a4=ax(f,151,127,50,42)
 for j,cat in enumerate(['Ordinary unplanned','Planned']):
  for dy,out,col in [(-.2,'frequency',BLUE),(.2,'duration_component',TEAL)]:
   z=m[(m.category==cat)&(m.specification=='Within operator + density')&(m.term=='log_density')&(m.outcome==out)].iloc[0];err(a4,z,j*2+dy,col)
 a4.axvline(0,color='#91999d',ls='--',lw=.75);a4.set_xlim(-65,20);a4.set_ylim(3,-1);a4.set_xticks([-60,-30,0]);a4.set_yticks([0,2],['Ordinary\nunplanned','Planned']);a4.set_xlabel('Density-IQR response (%)');clean(a4);title(f,a4,'d','A process-specific spatial gradient',offset=42)
 legend(f,[Line2D([],[],marker='o',color=BLUE,ms=4),Line2D([],[],marker='o',color=TEAL,ms=4)],['Frequency','Duration'],112,115,2,10)
 save(f,'Fig05_Brazil_supply_geography')

if __name__=='__main__':
 import sys
 funcs={'china':china,'usa':usa,'europe':europe,'brazil':brazil}
 for k in (sys.argv[1:] or funcs):funcs[k]()
