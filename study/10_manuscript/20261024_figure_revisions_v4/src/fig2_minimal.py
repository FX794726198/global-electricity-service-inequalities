"""Fig. 2 minimal revision (v5): original v37 layout for panel f, plus small markers for leading-cause switches.
Changes relative to the original: (1) diamonds left of the first panel mark the 19 profiles whose leading
category differs between occurrence and time/energy shares; (2) Turkish profile labels use full region names.
No bold labels, no extra numeric column, no in-figure footnotes; maps a-e are unchanged."""
from pathlib import Path
import sys, json, itertools, hashlib
P=Path(__file__).resolve().parents[1]
B=P.parent/'20261009_global_blackout_inequalities_v40'
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Wedge,Circle
from matplotlib.offsetbox import DrawingArea,AnnotationBbox
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.patheffects as pe
S=B/'source_data';F=P/'figures';Q=P/'qa'
sp=S/'fig2_profiles.csv';qp=S/'fig2_shares.csv'
s=pd.read_csv(sp if sp.exists() else S/'profiles_original.csv').set_index('profile_id')
q=pd.read_csv(qp if qp.exists() else S/'paired_shares_original.csv')
q['occurrence_share_normalized_pct']=100*q.occurrence_share_pct/q.groupby('profile_id').occurrence_share_pct.transform('sum')
if 'paired' not in s:s['paired']=True
s['paired']=s.paired.fillna(True).astype(bool)
world=gpd.read_parquet(P.parent/'20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
world=world[world.ISO3.ne('ATA')].copy();world.geometry=world.geometry.simplify(.03,preserve_topology=True)
br=gpd.read_parquet(S/'brazil.geoparquet').to_crs(4326)
eu=gpd.read_parquet(S/'nordic.geoparquet').to_crs(4326)
ny=gpd.read_parquet(S/'ny.geoparquet').to_crs(4326)
CA=['Weather','Other environment','Equipment / electrical','Human / operation','Supply constraint','Planned','Other specified','Other / unresolved']
COL=dict(zip(CA,['#527F9A','#62947D','#8B75A2','#CA9C4C','#C36F62','#70A8AF','#A79B84','#C8CCCE']))
INK='#293942';BLUE='#356F98';RED='#BD5264'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'xtick.major.width':.5,'ytick.major.width':.5,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
W=7.4;N=len(s);matrix_height=max(3.2,N*.114);H=4.96+matrix_height+.49
fig=plt.figure(figsize=(W,H),facecolor='white')
def ax(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def txt(x,y,t,**kw):return fig.text(x/W,1-y/H,t,ha=kw.pop('ha','left'),va=kw.pop('va','top'),color=INK,**kw)
def letter(x,y,c):txt(x,y,c,fontsize=10.5,fontweight='bold')
def frame(a,bounds,xt,yt,fs=7):
    world.plot(ax=a,color='#EFF0EE',edgecolor='#C1C8CA',linewidth=.25,aspect='auto',zorder=1)
    a.set(xlim=bounds[:2],ylim=bounds[2:]);a.set_aspect('auto')
    a.set_xticks(xt);a.set_yticks(yt)
    a.set_xticklabels([f'{abs(v):g}°'+('W' if v<0 else 'E' if v>0 else '') for v in xt],fontsize=fs)
    a.set_yticklabels([f'{abs(v):g}°'+('S' if v<0 else 'N' if v>0 else '') for v in yt],fontsize=fs)
    a.tick_params(length=2,pad=1.6);a.grid(color='#D0D6D8',ls=(0,(2,2)),lw=.4,zorder=2)
    for sp in a.spines.values():sp.set(color='#7C878D',linewidth=.55)
PIES=[]
def pie(a,pid,loc=None,size=10,where='a'):
    row=s.loc[pid];true=(float(row.longitude),float(row.latitude));loc=true if loc is None else loc
    z=q[q.profile_id.eq(pid)].set_index('cause_group').occurrence_share_normalized_pct
    assert np.isclose(z.sum(),100,atol=.015),(pid,z.sum())
    if not np.allclose(loc,true):a.plot([true[0],loc[0]],[true[1],loc[1]],c='#6C7A80',lw=.45,zorder=5)
    da=DrawingArea(size,size,0,0);angle=90
    for c in CA:
        v=float(z.get(c,0));v=0 if pd.isna(v) else v;nxt=angle+3.6*v
        if v>0:da.add_artist(Wedge((size/2,size/2),size/2,angle,nxt,facecolor=COL[c],edgecolor='white',linewidth=.18))
        angle=nxt
    da.add_artist(Circle((size/2,size/2),size/2,facecolor='none',edgecolor=INK,linewidth=.5,linestyle='-' if row.paired else '--'))
    ab=AnnotationBbox(da,loc,frameon=False,pad=0,box_alignment=(.5,.5),zorder=8);a.add_artist(ab)
    PIES.append({'profile_id':pid,'panel':where,'true_lon_lat':true,'display_lon_lat':loc,'size_pt':size})
def label(a,name,xy,off=(3,4),ha='left',fs=6.6):
    return a.annotate(name,xy,xytext=off,textcoords='offset points',ha=ha,va='center',fontsize=fs,color=INK,zorder=10,path_effects=[pe.withStroke(linewidth=1.4,foreground='white')])
EXT={'b':(-12,45,34,74),'c':(-79,-32,-36,7),'d':(67,150,-15,57),'e':(-81.4,-71.5,40.5,46.5)}
def inside(ext):return s[s.longitude.between(ext[0],ext[1])&s.latitude.between(ext[2],ext[3])].index.tolist()
# a: each circle shows the complete occurrence composition, with no magnitude encoding.
a=ax(.36,.17,6.94,1.97);frame(a,(-180,180,-58,83),[-120,-60,0,60,120],[-30,0,30,60],7.5);letter(.12,.01,'a')
global_pos={
 'NORDIC_NOR':(5,70),'NORDIC_SWE':(15,64),'NORDIC_FIN':(28,73),'FIN_DISTRIBUTION':(39,69),
 'NORDIC_EST':(38,60),'NORDIC_LVA':(36,52),'NORDIC_LTU':(25,51),'NORDIC_DNK':(6,56),'GBR_SSEN2':(-5,60),
 'TUR_ULUDAG_2025':(24,35),'TUR_SEDAS_2025':(32,40),'TUR_BASKENT_2025':(43,45),'TUR_TOROSLAR_2025':(43,35),
 'DEU_BNETZA_LV_2024':(13,49),'CHE_ELCOM91_2024':(6,42),'ITA_TERNA_HV_ALL_TYPES':(15,36),'PRT_REN_2023_2024':(-9,39),
 'IND_TPDDL_2024_UNPLANNED_ROWS':(73,34),'IND_BRPL_UNPLANNED_2025_12':(74,21),
 'THA_EGAT_FORCED_2024':(104,19),'MYS_PENINSULAR_UNSCHEDULED_2020':(102,-7),'MYS_SARAWAK_CAUSES_2020':(117,-7),'MYS_SABAH_UNSCHEDULED_2020':(132,-5),
 'BTN_TRANSMISSION_2023':(89,31),'BGD_PGCB_2024_2025':(94,19),
 'SEN_SENELEC':(-18,19),'GIN_EDG2016':(-9,6)}
for pid,row in s.iterrows():
    size=6.5 if pid.startswith(('NORDIC','TUR_','DEU_','CHE_','ITA_','PRT_')) or pid in ['FIN_DISTRIBUTION','GBR_SSEN2'] else 10.2
    pie(a,pid,global_pos.get(pid),size,'a')
for k,ext in EXT.items():
    x0,x1,y0,y1=ext;a.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='#667E8A',lw=.6,zorder=3))
    label(a,k,(x1,y1),(2,4),fs=8.5).set_fontweight('bold')
world_labels={'NERC_NORTH_AMERICA':('N. America',(-1,10),'center'),'JAM_JPS':('Jamaica',(-6,-9),'right'),
 'BRA_AGGREGATE2623':('Brazil',(6,-4),'left'),'NZL_QUALIFIED21':('New Zealand',(-7,-1),'right'),
 'NORDIC_ISL':('ISL',(-6,0),'right'),'SEN_SENELEC':('Senegal',(-6,8),'right'),
 'GIN_EDG2016':('Guinea',(5,-8),'left'),'RWA_REG':('Rwanda',(-6,-6),'right'),'ETH_BEDELE7':('Ethiopia',(6,4),'left')}
for pid,(name,off,ha) in world_labels.items():
    if pid in s.index:label(a,name,global_pos.get(pid,(s.loc[pid,'longitude'],s.loc[pid,'latitude'])),off,ha,6.7)
# b-e: exact data bounds, equally tall and explicitly complete for contained profile points.
ys=2.42;hh=1.42;xs=[.36,2.35,4.00,6.27];widths=[1.65,1.28,1.88,1.03];axs={};represented={}
for k,x,ww,xt,yt in zip('bcde',xs,widths,[[0,30],[-60,-40],[80,120],[-80,-75]],[[40,60],[-20,0],[0,30],[42,45]]):
    axis=ax(x,ys,ww,hh);frame(axis,EXT[k],xt,yt,6.7);txt(x+.035,ys+.025,k,fontsize=10.5,fontweight='bold',bbox={'facecolor':'white','edgecolor':'none','pad':.4,'alpha':.9});axs[k]=axis;represented[k]=set()
b,c,d,e=[axs[x] for x in 'bcde']
# Country fills give leading occurrence class; all regional profile pies remain visible.
lead=q.loc[q.groupby('profile_id').occurrence_share_pct.idxmax()].set_index('profile_id').cause_group
eu['fill']=eu.profile_id.map(lead).map(COL);eu.plot(ax=b,color=eu['fill'],edgecolor='white',linewidth=.35,aspect='auto',zorder=3)
epos={'GBR_SSEN2':(-3,57),'NORDIC_NOR':(8,67),'NORDIC_SWE':(18,63),'NORDIC_FIN':(29,70),'FIN_DISTRIBUTION':(39,64),'NORDIC_DNK':(8,56),'NORDIC_LTU':(25,50),'NORDIC_LVA':(29,56),'NORDIC_EST':(38,55),'DEU_BNETZA_LV_2024':(12,50),'CHE_ELCOM91_2024':(5,44),'ITA_TERNA_HV_ALL_TYPES':(15,40),'PRT_REN_2023_2024':(-7,39),'TUR_ULUDAG_2025':(25,36),'TUR_SEDAS_2025':(30,42),'TUR_BASKENT_2025':(39,43),'TUR_TOROSLAR_2025':(39,36)}
for pid in inside(EXT['b']):
    pie(b,pid,epos.get(pid),8.8,'b');represented['b'].add(pid)
label(b,'UK',epos['GBR_SSEN2'],(0,-9),'center',6)
label(b,'FIN-D',epos['FIN_DISTRIBUTION'],(0,-9),'center',6)
br.plot(ax=c,color=br.leading_occurrence.map(COL),edgecolor='white',linewidth=.025,aspect='auto',zorder=3)
represented['c'].add('BRA_AGGREGATE2623')
for pid in inside(EXT['c']):
    if pid!='BRA_AGGREGATE2623':pie(c,pid,None,11,'c');represented['c'].add(pid)
apos={'CHN_NEA_LINES':(115,34),'IND_TPDDL_2024_UNPLANNED_ROWS':(74,44),'IND_BRPL_UNPLANNED_2025_12':(74,18),'BTN_TRANSMISSION_2023':(92,44),'BGD_PGCB_2024_2025':(91,18),'PHL_CAGAYAN_DE_ORO_2015_2022':(139,16),'JPN_TOHOKU_RESTORED_GROUPS_20261005':(143,45),'THA_EGAT_FORCED_2024':(108,13),'MYS_PENINSULAR_UNSCHEDULED_2020':(96,-6),'MYS_SARAWAK_CAUSES_2020':(113,-6),'MYS_SABAH_UNSCHEDULED_2020':(131,-6)}
for pid in inside(EXT['d']):
    loc=apos.get(pid,(s.loc[pid,'longitude'],s.loc[pid,'latitude']));pie(d,pid,loc,11.5,'d');represented['d'].add(pid)
    name={'CHN_NEA_LINES':'CHN','IND_TPDDL_2024_UNPLANNED_ROWS':'Delhi N/W','IND_BRPL_UNPLANNED_2025_12':'Delhi S/W','BGD_PGCB_2024_2025':'BGD','BTN_TRANSMISSION_2023':'BTN','PHL_CAGAYAN_DE_ORO_2015_2022':'Cagayan','MYS_PENINSULAR_UNSCHEDULED_2020':'Peninsula','MYS_SARAWAK_CAUSES_2020':'Sarawak','MYS_SABAH_UNSCHEDULED_2020':'Sabah'}.get(pid,s.loc[pid,'country_iso3'])
    label(d,name,loc,(0,-9),'center',6.0)
ny.plot(ax=e,color=ny.leading_occurrence.map(COL),edgecolor='white',linewidth=.4,aspect='auto',zorder=3);represented['e'].add('USA_NY_QUALIFIED')
for pid in inside(EXT['e']):
    if pid!='USA_NY_QUALIFIED':pie(e,pid,None,10,'e');represented['e'].add(pid)
for k,axis in axs.items():axis.set(xlim=EXT[k][:2],ylim=EXT[k][2:]);axis.set_aspect('auto')
legendnames=['Natural hazard / weather','Other environment','Equipment / electrical','Human / operation','Supply constraint','Planned','Other specified','Other / unresolved']
handles=[Line2D([],[],marker='s',ls='',ms=4.8,mec='none',mfc=COL[c]) for c in CA]
fig.legend(handles,legendnames,loc='upper center',bbox_to_anchor=(.53,1-4.11/H),ncol=4,fontsize=7.0,columnspacing=1.5,handletextpad=.35,handlelength=.7,borderaxespad=0,labelspacing=.45)
letter(.12,4.48,'f')

names={'BRA_AGGREGATE2623':'Brazil · 2,623 areas','BGD_PGCB_2024_2025':'Bangladesh · assets','CHN_NEA_LINES':'China · lines','FIN_DISTRIBUTION':'Finland · distribution','NORDIC_FIN':'Finland · transmission','TUR_BASKENT_2025':'Türkiye · Ankara–W. Black Sea','TUR_TOROSLAR_2025':'Türkiye · E. Mediterranean','TUR_ULUDAG_2025':'Türkiye · S. Marmara','TUR_SEDAS_2025':'Türkiye · Bolu–Sakarya','PHL_CAGAYAN_DE_ORO_2015_2022':'Philippines · Cagayan','BTN_TRANSMISSION_2023':'Bhutan · transmission','NERC_NORTH_AMERICA':'N. America · lines','ETH_BEDELE7':'Ethiopia · 7 feeders','USA_NY_QUALIFIED':'United States · New York'}
s['figure_name']=[names.get(pid,r.display_name) for pid,r in s.iterrows()]
# Leading-category switch (definition unchanged from the audited v4 revision).
ranking=[]
for pid,r in s.iterrows():
    gg=q[q.profile_id.eq(pid)].set_index('cause_group')
    pp=gg.occurrence_share_pct.fillna(0);pair=bool(r.paired);lead_occ=pp.idxmax()
    if pair:
        qq=gg.burden_share_pct.fillna(0);lead_loss=qq.idxmax()
        shift=float(.5*(100*pp/pp.sum()-100*qq/qq.sum()).abs().sum());changed=lead_occ!=lead_loss
        kind=('other_involved' if {lead_occ,lead_loss}&{'Other specified','Other / unresolved'} else 'known_to_known') if changed else 'unchanged'
    else:
        lead_loss=None;shift=np.nan;changed=False;kind='occurrence_only'
    ranking.append(dict(profile_id=pid,label=r.figure_name,paired=pair,leader_changed=changed,change_kind=kind,occurrence_leader=lead_occ,loss_leader=lead_loss,composition_shift_pp=shift))
rank=pd.DataFrame(ranking).set_index('profile_id')
assert len(rank)==40 and rank.paired.sum()==36 and rank.leader_changed.sum()==19
assert (rank.change_kind=='known_to_known').sum()==10 and (rank.change_kind=='other_involved').sum()==9
# Original v37 layout: alphabetical rows, full-width matrix, no extra columns.
order=s.sort_values('figure_name').index.tolist()
left=1.56;gap=.018;cw=(7.30-left-7*gap)/8;top=4.96
headers=['Natural hazard /\nweather','Other\nenvironment','Equipment /\nelectrical','Human /\noperation','Supply\nconstraint','Planned','Other\nspecified','Other /\nunresolved']
first=None
for j,category in enumerate(CA):
    aa=ax(left+j*(cw+gap),top,cw,matrix_height)
    if j==0:first=aa
    aa.set(xlim=(-12,112),ylim=(len(order)-.5,-.5));aa.set_xticks([0,50,100]);aa.tick_params(axis='x',labelsize=6.5,pad=2,length=2)
    aa.set_yticks(range(len(order)));aa.set_yticklabels([s.loc[z,'figure_name'] for z in order] if j==0 else [],fontsize=7.0);aa.tick_params(axis='y',length=0,pad=10 if j==0 else 4)
    for i,pid in enumerate(order):
        if i%2==0:aa.axhspan(i-.49,i+.49,color='#F2F4F3',lw=0)
        v=q[(q.profile_id.eq(pid))&q.cause_group.eq(category)]
        if v.empty or ('category_reported' in v and str(v.iloc[0].category_reported).lower()=='false'):
            aa.text(50,i,'–',color='#A5ABAE',ha='center',va='center',fontsize=8);continue
        r=v.iloc[0];x=r.occurrence_share_pct;y=r.burden_share_pct
        if pd.notna(y):aa.plot([x,y],[i,i],c='#929CA1',lw=.7)
        if pd.notna(x):aa.scatter(x,i,s=11,facecolor='white',edgecolor=BLUE,lw=.7,zorder=3)
        if pd.notna(y):aa.scatter(y,i,s=10.5,facecolor=RED,edgecolor='white',lw=.25,zorder=4)
    for v in [0,50,100]:aa.axvline(v,color='#D6DCDE',lw=.35,zorder=0)
    for spine in ['top','left','right']:aa.spines[spine].set_visible(False)
    aa.spines['bottom'].set_bounds(0,100)
    txt(left+j*(cw+gap)+cw/2,top-.29,headers[j],ha='center',fontsize=6.8,linespacing=1.0)
# Small switch markers sit between each label and the first panel.
from matplotlib.transforms import blended_transform_factory
bt=blended_transform_factory(first.transAxes,first.transData)
mx=-(5.2/72)/cw
for i,pid in enumerate(order):
    r=rank.loc[pid]
    if r.leader_changed:
        first.plot([mx],[i],transform=bt,marker='D',ms=3.0,mfc=INK if r.change_kind=='known_to_known' else 'white',mec=INK,mew=.6,ls='',clip_on=False,zorder=6)
fig.legend([Line2D([],[],marker='o',ls='',mfc='white',mec=BLUE,ms=4),Line2D([],[],marker='o',ls='',mfc=RED,mec='white',ms=4),
            Line2D([],[],marker='D',ls='',mfc=INK,mec=INK,mew=.6,ms=3.2),Line2D([],[],marker='D',ls='',mfc='white',mec=INK,mew=.6,ms=3.2)],
           ['Occurrence','Total duration / energy loss','Switch between identified causes','Switch involving other / unresolved'],
           ncol=4,loc='upper center',bbox_to_anchor=(.53,1-4.45/H),fontsize=7.2,handletextpad=.35,columnspacing=1.3,borderaxespad=0)
txt(4.42,top+matrix_height+.25,'Share within profile (%)',ha='center',fontsize=8)
OUTDIR=P/'minimal_v5';OUTDIR.mkdir(exist_ok=True);F=OUTDIR;Q=OUTDIR
rank.assign(display_order=[order.index(p)+1 for p in rank.index]).to_csv(OUTDIR/'fig2_minimal_profile_order.csv')

for _ax in fig.axes:_ax.set_xlabel('');_ax.set_ylabel('')  # newer geopandas adds CRS axis labels
fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds={}
for k,axis in axs.items():
    actual=[*axis.get_xlim(),*axis.get_ylim()];expected=list(EXT[k]);missing=set(inside(EXT[k]))-represented[k]
    assert np.allclose(actual,expected),(k,actual,expected)
    assert not missing,(k,missing)
    bounds[k]={'overview_rectangle':expected,'inset_extent':actual,'profiles_in_extent':inside(EXT[k]),'represented':sorted(represented[k]),'missing':sorted(missing)}
texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
coll=[]
for (t,bx),(u,by) in itertools.combinations(texts,2):
    ix=min(bx.x1,by.x1)-max(bx.x0,by.x0);iy=min(bx.y1,by.y1)-max(bx.y0,by.y0)
    if ix>.8 and iy>.8:coll.append({'a':t.get_text(),'b':u.get_text(),'overlap_px':[ix,iy]})
outside=[t.get_text() for t,b in texts if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5]
pie_overlaps=[]
for u,v in itertools.combinations(PIES,2):
    if u['panel']!=v['panel']:continue
    axis=a if u['panel']=='a' else axs[u['panel']]
    xy1=axis.transData.transform(u['display_lon_lat']);xy2=axis.transData.transform(v['display_lon_lat'])
    distance=np.linalg.norm(xy1-xy2)*72/fig.dpi
    if distance<(u['size_pt']+v['size_pt'])/2:pie_overlaps.append({'panel':u['panel'],'one':u['profile_id'],'two':v['profile_id'],'distance_pt':distance})
neighbor_intrusions=[]
neighbor_clearance=[]
for key,axis in axs.items():
    for text in axis.get_xticklabels()+axis.get_yticklabels():
        if not text.get_visible() or not text.get_text():continue
        box=text.get_window_extent(renderer)
        for other_key,other in axs.items():
            if other_key==key:continue
            framebox=other.get_window_extent(renderer)
            ix=min(box.x1,framebox.x1)-max(box.x0,framebox.x0);iy=min(box.y1,framebox.y1)-max(box.y0,framebox.y0)
            if ix>0 and iy>0:neighbor_intrusions.append({'panel':key,'tick':text.get_text(),'neighbor':other_key,'overlap_px':[ix,iy]})
        if key!='b' and text in axis.get_yticklabels():
            previous=axs[chr(ord(key)-1)].get_window_extent(renderer)
            neighbor_clearance.append({'panel':key,'tick':text.get_text(),'clearance_pt':float((box.x0-previous.x1)*72/fig.dpi)})
assert not neighbor_intrusions,neighbor_intrusions
json.dump({'tick_neighbor_intrusions':neighbor_intrusions,'tick_clearance_pt':neighbor_clearance,'insets':bounds,'pies':PIES,'pie_overlaps':pie_overlaps,'text_overlap_candidates':coll,'outside_canvas':outside,'size_inches':[W,H],'profiles':N},open(Q/'fig2_geometry_layout.json','w'),indent=2)
# Preserve the source figure geometry; trim only uniform blank page borders.
from PIL import Image,ImageChops
for ext in ['svg','pdf']:
    fig.savefig(F/('Fig2_minimal.'+ext),bbox_inches='tight',pad_inches=.025)
fig.savefig(F/'Fig2_minimal.png',dpi=600,bbox_inches='tight',pad_inches=.025)
preview=Image.open(F/'Fig2_minimal.png');preview.thumbnail((1600,2200));preview.save(Q/'fig2_minimal_preview.png')
source_files=[sp,qp,P.parent/'20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet',S/'brazil.geoparquet',S/'nordic.geoparquet',S/'ny.geoparquet']
meta={'figure':'Fig2','profile_count':40,'paired':36,'leader_changes':19,'known_to_known':10,'other_involved':9,'metric':'Total variation distance between normalized occurrence and time/energy cause compositions, in percentage points. It measures whole-composition change, not only leader reversal.','order':'Alphabetical (original v37 layout); 19 leader switches marked by diamonds','source_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in source_files},'outside_canvas':outside,'overlap_candidates':coll,'all_profile_ids_preserved':set(order)==set(s.index),'map_support_preserved':bounds,'word_files_modified':False}
(Q/'fig2_minimal_validation.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({'output':str(F/'Fig2_minimal.png'),'overlap_candidates':coll,'outside_canvas':outside,'counts':[40,36,19,10,9]},indent=2))

