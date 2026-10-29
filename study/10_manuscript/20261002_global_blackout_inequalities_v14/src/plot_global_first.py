"""Four new main composites: three global arguments and one regional explanation."""
from figure_style import *
from matplotlib.path import Path as MPath

W=gpd.read_parquet(R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
W=W[W.ISO3.ne('ATA')].copy()
QCOL=[BLUE,'#82A4BA','#C88D8D',RED]
CAUSES=['Weather','Other environment','Human / operation','Equipment / electrical','Supply constraint','Planned','Other / unresolved']
CC=['#3A73A6','#7EA99B','#B94F59','#CB9A48','#8C78A5','#CFB99C','#D4D8D9']

def world(a,d=None,key=None,cmap=None,norm=None):
    z=W.copy()
    if d is None:z.plot(ax=a,color='#eef0ee',edgecolor='#b6bdc0',lw=.3,aspect='auto')
    else:
        z=z.merge(d[['ISO3',key]],on='ISO3',how='left',validate='many_to_one')
        z.plot(column=key,ax=a,cmap=cmap,norm=norm,edgecolor='#b2b9bc',lw=.27,aspect='auto',missing_kwds={'color':'#e6e8e6'})
    a.set(xlim=(-180,180),ylim=(-58,82));a.set_aspect('auto')
    geographic_frame(a,4326,longitudes=[-120,-60,0,60,120],latitudes=[-30,0,30,60],fontsize=9.5)

def firms():
    d=dat('enterprise_latest_profiles');f=page(161)
    cm=LinearSegmentedColormap.from_list('firm_pct',['#f3f6f3','#c4d9d0','#76b4ad','#397e92','#1b475f'])
    norm=BoundaryNorm([0,10,25,50,75,90,100],cm.N)
    a=ax(f,13,11,121,47);world(a,d,'prevalence_pct',cm,norm)
    title(f,a,'a','Firms experiencing interruptions · 159 economies',offset=9)
    c=ax(f,13,87,121,47);world(c,d,'generator_pct',cm,norm)
    title(f,c,'c','Firms owning or sharing a generator',offset=9)
    for y,label in [(66,'Firms experiencing interruptions (%)'),(142,'Firms owning or sharing a generator (%)')]:
        cb(f,cm,norm,[31,y,90,2.3],[0,25,50,75,100],['0','25','50','75','100'],label)
    b=ax(f,160,11,43,47);z=d.dropna(subset=['gdp']).copy()
    b.scatter(z.gdp/1000,z.monthly_frequency,s=15,color=BLUE,alpha=.6,edgecolor='white',lw=.35)
    z['bin']=pd.qcut(z.gdp,6,labels=False)
    med=z.groupby('bin')[['gdp','monthly_frequency']].median()
    b.plot(med.gdp/1000,med.monthly_frequency,color=INK,lw=1.5,marker='o',ms=3)
    b.set(xscale='log',yscale='symlog',ylim=(-.03,120),xlim=(.9,150));b.set_yscale('symlog',linthresh=.15,linscale=.65)
    logticks(b,'x',[1,10,100]);logticks(b,'y',[0,.1,1,10,100]);b.set_xlabel('GDP/person\n(thousand PPP $)');b.set_ylabel('Outages per typical month')
    clean(b);title(f,b,'b','Economic gradient',offset=13)
    e=ax(f,160,87,43,47)
    e.scatter(d.prevalence_pct,d.generator_pct,s=17,color=TEAL,alpha=.62,edgecolor='white',lw=.3)
    e.set(xlim=(-3,103),ylim=(-3,103));e.set_xticks([0,50,100]);e.set_yticks([0,50,100])
    e.set_xlabel('Firms with\ninterruptions (%)');e.set_ylabel('Firms with a generator (%)')
    clean(e);title(f,e,'d','Backup and exposure',offset=13)
    save(f,'Fig02_global_firm_experience')

def cause_atlas():
    d=dat('global_cause_atlas');pr=dat('global_cause_atlas_profiles').set_index('profile_id');f=page(218)
    a=ax(f,14,10,187,66);world(a)
    locs={'NORDIC_DNK':(10,56),'NORDIC_EST':(26,59),'NORDIC_FIN':(27,66),'NORDIC_ISL':(-19,65),
          'NORDIC_LVA':(24,57),'NORDIC_LTU':(24,54.5),'NORDIC_NOR':(8,63),'NORDIC_SWE':(16,62),
          'FIN_DISTRIBUTION':(31,63),'USA_NY_QUALIFIED':(-75,43),'JAM_JPS':(-77,18),'SEN_SENELEC':(-14,14),
          'RWA_REG':(30,-2),'NZL_QUALIFIED21':(173,-41),'BRA_AGGREGATE689':(-49,-18)}
    def dot(aa,pid,x,y,lab=True):
        row=pr.loc[pid];col=CC[CAUSES.index(row.leading_burden)]
        aa.scatter(x,y,s=70,facecolor=col,edgecolor=INK,lw=.7,zorder=4)
        if lab:
            offset=(-6,-7) if row.country_iso3=='NZL' else (4,-7)
            aa.annotate(row.country_iso3,(x,y),xytext=offset,textcoords='offset points',fontsize=9.5,
                        ha='right' if row.country_iso3=='NZL' else 'left',color=INK,zorder=5)
    for pid,(x,y) in locs.items():
        if not pid.startswith('NORDIC') and pid!='FIN_DISTRIBUTION':dot(a,pid,x,y)
    a.add_patch(Rectangle((-28,50),64,24,fill=False,edgecolor='#6e7e86',lw=.8,zorder=5))
    # Dedicated inset avoids nine overlapping European markers on the world map.
    inset=ax(f,126,14,73,32)
    W.plot(ax=inset,color='#f1f3f1',edgecolor='#bdc4c7',lw=.35,aspect='auto')
    inset.set(xlim=(-29,40),ylim=(51,72));inset.set_aspect('auto');inset.set_xticks([]);inset.set_yticks([])
    for spine in inset.spines.values():spine.set_color('#77838a');spine.set_linewidth(.65)
    offsets={'NORDIC_DNK':(-7,-12),'NORDIC_EST':(6,1),'NORDIC_FIN':(-18,6),'NORDIC_ISL':(5,1),'NORDIC_LVA':(5,-1),'NORDIC_LTU':(5,-4),'NORDIC_NOR':(-19,3),'NORDIC_SWE':(-14,-8),'FIN_DISTRIBUTION':(4,0)}
    for pid,(dx,dy) in offsets.items():
        x,y=locs[pid];dot(inset,pid,x,y,False)
        txt='FIN-D' if pid=='FIN_DISTRIBUTION' else pr.loc[pid].country_iso3
        inset.annotate(txt,(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=9.0,ha='left',va='center')
    title(f,a,'a','Dominant burden class · observed country and regional profiles',offset=9)
    # Native percentages are not rescaled to force rounded values to 100.
    order=['SEN_SENELEC','RWA_REG','JAM_JPS','BRA_AGGREGATE689','USA_NY_QUALIFIED','NZL_QUALIFIED21','FIN_DISTRIBUTION',
        'NORDIC_DNK','NORDIC_EST','NORDIC_FIN','NORDIC_ISL','NORDIC_LVA','NORDIC_LTU','NORDIC_NOR','NORDIC_SWE']
    labels=['Senegal · network','Rwanda · reported network','Jamaica · distribution','Brazil · sampled areas','New York · sampled regions','New Zealand · sampled regions',
        'Finland · distribution','Denmark · transmission','Estonia · transmission','Finland · transmission',
        'Iceland · transmission','Latvia · transmission','Lithuania · transmission','Norway · transmission','Sweden · transmission']
    b=ax(f,61,95,104,96)
    for i,pid in enumerate(order):
        z=d[d.profile_id.eq(pid)].set_index('cause_group').reindex(CAUSES)
        for dy,key in [(-.19,'occurrence_share_pct'),(.19,'burden_share_pct')]:
            left=0
            for cause,col in zip(CAUSES,CC):
                v=z.loc[cause,key]
                if pd.notna(v):b.barh(i+dy,v,left=left,height=.33,color=col,edgecolor='white',lw=.15);left+=v
    b.set(xlim=(0,102),ylim=(14.65,-.65));b.set_xticks([0,25,50,75,100]);b.set_yticks(range(15),labels,fontsize=9.2)
    b.tick_params(axis='y',length=0);b.set_xlabel('Share within the same source and period (%)');clean(b)
    title(f,b,'b','Upper: occurrence   Lower: burden',offset=55)
    # Period column stays outside the plotted bars, aligned with their source rows.
    for i,pid in enumerate(order):b.text(1.035,i,pr.loc[pid,'period'].replace('FY',''),transform=b.get_yaxis_transform(),va='center',fontsize=9.0)
    text(f,169,91,'Observation period',fontsize=9.7,va='bottom')
    handles=[Patch(facecolor=x) for x in CC]
    legend(f,handles,[x.replace(' / ',' /\n') if x in ['Equipment / electrical','Other / unresolved'] else x for x in CAUSES],109,209,4,9.4)
    save(f,'Fig03_global_cause_atlas')

def regional_explanations():
    f=page(158)
    a=ax(f,18,12,74,48);q=dat('china_absolute_class_income').query("hazard_context=='All days'")
    for cat,col in [('Planned',BLUE),('Unplanned',RED)]:
        z=q[q.category.eq(cat)].sort_values('quartile');a.plot(range(4),z.frequency_per10000_fullwindow,color=col,lw=1.8,marker='o',ms=4,label=cat)
    a.set_xticks(range(4),['Q1','Q2','Q3','Q4']);a.set_ylim(0,6);a.set_ylabel('Notices per 10,000 residents');a.set_xlabel('County income quartile');clean(a)
    a.legend(loc='center left',bbox_to_anchor=(.02,.53),fontsize=9.8);title(f,a,'a','China: opposing occurrence patterns',offset=11)
    b=ax(f,128,12,74,48);m=dat('zero_inclusive_models');m=m[(m.region=='Brazil')&(m.specification=='Within operator + density')&(m.term=='log_density')]
    for j,cat in enumerate(['Ordinary unplanned','Planned']):
        for dy,out,col in [(-.18,'frequency',BLUE),(.18,'duration_component',TEAL)]:
            z=m[(m.category==cat)&(m.outcome==out)].iloc[0]
            b.errorbar(z.contrast_pct,j+dy,xerr=[[z.contrast_pct-z.contrast_lo_pct],[z.contrast_hi_pct-z.contrast_pct]],fmt='o',color=col,ms=4.5,capsize=2,lw=1.3)
    b.set(xlim=(-58,18),ylim=(1.6,-.6));b.set_xticks([-40,-20,0]);b.set_yticks([0,1],['Unplanned','Planned']);b.axvline(0,color=GREY,ls=':',lw=.8)
    b.set_xlabel('Density-IQR response (%)');clean(b);title(f,b,'b','Brazil: supply geography',offset=33)
    legend(f,[Line2D([],[],marker='o',color=BLUE,lw=0),Line2D([],[],marker='o',color=TEAL,lw=0)],['Frequency','Duration'],165,77,2,10)
    c=ax(f,35,91,62,47);dc=csv('duration_composition_summary').set_index('region');regions=['China','United States','Europe','Brazil','New Zealand']
    for j,region in enumerate(regions):
        z=dc.loc[region];neg=pos=0
        for k,col in [('within_pct_of_high',BLUE),('composition_pct_of_high',GOLD)]:
            v=z[k];c.barh(j,v,left=pos if v>=0 else neg,color=col,height=.6)
            if v>=0:pos+=v
            else:neg+=v
        c.scatter(z.gap_pct_of_high,j,s=21,marker='D',color=INK,edgecolor='white',lw=.5,zorder=4)
    c.set(xlim=(-40,82),ylim=(4.6,-.6));c.set_yticks(range(5),['China','United States','Europe','Brazil','New Zealand'],fontsize=9.5);c.set_xticks([-40,0,40,80])
    c.axvline(0,color=GREY,ls=':',lw=.8);c.set_xlabel('Low–high duration gap\n(% of high-income duration)');clean(c)
    title(f,c,'c','Composition can conceal duration gaps',offset=28)
    legend(f,[Patch(color=BLUE),Patch(color=GOLD),Line2D([],[],color=INK,marker='D',lw=0,ms=4)],['Within class','Composition','Net'],60,77,3,9.1)
    e=ax(f,144,91,58,47);r=dat('cause_persistence_unit_ratios');s=dat('cause_persistence_summary')
    order=[('New Zealand','Weather'),('New York','Weather'),('New Zealand','Lightning'),('New York','Lightning')]
    for j,(country,cause) in enumerate(order):
        vals=r[(r.source==country)&(r.cause==cause)].sort_values('system');col=TEAL if country=='New Zealand' else RED
        e.scatter(vals.ratio,j+np.linspace(-.19,.19,len(vals)),s=13,color=col,alpha=.5,lw=0)
        z=s[(s.source==country)&(s.cause==cause)].iloc[0]
        e.errorbar(z.geometric_ratio,j,xerr=[[z.geometric_ratio-z.ci_low],[z.ci_high-z.geometric_ratio]],fmt='D',mfc='white',mec=INK,color=INK,ms=4.6,lw=1.1,capsize=2,zorder=4)
    e.set(xscale='log',xlim=(.12,95),ylim=(3.65,-.65));logticks(e,'x',[.1,1,10,100]);e.set_xticks([.1,1,10,100],['0.1','1','10','100']);e.set_xlim(.1,100)
    e.set_yticks(range(4),['NZ · weather','NY · weather','NZ · lightning','NY · lightning'],fontsize=9.4);e.axvline(1,color=GREY,ls=':',lw=.8)
    e.set_xlabel('Cause / human-error\nmean duration');clean(e);title(f,e,'d','Cause-specific persistence',offset=34)
    save(f,'Fig05_regional_interpretation')

def global_burden():
    f=page(186);d=dat('global_benchmark_context')
    cm=LinearSegmentedColormap.from_list('burden',['#faf4e8','#edcc95','#d7926d','#b44e5d','#702745']);norm=LogNorm(.1,1000)
    a=ax(f,14,10,187,69);world(a,d,'burden',cm,norm);title(f,a,'a','Annual interruption burden · 152 economy benchmarks',offset=9)
    cb(f,cm,norm,[47,87,124,2.3],[.1,1,10,100,1000],['0.1','1','10','100','1,000'],'Hours without supply per customer-year',extend='both')
    b=ax(f,20,116,76,51);q=csv('Fig07_quartile_metrics')
    for metric,col,lab in [('saifi',BLUE,'Frequency'),('caidi_h',TEAL,'Duration'),('saidi_h',RED,'Annual burden')]:
        z=q[q.metric==metric].set_index('quartile').reindex(['Q1','Q2','Q3','Q4'])
        b.plot(range(4),z.relative_to_q4,color=col,lw=1.8,marker='o',ms=4,label=lab)
        b.fill_between(range(4),z.relative_ci_low,z.relative_ci_high,color=col,alpha=.12,lw=0)
    b.set(yscale='log',ylim=(.68,150),xlim=(-.15,3.15));b.axhline(1,color=GREY,ls=':',lw=.7);b.set_xticks(range(4),['Q1','Q2','Q3','Q4']);logticks(b,'y',[1,10,100])
    b.set_ylabel('Relative to highest-GDP quartile');b.set_xlabel('GDP/person quartile · lower → higher');clean(b);title(f,b,'b','Recurrence carries the global burden gap',offset=12)
    legend(f,[Line2D([],[],color=c,lw=1.8) for c in [BLUE,TEAL,RED]],['Frequency','Duration','Burden'],58,106,3,9.7)
    c=ax(f,150,116,51,51);m=dat('global_paired_income_models').query("specification=='Continent adjusted'")
    for j,source in enumerate(['Business-city benchmark','Conditional enterprise survey']):
        c.axhspan(j-.38,j+.38,color='#f2f4f3',zorder=0)
        for dy,out,col in [(-.16,'Frequency',BLUE),(.16,'Duration',TEAL)]:
            z=m[(m.source==source)&(m.outcome==out)].iloc[0]
            c.errorbar(z.response_pct,j+dy,xerr=[[z.response_pct-z.response_low],[z.response_high-z.response_pct]],fmt='o',color=col,ms=5,lw=1.2,capsize=2)
    c.axvline(0,color=GREY,ls=':',lw=.8);c.set(xlim=(-76,18),ylim=(1.5,-.5));c.set_xticks([-60,-30,0]);c.set_yticks([0,1],['City/utility\n(n = 152)','Enterprise\n(n = 140)'],fontsize=9.7)
    c.set_xlabel('Response per GDP doubling (%)');clean(c);title(f,c,'c','Independent economic contrasts',offset=35)
    save(f,'Fig06_global_burden_and_inequality')

def continents():
    f=page(114);d=dat('global_continent_decomposition');order=['All economies','Africa','Asia','Europe','North America']
    a=ax(f,45,13,85,77)
    for j,region in enumerate(order):
        for dy,metric,col in [(-.23,'frequency',BLUE),(0,'duration',TEAL),(.23,'burden',RED)]:
            z=d[(d.region==region)&(d.metric==metric)].iloc[0]
            a.errorbar(z.ratio,j+dy,xerr=[[z.ratio-z.ci_low],[z.ci_high-z.ratio]],fmt='o',color=col,ms=4,capsize=2,lw=1.3)
    a.axvline(1,color=GREY,ls=':',lw=.8);a.set(xscale='log',xlim=(.3,320),ylim=(4.6,-.6));logticks(a,'x',[1,10,100]);a.set_yticks(range(5),['Global · 152','Africa · 30','Asia · 41','Europe · 41','N. America · 20']);a.set_xlabel('Lowest / highest GDP quartile within each group');clean(a)
    title(f,a,'a','Global inequality within geographical strata',offset=40)
    for j,region in enumerate(order):
        z=d[(d.region==region)&(d.metric=='frequency')].iloc[0]
        text(f,145,24+j*14,f'{100*z.frequency_log_share:.1f}%',fontsize=12,color=BLUE,fontweight='bold')
    text(f,141,11,'Frequency share\nof log burden gap',fontsize=10.2,va='top')
    legend(f,[Line2D([],[],color=c,marker='o',ms=4) for c in [BLUE,TEAL,RED]],['Frequency','Duration','Annual burden'],111,108,3,10)
    save(f,'SuppFig01_geographic_decomposition')

def fine_causes():
    d=dat('global_cause_native_crosswalk');f=page(111)
    a=ax(f,40,13,61,77);b=ax(f,144,13,58,77)
    labels={'DNK':'Denmark','EST':'Estonia','FIN':'Finland','ISL':'Iceland','LVA':'Latvia','LTU':'Lithuania','NOR':'Norway','SWE':'Sweden'}
    z=d[d.profile_id.str.startswith('NORDIC') & d.cause_native.eq('Lightning')].sort_values('country_iso3')
    for j,(_,r) in enumerate(z.iterrows()):
        a.plot([r.occurrence_share_pct,r.burden_share_pct],[j,j],color='#b7bfc2',lw=2,zorder=1)
        a.scatter(r.occurrence_share_pct,j,color=BLUE,s=26,zorder=2)
        a.scatter(r.burden_share_pct,j,color=RED,s=26,zorder=2)
    a.set(ylim=(7.6,-.6),xlim=(-1,42));a.set_xticks([0,20,40]);a.set_yticks(range(8),[labels[x] for x in z.country_iso3]);a.set_xlabel('Lightning share (%)');clean(a)
    title(f,a,'a','Nordic and Baltic transmission',offset=34)
    zz=d[d.profile_id.eq('FIN_DISTRIBUTION') & d.cause_group.eq('Weather')].sort_values('occurrence_share_pct',ascending=False)
    for j,(_,r) in enumerate(zz.iterrows()):
        b.plot([r.occurrence_share_pct,r.burden_share_pct],[j,j],color='#b7bfc2',lw=2)
        b.scatter(r.occurrence_share_pct,j,color=BLUE,s=30,zorder=3);b.scatter(r.burden_share_pct,j,color=RED,s=30,zorder=3)
    b.set(ylim=(len(zz)-.4,-.6),xlim=(-1,50));b.set_xticks([0,25,50]);b.set_yticks(range(len(zz)),zz.cause_native.str.replace(' / ',' /\n',regex=False),fontsize=9.5);b.set_xlabel('Cause share (%)');clean(b)
    title(f,b,'b','Finnish distribution',offset=26)
    legend(f,[Line2D([],[],color=c,marker='o',lw=0) for c in [BLUE,RED]],['Occurrence','Burden'],110,106,2,10)
    save(f,'SuppFig02_hazard_specific_burden')

if __name__=='__main__':
    import sys
    tasks={'firms':firms,'causes':cause_atlas,'regions':regional_explanations,'burden':global_burden,'continents':continents,'fine_causes':fine_causes}
    for k in (sys.argv[1:] or tasks):tasks[k]()
