"""Household evidence, rendered with the established manuscript figure system."""
from figure_style import *

W=gpd.read_parquet(R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet').to_crs(4326)
W=W[W.ISO3.ne('ATA')]
C=pd.read_csv(P/'source_data/afrobarometer_country_contrasts.csv')
A=C[C.admitted].copy()
SHORT={'Cabo Verde':'Cabo Verde','Cote d\'Ivoire':'Côte d’Ivoire','Côte d’Ivoire':'Côte d’Ivoire',
       'Sao Tome and Principe':'São Tomé & Príncipe','São Tomé and Príncipe':'São Tomé & Príncipe','Swaziland':'Eswatini'}


def households():
    f=page(208)
    cm=LinearSegmentedColormap.from_list('poor_gap',[BLUE,'#faf9f5',RED])
    norm=TwoSlopeNorm(vmin=-40,vcenter=0,vmax=40)
    a=ax(f,12,10,81,82)
    z=W.merge(A[['iso3','conditional_gap']],left_on='ISO3',right_on='iso3',how='left',validate='many_to_one')
    z['gap_pp']=100*z.conditional_gap
    z.plot(ax=a,color='#e7e9e7',edgecolor='#b7bec0',lw=.35,aspect='auto')
    z[z.gap_pp.notna()].plot(column='gap_pp',ax=a,cmap=cm,norm=norm,edgecolor='white',lw=.38,aspect='auto')
    # Small islands are shown at their actual locations, not visually enlarged polygons.
    for iso,x,y in [('CPV',-23.7,15.1),('STP',6.6,.3),('SYC',55.5,-4.7)]:
        gap=float(A.loc[A.iso3==iso,'conditional_gap'].iloc[0])*100
        a.scatter(x,y,s=25,facecolor=cm(norm(gap)),edgecolor='#455057',lw=.6,zorder=4)
    a.set(xlim=(-28,60),ylim=(-37,39));a.set_aspect('auto')
    geographic_frame(a,4326,longitudes=[0,30,60],latitudes=[-30,0,30],fontsize=9.2)
    title(f,a,'a','Inequality after grid connection',offset=9)
    cb(f,cm,norm,[18,100,67,2.5],[-40,-20,0,20,40],['−40','−20','0','20','40'],
       'Unreliable supply gap (percentage points)')

    b=ax(f,137,10,66,82)
    order=A.sort_values('service_gap',ascending=False).reset_index(drop=True)
    for i,row in order.iterrows():
        conn=100*row.connection_component;rel=100*row.reliability_component
        b.barh(i,conn,height=.7,color=BLUE,alpha=.88)
        b.barh(i,rel,left=max(conn,0) if rel>=0 else 0,height=.7,color=RED,alpha=.86)
        b.plot(100*row.service_gap,i,marker='o',ms=2.5,color=INK,lw=0)
    b.set_yticks(range(len(order)),[SHORT.get(x,x) for x in order.country],fontsize=9)
    b.tick_params(axis='y',length=0);b.set(xlim=(-7,56),ylim=(len(order)-.45,-.65))
    b.set_xticks([0,20,40]);b.set_xlabel('Reliable-service gap (pp)',fontsize=9.8)
    b.axvline(0,color='#7f898d',lw=.55);clean(b)
    title(f,b,'b','Two components of service inequality',offset=37)
    legend(f,[Patch(facecolor=BLUE),Patch(facecolor=RED),Line2D([],[],marker='o',color=INK,lw=0,ms=3)],['Connection','Reliability','Total gap'],167,110,2,9.3)
    c=ax(f,15,126,73,60)
    countries=[('Cambodia','cambodia_income_quintiles.csv',BLUE),
               ('Ethiopia','ethiopia_complete_quintiles.csv',TEAL),
               ('Kenya','kenya_complete_quintiles.csv',GOLD)]
    for country,fn,col in countries:
        q=pd.read_csv(P/'source_data'/fn)
        c.fill_between(q.quintile,100*q.connection_lo,100*q.connection_hi,color=col,alpha=.10,lw=0)
        c.plot(q.quintile,100*q.connection,'o-',color=col,ms=4,lw=1.5,label=country)
    c.set(xlim=(.8,5.2),ylim=(0,100),ylabel='Households connected to grid (%)',xlabel='Living-standard quintile')
    c.set_xticks(range(1,6),['Q1','Q2','Q3','Q4','Q5']);c.set_yticks([0,25,50,75,100]);clean(c)
    c.legend(loc='upper left',fontsize=9.2,labelspacing=.32,handlelength=1.4,borderaxespad=.3)
    title(f,c,'c','Unequal access to the grid',offset=10)
    d=ax(f,132,126,71,60)
    for i,(country,_,_) in enumerate(countries):
        fn='cambodia_summary.json' if country=='Cambodia' else country.lower()+'_complete_summary.json'
        s=json.loads((P/'metadata'/fn).read_text())
        if i%2==0:d.axhspan(i-.43,i+.43,color='#f0f2f2',zorder=0)
        for j,(key,col) in enumerate([('frequency_week',BLUE),('duration_hours',TEAL),('hours_week',RED)]):
            z=s['contrasts'][key];y=i+(j-1)*.22
            d.plot([z['ratio_lo'],z['ratio_hi']],[y,y],color=col,lw=1.3)
            d.scatter(z['ratio'],y,s=27,color=col,edgecolor='white',lw=.5,zorder=3)
    d.axvline(1,color=GREY,lw=.8,ls='--');d.set(xscale='log',xlim=(.125,16),ylim=(2.6,-.6),xlabel='Lowest / highest quintile')
    d.set_yticks(range(3),[r[0] for r in countries]);d.tick_params(axis='y',length=0);logticks(d,'x',[.25,1,4,16]);clean(d)
    title(f,d,'d','Interruption after connection',offset=29)
    legend(f,[Line2D([],[],marker='o',color=co,lw=1.3,ms=4) for co in [BLUE,TEAL,RED]],
           ['Frequency','Duration','Hours'],155,203,3,9.0)
    save(f,'Fig04_household_service_inequality')


def supplementary():
    f=page(114)
    s=pd.read_csv(P/'source_data/afrobarometer_settlement_contrasts.csv').sort_values('total_gap',ascending=False)
    a=ax(f,40,13,60,72)
    for i,row in enumerate(s.itertuples()):
        a.plot([row.total_gap*100,row.standardized_gap*100],[i,i],color='#c3c9ca',lw=1.5)
        a.scatter(row.total_gap*100,i,s=28,facecolor='white',edgecolor=BLUE,lw=1.3,zorder=3)
        a.scatter(row.standardized_gap*100,i,s=26,color=RED,edgecolor='white',lw=.4,zorder=4)
    a.set_yticks(range(len(s)),[SHORT.get(x,x) for x in s.country],fontsize=9.6)
    a.set(ylim=(len(s)-.4,-.6),xlim=(-20,35));a.set_xticks([-20,0,20]);a.axvline(0,lw=.65,ls=':',color=GREY)
    a.set_xlabel('Unreliable supply gap (pp)');a.tick_params(axis='y',length=0);clean(a)
    title(f,a,'a','Settlement composition',offset=33)
    legend(f,[Line2D([],[],marker='o',mfc='white',mec=BLUE,lw=0),Line2D([],[],marker='o',color=RED,lw=0)],['Observed','Standardized'],72,106,2,9.4)
    b=ax(f,133,13,69,72)
    alt=pd.read_csv(P/'source_data/afrobarometer_four_item_contrasts.csv')
    both=A.merge(alt[alt.admitted],on='iso3',suffixes=('_five','_four'))
    b.plot([-15,45],[-15,45],lw=.8,ls=':',color=GREY)
    b.scatter(both.conditional_gap_five*100,both.conditional_gap_four*100,color=TEAL,edgecolor='white',s=36,lw=.5)
    b.set(xlim=(-10,40),ylim=(-10,40),xlabel='Five-item poverty gap (pp)',ylabel='Four-item poverty gap (pp)')
    b.set_xticks([0,20,40]);b.set_yticks([0,20,40]);clean(b)
    title(f,b,'b','Poverty construct',offset=13)
    save(f,'SuppFig08_household_interpretation')


def multicountry():
    f=page(200)
    cases=[('Cambodia','cambodia_income_quintiles.csv'),('Ethiopia','ethiopia_complete_quintiles.csv'),('Kenya','kenya_complete_quintiles.csv')]
    keys=['connection','frequency_week','duration_hours','hours_week']
    labels=['Connected (%)','Interruptions / week','Hours / interruption','Interrupted hours / week']
    for i,(country,fn) in enumerate(cases):
        q=pd.read_csv(P/'source_data'/fn)
        for j,(key,lab) in enumerate(zip(keys,labels)):
            a=ax(f,15+j*50,13+i*62,38,43)
            mult=100 if j==0 else 1
            col=[GREY,BLUE,TEAL,RED][j]
            a.fill_between(q.quintile,mult*q[key+'_lo'],mult*q[key+'_hi'],color=col,alpha=.15,lw=0)
            a.plot(q.quintile,mult*q[key],'o-',color=col,lw=1.3,ms=3.6)
            if country!='Cambodia' and j>0:
                other=pd.read_csv(P/'source_data'/f'{country.lower()}_component_zero_quintiles.csv')
                a.plot(other.quintile,other[key],ls='--',color=col,lw=1.1)
            a.set(xlim=(.7,5.3),ylim=(0,100 if j==0 else 1.07*max(q[key+'_hi'])),ylabel=lab)
            a.yaxis.labelpad=2
            a.set_xticks(range(1,6),['1','2','3','4','5']);a.tick_params(labelsize=9);clean(a)
            if i==2:a.set_xlabel('Quintile',fontsize=9.5)
            title(f,a,chr(97+i*4+j),country if j==0 else '',offset=11)
    legend(f,[Line2D([],[],color=INK,lw=1.5),Line2D([],[],color=INK,lw=1.2,ls='--')],
           ['Paired primary estimate','Blank component set to zero'],107,198,2,9.2)
    save(f,'SuppFig09_multicountry_households')


if __name__=='__main__':
    households()
    supplementary()
    multicountry()
