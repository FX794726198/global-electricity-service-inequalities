"""Cause comparisons, including the independent result that does not replicate."""
from figure_style import *

f=page(176)
cm=LinearSegmentedColormap.from_list('causefreq',['#edf3f1','#c2d5d8','#87b5c3','#467eac','#234b78'])
norm=BoundaryNorm([0,.05,.1,.25,.5,1,2.5],cm.N)
nz=dat('nz_explicit21_cause_context')
nzg=gpd.read_parquet(P/'data/nz_explicit21_map.parquet').to_crs(2193)
if 'frequency' in nzg: nzg=nzg.drop(columns='frequency')
nzg=nzg.merge(nz[nz.category.eq('Weather')][['system','frequency']],on='system',validate='one_to_one')
a=ax(f,17,12,55,68)
world=gpd.read_parquet(R/'10_manuscript/20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/world_geometry.geoparquet')
world[world.ISO3.eq('NZL')].to_crs(2193).plot(ax=a,color='#eceeed',edgecolor='#abb1b4',lw=.4)
nzg.plot(column='frequency',ax=a,cmap=cm,norm=norm,edgecolor='white',lw=.45)
extent(a,nzg.total_bounds,55/68,pad=1.13)
geographic_frame(a,2193,longitudes=[168,174,180],latitudes=[-45,-40,-35],fontsize=9.3)
title(f,a,'a','New Zealand · 21 networks',offset=11)

ny=dat('ny_regional_cause_context')
nyg=gpd.read_parquet(P/'data/ny_ng_electric_service_regions.parquet').to_crs(5070)
nyg=nyg.merge(ny[ny.category.eq('Recorded weather')][['region','frequency']],left_on='Region',right_on='region',validate='one_to_one')
b=ax(f,102,12,98,68)
us=gpd.read_parquet(P/'data/us_same49_map.parquet').to_crs(5070)
us[us.state.eq('NY')].plot(ax=b,color='#eceeed',edgecolor='#abb1b4',lw=.5)
nyg.plot(column='frequency',ax=b,cmap=cm,norm=norm,edgecolor='white',lw=.6)
extent(b,nyg.total_bounds,98/68,pad=1.13)
geographic_frame(b,5070,longitudes=[-79,-76,-73],latitudes=[42,43,44,45],fontsize=9.3)
title(f,b,'b','New York · 8 regions',offset=11)
cb(f,cm,norm,[33,92,141,2.2],[0,.1,.5,1,2.5],['0','0.1','0.5','1','2.5'],
   'Recorded-weather interruptions per customer-year',extend='max')

c=ax(f,51,114,48,42)
defs=[('ny_nz_narrow_human_error_contrasts','Weather minus operating-error frequency','NZ: weather',TEAL,'o','New Zealand narrow human error'),
      ('ny_regional_income_contrasts','Weather minus operating-error frequency','NY: weather',GOLD,'o','National Grid New York'),
      ('ny_common_named_cause_contrasts','Lightning minus human-error frequency','NZ: lightning',TEAL,'s','New Zealand'),
      ('ny_common_named_cause_contrasts','Lightning minus human-error frequency','NY: lightning',GOLD,'s','New York')]
source=[]
for j,(stem,contrast,label,col,mk,region) in enumerate(defs):
    z=dat(stem).query("specification=='Density adjusted'")
    z=z[z.contrast.eq(contrast)&z.region.eq(region)].iloc[0]
    c.errorbar(z.contrast_pct,j,xerr=[[z.contrast_pct-z.contrast_lo_pct],[z.contrast_hi_pct-z.contrast_pct]],
        fmt=mk,color=col,mfc='white' if j>=2 else col,mec=col,ms=4.5,capsize=2,lw=1.2)
    source.append(dict(label=label,**z.to_dict()))
c.axvline(0,color='#959da1',ls=':',lw=.9)
c.set(xlim=(-100,270),ylim=(3.6,-.6))
c.set_xticks([0,100,200]);c.set_yticks(range(4),['NZ: weather','NY: weather','NZ: lightning','NY: lightning'],fontsize=9.5)
c.set_xlabel('Cause–human-error frequency\nresponse contrast (%)',fontsize=9.7)
clean(c);title(f,c,'c','Cause-specific income contrasts',offset=39)
pd.DataFrame(source).to_csv(P/'source_data/Fig06_income_cause_comparison.csv',index=False)

d=ax(f,141,114,59,42)
rr=pd.read_csv(P/'data/ny_process_stage_region_ratios.csv')
su=pd.read_csv(P/'data/ny_process_stage_summary.csv').set_index('metric')
for j,(key,col) in enumerate([('scope',BLUE),('duration',RED)]):
    vals=rr[rr.metric.eq(key)].sort_values('region')
    jitter=np.linspace(-.18,.18,len(vals))
    d.scatter(vals.ratio,j+jitter,s=18,color=col,alpha=.55,edgecolors='none',zorder=2)
    z=su.loc[key]
    d.errorbar(z.geometric_ratio,j,xerr=[[z.geometric_ratio-z.ci_low],[z.ci_high-z.geometric_ratio]],
        fmt='D',color=INK,mfc='white',ms=4.7,lw=1.25,capsize=2.5,zorder=3)
d.axvline(1,color='#959da1',ls=':',lw=.9)
d.set(xscale='log',xlim=(.08,32),ylim=(1.6,-.6))
logticks(d,'x',[.1,1,10]);d.set_yticks([0,1],['Customer\nfootprint','Duration'],fontsize=9.5)
d.set_xlabel('Weather / operator error\nwithin the same New York region',fontsize=9.7)
clean(d);title(f,d,'d','Different disruption processes',offset=24)
su.to_csv(P/'source_data/Fig06_NY_process_stages.csv')
save(f,'Fig06_cause_specific_inequality_and_processes')
