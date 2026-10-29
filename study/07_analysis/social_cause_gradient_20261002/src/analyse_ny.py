"""Official census-poverty context for eight already qualified New York regions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import block_diag
from scipy.stats import t
P=Path(__file__).resolve().parents[1];D=P/'data';M=P/'metadata'
N=P.parent/'inequality_process_linkage_20261001/components/ny_regional_causes'
cols=['GEO_ID','B17001_E001','B17001_E002','B17001_M001','B17001_M002']
chunks=[]
for chunk in pd.read_csv(P/'raw/acsdt5y2022-b17001.dat',sep='|',dtype=str,usecols=cols,chunksize=50000):
    chunks.append(chunk[chunk.GEO_ID.str.startswith('1400000US36')])
a=pd.concat(chunks,ignore_index=True)
a['GEOID']=a.GEO_ID.str.removeprefix('1400000US')
for c in cols[1:]:a[c]=pd.to_numeric(a[c],errors='coerce').where(lambda x:x>=0)
assert not a.GEOID.duplicated().any()
a.to_parquet(D/'ny_tract_poverty_2018_2022.parquet',index=False)
cross=pd.read_parquet(N/'data/tract_region_crosswalk.parquet')
cross=cross[['GEOID','Region','fraction','admitted']].merge(a,on='GEOID',how='left',validate='one_to_one')
cross['valid_poverty']=cross.B17001_E001.gt(0)&cross.B17001_E002.ge(0)&cross.B17001_E002.le(cross.B17001_E001)
cross.to_parquet(D/'ny_poverty_crosswalk.parquet',index=False)
q=cross[cross.admitted&cross.valid_poverty].copy()
c=q.groupby('Region').agg(poverty_universe=('B17001_E001','sum'),below_poverty=('B17001_E002','sum'),
    tracts=('GEOID','size'),poverty_moe_rss=('B17001_M002',lambda s:np.sqrt(np.sum(s*s))))
c['poverty_share_pct']=100*c.below_poverty/c.poverty_universe
c['intersecting_tract_universe']=cross[cross.valid_poverty].groupby('Region').B17001_E001.sum()
c['strict_to_intersecting_universe']=c.poverty_universe/c.intersecting_tract_universe
context=pd.read_parquet(N/'data/regional_income_context.parquet').set_index('Region')
c=c.join(context[['area_km2','log_density','mean_household_income']]).reset_index().rename(columns={'Region':'region'})
assert len(c)==8 and c.poverty_share_pct.notna().all()
c['poverty_5pp']=c.poverty_share_pct/5
native=pd.read_parquet(N/'data/regional_category_year.parquet')
assert native.year.between(2021,2024).all()
assert native.groupby(['region','category']).year.nunique().eq(4).all()
den=pd.read_parquet(N/'data/regional_customer_denominators.parquet')
den=den[den.year.between(2021,2024)].groupby('region').customers_served.sum().rename('customer_years')
g=native.groupby(['region','category'])[['customer_interruptions','customer_hours','interruption_events']].sum().reset_index()
tot=g.groupby('region')[['customer_interruptions','customer_hours','interruption_events']].sum().reset_index();tot['category']='All causes'
g=pd.concat([g,tot],ignore_index=True).merge(den,on='region',validate='many_to_one')
g['frequency']=g.customer_interruptions/g.customer_years;g['burden_hours']=g.customer_hours/g.customer_years
g['conditional_duration_h']=g.customer_hours/g.customer_interruptions.where(g.customer_interruptions>0)
g=g.merge(c,on='region',validate='many_to_one')
c.to_csv(D/'ny_poverty_context.csv',index=False);g.to_csv(D/'ny_poverty_cause_stages.csv',index=False)
rows=[];audit=[]
for cause,z in g.groupby('category'):
    if z.frequency.sum()==0 or z.burden_hours.sum()==0:
        audit.append(dict(cause=cause,status='All zero; no finite gradient'));continue
    for spec in ['Unadjusted','Density adjusted']:
        z=z.reset_index(drop=True)
        X=pd.DataFrame({'const':np.ones(len(z)),'poverty_5pp':z.poverty_5pp})
        if spec=='Density adjusted':X['log_density']=z.log_density
        fits=[];scores=[];breads=[]
        for col in ['frequency','burden_hours']:
            fit=sm.GLM(z[col],X,family=sm.families.Poisson()).fit(maxiter=300,tol=1e-10)
            mu=np.asarray(fit.fittedvalues);fits.append(fit)
            scores.append(np.asarray(X)*(z[col].to_numpy()-mu)[:,None])
            breads.append(np.linalg.inv(np.asarray(X).T@(mu[:,None]*np.asarray(X))))
        n,k=X.shape;S=np.column_stack(scores);B=block_diag(*breads)
        cov=B@(S.T@S*n/(n-k))@B;params=np.r_[fits[0].params,fits[1].params]
        for metric,ws in [('Frequency',[(1,1)]),('Annual burden',[(k+1,1)]),
            ('Conditional duration',[(k+1,1),(1,-1)]),('Frequency versus duration',[(1,2),(k+1,-1)])]:
            L=np.zeros(2*k)
            for i,w in ws:L[i]=w
            b=float(L@params)
            # Sum squared projected influences directly: mathematically identical
            # to L cov L, without subtractive cancellation in sparse causes.
            projected=S@(B@L);v=float(projected@projected*n/(n-k))
            se=np.sqrt(v);crit=t.ppf(.975,n-k)
            rows.append(dict(category=cause,specification=spec,metric=metric,regions=n,
                poverty_contrast_pp=5,contrast_pct=100*np.expm1(b),lower_pct=100*np.expm1(b-crit*se),
                upper_pct=100*np.expm1(b+crit*se),beta=b,se=se,df=n-k,
                p_value=2*t.sf(abs(b/se),n-k) if se>0 else np.nan,
                positive_regions=int(z.frequency.gt(0).sum()),
                interpretation='Sparse positive support' if z.frequency.gt(0).sum()<4 else 'Exploratory regional association'))
        audit.append(dict(cause=cause,specification=spec,status='Estimated',converged=all(f.converged for f in fits)))
pd.DataFrame(rows).to_csv(D/'ny_poverty_stage_gradients.csv',index=False)
info=dict(regions=8,tracts=len(q),strict_tract_threshold=.9,poverty_period='2018–2022',outage_period='2021–2024',
    poverty_share_range=c.poverty_share_pct.agg(['min','max']).to_dict(),
    admitted_tracts_missing_poverty=int((cross.admitted&~cross.valid_poverty).sum()),models=audit,
    independent_observations='8 geographic regions, not 32 region-years',
    uncertainty='Joint HC1 covariance; t on n minus parameter count; conditional on ACS published estimates')
(M/'ny_poverty_audit.json').write_text(json.dumps(info,indent=2))
print(c[['region','tracts','poverty_share_pct','strict_to_intersecting_universe']].round(3).to_string(index=False))
print(pd.DataFrame(rows).query("specification=='Density adjusted' & metric!='Frequency versus duration'")[
    ['category','metric','contrast_pct','lower_pct','upper_pct']].round(2).to_string(index=False))
