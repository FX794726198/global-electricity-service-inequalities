"""Direct contrasts of process-specific slopes, with paired covariance."""
from analyse_zero_inclusive import *
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

out=[]
def contrasts(z,X,ys,definitions,meta,delta,groups=None):
    x=X.to_numpy(float);n,p=x.shape;k=len(ys);B=np.zeros((k*p,k*p));S=[];coef=[];ranks=[]
    for j,y in enumerate(ys.values()):
        fit=sm.GLM(y,x,family=sm.families.Poisson()).fit(maxiter=200,tol=1e-10)
        assert fit.converged
        mu=np.asarray(fit.fittedvalues);bi=np.linalg.pinv(x.T@(mu[:,None]*x));B[j*p:(j+1)*p,j*p:(j+1)*p]=bi
        score=x*(np.asarray(y)-mu)[:,None]
        if groups is None:
            # HC3-style leverage correction, equation-specific (small NZ frame).
            h=mu*np.einsum('ij,jk,ik->i',x,bi,x);score=score/np.maximum(1-h,.001)[:,None]
        S.append(score);coef.extend(fit.params)
    S=np.hstack(S)
    if groups is None:V=S.T@S;df=n-p
    else:
        joint=pd.Series(list(zip(groups[0],groups[1]))).astype(str)
        V=meat(S,groups[0],p)+meat(S,groups[1],p)-meat(S,joint,p);df=min(pd.Series(groups[0]).nunique(),pd.Series(groups[1]).nunique())-1
    cov=B@V@B;coef=np.array(coef);crit=t.ppf(.975,df)
    for term,ds in definitions.items():
        j=X.columns.get_loc(term)
        for label,weights in ds.items():
            L=np.zeros(k*p)
            for outcome,w in weights.items():L[list(ys).index(outcome)*p+j]=w
            b=L@coef;v=L@cov@L;se=np.sqrt(v) if v>=0 else np.nan;lo=b-crit*se;hi=b+crit*se;dd=delta[term]
            out.append(dict(**meta,term=term,contrast=label,n=n,df=df,beta=b,lo=lo,hi=hi,p=2*t.sf(abs(b/se),df),
                contrast_pct=100*np.expm1(b*dd),contrast_lo_pct=100*np.expm1(lo*dd),contrast_hi_pct=100*np.expm1(hi*dd),covariance='joint two-way cluster' if groups else 'joint HC3-style sandwich'))

def china():
    a=pd.read_parquet(D/'china_common_counties.parquet');delta=a.log_income.quantile(.75)-a.log_income.quantile(.25)
    # Rate models weight counties equally. The offset/count PPML analysis has a
    # population-weighted score; keep the difference explicit in output.
    ys={'Fp':a.planned_count/a['pop'],'Hp':a.planned_hours/a['pop'],'Fu':a.unplanned_count/a['pop'],'Hu':a.unplanned_hours/a['pop'],
        'Fa':a.all_notices_count/a['pop'],'Ha':a.all_notices_hours/a['pop']}
    defs={'Planned minus unplanned frequency':{'Fp':1,'Fu':-1},'Planned minus unplanned duration':{'Hp':1,'Fp':-1,'Hu':-1,'Fu':1},
        'Unplanned frequency minus duration':{'Fu':2,'Hu':-1},'Planned frequency':{'Fp':1},'Unplanned frequency':{'Fu':1},
        'Planned duration':{'Hp':1,'Fp':-1},'Unplanned duration':{'Hu':1,'Fu':-1},'All frequency':{'Fa':1},'All duration':{'Ha':1,'Fa':-1}}
    for spec in ['County-equal rates','County-equal rates + population and hazard']:
        X=pd.DataFrame({'const':1.,'log_income':a.log_income},index=a.index)
        if '+' in spec:X['log_population']=a.log_population;X['hazard_fraction']=a.hazard_fraction
        contrasts(a,X,ys,{'log_income':defs},dict(region='China',specification=spec),{'log_income':delta})

def brazil():
    a=pd.read_parquet(D/'brazil_2021_joint_context.parquet');m=a.drop_duplicates('municipality_code');di=m.log_income.quantile(.75)-m.log_income.quantile(.25);dd=a.log_density.quantile(.75)-a.log_density.quantile(.25)
    ys={'Fp':a.FECIP,'Hp':a.DECIP,'Fu':a.FECIND,'Hu':a.DECIND}
    defs={'Planned minus ordinary-unplanned frequency':{'Fp':1,'Fu':-1},'Planned minus ordinary-unplanned duration':{'Hp':1,'Fp':-1,'Hu':-1,'Fu':1}}
    for spec in ['Economic context','Within operator + density']:
        X=pd.DataFrame({'const':1.,'log_income':a.log_income},index=a.index);ds={'log_income':defs};delta={'log_income':di}
        if spec.startswith('Within'):
            X['log_density']=a.log_density;X=pd.concat([X,pd.get_dummies(a.NumCNPJ.astype(str),drop_first=True,dtype=float)],axis=1);ds['log_density']=defs;delta['log_density']=dd
        contrasts(a,X,ys,ds,dict(region='Brazil',specification=spec),delta,groups=[a.municipality_code,a.NumCNPJ])

if __name__=='__main__':
    china();brazil()
    save(pd.DataFrame(out),'direct_process_slope_contrasts')
    M.joinpath('process_contrast_manifest.json').write_text(json.dumps(receipts,indent=2))
    print(pd.DataFrame(out)[['region','specification','term','contrast','n','contrast_pct','contrast_lo_pct','contrast_hi_pct','p']].to_string(index=False))
