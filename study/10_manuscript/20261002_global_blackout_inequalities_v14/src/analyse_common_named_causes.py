"""Definition diagnostic: explicitly named lightning versus human/operating error."""
from pathlib import Path
import numpy as np
import pandas as pd
import analyse_process_contrasts as cp
P=Path(__file__).resolve().parents[1];D=P/'data'
a=pd.read_parquet(D/'ny_common_named_cause_year.parquet')
defs={'Lightning minus human-error frequency':{'Fl':1,'Fh':-1},
      'Lightning minus human-error duration':{'Hl':1,'Fl':-1,'Hh':-1,'Fh':1},
      'Lightning frequency':{'Fl':1},'Lightning duration':{'Hl':1,'Fl':-1},
      'Human-error frequency':{'Fh':1},'Human-error duration':{'Hh':1,'Fh':-1}}
for name in ['New York','New Zealand']:
    c=(pd.read_parquet(D/'ny_regional_income_context.parquet').rename(columns={'Region':'system'})
       if name=='New York' else pd.read_parquet(D/'nz_explicit21_cause_context.parquet'))
    c=c.drop_duplicates('system').set_index('system')
    v=a[a.source.eq(name)].groupby(['system','category'])[['frequency','burden_h']].mean().unstack().loc[c.index]
    X=pd.DataFrame({'const':1.,'log_income':c.log_income,'log_density':c.log_density})
    ys={'Fl':v['frequency','Lightning'],'Hl':v['burden_h','Lightning'],
        'Fh':v['frequency','Error'],'Hh':v['burden_h','Error']}
    cp.contrasts(c,X,ys,{'log_income':defs},dict(region=name,specification='Density adjusted'),
                 {'log_income':c.log_income.quantile(.75)-c.log_income.quantile(.25)})
o=pd.DataFrame(cp.out);o.to_parquet(D/'ny_common_named_cause_contrasts.parquet',index=False)
o.to_csv(P/'source_data/ny_common_named_cause_contrasts.csv',index=False)
print(o[['region','contrast','contrast_pct','contrast_lo_pct','contrast_hi_pct']].to_string(index=False))
