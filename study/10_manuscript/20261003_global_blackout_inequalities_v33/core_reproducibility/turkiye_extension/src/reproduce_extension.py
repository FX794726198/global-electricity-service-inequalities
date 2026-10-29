"""Independent reconstruction from published indices and native cause aggregates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
annual=pd.read_csv(P/'data/annual_settlement_indices_2025.csv')
expected=pd.read_csv(P/'data/key_regional_results_2025.csv').set_index('provider')
native=pd.read_csv(P/'data/native_cause_settlement_totals_2025.csv')
old=P/'data/sedas_cause_settlement_metrics_2025.csv'
if not old.exists():old=P.parent/'turkiye_cause_extension_20261002/data/cause_settlement_metrics_2025.csv'
sedas=pd.read_csv(old)
checks=[]
def check(label,found,wanted):
 assert np.isclose(found,wanted,rtol=1e-10,atol=1e-9),(label,found,wanted)
 checks.append(dict(check=label,found=float(found),expected=float(wanted),passed=True))
for provider,g in annual.query("notification=='Unannounced' and voltage=='All'").groupby('provider'):
 g=g.set_index('settlement');r=g.loc['Rural'];u=g.loc['Urban']
 dr=r.hours/r.frequency;du=u.hours/u.frequency
 fpart=(r.frequency-u.frequency)*(dr+du)/2
 dpart=(dr-du)*(r.frequency+u.frequency)/2
 e=expected.loc[provider]
 for key,val in dict(frequency_ratio=r.frequency/u.frequency,duration_ratio=dr/du,burden_ratio=r.hours/u.hours,
                     hour_gap=r.hours-u.hours,frequency_component_h=fpart,duration_component_h=dpart,
                     frequency_share_pct=100*fpart/(r.hours-u.hours)).items():
  check(provider+' '+key,val,e[key])
 if provider=='uludag':
  assert pd.isna(e.weather_gap_h) and pd.isna(e.weather_share_of_gap_pct)
  continue
 if provider=='sedas':
  z=sedas[sedas.voltage.eq('All') & sedas.cause_group.isin(['Wind / lightning','Ice loading'])]
  hh=z.groupby('settlement').customer_hours.sum()
 else:
  # Independent literal native-weather selection, not the broader classified table.
  z=native[native.provider.eq(provider)&native.notification.eq('Bildirimsiz')&native.cause_native.str.contains('Olumsuz Hava Sartlari',regex=False)]
  hh=z.groupby('settlement').customer_hours.sum()
 weather_gap=hh['Rural']/r.users-hh['Urban']/u.users
 check(provider+' explicit-weather hour gap',weather_gap,e.weather_gap_h)
 check(provider+' explicit-weather percentage',100*weather_gap/(r.hours-u.hours),e.weather_share_of_gap_pct)
check('Registered supply accounts',annual.query("notification=='Unannounced' and voltage=='All' and settlement=='All'").users.sum(),15044310)
(P/'metadata/independent_reproduction.json').write_text(json.dumps(dict(checks=checks,all_passed=True,weather_unavailable_preserved=True),indent=2))
print(len(checks),'independent numerical checks passed')
