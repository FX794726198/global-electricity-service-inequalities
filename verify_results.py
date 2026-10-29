#!/usr/bin/env python3
"""Compare regenerated results with the frozen manuscript-analysis tables."""
from pathlib import Path
import argparse,json,platform,sys
from importlib.metadata import version
import pandas as pd
from reproduce import STEPS,SUPPLEMENT,FIGURES

def main():
    root=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=root/'outputs/reproduction')
    args=ap.parse_args();out=args.output.resolve()
    run=json.loads((out/'run_report.json').read_text())
    required=list(STEPS)
    has_supplement=any(s in run['steps'] for s in SUPPLEMENT)
    if has_supplement:required+=list(SUPPLEMENT)
    for name in required:
        assert name in run['steps'],f'Missing run step: {name}'
        assert run['steps'][name]['returncode']==0,f'Failed run step: {name}'

    frozen=root/'study';generated=out/'study'
    v33=Path('10_manuscript/20261003_global_blackout_inequalities_v33/source_data')
    v37=Path('10_manuscript/20261003_global_blackout_inequalities_v37/source_data')
    v40=Path('10_manuscript/20261009_global_blackout_inequalities_v40/source_data')
    paths=list((frozen/v37).glob('*.csv'))+list((frozen/v33).glob('gradient_v31_*.csv'))
    paths += [frozen/v40/n for n in ['fig2_profiles.csv','fig2_shares.csv']]
    checked=[]
    for path in sorted(paths):
        if path.name.startswith('.'):continue
        relative=path.relative_to(frozen)
        a=pd.read_csv(path);b=pd.read_csv(generated/relative)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-9,atol=1e-8,obj=str(relative))
        checked.append({'path':str(relative),'rows':len(a),'columns':len(a.columns)})
    assert len(checked)==36,f'Expected 36 frozen table comparisons; got {len(checked)}'

    profiles=pd.read_csv(generated/v40/'fig2_profiles.csv')
    assert len(profiles)==40 and profiles.paired.sum()==36
    recent=pd.read_csv(generated/v37/'recent_country_year_available.csv')
    assert recent.country_iso3.nunique()==31
    poverty=json.loads((generated/v37.parent/'analysis/poverty_priority/self_checks.json').read_text())
    assert poverty['status']=='pass' and poverty['cohort_areas']==703 and poverty['municipalities']==275
    assert poverty['absent_external_categories_imputed_zero'] is False
    names=list(FIGURES) if has_supplement else ['Figure1','Figure2','Figure3','Figure4','FigureS6','FigureS8']
    for name in names:
        for ext in ['png','pdf','svg']:
            f=out/'figures'/f'{name}.{ext}'
            assert f.is_file() and f.stat().st_size>100,f'Missing/empty figure: {name}.{ext}'
    result={'status':'pass','scope':'Numerical package reproduction from processed inputs, not a fresh raw-provider audit',
        'python':sys.version.split()[0],'platform':platform.system(),'machine':platform.machine(),
        'versions':{n:version(n) for n in ['numpy','pandas','scipy','statsmodels','matplotlib','geopandas','shapely','pyproj','pyarrow','Pillow','tabulate','resvg-py']},
        'execution_steps':run['steps'],'frozen_table_comparisons':checked,'table_tolerance':{'rtol':1e-9,'atol':1e-8},
        'figures_verified':names,'cause_profiles':40,'paired_cause_profiles':36,'recent_countries':31,
        'brazil_service_areas':703,'brazil_municipalities':275}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS: {len(required)} execution steps, {len(checked)} frozen table comparisons and {len(names)} figures in three formats.')

if __name__=='__main__':main()
