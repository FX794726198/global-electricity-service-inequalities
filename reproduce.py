#!/usr/bin/env python3
"""Run the supplied analyses in an isolated output directory, preserving inputs."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys, time

ROOT=Path(__file__).resolve().parent
V33='10_manuscript/20261003_global_blackout_inequalities_v33'
V37='10_manuscript/20261003_global_blackout_inequalities_v37'
V38='10_manuscript/20261009_global_blackout_inequalities_v38'
V14='10_manuscript/20261002_global_blackout_inequalities_v14'
REV='10_manuscript/20261024_figure_revisions_v4'
STEPS={
 'core_checks':V33+'/core_reproducibility/src/reproduce_key_results.py',
 'historical':V33+'/src/gradient_v31.py',
 'regional':V33+'/core_reproducibility/class_gap_extension/src/analyse.py',
 'regional_checks':V33+'/core_reproducibility/class_gap_extension/src/verify.py',
 'turkiye_checks':V33+'/core_reproducibility/turkiye_extension/src/reproduce_extension.py',
 'recent':V37+'/src/analyse_recent_validation.py',
 'priorities':V37+'/src/analyse_cause_priority.py',
 'brazil_poverty':V37+'/src/analyse_poverty_priority.py',
 'cause_profiles':'10_manuscript/20261009_global_blackout_inequalities_v40/src/integrate_causes_v40.py',
 'figure1':V38+'/src/draw_merged_fig1_v38.py',
 'figure2':REV+'/src/fig2_minimal.py',
 'figure3':REV+'/src/fig3_minimal.py',
 'figure4':V33+'/src/households_v33.py',
 'figureS6':V37+'/src/plot_recent_validation.py',
 'figureS8':V37+'/src/draw_priority_fig4.py',
}
SUPPLEMENT={
 'figuresS1_S2_S4':[V14+'/src/plot_global_first.py','continents','fine_causes','firms'],
 'figureS3':'07_analysis/review_resolution_20261002/src/plot_definition_check.py',
 'figureS5':V33+'/core_reproducibility/src/plot_supp23_recent_v30.py',
 'figureS7':[V14+'/src/plot_main_figures.py','europe'],
 'figureS9':'07_analysis/review_resolution_20261002/src/plot_main.py',
 'figureS10':'07_analysis/turkiye_cause_extension_20261002/src/plot_settlement_inequality.py',
 'figureS11':'07_analysis/cause_conditioned_service_inequality_20261002/src/plot.py',
 'figureS12':V14+'/src/plot_cause_comparison.py',
 'figureS13':['07_analysis/brazil_detailed_cause_extension_20261002/src/plot_figures.py','supplement'],
 'figureS14':'07_analysis/social_cause_gradient_20261002/src/plot.py',
 'figureS15':'07_analysis/us_poverty_persistence_20261002/src/plot.py',
 'figuresS16_S17':V14+'/src/plot_household_inequality.py',
 'figureS18':'07_analysis/grid_reach_inequality_20261002/src/plot.py',
}
ALL_STEPS={**STEPS,**SUPPLEMENT}
FIGURES={
 'Figure1':V38+'/figures/Fig01_global_economic_service_inequality',
 'Figure2':REV+'/minimal_v5/Fig2_minimal',
 'Figure3':REV+'/minimal_v5/Fig3_minimal',
 'Figure4':V33+'/figures/Fig05_household_reliable_service',
 'FigureS6':V37+'/figures/SuppFig18_recent_year_validation',
 'FigureS8':V37+'/figures/SuppFig20_matched_priority_comparison',
 'FigureS1':V14+'/figures/SuppFig01_geographic_decomposition',
 'FigureS2':V14+'/figures/SuppFig02_hazard_specific_burden',
 'FigureS3':'07_analysis/review_resolution_20261002/figures/SuppFig19_benchmark_definition_support',
 'FigureS4':V14+'/figures/Fig02_global_firm_experience',
 'FigureS5':V33+'/core_reproducibility/figures/SuppFig23_recent_cross_source_checks',
 'FigureS7':V14+'/figures/Fig04_Europe_recurrence_and_duration',
 'FigureS9':'07_analysis/review_resolution_20261002/figures/Fig05_regional_interpretation',
 'FigureS10':'07_analysis/turkiye_cause_extension_20261002/figures/SuppFig17_turkiye_settlement_inequality',
 'FigureS11':'07_analysis/cause_conditioned_service_inequality_20261002/figures/SuppFig18_shared_incident_inequality',
 'FigureS12':V14+'/figures/Fig06_cause_specific_inequality_and_processes',
 'FigureS13':'07_analysis/brazil_detailed_cause_extension_20261002/figures/SuppFig12_brazil_fine_cause_persistence',
 'FigureS14':'07_analysis/social_cause_gradient_20261002/figures/SuppFig13_resident_disadvantage_and_interruption',
 'FigureS15':'07_analysis/us_poverty_persistence_20261002/figures/US_poverty_weather_persistence',
 'FigureS16':V14+'/figures/SuppFig08_household_interpretation',
 'FigureS17':V14+'/figures/SuppFig09_multicountry_households',
 'FigureS18':'07_analysis/grid_reach_inequality_20261002/figures/SuppFig16_connection_barriers',
}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/reproduction')
    ap.add_argument('--steps',nargs='+',choices=list(ALL_STEPS),default=None)
    ap.add_argument('--supplement',action='store_true',help='Also redraw supported supplementary figures from processed tables.')
    ap.add_argument('--resume',action='store_true',help='Use an existing isolated study copy.')
    args=ap.parse_args();output=args.output.resolve();study=output/'study'
    if not args.resume:
        if study.exists():raise SystemExit(f'{study} already exists; choose another output or use --resume')
        shutil.copytree(ROOT/'study',study,ignore=shutil.ignore_patterns('._*','.DS_Store','__pycache__','*.pyc'))
    elif not study.is_dir():raise SystemExit('--resume requires an existing output study')
    logs=output/'logs';logs.mkdir(parents=True,exist_ok=True)
    for code in study.rglob('src'):
        if code.is_dir():
            for name in ['figures','source_data','metadata','analysis','qa']:(code.parent/name).mkdir(exist_ok=True)
    report_path=output/'run_report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {'steps':{}}
    env=os.environ.copy();env.update(MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(output/'matplotlib_cache'))
    selected=args.steps if args.steps else list(ALL_STEPS if args.supplement else STEPS)
    for name in selected:
        start=time.monotonic();log=logs/(name+'.log')
        print(f'Running {name}',flush=True)
        with log.open('w') as f:
            spec=ALL_STEPS[name];spec=[spec] if isinstance(spec,str) else spec
            result=subprocess.run([sys.executable,str(study/spec[0]),*spec[1:]],cwd=study,env=env,stdout=f,stderr=subprocess.STDOUT)
        report['steps'][name]={'returncode':result.returncode,'seconds':round(time.monotonic()-start,2),'log':str(log.relative_to(output))}
        report_path.write_text(json.dumps(report,indent=2))
        if result.returncode:
            print(log.read_text()[-5000:]);raise SystemExit(result.returncode)
        print(f'Passed {name} ({report["steps"][name]["seconds"]} s)',flush=True)
    exported=output/'figures';exported.mkdir(exist_ok=True)
    for name,stem in FIGURES.items():
        for ext in ['png','pdf','svg']:
            path=study/(stem+'.'+ext)
            if path.exists():shutil.copyfile(path,exported/(name+'.'+ext))
    print(f'Results: {output}',flush=True)

if __name__=='__main__':main()
