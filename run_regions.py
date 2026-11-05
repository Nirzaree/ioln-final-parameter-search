"""Run notebooks 03 -> 04 -> 06 -> 07 for several regions, one after another, unattended.

    python run_regions.py --regions HIM PEN DES IGP WGR ICR NEI --trials 60 --fresh
    python run_regions.py --regions HIM PEN DES IGP WGR ICR NEI --trials 100 --force     # later: adds the missing 40 trials to each study

* Each notebook is executed with nbconvert; the executed copy (with all output) goes to logs/executed/{REGION}_{notebook}.ipynb.
* A region whose notebook fails is marked FAILED in logs/status.json and the next region starts.
* A region whose {REGION}_pareto_frontier.csv already exists is skipped (use --force to rerun it; the study resumes).
* --fresh removes the old study/results of a region first (use it once: the split changed, so old trials are not comparable).
* Notebook 07 keeps its study in data/optuna_regional_gee_configs/{REGION}_study.db, so an interrupted region resumes
  (just run the same command again).
Keep the computer awake while it runs.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

NOTEBOOKS = ['03_split_and_years', '04_baseline_feature_ranking', '06_collinearity_pruning', '07_optuna_study']   # add 08_confirmation_folds with --steps after 07 is done
OUT = Path('data/optuna_regional_gee_configs')
LOGS = Path('logs'); (LOGS / 'executed').mkdir(parents=True, exist_ok=True)


ap = argparse.ArgumentParser()
ap.add_argument('--regions', nargs='+', required=True)
ap.add_argument('--trials', type=int, default=60)
ap.add_argument('--force', action='store_true')
ap.add_argument('--fresh', action='store_true', help='delete the region\'s old Optuna study (.db) and result files before running (needed once for the new split)')
ap.add_argument('--rf-jobs', type=int, default=2, help='cores per random forest fit in notebook 07')
ap.add_argument('--steps', nargs='+', default=NOTEBOOKS, help='notebooks to run per region, e.g. --steps 07_optuna_study (default: 03 04 06 07)')
ap.add_argument('--tag', default='', help='name for this run; gives it its own logs/status_{tag}.json (use when running several drivers at once)')
args = ap.parse_args()
STATUS = LOGS / (f'status_{args.tag}.json' if args.tag else 'status.json')

status = json.load(open(STATUS)) if STATUS.exists() else {}
def save(): json.dump(status, open(STATUS, 'w'), indent=2)

for region in args.regions:
    if (OUT / f'{region}_pareto_frontier.csv').exists() and not args.force and not args.fresh:
        print(f'[{region}] already has a Pareto frontier, skipping (use --force)'); continue
    if args.fresh:
        for p in OUT.glob(f'{region}_*'):
            if p.suffix in ('.db', '.csv', '.json', '.html', '.png'): p.unlink()
        print(f'[{region}] old study and results removed')
    env = dict(os.environ, REGIONS=region, N_TRIALS=str(args.trials), RF_JOBS=str(args.rf_jobs))
    t0 = time.time(); status[region] = {'state': 'RUNNING', 'step': None}; save()
    ok = True
    for nb in args.steps:
        status[region]['step'] = nb; save()
        print(f'[{region}] {nb} ...', flush=True)
        cmd = [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', f'{nb}.ipynb',
               '--output', f'{region}_{nb}', '--output-dir', str(LOGS / 'executed'),
               '--ExecutePreprocessor.timeout=-1']
        with open(LOGS / f'{region}_{nb}.log', 'w') as lf:
            rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            status[region] = {'state': 'FAILED', 'step': nb, 'log': str(LOGS / f'{region}_{nb}.log'),
                              'minutes': round((time.time() - t0) / 60, 1)}
            save(); ok = False; print(f'[{region}] FAILED at {nb} (see {LOGS}/{region}_{nb}.log)', flush=True); break
    if ok:
        status[region] = {'state': 'DONE', 'minutes': round((time.time() - t0) / 60, 1)}; save()
        print(f'[{region}] done in {status[region]["minutes"]} min', flush=True)
print(json.dumps(status, indent=2))
