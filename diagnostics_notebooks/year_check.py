"""Do the best configurations change from year to year?  (run AFTER the block-split lookups are rebuilt)

Usage in a notebook (same variables as 3d):
    from year_check import years_available, pick_configs, evaluate, report
    cnt = years_available(region); years = list(cnt.year[cnt.gated_rows >= 20_000])      # your "enough data" rule
    cfgs = pick_configs(f'data/optuna_regional_gee_configs/{region}_all_trials.json', n=10)
    res = evaluate(region, cfgs, years, all_ranked, uncorr_ranked, ref_col_series)
    report(res, search_years=[1990, 2000, 2006, 2013, 2024])
The model fitting and sampling replicate the 3d objective exactly (RF params, sampling strategies, feature sets),
but every year gets its OWN independent training draw (seeded by year and draw seed)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def years_available(region, data_base=None):
    import polars as pl
    import spatial_split as ss
    base = data_base or ss.DATA_BASE
    rows = []
    for d in sorted((base / region).glob('year=*')):
        n = pl.scan_parquet(d / '*.parquet').filter(ss.QUALITY).select(pl.len()).collect().item()
        rows.append(dict(year=int(d.name.split('=')[1]), gated_rows=int(n)))
    return pd.DataFrame(rows)


def _pareto_mask(f1, cx):
    keep = []
    for i in range(len(f1)):
        dom = ((f1 >= f1[i]) & (cx <= cx[i]) & ((f1 > f1[i]) | (cx < cx[i]))).any()
        keep.append(not dom)
    return np.array(keep)


def pick_configs(trials_json, n=8):
    """n configs spread over log-cost, for the confirmation step (08).
    Frontier configs only (when there are at least n of them): the cheapest, the peak-F1 one, and the frontier points nearest to
    evenly spaced log-cost targets in between. With fewer frontier points than n, the rest are filled with other trials at
    evenly spaced costs. Exact duplicates (same parameters) count once."""
    df = pd.read_json(trials_json)
    df = df[df.state == 'COMPLETE'].reset_index(drop=True)
    pcols = [c for c in df.columns if c.startswith('params_')]
    df = df.drop_duplicates(subset=pcols).reset_index(drop=True)
    f1, cx = df.values_0.to_numpy(), df.values_1.to_numpy()
    lc = np.log10(cx)
    front = list(np.flatnonzero(_pareto_mask(f1, cx)))
    chosen = []
    if len(front) >= n:
        must = [front[int(np.argmin(cx[front]))], front[int(np.argmax(f1[front]))]]
        chosen = list(dict.fromkeys(must))
        targets = np.linspace(lc[front].min(), lc[front].max(), n)
        for q in targets:
            if len(chosen) >= n:
                break
            order = sorted(front, key=lambda i: abs(lc[i] - q))
            nxt = next((i for i in order if i not in chosen), None)
            if nxt is not None:
                chosen.append(nxt)
    else:
        chosen = list(front)
        for q in np.linspace(lc.min(), lc.max(), n - len(chosen) + 2)[1:-1]:
            order = np.argsort(np.abs(lc - q))
            chosen.append(next(i for i in order if i not in chosen))
    chosen = sorted(chosen, key=lambda i: cx[i])
    out = []
    for i in chosen:
        r = df.iloc[i]
        mi = r.get('params_max_input_features')
        out.append(dict(trial=int(r.number), sampling_strategy=r.params_sampling_strategy,
                        feature_strategy=r.params_feature_strategy,
                        max_input_features=None if pd.isna(mi) else int(mi),
                        train_fraction=float(r.params_train_fraction),
                        minLeafPopulation=int(r.params_minLeafPopulation),
                        numberOfTrees=int(r.params_numberOfTrees),
                        old_f1=float(r.values_0), complexity=float(r.values_1)))
    return out


def _balanced(y, target_n, ref_series, rng):                 # identical to 3d get_balanced_indices
    classes = np.unique(y)
    present = ref_series.reindex(classes).fillna(0.0)
    scaled = (np.sqrt(present) / sum(np.sqrt(present))) * target_n
    sel = []
    for c in classes:
        idx = np.where(y == c)[0]
        k = min(len(idx), max(3, int(np.round(scaled.get(c, 0)))))
        sel.extend(rng.choice(idx, size=k, replace=False))
    return np.array(sel)


def _draw(y, strategy, frac, ref_series, rng):               # identical logic to 3d objective
    target_n = int(len(y) * frac)
    if frac >= 0.99 and strategy == 'stratified':
        return np.arange(len(y))
    if strategy == 'reference_balanced' and ref_series is not None:
        return _balanced(y, target_n, ref_series, rng)
    out = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        out.extend(rng.choice(idx, size=min(len(idx), max(3, int(target_n * len(idx) / len(y)))), replace=False))
    return np.array(out)


def evaluate(region, configs, years, all_ranked, uncorr_ranked, ref_series=None, seeds=(0, 1), n_jobs=4, load=None, scored=None,
             folds=(0,)):
    """scored: classes that enter macro-F1 (default: ss.scored_classes(region), same as notebook 07).
    folds: block-split folds (0 = the split used by 04-07; 1, 2, ... = other block assignments, see ss.build_block_splits(fold=)).
    Each row has f1 (val) and f1_test (test) for one config x fold x year x seed."""
    import spatial_split as ss
    if scored is None:
        scored = ss.scored_classes(region)[0]
    if load is None:
        load = lambda r, y, cols, fold=0: ss.load_year_split(r, y, cols, gated=True, fold=fold)
    idx_of = {f: i for i, f in enumerate(all_ranked)}
    rows = []

    def score(clf, X, y, cols):
        k = np.isin(y, scored)                                    # rows of unscored classes are dropped
        if k.sum() == 0:
            return np.nan
        return f1_score(y[k], clf.predict(X[k][:, cols]), labels=scored, average='macro', zero_division=0)

    for fold in folds:
        for yr in years:
            sp = load(region, yr, all_ranked, fold) if fold else load(region, yr, all_ranked)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = sp['train'], sp['val'], sp['test']
            if len(ytr) == 0 or len(yva) == 0:
                print(f'[{region}] fold {fold} {yr}: empty train/val, skipped'); continue
            for ci, c in enumerate(configs):
                names = uncorr_ranked if c['feature_strategy'] == 'uncorrelated' else all_ranked[:int(c['max_input_features'])]
                cols = [idx_of[f] for f in names]
                for s in seeds:
                    rng = np.random.RandomState(1000 * s + int(yr))        # independent draw for each year and seed
                    ch = _draw(ytr, c['sampling_strategy'], c['train_fraction'], ref_series, rng)
                    clf = RandomForestClassifier(n_estimators=c['numberOfTrees'], min_samples_leaf=c['minLeafPopulation'],
                                                 max_features='sqrt', max_depth=None, n_jobs=n_jobs, random_state=s)
                    clf.fit(Xtr[ch][:, cols], ytr[ch])
                    rows.append(dict(region=region, cfg=ci, trial=c.get('trial'), fold=fold, year=int(yr), seed=s,
                                     f1=score(clf, Xva, yva, cols), f1_test=score(clf, Xte, yte, cols),
                                     n_train=len(ch), n_val=int(np.isin(yva, scored).sum()), n_test=int(np.isin(yte, scored).sum())))
            print(f'[{region}] fold {fold} year {yr} done')
    return pd.DataFrame(rows)


def report(res, search_years=None):
    M = res.groupby(['cfg', 'year']).f1.mean().unstack()                     # configs x years
    out = {}
    g, rm, cm = M.values.mean(), M.mean(1), M.mean(0)
    ss_cfg = M.shape[1] * ((rm - g) ** 2).sum(); ss_yr = M.shape[0] * ((cm - g) ** 2).sum()
    ss_int = ((M.sub(rm, axis=0).sub(cm, axis=1) + g) ** 2).sum().sum(); tot = ss_cfg + ss_yr + ss_int
    print('Share of F1 variance explained by:  config %.0f%%   year (all configs move together) %.0f%%   '
          'config x year (ranking changes) %.0f%%' % (100 * ss_cfg / tot, 100 * ss_yr / tot, 100 * ss_int / tot))
    seed_sd = res.groupby(['cfg', 'year']).f1.std().mean()
    print('Typical F1 SD between training draws of the same config and year: %.4f' % seed_sd)
    yr = pd.DataFrame({'mean_f1_all_configs': cm.round(4), 'best_config_f1': M.max(0).round(4),
                       'best_cfg': M.idxmax(0), 'n_train_med': res.groupby('year').n_train.median()})
    print('\nPer year:\n', yr.to_string())
    C = M.corr(method='spearman'); off = C.values[~np.eye(len(C), dtype=bool)]
    print('\nRank agreement of configs between years (Spearman): mean %.2f, worst pair %.2f' % (off.mean(), off.min()))
    cfg = pd.DataFrame({'mean_f1': rm.round(4), 'worst_year_f1': M.min(1).round(4), 'worst_year': M.idxmin(1),
                        'sd_across_years': M.std(1).round(4)})
    print('\nPer config:\n', cfg.to_string())
    out.update(M=M, per_year=yr, per_cfg=cfg, rank_corr=C)
    if search_years:
        S = [y for y in search_years if y in M.columns]; H = [y for y in M.columns if y not in S]
        if H:
            a, b = M[S].mean(1), M[H].mean(1)
            top = a.idxmax()
            print('\nPick configs on %s, check on the other years %s:' % (S, H))
            print('  Spearman(mean over search years, mean over other years) = %.2f' % a.corr(b, method='spearman'))
            print('  config chosen on search years scores %.4f on other years; best config there scores %.4f (regret %.4f)'
                  % (b[top], b.max(), b.max() - b[top]))
            out['regret'] = float(b.max() - b[top])
    return out


def tier_summary(res, configs):
    """One row per config (cheap -> expensive) for the group guidelines.
    f1_val_* : val F1 over all folds x years (median, 10th-90th percentile).  fold_sd : SD of the fold means (spatial split noise).
    year_sd  : SD of the year means (year-to-year range).  f1_test_fold0 : F1 on the untouched 10% test split (fold 0), median over years."""
    out = []
    for ci, c in enumerate(configs):
        r = res[res.cfg == ci]
        fm = r.groupby('fold').f1.mean(); ym = r.groupby('year').f1.mean()
        out.append(dict(cfg=ci, trial=c.get('trial'), complexity=c.get('complexity'), search_f1=c.get('old_f1'),
                        f1_val_median=r.f1.median(), f1_val_p10=r.f1.quantile(.1), f1_val_p90=r.f1.quantile(.9),
                        fold_sd=fm.std() if len(fm) > 1 else np.nan, year_sd=ym.std(),
                        f1_test_fold0=r[r.fold == 0].groupby('year').f1_test.mean().median()))
    return pd.DataFrame(out).sort_values('complexity').reset_index(drop=True)
