"""Does the feature RANKING limit what top-N features can do?  (side test for notebook 10)

For a few years of a region: rank the bands on THAT year's train pixels, then score the classifier on that year's val pixels using
the top N bands, for several N.  The same is done with
  pooled       the ranking used in the study (notebook 04: train pixels pooled over all usable years)
  random       random sets of N bands (the 'does ranking help at all' baseline)
  decorrelated the per-year ranking with bands dropped when |r| > thr with an already chosen, higher-ranked band
  study_uncorr the pruned band set of notebook 06 (one point)
One fixed classifier (100 trees, min leaf 2) and one training draw per year are shared by all methods, so only the band set differs.
Ranking never sees val pixels."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from year_check import _draw

NS = (5, 10, 20, 40, 80, 'all')


def _rf(n_jobs, seed, **kw):
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_features='sqrt', n_jobs=n_jobs, random_state=seed, **kw)


def per_year_importance(Xtr, ytr, rng, n_jobs=4, cap=100_000):
    idx = rng.choice(len(ytr), size=min(cap, len(ytr)), replace=False)          # same recipe as notebook 04
    rf = RandomForestClassifier(n_estimators=100, max_depth=14, n_jobs=n_jobs, random_state=42).fit(Xtr[idx], ytr[idx])
    return rf.feature_importances_


def decorrelated_order(order, X, thr=0.8):
    """Greedy: walk down the ranking, keep a band only if |r| <= thr with every band kept so far."""
    C = np.abs(np.corrcoef(X, rowvar=False))
    keep = []
    for j in order:
        if all(C[j, k] <= thr for k in keep):
            keep.append(int(j))
    return keep


def _score(clf, Xva, yva, cols, scored):
    return f1_score(yva, clf.predict(Xva[:, cols]), labels=scored, average='macro', zero_division=0)


def check_region(region, years, all_ranked, load, scored, pooled_imp=None, uncorr=None, ns=NS, n_random=5, seeds=(0,),
                 n_jobs=4, train_cap=40_000, thr=0.8, verbose=True):
    """load(region, year, all_ranked) -> {'train': (X, y), 'val': (X, y), ...} with columns in the order of all_ranked
    (all_ranked is sorted by the pooled ranking, so column j is the pooled j-th band).  Returns (scores, agreement) data frames."""
    p = len(all_ranked)
    nlist = [p if n == 'all' else int(n) for n in ns]
    rows, agree, tops = [], [], {}
    uncorr_idx = [all_ranked.index(f) for f in uncorr] if uncorr else None
    for yr in years:
        sp = load(region, yr, all_ranked)
        (Xtr, ytr), (Xva, yva) = sp['train'], sp['val']
        k = np.isin(yva, scored); Xva, yva = Xva[k], yva[k]
        if len(ytr) == 0 or len(yva) == 0:
            continue
        rng0 = np.random.RandomState(int(yr))
        imp = per_year_importance(Xtr, ytr, rng0, n_jobs)
        order_y = list(np.argsort(-imp, kind='stable'))
        order_p = list(range(p))                                   # pooled ranking = column order
        sub = Xtr[rng0.choice(len(ytr), size=min(30_000, len(ytr)), replace=False)]
        order_d = decorrelated_order(order_y, sub, thr)
        tops[yr] = set(order_y[:20])
        a = dict(year=int(yr), top20_jaccard_vs_pooled=len(tops[yr] & set(order_p[:20])) / len(tops[yr] | set(order_p[:20])))
        if pooled_imp is not None:
            a['importance_spearman_vs_pooled'] = pd.Series(imp).corr(pd.Series(pooled_imp), method='spearman')
        a['n_decorrelated'] = len(order_d)
        agree.append(a)
        for s in seeds:
            rng = np.random.RandomState(1000 * s + int(yr))
            ch = _draw(ytr, 'stratified', min(1.0, train_cap / len(ytr)), None, rng)
            Xs, ys = Xtr[ch], ytr[ch]

            def run(method, cols, n, draw=0):
                clf = _rf(n_jobs, s).fit(Xs[:, cols], ys)
                rows.append(dict(region=region, year=int(yr), seed=s, method=method, n=int(n), draw=draw,
                                 f1=_score(clf, Xva, yva, cols, scored)))
            for n in nlist:
                run('per_year', order_y[:n], n)
                run('pooled', order_p[:n], n)
                if n < p:
                    for d in range(n_random):
                        run('random', list(np.random.RandomState(7919 * int(yr) + 31 * n + d).choice(p, n, replace=False)), n, d)
                if n <= len(order_d):
                    run('decorrelated', order_d[:n], n)
            if uncorr_idx:
                run('study_uncorr', uncorr_idx, len(uncorr_idx))
        if verbose:
            print(f'[{region}] {yr} done')
    sc = pd.DataFrame(rows)
    ag = pd.DataFrame(agree)
    if len(tops) > 1:
        ys = list(tops)
        pj = [len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for i, a in enumerate(ys) for b in ys[i + 1:]]
        ag.attrs['mean_year_vs_year_top20_jaccard'] = float(np.mean(pj))
    return sc, ag


def summarize(sc, tols=(0.005, 0.01)):
    """Mean F1 over years per method and N; smallest N within tol of the all-band F1."""
    m = sc.groupby(['method', 'n']).f1.mean().unstack(0)
    full = m.loc[m.index.max(), 'pooled']
    res = {'f1_all_bands': full}
    for t in tols:
        for meth in ('per_year', 'pooled', 'decorrelated', 'random'):
            if meth in m:
                ok = m.index[m[meth] >= full - t]
                res[f'{meth}_N_within_{t}'] = int(ok.min()) if len(ok) else np.nan
    return m, res


def plot_region(sc, region, path, uncorr_n=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    g = sc.groupby(['method', 'n']).f1
    for meth, col in (('per_year', '#1f77b4'), ('pooled', '#d95f02'), ('decorrelated', '#2ca02c')):
        s = sc[sc.method == meth].groupby('n').f1.mean()
        if len(s):
            ax.plot(s.index, s.values, marker='o', color=col, label={'per_year': 'ranked per year', 'pooled': 'pooled ranking (study)', 'decorrelated': 'per-year, decorrelated'}[meth])
    r = sc[sc.method == 'random']
    if len(r):
        mean = r.groupby('n').f1.mean(); lo = r.groupby('n').f1.min(); hi = r.groupby('n').f1.max()
        ax.plot(mean.index, mean.values, color='#777777', linestyle='--', marker='o', label='random bands (mean, min to max)')
        ax.fill_between(mean.index, lo.values, hi.values, color='#777777', alpha=0.15)
    u = sc[sc.method == 'study_uncorr']
    if len(u):
        ax.scatter(u.groupby('n').f1.mean().index, u.groupby('n').f1.mean().values, marker='*', s=140, color='#a03a63', zorder=5, label='study pruned set (06)')
    ax.set_xscale('log'); ax.set_xlabel('number of bands'); ax.set_ylabel('macro-F1 on val (mean over years)')
    ax.set_title(f'{region}: F1 against number of ranked bands'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
