"""Block-size checks to run BEFORE the final Optuna study.
1) sweep(): F1 under a random pixel split vs block splits of increasing size (several random block draws).
2) pick_block(): smallest block size after which F1 stops falling.
3) same_class_vs_distance(): how far apart two pixels must be before knowing one tells you little about the other."""
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import spatial_split as ss

CHECKS_VERSION = 'coord-scale-fix-2'


def _load_year(region, year, gated=True):
    df = pl.read_parquet(ss.DATA_BASE / region / f'year={year}' / '*.parquet').with_columns(ss._remap())
    return df.filter(ss.QUALITY) if gated else df


def _block_split(df, block_deg, seed, fracs=(0.80, 0.15, 0.05)):
    """Random block assignment for ONE seed (no class-balance seed search, so seeds are independent draws)."""
    blk = df.select(ss._block_cols(block_deg)).unique().sort(['bx', 'by'])   # sorted: same seed -> same split on every run
    u = np.random.default_rng(seed).random(blk.height)
    g = np.searchsorted(np.cumsum(fracs), u, side='right').clip(max=2)
    blk = blk.with_columns(pl.Series('split', ss.SPLIT_NAMES[g]))
    return df.with_columns(*ss._block_cols(block_deg)).join(blk, on=['bx', 'by'], how='left')


def _fit_score(X, y, tr, va, n_trees, leaf, max_train, seed):
    if len(tr) > max_train:
        tr = np.random.default_rng(seed).choice(tr, max_train, replace=False)
    clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=leaf, max_features='sqrt',
                                 n_jobs=4, random_state=seed).fit(X[tr], y[tr])
    pred = clf.predict(X[va])
    # classes that exist in training but are absent from the held-out part (cannot be scored)
    missing = len(set(np.unique(y[tr])) - set(np.unique(y[va])))
    return f1_score(y[va], pred, average='macro'), missing


def sweep(region, years, cols, block_sizes=(0.005, 0.01, 0.02, 0.05, 0.1), seeds=(0, 1, 2),
          n_trees=100, leaf=2, max_train=60_000, drop_xy=False, gated=True):
    """One fixed, representative configuration. Run with drop_xy=False (harder case) and True."""
    use = [c for c in cols if not (drop_xy and c in ('lat', 'lon'))]
    assert hasattr(ss, 'COORD_SCALE'), 'old spatial_split loaded: replace the files, then restart the kernel'
    ss._check_units(region)
    print(f'spatial_checks {CHECKS_VERSION}; COORD_SCALE={ss.COORD_SCALE}')
    rows = []
    for yr in years:
        df = _load_year(region, yr, gated)
        X = df.select(use).to_numpy().astype(np.float32)
        y = df['class'].to_numpy().astype(np.int32)
        for seed in seeds:                                   # old behaviour: random pixels, 80/20
            tr, va = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=seed)
            f1, miss = _fit_score(X, y, tr, va, n_trees, leaf, max_train, seed)
            rows.append(dict(region=region, year=yr, scheme='random pixels', block_deg=0.0, seed=seed,
                             f1=f1, n_blocks=np.nan, n_rows=len(y), n_val=len(va), classes_missing_in_val=miss, xy=not drop_xy))
        for b in block_sizes:
            for seed in seeds:
                d = _block_split(df, b, seed)
                s = d['split'].to_numpy()
                tr, va = np.flatnonzero(s == 'train'), np.flatnonzero(s == 'val')
                f1, miss = _fit_score(X, y, tr, va, n_trees, leaf, max_train, seed)
                rows.append(dict(region=region, year=yr, scheme=f'blocks {b}', block_deg=b, seed=seed, f1=f1,
                                 n_blocks=d.select(['bx', 'by']).unique().height, n_rows=len(y), n_val=len(va),
                                 classes_missing_in_val=miss, xy=not drop_xy))
    return pd.DataFrame(rows)


def summarise(res):
    """Mean F1 over years and seeds, spread between block draws, and the drop relative to random pixels."""
    per = res.groupby(['xy', 'scheme', 'block_deg', 'year', 'seed']).f1.mean().reset_index()
    g = per.groupby(['xy', 'scheme', 'block_deg'])
    out = g.f1.mean().to_frame('f1_mean')
    out['f1_sd_over_seeds'] = per.groupby(['xy', 'scheme', 'block_deg', 'year']).f1.std().groupby(level=[0, 1, 2]).mean()
    out['n_blocks'] = res.groupby(['xy', 'scheme', 'block_deg']).n_blocks.mean()
    out['rows_per_block'] = res.groupby(['xy', 'scheme', 'block_deg']).n_rows.mean() / out['n_blocks']
    out['val_rows'] = res.groupby(['xy', 'scheme', 'block_deg']).n_val.mean()
    out['classes_missing_in_val'] = res.groupby(['xy', 'scheme', 'block_deg']).classes_missing_in_val.mean()
    out = out.reset_index().sort_values(['xy', 'block_deg'])
    big = out[out.block_deg == out.block_deg.max()]
    if (big.rows_per_block < 1.5).any():
        print('WARNING: blocks hold about one row each even at the largest size, so they are not grouping neighbouring '
              'pixels. Check the lat/lon units (spatial_split.COORD_SCALE).')
    base = out[out.scheme == 'random pixels'].set_index('xy').f1_mean
    out['drop_vs_random_pts'] = [100 * (b - base[x]) for x, b in zip(out.xy, out.f1_mean)]
    return out


def pick_block(summary, xy=True, tol=None, min_blocks=1000):
    """Smallest block size from which F1 stays flat (within `tol`) over the NEXT TWO larger sizes,
    and which still leaves at least `min_blocks` blocks. tol defaults to 0.5 points (it is NOT widened
    by seed noise, which would make a falling curve look flat). Returns (None, tol) when F1 never flattens;
    then the block size is a judgement, not something this function can read off."""
    s = summary[(summary.xy == xy) & (summary.block_deg > 0)].sort_values('block_deg').reset_index(drop=True)
    tol = tol if tol is not None else 0.005
    for i in range(len(s) - 2):
        flat = abs(s.f1_mean[i] - s.f1_mean[i + 1]) <= tol and abs(s.f1_mean[i + 1] - s.f1_mean[i + 2]) <= tol
        if flat and s.n_blocks[i] >= min_blocks:
            return float(s.block_deg[i]), tol
    return None, tol


def same_class_vs_distance(region, year, n=50_000, r_km=(0.25, 0.5, 1, 2, 3, 5, 10, 20), seed=0, gated=True):
    """P(two pixels share a class | distance) against the chance level. Where it levels off is the
    distance beyond which neighbouring pixels stop carrying information about each other."""
    from scipy.spatial import cKDTree
    df = _load_year(region, year, gated).select('lat', 'lon', 'class')
    df = df.sample(n=min(n, df.height), seed=seed)
    lat, lon, y = df['lat'].to_numpy() / ss.COORD_SCALE, df['lon'].to_numpy() / ss.COORD_SCALE, df['class'].to_numpy()
    xy = np.column_stack([lon * np.cos(np.deg2rad(lat.mean())) * 111.32, lat * 110.57])   # km
    edges = np.array(r_km, dtype=float)
    tot = cKDTree(xy).count_neighbors(cKDTree(xy), edges) - len(xy)          # remove self-pairs
    same = np.zeros(len(edges))
    for c in np.unique(y):
        t = cKDTree(xy[y == c]); same += t.count_neighbors(t, edges) - t.n
    tot_b, same_b = np.diff(np.r_[0, tot]), np.diff(np.r_[0, same])
    base = float(sum(np.mean(y == c) ** 2 for c in np.unique(y)))
    p = same_b / np.maximum(tot_b, 1)
    return pd.DataFrame({'up_to_km': edges, 'pairs': tot_b, 'p_same_class': p.round(3),
                         'chance_level': round(base, 3), 'ratio_to_chance': (p / base).round(2)})