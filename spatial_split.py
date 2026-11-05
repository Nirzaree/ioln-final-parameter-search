"""Train/val/test split by location, built once per region from ALL years, so a place has the same split in every year.

SPLIT_UNIT = 'polygon': every sampled polygon goes whole to one split. A polygon is identified by its area_m2 (the same
                         value on every pixel of the polygon, in every year). The polygon is placed by the grid block
                         (BLOCK_DEG) that contains its centroid, so polygons in the same block also share a split and
                         neighbouring polygons are separated as well.
SPLIT_UNIT = 'block'   : plain grid blocks of BLOCK_DEG (polygons can straddle a block edge).
Notebooks keep calling build_block_splits / add_split / load_year_split; the unit is chosen here."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

DATA_BASE = Path('data/regionwise_yearwise_samples')
SPLIT_DIR = Path('data/spatial_splits')
STUDY_DIR = Path('data/study_years')   # per-region usable / search years, written by 03_split_and_years.ipynb
SPLIT_UNIT = 'polygon'                 # 'polygon' or 'block'
BLOCK_DEG = 0.01                       # ~1.1 km blocks
FRACS = (0.70, 0.20, 0.10)             # train, val, test
SPLIT_NAMES = np.array(['train', 'val', 'test'])
MIN_VAL_PX, MIN_TEST_PX = 200, 50      # warn if a class has fewer pixels than this

_old = [5, 3, 4, 12, 11, 66, 19, 36, 9, 41, 24, 23, 75, 30, 25, 33, 76, 34, 61]
_new = [5, 3, 4, 12, 11, 66, 19, 14, 14, 14, 22, 22, 22, 22, 22, 33, 76, 34, 61]
CLASS_REMAP = dict(zip(_old, _new))
QUALITY = (pl.col('usable_count') >= 5) & (pl.col('quarters_present') >= 3)

SCORE_FEW_POLY = True   # score classes with < MIN_POLYS polygons too (their val/test come from the smaller polygons); False = leave them out of macro-F1
MIN_POLYS = 10         # a class with fewer polygons than this in the region is not scored (kept in training, listed as a data gap)
COORD_SCALE = 10_000    # lat/lon columns hold degrees x 10,000 (2_EDA divides them by 10000 to get EPSG:4326)


def _remap():
    return pl.col('class').replace_strict(CLASS_REMAP, default=pl.col('class')).cast(pl.Int32)


def _check_units(region, n=2000):
    """Fail loudly if lat/lon do not look like degrees after dividing by COORD_SCALE."""
    s = pl.scan_parquet(DATA_BASE / region / '**/*.parquet').select(['lat', 'lon']).head(n).collect()
    lat, lon = s['lat'].to_numpy() / COORD_SCALE, s['lon'].to_numpy() / COORD_SCALE
    if not (5 < np.nanmin(lat) and np.nanmax(lat) < 40 and 65 < np.nanmin(lon) and np.nanmax(lon) < 100):
        raise ValueError(f'{region}: lat/lon do not look like degrees after dividing by COORD_SCALE={COORD_SCALE} '
                         f'(lat {np.nanmin(lat):.2f}..{np.nanmax(lat):.2f}, lon {np.nanmin(lon):.2f}..{np.nanmax(lon):.2f})')


def _block_cols(block_deg):
    return [(pl.col('lon') / COORD_SCALE / block_deg).floor().cast(pl.Int64).alias('bx'),
            (pl.col('lat') / COORD_SCALE / block_deg).floor().cast(pl.Int64).alias('by')]


def _poly_key(block_deg=BLOCK_DEG):
    """Polygon id proxy = area_m2 (identical on every pixel of a polygon and in every year).
    Rows without area_m2 fall back to their own grid block so they still get a split."""
    bx = (pl.col('lon') / COORD_SCALE / block_deg).floor().cast(pl.Int64).cast(pl.Utf8)
    by = (pl.col('lat') / COORD_SCALE / block_deg).floor().cast(pl.Int64).cast(pl.Utf8)
    return (pl.when(pl.col('area_m2').is_null())
              .then(pl.concat_str([pl.lit('nopoly_'), bx, pl.lit('_'), by]))
              .otherwise(pl.col('area_m2').round(3).cast(pl.Utf8))
              .alias('poly'))


def _repair(M, g, Yb=None, n_years=None, min_year_frac=0.9, skip=None, pin=None):
    """Make sure every class is present in val AND test (and keeps train mass), moving as few blocks as possible.
    For each missing (class, split), pick a train block that holds the class, preferring blocks where the class is present in
    >= min_year_frac of the years (Yb: blocks x classes, number of years the class is present; None = ignore), and among those
    the block whose mass of that class is closest to the split's target share. Rarest classes first."""
    g = g.copy()
    tot, target = M.sum(0), np.array(FRACS)
    order = np.argsort((M > 0).sum(0))                       # rarest class (fewest blocks) first
    for c in order:
        if skip is not None and skip[c]:
            continue                                           # unscored class: stays wherever the optimiser put it (train)
        for s in (2, 1):                                       # test first: it is the smaller, harder split
            thr = min_year_frac * Yb[:, c].max() if Yb is not None and n_years else 0
            here = (g == s) & (M[:, c] > 0) & ((Yb[:, c] >= thr) if thr else True)
            if here.any():                                     # class already present (in enough years) in this split
                continue
            cand = np.where((g == 0) & (M[:, c] > 0) & ((pin < 0) if pin is not None else True))[0]
            train_tot = M[g == 0].sum(0)
            keep = [b for b in cand if (train_tot - M[b]).min() > 0]      # prefer blocks that leave every class in train
            cand = np.array(keep) if keep else cand
            if len(cand) == 0:
                continue
            if thr:
                ok = cand[Yb[cand, c] >= thr]
                cand = ok if len(ok) else cand[np.argsort(-Yb[cand, c])[:max(1, len(cand) // 4)]]
            b = cand[np.argmin(np.abs(M[cand, c] / tot[c] - target[s]))]
            g[b] = s
    return g


def _loss(part, partT, tot, skip=None):
    """Split quality. part/partT: 3 x classes pixel mass per split (partT = only blocks present in enough years)."""
    sh = part / tot
    keep = np.ones(part.shape[1], bool) if skip is None else ~skip
    L = (((sh[1, keep] - FRACS[1]) / FRACS[1]) ** 2).sum() + 0.25 * (((sh[2, keep] - FRACS[2]) / FRACS[2]) ** 2).sum()
    L += 100.0 * int((partT[1:][:, keep] <= 0).sum())   # scored class absent (in enough years) from val or test
    L += 1000.0 * int((part[0] <= 0).sum())             # any class absent from train
    if skip is not None and skip.any():                 # unscored (too few polygons) classes: keep them in train
        L += 50.0 * ((sh[1:, skip].sum(0)) ** 2).sum()
    return L


def _optimise(M, g0, Yb=None, n_years=None, min_year_frac=0.9, iters=40000, seed=0, skip=None, pin=None):
    """Local search (simulated annealing) over block->split assignments. Minimises the squared relative deviation of every
    class's val share from 15% (and, at lower weight, test share from 5%) so no class ends up with a val share of 50%+ just
    because one big polygon landed there; presence in val/test (in >= min_year_frac of years) is a hard-ish constraint."""
    tot = M.sum(0)
    if Yb is not None and n_years:
        MT = M * (Yb >= min_year_frac * Yb.max(0))
    else:
        MT = M
    rng = np.random.default_rng(seed)
    g = g0.copy()
    part = np.stack([M[g == k].sum(0) for k in range(3)]); partT = np.stack([MT[g == k].sum(0) for k in range(3)])
    cur = _loss(part, partT, tot, skip); best, gbest = cur, g.copy()
    n = len(M); T0 = max(cur, 1.0) * 0.02
    if pin is not None:
        g = np.where(pin >= 0, pin, g)
        part = np.stack([M[g == k].sum(0) for k in range(3)]); partT = np.stack([MT[g == k].sum(0) for k in range(3)])
        cur = _loss(part, partT, tot, skip); best, gbest = cur, g.copy()
    nz = np.where((M.sum(1) > 0) & ((pin < 0) if pin is not None else True))[0]
    iters = max(iters, 60 * len(nz))
    for it in range(iters):
        T = T0 * (1 - it / iters) + 1e-9
        b = nz[rng.integers(len(nz))]; k0 = g[b]
        k1 = rng.integers(3)
        if k1 == k0:
            continue
        part[k0] -= M[b]; part[k1] += M[b]; partT[k0] -= MT[b]; partT[k1] += MT[b]
        new = _loss(part, partT, tot, skip)
        if new <= cur or rng.random() < np.exp((cur - new) / T):
            g[b] = k1; cur = new
            if cur < best:
                best, gbest = cur, g.copy()
        else:
            part[k0] += M[b]; part[k1] -= M[b]; partT[k0] += MT[b]; partT[k1] -= MT[b]
    return gbest, best


def _assign(M, n_seeds, Yb=None, n_years=None, skip=None, pin=None, fold=0):
    """Random assignment of units (rows of M) to train/val/test; keep the seed whose per-class shares are closest to the FRACS targets,
    then improve it by local search, then repair any class still absent from val or test."""
    tot, target, cum = M.sum(0), np.array(FRACS), np.cumsum(FRACS)
    best = (np.inf, None, None)
    for seed in range(fold * 100000, fold * 100000 + n_seeds):     # fold k = another independent draw of the block assignment
        u = np.random.default_rng(seed).random(len(M))
        g = np.searchsorted(cum, u, side='right').clip(max=2)
        part = np.stack([M[g == k].sum(0) for k in range(3)])
        share = part / tot
        err = (((share - target[:, None]) / target[:, None]) ** 2).sum()
        err += 100.0 * int((part[1:] == 0).sum())          # penalise any class that is absent from val or test
        if err < best[0]:
            best = (err, seed, g)
    g, _ = _optimise(M, best[2], Yb, n_years, seed=fold, pin=pin)
    return best[1], _repair(M, g, Yb, n_years, skip=skip, pin=pin)


def _class_report(M, g, class_cols, tag, region, what, npoly=None):
    tot = M.sum(0)
    npoly = npoly or {}
    rep = pl.DataFrame({'class': [int(c) for c in class_cols],
                        'n_polygons': [int(npoly.get(c, -1)) for c in class_cols],
                        'scored': [bool(npoly.get(c, MIN_POLYS) >= MIN_POLYS) for c in class_cols],
                        **{f'{n}_px': M[g == k].sum(0).astype(int).tolist() for k, n in enumerate(SPLIT_NAMES)},
                        **{f'{n}_share': (M[g == k].sum(0) / tot).round(3).tolist() for k, n in enumerate(SPLIT_NAMES)}})
    rep.write_csv(SPLIT_DIR / f'{tag}_class_report.csv')
    print(f'[{region}] {what}. Class report -> {tag}_class_report.csv')
    gap = rep.filter(~pl.col('scored'))
    if gap.height:
        print(f'[{region}] DATA GAP: classes with fewer than {MIN_POLYS} polygons (low confidence; largest polygons in train, val/test from the smaller ones):\n'
              f'{gap.select(["class", "n_polygons", "train_px", "val_px", "test_px"])}')
    rep = rep.filter(pl.col('scored'))
    low = rep.filter((pl.col('val_px') < MIN_VAL_PX) | (pl.col('test_px') < MIN_TEST_PX))
    skew = rep.filter((pl.col('val_share') > 0.30) | (pl.col('train_share') < 0.55))
    if skew.height:
        print(f'[{region}] NOTE classes whose val share is above 30% or train share below 55% (few, large polygons):\n{skew}')
    if low.height:
        print(f'[{region}] WARNING classes with thin val/test coverage:\n{low}')



# ------------------------------------------------------------------ which years are used (per region)
def scored_classes(region, block_deg=BLOCK_DEG):
    """Classes with at least MIN_POLYS polygons in the region (read from the class report written with the split).
    Only these enter macro-F1; the others stay in training and are listed as data gaps."""
    base, yrs = _study_lf(region)
    p = SPLIT_DIR / f'{region}_poly_block{block_deg}_s{COORD_SCALE}_{_ytag(yrs)}_class_report.csv'
    r = pd.read_csv(p)
    if 'scored' not in r.columns:
        raise RuntimeError(f'{p.name} comes from the old split; rebuild it: FORCE_SPLIT=1 (03_split_and_years.ipynb)')
    if SCORE_FEW_POLY:
        return sorted(int(c) for c in r['class']), sorted(int(c) for c in r.loc[~r.scored, 'class'])   # second list = low-confidence classes
    return sorted(int(c) for c in r.loc[r.scored, 'class']), sorted(int(c) for c in r.loc[~r.scored, 'class'])


def study_years(region):
    """Usable years chosen in 03_split_and_years.ipynb; None if that step has not been run (= all years, old behaviour)."""
    p = STUDY_DIR / f'{region}.json'
    return json.load(open(p))['usable_years'] if p.exists() else None


def search_years(region):
    p = STUDY_DIR / f'{region}.json'
    if not p.exists():
        raise FileNotFoundError(f'{p} not found: run 03_split_and_years.ipynb for {region} first')
    return json.load(open(p))['search_years']


def save_study_years(region, usable, search, table=None):
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    with open(STUDY_DIR / f'{region}.json', 'w') as f:
        json.dump({'region': region, 'usable_years': [int(y) for y in usable], 'search_years': [int(y) for y in search],
                   'table': table.to_dicts() if table is not None else None}, f, indent=2)


def scan_years(region, years=None):
    """LazyFrame over the given years (all years if None)."""
    if not years:
        return pl.scan_parquet(DATA_BASE / region / '**/*.parquet')
    files = [str(p) for y in years for p in sorted((DATA_BASE / region / f'year={y}').glob('*.parquet'))]
    return pl.scan_parquet(files)


def _ytag(years):
    if not years:
        return 'allyears'
    return f'y{len(years)}_' + hashlib.md5(','.join(map(str, years)).encode()).hexdigest()[:6]


def usable_years(region, frac=0.85, min_cls_frac=0.75):
    """A year is usable if every class is present after the quality gate, at least `frac` of the region's best polygon
    count survives, and its rarest class keeps at least `min_cls_frac` of its best polygon count."""
    rows = []
    for f in sorted((DATA_BASE / region).glob('year=*')):
        g = (pl.scan_parquet(f / '*.parquet').select('area_m2', 'class', 'usable_count', 'quarters_present')
               .with_columns(_remap()).filter(QUALITY).collect())
        per = g.group_by('class').agg(pl.col('area_m2').n_unique().alias('n'))
        rows.append(dict(year=int(f.name.split('=')[1]), rows=g.height, polys=g['area_m2'].n_unique(),
                         n_cls=per.height, min_cls=int(per['n'].min()) if per.height else 0))
    t = pl.DataFrame(rows)
    ok = ((pl.col('n_cls') == t['n_cls'].max()) & (pl.col('polys') >= frac * t['polys'].max())
          & (pl.col('min_cls') >= min_cls_frac * t['min_cls'].max()))
    t = t.with_columns(ok.alias('usable'))
    return t.filter(pl.col('usable'))['year'].to_list(), t


def spread(years, k=6):
    """k evenly spaced years out of a sorted list."""
    years = sorted(years)
    if len(years) <= k:
        return list(years)
    return [years[round(i * (len(years) - 1) / (k - 1))] for i in range(k)]


def plan_years(region, n_spread=8, stress_years=None, frac=0.85, min_cls_frac=0.75, save=True, verbose=True):
    """Usable years + search years for one region, saved to data/study_years/{region}.json (what 04/06/07 read).
    Optional hand edits in data/study_years/overrides.json, e.g.
        {"HIM": {"exclude": [1993, 1994]}, "PEN": {"search": [2001, 2005, 2010, 2015, 2020, 2024]}}
    exclude: years removed from the usable pool. search: explicit search years (must be usable).
    Without 'search': n_spread evenly spread usable years + one stress year (first of stress_years that is free, else the
    usable year outside the spread with the fewest gated rows)."""
    usable, table = usable_years(region, frac, min_cls_frac)
    ov_path = STUDY_DIR / 'overrides.json'
    ov = (json.load(open(ov_path)) if ov_path.exists() else {}).get(region, {})
    usable = [y for y in usable if y not in set(ov.get('exclude', []))]
    if len(usable) < 3:
        raise ValueError(f'{region}: only {len(usable)} usable years')
    rows = {r['year']: r['rows'] for r in table.to_dicts()}
    if ov.get('search'):
        search = sorted(y for y in ov['search'] if y in usable)
        stress = None
    else:
        search = spread(usable, n_spread)
        rest = [y for y in usable if y not in search]
        stress = next((y for y in (stress_years or []) if y in rest), None)
        if stress is None and rest:
            stress = min(rest, key=lambda y: rows.get(y, 0))
        if stress is not None:
            search = sorted(search + [stress])
    if save:
        save_study_years(region, usable, search, table)
    if verbose:
        print(f'[{region}] usable {len(usable)} years ({usable[0]}-{usable[-1]}); search {len(search)}: {search}'
              + (f'; stress {stress}' if stress else '') + (' [overrides applied]' if ov else ''))
    return {'usable': usable, 'search': search, 'stress': stress, 'table': table}


def _study_lf(region):
    """Rows used to build the split: gated rows of the usable years (all rows of all years if step 03 has not been run)."""
    yrs = study_years(region)
    lf = scan_years(region, yrs)
    return (lf.filter(QUALITY) if yrs else lf), yrs


# ------------------------------------------------------------------ plain grid blocks
def _build_grid_splits(region, block_deg, n_seeds, force):
    base, yrs = _study_lf(region)
    tag = f'{region}_block{block_deg}_s{COORD_SCALE}_{_ytag(yrs)}'   # units and years in the name: stale lookups are not reused
    path = SPLIT_DIR / f'{tag}.parquet'
    if path.exists() and not force:
        return pl.read_parquet(path)
    _check_units(region)
    counts = (base
              .select(['lat', 'lon', 'class'])
              .with_columns(*_block_cols(block_deg), _remap())
              .group_by(['bx', 'by', 'class']).len().collect())
    wide = (counts.pivot(on='class', index=['bx', 'by'], values='len').fill_null(0).sort(['bx', 'by']))
    class_cols = sorted([c for c in wide.columns if c not in ('bx', 'by')], key=int)
    M = wide.select(class_cols).to_numpy().astype(np.float64)
    seed, g = _assign(M, n_seeds)
    lk = wide.select('bx', 'by').with_columns(pl.Series('split', SPLIT_NAMES[g]))
    lk.write_parquet(path)
    _class_report(M, g, class_cols, tag, region, f'{len(M):,} blocks of {block_deg} deg, seed {seed}')
    return lk


# ------------------------------------------------------------------ whole polygons, placed by centroid block
def _poly_core(poly, lat_deg, lon_deg, cls, block_deg, n_seeds, yp=None, n_years=None, skip_classes=(), fold=0):
    """Pure numpy/pandas part (unit-tested). Returns polygon table P (with split code per polygon), class matrix M, class ids."""
    d = pd.DataFrame({'poly': poly, 'lat': lat_deg, 'lon': lon_deg, 'cls': cls})
    P = d.groupby('poly').agg(clat=('lat', 'mean'), clon=('lon', 'mean'), lat_min=('lat', 'min'), lat_max=('lat', 'max'),
                              lon_min=('lon', 'min'), lon_max=('lon', 'max'), n_px=('cls', 'size'), n_cls=('cls', 'nunique'))
    P['bx'] = np.floor(P.clon / block_deg).astype(np.int64)
    P['by'] = np.floor(P.clat / block_deg).astype(np.int64)
    cnt = d.groupby(['poly', 'cls']).size().unstack(fill_value=0).loc[P.index]
    class_cols = list(cnt.columns)
    cnt['bx'], cnt['by'] = P.bx.to_numpy(), P.by.to_numpy()
    B = cnt.groupby(['bx', 'by']).sum()                      # blocks (of polygon centroids) x classes
    M = B[class_cols].to_numpy(np.float64)
    Yb = None
    if yp is not None:                                       # yp: polygons x classes = number of usable years the class is present
        Y = yp.reindex(index=P.index, columns=class_cols).fillna(0)
        Y['bx'], Y['by'] = P.bx.to_numpy(), P.by.to_numpy()
        Yb = Y.groupby(['bx', 'by']).max().reindex(B.index)[class_cols].to_numpy(np.float64)
    pin = np.full(len(M), -1)
    few = [c for c in class_cols if (cnt[c] > 0).sum() < MIN_POLYS]
    cnt_np = cnt[class_cols]
    for c in sorted(few, key=lambda c: (cnt[c] > 0).sum()):          # rarest first
        polys = cnt_np.index[(cnt_np[c] > 0).to_numpy()]
        order = cnt_np.loc[polys, c].sort_values(kind='stable')      # ascending size: smallest polygons are used for test / val
        n = len(order)
        n_te = 1 if n >= 3 else 0
        n_va = max(1, int(round(0.2 * n))) if n >= 2 else 0
        sp = np.array([2] * n_te + [1] * n_va + [0] * (n - n_te - n_va))
        for key, k in zip(order.index, sp):
            b = B.index.get_loc((int(P.loc[key, 'bx']), int(P.loc[key, 'by'])))
            if pin[b] < 0:
                pin[b] = k
    skip = np.array([c in set(few) for c in class_cols])             # classes whose val/test is fixed by the rule above
    seed, g = _assign(M, n_seeds, Yb, n_years, skip, pin, fold)
    gb = pd.Series(g, index=B.index)
    P['split_code'] = gb.reindex(pd.MultiIndex.from_arrays([P.bx.to_numpy(), P.by.to_numpy()])).to_numpy()
    return P, M, class_cols, seed, g


def _poly_diagnostics(P, region):
    ext_km = np.maximum((P.lat_max - P.lat_min) * 111.0,
                        (P.lon_max - P.lon_min) * 111.0 * np.cos(np.deg2rad(P.clat)))
    P['extent_km'] = ext_km
    q = lambda s, qs: ', '.join(f'{x:.1f}' for x in np.quantile(s, qs))
    print(f'[{region}] {len(P):,} polygons; pixels per polygon (median, p90, p99): {q(P.n_px, [.5, .9, .99])}; '
          f'extent km (median, p90, p99): {q(ext_km, [.5, .9, .99])}')
    bad = int((P.n_cls > 1).sum()) + int((ext_km > 5).sum())
    if bad:
        print(f'[{region}] WARNING {int((P.n_cls > 1).sum())} keys hold more than one class and {int((ext_km > 5).sum())} '
              f'span > 5 km: probably different polygons sharing one area_m2. See the polygon report.')


def _years_present(region, yrs, block_deg):
    """polygons x classes: in how many of the study years does the (gated) polygon carry that class? Used to pick val/test
    polygons that exist in (nearly) every year, so each year's val and test hold every class."""
    if not yrs:
        return None, None
    parts = []
    for y in yrs:
        parts.append(pl.scan_parquet(DATA_BASE / region / f'year={y}' / '*.parquet')
                       .select(['area_m2', 'class', 'usable_count', 'quarters_present', 'lat', 'lon'])
                       .with_columns(_remap()).filter(QUALITY)
                       .with_columns(_poly_key(block_deg)).select(['poly', 'class']).unique().collect())
    cnt = pl.concat(parts).group_by(['poly', 'class']).len().to_pandas()
    yp = cnt.pivot(index='poly', columns='class', values='len').fillna(0)
    yp.index = yp.index.astype(str)
    return yp, len(yrs)


def _build_poly_splits(region, block_deg, n_seeds, force, fold=0):
    base, yrs = _study_lf(region)
    tag = f'{region}_poly_block{block_deg}_s{COORD_SCALE}_{_ytag(yrs)}' + (f'_f{fold}' if fold else '')
    path = SPLIT_DIR / f'{tag}.parquet'
    if path.exists() and not force:
        return pl.read_parquet(path)
    _check_units(region)
    print(f'[{region}] building split from {"all years, ungated" if not yrs else str(len(yrs)) + " usable years, quality-gated"}')
    df = (base
          .select(['lat', 'lon', 'class', 'area_m2'])
          .with_columns(_poly_key(block_deg), _remap()).collect())
    n_null = df['area_m2'].null_count()
    if n_null:
        print(f'[{region}] WARNING {n_null:,} rows have no area_m2; they are split by grid block instead.')
    yp, n_years = _years_present(region, yrs, block_deg)
    pc = pd.DataFrame({'poly': df['poly'].to_list(), 'cls': df['class'].to_numpy()}).drop_duplicates()
    npoly = {int(k): int(v) for k, v in pc.groupby('cls').size().items()}
    skip_classes = [k for k, v in npoly.items() if v < MIN_POLYS]
    P, M, class_cols, seed, g = _poly_core(np.array(df['poly'].to_list(), dtype=object),
                                           df['lat'].to_numpy() / COORD_SCALE, df['lon'].to_numpy() / COORD_SCALE,
                                           df['class'].to_numpy(), block_deg, n_seeds, yp, n_years, skip_classes, fold)
    lk = pl.DataFrame({'poly': [str(k) for k in P.index], 'split': SPLIT_NAMES[P.split_code.to_numpy()].tolist()})
    lk.write_parquet(path)
    _poly_diagnostics(P, region)
    P.to_csv(SPLIT_DIR / f'{tag}_polygon_report.csv')
    _class_report(M, g, class_cols, tag, region,
                  f'{len(P):,} polygons in {len(M):,} centroid blocks of {block_deg} deg, seed {seed}', npoly)
    return lk


# ------------------------------------------------------------------ public API (unchanged names)
def build_block_splits(region, block_deg=BLOCK_DEG, n_seeds=300, force=False, fold=0):
    """fold=0 is THE split used by notebooks 04-07 (its test part stays untouched). fold>=1 are alternative block assignments for the
    multi-fold confirmation (08); classes with < MIN_POLYS polygons keep the same pinned polygons in every fold."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if SPLIT_UNIT == 'polygon':
        return _build_poly_splits(region, block_deg, n_seeds, force, fold)
    return _build_grid_splits(region, block_deg, n_seeds, force)


def add_split(df, region, block_deg=BLOCK_DEG, fold=0):
    """Adds a 'split' column to a DataFrame or LazyFrame that has lat/lon (and area_m2 in polygon mode)."""
    lk = build_block_splits(region, block_deg, fold=fold)
    if isinstance(df, pl.LazyFrame):
        lk = lk.lazy()
    if SPLIT_UNIT == 'polygon':
        return df.with_columns(_poly_key(block_deg)).join(lk, on='poly', how='left').drop('poly')
    return (df.with_columns(*_block_cols(block_deg))
              .join(lk, on=['bx', 'by'], how='left').drop('bx', 'by'))


def load_year_split(region, year, columns, block_deg=BLOCK_DEG, gated=False, fold=0):
    """Returns {'train': (X, y), 'val': (X, y), 'test': (X, y)} for one year."""
    df = (pl.read_parquet(DATA_BASE / region / f'year={year}' / '*.parquet')
          .with_columns(_remap()))
    if gated:
        df = df.filter(QUALITY)
    df = add_split(df, region, block_deg, fold)
    assert df['split'].null_count() == 0, ('rows without a split assignment: a polygon is missing from the lookup '
                                           '(was the lookup built for a different set of years?)')
    out = {}
    for s in SPLIT_NAMES:
        d = df.filter(pl.col('split') == s)
        out[str(s)] = (d.select(columns).to_numpy().astype(np.float32),
                       d['class'].to_numpy().astype(np.int32))
    return out



def split_summary(region, years=None, gated=True):
    """Per year: pixels and polygons in train/val/test, and classes that are missing from val or test."""
    years = years or study_years(region)
    rows = []
    for y in years:
        df = pl.read_parquet(DATA_BASE / region / f'year={y}' / '*.parquet').with_columns(_remap())
        if gated:
            df = df.filter(QUALITY)
        df = add_split(df, region)
        allc = set(df['class'].unique().to_list())
        r = {'year': int(y)}
        for s in SPLIT_NAMES:
            d = df.filter(pl.col('split') == s)
            r[f'{s}_px'] = d.height
            r[f'{s}_polys'] = d['area_m2'].n_unique()
            if s != 'train':
                r[f'{s}_missing'] = ','.join(map(str, sorted(allc - set(d['class'].unique().to_list()))))
        rows.append(r)
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ checks to run before trusting polygon mode
def polygon_check(region, years=None):
    """Is area_m2 a usable polygon id?  Are the same pixels (system:index) sampled in every year?"""
    folders = sorted((DATA_BASE / region).glob('year=*'))
    if years is None:
        years = [folders[0], folders[len(folders) // 2], folders[-1]]
    else:
        years = [DATA_BASE / region / f'year={y}' for y in years]
    cols = ['system:index', 'area_m2', 'lat', 'lon', 'class']
    dfs = {}
    for f in years:
        d = pl.read_parquet(f / '*.parquet').select(cols).with_columns(_poly_key())
        dfs[f.name] = d
        g = d.group_by('poly').len()['len'].to_numpy()
        print(f'{f.name}: {d.height:,} rows, {d["area_m2"].null_count()} without area_m2, {len(g):,} polygons, '
              f'pixels/polygon median {np.median(g):.0f}, p90 {np.quantile(g, .9):.0f}')
    names = list(dfs)
    a, b = dfs[names[0]], dfs[names[-1]]
    j = a.join(b, on='system:index', how='inner', suffix='_b')
    same_xy = int(((j['lat'] == j['lat_b']) & (j['lon'] == j['lon_b'])).sum())
    same_area = int((j['area_m2'] == j['area_m2_b']).sum())
    print(f'\n{names[0]} vs {names[-1]}: {j.height:,} shared system:index values '
          f'({100 * j.height / a.height:.1f}% of {names[0]}); same lat/lon on {same_xy:,}; same area_m2 on {same_area:,}')
    print('If shared == 0 the two years sample DIFFERENT pixels; if shared is large and lat/lon agree, the same pixels are reused.')
    # does one area_m2 stay compact?  (different polygons sharing a value would be spread out)
    d = a.with_columns((pl.col('lat') / COORD_SCALE).alias('la'), (pl.col('lon') / COORD_SCALE).alias('lo'))
    e = d.group_by('poly').agg(((pl.col('la').max() - pl.col('la').min()) * 111).alias('dy_km'),
                               ((pl.col('lo').max() - pl.col('lo').min()) * 111).alias('dx_km'),
                               pl.col('class').n_unique().alias('n_cls'))
    ext = np.maximum(e['dy_km'].to_numpy(), e['dx_km'].to_numpy())
    print(f'polygon extent km (median, p90, p99, max): {np.median(ext):.2f}, {np.quantile(ext, .9):.2f}, '
          f'{np.quantile(ext, .99):.2f}, {ext.max():.2f};  keys with >1 class: {int((e["n_cls"] > 1).sum())}')


# ---- diagnostic: how much does the split type change F1? ----
def split_sensitivity(region, year, cols, block_sizes=(0.005, 0.01, 0.05, 0.1),
                      n_trees=100, leaf=3, max_train=100_000, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    def fit_score(Xtr, ytr, Xv, yv):
        if len(Xtr) > max_train:
            idx = np.random.default_rng(seed).choice(len(Xtr), max_train, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=leaf,
                                     max_features='sqrt', n_jobs=4, random_state=seed).fit(Xtr, ytr)
        return f1_score(yv, clf.predict(Xv), average='macro')

    df = (pl.read_parquet(DATA_BASE / region / f'year={year}' / '*.parquet').with_columns(_remap()))
    X, y = df.select(cols).to_numpy().astype(np.float32), df['class'].to_numpy()
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    res = {'random pixel split (old)': fit_score(Xtr, ytr, Xv, yv)}
    for b in block_sizes:
        sp = load_year_split(region, year, cols, block_deg=b)
        res[f'blocks {b} deg'] = fit_score(*sp['train'], *sp['val'])
    return pd.Series(res)
