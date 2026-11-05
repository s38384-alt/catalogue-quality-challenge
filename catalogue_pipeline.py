"""Catalogue quality pipeline: features, seller-disjoint CV, inference, controlled comparisons.

Same features and LightGBM settings as the team's original pipeline, with the validation fixes:
  * fixed number of trees; the original early-stopped on the held-out fold, i.e. chose the
    model size with validation labels;
  * every learned statistic (phrase purity, modal category, modal pack size, price/weight
    medians and MADs, categorical levels) is fitted on the training fold only; the original
    computed price/weight medians on all rows of each frame (validation and test included)
    and categorical levels on train+test;
  * deterministic LightGBM with a fixed thread count, library versions recorded.

Run:  python catalogue_pipeline.py --data-dir data/public --out results

Features are organised in named groups so a comparison changes exactly one group:
  base      raw attributes, lengths, missingness flags, category word in text, categoricals
  quantity  main-item count parsed from text vs pack_size   (extractor is swappable)
  phrase    fold-fitted per-product statistics: category purity/contradiction, modal pack size
  anomaly   fold-fitted robust z-scores of price and weight (package and per item) per product
  composite rule_score / extreme_AND_mismatch, built only from the groups that are active
"""
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
N_TREES = 400
CATS = ['declared_category', 'brand', 'condition']
WORD2NUM = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
            'eight': 8, 'nine': 9, 'ten': 10, 'zero': 0, 'single': 1}
NUMW = 'one|two|three|four|five|six|seven|eight|nine|ten'


# --------------------------------------------------------------------------- folds
def fold_assignment(sellers, n_splits=5, seed=SEED):
    """Organisers' seller-disjoint folds: SHA-256('42:<seller_id>')[:16] mod 5. Label-free."""
    return np.array([int(hashlib.sha256(f'{seed}:{s}'.encode('utf-8')).hexdigest()[:16], 16)
                     % n_splits for s in sellers.astype(str)])


# --------------------------------------------------------------- quantity extractors
def qty_first_pcs(text):
    """Starter heuristic from the organisers' baseline: first '<digits> pcs' in the text.
    Picks up free accessories and ignores quantities written as words."""
    m = re.search(r'(\d+)\s*pcs', text)
    return int(m.group(1)) if m else None


def qty_main_items(text):
    """Team parser: removes the free-accessory sentence, then tries main-item patterns
    (digits and number words) in a fixed priority order."""
    t = re.sub(r'also includes \d+ pcs? of free accessory[^.]*\.?', ' ', text)
    t = re.sub(r'free accessory[^.]*\.?', ' ', t)
    pats = [r'included main items:\s*(\d+)',
            r'(\d+)\s+individually boxed main items', rf'({NUMW})\s+individually boxed main items',
            r'(\d+)\s+main items', rf'({NUMW})\s+main items',
            rf'pack of ({NUMW}) main items', r'pack of (\d+) main items',
            r'bundle quantity:\s*(\d+)', r'(\d+)\s*-?pack', r'sold as a (\d+)-pack',
            r'\(quantity\s*(\d+)\)', r'quantity\s*(\d+)', r'(\d+)\s*units',
            r'package contains\s*(\d+)', r'(\d+)\s*pcs', rf'pack of ({NUMW})']
    for p in pats:
        m = re.search(p, t)
        if m:
            v = m.group(1)
            return WORD2NUM.get(v) if v.isalpha() else int(v)
    return None


EXTRACTORS = {'first_pcs': qty_first_pcs, 'main_items': qty_main_items}


# ------------------------------------------------------------ deterministic features
def product_phrase(title, desc):
    """Product noun: '<noun> by <Brand>' in the description, else the title head."""
    m = re.search(r'([a-z][a-z\s\-/]{2,40}?)\s+by\s+(?:mira|noro|velto|tenso|aster)', desc)
    if m:
        return m.group(1).strip()
    m = re.search(r'([a-z][a-z\s\-/]{2,40}?)\s*[—\-/]', title)
    return m.group(1).strip() if m else ''


def base_frame(df, extractor):
    """Label-free, row-wise features. Nothing is learned here."""
    x = df.copy()
    title = x.title.fillna('').astype(str).str.lower()
    desc = x.description.fillna('').astype(str).str.lower()
    x['text'] = (title + ' ' + desc).str.replace(r'\s+', ' ', regex=True).str.strip()
    # base
    x['text_is_short'] = (x.text.str.len() < 20).astype(int)
    for c in ['title', 'description', 'declared_category', 'brand', 'condition']:
        x['isna_' + c] = x[c].isna().astype(int)
    x['isna_price'] = x.price_pln.isna().astype(int)
    x['isna_weight'] = x.shipping_weight_kg.isna().astype(int)
    for c in ['price_pln', 'shipping_weight_kg', 'pack_size']:
        x['log_' + c] = np.log1p(x[c].clip(lower=0))
    x['price_per_item'] = np.log1p(x.price_pln / x.pack_size.clip(lower=1))
    x['mass_per_item'] = np.log1p(x.shipping_weight_kg / x.pack_size.clip(lower=1))
    x['price_per_weight'] = np.log1p(x.price_pln / (x.shipping_weight_kg + 1e-3))
    x['title_len'], x['desc_len'] = title.str.len(), desc.str.len()
    x['desc_title_ratio'] = x.desc_len / (x.title_len + 1)
    cat = x.declared_category.fillna('missing').astype(str).str.lower()
    x['category_in_text'] = [int(c in t) for c, t in zip(cat, x.text)]
    # quantity (depends on the extractor)
    x['main_items'] = pd.to_numeric(x.text.map(EXTRACTORS[extractor]), errors='coerce')
    m = re.compile(r'also includes (\d+) pcs? of free accessory')
    x['free_acc'] = x.text.map(lambda t: int(m.search(t).group(1)) if m.search(t) else 0)
    pk, mi = x.pack_size, x.main_items
    x['mvp_mismatch'] = (pk.notna() & mi.notna() & (pk != mi)).astype(int)
    x['pack_vs_main'] = pk - mi
    x['pack_vs_main_abs'] = (pk - mi).abs()
    # product key used by the fold-fitted groups
    x['product_phrase'] = [re.sub(r'\s+', ' ', product_phrase(t, d)).strip()
                           for t, d in zip(title, desc)]
    x['grp_pc'] = x.product_phrase + '||' + x.declared_category.astype(str)
    return x


GROUP_COLUMNS = {
    'base': ['log_price_pln', 'log_shipping_weight_kg', 'log_pack_size', 'price_per_item',
             'mass_per_item', 'price_per_weight', 'title_len', 'desc_len', 'desc_title_ratio',
             'isna_title', 'isna_description', 'isna_declared_category', 'isna_brand',
             'isna_condition', 'isna_price', 'isna_weight', 'category_in_text', 'text_is_short'],
    'quantity': ['main_items', 'free_acc', 'mvp_mismatch', 'pack_vs_main', 'pack_vs_main_abs'],
    'phrase': ['phrase_purity', 'cat_contradicts_strict', 'pack_anomaly_pp'],
    # 8 z-scores + 6 per-item flags + 3 summaries = 17 columns (as in the team's FEATS)
    'anomaly': [f'{v}_{k}' for v in ['price', 'weight', 'price_item', 'weight_item']
                for k in ['z_pp2', 'z_pc']]
               + [f'{v}_{k}' for v in ['price_item', 'weight_item']
                  for k in ['hi_pp2', 'extreme_pp2', 'extreme_pc']]
               + ['any_extreme_anom', 'anom_pp_minus_pc', 'anom_wt_pp_minus_pc'],
}
ANOM_VALUES = {
    'price': lambda d: d.price_pln,
    'weight': lambda d: d.shipping_weight_kg,
    'price_item': lambda d: d.price_pln / d.pack_size.clip(lower=1),
    'weight_item': lambda d: d.shipping_weight_kg / d.pack_size.clip(lower=1),
}


# ---------------------------------------------------------- fold-fitted feature groups
def fit_transform_fold(x_fit, frames, groups):
    """Fit per-product statistics on the training fold `x_fit` and apply them to each frame."""
    out = [f.copy() for f in frames]
    if 'phrase' in groups:
        def purity(s):
            s = s.dropna()
            if len(s) < 5:
                return (0.0, None)
            vc = s.value_counts(normalize=True)
            return (float(vc.iloc[0]), vc.index[0])
        st = x_fit.groupby('product_phrase').declared_category.agg(purity)
        pur = {p: v[0] for p, v in st.items()}
        modal = {p: v[1] for p, v in st.items()}
        pmode = x_fit.groupby('product_phrase').pack_size.agg(
            lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan).to_dict()
        for f in out:
            f['phrase_purity'] = f.product_phrase.map(pur).fillna(0.0)
            mc = f.product_phrase.map(modal)
            f['cat_contradicts_strict'] = ((f.phrase_purity >= 0.90) & mc.notna()
                                           & f.declared_category.notna()
                                           & (mc != f.declared_category)).astype(int)
            pm = f.product_phrase.map(pmode)
            f['pack_anomaly_pp'] = (pm.notna() & f.pack_size.notna()
                                    & (f.pack_size != pm)).astype(int)
    if 'anomaly' in groups:
        vc_pp = x_fit.product_phrase.value_counts()
        vc_pc = x_fit.grp_pc.value_counts()
        trusted_pp, trusted_pc = set(vc_pp[vc_pp >= 8].index), set(vc_pc[vc_pc >= 6].index)
        stats = {}
        for name, fn in ANOM_VALUES.items():
            v = fn(x_fit)
            for key in ['product_phrase', 'grp_pc']:
                med = v.groupby(x_fit[key]).median()
                mad = (v - x_fit[key].map(med)).abs().groupby(x_fit[key]).median()
                stats[name, key] = (med, mad)
        for f in out:
            for name, fn in ANOM_VALUES.items():
                v = fn(f)
                for key, suf in [('product_phrase', 'pp2'), ('grp_pc', 'pc')]:
                    med, mad = stats[name, key]
                    f[f'{name}_z_{suf}'] = (v - f[key].map(med)) / (f[key].map(mad) + 1e-6)
                zpp, zpc = f[f'{name}_z_pp2'].abs(), f[f'{name}_z_pc'].abs()
                ok_pp, ok_pc = f.product_phrase.isin(trusted_pp), f.grp_pc.isin(trusted_pc)
                f[f'{name}_hi_pp2'] = ((zpp > 4) & ok_pp).astype(int)
                f[f'{name}_extreme_pp2'] = ((zpp > 6) & ok_pp).astype(int)
                f[f'{name}_extreme_pc'] = ((zpc > 6) & ok_pc).astype(int)
            f['any_extreme_anom'] = (f.price_item_extreme_pp2 | f.weight_item_extreme_pp2
                                     | f.price_item_extreme_pc | f.weight_item_extreme_pc).astype(int)
            f['anom_pp_minus_pc'] = (f.price_item_z_pp2.abs() - f.price_item_z_pc.abs()).fillna(0)
            f['anom_wt_pp_minus_pc'] = (f.weight_item_z_pp2.abs() - f.weight_item_z_pc.abs()).fillna(0)
    # composite features use only the active groups
    for f in out:
        score = pd.Series(0.0, index=f.index)
        if 'quantity' in groups:
            score += 10.0 * f.mvp_mismatch
        if 'phrase' in groups:
            score += 5.0 * f.cat_contradicts_strict
        if 'anomaly' in groups:
            score += 2.0 * f.any_extreme_anom + f.price_item_hi_pp2 + f.weight_item_hi_pp2
        f['rule_score'] = score
        if {'quantity', 'anomaly'} <= set(groups):
            f['extreme_AND_mismatch'] = f.any_extreme_anom * f.mvp_mismatch
    # categorical levels learned on the training fold only
    for c in CATS:
        levels = pd.Categorical(x_fit[c].fillna('missing')).categories
        for f in out:
            f[c] = pd.Categorical(f[c].fillna('missing'), categories=levels)
    return out


def feature_list(groups):
    cols = [c for g in ['base', 'quantity', 'phrase', 'anomaly'] if g in groups
            for c in GROUP_COLUMNS[g]]
    cols.append('rule_score')
    if {'quantity', 'anomaly'} <= set(groups):
        cols.append('extreme_AND_mismatch')
    return cols + CATS


def make_model(fold):
    """Team LightGBM settings with a fixed number of trees, deterministic.
    The original also set subsample=0.85, which LightGBM ignores while subsample_freq=0
    (the default), so it is left out: the fitted model is the same."""
    return lgb.LGBMClassifier(n_estimators=N_TREES, learning_rate=0.03, num_leaves=95,
                              min_child_samples=20, colsample_bytree=0.85,
                              reg_lambda=1.0, objective='binary', verbosity=-1, n_jobs=4,
                              deterministic=True, force_row_wise=True, random_state=SEED + fold)


# ----------------------------------------------------------------------------- CV run
def run_cv(train, test, extractor, groups, name, out_dir, data_dir=None):
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = train.target.to_numpy()
    x = base_frame(train.drop(columns='target'), extractor)
    xt = base_frame(test, extractor)
    folds = fold_assignment(train.seller_id)
    cols = feature_list(groups)
    oof, pred, per_fold, fold_sec = np.zeros(len(train)), np.zeros(len(test)), [], []
    for k in range(5):
        t0 = time.perf_counter()
        fit, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        assert set(train.seller_id.iloc[fit]).isdisjoint(train.seller_id.iloc[val])
        xf, xv, xtt = fit_transform_fold(x.iloc[fit], [x.iloc[fit], x.iloc[val], xt], groups)
        model = make_model(k + 1).fit(xf[cols], y[fit], categorical_feature=CATS)
        oof[val] = model.predict_proba(xv[cols])[:, 1]
        pred += model.predict_proba(xtt[cols])[:, 1] / 5
        per_fold.append(float(roc_auc_score(y[val], oof[val])))
        fold_sec.append(time.perf_counter() - t0)
    report = {
        'name': name, 'extractor': extractor, 'groups': list(groups), 'features': cols,
        'n_features': len(cols), 'oof_roc_auc': float(roc_auc_score(y, oof)),
        'oof_average_precision': float(average_precision_score(y, oof)),
        'fold_roc_auc': per_fold, 'mean_fold_roc_auc': float(np.mean(per_fold)),
        'std_fold_roc_auc': float(np.std(per_fold)), 'fold_seconds': fold_sec,
        'elapsed_seconds': time.perf_counter() - started, 'seed': SEED, 'n_trees': N_TREES,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                     'lightgbm': lgb.__version__},
        'data_sha256': ({f: hashlib.sha256((Path(data_dir) / f).read_bytes()).hexdigest()
                         for f in ['train.csv', 'test.csv', 'sample_submission.csv']}
                        if data_dir else None),
        'fold_assignment_sha256': hashlib.sha256(
            pd.DataFrame({'id': train.id, 'fold': folds + 1}).to_csv(index=False, lineterminator='\n')
            .encode('utf-8')).hexdigest(),
    }
    pd.DataFrame({'id': train.id, 'seller_id': train.seller_id, 'target': y,
                  'prediction': oof, 'fold': folds + 1}).to_csv(out_dir / f'oof_{name}.csv', index=False)
    pd.DataFrame({'id': test.id, 'target': pred}).to_csv(out_dir / f'submission_{name}.csv', index=False)
    (out_dir / f'cv_{name}.json').write_text(json.dumps(report, indent=2))
    return report


def seller_bootstrap_diff(oof_a, oof_b, n_boot=1000, seed=SEED):
    """95% interval of AUC(b) - AUC(a), resampling whole sellers (the grouping unit)."""
    a = oof_a.set_index('id')
    b = oof_b.set_index('id').loc[a.index]
    sellers = a.seller_id.unique()
    idx = a.groupby('seller_id').indices
    rng = np.random.default_rng(seed)
    y, pa, pb = a.target.to_numpy(), a.prediction.to_numpy(), b.prediction.to_numpy()
    diffs = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[s] for s in rng.choice(sellers, len(sellers))])
        diffs.append(roc_auc_score(y[rows], pb[rows]) - roc_auc_score(y[rows], pa[rows]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


CONFIGS = {
    # shared reference configuration for both comparisons
    'ref': ('main_items', ('base', 'quantity', 'phrase', 'anomaly')),
    # comparison 1: only the quantity extractor changes
    'c1_first_pcs': ('first_pcs', ('base', 'quantity', 'phrase', 'anomaly')),
    # comparison 2: only the price/weight anomaly group is removed
    'c2_no_anomaly': ('main_items', ('base', 'quantity', 'phrase')),
}


FINAL = 'ref'   # configuration used for the Kaggle submission and the error analysis


def resolve_data_dir():
    for p in [Path('data/public'), Path('data'), *sorted(Path('/kaggle/input').rglob('train.csv'))]:
        d = p if p.is_dir() else p.parent
        if (d / 'train.csv').exists() and (d / 'sample_submission.csv').exists():
            return d
    raise FileNotFoundError('competition data not found; pass --data-dir')


def make_submission(data_dir, out_dir, config=FINAL):
    """Seller-disjoint CV of the chosen configuration + test predictions (mean of 5 fold models)."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    sample = pd.read_csv(data_dir / 'sample_submission.csv')
    assert train.target.isin([0, 1]).all()
    assert set(train.seller_id).isdisjoint(test.seller_id)
    assert train.id.is_unique and test.id.is_unique and set(train.id).isdisjoint(test.id)
    assert sample.id.tolist() == test.id.tolist()
    extractor, groups = CONFIGS[config]
    report = run_cv(train, test, extractor, groups, config, out_dir, data_dir)
    pred = pd.read_csv(out_dir / f'submission_{config}.csv')
    assert pred.id.tolist() == sample.id.tolist()
    assert np.isfinite(pred.target).all() and pred.target.between(0, 1).all()
    sample['target'] = pred.target.to_numpy()
    sample.to_csv(out_dir / 'submission.csv', index=False)
    return report


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out', default='results')
    ap.add_argument('--config', default=FINAL, choices=list(CONFIGS))
    a = ap.parse_args()
    r = make_submission(a.data_dir or resolve_data_dir(), a.out, a.config)
    print(json.dumps({k: r[k] for k in ['name', 'oof_roc_auc', 'oof_average_precision',
                                        'fold_roc_auc', 'elapsed_seconds', 'versions',
                                        'fold_assignment_sha256']}, indent=2))
