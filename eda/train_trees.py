"""Train and compare tree models for the day-ahead renewable share, tracked in MLflow.

Each model is tuned once by time-ordered cross-validation on the days before the walk-forward starts (January
2025), then refitted monthly with those settings. Every run is compared with the baseline and with the best
linear model. LightGBM models also log SHAP explanations: which features moved each prediction, and how.

    python eda/train_trees.py
"""
import hashlib
import json

import matplotlib.pyplot as plt
import pandas as pd

import experiment as ex
import tree_models as tm
from train_linear import LINEAR

DEV = ex.data[ex.data.index < ex.WALK_START]


def tuned_settings():
    """Grid search per model on the days before the walk-forward, cached by grid and data."""
    digest = hashlib.sha256(f'{json.dumps(tm.GRIDS, sort_keys=True)}|{pd.util.hash_pandas_object(DEV).sum()}|{tm.SEED}'.encode()).hexdigest()[:16]
    path = ex.CACHE / f'tree-tuning-{digest}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        return saved['best'], {k: pd.DataFrame(v) for k, v in saved['tables'].items()}
    best, tables = {}, {}
    for name in tm.GRIDS:
        best[name], tables[name] = tm.tune(name, DEV[ex.FEATURES], DEV['target'])
        print(f'tuned {name:20s} {best[name]}  CV MAE {tables[name]["cv_mae"].iloc[0]:.3f}', flush=True)
    path.write_text(json.dumps({'best': best, 'tables': {k: v.to_dict('list') for k, v in tables.items()}}))
    return best, tables


BEST, TABLES = tuned_settings()
BEST = {name: tm.typed(name, params) for name, params in BEST.items()}


def shap_charts(name, est, X, folder):
    """For LightGBM: how the top features move predictions, and why the latest day's prediction came out as it did."""
    if not isinstance(est, tm.Boosted) or est.kind != 'lightgbm':
        return []
    contrib = est.contributions(X)
    features = contrib.drop(columns='base_value')
    top = features.abs().mean().sort_values(ascending=False).head(4).index
    paths = []
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, feature in zip(axes, top):
        ax.scatter(X[feature], features[feature], s=6, alpha=0.5, color=ex.BLUE)
        ax.axhline(0, color=ex.GREY, lw=0.8)
        ax.set(title=feature, xlabel='feature value', ylabel='contribution, points')
    fig.suptitle('How the top features move the prediction (SHAP dependence)', fontweight='bold')
    paths.append(ex.save(fig, folder, '14_shap_dependence.png'))

    day = X.index[-1]
    row = features.loc[day].sort_values(key=abs, ascending=False).head(12)[::-1]
    base = contrib.loc[day, 'base_value']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([f'{f} = {X.loc[day, f]:.3g}' for f in row.index], row.values, color=[ex.BLUE if v >= 0 else ex.ORANGE for v in row.values])
    ax.axvline(0, color=ex.GREY, lw=0.8)
    ax.set(title=f'Why the model predicted {base + features.loc[day].sum():.1f}% for {day:%d %b %Y}',
           xlabel=f'contribution, points (starting from the average prediction, {base:.1f}%)')
    paths.append(ex.save(fig, folder, '15_shap_latest_day.png'))

    recent = features.tail(60).stack().rename('contribution').reset_index()
    recent.columns = ['target_date', 'feature', 'contribution']
    recent['feature_value'] = [X.loc[d, f] for d, f in zip(recent['target_date'], recent['feature'])]
    recent['base_value'] = recent['target_date'].map(contrib['base_value'])
    # Logged with the run's tables: the same shape as ml.explanations in the feature store
    (folder / 'tables').mkdir(exist_ok=True)
    recent.to_csv(folder / 'tables' / 'shap_last_60_days.csv', index=False)
    return paths


TREES = ex.Family(
    name='tree',
    catalog=lambda: tm.models(ex.FEATURES, BEST),
    explain=tm.importance,
    explain_title='Feature importance (final model): mean |SHAP| for LightGBM, impurity for forests',
    explain_label='importance',
    # Library internals of the fitted trees, which skops only loads once listed
    trusted=['numpy.dtype', 'collections.OrderedDict', 'tree_models.Boosted', 'tree_models.DropEmptyColumns', 'sklearn.tree._tree.Tree',
             'sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor', 'lightgbm.basic.Booster', 'lightgbm.sklearn.LGBMRegressor'],
    code=ex.EDA / 'tree_models.py',
    interval=tm.interval,
    extra_params=lambda name: {'tuning': 'TimeSeriesSplit(5) grid search once, on the days before 2025',
                               'grid': json.dumps(tm.GRIDS[name]) if name in tm.GRIDS else None,
                               **({f'tuned_{k}': v for k, v in BEST[name].items()} if name in BEST else {}),
                               'monotonic_constraints': f'{sum(1 for c in tm.monotone_constraints(ex.FEATURES) if c)} features'
                               if name == 'lightgbm_monotonic' else None,
                               'missing_values': 'native' if name in ('hist_gbm', 'lightgbm', 'lightgbm_monotonic') else 'median imputation'},
    extra_charts=shap_charts,
    extra_tables=lambda name: {'tuning_cv_results.csv': TABLES[name]} if name in TABLES else {},
    permutation=True,
    references={'baseline': (LINEAR, 'ols_gold8'), 'best_linear': (LINEAR, 'bayesian_ridge')},
)


if __name__ == '__main__':
    ex.run_suite(TREES, extra_suite_params={'tuned_on': f'{DEV.index.min():%Y-%m-%d} to {DEV.index.max():%Y-%m-%d}'})
