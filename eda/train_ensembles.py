"""Train and compare ensembles of the linear and tree models, tracked in MLflow.

An ensemble's walk-forward predictions are built from its members' cached walk-forward predictions, so nothing
is refitted month by month:
- a fixed blend averages its members with equal weights;
- a stacked ensemble learns non-negative weights each month from the members' earlier out-of-sample predictions
  only (at least 90 days of them; equal weights before that), so a month's weights never see that month.

The saved model fits every member on all days and applies the weights learned from all out-of-sample days (for the
2026 holdout, only those before 2026). Every run is compared with the baseline, the best linear and the best tree model.

    python eda/train_ensembles.py
"""
import numpy as np
import pandas as pd

import ensemble_models as em
import experiment as ex
from train_linear import LINEAR
from train_trees import TREES

# The members an ensemble may use: the strongest of each family
CANDIDATES = {'bayesian_ridge': LINEAR, 'ridge_loo': LINEAR, 'huber': LINEAR,
              'lightgbm': TREES, 'lightgbm_monotonic': TREES, 'hist_gbm': TREES, 'extra_trees': TREES}
MIN_HISTORY, RECENT_DAYS = 90, 180
ENSEMBLES = {
    'blend_bayes_lgbm': ('fixed', ['bayesian_ridge', 'lightgbm'],
                         'Equal average of Bayesian ridge (best linear) and LightGBM (best tree)'),
    'blend_bayes_lgbm_monotonic': ('fixed', ['bayesian_ridge', 'lightgbm_monotonic'],
                                   'Equal average of Bayesian ridge and monotonic LightGBM: the most defensible pair'),
    'blend_four': ('fixed', ['bayesian_ridge', 'huber', 'lightgbm', 'hist_gbm'],
                   'Equal average of the two best linear and the two best boosted models'),
    'stacked_all': ('stacked', list(CANDIDATES),
                    'Weights learned monthly by non-negative least squares on all earlier out-of-sample predictions'),
    'stacked_recent': ('stacked_recent', list(CANDIDATES),
                       f'Weights learned monthly on the last {RECENT_DAYS} days of out-of-sample predictions: adapts to drift'),
}

WALK = {name: ex.walk_forward(family, name)[0] for name, family in CANDIDATES.items()}
PRED = pd.DataFrame({name: wf['pred'] for name, wf in WALK.items()})
ACTUAL = WALK['bayesian_ridge']['actual']
assert all(wf.index.equals(PRED.index) for wf in WALK.values())


def weights_for(kind, members, before):
    """The members' weights for predicting from `before` on, using only out-of-sample predictions before it."""
    equal = np.full(len(members), 1 / len(members))
    if kind == 'fixed':
        return equal
    history = PRED.loc[PRED.index < before, members]
    if kind == 'stacked_recent':
        history = history.tail(RECENT_DAYS)
    if len(history) < MIN_HISTORY:
        return equal
    return em.nnls_weights(history.to_numpy(), ACTUAL.loc[history.index].to_numpy())


def ensemble_walk_forward(name):
    kind, members, _ = ENSEMBLES[name]
    parts, chosen = [], {}
    for month in sorted(set(PRED.index.to_period('M'))):
        here = PRED.index.to_period('M') == month
        weights = weights_for(kind, members, month.start_time)
        parts.append(pd.DataFrame({'actual': ACTUAL[here], 'pred': PRED.loc[here, members].to_numpy() @ weights,
                                   'own_low': np.nan, 'own_high': np.nan}, index=PRED.index[here]))
        if kind != 'fixed':
            chosen[month] = {f'chosen_w_{member}': float(w) for member, w in zip(members, weights)}
    wf = pd.concat(parts)
    ex.conformal(wf)
    return wf, pd.DataFrame.from_dict(chosen, orient='index')


def catalog(before=None):
    """Ensembles with their members unfitted and weights learned from out-of-sample predictions before `before`."""
    out = {}
    for name, (kind, members, description) in ENSEMBLES.items():
        member_list = [(m, *CANDIDATES[m].catalog()[m][:2]) for m in members]
        weights = weights_for(kind, members, before or pd.Timestamp.max)
        out[name] = (em.Blend(member_list, [float(w) for w in weights]), ex.FEATURES, description)
    return out


ENSEMBLE = ex.Family(
    name='ensemble',
    catalog=catalog,
    holdout_catalog=lambda: catalog(ex.HOLDOUT_START),
    walk_forward=ensemble_walk_forward,
    explain=lambda est, X: pd.Series(est.weights, index=[name for name, _, _ in est.members]),
    explain_title='Ensemble weights (final model)',
    explain_label='weight',
    trusted=sorted(set(LINEAR.trusted + TREES.trusted + ['ensemble_models.Blend'])),
    code=[ex.EDA / 'linear_models.py', ex.EDA / 'tree_models.py', ex.EDA / 'ensemble_models.py'],
    extra_params=lambda name: {'combination': ENSEMBLES[name][0], 'members': ', '.join(ENSEMBLES[name][1]),
                               'weights_learned_from': None if ENSEMBLES[name][0] == 'fixed'
                               else f'earlier out-of-sample predictions, at least {MIN_HISTORY} days'
                               + (f', last {RECENT_DAYS} days' if ENSEMBLES[name][0] == 'stacked_recent' else '')}
    if name in ENSEMBLES else {},
    references={'baseline': (LINEAR, 'ols_gold8'), 'best_linear': (LINEAR, 'bayesian_ridge'), 'best_tree': (TREES, 'lightgbm')},
)


if __name__ == '__main__':
    final = {name: catalog()[name][0].weights for name in ENSEMBLES}
    for name, weights in final.items():
        print(f'{name:28s} final weights', dict(zip(ENSEMBLES[name][1], np.round(weights, 3))))
    ex.run_suite(ENSEMBLE, extra_suite_params={'candidates': ', '.join(CANDIDATES)})
