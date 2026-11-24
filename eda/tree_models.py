"""Tree models for the day-ahead renewable share, as scikit-learn estimators.

Unlike the linear models, hyperparameters are tuned once, on the days before the walk-forward starts, by
time-ordered cross-validation over a small grid (tune()). The walk-forward then refits with those settings every
month. Tuning inside every refit, as the linear suite does, would cost a grid search per month for little gain.

MLflow reloads custom classes from this file (it is logged with each model as code).
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline

SEED = 7
LOCATIONS = ('dogger_bank', 'east_anglia_offshore', 'irish_sea', 'scottish_highlands', 'southern_england')
# Monotonic constraints: more forecast wind (or wind power) can only raise the predicted share; more residual
# demand (what gas and imports must cover) can only lower it. Everything else is unconstrained
INCREASING = {
    'fc_wind_share_of_demand', 'fc_wind_solar_share_of_demand', 'fc_share_estimate', 'fc_wind_mean_mw', 'fc_wind_onshore_mean_mw',
    'fc_wind_offshore_mean_mw', 'fc_wind_solar_mean_mw', 'fc_wind_calibrated_mw', 'fc_wind_min_mw', 'fc_wind_max_mw', 'wf_mean_mw',
    'wf_calibrated_mw', 'wf_min_mw', 'wf_max_mw', 'wx_power_mean', 'wx_power_offshore', 'wx_power_min_location', 'wx_power_min_hour',
    'wx_wind_100m_mean_kmh', 'wx_wind_10m_mean_kmh', 'wx_power_x_calibration', *(f'wx_power_{loc}' for loc in LOCATIONS)}
DECREASING = {'fc_residual_demand_mean_mw', 'fc_residual_demand_max_mw', 'fc_residual_demand_min_mw'}

GRIDS = {
    'random_forest': {'min_samples_leaf': [2, 5, 10], 'max_features': [0.2, 0.33, 0.5]},
    'extra_trees': {'min_samples_leaf': [2, 5, 10], 'max_features': [0.2, 0.33, 0.5]},
    'hist_gbm': {'learning_rate': [0.03, 0.06], 'max_depth': [2, 3, 5], 'max_iter': [200, 400], 'min_samples_leaf': [10, 20]},
    'lightgbm': {'num_leaves': [4, 8, 16], 'learning_rate': [0.03, 0.06], 'n_estimators': [200, 400], 'min_child_samples': [10, 20]},
    'lightgbm_monotonic': {'num_leaves': [4, 8, 16], 'learning_rate': [0.03, 0.06], 'n_estimators': [200, 400],
                           'min_child_samples': [10, 20]},
}
DESCRIPTIONS = {
    'random_forest': 'Random forest: many deep trees on bootstrap samples, averaged',
    'extra_trees': 'Extremely randomised trees: like a random forest, with random split points',
    'hist_gbm': 'Gradient boosting (scikit-learn histogram): shallow trees added one at a time, each fixing the last',
    'lightgbm': 'Gradient boosting (LightGBM), with its own 10th-90th percentile interval',
    'lightgbm_monotonic': 'LightGBM with monotonic constraints: more forecast wind can never lower the prediction, more residual demand '
                          'can never raise it',
}


def monotone_constraints(features):
    return [1 if f in INCREASING else -1 if f in DECREASING else 0 for f in features]


class DropEmptyColumns(BaseEstimator, TransformerMixin):
    """Drop columns with no values in the training rows (early refits predate some sources); trees can't bin them."""

    def fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if X[c].notna().any()]
        return self

    def transform(self, X):
        return X[self.columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


class Boosted(BaseEstimator, RegressorMixin):
    """A gradient-boosted model (LightGBM or scikit-learn's histogram booster) that handles missing values natively,
    can be monotonic, and fits two quantile models alongside for its own 80% interval."""

    def __init__(self, kind='lightgbm', params=None, monotonic=False, interval=True):
        self.kind = kind
        self.params = params
        self.monotonic = monotonic
        self.interval = interval

    def _make(self, quantile=None):
        params = dict(self.params or {})
        if self.kind == 'lightgbm':
            base = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=1)
            # LightGBM can't combine monotonic constraints with the quantile objective: only the point model is constrained
            if self.monotonic and quantile is None:
                base['monotone_constraints'] = monotone_constraints(self.columns_)
            if quantile is not None:
                base.update(objective='quantile', alpha=quantile)
            return LGBMRegressor(**base, **params)
        base = dict(l2_regularization=1.0, random_state=SEED)
        if self.monotonic:
            base['monotonic_cst'] = monotone_constraints(self.columns_)
        if quantile is not None:
            base.update(loss='quantile', quantile=quantile)
        return HistGradientBoostingRegressor(**base, **params)

    def fit(self, X, y):
        self.columns_ = [c for c in X.columns if X[c].notna().any()]
        X = X[self.columns_]
        self.model_ = self._make().fit(X, y)
        self.quantile_models_ = {q: self._make(q).fit(X, y) for q in (0.1, 0.9)} if self.interval else {}
        return self

    def predict(self, X):
        return self.model_.predict(X[self.columns_])

    def predict_interval(self, X):
        if not self.quantile_models_:
            return None
        low, high = (self.quantile_models_[q].predict(X[self.columns_]) for q in (0.1, 0.9))
        return np.minimum(low, high), np.maximum(low, high)

    def contributions(self, X):
        """Per-day, per-feature contributions to the prediction (SHAP values, computed by LightGBM itself), plus the
        base value in the last column. None for the scikit-learn booster."""
        if self.kind != 'lightgbm':
            return None
        values = self.model_.predict(X[self.columns_], pred_contrib=True)
        return pd.DataFrame(values, index=X.index, columns=[*self.columns_, 'base_value'])


def forest(kind, params=None):
    model = RandomForestRegressor if kind == 'random_forest' else ExtraTreesRegressor
    return Pipeline([('drop_empty', DropEmptyColumns()), ('impute', SimpleImputer(strategy='median')),
                     ('model', model(n_estimators=300, n_jobs=-1, random_state=SEED, **(params or {})))])


def typed(name, params):
    """Settings in their grid's own types: a results table (or JSON) turns the integer 5 into 5.0, which trees reject."""
    if not params:
        return params
    return {k: type(GRIDS[name][k][0])(v) if k in GRIDS.get(name, {}) else v for k, v in params.items()}


def make(name, params=None, interval=True):
    params = typed(name, params)
    if name in ('random_forest', 'extra_trees'):
        return forest(name, params)
    if name == 'hist_gbm':
        return Boosted('hist_gbm', params, interval=interval)
    return Boosted('lightgbm', params, monotonic=(name == 'lightgbm_monotonic'), interval=interval)


def tune(name, X, y, folds=5):
    """Grid search with time-ordered folds on the given (pre-walk-forward) days. Returns the best settings and the table."""
    rows = []
    for params in ParameterGrid(GRIDS[name]):
        maes = []
        for fit_idx, val_idx in TimeSeriesSplit(n_splits=folds).split(X):
            est = make(name, params, interval=False).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            maes.append(float(np.abs(y.iloc[val_idx] - est.predict(X.iloc[val_idx])).mean()))
        rows.append({**params, 'cv_mae': float(np.mean(maes)), 'cv_mae_std': float(np.std(maes))})
    table = pd.DataFrame(rows).sort_values('cv_mae').reset_index(drop=True)
    best = typed(name, {k: v for k, v in table.iloc[0].items() if k in GRIDS[name]})
    return best, table


def models(features, tuned):
    """name -> (estimator, feature list, description), built with the tuned settings."""
    return {name: (make(name, tuned.get(name)), features, DESCRIPTIONS[name]) for name in GRIDS}


# --- Reading a fitted model back ----------------------------------------------------------------------------------

def importance(est, X):
    """How much each feature matters to the fitted model: mean |SHAP| for LightGBM, impurity importance for forests,
    None for the scikit-learn booster (its permutation importance on 2026 is logged instead)."""
    if isinstance(est, Boosted):
        contrib = est.contributions(X)
        return None if contrib is None else contrib.drop(columns='base_value').abs().mean()
    model = est.named_steps['model']
    return pd.Series(model.feature_importances_, index=est[:-1].get_feature_names_out())


def interval(est, X):
    return est.predict_interval(X) if isinstance(est, Boosted) else None
