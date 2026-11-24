"""Linear models for the day-ahead renewable share, as scikit-learn estimators.

Each model is a complete pipeline (fill gaps, scale, fit), so the object MLflow saves is the whole thing and
predicts straight from the feature table. Hyperparameters are chosen inside each fit by time-ordered
cross-validation, so refitting on new data re-tunes them without peeking at later days.

MLflow reloads custom classes from this file (it is logged with each model as code).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNetCV, HuberRegressor, LassoCV, LinearRegression, QuantileRegressor, RidgeCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The share is a percentage; a logit needs it strictly inside (0, 100)
LOGIT_FLOOR, LOGIT_CEILING = 0.5, 99.5
RIDGE_ALPHAS = np.logspace(-2, 4, 30)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
HUBER_GRID = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0], 'epsilon': [1.35, 2.0]}


def time_cv():
    """Five folds, each validating on days after the ones it trained on."""
    return TimeSeriesSplit(n_splits=5)


def pipeline(estimator, scale=True):
    """Fill missing values with the training median, standardise, then fit the estimator."""
    steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))
    return Pipeline([*steps, ('model', estimator)])


def to_logit(y):
    p = np.clip(np.asarray(y, dtype=float), LOGIT_FLOOR, LOGIT_CEILING) / 100
    return np.log(p / (1 - p))


def from_logit(z):
    return 100 / (1 + np.exp(-np.asarray(z, dtype=float)))


class CompleteRowsOnly(BaseEstimator, RegressorMixin):
    """Fit only on days where every feature is present, as gold's is_trainable does; prediction still fills gaps.
    Used for the baseline so it matches the baseline in eda/model.py."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        keep = X.notna().all(axis=1).to_numpy()
        self.estimator_ = clone(self.estimator).fit(X[keep], np.asarray(y, dtype=float)[keep])
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class GapToEstimate(BaseEstimator, RegressorMixin):
    """Predict the gap between the target and a physical estimate of it, then add the estimate back.

    The estimate (fc_share_estimate) is built from forecasts with no fitting; the model only learns its
    corrections: curtailment, imports, forecast bias. Days without an estimate fall back to the training median.
    """

    def __init__(self, regressor, base='fc_share_estimate'):
        self.regressor = regressor
        self.base = base

    def _base(self, X):
        return X[self.base].fillna(self.fallback_).to_numpy(dtype=float)

    def fit(self, X, y):
        self.fallback_ = float(np.nanmedian(y))
        self.regressor_ = clone(self.regressor).fit(X, np.asarray(y, dtype=float) - self._base(X))
        return self

    def predict(self, X):
        return self._base(X) + self.regressor_.predict(X)


class QuantileInterval(BaseEstimator, RegressorMixin):
    """Linear quantile regression at the 10th, 50th and 90th percentiles: the median is the prediction, the
    other two an 80% interval fitted directly rather than assumed from a normal distribution."""

    def __init__(self, alpha=0.01, quantiles=(0.1, 0.5, 0.9)):
        self.alpha = alpha
        self.quantiles = quantiles

    def fit(self, X, y):
        self.models_ = {q: pipeline(QuantileRegressor(quantile=q, alpha=self.alpha, solver='highs')).fit(X, y)
                        for q in self.quantiles}
        return self

    def predict(self, X):
        return self.models_[0.5].predict(X)

    def predict_interval(self, X):
        low, high = self.models_[min(self.quantiles)].predict(X), self.models_[max(self.quantiles)].predict(X)
        return np.minimum(low, high), np.maximum(low, high)


def models(features, gold_features, selected_features):
    """name -> (estimator, feature list, description). Rebuilt fresh for every fit."""
    elastic = lambda: ElasticNetCV(l1_ratio=L1_RATIOS, alphas=40, cv=time_cv(), max_iter=50000)  # noqa: E731
    return {
        'ols_gold8': (CompleteRowsOnly(pipeline(LinearRegression())), gold_features,
                      'Baseline: ordinary least squares on the eight features gold has today, fitted on complete days'),
        'ols_all': (pipeline(LinearRegression()), features,
                    'Ordinary least squares on all 131 features, no regularisation: shows what collinearity costs'),
        'ridge': (pipeline(RidgeCV(alphas=RIDGE_ALPHAS, cv=time_cv())), features,
                  'Ridge (L2), penalty tuned on time-ordered folds: shrinks correlated features towards sharing the weight'),
        'ridge_loo': (pipeline(RidgeCV(alphas=RIDGE_ALPHAS)), features,
                      'Ridge (L2), penalty tuned by efficient leave-one-out, as in eda/model.py'),
        'lasso': (pipeline(LassoCV(alphas=40, cv=time_cv(), max_iter=50000)), features,
                  'Lasso (L1): sets weak features to exactly zero'),
        'elastic_net': (pipeline(elastic()), features,
                        'Elastic net: L1 and L2 together, the mix chosen by cross-validation'),
        'elastic_net_logit': (TransformedTargetRegressor(regressor=pipeline(elastic()), func=to_logit, inverse_func=from_logit,
                                                         check_inverse=False), features,
                              'Elastic net on the logit of the share: predictions stay inside 0-100%'),
        'huber': (pipeline(GridSearchCV(HuberRegressor(max_iter=5000), HUBER_GRID, cv=time_cv(),
                                        scoring='neg_mean_absolute_error')), features,
                  'Huber regression: squared loss for small errors, absolute loss for large ones, so outlier days count less'),
        'bayesian_ridge': (pipeline(BayesianRidge()), features,
                           'Bayesian ridge: the penalty is learned from the data, and every prediction comes with a standard deviation'),
        'ridge_gap_to_estimate': (GapToEstimate(pipeline(RidgeCV(alphas=RIDGE_ALPHAS, cv=time_cv()))), features,
                                  'Hybrid: ridge predicts the correction to the physical estimate fc_share_estimate'),
        'quantile': (QuantileInterval(alpha=0.01), selected_features,
                     'Quantile regression on the 30 selected features: median prediction and a fitted 10th-90th percentile interval'),
    }


# --- Reading a fitted model back ----------------------------------------------------------------------------------

def _last_linear(est):
    """The innermost fitted linear model and the pipeline that feeds it."""
    if isinstance(est, TransformedTargetRegressor):
        return _last_linear(est.regressor_)
    if isinstance(est, CompleteRowsOnly):
        return _last_linear(est.estimator_)
    if isinstance(est, GapToEstimate):
        return _last_linear(est.regressor_)
    if isinstance(est, QuantileInterval):
        return _last_linear(est.models_[0.5])
    if isinstance(est, Pipeline):
        inner = est.steps[-1][1]
        inner = inner.best_estimator_ if isinstance(inner, GridSearchCV) else inner
        return est, inner
    raise TypeError(type(est))


def coefficients(est):
    """Standardised coefficients of the fitted model, by feature (the scale step makes them comparable)."""
    pipe, linear = _last_linear(est)
    names = pipe[:-1].get_feature_names_out()
    return pd.Series(np.ravel(linear.coef_), index=names)


def chosen_hyperparameters(est):
    """The hyperparameters cross-validation picked in this fit."""
    pipe, linear = _last_linear(est)
    inner = pipe.steps[-1][1]
    out = {}
    if isinstance(inner, GridSearchCV):
        out.update({f'chosen_{k}': v for k, v in inner.best_params_.items()})
    for attribute in ('alpha_', 'l1_ratio_', 'lambda_'):
        if hasattr(linear, attribute):
            out[f'chosen_{attribute.rstrip("_")}'] = float(getattr(linear, attribute))
    if isinstance(est, QuantileInterval):
        out['chosen_alpha'] = est.alpha
    return out


def interval(est, X):
    """A model's own 80% interval, where it has one: fitted quantiles, or the Bayesian standard deviation."""
    if isinstance(est, QuantileInterval):
        return est.predict_interval(X)
    pipe, linear = _last_linear(est) if isinstance(est, Pipeline) else (None, None)
    if isinstance(linear, BayesianRidge):
        mean, std = est.predict(X, return_std=True)
        return mean - 1.2816 * std, mean + 1.2816 * std
    return None
