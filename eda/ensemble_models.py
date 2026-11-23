"""Ensembles for the day-ahead renewable share: weighted averages of linear and tree models.

Blend fits each member on its own features and averages their predictions with fixed weights. The weights are
decided outside (train_ensembles.py): equal for a plain blend, or learned by nnls_weights() from members'
out-of-sample predictions for a stacked one.

MLflow reloads custom classes from this file (it is logged with each model as code).
"""
import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone


def nnls_weights(predictions, actual):
    """Non-negative weights, summing to 1, that best reproduce the actual values from the members' predictions
    (least squares). Falls back to equal weights if every weight comes out zero."""
    predictions = np.asarray(predictions, dtype=float)
    weights, _ = nnls(predictions, np.asarray(actual, dtype=float))
    total = weights.sum()
    return weights / total if total > 0 else np.full(predictions.shape[1], 1 / predictions.shape[1])


class Blend(BaseEstimator, RegressorMixin):
    """A weighted average of models. members: [(name, estimator, feature list)]; weights: one per member."""

    def __init__(self, members, weights):
        self.members = members
        self.weights = weights

    def fit(self, X, y):
        self.fitted_ = [(name, clone(est).fit(X[cols], y), cols) for name, est, cols in self.members]
        return self

    def member_predictions(self, X):
        return {name: est.predict(X[cols]) for name, est, cols in self.fitted_}

    def predict(self, X):
        preds = self.member_predictions(X)
        return sum(w * preds[name] for w, (name, _, _) in zip(self.weights, self.fitted_))
