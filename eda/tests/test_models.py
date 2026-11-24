"""Behaviour tests for the model code that ships inside the registered models (linear, tree, ensemble, champion).

Synthetic data only, so they run in CI without the feature parquet or the MLflow store:
    pytest eda/tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import champion_model as cm  # noqa: E402
import ensemble_models as em  # noqa: E402
import linear_models as lm  # noqa: E402
import tree_models as tm  # noqa: E402

N = 240
FEATURES = ['fc_wind_share_of_demand', 'fc_share_estimate', 'ndf_mean_mw', 'wx_temperature_mean_c', 'noise_a', 'noise_b']


@pytest.fixture(scope='module')
def data():
    """A share driven mostly by forecast wind, with gaps, in date order like the feature store."""
    rng = np.random.default_rng(0)
    index = pd.date_range('2024-01-01', periods=N, freq='D', name='target_date')
    wind = rng.uniform(5, 70, N)
    X = pd.DataFrame({'fc_wind_share_of_demand': wind,
                      'fc_share_estimate': wind * 0.9 + 5 + rng.normal(0, 2, N),
                      'ndf_mean_mw': rng.normal(28000, 3000, N),
                      'wx_temperature_mean_c': rng.normal(10, 5, N),
                      'noise_a': rng.normal(size=N),
                      'noise_b': rng.normal(size=N)}, index=index)
    y = pd.Series(np.clip(0.8 * wind + 8 - 0.0002 * (X['ndf_mean_mw'] - 28000) + rng.normal(0, 3, N), 0, 100), index=index)
    X.iloc[rng.choice(N, 15, replace=False), 3] = np.nan  # missing weather on some days
    return X, y


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


# --- Linear ------------------------------------------------------------------------------------------------------

def test_logit_round_trip_stays_inside_bounds():
    y = np.array([0.0, 0.5, 30.0, 99.5, 100.0])
    back = lm.from_logit(lm.to_logit(y))
    assert np.all((back >= lm.LOGIT_FLOOR - 1e-9) & (back <= lm.LOGIT_CEILING + 1e-9))
    assert back[2] == pytest.approx(30.0)


@pytest.mark.parametrize('name', ['ols_gold8', 'ols_all', 'ridge', 'ridge_loo', 'lasso', 'elastic_net', 'elastic_net_logit',
                                  'huber', 'bayesian_ridge', 'ridge_gap_to_estimate', 'quantile'])
def test_every_linear_model_fits_and_beats_the_mean(data, name):
    X, y = data
    est, cols, description = lm.models(FEATURES, FEATURES[:3], FEATURES[:4])[name]
    train, test = slice(0, 180), slice(180, N)
    est.fit(X.iloc[train][cols], y.iloc[train])
    pred = est.predict(X.iloc[test][cols])
    assert description and len(pred) == N - 180 and np.isfinite(pred).all()
    assert mae(y.iloc[test], pred) < mae(y.iloc[test], np.full(N - 180, y.iloc[train].mean()))
    if name not in ('quantile',):
        assert np.isfinite(lm.coefficients(est)).all()
    assert isinstance(lm.chosen_hyperparameters(est), dict)


def test_complete_rows_only_skips_days_with_gaps(data):
    X, y = data
    est = lm.CompleteRowsOnly(lm.pipeline(lm.LinearRegression())).fit(X, y)
    assert est.estimator_[-1].n_features_in_ == X.shape[1]
    assert np.isfinite(est.predict(X)).all()  # prediction still fills the gaps


def test_gap_to_estimate_adds_the_physical_estimate_back(data):
    X, y = data
    est = lm.GapToEstimate(lm.pipeline(lm.LinearRegression())).fit(X, y)
    gap = est.regressor_.predict(X)
    np.testing.assert_allclose(est.predict(X), X['fc_share_estimate'].to_numpy() + gap)
    missing = X.assign(fc_share_estimate=np.nan)
    np.testing.assert_allclose(est.predict(missing), est.fallback_ + est.regressor_.predict(missing))


def test_linear_intervals_are_ordered(data):
    X, y = data
    for est in (lm.pipeline(lm.BayesianRidge()), lm.QuantileInterval(alpha=0.01)):
        est.fit(X, y)
        low, high = lm.interval(est, X)
        assert np.all(low <= high)
        covered = np.mean((y >= low) & (y <= high))
        assert 0.6 < covered < 0.98  # roughly 80%, in sample


# --- Trees -------------------------------------------------------------------------------------------------------

def test_typed_restores_the_grid_types():
    assert tm.typed('lightgbm', {'num_leaves': 8.0, 'learning_rate': 0.03, 'other': 1.5}) == {
        'num_leaves': 8, 'learning_rate': 0.03, 'other': 1.5}
    assert isinstance(tm.typed('lightgbm', {'n_estimators': 200.0})['n_estimators'], int)


@pytest.mark.parametrize('name', list(tm.GRIDS))
def test_every_tree_model_fits_with_an_empty_column(data, name):
    X, y = data
    X = X.assign(empty_source=np.nan)  # a source that started after these days
    params = {k: v[0] for k, v in tm.GRIDS[name].items()}
    est = tm.make(name, params).fit(X.iloc[:180], y.iloc[:180])
    pred = est.predict(X.iloc[180:])
    assert np.isfinite(pred).all()
    assert mae(y.iloc[180:], pred) < mae(y.iloc[180:], np.full(N - 180, y.iloc[:180].mean()))
    importance = tm.importance(est, X.iloc[180:])
    assert importance is None or 'empty_source' not in importance.index
    bounds = tm.interval(est, X.iloc[180:])
    if bounds is not None:
        assert np.all(bounds[0] <= bounds[1])


def test_lightgbm_contributions_add_up_to_the_prediction(data):
    X, y = data
    est = tm.make('lightgbm', {'n_estimators': 50, 'num_leaves': 8}, interval=False).fit(X, y)
    contrib = est.contributions(X)
    np.testing.assert_allclose(contrib.sum(axis=1), est.predict(X), atol=1e-6)


def test_monotonic_model_never_falls_as_forecast_wind_rises(data):
    X, y = data
    est = tm.make('lightgbm_monotonic', {'n_estimators': 100, 'num_leaves': 8}, interval=False).fit(X, y)
    grid = np.linspace(5, 70, 30)
    for _, row in X.sample(20, random_state=1).iterrows():
        sweep = pd.DataFrame([row] * len(grid)).assign(fc_wind_share_of_demand=grid)
        assert np.all(np.diff(est.predict(sweep)) >= -1e-9)


# --- Ensembles ---------------------------------------------------------------------------------------------------

def test_nnls_weights_recover_a_known_mix():
    rng = np.random.default_rng(1)
    members = rng.normal(40, 10, size=(300, 3))
    weights = em.nnls_weights(members, members @ np.array([0.7, 0.3, 0.0]))
    np.testing.assert_allclose(weights, [0.7, 0.3, 0.0], atol=1e-8)


def test_nnls_weights_fall_back_to_equal():
    weights = em.nnls_weights(np.ones((10, 4)), -np.ones(10))
    np.testing.assert_allclose(weights, 0.25)


def test_blend_is_the_weighted_average_of_its_members(data):
    X, y = data
    members = [('ridge', lm.pipeline(lm.BayesianRidge()), FEATURES),
               ('lgbm', tm.make('lightgbm', {'n_estimators': 50}, interval=False), FEATURES[:4])]
    blend = em.Blend(members, [0.25, 0.75]).fit(X, y)
    parts = blend.member_predictions(X)
    np.testing.assert_allclose(blend.predict(X), 0.25 * parts['ridge'] + 0.75 * parts['lgbm'])


# --- Champion ----------------------------------------------------------------------------------------------------

class Fixed:
    def __init__(self, values):
        self.values = values

    def predict(self, X):
        return np.asarray(self.values, dtype=float)


def test_champion_interval_is_the_point_plus_minus_the_half_width_clipped():
    X = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=pd.Index(['d1', 'd2', 'd3']))
    out = cm.ChampionModel(Fixed([50.0, 2.0, 103.0]), 5.0, {}).predict(None, X)
    assert list(out.columns) == ['predicted_share', 'lower_80', 'upper_80'] and list(out.index) == list(X.index)
    assert out.loc['d1'].tolist() == [50.0, 45.0, 55.0]
    assert out.loc['d2'].tolist() == [2.0, 0.0, 7.0]
    assert out.loc['d3'].tolist() == [100.0, 98.0, 100.0]


def test_champion_survives_an_mlflow_save_and_load(data, tmp_path):
    mlflow = pytest.importorskip('mlflow')
    X, y = data
    blend = em.Blend([('ridge', lm.pipeline(lm.BayesianRidge()), FEATURES),
                      ('lgbm', tm.make('lightgbm', {'n_estimators': 50}, interval=False), FEATURES)], [0.5, 0.5]).fit(X, y)
    eda = Path(__file__).resolve().parents[1]
    path = tmp_path / 'champion'
    mlflow.pyfunc.save_model(str(path), python_model=cm.ChampionModel(blend, 5.363, {'coverage': 0.8}),
                             code_paths=[str(eda / f) for f in ('champion_model.py', 'ensemble_models.py', 'linear_models.py',
                                                                'tree_models.py')],
                             input_example=X.head(3))
    loaded = mlflow.pyfunc.load_model(str(path))
    out = loaded.predict(X.tail(10))
    np.testing.assert_allclose(out['predicted_share'], np.clip(blend.predict(X.tail(10)), 0, 100))
    np.testing.assert_allclose(out['upper_80'] - out['predicted_share'], np.minimum(5.363, 100 - out['predicted_share']))
