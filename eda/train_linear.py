"""Train and compare linear models for the day-ahead renewable share, tracked in MLflow.

The models are in linear_models.py; the walk-forward, metrics, charts and logging in experiment.py. Each model
tunes its penalty inside every monthly refit by time-ordered cross-validation (ridge_loo by leave-one-out).

    python eda/train_linear.py
    mlflow ui --backend-store-uri sqlite:///eda/out/mlflow/mlflow.db     # then open http://127.0.0.1:5000
"""
import json

import experiment as ex
import linear_models as lm

LINEAR = ex.Family(
    name='linear',
    catalog=lambda: lm.models(ex.FEATURES, ex.GOLD, ex.SELECTED),
    explain=lambda est, X: lm.coefficients(est),
    explain_title='Largest standardised coefficients (final model)',
    explain_label='effect of one standard deviation',
    trusted=['numpy.dtype', 'sklearn.model_selection._split.TimeSeriesSplit', 'linear_models.CompleteRowsOnly',
             'linear_models.GapToEstimate', 'linear_models.QuantileInterval', 'linear_models.to_logit', 'linear_models.from_logit'],
    code=ex.EDA / 'linear_models.py',
    chosen=lm.chosen_hyperparameters,
    interval=lm.interval,
    extra_params=lambda name: {
        'imputation': 'median', 'scaling': 'standard',
        'tuning': 'leave-one-out' if name == 'ridge_loo' else 'TimeSeriesSplit(5) inside every monthly refit',
        'ridge_alpha_grid': 'logspace(-2, 4, 30)' if 'ridge' in name else None,
        'l1_ratio_grid': str(lm.L1_RATIOS) if 'elastic' in name else None,
        'huber_grid': json.dumps(lm.HUBER_GRID) if name == 'huber' else None},
)
LINEAR.references = {'baseline': (LINEAR, 'ols_gold8')}


if __name__ == '__main__':
    board = ex.run_suite(LINEAR)
    ex.client.set_registered_model_alias(ex.REGISTERED_MODEL, 'baseline', str(int(board.loc['ols_gold8', 'version'])))
