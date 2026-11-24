"""Package the chosen champion with its calibrated interval, register it, and point the champion alias at it.

Selection rule (decided before looking at which model would win it):
  1. Among all registered models, the lowest walk-forward MAE.
  2. Significantly more accurate than the best model of every single family (Diebold-Mariano, 5%).
  3. Its 2026 holdout MAE agrees with its walk-forward MAE (within the walk-forward's 95% interval).
  -> blend_bayes_lgbm (version 17): MAE 3.303, better than the best linear (p < 0.001) and the best tree (p = 0.003);
     holdout 3.334, inside 3.12-3.54.

The interval comes from calibrate_intervals.py (eda/out/mlflow/cache/interval_calibration.json).

    python eda/package_champion.py                       # local MLflow registry
    python eda/package_champion.py --target databricks   # Unity Catalog registry, as pulse.ml.renewable_share_day_ahead
    python eda/package_champion.py --target both
"""
import argparse
import json

import cloudpickle
import lightgbm
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import experiment as ex
from champion_model import ChampionModel

POINT_MODEL = ('blend_bayes_lgbm', 17)
SELECTION_RULE = ('lowest walk-forward MAE; significantly better (Diebold-Mariano, 5%) than the best linear and best tree '
                  'models; 2026 holdout MAE inside the walk-forward 95% interval')
UC_MODEL = 'pulse.ml.renewable_share_day_ahead'
CODE = [ex.EDA / f for f in ('linear_models.py', 'tree_models.py', 'ensemble_models.py', 'champion_model.py')]
# Pinned where the pickled objects depend on the exact version; the rest only need a compatible one
REQUIREMENTS = [f'scikit-learn=={sklearn.__version__}', f'lightgbm=={lightgbm.__version__}', f'cloudpickle=={cloudpickle.__version__}',
                f'numpy>={np.__version__.split(".")[0]}.0', 'scipy', 'pandas>=2.2', 'mlflow']


def package(point_model, calibration, point_run, registry_name, local_client):
    """Log the champion package in the current tracking server and register it. Returns the new version."""
    X = ex.data[ex.FEATURES]
    champion = ChampionModel(point_model, calibration['q'], calibration)
    output = champion.predict(None, X.head(50))
    calib_run = local_client.search_runs([mlflow.get_experiment_by_name(ex.EXPERIMENT).experiment_id] if local_client else [],
                                         filter_string="tags.mlflow.runName = 'interval-calibration'", max_results=1) if local_client else []
    with mlflow.start_run(run_name='champion-package', description='The champion ensemble with its calibrated 80% interval') as run:
        mlflow.set_tags({'model_family': 'champion', 'stage': 'production-candidate', 'git_commit': ex.COMMIT,
                         'point_model': f'{POINT_MODEL[0]} (local registry version {POINT_MODEL[1]})'})
        mlflow.log_params({'point_model': POINT_MODEL[0], 'point_model_local_version': POINT_MODEL[1],
                           'point_model_run_id': point_run.info.run_id, 'members': 'bayesian_ridge, lightgbm', 'weights': '0.5, 0.5',
                           'selection_rule': SELECTION_RULE, 'interval_method': calibration['method'],
                           'interval_half_width': round(calibration['q'], 4), 'interval_calibrated_through': calibration['calibrated_through'],
                           'interval_window_days': calibration['window_days'], 'feature_set_version': 'v1', 'n_features': len(ex.FEATURES)})
        metrics = {k: v for k, v in point_run.data.metrics.items() if k.startswith(('wf_', 'holdout2026_', 'dm_', 'mae_gain', 'interval80'))}
        if calib_run:
            m = calib_run[0].data.metrics
            metrics.update({f'calibration_{k.removeprefix(calibration["method"] + "_")}': v for k, v in m.items()
                            if k.startswith(calibration['method'] + '_')})
        mlflow.log_metrics(metrics)
        info = mlflow.pyfunc.log_model(name='model', python_model=champion, code_paths=[str(c) for c in CODE],
                                       signature=infer_signature(X.head(50), output), input_example=X.dropna().tail(3),
                                       pip_requirements=REQUIREMENTS, registered_model_name=registry_name)
        return info.registered_model_version, run.info.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['local', 'databricks', 'both'], default='local', help='which registry to register in')
    args = parser.parse_args()

    local = MlflowClient()
    version = local.get_model_version(ex.REGISTERED_MODEL, str(POINT_MODEL[1]))
    assert version.tags.get('model') == POINT_MODEL[0], version.tags
    point_model = mlflow.sklearn.load_model(f'models:/{ex.REGISTERED_MODEL}/{POINT_MODEL[1]}')
    point_run = local.get_run(version.run_id)
    calibration = json.loads((ex.CACHE / 'interval_calibration.json').read_text())

    # Same predictions as the registered point model, or stop
    X = ex.data[ex.FEATURES].tail(30)
    assert np.allclose(point_model.predict(X), mlflow.pyfunc.load_model(f'models:/{ex.REGISTERED_MODEL}/{POINT_MODEL[1]}').predict(X))

    new_version = local.get_model_version_by_alias(ex.REGISTERED_MODEL, 'champion').version
    if args.target in ('local', 'both'):
        new_version, run_id = package(point_model, calibration, point_run, ex.REGISTERED_MODEL, local)
        local.set_registered_model_alias(ex.REGISTERED_MODEL, 'champion', new_version)
        for key, value in (('model', 'champion: blend_bayes_lgbm + calibrated interval'), ('family', 'champion'),
                           ('selection_rule', SELECTION_RULE), ('promoted', f'{pd.Timestamp.now():%Y-%m-%d}'),
                           ('wf_mae', f"{point_run.data.metrics['wf_mae']:.3f}")):
            local.set_model_version_tag(ex.REGISTERED_MODEL, new_version, key, value)
        print(f'local registry: {ex.REGISTERED_MODEL} version {new_version} is now @champion (run {run_id})')

    if args.target in ('databricks', 'both'):
        from databricks.sdk import WorkspaceClient
        user = WorkspaceClient().current_user.me().user_name
        mlflow.set_tracking_uri('databricks')
        mlflow.set_registry_uri('databricks-uc')
        mlflow.set_experiment(f'/Users/{user}/renewable-share-day-ahead')
        uc_version, uc_run = package(point_model, calibration, point_run, UC_MODEL, None)
        uc = MlflowClient(registry_uri='databricks-uc')
        uc.set_registered_model_alias(UC_MODEL, 'champion', uc_version)
        uc.set_model_version_tag(UC_MODEL, uc_version, 'selection_rule', SELECTION_RULE)
        uc.set_model_version_tag(UC_MODEL, uc_version, 'local_registry_version', str(new_version))
        print(f'Unity Catalog: {UC_MODEL} version {uc_version} is now @champion (run {uc_run})')


if __name__ == '__main__':
    main()
