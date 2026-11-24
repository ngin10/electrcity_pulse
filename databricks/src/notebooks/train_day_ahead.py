# Databricks notebook source
# MAGIC %md
# MAGIC # Monthly retraining: refresh the champion on the latest features, and register it as a challenger if it passes the gate
# MAGIC
# MAGIC 1. **Specification:** cloned from the registered `@champion` (same members, same settings, same model code), so
# MAGIC    the validated design is what gets retrained; only the data is new.
# MAGIC 2. **Walk-forward:** monthly refits from January 2025, each month predicted only by a model fitted on earlier days.
# MAGIC 3. **Interval:** recalibrated, the conformal 80th percentile of the walk-forward's absolute errors.
# MAGIC 4. **Final fit** on every day with a known outcome.
# MAGIC 5. **Gate** (`pulse/training.py`): out-of-sample MAE within the validated range, no recent degradation or bias,
# MAGIC    coverage consistent with 80%, predictions close to the champion's, valid output.
# MAGIC 6. **Register** in Unity Catalog with the alias `@challenger` if it passes. Promotion to `@champion` is a separate,
# MAGIC    manual step (`promote_model.py`), unless the job parameter `auto_promote` is `true`.
# MAGIC
# MAGIC Everything is logged to the MLflow experiment `renewable-share-day-ahead`, including a failed gate.

# COMMAND ----------

# The champion's own library versions: its specification is unpickled with them, and the challenger is saved with them
import shutil

import mlflow

dbutils.widgets.text('catalog', 'pulse')  # noqa: F821
dbutils.widgets.text('model_alias', 'champion')  # noqa: F821
dbutils.widgets.dropdown('auto_promote', 'false', ['false', 'true'])  # noqa: F821
mlflow.set_registry_uri('databricks-uc')
_uri = f"models:/{dbutils.widgets.get('catalog')}.ml.renewable_share_day_ahead@{dbutils.widgets.get('model_alias')}"  # noqa: F821
shutil.copy(mlflow.pyfunc.get_model_dependencies(_uri), '/tmp/pulse_model_requirements.txt')

# COMMAND ----------

# MAGIC %pip install -r /tmp/pulse_model_requirements.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import hashlib
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath('..'))  # src/, for the pulse package

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone

from pulse import training as tr

CATALOG = dbutils.widgets.get('catalog')  # noqa: F821
ALIAS = dbutils.widgets.get('model_alias')  # noqa: F821
AUTO_PROMOTE = dbutils.widgets.get('auto_promote') == 'true'  # noqa: F821
MODEL = f'{CATALOG}.ml.renewable_share_day_ahead'
FEATURES_TABLE = f'{CATALOG}.gold.feature_store_day_ahead'
TARGET = 'target_renewable_share'
sql = spark.sql  # noqa: F821
spark.conf.set('spark.sql.session.timeZone', 'UTC')  # noqa: F821

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient(registry_uri='databricks-uc')
champion_version = client.get_model_version_by_alias(MODEL, ALIAS).version
champion_uri = f'models:/{MODEL}/{champion_version}'
champion = mlflow.pyfunc.load_model(champion_uri)  # also puts the model's code folder on sys.path
inputs = champion.metadata.get_input_schema().input_names()

# The model code travels with the model: reuse the champion's copy, so the challenger is saved with the same code
model_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=champion_uri))
code_paths = sorted(str(p) for p in (model_dir / 'code').glob('*.py'))
requirements = '/tmp/pulse_model_requirements.txt'
from champion_model import ChampionModel  # noqa: E402  (from the champion's code folder)

spec = clone(champion.unwrap_python_model().point_model)
print(f'{MODEL} version {champion_version} (@{ALIAS}); specification: '
      f'{[name for name, _, _ in spec.members]} with weights {spec.weights}; code {[Path(p).name for p in code_paths]}')

# COMMAND ----------

# MAGIC %md ## Training data: every day with a known outcome, and the feature table version it came from

# COMMAND ----------

table_version = sql(f'DESCRIBE HISTORY {FEATURES_TABLE} LIMIT 1').first()['version']
frame = (spark.table(FEATURES_TABLE).select('target_date', 'feature_set_version', *inputs, TARGET)  # noqa: F821
         .toPandas().set_index('target_date').sort_index())
frame.index = pd.to_datetime(frame.index)
feature_set_versions = sorted(frame['feature_set_version'].dropna().unique())
known = frame[frame[TARGET].notna()]
X, y = known[inputs].astype(float), known[TARGET].astype(float)
latest = frame[inputs].astype(float).tail(tr.AGREEMENT_DAYS)
digest = hashlib.sha256(pd.util.hash_pandas_object(known[[*inputs, TARGET]], index=True).values.tobytes()).hexdigest()[:16]
print(f'feature table version {table_version}; {len(known)} days with an outcome, {X.index.min():%Y-%m-%d} to '
      f'{X.index.max():%Y-%m-%d}; feature set {feature_set_versions}; data digest {digest}')

# COMMAND ----------

# MAGIC %md ## Walk-forward, calibration, final fit and the gate

# COMMAND ----------

wf = tr.walk_forward(spec, X, y)
abs_err = (wf['actual'] - wf['pred']).abs()
half_width = tr.conformal_half_width(abs_err)
calibration = {'method': 'absolute_all', 'q': half_width, 'alpha': 1 - tr.COVERAGE, 'window_days': int(len(wf)),
               'calibrated_through': f'{wf.index.max():%Y-%m-%d}'}

final = clone(spec).fit(X, y)
challenger = ChampionModel(final, half_width, calibration)
challenger_latest = challenger.predict(None, latest)['predicted_share'].to_numpy()
champion_latest = champion.predict(latest)['predicted_share'].to_numpy()
passed, checks = tr.gate(wf, len(X), challenger_latest, champion_latest)

print(f'walk-forward: {len(wf)} days, MAE {abs_err.mean():.3f}; interval half-width {half_width:.3f} '
      f'(champion {champion.unwrap_python_model().half_width:.3f})')
for c in checks:
    print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['check']:17s} {c['value']:8.3f}  limit {c['limit']:6.2f}  {c['detail']}")
print('GATE PASSED' if passed else 'GATE FAILED: no challenger is registered')

# COMMAND ----------

# MAGIC %md ## Log, and register the challenger if it passed

# COMMAND ----------

user = sql('SELECT current_user()').first()[0]
mlflow.set_experiment(f'/Users/{user}/renewable-share-day-ahead')
new_version = None
with mlflow.start_run(run_name='monthly-retrain', description='Champion specification refitted on the latest feature store') as run:
    mlflow.set_tags({'model_family': 'champion', 'stage': 'retraining', 'gate': 'passed' if passed else 'failed',
                     'source_champion_version': champion_version, 'auto_promote': str(AUTO_PROMOTE).lower()})
    mlflow.log_params({'source_model': f'{MODEL}/{champion_version}', 'members': ', '.join(n for n, _, _ in spec.members),
                       'weights': ', '.join(str(w) for w in spec.weights), 'feature_table': FEATURES_TABLE,
                       'feature_table_version': table_version, 'feature_set_version': ','.join(feature_set_versions),
                       'n_features': len(inputs), 'training_days': len(X), 'trained_from': f'{X.index.min():%Y-%m-%d}',
                       'trained_through': f'{X.index.max():%Y-%m-%d}', 'data_digest': digest, 'walk_start': f'{tr.WALK_START:%Y-%m-%d}',
                       'interval_method': calibration['method'], 'interval_half_width': round(half_width, 4)})
    err = wf['actual'] - wf['pred']
    mlflow.log_metrics({'wf_mae': float(err.abs().mean()), 'wf_rmse': float(np.sqrt((err ** 2).mean())), 'wf_bias': float(err.mean()),
                        'wf_r2': float(1 - (err ** 2).sum() / ((wf['actual'] - wf['actual'].mean()) ** 2).sum()),
                        **{f"gate_{c['check']}": c['value'] for c in checks}, 'gate_passed': float(passed)})
    mlflow.log_dict({'passed': passed, 'checks': checks}, 'gate.json')
    mlflow.log_table(wf.rename_axis('target_date').reset_index().assign(target_date=lambda d: d['target_date'].dt.strftime('%Y-%m-%d')),
                     'walk_forward_predictions.json')
    if passed:
        info = mlflow.pyfunc.log_model(name='model', python_model=challenger, code_paths=code_paths, pip_requirements=requirements,
                                       signature=infer_signature(X.head(50), challenger.predict(None, X.head(50))),
                                       input_example=X.dropna().tail(3), registered_model_name=MODEL)
        new_version = info.registered_model_version

if new_version:
    client.set_registered_model_alias(MODEL, 'challenger', new_version)
    for key, value in (('gate', 'passed'), ('wf_mae', f"{err.abs().mean():.3f}"), ('trained_through', f'{X.index.max():%Y-%m-%d}'),
                       ('feature_table_version', str(table_version)), ('source_champion_version', champion_version),
                       ('training_run_id', run.info.run_id)):
        client.set_model_version_tag(MODEL, new_version, key, value)
    print(f'{MODEL} version {new_version} is now @challenger (run {run.info.run_id})')
    if AUTO_PROMOTE:
        client.set_registered_model_alias(MODEL, 'previous_champion', champion_version)
        client.set_registered_model_alias(MODEL, 'champion', new_version)
        client.set_model_version_tag(MODEL, new_version, 'promoted', f'{pd.Timestamp.now(tz="UTC"):%Y-%m-%d} automatically by {user}')
        print(f'auto_promote: version {new_version} is now @champion; version {champion_version} is @previous_champion')
    else:
        print('To promote it after review, run the pulse_promote_model job (or promote_model.py) with version '
              f'{new_version}.')
else:
    print(f'Nothing registered; the failed checks are in run {run.info.run_id} (gate.json).')
