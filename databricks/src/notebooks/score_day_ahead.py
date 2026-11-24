# Databricks notebook source
# MAGIC %md
# MAGIC # Day-ahead scoring: predict tomorrow's renewable share with the champion model
# MAGIC
# MAGIC Runs after the feature store, after the 18:00 UTC cutoff, when tomorrow's forecasts exist. It loads
# MAGIC `pulse.ml.renewable_share_day_ahead@champion` and scores every target day **after the last day with a known
# MAGIC outcome** that this model version hasn't scored. Past days are never re-predicted: a prediction made after the
# MAGIC fact would flatter the monitoring. (An old day whose target is missing because its data is incomplete is not
# MAGIC a future day, so it is skipped too.)
# MAGIC
# MAGIC **Readiness.** A day needs its day-ahead wind and demand forecasts to be scored at all. A day whose day-ahead
# MAGIC weather is missing is still scored, but flagged `weather_missing` in `data_quality`, and its interval is widened
# MAGIC (the error roughly doubles without weather). Monitoring tracks flagged days separately.
# MAGIC
# MAGIC | Table | Written |
# MAGIC |---|---|
# MAGIC | `ml.predictions` | The prediction and its 80% interval, with the model version (merged on day, model and version) |
# MAGIC | `ml.features_served` | Appended: the exact feature values the model received, and the feature table's Delta version |
# MAGIC | `ml.explanations` | Each feature's contribution. The champion is 0.5 × Bayesian ridge + 0.5 × LightGBM, so its prediction is exactly 0.5 × (ridge intercept + each feature's linear term) + 0.5 × (LightGBM base value + each feature's SHAP value) |

# COMMAND ----------

# The model's own library versions: it was pickled with them, so it must be loaded with them
import shutil

import mlflow

dbutils.widgets.text('catalog', 'pulse')  # noqa: F821
dbutils.widgets.text('model_alias', 'champion')  # noqa: F821
mlflow.set_registry_uri('databricks-uc')
_uri = f"models:/{dbutils.widgets.get('catalog')}.ml.renewable_share_day_ahead@{dbutils.widgets.get('model_alias')}"  # noqa: F821
shutil.copy(mlflow.pyfunc.get_model_dependencies(_uri), '/tmp/pulse_model_requirements.txt')

# COMMAND ----------

# MAGIC %pip install -r /tmp/pulse_model_requirements.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import json
import os
import sys

sys.path.append(os.path.abspath('..'))  # src/, for the pulse package

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from pulse.monitoring import weather_quality

# Without the day-ahead weather the error roughly doubles. Such days are still scored, flagged weather_missing, and
# their 80% interval is widened by this factor: the ratio of the 80th percentiles of |error| without and with
# weather on the 2026 days (eda/validate_champion.py, weather_missing_interval; 5.20 / 2.72)
WEATHER_MISSING_WIDENING = 1.913

CATALOG = dbutils.widgets.get('catalog')  # noqa: F821
ALIAS = dbutils.widgets.get('model_alias')  # noqa: F821
MODEL = f'{CATALOG}.ml.renewable_share_day_ahead'
FEATURES_TABLE = f'{CATALOG}.gold.feature_store_day_ahead'
PREDICTIONS, SERVED, EXPLANATIONS = (f'{CATALOG}.ml.{t}' for t in ('predictions', 'features_served', 'explanations'))
sql = spark.sql  # noqa: F821
spark.conf.set('spark.sql.session.timeZone', 'UTC')  # noqa: F821

mlflow.set_registry_uri('databricks-uc')
version = MlflowClient(registry_uri='databricks-uc').get_model_version_by_alias(MODEL, ALIAS).version
model = mlflow.pyfunc.load_model(f'models:/{MODEL}/{version}')
inputs = model.metadata.get_input_schema().input_names()
print(f'{MODEL} version {version} (@{ALIAS}), {len(inputs)} input features')

# COMMAND ----------

# MAGIC %md ## Which days to score

# COMMAND ----------

table_version = sql(f'DESCRIBE HISTORY {FEATURES_TABLE} LIMIT 1').first()['version']
already = sql(f"SELECT target_date FROM {PREDICTIONS} WHERE model_name = '{MODEL}' AND model_version = '{version}'")
last_known = spark.table(FEATURES_TABLE).where(F.col('target_renewable_share').isNotNull()).agg(F.max('target_date')).first()[0]  # noqa: F821
todo = (spark.table(FEATURES_TABLE).where(F.col('target_date') > F.lit(last_known))  # noqa: F821
        .join(already, 'target_date', 'left_anti'))
rows = todo.select('target_date', 'feature_set_version', *inputs).toPandas().set_index('target_date').sort_index()
# A day-ahead prediction needs the day-ahead forecasts; without them the row isn't ready yet
ready = rows[rows[['fc_wind_share_of_demand', 'ndf_mean_mw']].notna().all(axis=1)]
print(f'last day with a known outcome: {last_known}; {len(rows)} days after it to score; '
      f'{len(ready)} have their forecasts: {list(ready.index)}')

# COMMAND ----------

# MAGIC %md ## Predict, and explain each prediction

# COMMAND ----------

now = pd.Timestamp.now(tz='UTC').tz_localize(None)
if len(ready):
    X = ready[inputs].astype(float)
    out = model.predict(X)
    # Readiness: flag days with missing weather and widen their interval rather than hold them back
    quality = weather_quality(X)
    half_width = model.unwrap_python_model().half_width
    widen = quality == 'weather_missing'
    wide = half_width * WEATHER_MISSING_WIDENING
    out.loc[widen, 'lower_80'] = (out.loc[widen, 'predicted_share'] - wide).clip(0, 100)
    out.loc[widen, 'upper_80'] = (out.loc[widen, 'predicted_share'] + wide).clip(0, 100)
    out['data_quality'] = quality
    print(f'data quality: {quality.value_counts().to_dict()}'
          + (f'; interval widened to +/-{wide:.2f} on {int(widen.sum())} day(s)' if widen.any() else ''))

    # Exact decomposition of the ensemble: 0.5 x linear terms + 0.5 x LightGBM SHAP values
    blend = model.unwrap_python_model().point_model
    members = {name: (est, cols) for name, est, cols in blend.fitted_}
    weights = dict(zip([name for name, _, _ in blend.fitted_], blend.weights))
    ridge, ridge_cols = members['bayesian_ridge']
    transformed = ridge[:-1].transform(X[ridge_cols])
    linear = pd.DataFrame(transformed * ridge[-1].coef_, index=X.index, columns=ridge[:-1].get_feature_names_out())
    lgbm, lgbm_cols = members['lightgbm']
    shap = lgbm.contributions(X[lgbm_cols])
    contributions = (weights['bayesian_ridge'] * linear.reindex(columns=inputs, fill_value=0.0)
                     + weights['lightgbm'] * shap.drop(columns='base_value').reindex(columns=inputs, fill_value=0.0))
    base = weights['bayesian_ridge'] * ridge[-1].intercept_ + weights['lightgbm'] * shap['base_value']
    gap = (base + contributions.sum(axis=1) - out['predicted_share']).abs().max()
    print(f'explanations reproduce the predictions to within {gap:.2e} points')

    predictions = out.assign(model_name=MODEL, model_version=str(version), feature_set_version=ready['feature_set_version'],
                             predicted_at=now).rename_axis('target_date').reset_index()
    explanations = (contributions.stack().rename('contribution').rename_axis(['target_date', 'feature']).reset_index()
                    .assign(feature_value=lambda d: [X.loc[t, f] for t, f in zip(d['target_date'], d['feature'])],
                            base_value=lambda d: d['target_date'].map(base), model_name=MODEL, model_version=str(version),
                            method='0.5 x Bayesian ridge linear terms + 0.5 x LightGBM SHAP', explained_at=now))
    served = pd.DataFrame({'target_date': X.index, 'model_name': MODEL, 'model_version': str(version),
                           'feature_set_version': ready['feature_set_version'].to_numpy(), 'feature_table_version': int(table_version),
                           # JSON text, turned into MAP<STRING, DOUBLE> by the INSERT below (a dict column would arrive as a struct)
                           'features_json': [json.dumps({k: float(v) for k, v in row.items() if pd.notna(v)}) for _, row in X.iterrows()],
                           'served_at': now})
    print(predictions[['target_date', 'predicted_share', 'lower_80', 'upper_80', 'data_quality']].round(2).to_string(index=False))

# COMMAND ----------

# MAGIC %md ## Write

# COMMAND ----------

def merge(frame, table, keys):
    """Insert new rows and replace rows with the same key (rerunning the notebook writes nothing twice)."""
    spark.createDataFrame(frame).createOrReplaceTempView('incoming')  # noqa: F821
    columns = list(frame.columns)
    on = ' AND '.join(f't.{k} = s.{k}' for k in keys)
    sql(f"""MERGE INTO {table} AS t USING incoming AS s ON {on}
            WHEN MATCHED THEN UPDATE SET {', '.join(f't.{c} = s.{c}' for c in columns)}
            WHEN NOT MATCHED THEN INSERT ({', '.join(columns)}) VALUES ({', '.join(f's.{c}' for c in columns)})""")


def count(table):
    return sql(f"SELECT count(*) FROM {table} WHERE model_name = '{MODEL}' AND model_version = '{version}'").first()[0]


if len(ready):
    for frame in (predictions, explanations, served):
        frame['target_date'] = pd.to_datetime(frame['target_date']).dt.date
    before = {t: count(t) for t in (PREDICTIONS, EXPLANATIONS, SERVED)}
    merge(predictions, PREDICTIONS, ['target_date', 'model_name', 'model_version'])
    merge(explanations[['target_date', 'model_name', 'model_version', 'feature', 'feature_value', 'contribution', 'base_value', 'method',
                        'explained_at']], EXPLANATIONS, ['target_date', 'model_name', 'model_version', 'feature'])
    spark.createDataFrame(served).createOrReplaceTempView('served_rows')  # noqa: F821
    sql(f"""INSERT INTO {SERVED} (target_date, model_name, model_version, feature_set_version, feature_table_version, features, served_at)
            SELECT target_date, model_name, model_version, feature_set_version, feature_table_version,
                   from_json(features_json, 'MAP<STRING, DOUBLE>'), served_at FROM served_rows""")
    after = {t: count(t) for t in (PREDICTIONS, EXPLANATIONS, SERVED)}
    expected = {PREDICTIONS: len(predictions), EXPLANATIONS: len(explanations), SERVED: len(served)}
    # Fail loudly if a write didn't land
    for table, n in expected.items():
        assert after[table] - before[table] == n, f'{table}: expected {n} new rows, found {after[table] - before[table]}'
    print(f'wrote {len(predictions)} predictions, {len(explanations):,} explanation rows, {len(served)} served-feature rows')
else:
    print('nothing to score: no day with forecasts and an unknown outcome is waiting')
