# Databricks notebook source
# MAGIC %md
# MAGIC # Model monitoring: is the day-ahead champion still doing what it was validated to do?
# MAGIC
# MAGIC Runs after scoring. Four checks, each graded `ok`, `warn` or `alert` (rules and thresholds in `pulse/monitoring.py`):
# MAGIC
# MAGIC | Check | Compares | Against |
# MAGIC |---|---|---|
# MAGIC | **accuracy** | the last 30 days with a known outcome: MAE, bias, 80% interval coverage (Kupiec test) | the validated MAE of 3.30 and 80% coverage; overall and for each `data_quality` |
# MAGIC | **feature_drift** | every input the model received over the last 30 scored days (`ml.features_served`), except calendar and holiday features: PSI and KS | the same time of year (±15 days) in the year before the model version was registered; PSI thresholds allow for the small sample |
# MAGIC | **prediction_drift** | the last 30 predictions | the model's own predictions over those training days |
# MAGIC | **freshness** | per source, the share of its features present at the 18:00 UTC cutoff on the latest scored day | the share usually present: below 90% of it warns, below 50% alerts |
# MAGIC
# MAGIC Results go to `ml.monitoring`, one row per check, subject and metric, replaced if rerun the same day. **This
# MAGIC notebook only records: it never fails the job.** Retraining is a decision for a person looking at the alerts.

# COMMAND ----------

# The model's own library versions, as in scoring: prediction drift needs the model to predict its training year
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

import os
import sys

sys.path.append(os.path.abspath('..'))  # src/, for the pulse package

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from pulse import monitoring as mon

CATALOG = dbutils.widgets.get('catalog')  # noqa: F821
ALIAS = dbutils.widgets.get('model_alias')  # noqa: F821
MODEL = f'{CATALOG}.ml.renewable_share_day_ahead'
FEATURES_TABLE = f'{CATALOG}.gold.feature_store_day_ahead'
PREDICTIONS, SERVED, CATALOGUE, MONITORING = (f'{CATALOG}.ml.{t}' for t in ('predictions', 'features_served', 'feature_catalogue',
                                                                             'monitoring'))
TARGET = 'target_renewable_share'
sql = spark.sql  # noqa: F821
spark.conf.set('spark.sql.session.timeZone', 'UTC')  # noqa: F821

mlflow.set_registry_uri('databricks-uc')
details = MlflowClient(registry_uri='databricks-uc').get_model_version_by_alias(MODEL, ALIAS)
version = details.version
registered = pd.Timestamp(details.creation_timestamp, unit='ms').normalize()
model = mlflow.pyfunc.load_model(f'models:/{MODEL}/{version}')
inputs = model.metadata.get_input_schema().input_names()
print(f'{MODEL} version {version} (@{ALIAS}), registered {registered:%Y-%m-%d}')

# COMMAND ----------

# MAGIC %md ## Load: the training year, what the model was served, and the outcomes

# COMMAND ----------

store = spark.table(FEATURES_TABLE)  # noqa: F821
reference = (store.where((F.col('target_date') < F.lit(registered.date()))
                         & (F.col('target_date') >= F.lit((registered - pd.Timedelta(days=365)).date()))
                         & F.col(TARGET).isNotNull())
             .select('target_date', *inputs).toPandas().set_index('target_date').sort_index().astype(float))

served_rows = (spark.table(SERVED).where((F.col('model_name') == MODEL) & (F.col('model_version') == str(version)))  # noqa: F821
               .select('target_date', 'features', 'served_at').toPandas()
               # a day served twice (a rerun) counts once, as last served
               .sort_values('served_at').drop_duplicates('target_date', keep='last').set_index('target_date').sort_index())
served = pd.DataFrame(list(served_rows['features']), index=served_rows.index).reindex(columns=inputs).astype(float)
recent_served = served.tail(mon.WINDOW_DAYS)

predictions = (spark.table(PREDICTIONS).where((F.col('model_name') == MODEL) & (F.col('model_version') == str(version)))  # noqa: F821
               .join(store.select('target_date', F.col(TARGET).alias('actual')), 'target_date', 'left')
               .select('target_date', 'predicted_share', 'lower_80', 'upper_80', 'data_quality', 'actual')
               .toPandas().set_index('target_date').sort_index())
predictions['data_quality'] = predictions['data_quality'].fillna('not_recorded')
print(f'reference: {len(reference)} days ({reference.index.min()} to {reference.index.max()}); served: {len(served)} days; '
      f'predictions: {len(predictions)}, {int(predictions["actual"].notna().sum())} with a known outcome')

# COMMAND ----------

# MAGIC %md ## Check

# COMMAND ----------

rows = []

# Accuracy: overall, and separately for each data-quality flag (weather_missing days are expected to be worse)
known = predictions.dropna(subset=['actual']).tail(mon.WINDOW_DAYS)
rows += mon.accuracy(known, 'all')
for quality, group in known.groupby('data_quality'):
    rows += mon.accuracy(group, quality)

# Drift: against the same time of year in the training year, so the seasons aren't mistaken for drift
season = mon.seasonal_reference(reference, recent_served.index)
rows += mon.feature_drift(season, recent_served, inputs)

recent_predictions = predictions['predicted_share'].tail(mon.WINDOW_DAYS)
season_for_predictions = mon.seasonal_reference(reference, recent_predictions.index)
rows += mon.prediction_drift(model.predict(season_for_predictions)['predicted_share'] if len(season_for_predictions) else [],
                             recent_predictions)
print(f'drift reference: {len(season)} training days within {mon.SEASON_MARGIN_DAYS} days of the year of the recent ones')

catalogue = spark.table(CATALOGUE).select('feature', 'source_table').toPandas()  # noqa: F821
rows += mon.freshness(recent_served, dict(zip(catalogue['feature'], catalogue['source_table'])))

results = pd.DataFrame(rows, columns=mon.ROW_COLUMNS)
for c in ('window_start', 'window_end'):
    results[c] = pd.to_datetime(results[c]).dt.date
results.insert(0, 'model_version', str(version))
results.insert(0, 'model_name', MODEL)
results.insert(0, 'run_date', pd.Timestamp.now(tz='UTC').date())
results['monitored_at'] = pd.Timestamp.now(tz='UTC').tz_localize(None)

summary = mon.summary(rows)
for check, counts in summary.items():
    print(f'{check:17s} worst: {counts["worst"]:17s} {({k: v for k, v in counts.items() if k != "worst"})}')
flagged = results[results['status'].isin(['warn', 'alert'])].sort_values(['status', 'check_name'])
if flagged.empty:  # display() can't build a table from an empty frame
    print('no warnings or alerts')
else:
    display(flagged[['check_name', 'subject', 'metric', 'value', 'warn_at', 'alert_at', 'status', 'n', 'details']])  # noqa: F821

# COMMAND ----------

# MAGIC %md ## Write

# COMMAND ----------

keys = ['run_date', 'model_name', 'model_version', 'check_name', 'subject', 'metric']
columns = list(results.columns)
# NaN must arrive as NULL (Spark would otherwise store a NaN double)
spark.createDataFrame(results.astype(object).where(results.notna(), None)).createOrReplaceTempView('incoming')  # noqa: F821
sql(f"""MERGE INTO {MONITORING} AS t USING incoming AS s ON {' AND '.join(f't.{k} = s.{k}' for k in keys)}
        WHEN MATCHED THEN UPDATE SET {', '.join(f't.{c} = s.{c}' for c in columns)}
        WHEN NOT MATCHED THEN INSERT ({', '.join(columns)}) VALUES ({', '.join(f's.{c}' for c in columns)})""")
written = sql(f"SELECT count(*) FROM {MONITORING} WHERE run_date = current_date() AND model_version = '{version}'").first()[0]
print(f'{len(results)} monitoring rows for today; {written} in {MONITORING}')
# Records only: alerts are for a person to review, and never fail the job
