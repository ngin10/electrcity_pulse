# Databricks notebook source
# MAGIC %md
# MAGIC # Feature store: build and version the day-ahead features
# MAGIC
# MAGIC Runs in the daily job after silver. Builds every day-ahead feature (`pulse/feature_store.py`, the same code the
# MAGIC EDA runs) and keeps them as versioned Delta tables:
# MAGIC
# MAGIC | Table | What it holds |
# MAGIC |---|---|
# MAGIC | `gold.feature_store_day_ahead` | One row per target day: 131 features and the target. Primary key `target_date`. |
# MAGIC | `ml.feature_catalogue` | One row per feature: definition, source table, when it is known, the version that added it. |
# MAGIC | `ml.features_served` | Append-only: the exact feature row behind each prediction. Filled by the scoring job. |
# MAGIC | `ml.predictions` | Each prediction and its 80% interval, with the model version. Filled by the scoring job. |
# MAGIC | `ml.explanations` | Each prediction's feature contributions (SHAP). Filled by the scoring job. |
# MAGIC
# MAGIC **Versioning.** Every write is a Delta version, kept for a year. A rerun only rewrites days whose features changed
# MAGIC (compared by a hash of the values), so the history shows real changes, not reruns. Each row also records the
# MAGIC `feature_set_version` (the feature logic) and `code_version` (the Git commit) that produced it.
# MAGIC
# MAGIC **Rollback.** See the last cell.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath('..'))  # src/, for the pulse package

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T

from pulse import feature_store as fs
from pulse.quality import sql_str
from pulse.settings import CATALOG_DEFAULT, schema

# Job parameters (set by resources/job.yml); the defaults suit a run by hand on Free Edition
dbutils.widgets.text('catalog', CATALOG_DEFAULT)  # noqa: F821
dbutils.widgets.text('code_version', '')  # noqa: F821
CATALOG = dbutils.widgets.get('catalog')  # noqa: F821
CODE_VERSION = dbutils.widgets.get('code_version') or 'unknown'  # noqa: F821

SILVER, GOLD, ML = (schema(CATALOG, layer) for layer in ('silver', 'gold', 'ml'))
FEATURES = f'{GOLD}.feature_store_day_ahead'
CATALOGUE = f'{ML}.feature_catalogue'
SERVED, PREDICTIONS, EXPLANATIONS = f'{ML}.features_served', f'{ML}.predictions', f'{ML}.explanations'
# Keep every version for a year: VACUUM removes old files after 7 days by default, which would end time travel
RETENTION = ("TBLPROPERTIES ('delta.logRetentionDuration' = 'interval 365 days', "
             "'delta.deletedFileRetentionDuration' = 'interval 365 days')")
TARGET = 'target_renewable_share'
META_COLUMNS = ('feature_hash', 'feature_set_version', 'code_version', 'computed_at')

# The notebook's Spark session, provided by Databricks
sql, table = spark.sql, spark.table  # noqa: F821
# Timestamps reach pandas as naive UTC, as feature_store expects
spark.conf.set('spark.sql.session.timeZone', 'UTC')  # noqa: F821

# COMMAND ----------

# MAGIC %md ## Build the features from silver

# COMMAND ----------

raw = {table: spark.read.table(f'{SILVER}.{table}').select(*columns).toPandas()  # noqa: F821
       for table, columns in fs.SILVER_COLUMNS.items()}
frame, meta = fs.build(fs.prepare(raw))
feature_names = [m['feature'] for m in meta]

rows = frame.rename(columns={'target': TARGET}).rename_axis('target_date').reset_index()
rows['target_date'] = rows['target_date'].dt.date
rows['feature_hash'] = fs.feature_hash(rows, feature_names)
rows['feature_set_version'] = fs.FEATURE_SET_VERSION
rows['code_version'] = CODE_VERSION
rows['computed_at'] = pd.Timestamp.now(tz='UTC').tz_localize(None)

row_schema = T.StructType(
    [T.StructField('target_date', T.DateType(), False)]
    + [T.StructField(name, T.DoubleType()) for name in [*feature_names, TARGET]]
    + [T.StructField(name, T.StringType()) for name in META_COLUMNS[:3]]
    + [T.StructField('computed_at', T.TimestampType())])
built = spark.createDataFrame(rows[row_schema.fieldNames()], schema=row_schema)  # noqa: F821
# pandas marks a missing value as NaN; in the table it must be NULL, which aggregates and models treat as missing
built = built.select(*[F.when(F.isnan(c), None).otherwise(F.col(c)).alias(c) if c in (*feature_names, TARGET) else F.col(c)
                       for c in row_schema.fieldNames()])
print(f'{len(rows):,} target days, {len(feature_names)} features, '
      f'{rows["target_date"].min()} to {rows["target_date"].max()}')

# COMMAND ----------

# MAGIC %md ## Create the tables once

# COMMAND ----------

definitions = {m['feature']: m['definition'] for m in meta}
sql(f"CREATE SCHEMA IF NOT EXISTS {ML} COMMENT 'Feature catalogue, served features, predictions and explanations'")


def feature_column(name):
    return f'{name} DOUBLE COMMENT {sql_str(definitions[name])}'


sql(f"""
CREATE TABLE IF NOT EXISTS {FEATURES} (
  target_date DATE NOT NULL COMMENT 'The day being predicted (UK date). Every feature uses only what was known at 18:00 UTC the day before',
  {', '.join(feature_column(n) for n in feature_names)},
  {TARGET} DOUBLE COMMENT 'The label: renewable share of GB generation on the target day, %. NULL until the day is complete',
  feature_hash STRING COMMENT 'Fingerprint of the feature values: a change means the row was recomputed with different values',
  feature_set_version STRING COMMENT 'Version of the feature logic (pulse/feature_store.py FEATURE_SET_VERSION)',
  code_version STRING COMMENT 'Git commit of the code that wrote the row',
  computed_at TIMESTAMP COMMENT 'When the row was last written (UTC)',
  CONSTRAINT feature_store_day_ahead_pk PRIMARY KEY (target_date)
) COMMENT 'Day-ahead features for the renewable-share model, one row per target day. Versioned: query an earlier state with VERSION AS OF.'
{RETENTION}""")

sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOGUE} (
  feature STRING NOT NULL COMMENT 'Column name in the feature table',
  source_table STRING COMMENT 'The silver table it is built from (or calendar)',
  definition STRING COMMENT 'What it measures',
  known STRING COMMENT 'When its value is known relative to the target day',
  available_from STRING COMMENT 'When a source row counts as available: the leakage rule',
  data_type STRING,
  feature_set_version STRING COMMENT 'The feature-logic version that last wrote this entry',
  first_version STRING COMMENT 'The feature-logic version that added the feature',
  in_current_version BOOLEAN COMMENT 'FALSE once a later version drops the feature (its column stays in the feature table)',
  updated_at TIMESTAMP,
  CONSTRAINT feature_catalogue_pk PRIMARY KEY (feature)
) COMMENT 'One row per feature: definition, source and leakage rule' {RETENTION}""")

sql(f"""
CREATE TABLE IF NOT EXISTS {SERVED} (
  target_date DATE NOT NULL,
  model_name STRING NOT NULL COMMENT 'Registered model name',
  model_version STRING NOT NULL COMMENT 'Registered model version',
  feature_set_version STRING,
  feature_table_version BIGINT COMMENT 'Delta version of the feature table the row was read from',
  features MAP<STRING, DOUBLE> COMMENT 'The exact feature values the model received',
  served_at TIMESTAMP
) COMMENT 'Append-only: the feature row behind every prediction, as the model saw it. Never updated.' {RETENTION}""")

sql(f"""
CREATE TABLE IF NOT EXISTS {PREDICTIONS} (
  target_date DATE NOT NULL,
  model_name STRING NOT NULL,
  model_version STRING NOT NULL,
  predicted_share DOUBLE COMMENT 'Predicted renewable share, %',
  lower_80 DOUBLE COMMENT 'Lower end of the 80% prediction interval, %',
  upper_80 DOUBLE COMMENT 'Upper end of the 80% prediction interval, %',
  feature_set_version STRING,
  predicted_at TIMESTAMP,
  CONSTRAINT predictions_pk PRIMARY KEY (target_date, model_name, model_version)
) COMMENT 'One row per prediction' {RETENTION}""")

sql(f"""
CREATE TABLE IF NOT EXISTS {EXPLANATIONS} (
  target_date DATE NOT NULL,
  model_name STRING NOT NULL,
  model_version STRING NOT NULL,
  feature STRING NOT NULL,
  feature_value DOUBLE,
  contribution DOUBLE COMMENT 'How much this feature moved the prediction from the base value, points of share',
  base_value DOUBLE COMMENT 'The average prediction of the model, before any feature is considered',
  method STRING COMMENT 'How contributions were computed, e.g. shap',
  explained_at TIMESTAMP,
  CONSTRAINT explanations_pk PRIMARY KEY (target_date, model_name, model_version, feature)
) COMMENT 'Why each prediction came out as it did: one row per prediction and feature' {RETENTION}""")

# COMMAND ----------

# MAGIC %md ## Keep the feature table in step with the feature logic
# MAGIC
# MAGIC A new feature becomes a new column. A dropped feature keeps its column (old rows still hold it), and the
# MAGIC catalogue marks it `in_current_version = FALSE`. Column comments follow the definitions.

# COMMAND ----------

existing = {f.name: f.metadata.get('comment') for f in table(FEATURES).schema.fields}
new_columns = [n for n in feature_names if n not in existing]
if new_columns:
    sql(f"ALTER TABLE {FEATURES} ADD COLUMNS ({', '.join(feature_column(n) for n in new_columns)})")
for name in feature_names:
    if name in existing and existing[name] != definitions[name]:
        sql(f'ALTER TABLE {FEATURES} ALTER COLUMN {name} COMMENT {sql_str(definitions[name])}')
print(f'{len(new_columns)} new columns')

# COMMAND ----------

# MAGIC %md ## Write only what changed
# MAGIC
# MAGIC A day is written when it is new, when its feature values changed (the hash differs), when its target arrived or
# MAGIC changed, or when a new feature-logic version produced it. Unchanged days are left alone, so each Delta version
# MAGIC is a real change.

# COMMAND ----------

current = table(FEATURES).select('target_date', F.col('feature_hash').alias('old_hash'),
                                       F.col(TARGET).alias('old_target'), F.col('feature_set_version').alias('old_version'))
changed = (built.join(current, 'target_date', 'left')
           .where(F.col('old_hash').isNull() | (F.col('feature_hash') != F.col('old_hash'))
                  | ~F.col(TARGET).eqNullSafe(F.col('old_target')) | (F.col('feature_set_version') != F.col('old_version')))
           .select(*row_schema.fieldNames()))
n_changed = changed.count()

if n_changed:
    changed.createOrReplaceTempView('feature_rows')
    columns = row_schema.fieldNames()
    sql(f"""
    MERGE INTO {FEATURES} AS t USING feature_rows AS s ON t.target_date = s.target_date
    WHEN MATCHED THEN UPDATE SET {', '.join(f't.{c} = s.{c}' for c in columns)}
    WHEN NOT MATCHED THEN INSERT ({', '.join(columns)}) VALUES ({', '.join(f's.{c}' for c in columns)})""")
print(f'{n_changed:,} days written')

# COMMAND ----------

# MAGIC %md ## The catalogue

# COMMAND ----------

cat = fs.catalogue(meta)
cat['first_version'] = fs.FEATURE_SET_VERSION
cat['in_current_version'] = True
cat['updated_at'] = pd.Timestamp.now(tz='UTC').tz_localize(None)
spark.createDataFrame(cat).createOrReplaceTempView('catalogue_rows')  # noqa: F821
described = ('source_table', 'definition', 'known', 'available_from', 'data_type')
sql(f"""
MERGE INTO {CATALOGUE} AS t USING catalogue_rows AS s ON t.feature = s.feature
WHEN MATCHED AND (NOT t.in_current_version OR {' OR '.join(f'NOT (t.{c} <=> s.{c})' for c in described)})
  THEN UPDATE SET {', '.join(f't.{c} = s.{c}' for c in described)}, t.feature_set_version = s.feature_set_version,
                  t.in_current_version = TRUE, t.updated_at = s.updated_at
WHEN NOT MATCHED THEN INSERT *
WHEN NOT MATCHED BY SOURCE AND t.in_current_version THEN UPDATE SET t.in_current_version = FALSE""")

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

history = sql(f'DESCRIBE HISTORY {FEATURES} LIMIT 5').select('version', 'timestamp', 'operation', 'operationMetrics')
display(history)  # noqa: F821
latest = table(FEATURES).agg(
    F.count('*').alias('days'),
    F.max(F.when(F.col('fc_wind_share_of_demand').isNotNull(), F.col('target_date'))).alias('last_day_with_forecasts'),
    F.max(F.when(F.col(TARGET).isNotNull(), F.col('target_date'))).alias('last_day_with_target')).first()
print(f'{latest.days:,} days stored; forecasts through {latest.last_day_with_forecasts}; target through {latest.last_day_with_target}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading an earlier version, and rolling back
# MAGIC
# MAGIC ```sql
# MAGIC -- Every version, with what changed
# MAGIC DESCRIBE HISTORY pulse.gold.feature_store_day_ahead;
# MAGIC
# MAGIC -- The table as it was at a version or a time
# MAGIC SELECT * FROM pulse.gold.feature_store_day_ahead VERSION AS OF 12;
# MAGIC SELECT * FROM pulse.gold.feature_store_day_ahead TIMESTAMP AS OF '2026-11-01';
# MAGIC
# MAGIC -- What changed for one day between two versions
# MAGIC SELECT 'now' AS state, * FROM pulse.gold.feature_store_day_ahead WHERE target_date = '2026-10-15'
# MAGIC UNION ALL
# MAGIC SELECT 'v12', * FROM pulse.gold.feature_store_day_ahead VERSION AS OF 12 WHERE target_date = '2026-10-15';
# MAGIC
# MAGIC -- Roll the whole table back (this is itself a new version, so it can be undone)
# MAGIC RESTORE TABLE pulse.gold.feature_store_day_ahead TO VERSION AS OF 12;
# MAGIC ```
# MAGIC
# MAGIC To roll back the feature **logic**, revert the commit in Git, set `FEATURE_SET_VERSION` to a new value and rerun:
# MAGIC every day is rewritten with the reverted logic, and the old rows stay readable in the history.
