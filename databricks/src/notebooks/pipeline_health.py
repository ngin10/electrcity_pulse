# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline health
# MAGIC
# MAGIC Runs after the pipelines in the daily job. Shows each layer's state and **fails the task** on a problem,
# MAGIC so the job turns red and notifies you:
# MAGIC - **bronze**: landing files not loaded, rows that didn't fit the schema, no new data for 2+ days;
# MAGIC - **silver**: rows rejected from the last 2 landing days, or recent days with missing periods;
# MAGIC - **gold**: no trainable feature row for a recent day.

# COMMAND ----------

import os
import sys
from datetime import date, timedelta

sys.path.append(os.path.abspath('..'))  # src/, for the shared settings

from pyspark.sql import functions as F

from pulse.settings import BRONZE_TABLES, CATALOG_DEFAULT, landing_volume, schema

# Job parameters (set by resources/job.yml); the defaults suit a run by hand on Free Edition
dbutils.widgets.text('catalog', CATALOG_DEFAULT)  # noqa: F821
dbutils.widgets.text('landing_path', '')  # noqa: F821
CATALOG = dbutils.widgets.get('catalog')  # noqa: F821
LANDING = dbutils.widgets.get('landing_path') or landing_volume(CATALOG)  # noqa: F821
BRONZE, SILVER, GOLD = (schema(CATALOG, layer) for layer in ('bronze', 'silver', 'gold'))
problems = []

# COMMAND ----------

# MAGIC %md ## Bronze: every landing file loaded, every row fits the schema, data still arriving

# COMMAND ----------


def landing_file_count(table):
    """.parquet files in one landing folder (in GCS, or in the volume copy)."""
    total = 0
    for partition in dbutils.fs.ls(f'{LANDING}/{table}'):  # noqa: F821 (provided by Databricks)
        if partition.name.startswith('ingest_date='):
            total += sum(1 for f in dbutils.fs.ls(partition.path) if f.name.endswith('.parquet'))  # noqa: F821
    return total


rows = []
for table in BRONZE_TABLES:
    df = spark.table(f'{BRONZE}.{table}')  # noqa: F821
    stats = df.agg(
        F.count('*').alias('rows'),
        F.countDistinct('source_file').alias('files_loaded'),
        F.max('ingest_date').alias('last_ingest_date'),
        F.max('loaded_at').alias('last_loaded_at'),
        F.sum(F.col('_rescued_data').isNotNull().cast('int')).alias('rescued_rows'),
    ).first().asDict()
    stats['table'] = table
    stats['files_in_landing'] = landing_file_count(table)
    rows.append(stats)

bronze = (spark.createDataFrame(rows)  # noqa: F821
          .withColumn('files_missing', F.col('files_in_landing') - F.col('files_loaded'))
          .withColumn('days_since_ingest', F.datediff(F.current_date(), 'last_ingest_date'))
          .select('table', 'rows', 'files_in_landing', 'files_loaded', 'files_missing', 'last_ingest_date',
                  'days_since_ingest', 'rescued_rows', 'last_loaded_at'))
display(bronze)  # noqa: F821

for r in bronze.where('files_missing > 0 OR rescued_rows > 0 OR days_since_ingest > 2').collect():
    problems.append(f'bronze.{r.table}: {r.files_missing} files not loaded, {r.rescued_rows} rescued rows, '
                    f'{r.days_since_ingest} days since the last landing')

# COMMAND ----------

# MAGIC %md ## Silver: new rejections, and gaps in recent days

# COMMAND ----------

recent = date.today() - timedelta(days=2)
rejected = (spark.table(f'{SILVER}.rejected_summary')  # noqa: F821
            .where(F.col('ingest_date') >= F.lit(recent))
            .orderBy(F.desc('count')))
display(rejected)  # noqa: F821
for r in rejected.collect():
    problems.append(f'silver.{r.source_table}: {r["count"]} rows rejected for {r.reason} (landed {r.ingest_date})')

# Days from a month ago to the day before yesterday: today and yesterday are still filling in.
# NESO publishes about three weeks behind, so its window is earlier
gaps = (spark.table(f'{SILVER}.completeness')  # noqa: F821
        .where(~F.col('is_complete'))
        .where(F.col('settlement_date') <= F.date_sub(F.current_date(),
                                                      F.when(F.col('source_table') == 'neso_demand', 25).otherwise(2)))
        .where(F.col('settlement_date') >= F.date_sub(F.current_date(), 30))
        .orderBy('source_table', 'settlement_date'))
display(gaps)  # noqa: F821
for r in gaps.collect():
    problems.append(f'silver.{r.source_table}: {r.settlement_date} has {r.periods_present} of '
                    f'{r.periods_expected} periods')

# COMMAND ----------

# MAGIC %md ## Gold: features ready for recent days

# COMMAND ----------

features = spark.table(f'{GOLD}.features_day_ahead')  # noqa: F821
summary = features.agg(F.max(F.when('is_trainable', F.col('target_date'))).alias('last_trainable_date'),
                       F.sum(F.col('is_trainable').cast('int')).alias('trainable_days'),
                       F.count('*').alias('days')).first()
display(spark.createDataFrame([summary.asDict()]))  # noqa: F821
if summary.last_trainable_date is None or (date.today() - summary.last_trainable_date).days > 3:
    problems.append(f'gold.features_day_ahead: last trainable day is {summary.last_trainable_date}')

# COMMAND ----------

if problems:
    raise RuntimeError('Pipeline health check failed:\n- ' + '\n- '.join(problems))
print('All checks passed.')
