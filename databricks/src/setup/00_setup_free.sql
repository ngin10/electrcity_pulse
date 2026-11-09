-- Databricks notebook source
-- MAGIC %md
-- MAGIC # One-time setup on Databricks Free Edition: the catalog, schemas and landing volume
-- MAGIC
-- MAGIC For the `free` bundle target (the default). Run once, top to bottom, in the SQL editor or a notebook.
-- MAGIC
-- MAGIC Free Edition can't reach your own GCS bucket, so bronze reads a **volume** instead: a folder inside Unity
-- MAGIC Catalog that `databricks/tools/sync_landing.py` keeps in step with the GCS landing zone. No credentials
-- MAGIC or external locations are needed.
-- MAGIC
-- MAGIC **If `CREATE CATALOG` fails** (Free Edition may not allow new catalogs), replace `pulse` with `workspace`,
-- MAGIC the built-in catalog, everywhere below, and deploy with
-- MAGIC `databricks bundle deploy --var="catalog=workspace" --var="landing_path=/Volumes/workspace/bronze/landing"`.

-- COMMAND ----------

CREATE CATALOG IF NOT EXISTS pulse COMMENT 'Electricity Pulse: UK electricity generation lakehouse';

-- COMMAND ----------

CREATE SCHEMA IF NOT EXISTS pulse.bronze
  COMMENT 'Raw, append-only copies of the landing files, one table per landing folder. Not deduplicated.';
CREATE SCHEMA IF NOT EXISTS pulse.silver
  COMMENT 'Parsed, typed, checked and deduplicated, one table per source; rejected rows in silver.rejected.';
CREATE SCHEMA IF NOT EXISTS pulse.gold
  COMMENT 'Business-ready tables and day-ahead model features.';

-- COMMAND ----------

-- The landing files, copied from gs://databricks_1/electricity_gen_by_fuel/landing with the same folder layout
CREATE VOLUME IF NOT EXISTS pulse.bronze.landing
  COMMENT 'Copy of the GCS landing zone, kept in step by databricks/tools/sync_landing.py. Append-only.';

-- COMMAND ----------

-- Empty until the first sync; afterwards it lists the 15 landing folders
LIST '/Volumes/pulse/bronze/landing/';
