-- Databricks notebook source
-- MAGIC %md
-- MAGIC # One-time setup: access to the landing zone, and the catalog
-- MAGIC
-- MAGIC Run once, top to bottom, in a SQL warehouse or a notebook attached to one.
-- MAGIC
-- MAGIC **Before this notebook**, create the storage credential in the UI (SQL can't create one on GCP):
-- MAGIC 1. Catalog > External data > Credentials > Create credential: type *GCP Service Account*, name `pulse_gcs_cred`.
-- MAGIC 2. Copy the service account email Databricks shows, and give it read-only access to the bucket:
-- MAGIC    ```
-- MAGIC    gcloud storage buckets add-iam-policy-binding gs://databricks_1 --member="serviceAccount:THE_EMAIL" --role="roles/storage.objectViewer"
-- MAGIC    gcloud storage buckets add-iam-policy-binding gs://databricks_1 --member="serviceAccount:THE_EMAIL" --role="roles/storage.legacyBucketReader"
-- MAGIC    ```
-- MAGIC
-- MAGIC The ingestion code writes with a create-only account; Databricks reads with a read-only one.

-- COMMAND ----------

-- Unity Catalog's permission to use the landing path, through that credential
CREATE EXTERNAL LOCATION IF NOT EXISTS pulse_landing
  URL 'gs://databricks_1/electricity_gen_by_fuel/landing'
  WITH (STORAGE CREDENTIAL pulse_gcs_cred)
  COMMENT 'Raw landing files written by the electrcity_pulse ingestion code (create-only writer)';

-- COMMAND ----------

-- Should list the ingest_date= folders. A permission error means step 2 above is missing
LIST 'gs://databricks_1/electricity_gen_by_fuel/landing/fuelhh/';

-- COMMAND ----------

-- If this fails with "no storage location", add: MANAGED LOCATION 'gs://<a bucket for managed tables>/pulse'
CREATE CATALOG IF NOT EXISTS pulse COMMENT 'Electricity Pulse: UK electricity generation lakehouse';

CREATE SCHEMA IF NOT EXISTS pulse.bronze
  COMMENT 'Raw, append-only copies of the GCS landing files, one table per landing folder. Not deduplicated.';
CREATE SCHEMA IF NOT EXISTS pulse.silver
  COMMENT 'Parsed, typed, checked and deduplicated, one table per source; rejected rows in pulse.silver.rejected.';
CREATE SCHEMA IF NOT EXISTS pulse.gold
  COMMENT 'Business-ready tables and day-ahead model features.';
