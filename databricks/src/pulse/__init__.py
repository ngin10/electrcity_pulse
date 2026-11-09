"""Shared code for the Databricks side of Electricity Pulse: bronze, silver and gold.

The pipeline files in src/pipelines/ only declare tables. Everything they compute lives here as plain
DataFrame functions, so it can be unit-tested with a local SparkSession (see databricks/tests/).
"""
