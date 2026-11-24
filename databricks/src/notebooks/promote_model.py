# Databricks notebook source
# MAGIC %md
# MAGIC # Promote a model version to `@champion`, or roll back
# MAGIC
# MAGIC The human approval step after monthly retraining. Run it once you've reviewed the challenger's training run
# MAGIC (its gate checks are in `gate.json`).
# MAGIC
# MAGIC - **Promote:** leave `version` empty to promote the current `@challenger`, or give a version number.
# MAGIC - **Roll back:** give the `@previous_champion` version.
# MAGIC
# MAGIC It refuses a version that neither passed the retraining gate nor was the previous champion. The champion it
# MAGIC replaces becomes `@previous_champion`, so every promotion can be undone the same way.
# MAGIC
# MAGIC Scoring picks up the new `@champion` at its next run. Monitoring starts afresh for the new version: it needs
# MAGIC about two weeks of predictions before its checks leave `insufficient_data`.

# COMMAND ----------

import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

dbutils.widgets.text('catalog', 'pulse')  # noqa: F821
dbutils.widgets.text('version', '')  # noqa: F821
CATALOG = dbutils.widgets.get('catalog')  # noqa: F821
MODEL = f'{CATALOG}.ml.renewable_share_day_ahead'
client = MlflowClient(registry_uri='databricks-uc')


def alias_version(alias):
    try:
        return client.get_model_version_by_alias(MODEL, alias).version
    except MlflowException:
        return None


champion = alias_version('champion')
previous = alias_version('previous_champion')
target = dbutils.widgets.get('version').strip() or alias_version('challenger')  # noqa: F821
if not target:
    raise ValueError('No version given and no @challenger to promote')

details = client.get_model_version(MODEL, target)
gate = details.tags.get('gate')
print(f'champion: {champion}; previous champion: {previous}; promoting: {target} (gate: {gate}, '
      f"walk-forward MAE {details.tags.get('wf_mae')}, trained through {details.tags.get('trained_through')})")

if target == champion:
    raise ValueError(f'Version {target} is already @champion')
if gate != 'passed' and target != previous:
    raise ValueError(f'Version {target} did not pass the retraining gate and is not the previous champion: refusing')

# COMMAND ----------

user = spark.sql('SELECT current_user()').first()[0]  # noqa: F821
now = f'{pd.Timestamp.now(tz="UTC"):%Y-%m-%d %H:%M} UTC'
client.set_registered_model_alias(MODEL, 'previous_champion', champion)
client.set_registered_model_alias(MODEL, 'champion', target)
client.set_model_version_tag(MODEL, target, 'promoted', f'{now} by {user}' + (' (rollback)' if target == previous else ''))
client.set_model_version_tag(MODEL, champion, 'retired', f'{now} by {user}')
if alias_version('challenger') == target:
    client.delete_registered_model_alias(MODEL, 'challenger')
print(f'Version {target} is now @champion; version {champion} is @previous_champion. To undo, run this again with version {champion}.')
