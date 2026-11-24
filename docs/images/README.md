# Databricks screenshots

Screenshots of the parts of the deployment that only the Databricks UI shows. The README's
"The model in production" section has a commented-out table that uses these exact file names; uncomment it once
they are here.

| File | What to capture | Where in Databricks |
|---|---|---|
| `scoring-job.png` | A successful run's task graph: bronze → silver → feature_store → score → monitor, all green | Jobs & Pipelines → `pulse_day_ahead_scoring` → a successful run |
| `model-registry.png` | The model with its versions and aliases: v1 `@champion`, v2 `@challenger` | Catalog → `pulse` → `ml` → Models → `renewable_share_day_ahead` |
| `retraining-run.png` | The `monthly-retrain` run's metrics, the `gate_*` values and the `gate = passed` tag | Experiments → `renewable-share-day-ahead` → `monthly-retrain` |

Crop to the relevant panel, roughly 1200–1600 px wide, and check that no token, email address or other personal
detail is visible before committing.
