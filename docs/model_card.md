# Model card: day-ahead renewable share

**`pulse.ml.renewable_share_day_ahead@champion`**: Unity Catalog version 1 (local MLflow version 22)

## What it does

Predicts, at 18:00 UTC on day D−1, the share of Great Britain's transmission-metered electricity generation on day D that will come from wind, hydro and biomass. It returns:

| Output | Meaning |
|---|---|
| `predicted_share` | The prediction, % |
| `lower_80`, `upper_80` | An 80% prediction interval: the prediction ± 5.36 points, floored at 0 and capped at 100 |

## How it works

- **The model:** the equal average of two models, each using all 131 day-ahead features in `pulse.gold.feature_store_day_ahead` (feature set `v1`):
  - **Bayesian ridge regression**, a linear model whose penalty is learned from the data;
  - **LightGBM**, gradient-boosted trees.
- **The interval:** conformal, from the 80th percentile of earlier out-of-sample absolute errors.
- **The inputs:** every feature uses only data published by the 18:00 UTC cutoff, mainly the day-ahead wind, solar and demand forecasts and the day-ahead weather.

## How well it works

Out of sample, from January 2025 to November 2026 (674 days):

| | Mean absolute error |
|---|---|
| **This model** | **3.30 points** (95% interval 3.12–3.54); R² 0.907 |
| Best single tree model (LightGBM) | 3.51 |
| Best single linear model (Bayesian ridge) | 3.59 |
| Previous gold baseline | 4.47 |
| 2026 holdout | 3.33 |

The 80% interval covers 78.2% of days, which is statistically consistent with 80%.

## Intended use

- **For:** analytics on the GB grid's renewable share; a demonstration of the Pulse data and MLOps stack.
- **Not for:** trading or hedging decisions, regulated advice, or anything involving embedded (distribution-connected) solar and wind, which the target excludes.

## Known limitations

- **Interval misses come in runs.** It's right about 80% of the time overall, but turbulent weeks can see several misses in a row.
- **Very windy days are under-predicted,** by about 3 points above 65% renewable share.
- **Missing day-ahead weather roughly doubles the error.** Such days are still scored but flagged `weather_missing`, and their interval is widened 1.91× (to about ±10.3 points), the ratio of the 80th percentiles of absolute error without and with weather.
- **Six sources have assumed publication times,** not observed ones.
- **About two years of history,** so rare weather is barely represented.
- **The definition matters:** the target excludes solar, so it isn't the "renewable share" usually quoted in the news.

## Running it

- **Schedule:** the job `pulse_day_ahead_scoring` runs at 18:45 UTC, after refreshing the features.
- **Which days:** every day after the last day with a known outcome that this model version hasn't already scored.
- **Outputs:**
  - `pulse.ml.predictions`: the prediction and interval, with a `data_quality` flag (`ok`, `weather_partial`, `weather_missing`);
  - `pulse.ml.features_served`: the exact inputs the model saw;
  - `pulse.ml.explanations`: each feature's contribution, which adds up exactly to the prediction.

## Monitoring

After each scoring run, `monitor_day_ahead.py` writes `ok` / `warn` / `alert` results to `pulse.ml.monitoring` (rules in `databricks/src/pulse/monitoring.py`):

| Check | Alert when |
|---|---|
| Accuracy, last 30 days with an outcome, overall and per `data_quality` | MAE above 4.5; \|bias\| above 2 points; coverage outside 65–92% or Kupiec p below 0.05 |
| Feature drift, last 30 served days, against the same time of year (±15 days) before registration | PSI or KS beyond the 99th percentile of what 30 days from that season produce by chance (7-day block bootstrap) |
| Prediction drift | the same, for the predictions |
| Freshness | a source's features present at the cutoff fall below half their usual share |

Monitoring records and never fails the job. Drift alone isn't a reason to retrain: in a dry run on October 2026, 17 features had drifted for real reasons (solar capacity growth, nuclear back from outages, prices doubling) while the predictions and accuracy were unaffected.

**Retraining:** monthly, with the interval recalibrated at the same time, or sooner if the accuracy checks alert. A replacement becomes `@champion` only if it meets the selection rule in the validation report.

## Lineage and evidence

- **Full validation:** [model_validation_report.md](model_validation_report.md).
- **Experiments:** MLflow experiment `renewable-share-day-ahead`: linear, tree and ensemble suites, interval calibration and the validation run.
- **Code:** `eda/` (training and evaluation), `databricks/src/pulse/feature_store.py` (features), `databricks/src/notebooks/score_day_ahead.py` (scoring).

**Owner:** the project owner. **Status:** developer-validated; awaiting independent review.
