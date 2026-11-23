# Model validation report: day-ahead renewable share

| | |
|---|---|
| **Model** | `renewable_share_day_ahead@champion`: equal average of Bayesian ridge and LightGBM, with a calibrated 80% interval |
| **Registry** | Unity Catalog `pulse.ml.renewable_share_day_ahead` version 1; local MLflow `renewable_share_day_ahead` version 22 |
| **Point model** | Local version 17 (`blend_bayes_lgbm`), MLflow run `d92f4e59276e47dc81cb5df53a79071d` |
| **Evidence runs** | Interval calibration `7b831968d70d4ab98184b553ed735c91`; validation `36bf1bafc9df4210a0b9f24073334ef4`; package `07d9ec4a030d424ca0e8916d0b4d1a99` |
| **Data** | Feature store `pulse.gold.feature_store_day_ahead`, feature set `v1` (131 features), target days 1 Nov 2023 to 5 Nov 2026 |
| **Code** | Branch `feature-store`, `eda/` (experiments) and `databricks/src/notebooks/score_day_ahead.py` (scoring) |
| **Prepared** | 22 November 2026 |
| **Status** | **Developer validation.** Not independently validated: the person who built the model prepared this report. It is the evidence pack for an independent reviewer. |

## 1. Summary

The model predicts, at 18:00 UTC on day D−1, the share of Great Britain's transmission-metered electricity generation on day D that came from wind, hydro and biomass.

Out of sample, on 674 days from January 2025 to November 2026, it has a mean absolute error of **3.30 points** (95% interval 3.12–3.54). It is significantly more accurate than:
- the current gold baseline (4.47);
- the best linear model (3.59; p < 0.001);
- the best tree model (3.51; p = 0.003).

Its 2026 holdout error (3.33) agrees with the walk-forward error.

The 80% interval covers 78.2% of days, which is consistent with 80% (Kupiec p = 0.28). However, its misses cluster in time (Christoffersen p < 0.001).

**Findings that limit use:**
1. Misses of the interval cluster: after a missed day, the next is missed 35% of the time, against 18% after a hit.
2. Very windy days are under-predicted: above 65% renewable share, the bias is +3.1 points.
3. Accuracy depends heavily on the day-ahead weather. When it is missing, error roughly doubles.
4. Six of the fifteen source tables have assumed, not observed, publication times.

**Suggested conclusion for the reviewer:** fit for its intended use (analytics and demonstration) subject to the conditions in section 9. Not fit for trading or for any decision where a missed interval carries financial consequences.

## 2. Purpose and intended use

| | |
|---|---|
| **Predicts** | Daily renewable share of GB generation (%), with an 80% prediction interval |
| **Made at** | 18:00 UTC on the day before. Every input is restricted to what was published by then. |
| **Intended use** | Portfolio demonstration; analytics on the GB grid; a component and testbed for the Pulse risk engine |
| **Not intended for** | Trading or hedging decisions; regulated advice; estimating embedded (distribution-connected) solar or wind, which the target excludes |
| **Users** | The project owner; reviewers of the project |

## 3. Data

- **Sources:** 15 silver tables (Elexon, NESO, Sheffield Solar, Carbon Intensity, Open-Meteo, GOV.UK), built into a 131-feature table with one row per target day. The National Gas prices are not used.
- **Target:** 100 × (wind + hydro + biomass energy) ÷ (all domestic generation), from Elexon FUELHH. Interconnector flows are excluded from the denominator. Only complete days are used.
- **Leakage control:** every source row has an availability time, and a feature uses a row only if it was available at the cutoff.
- **Leakage audit:** 12 random days were rebuilt from tables with every row published after the cutoff deleted. No feature value changed. The audit caught, and led to fixing, two leaks during development: NESO embedded generation used a day early, and an AGWS ratio that used republished values.
- **Assumed availability:** PV_Live, carbon intensity, the market index, system prices, NESO demand and both weather tables carry no publication time. Their availability is assumed (for example, 30 minutes after the half-hour ends). System prices hold the latest settlement run, which can differ from the first published value.
- **Known data issues:**
  - Elexon AGWS under-reported wind from April to November 2024. It feeds 4 low-importance features.
  - Day-ahead weather starts in March 2024.
  - NESO demand lags the other sources by about three weeks.

## 4. Method

- **Members:**
  - Bayesian ridge regression on all 131 features (median imputation, standardisation; the penalty is learned from the data).
  - LightGBM on all 131 features (missing values handled natively). Settings tuned once by time-ordered cross-validation on 2023–24: 400 trees, 16 leaves, learning rate 0.06, minimum 10 days per leaf.
- **Combination:** an equal-weight average. Learned (stacked) weights were tested and did worse (3.35–3.36 against 3.30): the weights' estimation noise outweighed their benefit on under two years of history.
- **Interval:** the prediction ± 5.36 points, the conformal 80th percentile of earlier out-of-sample absolute errors. Recalibrated whenever the model is retrained.
- **Selection rule, set before comparing candidates:**
  1. Lowest walk-forward MAE among the registered models.
  2. Significantly better than the best linear and best tree models (Diebold–Mariano test, 5%).
  3. 2026 holdout MAE inside the walk-forward MAE's 95% interval.

## 5. Performance

### 5.1 Walk-forward evaluation

From January 2025, each month is predicted by models fitted only on earlier days; then the month joins the training data. 674 days.

| Measure | Value |
|---|---|
| Mean absolute error | **3.30 points** (95% interval 3.12–3.54, 14-day block bootstrap) |
| Root mean squared error | 4.15 |
| R² | 0.907 |
| Bias (actual − predicted) | +0.05 |
| Median / 90th percentile / largest absolute error | 2.83 / 6.70 / 17.28 |
| Mean absolute percentage error | 7.7% |
| 2026 holdout (fitted before 2026, scored on 2026) | MAE 3.33, R² 0.899 |
| In-sample MAE (final model) | 1.56: expected, as LightGBM fits its training data closely |

### 5.2 Benchmarks

| Model | Walk-forward MAE | Champion's gain | Diebold–Mariano p |
|---|---|---|---|
| Training mean | 11.66 | 8.36 | < 0.001 |
| Share two days earlier | 12.75 | 9.45 | < 0.001 |
| Baseline: OLS on the 8 gold features | 4.47 | 1.17 | < 0.001 |
| Best linear: Bayesian ridge | 3.59 | 0.28 | < 0.001 |
| Best tree: LightGBM | 3.51 | 0.21 | 0.003 |
| **Champion** | **3.30** | – | – |

The champion beats the best linear model on 56% of days and the best tree model on 55%.

### 5.3 Stability

| Quarter | Days | MAE | Bias |
|---|---|---|---|
| 2025 Q1 | 90 | 2.89 | −1.33 |
| 2025 Q2 | 91 | 3.12 | +0.38 |
| 2025 Q3 | 92 | 3.34 | +0.78 |
| 2025 Q4 | 92 | 3.67 | +0.38 |
| 2026 Q1 | 90 | 2.94 | −0.42 |
| 2026 Q2 | 91 | 3.43 | −0.46 |
| 2026 Q3 | 92 | 3.58 | **+1.61** |
| 2026 Q4 (to 5 Nov) | 36 | 3.67 | −1.47 |

- **By season:** MAE ranges from 2.95 (spring) to 3.64 (autumn).
- **Monthly MAE** ranges from 1.50 (November 2026, a partial month) to 4.47 (July 2026).
- **Bias swings quarter to quarter.** 2026 Q3 under-predicted by 1.6 points on average. The cumulative sum of errors shows long one-directional runs: spells of under- and over-prediction, not random noise.

### 5.4 Performance by level of the share

| Actual share | Days | MAE | Bias |
|---|---|---|---|
| Under 25% | 35 | 2.52 | −1.66 |
| 25–35% | 143 | 2.79 | −1.48 |
| 35–45% | 153 | 3.19 | −0.43 |
| 45–55% | 158 | 3.63 | +0.68 |
| 55–65% | 134 | 3.47 | +0.79 |
| **Over 65%** | 51 | **4.20** | **+3.14** |

The model is pulled towards the middle: it over-predicts calm days and under-predicts very windy ones. Of the 41 days with errors above 8 points, 68% were under-predictions.

## 6. Uncertainty: the 80% interval

Seven conformal methods were compared on the same 611 days, in the way a VaR model is backtested.

| Method | Coverage | Kupiec p | Christoffersen p | Mean width | Interval score |
|---|---|---|---|---|---|
| **Absolute error, all history (chosen)** | 78.2% | 0.28 | < 0.001 | 10.4 | **14.72** |
| Absolute error, last 180 days | 79.2% | 0.63 | < 0.001 | 10.8 | 14.82 |
| Volatility-scaled | 80.0% | 0.98 | < 0.001 | 12.2 | 15.78 |
| Volatility-scaled, adaptive | 80.2% | 0.90 | 0.001 | 12.3 | 15.87 |
| Scaled by LightGBM's quantile spread (three variants) | 79.5–80.9% | 0.60–0.78 | ≤ 0.007 | 12.6–12.8 | 16.3–16.5 |

- **Kupiec test (miss rate):** no method's miss rate differs significantly from 20%.
- **Christoffersen test (clustering):** every method fails. After a missed day, the next day is missed 29–35% of the time, against 16–18% after a hit. The cause is the point predictions' autocorrelated errors (lag-1 0.33, lag-2 0.18), which no interval construction removes.
- **Selection:** the rule was the lowest interval score among methods passing both tests. None passes both, so the lowest interval score decided.
- **Consequence:** the interval is right on average, but misses arrive in runs. Over a week of turbulent conditions, more than one in five days can fall outside it.

## 7. Sensitivity and behaviour

Measured on the packaged model over the 2026 days.

| Change to the inputs | Mean change in prediction | Largest change |
|---|---|---|
| Forecast wind ÷ demand +10% | +0.21 points | 1.70 |
| Forecast wind ÷ demand −10% | −0.23 | 1.93 |
| All wind forecasts +10% | +0.99 | 3.15 |
| Demand forecast +5% | −0.85 | 2.19 |
| Temperature forecast −3 °C | +0.59 | 1.30 |
| **Day-ahead weather missing** | +1.13 (mean absolute 2.65) | **8.65** |

- **Directions are sensible:** more wind raises the prediction, and more demand lowers it.
- **Changing one feature moves the prediction only a little.** The wind signal is spread across many correlated features, so a change has to come through all of them (see "all wind forecasts +10%").
- **Missing weather matters a lot.** In-sample on 2026, MAE rises from 1.61 to 3.18 when the weather features are removed. The scoring job does not yet check for this: on 7 November 2026 it scored a day without weather.
- **Monotonicity:** the prediction rises by 5.4 points on average as forecast wind ÷ demand goes from 17% to 90%, all else fixed. Small local dips (at most 0.19 points) come from LightGBM's step-shaped response. They're too small to matter, but they mean the response isn't strictly monotonic.

## 8. Explainability

Each prediction decomposes exactly as

> 0.5 × (ridge intercept + each feature's linear term) + 0.5 × (LightGBM base value + each feature's SHAP value).

The reconstruction error is under 10⁻¹³ points. The scoring job stores this decomposition, per day and feature, in `ml.explanations`.

What drives the predictions (2026, mean absolute contribution in points):

| Feature | Contribution | | Source table | Total |
|---|---|---|---|---|
| Forecast wind ÷ forecast demand | 2.25 | | Day-ahead wind and solar forecast | 10.8 |
| Physical estimate of the share | 2.22 | | Day-ahead weather | 8.0 |
| Forecast offshore wind | 1.35 | | Generation (FUELHH) | 3.5 |
| Forecast demand | 1.25 | | REMIT outages | 2.4 |
| Forecast residual demand | 1.21 | | Demand forecast | 2.3 |
| Renewable share over yesterday's last 3 hours | 0.96 | | Wind forecast (WINDFOR) | 2.0 |

The model rests mainly on the day-ahead forecasts and weather, as it should for a day-ahead prediction.

## 9. Limitations and conditions of use

| # | Limitation | Impact | Condition or mitigation |
|---|---|---|---|
| 1 | Interval misses cluster | Runs of days outside the interval | Don't read the interval as independent day by day. Monitor coverage on a rolling 30 days. A point model that uses its recent errors (tested: MAE 3.28) would reduce it. |
| 2 | Very windy days under-predicted (+3.1 points above 65%) | Systematic under-estimate on the highest days | Report as a known bias. Consider a correction by level. |
| 3 | Weather missing roughly doubles error | Poor predictions on days with missing weather | **Required before relying on the job:** the scoring job must flag or hold days without day-ahead weather |
| 4 | Assumed publication times for 6 tables | Possible leakage if real publication is later | Confirm against the live feeds |
| 5 | Concept drift: the wind-to-share slope fell 13% from 2024 to 2026 | Accuracy decays without retraining | Retrain monthly, as evaluated |
| 6 | Short history: 674 out-of-sample days, about two weather years | Rare conditions (long calm spells, extreme cold) are barely represented | Re-validate after a further year |
| 7 | The two members are highly correlated (0.97), and their disagreement doesn't predict error (r = 0.02) | Disagreement can't be used as a warning signal | Use the calibrated interval instead |
| 8 | Packaged under Python 3.14; Databricks installs the model's pinned libraries at load | Environment drift could break loading | Load tested on Databricks (succeeded). Pin the environment in the job. |
| 9 | Target excludes solar and embedded generation | Not the "renewable share" in the everyday sense | State the definition wherever the number is shown |

## 10. Ongoing monitoring

| Check | Frequency | Threshold | Action |
|---|---|---|---|
| Input completeness: day-ahead forecasts and weather present | Each scoring run | Any key input missing | Hold or flag the prediction |
| Rolling 30-day MAE | Daily, once actuals arrive | Above 4.5 (beyond the worst month so far) | Investigate; consider early retraining |
| Rolling 30-day bias | Daily | Beyond ±2 points | Investigate drift |
| Rolling 30-day interval coverage | Daily | Outside 65–92% | Recalibrate |
| Feature drift (PSI against the same months a year earlier, top 20 features) | Weekly | PSI above 0.25 on an important feature | Investigate the source |
| Source health (AGWS ÷ FUELHH wind; NESO ÷ INDO demand) | Daily | Outside the normal band | Quarantine the source |

## 11. Reproducibility

- **Rebuild everything:** `python eda/build_silver.py`, `features.py`, `train_linear.py`, `train_trees.py`, `train_ensembles.py`, `calibrate_intervals.py`, `package_champion.py`, `validate_champion.py`.
- **Recorded on every run:** the Git commit, the feature-set version, the data range and the input dataset digest.
- **Caching:** walk-forward predictions are cached by model definition and data digest, so a rerun recomputes only what changed.
- **Registered model:** includes its input signature, an input example, its code and its pinned library versions.

## 12. Sign-off

| Role | Name | Decision | Date |
|---|---|---|---|
| Model developer | | | |
| Independent validator | | | |
| Model owner (approval for use) | | | |
