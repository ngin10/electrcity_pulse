"""The experiment machinery every model family shares: walk-forward evaluation, metrics, charts, MLflow logging.

A family (linear, tree, ...) describes its models with a Family: a catalogue of estimators and how to read them
back (coefficients or importances, chosen hyperparameters, own intervals). run_suite() then judges every model
the same way: from January 2025 each month is predicted by a model fitted only on earlier days, then added to the
training data. One parent run holds the comparison; each model is a nested run with:

- params:    the model, its features, target transform, tuning, data and code versions;
- metrics:   walk-forward accuracy (MAE, RMSE, R², bias, error percentiles, MAPE, max error), the MAE's 95%
             interval, Diebold-Mariano tests against the baseline and the best linear model, 80% interval coverage and
             width, residual autocorrelation, error by share band and season, a 2026 holdout, in-sample fit, and
             per-month series that MLflow charts by step;
- artifacts: diagnostic charts and tables;
- the model: fitted on every day, saved with its input signature and registered as a version of one model.

Tracking goes to a local MLflow database unless MLFLOW_TRACKING_URI says otherwise (e.g. "databricks").
"""
import hashlib
import json
import math
import os
import re
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy import stats
from sklearn.inspection import permutation_importance
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from lib import MONTHS, OUT, ROOT

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings('ignore')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
RNG = np.random.default_rng(7)
EDA = Path(__file__).resolve().parent
TRACKING = OUT / 'mlflow'
CACHE = TRACKING / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
EXPERIMENT = os.getenv('MLFLOW_EXPERIMENT_NAME', 'renewable-share-day-ahead')
REGISTERED_MODEL = os.getenv('PULSE_MODEL_NAME', 'renewable_share_day_ahead')
WALK_START = pd.Timestamp('2025-01-01')
HOLDOUT_START = pd.Timestamp('2026-01-01')
GOLD = ['fc_wind_solar_mean_mw', 'ndf_mean_mw', 'wx_wind_100m_mean_kmh', 'wx_temperature_mean_c', 'wx_radiation_mean_wm2',
        'fuel_share_lag2', 'fuel_share_lag7', 'cal_is_weekend']
BANDS = [(0, 25, 'under_25'), (25, 35, '25_35'), (35, 45, '35_45'), (45, 55, '45_55'), (55, 65, '55_65'), (65, 101, 'over_65')]
SEASONS = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 6: 'summer', 7: 'summer', 8: 'summer',
           9: 'autumn', 10: 'autumn', 11: 'autumn'}
INK, BLUE, ORANGE, GREY, GRID = '#1f2328', '#2a78d6', '#eb6834', '#8a8f98', '#e3e5e8'
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'axes.edgecolor': GREY, 'font.size': 9, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})

if 'MLFLOW_TRACKING_URI' not in os.environ:
    mlflow.set_tracking_uri(f'sqlite:///{(TRACKING / "mlflow.db").as_posix()}')
client = MlflowClient()
if mlflow.get_experiment_by_name(EXPERIMENT) is None:
    location = (TRACKING / 'artifacts').resolve().as_uri() if 'MLFLOW_TRACKING_URI' not in os.environ else None
    mlflow.create_experiment(EXPERIMENT, artifact_location=location,
                             tags={'project': 'electricity-pulse', 'target': 'daily renewable share of GB generation, %'})
mlflow.set_experiment(EXPERIMENT)

DATA_PATH = OUT / 'features_day_ahead.parquet'
frame = pd.read_parquet(DATA_PATH).set_index('target_date')
FEATURES = [c for c in frame.columns if c != 'target']
# The 30 features eda/features.py selected on the training years
SELECTED = json.loads((OUT / 'features.json').read_text(encoding='utf-8'))['summary']['selected_list']
data = frame[frame['target'].notna()]


def git(*args):
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


COMMIT, DIRTY = git('rev-parse', 'HEAD'), bool(git('status', '--porcelain', '--', 'eda', 'databricks/src'))


@dataclass
class Family:
    """What run_suite needs to know about one family of models."""
    name: str
    # () -> {model: (fresh estimator, feature list, description)}
    catalog: Callable[[], dict]
    # (fitted estimator, X) -> Series of coefficients or importances, or None
    explain: Callable
    explain_title: str
    explain_label: str
    # Types beyond plain scikit-learn that the saved models contain; MLflow's skops format only loads trusted types
    trusted: list
    # The module(s) defining the family's estimators, logged with each model so it can be reloaded anywhere
    code: Path | list
    # (fitted estimator) -> {chosen_<hyperparameter>: value}, recorded at every refit
    chosen: Callable = lambda est: {}
    # (fitted estimator, X) -> (low, high) of the model's own 80% interval, or None
    interval: Callable = lambda est, X: None
    # (model) -> extra params to log
    extra_params: Callable = lambda model: {}
    # (model, fitted estimator, X, folder) -> extra chart paths
    extra_charts: Callable = None
    # (model) -> {file name: DataFrame} extra tables to log
    extra_tables: Callable = lambda model: {}
    # Compute permutation importance on the 2026 holdout (meaningful for any model; slow for big ones)
    permutation: bool = False
    # Models whose walk-forward errors every run is compared against: label -> (Family, model)
    references: dict = field(default_factory=dict)
    # () -> catalogue for the 2026 holdout, when it must differ from the final one (e.g. weights learned before 2026)
    holdout_catalog: Callable = None
    # (model) -> (walk-forward predictions, chosen), when the family builds them another way (e.g. from other models')
    walk_forward: Callable = None


# --- Statistics ---------------------------------------------------------------------------------------------------

def block_ci(values, block=14, reps=1000):
    """95% interval of the mean by a 14-day moving-block bootstrap: neighbouring days' errors are related."""
    values = np.asarray(values, dtype=float)
    n, blocks = len(values), math.ceil(len(values) / block)
    starts = RNG.integers(0, n - block + 1, size=(reps, blocks))
    return np.percentile([values[(s[:, None] + np.arange(block)).ravel()[:n]].mean() for s in starts], [2.5, 97.5])


def diebold_mariano(err, ref_err):
    """Mean of |err| - |ref_err| with Newey-West errors: negative means more accurate than the reference."""
    d = (err.abs() - ref_err.abs()).to_numpy()
    fit = sm.OLS(d, np.ones(len(d))).fit(cov_type='HAC', cov_kwds={'maxlags': round(len(d) ** (1 / 3))})
    return float(fit.params[0]), float(fit.tvalues[0]), float(fit.pvalues[0])


def accuracy(actual, pred, prefix=''):
    err = actual - pred
    return {f'{prefix}mae': err.abs().mean(), f'{prefix}rmse': math.sqrt((err ** 2).mean()),
            f'{prefix}r2': 1 - (err ** 2).sum() / ((actual - actual.mean()) ** 2).sum(), f'{prefix}bias': err.mean()}


# --- Charts -------------------------------------------------------------------------------------------------------

def save(fig, folder, name):
    path = folder / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def charts(name, wf, err, base_err, explained, explain_title, explain_label, chosen, folder):
    """The diagnostic charts for one model's walk-forward predictions."""
    paths = []
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(wf.index, wf['actual'], color=INK, lw=1, label='Actual')
    ax.plot(wf.index, wf['pred'], color=BLUE, lw=1, label='Predicted')
    ax.set(title=f'{name}: actual and predicted share, walk-forward', ylabel='%')
    ax.legend(frameon=False, loc='upper left')
    paths.append(save(fig, folder, '01_actual_vs_predicted.png'))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(wf['pred'], wf['actual'], s=7, alpha=0.45, color=BLUE)
    lims = [min(wf['pred'].min(), wf['actual'].min()) - 2, max(wf['pred'].max(), wf['actual'].max()) + 2]
    ax.plot(lims, lims, color=ORANGE, lw=1.2, label='Perfect prediction')
    ax.set(xlim=lims, ylim=lims, xlabel='Predicted share (%)', ylabel='Actual share (%)', title='Predicted against actual')
    ax.legend(frameon=False)
    paths.append(save(fig, folder, '02_predicted_vs_actual.png'))

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axhline(0, color=GREY, lw=0.8)
    ax.plot(err.index, err, color=BLUE, lw=0.6, alpha=0.6, label='Daily error')
    ax.plot(err.index, err.rolling(30, min_periods=10).mean(), color=ORANGE, lw=1.6, label='30-day mean (bias)')
    ax.set(title='Errors over time: actual - predicted', ylabel='points')
    ax.legend(frameon=False, loc='upper left')
    paths.append(save(fig, folder, '03_errors_over_time.png'))

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.hist(err, bins=40, color=BLUE, alpha=0.8, density=True)
    xs = np.linspace(err.min(), err.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, err.mean(), err.std()), color=ORANGE, lw=1.4, label='Normal with the same mean and sd')
    ax.set(title='Distribution of errors', xlabel='points', ylabel='density')
    ax.legend(frameon=False)
    paths.append(save(fig, folder, '04_error_distribution.png'))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    (osm, osr), (slope, intercept, _) = stats.probplot(err, dist='norm')
    ax.scatter(osm, osr, s=6, color=BLUE)
    ax.plot(osm, slope * osm + intercept, color=ORANGE, lw=1.2)
    ax.set(title='Q-Q plot of errors against a normal', xlabel='Normal quantiles', ylabel='Error quantiles')
    paths.append(save(fig, folder, '05_error_qq.png'))

    lags = acf(err, nlags=14)[1:]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(1, 15), lags, color=BLUE)
    band = 1.96 / math.sqrt(len(err))
    ax.axhspan(-band, band, color=GRID, alpha=0.7, label='95% band for no autocorrelation')
    ax.set(title='Autocorrelation of errors', xlabel='lag (days)', ylabel='r')
    ax.legend(frameon=False)
    paths.append(save(fig, folder, '06_error_autocorrelation.png'))

    month = err.index.to_period('M')
    by_month = pd.DataFrame({'Baseline': base_err.abs(), name: err.abs()}).groupby(month).mean()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(by_month.index.astype(str), by_month['Baseline'], color=GREY, marker='o', ms=3, label='Baseline (ols_gold8)')
    ax.plot(by_month.index.astype(str), by_month[name], color=BLUE, marker='o', ms=3, label=name)
    ax.set(title='Mean absolute error by month', ylabel='points')
    ax.tick_params(axis='x', rotation=60)
    ax.legend(frameon=False)
    paths.append(save(fig, folder, '07_error_by_month.png'))

    band_err = pd.DataFrame({'Baseline': base_err.abs(), name: err.abs()})
    labels = pd.cut(wf['actual'], [b[0] for b in BANDS] + [101], labels=[b[2] for b in BANDS], right=False)
    by_band = band_err.groupby(labels, observed=True).mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(by_band))
    ax.bar(x - 0.2, by_band['Baseline'], 0.4, color=GREY, label='Baseline')
    ax.bar(x + 0.2, by_band[name], 0.4, color=BLUE, label=name)
    ax.set_xticks(x, [b.replace('_', '–') for b in by_band.index])
    ax.set(title='Error by level of the actual share', xlabel='actual share (%)', ylabel='MAE, points')
    ax.legend(frameon=False)
    paths.append(save(fig, folder, '08_error_by_share_band.png'))

    by_season = pd.DataFrame({'Baseline': base_err.abs(), name: err.abs()}).groupby(err.index.month).mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(by_season))
    ax.bar(x - 0.2, by_season['Baseline'], 0.4, color=GREY, label='Baseline')
    ax.bar(x + 0.2, by_season[name], 0.4, color=BLUE, label=name)
    ax.set_xticks(x, [MONTHS[m - 1] for m in by_season.index])
    ax.set(title='Error by month of year', ylabel='MAE, points')
    ax.legend(frameon=False)
    paths.append(save(fig, folder, '09_error_by_month_of_year.png'))

    tail = wf.tail(150)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    if tail['low'].notna().any():
        ax.fill_between(tail.index, tail['low'], tail['high'], color=BLUE, alpha=0.15, label='80% interval (conformal)')
    if 'own_low' in tail and tail['own_low'].notna().any():
        ax.plot(tail.index, tail['own_low'], color=ORANGE, lw=0.8, ls='--', label="Model's own 80% interval")
        ax.plot(tail.index, tail['own_high'], color=ORANGE, lw=0.8, ls='--')
    ax.plot(tail.index, tail['actual'], color=INK, lw=1, label='Actual')
    ax.plot(tail.index, tail['pred'], color=BLUE, lw=1, label='Predicted')
    ax.set(title='Last 150 days with the 80% prediction interval', ylabel='%')
    ax.legend(frameon=False, loc='upper left')
    paths.append(save(fig, folder, '10_prediction_interval.png'))

    if explained is not None and len(explained):
        top = explained.reindex(explained.abs().sort_values(ascending=False).index).head(25)[::-1]
        fig, ax = plt.subplots(figsize=(7, 6.5))
        ax.barh(top.index, top.values, color=[BLUE if v >= 0 else ORANGE for v in top.values])
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set(title=explain_title, xlabel=explain_label)
        paths.append(save(fig, folder, '11_feature_effects.png'))

    if chosen is not None and not chosen.empty:
        fig, axes = plt.subplots(1, chosen.shape[1], figsize=(4 * chosen.shape[1], 3), squeeze=False)
        for ax, column in zip(axes[0], chosen.columns):
            ax.plot(chosen.index.astype(str), chosen[column], color=BLUE, marker='o', ms=3)
            if column in ('chosen_alpha', 'chosen_lambda') and (chosen[column] > 0).all():
                ax.set_yscale('log')
            ax.set(title=f'{column.removeprefix("chosen_")} chosen at each refit')
            ax.tick_params(axis='x', rotation=70, labelsize=7)
        paths.append(save(fig, folder, '12_hyperparameters_by_refit.png'))
    return paths


# --- Walk forward ---------------------------------------------------------------------------------------------------

def cache_key(family, name):
    """Changes whenever the model's definition, its features or the data change."""
    est, cols, _ = family.catalog()[name]
    data_digest = pd.util.hash_pandas_object(data[cols + ['target']], index=True).sum()
    # repr() shows functions with their memory address, which changes every run: drop it
    definition = re.sub(r' at 0x[0-9a-fA-F]+', '', repr(est))
    return hashlib.sha256(f'{definition}|{cols}|{data_digest}|{WALK_START}'.encode()).hexdigest()[:16]


def walk_forward(family, name):
    """Monthly refits from WALK_START, cached on disk by model definition and data, so a logging failure never
    means retraining. Returns predictions, the model's own intervals and the chosen hyperparameters."""
    key = cache_key(family, name)
    wf_file, chosen_file = CACHE / f'{name}-{key}.parquet', CACHE / f'{name}-{key}-chosen.parquet'
    if wf_file.exists():
        chosen = pd.read_parquet(chosen_file) if chosen_file.exists() else pd.DataFrame()
        if len(chosen):
            chosen.index = pd.PeriodIndex(chosen.index, freq='M')
        return pd.read_parquet(wf_file), chosen
    months = pd.period_range(WALK_START, data.index.max(), freq='M')
    parts, chosen = [], {}
    for month in months:
        train = data[data.index < month.start_time]
        test = data[(data.index >= month.start_time) & (data.index <= month.end_time)]
        if test.empty:
            continue
        est, cols, _ = family.catalog()[name]
        est.fit(train[cols], train['target'])
        part = pd.DataFrame({'actual': test['target'], 'pred': est.predict(test[cols])}, index=test.index)
        own = family.interval(est, test[cols])
        part['own_low'], part['own_high'] = (own if own is not None else (np.nan, np.nan))
        parts.append(part)
        chosen[month] = family.chosen(est)
    wf = pd.concat(parts)
    conformal(wf)
    chosen = pd.DataFrame.from_dict(chosen, orient='index')
    wf.to_parquet(wf_file)
    if not chosen.empty:
        chosen.set_axis(chosen.index.astype(str)).to_parquet(chosen_file)
    return wf, chosen


def conformal(wf):
    """A split-conformal 80% interval: widen each month's predictions by the 80th percentile of earlier out-of-sample
    misses (at least 60 of them). In place."""
    wf['low'], wf['high'] = np.nan, np.nan
    abs_err = (wf['actual'] - wf['pred']).abs()
    for month in sorted(set(wf.index.to_period('M'))):
        earlier = abs_err[abs_err.index < month.start_time]
        here = wf.index.to_period('M') == month
        if len(earlier) >= 60:
            q = float(np.quantile(earlier, 0.8))
            wf.loc[here, 'low'], wf.loc[here, 'high'] = wf.loc[here, 'pred'] - q, wf.loc[here, 'pred'] + q


def residual_corrected(wf):
    """Add phi x the error from two days earlier. Two, not one: yesterday's actual share is incomplete at the 18:00
    cutoff. phi is estimated from earlier out-of-sample errors only (at least 60), so nothing looks ahead."""
    err = wf['actual'] - wf['pred']
    lag2 = err.shift(2, freq='D').reindex(err.index)
    out, phis = wf.copy(), {}
    for month in sorted(set(wf.index.to_period('M'))):
        past = pd.DataFrame({'e': err, 'lag': lag2})[err.index < month.start_time].dropna()
        phi = float(np.polyfit(past['lag'], past['e'], 1)[0]) if len(past) >= 60 else 0.0
        here = wf.index.to_period('M') == month
        out.loc[here, 'pred'] = wf.loc[here, 'pred'] + phi * lag2[here].fillna(0)
        phis[month] = {'chosen_phi': phi}
    conformal(out)
    out['own_low'] = out['own_high'] = np.nan
    return out, pd.DataFrame.from_dict(phis, orient='index')


# --- One model, one run -------------------------------------------------------------------------------------------

def feature_set_name(cols):
    return 'gold8' if cols == GOLD else ('selected30' if cols == SELECTED else 'all131')


def run_model(family, name, wf, chosen, refs, description, est_for_final, cols, extra_params=None):
    """Log one model's walk-forward results as a nested run, and register its final model."""
    folder = TRACKING / 'tmp' / family.name / name
    folder.mkdir(parents=True, exist_ok=True)
    err = wf['actual'] - wf['pred']
    base_err = refs['baseline']
    with mlflow.start_run(run_name=name, nested=True, description=description) as run:
        mlflow.set_tags({'model_family': family.name, 'stage': 'experiment', 'baseline': str(name == 'ols_gold8').lower(),
                         'git_commit': COMMIT, 'git_dirty': str(DIRTY).lower()})
        mlflow.log_input(mlflow.data.from_pandas(frame.reset_index(), source=str(DATA_PATH), name='features_day_ahead', targets='target'),
                         context='training')
        params = {'model': name, 'family': family.name,
                  'estimator': type(est_for_final).__name__ if est_for_final is not None else 'post-processing',
                  'feature_set': feature_set_name(cols), 'n_features': len(cols), 'target': 'daily renewable share of GB generation, %',
                  'target_transform': 'logit' if 'logit' in name else ('gap to fc_share_estimate' if 'gap' in name else 'none'),
                  'walk_forward_start': WALK_START.date().isoformat(), 'refit': 'monthly, all earlier days',
                  'interval': 'split-conformal 80% from earlier out-of-sample errors', 'forecast_cutoff': '18:00 UTC the day before',
                  'feature_set_version': 'v1', 'data_rows': len(data),
                  'data_first_day': data.index.min().date().isoformat(), 'data_last_day': data.index.max().date().isoformat(),
                  **family.extra_params(name), **(extra_params or {})}
        mlflow.log_params({k: v for k, v in params.items() if v is not None})

        # Accuracy, uncertainty, significance, calibration, diagnostics
        m = accuracy(wf['actual'], wf['pred'], 'wf_')
        lo, hi = block_ci(err.abs())
        m.update({'wf_mae_ci_low': lo, 'wf_mae_ci_high': hi, 'wf_median_abs_error': err.abs().median(),
                  'wf_p90_abs_error': err.abs().quantile(0.9), 'wf_max_abs_error': err.abs().max(),
                  'wf_mape': (100 * err.abs() / wf['actual'].clip(lower=1)).mean(), 'wf_days': len(wf)})
        for label, ref_err in refs.items():
            if (label == 'baseline' and name == 'ols_gold8') or label == f'best_{family.name}':
                continue
            diff, t, p = diebold_mariano(err, ref_err)
            m.update({f'dm_mae_difference_vs_{label}': diff, f'dm_t_vs_{label}': t, f'dm_pvalue_vs_{label}': p,
                      f'mae_gain_vs_{label}': ref_err.abs().mean() - err.abs().mean(),
                      f'days_better_than_{label}_pct': 100 * (err.abs() < ref_err.abs()).mean()})
        calibrated = wf['low'].notna()
        inside = (wf['actual'] >= wf['low']) & (wf['actual'] <= wf['high'])
        m.update({'interval80_coverage_pct': 100 * inside[calibrated].mean(), 'interval80_width': (wf['high'] - wf['low'])[calibrated].mean()})
        if wf['own_low'].notna().any():
            own = (wf['actual'] >= wf['own_low']) & (wf['actual'] <= wf['own_high'])
            m.update({'own_interval80_coverage_pct': 100 * own.mean(), 'own_interval80_width': (wf['own_high'] - wf['own_low']).mean()})
        ac = acf(err, nlags=7)
        m.update({'error_acf_lag1': ac[1], 'error_acf_lag2': ac[2], 'error_acf_lag7': ac[7],
                  'error_ljungbox_pvalue_lag7': float(acorr_ljungbox(err, lags=[7], return_df=True)['lb_pvalue'].iloc[0]),
                  'error_skew': stats.skew(err), 'error_excess_kurtosis': stats.kurtosis(err)})
        labels = pd.cut(wf['actual'], [b[0] for b in BANDS] + [101], labels=[b[2] for b in BANDS], right=False)
        for band, value in err.abs().groupby(labels, observed=True).mean().items():
            m[f'mae_share_{band}'] = value
        for season, value in err.abs().groupby(err.index.month.map(SEASONS)).mean().items():
            m[f'mae_{season}'] = value

        extra_paths, permuted = [], None
        if est_for_final is not None:
            # A single split as a second view: fitted before 2026, scored on 2026
            fresh, _, _ = (family.holdout_catalog or family.catalog)()[name]
            train26, test26 = data[data.index < HOLDOUT_START], data[data.index >= HOLDOUT_START]
            fresh.fit(train26[cols], train26['target'])
            m.update(accuracy(test26['target'], pd.Series(fresh.predict(test26[cols]), index=test26.index), 'holdout2026_'))
            if family.permutation:
                result = permutation_importance(fresh, test26[cols], test26['target'], n_repeats=5, random_state=7,
                                                scoring='neg_mean_absolute_error', n_jobs=1)
                permuted = pd.Series(result.importances_mean, index=cols).sort_values(ascending=False)

            # The final model, fitted on every scored day; its in-sample error shows how much it fits noise
            t0 = time.time()
            est_for_final.fit(data[cols], data['target'])
            m['final_fit_seconds'] = time.time() - t0
            m.update(accuracy(data['target'], pd.Series(est_for_final.predict(data[cols]), index=data.index), 'train_'))
            m['overfit_gap_mae'] = m['wf_mae'] - m['train_mae']
            explained = family.explain(est_for_final, data[cols])
            if family.name == 'linear' and explained is not None:
                m['n_nonzero_coefficients'] = int((explained.abs() > 1e-10).sum())
            mlflow.log_params({f'final_{k}': v for k, v in family.chosen(est_for_final).items()})
            if family.extra_charts:
                extra_paths = family.extra_charts(name, est_for_final, data[cols], folder)
        else:
            explained = None
        mlflow.log_metrics({k: float(v) for k, v in m.items() if v is not None and np.isfinite(v)})

        # Per-month series: MLflow draws these as charts against the step
        for step, (month, e) in enumerate(err.groupby(err.index.to_period('M'))):
            ts = int(month.start_time.timestamp() * 1000)
            mlflow.log_metric('monthly_mae', float(e.abs().mean()), step=step, timestamp=ts)
            mlflow.log_metric('monthly_bias', float(e.mean()), step=step, timestamp=ts)
        numeric = chosen.select_dtypes('number') if not chosen.empty else chosen
        for column in numeric.columns if not numeric.empty else []:
            for step, value in enumerate(numeric[column]):
                if np.isfinite(value):
                    mlflow.log_metric(f'refit_{column}', float(value), step=step)

        # Artifacts: charts and tables
        for path in charts(name, wf, err, base_err, explained, family.explain_title, family.explain_label, numeric, folder) + extra_paths:
            mlflow.log_artifact(str(path), artifact_path='charts')
        if permuted is not None:
            top = permuted.head(25)[::-1]
            fig, ax = plt.subplots(figsize=(7, 6.5))
            ax.barh(top.index, top.values, color=BLUE)
            ax.set(title='Permutation importance on the 2026 holdout', xlabel='rise in MAE when the feature is shuffled, points')
            mlflow.log_artifact(str(save(fig, folder, '13_permutation_importance_2026.png')), artifact_path='charts')
        tables = folder / 'tables'
        tables.mkdir(exist_ok=True)
        wf.rename_axis('target_date').to_csv(tables / 'walk_forward_predictions.csv')
        pd.DataFrame({'mae': err.abs().groupby(err.index.to_period('M')).mean(), 'bias': err.groupby(err.index.to_period('M')).mean()}) \
            .rename_axis('month').to_csv(tables / 'metrics_by_month.csv')
        if not chosen.empty:
            chosen.rename_axis('refit_month').to_csv(tables / 'hyperparameters_by_refit.csv')
        if explained is not None:
            explained.rename('effect').sort_values(key=abs, ascending=False).rename_axis('feature').to_csv(tables / 'feature_effects.csv')
        if permuted is not None:
            permuted.rename('mae_increase').rename_axis('feature').to_csv(tables / 'permutation_importance_2026.csv')
        for file_name, table in family.extra_tables(name).items():
            table.to_csv(tables / file_name, index=False)
        (tables / 'features.json').write_text(json.dumps(cols, indent=1))
        mlflow.log_artifacts(str(tables), artifact_path='tables')

        # The model, with its input signature, registered as a new version
        version = None
        if est_for_final is not None:
            complete = data[cols].dropna()
            example = complete.tail(3) if len(complete) else data[cols].tail(3).fillna(0)
            signature = infer_signature(data[cols].head(50), est_for_final.predict(data[cols].head(50)))
            code = family.code if isinstance(family.code, list) else [family.code]
            common = dict(name='model', signature=signature, input_example=example, code_paths=[str(c) for c in code],
                          registered_model_name=REGISTERED_MODEL)
            try:
                # skops: MLflow's default, which refuses to load types not listed as trusted
                info = mlflow.sklearn.log_model(est_for_final, skops_trusted_types=family.trusted, **common)
                mlflow.set_tag('serialization', 'skops')
            except mlflow.exceptions.MlflowException as error:
                untrusted = str(error).split('Untrusted types found in the file:')[-1].strip()
                print(f'  {name}: skops refused {untrusted}; saving with cloudpickle')
                info = mlflow.sklearn.log_model(est_for_final, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE, **common)
                mlflow.set_tags({'serialization': 'cloudpickle', 'skops_untrusted': untrusted[:250]})
            version = info.registered_model_version
            for key, value in (('model', name), ('family', family.name), ('wf_mae', f"{m['wf_mae']:.3f}")):
                client.set_model_version_tag(REGISTERED_MODEL, version, key, value)
        return {'model': name, 'run_id': run.info.run_id, 'version': version, **m}


# --- A suite: every model of one family -------------------------------------------------------------------------------

def run_suite(family, correct_best=True, extra_suite_params=None):
    """Walk every model forward, log each as a nested run under one parent, and point best_<family> at the winner."""
    names = list(family.catalog())
    refs = {label: (lambda w: w['actual'] - w['pred'])(walk_forward(ref_family, model)[0])
            for label, (ref_family, model) in family.references.items()}
    results, store = [], {}
    with mlflow.start_run(run_name=f'{family.name}-suite-{pd.Timestamp.now():%Y%m%d-%H%M}',
                          description=f'{family.name.capitalize()} models for the day-ahead renewable share, walk-forward from January 2025') as parent:
        mlflow.set_tags({'model_family': family.name, 'stage': 'experiment', 'git_commit': COMMIT, 'git_dirty': str(DIRTY).lower()})
        mlflow.log_params({'models': ', '.join(names), 'walk_forward_start': WALK_START.date().isoformat(), 'n_features_all': len(FEATURES),
                           'data_rows': len(data), 'references': ', '.join(f'{k}={v[1]}' for k, v in family.references.items()),
                           **(extra_suite_params or {})})
        for name in names:
            t0 = time.time()
            store[name] = family.walk_forward(name) if family.walk_forward else walk_forward(family, name)
            wf = store[name][0]
            print(f'{name:28s} walk-forward MAE {(wf.actual - wf.pred).abs().mean():.3f}  ({time.time() - t0:.0f}s)', flush=True)
        for name in names:
            est, cols, description = family.catalog()[name]
            wf, chosen = store[name]
            results.append(run_model(family, name, wf, chosen, refs, description, est, cols))
        best = min(results, key=lambda r: r['wf_mae'])['model']
        if correct_best:
            wf, phis = residual_corrected(store[best][0])
            corrected = f'{best}_residual_corrected'
            results.append(run_model(family, corrected, wf, phis, refs,
                                     f'{best} plus phi x its error two days earlier (yesterday is incomplete at the cutoff)', None,
                                     family.catalog()[best][1], {'base_model': best,
                                                                 'correction': 'phi x error at lag 2, refitted monthly on earlier errors'}))
            print(f'{corrected:28s} walk-forward MAE {results[-1]["wf_mae"]:.3f}')

        board = pd.DataFrame(results).set_index('model').sort_values('wf_mae')
        keep = ['wf_mae', 'wf_mae_ci_low', 'wf_mae_ci_high', 'wf_rmse', 'wf_r2', 'wf_bias', 'mae_gain_vs_baseline', 'dm_pvalue_vs_baseline',
                'mae_gain_vs_best_linear', 'dm_pvalue_vs_best_linear', 'interval80_coverage_pct', 'own_interval80_coverage_pct',
                'own_interval80_width', 'error_acf_lag1', 'holdout2026_mae', 'train_mae', 'n_nonzero_coefficients', 'version']
        board = board[[c for c in keep if c in board.columns]]
        folder = TRACKING / 'tmp' / family.name / '_suite'
        folder.mkdir(parents=True, exist_ok=True)
        board.to_csv(folder / 'leaderboard.csv')
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(board) + 1.5))
        y = np.arange(len(board))[::-1]
        ax.errorbar(board['wf_mae'], y, xerr=[board['wf_mae'] - board['wf_mae_ci_low'], board['wf_mae_ci_high'] - board['wf_mae']],
                    fmt='o', color=BLUE, ecolor=BLUE, capsize=3)
        for label, ref_err in refs.items():
            ax.axvline(ref_err.abs().mean(), color=GREY if label == 'baseline' else ORANGE, ls='--', lw=1, label=label.replace('_', ' '))
        ax.set_yticks(y, board.index)
        ax.set(title='Walk-forward MAE with 95% intervals', xlabel='points of renewable share')
        ax.legend(frameon=False)
        save(fig, folder, 'leaderboard.png')
        mlflow.log_artifacts(str(folder), artifact_path='suite')
        registered = board[board['version'].notna()]
        best_registered = registered['wf_mae'].idxmin()
        mlflow.log_params({'best_model': best_registered})
        mlflow.log_metric('best_wf_mae', float(board.loc[best_registered, 'wf_mae']))
        client.set_registered_model_alias(REGISTERED_MODEL, f'best_{family.name}', str(int(board.loc[best_registered, 'version'])))
        print('\n' + board.round(3).to_string())
        print(f'\nparent run {parent.info.run_id}; best {family.name} model {best_registered} '
              f'(version {int(board.loc[best_registered, "version"])}, alias best_{family.name})')
        return board
