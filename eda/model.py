"""A baseline model for the daily renewable share, and models that beat it.

Baseline: linear regression on the eight features gold has today. Challengers: ridge and gradient boosting on
the 131 engineered features (eda/features.py), and their blend. Everything is judged out of sample:

- Hyperparameters are tuned by time-series cross-validation on the days before 2025 only.
- Walk-forward: from January 2025 each month is predicted by models fitted on all earlier days, then the
  month is added to the training data. No prediction ever sees its own or a later day.

Writes eda/out/models/*.joblib (final models, fitted on every day), eda/out/predictions.parquet and
eda/out/model.json (for the report).

    python eda/model.py
"""
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf

from lib import MONTHS, OUT, GradientBoosting, bar, hist, line, num, sanitize, scatter, table

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(7)
MODELS_DIR = OUT / 'models'
MODELS_DIR.mkdir(exist_ok=True)
GOLD = ['fc_wind_solar_mean_mw', 'ndf_mean_mw', 'wx_wind_100m_mean_kmh', 'wx_temperature_mean_c', 'wx_radiation_mean_wm2', 'fuel_share_lag2',
        'fuel_share_lag7', 'cal_is_weekend']
WALK_START = pd.Timestamp('2025-01-01')
BASELINE, RIDGE, GBM, BLEND = 'Baseline: gold\'s 8 features, linear', 'Ridge, 131 features', 'Gradient boosting, 131 features', 'Blend of ridge and gradient boosting'
NAIVE_MEAN, NAIVE_LAG = 'Naive: mean of the training days', 'Naive: share 2 days earlier'

frame = pd.read_parquet(OUT / 'features_day_ahead.parquet').set_index('target_date')
FEATURES = [c for c in frame.columns if c != 'target']
data = frame[frame['target'].notna()]
dev = data[data.index < WALK_START]


def baseline_model():
    return make_pipeline(SimpleImputer(strategy='median'), LinearRegression())


def ridge_model():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3.5, 25)))


def gbm_model(params, **extra):
    return GradientBoosting(random_state=7, **params, **extra)


def fit_baseline(train):
    rows = train.dropna(subset=GOLD)
    return baseline_model().fit(rows[GOLD], rows['target'])


# --- Tune the gradient boosting on the days before the walk-forward starts ------------------------------------

grid = [{'learning_rate': lr, 'max_depth': depth, 'max_iter': iters, 'min_samples_leaf': leaf, 'l2_regularization': 1.0}
        for lr in (0.03, 0.06) for depth in (2, 3, 5) for iters in (200, 400) for leaf in (10, 20)]
splits = list(TimeSeriesSplit(n_splits=5).split(dev))
cv_rows = []
for params in grid:
    scores = []
    for fit_idx, val_idx in splits:
        fit, val = dev.iloc[fit_idx], dev.iloc[val_idx]
        scores.append(np.abs(val['target'] - gbm_model(params).fit(fit[FEATURES], fit['target']).predict(val[FEATURES])).mean())
    cv_rows.append((float(np.mean(scores)), float(np.std(scores)), params))
cv_rows.sort(key=lambda r: r[0])
PARAMS = cv_rows[0][2]
print('tuned:', PARAMS, f'cv MAE {cv_rows[0][0]:.2f}')

# --- Walk forward, a month at a time ----------------------------------------------------------------------------

months = pd.period_range(WALK_START, data.index.max(), freq='M')
parts = []
for month in months:
    train = data[data.index < month.start_time]
    test = data[(data.index >= month.start_time) & (data.index <= month.end_time)]
    if test.empty:
        continue
    ridge = ridge_model().fit(train[FEATURES], train['target'])
    gbm = gbm_model(PARAMS).fit(train[FEATURES], train['target'])
    low = gbm_model(PARAMS, loss='quantile', quantile=0.1).fit(train[FEATURES], train['target'])
    high = gbm_model(PARAMS, loss='quantile', quantile=0.9).fit(train[FEATURES], train['target'])
    out = pd.DataFrame({'actual': test['target'], BASELINE: fit_baseline(train).predict(test[GOLD]),
                        RIDGE: ridge.predict(test[FEATURES]), GBM: gbm.predict(test[FEATURES]),
                        NAIVE_MEAN: train['target'].mean(), NAIVE_LAG: test['fuel_share_lag2'].fillna(train['target'].mean()),
                        'low': low.predict(test[FEATURES]), 'high': high.predict(test[FEATURES]), 'train_days': len(train)}, index=test.index)
    parts.append(out)
wf = pd.concat(parts)
wf[BLEND] = (wf[RIDGE] + wf[GBM]) / 2
# The interval is centred on the gradient boosting quantiles; keep the blend inside it
wf['low'], wf['high'] = np.minimum(wf['low'], wf[BLEND]), np.maximum(wf['high'], wf[BLEND])
# Quantile models fitted on their own training data are too narrow out of sample. Widen each month's interval by the
# 80th percentile of how far earlier out-of-sample days fell outside theirs (conformalised quantile regression)
wf['raw_low'], wf['raw_high'] = wf['low'], wf['high']
miss = np.maximum(wf['raw_low'] - wf['actual'], wf['actual'] - wf['raw_high'])
wf['widen'] = np.nan
for month in months:
    earlier = miss[miss.index < month.start_time]
    if len(earlier) >= 60:
        wf.loc[wf.index.to_period('M') == month, 'widen'] = max(0.0, float(np.quantile(earlier, 0.8)))
calibrated = wf['widen'].notna()
wf['low'], wf['high'] = wf['raw_low'] - wf['widen'].fillna(0), wf['raw_high'] + wf['widen'].fillna(0)
WIDEN = max(0.0, float(np.quantile(miss, 0.8)))
ORDER = [NAIVE_LAG, NAIVE_MEAN, BASELINE, RIDGE, GBM, BLEND]
y = wf['actual']
errors = {name: y - wf[name] for name in ORDER}
BEST = min((RIDGE, GBM, BLEND), key=lambda n: errors[n].abs().mean())


def block_ci(values, fn=np.mean, block=14, reps=1000):
    values = np.asarray(values, dtype=float)
    n, blocks = len(values), math.ceil(len(values) / block)
    starts = RNG.integers(0, n - block + 1, size=(reps, blocks))
    return np.percentile([fn(values[(s[:, None] + np.arange(block)).ravel()[:n]]) for s in starts], [2.5, 97.5])


def diebold_mariano(a, b):
    d = (a.abs() - b.abs()).to_numpy()
    m = sm.OLS(d, np.ones(len(d))).fit(cov_type='HAC', cov_kwds={'maxlags': round(len(d) ** (1 / 3))})
    return float(m.params[0]), float(m.pvalues[0]), m.conf_int()[0]


def score_rows(err_by_model, actual, reference):
    rows, cis = [], {}
    for name, err in err_by_model.items():
        lo, hi = block_ci(err.abs())
        cis[name] = (err.abs().mean(), lo, hi)
        r2 = 1 - (err ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
        if name == reference:
            gain, p = '–', '–'
        else:
            diff, pv, _ = diebold_mariano(err, err_by_model[reference])
            gain, p = f'{-diff:+.2f}', ('< 0.001' if pv < 0.001 else f'{pv:.3f}')
        rows.append([name, f'{err.abs().mean():.2f}', f'{lo:.2f} to {hi:.2f}', f'{np.sqrt((err ** 2).mean()):.2f}', f'{err.abs().median():.2f}',
                     f'{err.abs().quantile(0.9):.2f}', f'{r2:.3f}', f'{err.mean():+.2f}', gain, p])
    return rows, cis


COLS = ['Model', 'MAE', 'MAE 95% interval', 'RMSE', 'Median error', '90th percentile error', 'R²', 'Bias', 'MAE gain over baseline', 'Diebold–Mariano p']
wf_rows, wf_ci = score_rows(errors, y, BASELINE)
for r in wf_rows:
    print('  ', ' | '.join(r))

# The same comparison on 2026 alone, with models fitted once on everything before it
train26, test26 = data[data.index < '2026-01-01'], data[data.index >= '2026-01-01']
p26 = {BASELINE: fit_baseline(train26).predict(test26[GOLD]), RIDGE: ridge_model().fit(train26[FEATURES], train26['target']).predict(test26[FEATURES]),
       GBM: gbm_model(PARAMS).fit(train26[FEATURES], train26['target']).predict(test26[FEATURES]),
       NAIVE_MEAN: np.full(len(test26), train26['target'].mean()), NAIVE_LAG: test26['fuel_share_lag2'].fillna(train26['target'].mean()).to_numpy()}
p26[BLEND] = (p26[RIDGE] + p26[GBM]) / 2
hold_rows, _ = score_rows({n: test26['target'] - p26[n] for n in ORDER}, test26['target'], BASELINE)

# --- Where the best model wins and loses ------------------------------------------------------------------------

best_err, base_err = errors[BEST], errors[BASELINE]
gain, gain_p, gain_ci = diebold_mariano(best_err, base_err)
by_month = pd.DataFrame({BASELINE: base_err.abs(), BEST: best_err.abs()}).groupby(y.index.to_period('M')).mean()
bands = pd.cut(y, [0, 25, 35, 45, 55, 65, 100], labels=['under 25%', '25–35%', '35–45%', '45–55%', '55–65%', 'over 65%'])
by_band = pd.DataFrame({BASELINE: base_err.abs(), BEST: best_err.abs(), 'bias': best_err, 'days': 1.0}).groupby(bands, observed=True).agg(
    {BASELINE: 'mean', BEST: 'mean', 'bias': 'mean', 'days': 'sum'})
by_season = pd.DataFrame({BASELINE: base_err.abs(), BEST: best_err.abs()}).groupby(y.index.month).mean()
inside = (y >= wf['low']) & (y <= wf['high'])
raw_inside = (y >= wf['raw_low']) & (y <= wf['raw_high'])
coverage_month = 100 * inside.groupby(y.index.to_period('M')).mean()
width = wf['high'] - wf['low']
wins = 100 * (best_err.abs() < base_err.abs()).mean()
resid_acf = acf(best_err, nlags=14)[1:]
base_acf = acf(base_err, nlags=14)[1:]
worst = best_err.abs().sort_values(ascending=False).head(10).index
worst_rows = [[f'{d:%Y-%m-%d}', f'{y[d]:.1f}', f'{wf.loc[d, BEST]:.1f}', f'{best_err[d]:+.1f}', f'{wf.loc[d, BASELINE]:.1f}', f'{base_err[d]:+.1f}',
               f"{frame.loc[d, 'fc_wind_share_of_demand']:.0f}", f"{frame.loc[d, 'wx_power_mean']:.2f}"] for d in worst]
month_rows = [[str(m), int((y.index.to_period('M') == m).sum()), f'{r[BASELINE]:.2f}', f'{r[BEST]:.2f}', f'{r[BASELINE] - r[BEST]:+.2f}',
               f'{coverage_month[m]:.0f}%'] for m, r in by_month.iterrows()]
band_rows = [[str(b), int(r['days']), f'{r[BASELINE]:.2f}', f'{r[BEST]:.2f}', f'{r["bias"]:+.2f}'] for b, r in by_band.iterrows()]
tail = wf.tail(150)

# --- Final models, fitted on every scored day -------------------------------------------------------------------

final = {'baseline': fit_baseline(data), 'ridge': ridge_model().fit(data[FEATURES], data['target']),
         'gbm': gbm_model(PARAMS).fit(data[FEATURES], data['target']),
         'gbm_low': gbm_model(PARAMS, loss='quantile', quantile=0.1).fit(data[FEATURES], data['target']),
         'gbm_high': gbm_model(PARAMS, loss='quantile', quantile=0.9).fit(data[FEATURES], data['target'])}
for name, model in final.items():
    joblib.dump({'model': model, 'features': GOLD if name == 'baseline' else FEATURES, 'interval_widening': WIDEN}, MODELS_DIR / f'{name}.joblib')
ahead = frame[frame['target'].isna() & frame['fc_wind_share_of_demand'].notna() & (frame.index > data.index.max())]
ahead_rows = []
for day, row in ahead.iterrows():
    X = row[FEATURES].to_frame().T.astype(float)
    blend = (final['ridge'].predict(X)[0] + final['gbm'].predict(X)[0]) / 2
    lo, hi = min(final['gbm_low'].predict(X)[0], blend) - WIDEN, max(final['gbm_high'].predict(X)[0], blend) + WIDEN
    ahead_rows.append([f'{day:%Y-%m-%d}', f'{blend:.1f}', f'{lo:.1f} to {hi:.1f}', f"{final['baseline'].predict(row[GOLD].to_frame().T.astype(float))[0]:.1f}",
                       f"{row['fc_wind_share_of_demand']:.0f}", f'{100 * row[FEATURES].notna().mean():.0f}%'])
wf.assign(best=wf[BEST]).rename(columns={BASELINE: 'baseline', RIDGE: 'ridge', GBM: 'gbm', BLEND: 'blend', NAIVE_MEAN: 'naive_mean', NAIVE_LAG: 'naive_lag'}) \
    .reset_index(names='target_date').to_parquet(OUT / 'predictions.parquet', index=False)

# --- For the report ---------------------------------------------------------------------------------------------

days = lambda idx: [d.strftime('%Y-%m-%d') for d in idx]
charts = [
    {'t': 'ci', 'title': 'Walk-forward error by model, with 95% intervals', 'sub': f'Mean absolute error in points of renewable share over {len(wf)} days from January 2025, '
     'each predicted by a model fitted only on earlier days. 14-day block bootstrap.', 'x': ORDER, 'est': [num(wf_ci[n][0]) for n in ORDER],
     'lo': [num(wf_ci[n][1]) for n in ORDER], 'hi': [num(wf_ci[n][2]) for n in ORDER], 'unit': 'points', 'ref': None, 'wide': True},
    line('Actual, baseline and best model: the last 150 days', 'Renewable share of GB generation.', days(tail.index),
         [('Actual', tail['actual']), ('Baseline', tail[BASELINE]), ('Best model', tail[BEST])], '%', wide=True),
    line('The best model with its 80% interval: the last 150 days', 'The band is the calibrated 80% interval.', days(tail.index),
         [('Actual', tail['actual']), ('Prediction', tail[BEST]), ('Lower bound', tail['low']), ('Upper bound', tail['high'])], '%', wide=True),
    scatter('Best model: predicted against actual', f'{BEST}, walk-forward.', wf[BEST], y, 'Predicted share (%)', 'Actual share (%)'),
    scatter('Baseline: predicted against actual', 'Gold\'s eight features, linear, walk-forward.', wf[BASELINE], y, 'Predicted share (%)', 'Actual share (%)'),
    line('Error by month', 'Mean absolute error of each month\'s out-of-sample predictions.', [str(m) for m in by_month.index],
         [('Baseline', by_month[BASELINE]), ('Best model', by_month[BEST])], 'points', digits=2, wide=True),
    bar('Error by level of the renewable share', 'Both models are weakest at the extremes; the best model gains most there.', by_band.index,
        [('Baseline', by_band[BASELINE].to_numpy()), ('Best model', by_band[BEST].to_numpy())], 'points', digits=2),
    bar('Error by month of year', 'Mean absolute error, walk-forward days pooled by calendar month.', [MONTHS[m - 1] for m in by_season.index],
        [('Baseline', by_season[BASELINE].to_numpy()), ('Best model', by_season[BEST].to_numpy())], 'points', digits=2),
    hist('Errors of the best model', 'Actual minus predicted.', best_err, 'points', bins=40, clip=None),
    hist('Errors of the baseline', 'Actual minus predicted.', base_err, 'points', bins=40, clip=None),
    bar('Is anything left in the errors?', 'Autocorrelation of each model\'s daily errors. Bars near zero mean the model has used what yesterday could tell it.',
        range(1, 15), [('Baseline', base_acf), ('Best model', resid_acf)], '', digits=3),
    bar('Share of days inside the 80% interval, by month', 'A well-calibrated interval holds 80% of days. Raw: the quantile models alone. Calibrated: widened by past misses, '
        'from the third month on.', [str(m) for m in coverage_month.index],
        [('Calibrated', coverage_month.to_numpy()), ('Raw', (100 * raw_inside.groupby(y.index.to_period('M')).mean()).to_numpy())], '%', digits=0),
]
tables = [
    table(f'Walk-forward from January 2025: {len(wf)} days, refitted monthly', COLS, wf_rows,
          'Errors in points of renewable share. Gain and p-value compare each model with the baseline on the same days.'),
    table(f'One split: fitted before 2026, scored on {len(test26)} days of 2026', COLS, hold_rows, 'The same models without monthly refitting.'),
    table('By month', ['Month', 'Days', 'Baseline MAE', 'Best model MAE', 'Gain', 'Inside 80% interval'], month_rows),
    table('By level of the renewable share', ['Actual share', 'Days', 'Baseline MAE', 'Best model MAE', 'Best model bias'], band_rows,
          'Bias is actual minus predicted: positive means the model predicted too low.'),
    table('The ten worst days for the best model', ['Day', 'Actual %', 'Predicted %', 'Error', 'Baseline %', 'Baseline error', 'Forecast wind ÷ demand %',
                                                    'Weather power'], worst_rows, 'Error is actual minus predicted.'),
    table('Gradient boosting: the five best settings in cross-validation', ['CV MAE', 'Std dev across folds', 'Learning rate', 'Depth', 'Trees', 'Min leaf'],
          [[f'{m:.2f}', f'{s:.2f}', p['learning_rate'], p['max_depth'], p['max_iter'], p['min_samples_leaf']] for m, s, p in cv_rows[:5]],
          f'Five time-ordered folds on the {len(dev)} days before 2025. The first row is used throughout.'),
    table('Predictions for days not yet scored', ['Target day', 'Best model %', '80% interval', 'Baseline %', 'Forecast wind ÷ demand %', 'Features present'],
          ahead_rows, 'From the final models, fitted on every scored day. Days at the edge of the data lack yesterday\'s actuals.'),
]
summary = {'baseline': BASELINE, 'best': BEST, 'days': len(wf), 'baseline_mae': num(base_err.abs().mean()), 'best_mae': num(best_err.abs().mean()),
           'gain': num(-gain), 'gain_lo': num(-gain_ci[1]), 'gain_hi': num(-gain_ci[0]), 'gain_p': gain_p, 'wins': num(wins, 1),
           'coverage': num(100 * inside[calibrated].mean(), 1), 'width': num(width[calibrated].mean(), 1), 'coverage_raw': num(100 * raw_inside.mean(), 1),
           'width_raw': num((wf['raw_high'] - wf['raw_low']).mean(), 1), 'widen': num(WIDEN, 2), 'calibrated_days': int(calibrated.sum()), 'best_r2': num(1 - (best_err ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3),
           'baseline_r2': num(1 - (base_err ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3), 'params': PARAMS, 'months': len(by_month),
           'months_won': int((by_month[BEST] < by_month[BASELINE]).sum())}
(OUT / 'model.json').write_text(json.dumps(sanitize({'summary': summary, 'charts': charts, 'tables': tables}), allow_nan=False, ensure_ascii=False,
                                           separators=(',', ':')), encoding='utf-8')
print('summary', summary)
print('holdout'); [print('  ', ' | '.join(r)) for r in hold_rows]
print('bands'); [print('  ', r) for r in band_rows]
print('season', by_season.round(2).to_dict())
print('acf best', np.round(resid_acf[:4], 2), 'baseline', np.round(base_acf[:4], 2), 'err skew', round(float(best_err.skew()), 2))
print('worst'); [print('  ', r) for r in worst_rows]
print('ahead', ahead_rows)
print('ridge alpha', final['ridge'][-1].alpha_)
