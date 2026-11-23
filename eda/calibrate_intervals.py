"""Calibrate the champion's 80% prediction interval, tracked in MLflow.

The champion is the equal average of Bayesian ridge and LightGBM. Its interval so far widened every prediction
by the 80th percentile of all earlier absolute errors, which held only 76-78% of days: errors grew over time
and a fixed width learned from the past fell behind. Five methods are compared, all conformal (calibrated on
earlier out-of-sample errors only), on the same days:

  absolute_all       |error|, every earlier day (the current method)
  absolute_recent    |error|, the last 180 days only: follows drift
  scaled_all         |error| / LightGBM's own 10-90% quantile spread, every earlier day: wide on hard days,
                     narrow on easy ones
  scaled_recent      the same, last 180 days
  scaled_adaptive    scaled, with adaptive conformal inference: the target miss rate is nudged daily by
                     whether recent days were missed (outcomes known two days later, as at the 18:00 cutoff)
  volatility_all     |error| / a 10-day exponentially weighted average of recent |errors| (known by the cutoff):
                     wide after turbulent days, narrow after calm ones, like a volatility model
  volatility_adaptive  the same, with adaptive conformal inference

Each is judged as a VaR model would be: coverage, the Kupiec test (is the miss rate 20%?), the Christoffersen
test (do misses cluster?), and the interval score (width plus a penalty for each miss; lower is better).
Chosen: the lowest interval score among methods that pass both tests.

    python eda/calibrate_intervals.py
"""
import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy import stats

import experiment as ex
from train_ensembles import ensemble_walk_forward
from train_trees import TREES

ALPHA = 0.2              # an 80% interval misses 20% of days
MIN_HISTORY = 60
RECENT = 180
GAMMA = 0.01             # adaptive conformal step size
DELAY = 2                # an outcome is known two days later (yesterday is incomplete at the cutoff)

point, _ = ensemble_walk_forward('blend_bayes_lgbm')
lgbm, _ = ex.walk_forward(TREES, 'lightgbm')
y, pred = point['actual'], point['pred']
# LightGBM's quantile models give a day-specific difficulty: half the width of its own 10-90% band
spread = ((lgbm['own_high'] - lgbm['own_low']) / 2).reindex(point.index)
spread = spread.clip(lower=spread.quantile(0.05))
abs_err = (y - pred).abs()
# Recent error size as known at the cutoff: an exponentially weighted average of |errors| up to DELAY days back
VOL_SPAN = 10
volatility = abs_err.ewm(span=VOL_SPAN).mean().shift(DELAY, freq='D').reindex(point.index)
volatility = volatility.clip(lower=abs_err.quantile(0.1))
SCALES = {'scaled': spread, 'volatility': volatility}


def conformal_quantile(scores, level):
    """The finite-sample conformal quantile of past scores."""
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * level) / n), method='higher'))


def intervals(method):
    """Daily half-widths using only scores from days whose outcome was known (at least DELAY days earlier)."""
    scale = SCALES.get(method.split('_')[0])
    scaled = scale is not None
    score = abs_err / scale if scaled else abs_err
    half, alpha_t, alphas = pd.Series(np.nan, index=point.index), ALPHA, {}
    days = point.index
    for i, day in enumerate(days):
        known = score[days < day - pd.Timedelta(days=DELAY - 1)]
        if 'recent' in method or 'adaptive' in method:
            known = known.tail(RECENT if 'recent' in method else 365)
        if len(known) < MIN_HISTORY:
            continue
        known = known.dropna()
        if len(known) < MIN_HISTORY or (scaled and not np.isfinite(scale[day])):
            continue
        level = 1 - alpha_t if method.endswith('adaptive') else 1 - ALPHA
        q = conformal_quantile(known.to_numpy(), min(max(level, 0.5), 0.995))
        half[day] = q * (scale[day] if scaled else 1.0)
        alphas[day] = alpha_t
        if method.endswith('adaptive') and i >= DELAY:
            # Update with the outcome that has just become known: the day DELAY days back
            past = days[i - DELAY]
            if np.isfinite(half.get(past, np.nan)):
                missed = abs(y[past] - pred[past]) > half[past]
                alpha_t = alpha_t + GAMMA * (ALPHA - float(missed))
    return half, pd.Series(alphas)


def kupiec(misses, p=ALPHA):
    """Likelihood-ratio test that the miss rate is p."""
    n, x = len(misses), int(misses.sum())
    rate = x / n
    if rate in (0, 1):
        return float('nan'), float('nan')
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - rate) - x * np.log(rate))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(misses):
    """Likelihood-ratio test that a miss today doesn't make a miss tomorrow more likely."""
    m = misses.astype(int).to_numpy()
    n00 = int(((m[:-1] == 0) & (m[1:] == 0)).sum()); n01 = int(((m[:-1] == 0) & (m[1:] == 1)).sum())
    n10 = int(((m[:-1] == 1) & (m[1:] == 0)).sum()); n11 = int(((m[:-1] == 1) & (m[1:] == 1)).sum())
    p01, p11, p = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1), (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(prob, a, b):
        return (a * np.log(1 - prob) if a and prob < 1 else 0) + (b * np.log(prob) if b and prob > 0 else 0)
    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return float(lr), float(stats.chi2.sf(lr, 1)), p01, p11


def interval_score(low, high, actual, alpha=ALPHA):
    """Winkler interval score: width, plus 2/alpha times how far outside the interval the actual fell."""
    return ((high - low) + (2 / alpha) * (low - actual).clip(lower=0) + (2 / alpha) * (actual - high).clip(lower=0)).mean()


METHODS = ['absolute_all', 'absolute_recent', 'scaled_all', 'scaled_recent', 'scaled_adaptive', 'volatility_all', 'volatility_adaptive']
halves = {m: intervals(m) for m in METHODS}
# Judge every method on the same days: those where all of them have an interval
common = point.index[np.all([halves[m][0].notna() for m in METHODS], axis=0)]
rows, misses_by = [], {}
for method in METHODS:
    half = halves[method][0][common]
    low, high = pred[common] - half, pred[common] + half
    miss = (y[common] < low) | (y[common] > high)
    misses_by[method] = miss
    _, kp = kupiec(miss)
    _, cp, p01, p11 = christoffersen(miss)
    seasons = miss.groupby(miss.index.month.map(ex.SEASONS)).mean()
    quarters = 1 - miss.groupby(miss.index.to_period('Q')).mean()
    rows.append({'method': method, 'coverage_pct': 100 * (1 - miss.mean()), 'mean_width': (high - low).mean(),
                 'median_width': (high - low).median(), 'interval_score': interval_score(low, high, y[common]),
                 'kupiec_p': kp, 'christoffersen_p': cp, 'miss_after_hit_pct': 100 * p01, 'miss_after_miss_pct': 100 * p11,
                 'worst_quarter_coverage_pct': 100 * quarters.min(), 'best_quarter_coverage_pct': 100 * quarters.max(),
                 **{f'coverage_{s}_pct': 100 * (1 - v) for s, v in seasons.items()}})
table = pd.DataFrame(rows).set_index('method')
passing = table[(table['kupiec_p'] > 0.05) & (table['christoffersen_p'] > 0.05)]
chosen = (passing if len(passing) else table)['interval_score'].idxmin()

# What the deployed model needs: the score quantile as of the last known day, from the chosen method
kind = chosen.split('_')[0]
score = (abs_err / SCALES[kind]) if kind in SCALES else abs_err
window = (score.tail(RECENT) if 'recent' in chosen else (score.tail(365) if 'adaptive' in chosen else score)).dropna()
final_alpha = float(halves[chosen][1].iloc[-1]) if chosen.endswith('adaptive') else ALPHA
final_q = conformal_quantile(window.to_numpy(), 1 - final_alpha)
calibration = {'method': chosen, 'scale': {'scaled': 'lightgbm 10-90% quantile half-spread',
                                           'volatility': f'{VOL_SPAN}-day EWMA of |error|, {DELAY} days back'}.get(kind),
               'scale_floor': {'scaled': float(spread.quantile(0.05)), 'volatility': float(abs_err.quantile(0.1))}.get(kind),
               'volatility_span': VOL_SPAN if kind == 'volatility' else None, 'q': final_q, 'alpha': final_alpha,
               'window_days': len(window), 'calibrated_through': window.index.max().date().isoformat(),
               'last_scale': float(SCALES[kind].dropna().iloc[-1]) if kind in SCALES else None}
(ex.CACHE / 'interval_calibration.json').write_text(json.dumps(calibration, indent=1))

with mlflow.start_run(run_name='interval-calibration', description='Calibrating the champion ensemble 80% interval') as run:
    mlflow.set_tags({'model_family': 'calibration', 'stage': 'experiment', 'point_model': 'blend_bayes_lgbm', 'git_commit': ex.COMMIT})
    mlflow.log_params({'alpha': ALPHA, 'min_history': MIN_HISTORY, 'recent_days': RECENT, 'adaptive_gamma': GAMMA, 'volatility_span': VOL_SPAN,
                       'outcome_delay_days': DELAY, 'evaluation_days': len(common), 'chosen_method': chosen,
                       'selection_rule': 'lowest interval score among methods passing Kupiec and Christoffersen at 5%'})
    for method, r in table.iterrows():
        mlflow.log_metrics({f'{method}_{k}': float(v) for k, v in r.items() if np.isfinite(v)})
    mlflow.log_metrics({'final_q': final_q, 'final_alpha': final_alpha})
    folder = ex.TRACKING / 'tmp' / 'calibration'
    folder.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(folder / 'calibration_methods.csv')
    (folder / 'calibration.json').write_text(json.dumps(calibration, indent=1))

    fig, ax = plt.subplots(figsize=(11, 3.6))
    for method in METHODS:
        cover = 100 * (1 - misses_by[method].groupby(misses_by[method].index.to_period('M')).mean())
        ax.plot(cover.index.astype(str), cover, marker='o', ms=3, lw=1.4 if method == chosen else 0.9,
                label=f'{method}{"  (chosen)" if method == chosen else ""}')
    ax.axhline(80, color=ex.GREY, ls='--', lw=1)
    ax.set(title='Monthly coverage of the 80% interval, by calibration method', ylabel='% of days inside')
    ax.tick_params(axis='x', rotation=60)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ex.save(fig, folder, 'coverage_by_month.png')

    half = halves[chosen][0][common]
    tail = point.loc[common].tail(150)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(tail.index, tail['pred'] - half[tail.index], tail['pred'] + half[tail.index], color=ex.BLUE, alpha=0.15,
                    label=f'80% interval ({chosen})')
    ax.plot(tail.index, tail['actual'], color=ex.INK, lw=1, label='Actual')
    ax.plot(tail.index, tail['pred'], color=ex.BLUE, lw=1, label='Predicted')
    ax.set(title='Last 150 days with the calibrated interval', ylabel='%')
    ax.legend(frameon=False, loc='upper left')
    ex.save(fig, folder, 'calibrated_interval.png')
    mlflow.log_artifacts(str(folder), artifact_path='calibration')
    print(table.round(3).to_string())
    print(f'\nchosen: {chosen}; deployed q = {final_q:.3f} (alpha {final_alpha:.3f}); run {run.info.run_id}')
