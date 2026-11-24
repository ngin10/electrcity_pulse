"""Monthly retraining of the day-ahead model: walk-forward evaluation, interval calibration and the promotion gate.

The retraining job refits the champion's own specification (cloned from the registered model: same members, same
settings, same code) on the latest feature store. Only the data changes, so the design that was validated is the
design that is retrained. A head-to-head against the champion would be meaningless (the same specification gives
the same walk-forward), so the gate asks instead: does the refreshed model still meet the validated standard?

Gate (every check must pass; thresholds from docs/model_validation_report.md):
  training_days       enough complete days to fit on
  walk_forward_mae    out-of-sample MAE since WALK_START within the validated 95% interval's upper end (3.54)
  recent_mae          walk-forward MAE over the last 90 days at most 4.5 (the monitoring alert level)
  recent_bias         |walk-forward bias| over the last 90 days at most 2 points
  coverage            the conformal 80% interval's walk-forward coverage consistent with 80% (Kupiec p >= 0.05)
  agreement           on the latest days, the challenger's predictions stay close to the champion's: a large jump
                      means something broke (data, code or library), not that the weather changed
  valid_output        every prediction finite and within 0-100

Plain pandas and numpy (scikit-learn only inside walk_forward), so the gate logic is unit-tested without Spark.
"""
import math

import numpy as np
import pandas as pd

WALK_START = pd.Timestamp('2025-01-01')
COVERAGE = 0.80
MIN_TRAINING_DAYS = 600
VALIDATED_MAE_UPPER = 3.54   # upper end of the champion's validated walk-forward MAE 95% interval
RECENT_DAYS = 90
RECENT_MAE_MAX = 4.5
RECENT_BIAS_MAX = 2.0
KUPIEC_P_MIN = 0.05
AGREEMENT_DAYS = 60
AGREEMENT_MEAN_MAX = 2.0     # mean |challenger - champion|, points
AGREEMENT_MAX = 8.0          # largest single-day difference, points


def walk_forward(spec, X, y, start=WALK_START):
    """Monthly refits from `start`: each month is predicted by a fresh clone of `spec` fitted only on earlier days.
    Returns a DataFrame of actual and pred, indexed by day."""
    from sklearn.base import clone

    parts = []
    for month in pd.period_range(start, X.index.max(), freq='M'):
        train = X.index < month.start_time
        test = (X.index >= month.start_time) & (X.index <= month.end_time)
        if not test.any() or train.sum() == 0:
            continue
        model = clone(spec).fit(X[train], y[train])
        parts.append(pd.DataFrame({'actual': y[test], 'pred': model.predict(X[test])}, index=X.index[test]))
    return pd.concat(parts) if parts else pd.DataFrame(columns=['actual', 'pred'])


def conformal_half_width(abs_errors, level=COVERAGE):
    """The finite-sample conformal quantile of out-of-sample absolute errors (as in eda/calibrate_intervals.py)."""
    scores = np.sort(np.asarray(abs_errors, dtype=float))
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    return float(np.quantile(scores, min(1.0, math.ceil((n + 1) * level) / n), method='higher'))


def rolling_conformal_coverage(wf, level=COVERAGE, min_history=60):
    """Hits and days for the walk-forward with each month's interval calibrated on earlier errors only."""
    abs_err = (wf['actual'] - wf['pred']).abs()
    hits = days = 0
    for month in sorted(set(wf.index.to_period('M'))):
        earlier = abs_err[abs_err.index < month.start_time]
        if len(earlier) < min_history:
            continue
        q = conformal_half_width(earlier, level)
        here = abs_err[wf.index.to_period('M') == month]
        hits += int((here <= q).sum())
        days += len(here)
    return hits, days


def kupiec_p(hits, n, coverage=COVERAGE):
    """Kupiec's proportion-of-failures test p-value (chi-square, one degree of freedom)."""
    if n == 0:
        return float('nan')
    x, p0 = n - hits, 1 - coverage
    pi = x / n

    def loglik(p):
        return (x * math.log(p) if x else 0.0) + ((n - x) * math.log(1 - p) if n - x else 0.0)

    lr = -2 * (loglik(p0) - (loglik(pi) if 0 < pi < 1 else 0.0))
    return math.erfc(math.sqrt(max(lr, 0.0) / 2))


def check(name, value, limit, passed, detail=''):
    return {'check': name, 'value': None if value is None else float(value), 'limit': float(limit), 'passed': bool(passed),
            'detail': detail}


def gate(wf, n_training_days, challenger_latest, champion_latest):
    """The promotion gate. wf: walk-forward actual and pred. challenger_latest, champion_latest: both models'
    predictions for the same latest days. Returns (passed, checks)."""
    err = wf['actual'] - wf['pred']
    recent = err[err.index > err.index.max() - pd.Timedelta(days=RECENT_DAYS)]
    hits, days = rolling_conformal_coverage(wf)
    p = kupiec_p(hits, days)
    diff = (np.asarray(challenger_latest, dtype=float) - np.asarray(champion_latest, dtype=float))
    out = np.asarray(challenger_latest, dtype=float)
    checks = [
        check('training_days', n_training_days, MIN_TRAINING_DAYS, n_training_days >= MIN_TRAINING_DAYS),
        check('walk_forward_mae', err.abs().mean(), VALIDATED_MAE_UPPER, err.abs().mean() <= VALIDATED_MAE_UPPER,
              f'{len(err)} out-of-sample days since {WALK_START:%Y-%m-%d}'),
        check('recent_mae', recent.abs().mean(), RECENT_MAE_MAX, recent.abs().mean() <= RECENT_MAE_MAX, f'last {len(recent)} days'),
        check('recent_bias', abs(recent.mean()), RECENT_BIAS_MAX, abs(recent.mean()) <= RECENT_BIAS_MAX,
              f'mean error {recent.mean():+.2f} (positive: under-predicting)'),
        check('coverage', p, KUPIEC_P_MIN, p >= KUPIEC_P_MIN, f'{hits}/{days} = {hits / days:.1%} inside; Kupiec p = {p:.3f}' if days else ''),
        check('agreement', np.abs(diff).mean(), AGREEMENT_MEAN_MAX,
              np.abs(diff).mean() <= AGREEMENT_MEAN_MAX and np.abs(diff).max() <= AGREEMENT_MAX,
              f'mean |difference| {np.abs(diff).mean():.2f}, largest {np.abs(diff).max():.2f} (limit {AGREEMENT_MAX}) over {len(diff)} days'),
        check('valid_output', int((~np.isfinite(out) | (out < 0) | (out > 100)).sum()), 0,
              bool(np.isfinite(out).all() and (out >= 0).all() and (out <= 100).all()), 'predictions outside 0-100 or not finite'),
    ]
    return all(c['passed'] for c in checks), checks
