"""Model monitoring for the day-ahead renewable-share model: the statistics and the rules that grade them.

Plain pandas and numpy, so the monitoring notebook and the unit tests share one implementation. Every check
returns rows of one shape (see ROW_COLUMNS), which the notebook writes to pulse.ml.monitoring. Monitoring only
records: it never fails a job (the owner's choice). A status of 'alert' is for a person to look at.

Four checks:
- accuracy          rolling MAE, bias and interval coverage against the plan in the validation report; Kupiec test
- feature_drift     PSI and two-sample KS of each input: recent served days against the same time of year in the
                    training period
- prediction_drift  the same for the predictions, against the model's own predictions over those training days
- freshness         per source, the share of its features the model received at the 18:00 cutoff on the latest day,
                    against the share it usually receives
"""
import math

import numpy as np
import pandas as pd

# The validated out-of-sample MAE of the champion (docs/model_validation_report.md), and the interval's coverage
BENCHMARK_MAE = 3.30
COVERAGE = 0.80
WINDOW_DAYS = 30
MIN_DAYS = {'accuracy': 14, 'feature_drift': 14, 'prediction_drift': 14, 'freshness': 1}
SEASON_MARGIN_DAYS = 15                   # drift reference: the same time of year, +/- this many days
DETERMINISTIC_PREFIXES = ('cal_', 'hol_')  # calendar and bank-holiday features: known in advance, can't drift

# Thresholds: (warn, alert). Accuracy ones follow the monitoring plan in docs/model_validation_report.md, section 10
MAE_POINTS = (4.0, 4.5)       # rolling 30-day MAE; 4.5 is beyond the worst month in validation
BIAS_POINTS = (1.5, 2.0)      # |mean error|, points of share
COVERAGE_BAND = ((0.70, 0.90), (0.65, 0.92))  # rolling 30-day coverage of the 80% interval: warn / alert outside
KUPIEC_P = (0.10, 0.05)       # coverage test p-value: lower is worse
PSI = (0.10, 0.25)            # the industry rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift
BLOCK_DAYS = 7                # drift noise is resampled in runs of a week of consecutive days
PRESENT = (0.90, 0.50)        # share of a source's features present: lower is worse

ROW_COLUMNS = ['check_name', 'subject', 'metric', 'value', 'warn_at', 'alert_at', 'status', 'n', 'window_start', 'window_end', 'details']


def grade(value, thresholds, higher_is_worse=True):
    """'ok', 'warn' or 'alert' for a value against (warn, alert) thresholds; 'no_data' for a missing value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'no_data'
    warn, alert = thresholds
    if higher_is_worse:
        return 'alert' if value >= alert else 'warn' if value >= warn else 'ok'
    return 'alert' if value <= alert else 'warn' if value <= warn else 'ok'


def row(check, subject, metric, value, thresholds=None, higher_is_worse=True, n=0, window=(None, None), details='', min_n=1):
    status = 'insufficient_data' if n < min_n else grade(value, thresholds, higher_is_worse) if thresholds else 'info'
    warn, alert = thresholds if thresholds else (None, None)
    value = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
    return dict(zip(ROW_COLUMNS, [check, subject, metric, value, warn, alert, status, int(n), window[0], window[1], details]))


# --- Statistics --------------------------------------------------------------------------------------------------

def psi(reference, current, bins=10):
    """Population stability index of current against reference, on the reference's quantile bins.

    Missing values get a bin of their own, so a source that stops arriving shows up as drift. Empty bins are
    floored at a small share so the logarithm stays finite.
    """
    reference, current = pd.Series(reference, dtype=float), pd.Series(current, dtype=float)
    ref_values = reference.dropna()
    if ref_values.empty or current.empty:
        return float('nan')
    edges = np.unique(np.quantile(ref_values, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(s):
        counts = np.histogram(s.dropna(), bins=edges)[0] if len(edges) > 1 else np.array([s.notna().sum()])
        counts = np.append(counts, s.isna().sum())
        return np.clip(counts / len(s), 1e-4, None)

    r, c = shares(reference), shares(current)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_thresholds(reference, n, bins=10, block=BLOCK_DAYS, reps=200, seed=0):
    """{'psi': (warn, alert), 'ks_stat': (warn, alert)} for n recent days compared with this reference.

    Both statistics are large by chance alone when n is small: PSI is about (bins - 1) * (1/n + 1/N) with no real
    change, so 30 days on 10 bins score about 0.3, above the textbook 0.25. And consecutive days are alike (weather
    lasts; a 28-day rolling mean over 30 days is nearly one number), so treating them as independent understates
    the noise further. The thresholds are therefore the 95th and 99th percentiles of each statistic for n days
    resampled from the reference in runs of `block` consecutive days (a moving-block bootstrap): "more different
    than a typical stretch of the same season last year". PSI never goes below the textbook (0.10, 0.25).
    Resampling the reference leaves out its own sampling error, so the real false-alarm rate is a few per cent,
    not 1%: with ~118 features, expect a handful of alerts on any day. Read drift as a pattern, not one feature.
    """
    reference = pd.Series(reference, dtype=float).sort_index().reset_index(drop=True)
    if n == 0 or reference.dropna().empty:
        return {'psi': PSI, 'ks_stat': (float('nan'), float('nan'))}
    rng = np.random.default_rng(seed)
    block = max(1, min(block, len(reference)))
    starts = np.arange(len(reference) - block + 1)
    null_psi, null_ks = [], []
    for _ in range(reps):
        picks = rng.choice(starts, int(np.ceil(n / block)))
        sample = reference.iloc[np.concatenate([np.arange(s, s + block) for s in picks])[:n]]
        null_psi.append(psi(reference, sample, bins))
        null_ks.append(ks_2samp(reference, sample)[0])
    return {'psi': (max(PSI[0], float(np.quantile(null_psi, 0.95))), max(PSI[1], float(np.quantile(null_psi, 0.99)))),
            'ks_stat': (float(np.nanquantile(null_ks, 0.95)), float(np.nanquantile(null_ks, 0.99)))}


def ks_2samp(reference, current):
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value, ignoring missing values (no SciPy needed)."""
    a, b = (np.sort(pd.Series(x, dtype=float).dropna().to_numpy()) for x in (reference, current))
    if len(a) == 0 or len(b) == 0:
        return float('nan'), float('nan')
    values = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, values, side='right') / len(a) - np.searchsorted(b, values, side='right') / len(b))))
    en = math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.2:  # the series below converges badly near zero, where the p-value is 1 to many decimals
        return d, 1.0
    # Kolmogorov distribution tail: 2 * sum (-1)^(k-1) exp(-2 k^2 lambda^2)
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def kupiec(hits, n, coverage=COVERAGE):
    """Kupiec's proportion-of-failures test: is the observed coverage consistent with the nominal one? (LR, p)."""
    if n == 0:
        return float('nan'), float('nan')
    x = n - hits  # misses
    pi_hat = x / n
    p0 = 1 - coverage

    def loglik(p):
        return (x * math.log(p) if x else 0.0) + ((n - x) * math.log(1 - p) if n - x else 0.0)

    lr = -2 * (loglik(p0) - (loglik(pi_hat) if 0 < pi_hat < 1 else 0.0))
    return lr, math.erfc(math.sqrt(max(lr, 0.0) / 2))  # chi-square with one degree of freedom


# --- Checks ------------------------------------------------------------------------------------------------------

def _window(index):
    return (min(index), max(index)) if len(index) else (None, None)


def grade_band(value, band=COVERAGE_BAND):
    """Grade a value that must stay inside a band (too few interval hits and too many are both wrong)."""
    (warn_lo, warn_hi), (alert_lo, alert_hi) = band
    return 'alert' if not alert_lo <= value <= alert_hi else 'warn' if not warn_lo <= value <= warn_hi else 'ok'


def accuracy(scored, segment='all'):
    """scored: one row per day with actual, predicted_share, lower_80, upper_80 (index = target_date), the most
    recent WINDOW_DAYS days with a known outcome. Returns MAE, bias and coverage rows."""
    scored = scored.dropna(subset=['actual'])
    n, window = len(scored), _window(scored.index)
    err = scored['actual'] - scored['predicted_share']
    hits = int(((scored['actual'] >= scored['lower_80']) & (scored['actual'] <= scored['upper_80'])).sum())
    mae = float(err.abs().mean()) if n else float('nan')
    bias = float(err.mean()) if n else float('nan')
    _, p = kupiec(hits, n)
    common = dict(n=n, window=window, min_n=MIN_DAYS['accuracy'])
    (warn_lo, warn_hi), (alert_lo, alert_hi) = COVERAGE_BAND
    coverage = row('accuracy', segment, 'coverage_80', hits / n if n else float('nan'), **common,
                   details=f'{hits}/{n} inside; warn outside {warn_lo:.0%}-{warn_hi:.0%}, alert outside {alert_lo:.0%}-{alert_hi:.0%}')
    if coverage['status'] == 'info':  # enough days: grade it against the band
        coverage['status'] = grade_band(coverage['value'])
    return [
        row('accuracy', segment, 'mae', mae, MAE_POINTS, details=f'validated {BENCHMARK_MAE}', **common),
        row('accuracy', segment, 'bias', abs(bias) if n else float('nan'), BIAS_POINTS,
            details=f'mean error {bias:+.2f} (positive: under-predicting)' if n else '', **common),
        coverage,
        row('accuracy', segment, 'kupiec_p', p, KUPIEC_P, higher_is_worse=False, **common),
    ]


def seasonal_reference(reference, current_index, margin_days=SEASON_MARGIN_DAYS):
    """The reference days within margin_days (by day of year, wrapping at New Year) of any current day.

    A 30-day window always differs from a whole year (October is windier than the annual mix, and day of year
    "drifts" by definition), so drift is judged against the same time of year, not the whole year.
    """
    if len(current_index) == 0 or len(reference) == 0:
        return reference.iloc[:0]
    ref_doy = pd.DatetimeIndex(reference.index).day_of_year.to_numpy()[:, None]
    cur_doy = np.unique(pd.DatetimeIndex(current_index).day_of_year.to_numpy())[None, :]
    gap = np.abs(ref_doy - cur_doy)
    gap = np.minimum(gap, 365 - gap).min(axis=1)
    return reference[gap <= margin_days]


def feature_drift(reference, current, features):
    """reference, current: one row per day, one column per feature. PSI and KS per feature, except the calendar
    and holiday features, which are known in advance and can't drift."""
    rows, window = [], _window(current.index)
    for f in features:
        if f.startswith(DETERMINISTIC_PREFIXES):
            continue
        ref, cur = reference[f] if f in reference else pd.Series(dtype=float), current[f] if f in current else pd.Series(dtype=float)
        common = dict(n=len(cur), window=window, min_n=MIN_DAYS['feature_drift'])
        d, p = ks_2samp(ref, cur)
        limits = drift_thresholds(ref, len(cur))
        missing = f'missing {cur.isna().mean():.0%} now vs {ref.isna().mean():.0%} in training' if len(cur) and len(ref) else ''
        rows.append(row('feature_drift', f, 'psi', psi(ref, cur), limits['psi'], details=missing, **common))
        rows.append(row('feature_drift', f, 'ks_stat', d, limits['ks_stat'], details=f'textbook KS p = {p:.3g} (assumes independent days)', **common))
    return rows


def prediction_drift(reference_predictions, current_predictions):
    cur = pd.Series(current_predictions, dtype=float)
    ref = pd.Series(reference_predictions, dtype=float)
    common = dict(n=len(cur), window=_window(cur.index), min_n=MIN_DAYS['prediction_drift'])
    d, p = ks_2samp(ref, cur)
    limits = drift_thresholds(ref, len(cur))
    details = f'mean {cur.mean():.1f} now vs {ref.mean():.1f} in training' if len(cur) else ''
    return [row('prediction_drift', 'predicted_share', 'psi', psi(ref, cur), limits['psi'], details=details, **common),
            row('prediction_drift', 'predicted_share', 'ks_stat', d, limits['ks_stat'], details=f'textbook KS p = {p:.3g}', **common)]


def freshness(served, source_of):
    """Did each source arrive in time for the latest scored day, as it usually does?

    served: one row per scored day (oldest first), one column per feature, exactly as the model received it.
    source_of: feature -> source table. The value is the share of a source's features present on the latest day,
    as a fraction of the share usually present (the median over the served days). Usual matters: NESO publishes
    weeks behind, so some of its features are always missing at the cutoff, and that is not news.
    """
    rows, window = [], _window(served.index[-1:])
    by_source = {}
    for feature, source in source_of.items():
        if feature in served:
            by_source.setdefault(source, []).append(feature)
    for source, cols in sorted(by_source.items()):
        daily = served[cols].notna().mean(axis=1)
        latest = float(daily.iloc[-1]) if len(daily) else float('nan')
        usual = float(daily.median()) if len(daily) else float('nan')
        relative = latest / usual if usual > 0 else (1.0 if latest == usual else float('nan'))
        rows.append(row('freshness', source, 'share_present_vs_usual', relative, PRESENT, higher_is_worse=False, n=len(daily),
                        window=window, details=f'{len(cols)} features; {latest:.0%} present on the latest day, usually {usual:.0%}',
                        min_n=MIN_DAYS['freshness']))
    return rows


def weather_quality(features):
    """The scoring job's readiness flag, per day: 'ok', 'weather_partial' (some weather inputs missing) or
    'weather_missing' (all of them; the error roughly doubles, so the interval is widened)."""
    wx = [c for c in features.columns if c.startswith('wx_')]
    if not wx:
        return pd.Series('ok', index=features.index)
    present = features[wx].notna()
    return pd.Series(np.where(~present.any(axis=1), 'weather_missing', np.where(~present.all(axis=1), 'weather_partial', 'ok')),
                     index=features.index)


def summary(rows):
    """The headline: the worst status per check, and how many subjects are at each status."""
    frame = pd.DataFrame(rows, columns=ROW_COLUMNS)
    order = {'alert': 0, 'warn': 1, 'ok': 2, 'info': 3, 'insufficient_data': 4, 'no_data': 5}
    out = {}
    for check, g in frame.groupby('check_name'):
        counts = g['status'].value_counts().to_dict()
        out[check] = {'worst': min(counts, key=order.get), **counts}
    return out
