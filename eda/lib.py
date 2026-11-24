"""Helpers for the EDA: loading, column profiles, completeness, and chart specs (plain dicts the page draws)."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'eda' / 'out'
SILVER = OUT / 'silver'
UK = 'Europe/London'
LINEAGE = {'fetched_at', 'source_file', 'record_index', 'kafka_partition', 'kafka_offset', 'ingested_at'}
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
PERIOD_LABELS = [f'{(p - 1) // 2:02d}:{"30" if (p - 1) % 2 else "00"}' for p in range(1, 49)]


def load(table):
    df = pd.read_parquet(SILVER / f'{table}.parquet')
    for c in df.columns:
        if c.endswith('_date'):
            df[c] = pd.to_datetime(df[c])
    return df


def num(x, digits=2):
    """A JSON-safe rounded number: NaN and infinity become None."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, digits)


def nums(values, digits=2):
    return [num(v, digits) for v in values]


def sanitize(o):
    """Make a nested structure JSON-safe."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return num(o, 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime('%Y-%m-%d %H:%M')
    if o is pd.NaT:
        return None
    return o


def days(index):
    return [d.strftime('%Y-%m-%d') for d in index]


# --- UK settlement time -------------------------------------------------------------------------------------

def period_start(dates, periods):
    """The UTC instant each settlement period starts."""
    local = pd.to_datetime(dates).dt.tz_localize(UK)
    return (local.dt.tz_convert('UTC') + pd.to_timedelta((periods - 1) * 30, unit='m')).dt.tz_localize(None)


def periods_expected(dates):
    """Settlement periods in each UK day: 46, 48 or 50."""
    d = pd.DatetimeIndex(dates)
    a, b = d.tz_localize(UK), (d + pd.Timedelta(days=1)).tz_localize(UK)
    return pd.Series(((b - a) / pd.Timedelta(minutes=30)).astype(int), index=d)


def uk_local(ts):
    """UTC-naive timestamps -> UK local (naive)."""
    return ts.dt.tz_localize('UTC').dt.tz_convert(UK).dt.tz_localize(None)


def completeness(df, date='settlement_date', period='settlement_period', group=None):
    """Checks for a table with one row per settlement period: gaps, impossible periods, clock-change days."""
    out = []
    groups = [(None, df)] if group is None else list(df.groupby(group))
    for name, g in groups:
        present = g.groupby(date)[period].nunique()
        expected = periods_expected(present.index)
        missing = (expected - present).clip(lower=0)
        span = pd.date_range(present.index.min(), present.index.max())
        label = '' if name is None else f' ({name})'
        out.append(check(f'Days with every period present{label}', int((missing == 0).sum()), len(present),
                         'good' if (missing == 0).all() else 'warn',
                         f'{int(missing.sum()):,} periods missing on {int((missing > 0).sum())} days'))
        absent = len(span) - len(present)
        out.append(check(f'Whole days absent inside the range{label}', absent, len(span),
                         'good' if absent == 0 else 'warn',
                         f'{present.index.min():%Y-%m-%d} to {present.index.max():%Y-%m-%d}', bad=True))
    exp_row = periods_expected(df[date].unique()).reindex(df[date]).to_numpy()
    bad = int(((df[period] < 1) | (df[period] > exp_row)).sum())
    out.append(check('Periods outside 1..46/48/50 for their day', bad, len(df), 'good' if bad == 0 else 'bad',
                     'the hard rule period_out_of_range', bad=True))
    if 'start_time' in df.columns:
        mismatch = int((df['start_time'] != period_start(df[date], df[period])).sum())
        out.append(check('start_time disagrees with date + period', mismatch, len(df),
                         'good' if mismatch == 0 else 'warn', 'the soft rule start_time_mismatch', bad=True))
    return out


def check(name, count, of, level, note='', bad=False):
    """One data-quality line. `bad` marks counts where zero is the good outcome."""
    return {'name': name, 'count': int(count), 'of': int(of), 'level': level, 'note': note, 'bad': bad}


def range_check(df, column, low, high, rule):
    s = df[column].dropna()
    n = int(((s < low) | (s > high)).sum())
    return check(f'{column} outside [{low:,}, {high:,}]', n, len(s), 'good' if n == 0 else 'bad',
                 f'the hard rule {rule}', bad=True)


# --- Column profile -----------------------------------------------------------------------------------------

def profile(df):
    numeric, other = [], []
    n = len(df)
    for c in df.columns:
        if c in LINEAGE:
            continue
        s = df[c]
        base = {'column': c, 'non_null_pct': num(100 * s.notna().mean(), 2), 'nulls': int(s.isna().sum()),
                'distinct': int(s.nunique())}
        if pd.api.types.is_bool_dtype(s) or s.dropna().isin([True, False]).all() and s.dtype == object:
            v = s.dropna().astype(bool)
            other.append({**base, 'type': 'boolean', 'detail': f'true {100 * v.mean():.1f}%' if len(v) else ''})
        elif pd.api.types.is_numeric_dtype(s):
            v = s.dropna().astype(float)
            if not len(v):
                other.append({**base, 'type': 'number', 'detail': 'always NULL'})
                continue
            q = v.quantile([.01, .05, .25, .5, .75, .95, .99]).to_numpy()
            iqr = q[4] - q[2]
            outliers = ((v < q[2] - 1.5 * iqr) | (v > q[4] + 1.5 * iqr)).mean() if iqr > 0 else 0
            numeric.append({**base, 'min': num(v.min()), 'p01': num(q[0]), 'p05': num(q[1]), 'p25': num(q[2]),
                            'median': num(q[3]), 'mean': num(v.mean()), 'p75': num(q[4]), 'p95': num(q[5]),
                            'p99': num(q[6]), 'max': num(v.max()), 'std': num(v.std()),
                            'cv': num(v.std() / v.mean(), 3) if v.mean() else None,
                            'skew': num(v.skew(), 2), 'kurtosis': num(v.kurt(), 2),
                            'zero_pct': num(100 * (v == 0).mean(), 2), 'negative_pct': num(100 * (v < 0).mean(), 2),
                            'outlier_pct': num(100 * outliers, 2)})
        elif pd.api.types.is_datetime64_any_dtype(s):
            v = s.dropna()
            fmt = '%Y-%m-%d' if c.endswith('_date') else '%Y-%m-%d %H:%M'
            other.append({**base, 'type': 'date' if c.endswith('_date') else 'timestamp',
                          'detail': f'{v.min():{fmt}} to {v.max():{fmt}}' if len(v) else 'always NULL'})
        else:
            counts = s.value_counts()
            top = ', '.join(f'{str(k)[:28]} ({100 * v / n:.1f}%)' for k, v in counts.head(3).items())
            other.append({**base, 'type': 'text', 'detail': top})
    return {'numeric': numeric, 'other': other}


# --- Chart specs --------------------------------------------------------------------------------------------

def line(title, sub, x, series, unit='', area=False, stack=False, digits=1, wide=False):
    return {'t': 'line', 'title': title, 'sub': sub, 'x': list(x), 'unit': unit, 'area': area, 'stack': stack,
            'wide': wide, 'series': [{'name': n, 'data': nums(d, digits)} for n, d in series]}


def bar(title, sub, x, series, unit='', horizontal=False, stack=False, digits=1, wide=False):
    return {'t': 'bar', 'title': title, 'sub': sub, 'x': [str(v) for v in x], 'unit': unit, 'stack': stack,
            'horizontal': horizontal, 'wide': wide,
            'series': [{'name': n, 'data': nums(d, digits)} for n, d in series]}


def hist(title, sub, values, unit='', bins=50, clip=(0.005, 0.995), digits=1):
    v = pd.Series(values).dropna().astype(float)
    lo, hi = (v.quantile(clip[0]), v.quantile(clip[1])) if clip else (v.min(), v.max())
    if lo == hi:
        lo, hi = v.min(), v.max()
    inside = v[(v >= lo) & (v <= hi)]
    counts, edges = np.histogram(inside, bins=bins, range=(lo, hi))
    clipped = len(v) - len(inside)
    if clipped:
        sub = f'{sub} {clipped:,} values beyond the {clip[0]:.1%}–{clip[1]:.1%} quantiles are not drawn.'.strip()
    return {'t': 'hist', 'title': title, 'sub': sub, 'unit': unit, 'edges': nums(edges, digits),
            'counts': [int(c) for c in counts], 'mean': num(v.mean(), digits), 'median': num(v.median(), digits)}


def heat(title, sub, x, y, z, unit='', diverging=False, digits=1, wide=False):
    return {'t': 'heat', 'title': title, 'sub': sub, 'x': [str(v) for v in x], 'y': [str(v) for v in y],
            'z': [nums(row, digits) for row in z], 'unit': unit, 'diverging': diverging, 'wide': wide}


def box(title, sub, groups, unit='', digits=1, wide=False):
    """groups: {name: values}. Whiskers are the 1st and 99th percentiles, so rare extremes don't set the scale."""
    names, stats = [], []
    for name, values in groups.items():
        v = pd.Series(values).dropna().astype(float)
        if len(v):
            names.append(str(name))
            stats.append(nums(v.quantile([.01, .25, .5, .75, .99]).to_numpy(), digits))
    return {'t': 'box', 'title': title, 'sub': f'{sub} Box: quartiles; whiskers: 1st to 99th percentile.'.strip(),
            'x': names, 'stats': stats, 'unit': unit, 'wide': wide}


def scatter(title, sub, x, y, x_label, y_label, sample=1500, digits=2, seed=7):
    d = pd.DataFrame({'x': np.asarray(x, dtype=float), 'y': np.asarray(y, dtype=float)}).dropna()
    r = d['x'].corr(d['y']) if len(d) > 2 else float('nan')
    rho = d['x'].corr(d['y'], method='spearman') if len(d) > 2 else float('nan')
    slope, intercept = np.polyfit(d['x'], d['y'], 1) if len(d) > 2 else (float('nan'), float('nan'))
    shown = d.sample(sample, random_state=seed) if len(d) > sample else d
    drawn = f' {len(shown):,} of {len(d):,} points drawn.' if len(d) > sample else f' {len(d):,} points.'
    return {'t': 'scatter', 'title': title, 'sub': f'{sub} Pearson r = {r:.2f}, Spearman ρ = {rho:.2f}.{drawn}',
            'points': [[num(a, digits), num(b, digits)] for a, b in zip(shown['x'], shown['y'])],
            'xLabel': x_label, 'yLabel': y_label,
            'fit': {'slope': num(slope, 6), 'intercept': num(intercept, 4), 'r': num(r, 3),
                    'x0': num(d['x'].min(), digits), 'x1': num(d['x'].max(), digits)}}


def month_period_heat(df, value, title, sub, unit, period='settlement_period', month=None, digits=0):
    """Mean of `value` by month of year (rows) and settlement period (columns, UK local time)."""
    m = df[month] if month else df['settlement_date'].dt.month
    grid = df[df[period].between(1, 48)].groupby([m, period])[value].mean().unstack()
    grid = grid.reindex(index=range(1, 13), columns=range(1, 49))
    return heat(title, sub, PERIOD_LABELS, MONTHS, grid.to_numpy(), unit, digits=digits, wide=True)


def month_hour_heat(local_time, values, title, sub, unit, digits=0):
    """Mean by month of year (rows) and UK local hour (columns), for hourly or half-hourly UTC data."""
    d = pd.DataFrame({'m': local_time.dt.month, 'h': local_time.dt.hour, 'v': values})
    grid = d.groupby(['m', 'h'])['v'].mean().unstack().reindex(index=range(1, 13), columns=range(24))
    return heat(title, sub, [f'{h:02d}:00' for h in range(24)], MONTHS, grid.to_numpy(), unit, digits=digits, wide=True)


def corr_heat(frame, title, sub, digits=2):
    c = frame.corr()
    return heat(title, sub, c.columns, c.index, c.to_numpy(), '', diverging=True, digits=digits)


def error_stats(forecast, actual):
    """Forecast accuracy: n, bias (forecast - actual), MAE, RMSE, MAE as a share of the mean actual, r."""
    d = pd.DataFrame({'f': forecast, 'a': actual}).dropna()
    if not len(d):
        return {'n': 0, 'bias': None, 'mae': None, 'cmae': None, 'rmse': None, 'nmae_pct': None, 'r': None}
    e = d['f'] - d['a']
    mean = d['a'].abs().mean()
    return {'n': len(d), 'bias': num(e.mean()), 'mae': num(e.abs().mean()), 'cmae': num((e - e.mean()).abs().mean()),
            'rmse': num(math.sqrt((e ** 2).mean())),
            'nmae_pct': num(100 * e.abs().mean() / mean) if mean else None, 'r': num(d['f'].corr(d['a']), 3)}


def table(title, columns, rows, note=''):
    return {'title': title, 'cols': list(columns), 'rows': [list(r) for r in rows], 'note': note}


def tile(label, value, note=''):
    return {'label': label, 'value': value, 'note': note}


def fmt_int(n):
    return f'{int(n):,}'


class GradientBoosting:
    """HistGradientBoostingRegressor that ignores columns with no values in the training rows (it cannot bin them)."""

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.columns = [c for c in X.columns if X[c].notna().any()]
        self.model = HistGradientBoostingRegressor(**self.params).fit(X[self.columns], y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.columns])
