"""Inferential statistics for the 15 silver tables: intervals, hypothesis tests, forecast evaluation, regressions.

Half-hourly values are strongly autocorrelated, so 50,000 of them are nowhere near 50,000 independent observations.
Tests therefore run on daily values, and wherever a mean or a difference is tested the standard error is
Newey-West (HAC) or the interval comes from a 14-day moving-block bootstrap. Tests that still assume
independence (rank tests, ANOVA, chi-square on counts) are marked "assumes independence": read their p-values
as optimistic and look at the effect size.

Reads eda/out/silver, writes eda/out/inference.json and eda/out/inference_digest.txt.

    python eda/inference.py
"""
import json
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests, kpss

from lib import DOW, MONTHS, OUT, load, num, period_start, periods_expected, sanitize, table

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(7)
COLS = ['Test', 'Question', 'Statistic', 'p-value', 'Estimate', '95% interval', 'Reading']
EST_COLS = ['Series', 'Days', 'Mean', 'Naive 95% interval', 'Robust 95% interval', 'Median', 'Median 95% interval',
            'Lag-1 autocorrelation', 'Effective days']
INDEP = ' Assumes independence.'
RESULT, PVALUES, DIGEST, CURRENT = {}, [], [], ['']


# --- Formatting ---------------------------------------------------------------------------------------------

def f(x, digits=2):
    x = num(x, 6)
    if x is None:
        return '–'
    a = abs(x)
    d = digits if a < 1000 else 0
    return f'{x:,.{d}f}'


def pf(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return '–'
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def ci(lo, hi, digits=2, unit=''):
    return f'{f(lo, digits)} to {f(hi, digits)}{unit}'


def row(test, question, stat, p, estimate='–', interval='–', reading='', bounded=None):
    """One test result. `bounded` replaces the p-value text for tests that only report a range (KPSS)."""
    if p is not None and not (isinstance(p, float) and math.isnan(p)):
        PVALUES.append((CURRENT[0], test, question, float(p)))
    return [test, question, stat, bounded or pf(p), estimate, interval, reading]


def verdict(p, yes, no, alpha=0.05):
    return yes if (p is not None and p < alpha) else no


# --- Estimators ---------------------------------------------------------------------------------------------

def lags(n):
    return max(1, int(round(n ** (1 / 3))))


def hac_ols(y, X):
    return sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': lags(len(y))})


def hac_mean(x):
    """Mean of a daily series with a Newey-West interval and the p-value of 'mean = 0'."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    m = hac_ols(x, np.ones(len(x)))
    lo, hi = m.conf_int()[0]
    return {'n': len(x), 'mean': m.params[0], 'lo': lo, 'hi': hi, 'p': m.pvalues[0], 't': m.tvalues[0]}


def block_bootstrap(data, fn, block=14, reps=1000):
    """Percentile interval of fn over moving-block resamples of the rows of data."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    blocks = math.ceil(n / block)
    starts = RNG.integers(0, n - block + 1, size=(reps, blocks))
    values = []
    for s in starts:
        idx = (s[:, None] + np.arange(block)).ravel()[:n]
        values.append(fn(data[idx]))
    return np.nanpercentile(values, [2.5, 97.5])


def corr_ci(a, b):
    d = pd.DataFrame({'a': a, 'b': b}).dropna()
    r = d['a'].corr(d['b'])
    lo, hi = block_bootstrap(d.to_numpy(), lambda x: np.corrcoef(x[:, 0], x[:, 1])[0, 1])
    return r, lo, hi, len(d)


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled if pooled else float('nan')


def effect_word(d):
    d = abs(d)
    return 'negligible' if d < 0.2 else 'small' if d < 0.5 else 'medium' if d < 0.8 else 'large'


def cramers_v(tab):
    chi2, p, dof, _ = st.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    v = math.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    return chi2, p, dof, v


def harmonics(index, pairs=2):
    t = (index - index[0]).days.to_numpy() / 365.25
    X = {'years': t}
    for k in range(1, pairs + 1):
        X[f'sin{k}'] = np.sin(2 * np.pi * k * t)
        X[f'cos{k}'] = np.cos(2 * np.pi * k * t)
    return pd.DataFrame(X, index=index)


def ci_chart(title, sub, labels, est, lo, hi, unit='', ref=None, wide=False, digits=2):
    return {'t': 'ci', 'title': title, 'sub': sub, 'x': [str(v) for v in labels], 'est': [num(v, digits) for v in est],
            'lo': [num(v, digits) for v in lo], 'hi': [num(v, digits) for v in hi], 'unit': unit, 'ref': ref, 'wide': wide}


# --- The battery every daily series gets ----------------------------------------------------------------------

def estimates(series, label, digits=1):
    s = series.dropna()
    x = s.to_numpy(dtype=float)
    n = len(x)
    se = x.std(ddof=1) / math.sqrt(n)
    h = hac_mean(x)
    med = block_bootstrap(x, np.median)
    r1 = pd.Series(x).autocorr(1)
    # How many independent days carry the same information about the mean
    n_eff = n * (se / ((h['hi'] - h['lo']) / (2 * 1.96))) ** 2
    return [label, f'{n:,}', f(x.mean(), digits), ci(x.mean() - 1.96 * se, x.mean() + 1.96 * se, digits),
            ci(h['lo'], h['hi'], digits), f(np.median(x), digits), ci(med[0], med[1], digits), f(r1, 2), f'{min(n, n_eff):,.0f}']


def battery(series, label, unit, digits=1):
    """Distribution, time-series and calendar tests for one daily series."""
    s = series.dropna()
    x = s.to_numpy(dtype=float)
    rows = []
    jb = st.jarque_bera(x)
    rows.append(row('Jarque–Bera', f'Is {label} normally distributed?', f'JB = {f(jb.statistic, 1)}', jb.pvalue,
                    f'skew {f(st.skew(x))}, excess kurtosis {f(st.kurtosis(x))}', '–',
                    verdict(jb.pvalue, 'Not normal.', 'Consistent with normal.') + INDEP))
    sw = st.shapiro(x[:5000])
    rows.append(row('Shapiro–Wilk', f'Is {label} normally distributed?', f'W = {f(sw.statistic, 4)}', sw.pvalue, '–', '–',
                    verdict(sw.pvalue, 'Not normal.', 'Consistent with normal.') + INDEP))
    ad = adfuller(x, autolag='AIC')
    rows.append(row('Augmented Dickey–Fuller', f'Does {label} have a unit root?', f'ADF = {f(ad[0])}', ad[1], f'{ad[2]} lags', '–',
                    verdict(ad[1], 'Unit root rejected: stationary.', 'Unit root not rejected.')))
    kp = kpss(x, regression='c', nlags='auto')
    bounded = '< 0.010' if kp[1] <= 0.01 else '> 0.100' if kp[1] >= 0.1 else None
    rows.append(row('KPSS', f'Is {label} stationary around a level?', f'KPSS = {f(kp[0], 3)}', kp[1], f'{kp[2]} lags', '–',
                    verdict(kp[1], 'Stationarity rejected.', 'Stationarity not rejected.'), bounded=bounded))
    lb = acorr_ljungbox(x, lags=[7, 30], return_df=True)
    for lag in (7, 30):
        rows.append(row('Ljung–Box', f'Is {label} autocorrelated up to {lag} days?', f'Q = {f(lb.loc[lag, "lb_stat"], 0)}',
                        lb.loc[lag, 'lb_pvalue'], '–', '–', verdict(lb.loc[lag, 'lb_pvalue'], 'Autocorrelated.', 'No autocorrelation found.')))
    X = sm.add_constant(harmonics(s.index).assign(weekend=(s.index.dayofweek >= 5).astype(float)))
    m = hac_ols(s, X)
    arch = het_arch(m.resid, nlags=7)
    rows.append(row('Engle ARCH LM', f'Does the variance of {label} cluster in time?', f'LM = {f(arch[0], 1)}', arch[1], '7 lags', '–',
                    verdict(arch[1], 'Volatility clusters.', 'No clustering found.')))
    lo, hi = m.conf_int().loc['years']
    rows.append(row('Trend regression (HAC)', f'Does {label} drift, after the season and the week?', f't = {f(m.tvalues["years"])}',
                    m.pvalues['years'], f'{f(m.params["years"], digits)} {unit} per year', ci(lo, hi, digits),
                    verdict(m.pvalues['years'], 'A trend.', 'No trend distinguishable.')))
    lo, hi = m.conf_int().loc['weekend']
    weekend, weekday = x[s.index.dayofweek >= 5], x[s.index.dayofweek < 5]
    d = cohen_d(weekend, weekday)
    rows.append(row('Weekend effect (HAC regression)', f'Do weekends differ from weekdays in {label}?', f't = {f(m.tvalues["weekend"])}',
                    m.pvalues['weekend'], f'{f(m.params["weekend"], digits)} {unit}; Cohen d {f(d)}', ci(lo, hi, digits),
                    verdict(m.pvalues['weekend'], f'Weekends differ ({effect_word(d)} effect).', 'No weekend effect distinguishable.')))
    mw = st.mannwhitneyu(weekend, weekday)
    rows.append(row('Mann–Whitney U', f'Do weekends and weekdays share a distribution of {label}?', f'U = {f(mw.statistic, 0)}', mw.pvalue,
                    f'rank-biserial {f(1 - 2 * mw.statistic / (len(weekend) * len(weekday)))}', '–',
                    verdict(mw.pvalue, 'They differ.', 'No difference found.') + INDEP))
    groups = [x[s.index.month == mth] for mth in range(1, 13)]
    kw = st.kruskal(*groups)
    an = st.f_oneway(*groups)
    eta = sum(len(g) * (g.mean() - x.mean()) ** 2 for g in groups) / ((x - x.mean()) ** 2).sum()
    rows.append(row('Kruskal–Wallis', f'Does {label} differ between months of the year?', f'H = {f(kw.statistic, 1)}', kw.pvalue,
                    f'η² = {f(eta)} (ANOVA F = {f(an.statistic, 1)})', '–',
                    verdict(kw.pvalue, f'Months differ; they explain {100 * eta:.0f}% of the variance.', 'No monthly pattern found.') + INDEP))
    a, b = s[s.index.year == 2024], s[s.index.year == 2025]
    if len(a) > 300 and len(b) > 300:
        a.index, b.index = a.index.strftime('%m-%d'), b.index.strftime('%m-%d')
        diff = (b - a).dropna()
        h = hac_mean(diff)
        rows.append(row('Paired year test (HAC)', f'Did {label} change from 2024 to 2025, same calendar day?', f't = {f(h["t"])}', h['p'],
                        f'{f(h["mean"], digits)} {unit}', ci(h['lo'], h['hi'], digits),
                        verdict(h['p'], '2025 differs from 2024.', 'No change distinguishable.')))
        ks = st.ks_2samp(a.to_numpy(), b.to_numpy())
        rows.append(row('Kolmogorov–Smirnov', f'Do 2024 and 2025 share a distribution of {label}?', f'D = {f(ks.statistic, 3)}', ks.pvalue,
                        '–', '–', verdict(ks.pvalue, 'The distributions differ.', 'No difference found.') + INDEP))
        lev = st.levene(a.to_numpy(), b.to_numpy())
        rows.append(row('Levene', f'Is {label} equally variable in 2024 and 2025?', f'W = {f(lev.statistic)}', lev.pvalue,
                        f'std dev {f(a.std(), digits)} vs {f(b.std(), digits)}', '–',
                        verdict(lev.pvalue, 'The spread changed.', 'No change in spread found.') + INDEP))
    return rows


def group_ci(series, by, labels, title, sub, unit, digits=1):
    """Mean with a Newey-West interval for each group of days."""
    est, lo, hi = [], [], []
    for g in range(len(labels)):
        h = hac_mean(series[by == g + (1 if labels is MONTHS else 0)])
        est.append(h['mean'])
        lo.append(h['lo'])
        hi.append(h['hi'])
    return ci_chart(title, sub, labels, est, lo, hi, unit, digits=digits)


def calendar_charts(series, label, unit, digits=1):
    s = series.dropna()
    return [group_ci(s, s.index.dayofweek, DOW, f'{label} by day of week, with 95% intervals', 'Mean of daily values; Newey–West interval.', unit, digits),
            group_ci(s, s.index.month, MONTHS, f'{label} by month of year, with 95% intervals', 'Mean of daily values; Newey–West interval.', unit, digits)]


# --- Forecast evaluation --------------------------------------------------------------------------------------

def bias_row(daily_error, what, unit, digits=0):
    h = hac_mean(daily_error)
    return row('Bias test (HAC)', f'Is the mean error of {what} zero?', f't = {f(h["t"])}', h['p'], f'{f(h["mean"], digits)} {unit}',
               ci(h['lo'], h['hi'], digits), verdict(h['p'], 'Biased ' + ('high.' if h['mean'] > 0 else 'low.'), 'No bias distinguishable.')), h


def mincer_zarnowitz(forecast, actual, what):
    d = pd.DataFrame({'f': forecast, 'a': actual}).dropna()
    m = hac_ols(d['a'], sm.add_constant(d['f']))
    joint = m.f_test('const = 0, f = 1')
    lo, hi = m.conf_int().loc['f']
    return row('Mincer–Zarnowitz (HAC)', f'Is {what} efficient: actual = 0 + 1 × forecast?', f'F = {f(float(joint.fvalue), 1)}',
               float(joint.pvalue), f'slope {f(m.params["f"], 3)}, intercept {f(m.params["const"], 1)}', ci(lo, hi, 3),
               verdict(float(joint.pvalue), 'Not efficient: a linear correction would improve it.', 'Efficiency not rejected.'))


def diebold_mariano(loss_a, loss_b, name_a, name_b, unit, digits=0):
    """Daily mean absolute error of A minus B, tested against zero with a Newey-West error."""
    diff = (loss_a - loss_b).dropna()
    h = hac_mean(diff)
    better = name_a if h['mean'] < 0 else name_b
    better = better[0].upper() + better[1:]
    return row('Diebold–Mariano', f'Is {name_a} more accurate than {name_b}?', f'DM = {f(h["t"])}', h['p'],
               f'{f(h["mean"], digits)} {unit} (absolute error, {name_a} − {name_b})', ci(h['lo'], h['hi'], digits),
               verdict(h['p'], f'{better} is more accurate.', 'No difference distinguishable.'))


# --- Shared inputs --------------------------------------------------------------------------------------------

def generation():
    fuel = load('fuelhh')
    wide = fuel.pivot(index=['settlement_date', 'settlement_period'], columns='fuel_type', values='generation_mw')
    ints = [c for c in wide.columns if c.startswith('INT')]
    dates = wide.index.get_level_values(0)
    domestic = wide.drop(columns=ints).sum(axis=1)
    renewable = wide[['WIND', 'NPSHYD', 'BIOMASS']].sum(axis=1)
    present = wide['WIND'].groupby(dates).count()
    complete = present == periods_expected(present.index)
    daily = pd.DataFrame({'share': 100 * renewable.groupby(dates).sum() / domestic.groupby(dates).sum(),
                          'wind': wide['WIND'].groupby(dates).mean(), 'gas': (wide['CCGT'] + wide['OCGT']).groupby(dates).mean(),
                          'imports': wide[ints].sum(axis=1).groupby(dates).mean(), 'domestic': domestic.groupby(dates).mean()})[complete]
    daily['gas_share'] = 100 * daily['gas'] / daily['domestic']
    start = period_start(pd.Series(dates), pd.Series(wide.index.get_level_values(1))).to_numpy()
    half_wind = pd.Series(wide['WIND'].to_numpy(), index=start)
    importing = (wide[ints] > 0).where(wide[ints].notna()).groupby(dates).mean()
    return daily, half_wind, importing


def finish(table_id, tables, charts=()):
    RESULT[table_id] = {'tables': list(tables), 'charts': list(charts)}
    DIGEST.append(f'\n===== {table_id}')
    for t in tables:
        DIGEST.append(f"  -- {t['title']}")
        for r in t['rows']:
            DIGEST.append('     ' + ' | '.join(str(c) for c in r))


NOTE = ('Daily values. HAC: Newey–West standard errors, which allow for autocorrelation. Rows marked "assumes independence" '
        'overstate their certainty on a time series; read the effect size.')

GEN, HALF_WIND, IMPORTING = generation()
WX = load('weather_historical')
TEMP = WX.groupby(WX['time'].dt.normalize())['temperature_2m_c'].mean()
WIND_SPEED = WX.groupby(WX['time'].dt.normalize())['wind_speed_100m_kmh'].mean()
BANK = load('bank_holidays')
EW_HOLIDAYS = BANK[BANK['division'] == 'england-and-wales'].set_index('holiday_date')['title']


# --- 1. fuelhh ------------------------------------------------------------------------------------------------

def fuelhh():
    CURRENT[0] = 'fuelhh'
    share = GEN['share']
    est = [estimates(share, 'Renewable share (%)'), estimates(GEN['wind'], 'Wind output (MW)', 0), estimates(GEN['gas'], 'Gas output (MW)', 0),
           estimates(GEN['imports'], 'Net imports (MW)', 0)]
    rows = battery(share, 'the renewable share', 'points')
    for a, b, name in (('wind', 'gas', 'wind and gas output'), ('wind', 'imports', 'wind output and net imports')):
        r, lo, hi, n = corr_ci(GEN[a], GEN[b])
        rows.append(row('Correlation (block bootstrap)', f'Are {name} related, day to day?', f'r = {f(r)}', None, f'r = {f(r)}', ci(lo, hi),
                        'Related.' if lo * hi > 0 else 'Interval includes zero.'))
    r, lo, hi, n = corr_ci(share, share.shift(1))
    rows.append(row('Correlation (block bootstrap)', 'How much of today is in yesterday?', f'r = {f(r)}', None, f'r = {f(r)}', ci(lo, hi), ''))
    labels, e, l, h = [], [], [], []
    for c in IMPORTING.mean().sort_values(ascending=False).index:
        v = 100 * IMPORTING[c].dropna()
        lo, hi = block_bootstrap(v.to_numpy(), np.mean)
        labels.append(c); e.append(v.mean()); l.append(lo); h.append(hi)
    charts = calendar_charts(share, 'Renewable share', '%') + [
        ci_chart('Share of half-hours each interconnector imports, with 95% intervals', '14-day block bootstrap over days.', labels, e, l, h, '%', ref=50)]
    finish('fuelhh', [table('Estimates with uncertainty', EST_COLS, est,
                            'Naive: treats days as independent. Robust: Newey–West. Effective days: how many independent days would give the robust interval.'),
                      table('Tests on the daily renewable share', COLS, rows, NOTE)], charts)


# --- 2. demand_outturn and 10. bank_holidays ------------------------------------------------------------------

def demand_and_holidays():
    CURRENT[0] = 'demand_outturn'
    d = load('demand_outturn')
    demand = d.groupby('settlement_date')['initial_demand_outturn_mw'].agg(['mean', 'count'])
    demand = demand.loc[demand['count'] >= 46, 'mean']
    frame = pd.DataFrame({'demand': demand, 'temp': TEMP}).dropna()
    frame['heating_degrees'] = (15.5 - frame['temp']).clip(lower=0)
    frame['saturday'] = (frame.index.dayofweek == 5).astype(float)
    frame['sunday'] = (frame.index.dayofweek == 6).astype(float)
    frame['bank_holiday'] = frame.index.isin(EW_HOLIDAYS.index).astype(float)
    X = sm.add_constant(pd.concat([frame[['heating_degrees', 'saturday', 'sunday', 'bank_holiday']], harmonics(frame.index)], axis=1))
    m = hac_ols(frame['demand'], X)
    names = {'heating_degrees': 'Each degree below 15.5 °C', 'saturday': 'Saturday', 'sunday': 'Sunday', 'bank_holiday': 'Bank holiday (England and Wales)',
             'years': 'Each year (trend)'}
    reg_rows = []
    for k, label in names.items():
        lo, hi = m.conf_int().loc[k]
        reg_rows.append([label, f(m.params[k], 0), ci(lo, hi, 0), f(m.tvalues[k]), pf(m.pvalues[k]), f'{100 * m.params[k] / frame["demand"].mean():+.1f}%'])
        PVALUES.append(('demand_outturn', 'Demand regression', label, float(m.pvalues[k])))
    resid_lb = acorr_ljungbox(m.resid, lags=[7], return_df=True)
    rows = battery(demand, 'daily demand', 'MW', 0)
    r, lo, hi, n = corr_ci(frame['demand'], frame['temp'])
    rows.append(row('Correlation (block bootstrap)', 'Is demand related to temperature?', f'r = {f(r)}', None, f'r = {f(r)}', ci(lo, hi), 'Strongly, and negatively.'))
    rows.append(row('Ljung–Box on regression residuals', 'Does the demand regression leave autocorrelation behind?',
                    f'Q = {f(resid_lb.loc[7, "lb_stat"], 0)}', resid_lb.loc[7, 'lb_pvalue'], f'R² = {f(m.rsquared, 3)}', '–',
                    verdict(resid_lb.loc[7, 'lb_pvalue'], 'Yes: hence the HAC errors.', 'No.')))
    finish('demand_outturn', [
        table('Estimates with uncertainty', EST_COLS, [estimates(demand, 'National demand (MW)', 0)],
              'Demand is so persistent that 1,100 days are worth far fewer independent ones.'),
        table('What moves daily demand: regression with Newey–West errors', ['Term', 'Effect MW', '95% interval', 't', 'p-value', 'Share of mean demand'],
              reg_rows, f'Daily mean demand on heating degrees, day type, bank holiday, a trend and two annual harmonics. R² = {m.rsquared:.3f}, {len(frame):,} days.'),
        table('Tests on daily demand', COLS, rows, NOTE)], calendar_charts(demand, 'National demand', 'MW', 0))

    CURRENT[0] = 'bank_holidays'
    lo, hi = m.conf_int().loc['bank_holiday']
    rows = [row('Bank holiday effect (HAC regression)', 'Does a bank holiday lower national demand, given the weather and the season?',
                f't = {f(m.tvalues["bank_holiday"])}', m.pvalues['bank_holiday'],
                f'{f(m.params["bank_holiday"], 0)} MW ({100 * m.params["bank_holiday"] / frame["demand"].mean():+.1f}%)', ci(lo, hi, 0),
                'Yes, and by more than a Sunday does.')]
    groups = {'Christmas Day': 'Christmas Day', 'Boxing Day': 'Boxing Day', 'New Year': 'New Year', 'Good Friday': 'Good Friday',
              'Easter Monday': 'Easter Monday', 'Early May': 'Early May', 'Spring': 'Spring bank', 'Summer': 'Summer bank'}
    titles = EW_HOLIDAYS.reindex(frame.index)
    dummies = pd.DataFrame({name: titles.fillna('').str.contains(key).astype(float) for name, key in groups.items()}, index=frame.index)
    X2 = sm.add_constant(pd.concat([frame[['heating_degrees', 'saturday', 'sunday']], dummies, harmonics(frame.index)], axis=1))
    m2 = hac_ols(frame['demand'], X2)
    conf = m2.conf_int()
    holiday_rows = [[name, int(dummies[name].sum()), f(m2.params[name], 0), ci(conf.loc[name, 0], conf.loc[name, 1], 0), pf(m2.pvalues[name]),
                     f'{100 * m2.params[name] / frame["demand"].mean():+.1f}%'] for name in groups]
    b = BANK.assign(weekday=BANK['holiday_date'].dt.dayofweek)
    observed = b['weekday'].value_counts().reindex(range(5)).fillna(0)
    gof = st.chisquare(observed)
    rows.append(row('Chi-square goodness of fit', 'Are bank holidays spread evenly over Monday to Friday?', f'χ² = {f(gof.statistic, 1)}', gof.pvalue,
                    f'Monday {100 * observed[0] / observed.sum():.0f}% of holidays', '–', verdict(gof.pvalue, 'No: Mondays dominate.', 'Evenly spread.')))
    tab = pd.crosstab(b['division'], np.where(b['weekday'] == 0, 'Monday', np.where(b['weekday'] == 4, 'Friday', 'Midweek')))
    chi2, p, dof, v = cramers_v(tab)
    rows.append(row('Chi-square independence', 'Do the three nations place holidays on different weekdays?', f'χ² = {f(chi2, 1)} ({dof} df)', p,
                    f'Cramér V = {f(v)}', '–', verdict(p, 'The pattern differs by nation.', 'No difference between nations found.')))
    per_year = b.groupby([b['holiday_date'].dt.year, 'division']).size().unstack()
    kw = st.kruskal(*[per_year[c].dropna() for c in per_year.columns])
    rows.append(row('Kruskal–Wallis', 'Do the nations have the same number of holidays a year?', f'H = {f(kw.statistic, 1)}', kw.pvalue,
                    ', '.join(f'{c}: {per_year[c].mean():.1f}' for c in per_year.columns), '–',
                    verdict(kw.pvalue, 'The counts differ.', 'No difference found.')))
    finish('bank_holidays', [
        table('Tests', COLS, rows, 'Demand effects come from the daily demand regression (heating degrees, day type, trend, annual harmonics; Newey–West errors).'),
        table('Effect of each holiday on national demand', ['Holiday', 'Days in the data', 'Effect MW', '95% interval', 'p-value', 'Share of mean demand'],
              holiday_rows, 'Each holiday against an ordinary weekday in the same weather and season. Two or three days each, so the intervals are wide.')],
        [ci_chart('Effect of each bank holiday on national demand, with 95% intervals', 'MW against an ordinary weekday, given weather and season.',
                  list(groups), [m2.params[n] for n in groups], [conf.loc[n, 0] for n in groups], [conf.loc[n, 1] for n in groups], 'MW', ref=0, wide=True, digits=0)])
    return demand, d


# --- 3. wind_forecast -----------------------------------------------------------------------------------------

def wind_forecast():
    CURRENT[0] = 'wind_forecast'
    w = load('wind_forecast')
    w['lead'] = (w['start_time'] - w['publish_time']).dt.total_seconds() / 3600
    actual = HALF_WIND.resample('1h').mean()
    w['actual'] = actual.reindex(w['start_time']).to_numpy()
    w['error'] = w['generation_mw'] - w['actual']
    w['day'] = w['start_time'].dt.normalize()
    rows, labels, e, l, h = [], [], [], [], []
    for name, lo_h, hi_h in (('0–6 h', 0, 6), ('6–12 h', 6, 12), ('12–24 h', 12, 24), ('24–36 h', 24, 36), ('36–48 h', 36, 48), ('48–66 h', 48, 67)):
        daily = w[w['lead'].between(lo_h, hi_h, inclusive='left')].groupby('day')['error'].mean()
        r, hm = bias_row(daily, f'the wind forecast {name} ahead', 'MW')
        rows.append(r); labels.append(name); e.append(hm['mean']); l.append(hm['lo']); h.append(hm['hi'])

    def at_lead(lo_h, hi_h):
        g = w[w['lead'].between(lo_h, hi_h, inclusive='left')].sort_values('publish_time').groupby('start_time').tail(1)
        return g.set_index('start_time')

    near, day, far = at_lead(3, 9), at_lead(27, 33), at_lead(51, 57)
    rows.append(mincer_zarnowitz(day['generation_mw'], day['actual'], 'the 27–33 hour forecast'))
    centred = {k: (g['error'] - g['error'].mean()).abs().groupby(g.index.normalize()).mean() for k, g in (('near', near), ('day', day), ('far', far))}
    rows.append(diebold_mariano(centred['near'], centred['day'], 'the 3–9 hour forecast', 'the 27–33 hour forecast', 'MW'))
    rows.append(diebold_mariano(centred['day'], centred['far'], 'the 27–33 hour forecast', 'the 51–57 hour forecast', 'MW'))
    # A forecast must beat what was already known: the wind 48 hours before the target hour
    naive = (actual.shift(48).reindex(day.index) - day['actual']).abs().groupby(day.index.normalize()).mean()
    rows.append(diebold_mariano(centred['day'], naive, 'the 27–33 hour forecast (bias removed)', 'wind 48 hours earlier', 'MW'))
    r, lo, hi, n = corr_ci(day['generation_mw'].groupby(day.index.normalize()).mean(), day['actual'].groupby(day.index.normalize()).mean())
    rows.append(row('Correlation (block bootstrap)', 'How closely does the day-ahead forecast follow actual wind, daily?', f'r = {f(r, 3)}', None,
                    f'r = {f(r, 3)}', ci(lo, hi, 3), ''))
    finish('wind_forecast', [table('Forecast evaluation', COLS, rows,
                                   'Errors are forecast minus FUELHH wind, averaged to days before testing. Accuracy comparisons remove each forecast\'s mean bias first, '
                                   'because the two feeds cover different fleets.')],
           [ci_chart('Forecast bias by lead time, with 95% intervals', 'Forecast − FUELHH wind. Newey–West interval on daily mean errors.', labels, e, l, h, 'MW',
                     ref=0, wide=True, digits=0)])


# --- 4. demand_forecast ---------------------------------------------------------------------------------------

def demand_forecast(outturn):
    CURRENT[0] = 'demand_forecast'
    d = load('demand_forecast')
    actual = outturn.set_index('start_time')['initial_demand_outturn_mw']
    d['actual'] = actual.reindex(d['start_time']).to_numpy()
    known = (d[d['publish_time'] < d['settlement_date'] - pd.Timedelta(hours=6)].sort_values('publish_time')
             .groupby(['settlement_date', 'settlement_period']).tail(1))
    known = known.assign(error=known['demand_mw'] - known['actual'])
    daily_error = known.groupby('settlement_date')['error'].mean()
    rows = []
    r, _ = bias_row(daily_error, 'the day-ahead demand forecast', 'MW')
    rows.append(r)
    rows.append(mincer_zarnowitz(known['demand_mw'], known['actual'], 'the day-ahead demand forecast'))
    loss = known.assign(loss=known['error'].abs()).groupby('settlement_date')['loss'].mean()
    for hours, name in ((48, 'demand two days earlier'), (168, 'demand a week earlier')):
        naive = (actual.shift(freq=pd.Timedelta(hours=hours)).reindex(known['start_time']).to_numpy() - known['actual']).abs()
        rows.append(diebold_mariano(loss, naive.groupby(known['settlement_date']).mean(), 'the day-ahead forecast', name, 'MW'))
    kw = st.kruskal(*[daily_error[daily_error.index.dayofweek == i].dropna() for i in range(7)])
    rows.append(row('Kruskal–Wallis', 'Does the forecast error depend on the day of the week?', f'H = {f(kw.statistic, 1)}', kw.pvalue, '–', '–',
                    verdict(kw.pvalue, 'Yes: some days are forecast high or low.', 'No.') + INDEP))
    wide = known[known['settlement_period'] <= 48].pivot_table(index='settlement_date', columns='settlement_period', values='error')
    labels, e, l, h = [], [], [], []
    for p in range(1, 49, 2):
        hm = hac_mean((wide[p] + wide[p + 1]) / 2)
        labels.append(f'{(p - 1) // 2:02d}:00'); e.append(hm['mean']); l.append(hm['lo']); h.append(hm['hi'])
    significant = sum(1 for a, b in zip(l, h) if a * b > 0)
    rows.append(row('Bias by hour (HAC)', 'In how many hours of the day is the bias distinguishable from zero?', f'{significant} of 24 hours', None,
                    f'{f(min(e), 0)} to {f(max(e), 0)} MW', '–', 'The bias has a daily shape.'))
    lb = acorr_ljungbox(daily_error.dropna(), lags=[7], return_df=True)
    rows.append(row('Ljung–Box', 'Are the daily forecast errors autocorrelated?', f'Q = {f(lb.loc[7, "lb_stat"], 0)}', lb.loc[7, 'lb_pvalue'],
                    f'lag-1 r = {f(daily_error.autocorr(1))}', '–',
                    verdict(lb.loc[7, 'lb_pvalue'], 'Yes: yesterday\'s error predicts today\'s.', 'No.')))
    finish('demand_forecast', [table('Forecast evaluation', COLS, rows, 'The latest forecast published before 18:00 UTC the day before, against INDO. ' + NOTE)],
           [ci_chart('Day-ahead bias by hour of day, with 95% intervals', 'Forecast − outturn, UK local time. Newey–West interval.', labels, e, l, h, 'MW',
                     ref=0, wide=True, digits=0)])
    return known.groupby('settlement_date')['demand_mw'].mean()


# --- 5 and 6. wind and solar ----------------------------------------------------------------------------------

PSR = ('Solar', 'Wind Offshore', 'Wind Onshore')


def wind_solar():
    CURRENT[0] = 'wind_solar_actual'
    a = load('wind_solar_actual')
    wide = a.pivot(index=['settlement_date', 'settlement_period'], columns='psr_type', values='quantity_mw')
    dates = wide.index.get_level_values(0)
    daily = wide.groupby(dates).mean()
    daily['wind'] = daily['Wind Offshore'] + daily['Wind Onshore']
    ratio = (daily['wind'] / GEN['wind']).dropna()
    ratio = ratio[GEN['wind'].reindex(ratio.index) > 1500]
    month = ratio.index.to_period('M')
    gap = (month >= '2024-04') & (month <= '2024-11')
    before, after = ratio[month < '2024-04'], ratio[month > '2024-11']
    rows = []
    X = sm.add_constant(pd.DataFrame({'gap': gap.astype(float), 'after': (month > '2024-11').astype(float)}, index=ratio.index))
    m = hac_ols(ratio, X)
    conf = m.conf_int()
    rows.append(row('Break regression (HAC)', 'Is the AGWS ÷ FUELHH wind ratio lower from April to November 2024?', f't = {f(m.tvalues["gap"])}',
                    m.pvalues['gap'], f'{f(m.params["gap"])} (from {f(m.params["const"])})', ci(conf.loc['gap', 0], conf.loc['gap', 1]),
                    'A break, not noise: the feed lost over half its wind.'))
    rows.append(row('Break regression (HAC)', 'Is the ratio different after the gap than before it?', f't = {f(m.tvalues["after"])}', m.pvalues['after'],
                    f'{f(m.params["after"])} ({f(before.mean())} before, {f(after.mean())} after)', ci(conf.loc['after', 0], conf.loc['after', 1]),
                    verdict(m.pvalues['after'], 'Yes: AGWS covers more wind since December 2024 than it did before the gap.', 'No.')))
    mw = st.mannwhitneyu(ratio[gap], ratio[~gap])
    rows.append(row('Mann–Whitney U', 'Do gap days and healthy days share a distribution of the ratio?', f'U = {f(mw.statistic, 0)}', mw.pvalue,
                    f'median {f(ratio[gap].median())} vs {f(ratio[~gap].median())}', '–', 'They do not overlap in any useful sense.' + INDEP))
    ks = st.ks_2samp(ratio[gap], ratio[~gap])
    rows.append(row('Kolmogorov–Smirnov', 'Same question, on the whole distribution.', f'D = {f(ks.statistic, 3)}', ks.pvalue, '–', '–',
                    'The distributions are almost disjoint.' + INDEP))
    healthy = daily[~daily.index.to_period('M').isin(pd.period_range('2024-04', '2024-11', freq='M'))]
    for x, y, name in (('Wind Offshore', 'Wind Onshore', 'offshore and onshore wind'), ('Solar', 'wind', 'solar and wind')):
        r, lo, hi, n = corr_ci(healthy[x], healthy[y])
        rows.append(row('Correlation (block bootstrap)', f'Are {name} related, day to day?', f'r = {f(r)}', None, f'r = {f(r)}', ci(lo, hi),
                        'Related.' if lo * hi > 0 else 'Interval includes zero.'))
    solar = daily['Solar']
    s24, s25 = solar[solar.index.year == 2024], solar[solar.index.year == 2025]
    s24.index, s25.index = s24.index.strftime('%m-%d'), s25.index.strftime('%m-%d')
    hm = hac_mean((s25 - s24).dropna())
    rows.append(row('Paired year test (HAC)', 'Did AGWS solar grow from 2024 to 2025, same calendar day?', f't = {f(hm["t"])}', hm['p'],
                    f'{f(hm["mean"], 0)} MW ({100 * hm["mean"] / s24.mean():+.0f}%)', ci(hm['lo'], hm['hi'], 0),
                    verdict(hm['p'], 'Yes.', 'No change distinguishable.')))
    labels = ['Before (Nov 2023 – Mar 2024)', 'Gap (Apr – Nov 2024)', 'After (Dec 2024 on)']
    parts = [hac_mean(before), hac_mean(ratio[gap]), hac_mean(after)]
    finish('wind_solar_actual', [
        table('Estimates with uncertainty', EST_COLS, [estimates(healthy['wind'], 'AGWS wind, healthy months (MW)', 0), estimates(solar, 'AGWS solar (MW)', 0),
                                                        estimates(ratio[~gap], 'AGWS ÷ FUELHH wind, healthy months', 3)]),
        table('Tests', COLS, rows, 'The ratio uses days with FUELHH wind above 1,500 MW. ' + NOTE)],
        [ci_chart('AGWS wind as a multiple of FUELHH wind, by regime, with 95% intervals', 'Mean of daily ratios; Newey–West interval.', labels,
                  [p['mean'] for p in parts], [p['lo'] for p in parts], [p['hi'] for p in parts], '×', ref=1, wide=True)])

    CURRENT[0] = 'wind_solar_forecast'
    fc = load('wind_solar_forecast')
    keyed = a.set_index(['psr_type', 'settlement_date', 'settlement_period'])['quantity_mw']
    fc['actual'] = keyed.reindex(pd.MultiIndex.from_frame(fc[['psr_type', 'settlement_date', 'settlement_period']])).to_numpy()
    fc = fc[~fc['settlement_date'].dt.to_period('M').isin(pd.period_range('2024-04', '2024-11', freq='M'))]
    fc['error'] = fc['quantity_mw'] - fc['actual']
    rows, labels, e, l, h = [], [], [], [], []
    for process in ('Day ahead', 'Intraday process', 'Intraday total'):
        for psr in PSR:
            g = fc[(fc['process_type'] == process) & (fc['psr_type'] == psr)]
            r, hm = bias_row(g.groupby('settlement_date')['error'].mean(), f'the {process.lower()} {psr.lower()} forecast', 'MW')
            rows.append(r); labels.append(f'{process} · {psr}'); e.append(hm['mean']); l.append(hm['lo']); h.append(hm['hi'])
    for psr in PSR:
        da = (fc[(fc['process_type'] == 'Day ahead') & (fc['psr_type'] == psr)].sort_values('publish_time')
              .groupby(['settlement_date', 'settlement_period'])[['quantity_mw', 'actual', 'error']].last())
        it = (fc[(fc['process_type'] == 'Intraday total') & (fc['psr_type'] == psr)].sort_values('publish_time')
              .groupby(['settlement_date', 'settlement_period'])[['quantity_mw', 'actual', 'error']].last())
        rows.append(mincer_zarnowitz(da['quantity_mw'], da['actual'], f'the day-ahead {psr.lower()} forecast'))
        both = pd.DataFrame({'da': da['error'].abs(), 'it': it['error'].abs()}).dropna()
        day_loss = both.groupby(level=0).mean()
        rows.append(diebold_mariano(day_loss['it'], day_loss['da'], f'the intraday {psr.lower()} forecast', 'the day-ahead one', 'MW'))
        wx = st.wilcoxon(day_loss['it'], day_loss['da'])
        rows.append(row('Wilcoxon signed-rank', f'Same comparison for {psr.lower()}, on ranks.', f'W = {f(wx.statistic, 0)}', wx.pvalue,
                        f'intraday better on {100 * (day_loss["it"] < day_loss["da"]).mean():.0f}% of days', '–',
                        verdict(wx.pvalue, 'A difference.', 'No difference found.') + INDEP))
    finish('wind_solar_forecast', [table('Forecast evaluation', COLS, rows, 'Scored against AGWS, leaving out April to November 2024, when AGWS under-reports. ' + NOTE)],
           [ci_chart('Forecast bias by process and technology, with 95% intervals', 'Forecast − AGWS actual. Newey–West interval on daily mean errors.',
                     labels, e, l, h, 'MW', ref=0, wide=True, digits=0)])
    da_all = load('wind_solar_forecast')
    da_all = da_all[da_all['process_type'] == 'Day ahead']
    return da_all.pivot_table(index=['settlement_date', 'settlement_period'], columns='psr_type', values='quantity_mw', aggfunc='last') \
        .groupby(level=0).mean().sum(axis=1, min_count=3)


# --- 7 and 8. prices ------------------------------------------------------------------------------------------

def prices():
    CURRENT[0] = 'system_prices'
    p = load('system_prices')
    v = 'system_sell_price_gbp_mwh'
    daily = p.groupby('settlement_date')[v].mean()
    rows = battery(daily, 'the daily system price', '£/MWh', 2)
    log_jb = st.jarque_bera(np.log(daily[daily > 0]))
    rows.append(row('Jarque–Bera on log price', 'Does taking logs make the daily price normal?', f'JB = {f(log_jb.statistic, 0)}', log_jb.pvalue,
                    f'skew {f(st.skew(np.log(daily[daily > 0])))}', '–', verdict(log_jb.pvalue, 'No, though the skew shrinks.', 'Yes.') + INDEP))
    short = p[p['net_imbalance_volume_mwh'] > 0].groupby('settlement_date')[v].mean()
    long_ = p[p['net_imbalance_volume_mwh'] < 0].groupby('settlement_date')[v].mean()
    hm = hac_mean((short - long_).dropna())
    rows.append(row('Paired test (HAC)', 'Is the price higher in short half-hours than in long ones, on the same day?', f't = {f(hm["t"])}', hm['p'],
                    f'£{f(hm["mean"])} per MWh', ci(hm['lo'], hm['hi']), 'Yes: the price depends on which side the system is.'))
    r, lo, hi, n = corr_ci(daily, GEN['wind'])
    rows.append(row('Correlation (block bootstrap)', 'Is the daily price related to wind output?', f'r = {f(r)}', None, f'r = {f(r)}', ci(lo, hi),
                    'Windy days are cheaper.' if hi < 0 else 'Interval includes zero.'))
    negative = 100 * (p[v] < 0).groupby(p['settlement_date']).mean()
    labels, e, l, h = [], [], [], []
    for year in (2024, 2025, 2026):
        x = negative[negative.index.year == year]
        lo, hi = block_bootstrap(x.to_numpy(), np.mean)
        labels.append(str(year)); e.append(x.mean()); l.append(lo); h.append(hi)
    lo, hi = block_bootstrap(negative.to_numpy(), np.mean)
    rows.append(row('Proportion (block bootstrap)', 'What share of half-hours has a negative price?', '–', None, f'{f(negative.mean())}%', ci(lo, hi, 2, '%'), ''))
    short_share = 100 * (p['net_imbalance_volume_mwh'] > 0).groupby(p['settlement_date']).mean()
    hm = hac_mean(short_share - 50)
    rows.append(row('Proportion test (HAC)', 'Is the system short half the time?', f't = {f(hm["t"])}', hm['p'], f'{f(hm["mean"] + 50, 1)}% short',
                    ci(hm['lo'] + 50, hm['hi'] + 50, 1, '%'), verdict(hm['p'], 'No: it is long more often than short.', 'Consistent with half.')))
    finish('system_prices', [
        table('Estimates with uncertainty', EST_COLS, [estimates(daily, 'System price (£/MWh)', 2), estimates(negative, 'Negative-price half-hours (% of day)', 2)]),
        table('Tests on the daily system price', COLS, rows, NOTE)],
        calendar_charts(daily, 'System price', '£/MWh', 2) + [
            ci_chart('Share of half-hours with a negative price, by year, with 95% intervals', '14-day block bootstrap over days. 2026 runs to 6 November.',
                     labels, e, l, h, '%', digits=2)])

    CURRENT[0] = 'market_index'
    m = load('market_index')
    apx = m[(m['data_provider'] == 'APXMIDP') & (m['price_gbp_mwh'] != 0)].groupby('settlement_date')['price_gbp_mwh'].mean()
    both = pd.DataFrame({'apx': apx, 'system': daily}).dropna()
    rows = []
    co = coint(both['system'], both['apx'])
    rows.append(row('Engle–Granger cointegration', 'Do the system price and the market price share a long-run path?', f't = {f(co[0])}', co[1], '–', '–',
                    verdict(co[1], 'Yes: they cannot drift apart for long.', 'Not established.')))
    hm = hac_mean(both['system'] - both['apx'])
    rows.append(row('Paired test (HAC)', 'Is the system price different from the market price on average?', f't = {f(hm["t"])}', hm['p'],
                    f'£{f(hm["mean"])} per MWh', ci(hm['lo'], hm['hi']), verdict(hm['p'], 'A small systematic premium.', 'No: being out of balance costs nothing on average.')))
    lev = st.levene(both['system'], both['apx'])
    rows.append(row('Levene', 'Is the system price more variable than the market price?', f'W = {f(lev.statistic, 1)}', lev.pvalue,
                    f'std dev £{f(both["system"].std())} vs £{f(both["apx"].std())}', '–', verdict(lev.pvalue, 'Yes.', 'No difference found.') + INDEP))
    for cause, effect in (('apx', 'system'), ('system', 'apx')):
        g = grangercausalitytests(both[[effect, cause]], maxlag=[2], verbose=False)[2][0]['ssr_ftest']
        names = {'apx': 'the market price', 'system': 'the system price'}
        rows.append(row('Granger causality (2 lags)', f'Does {names[cause]} help predict {names[effect]} a day on?', f'F = {f(g[0], 1)}', g[1], '–', '–',
                        verdict(g[1], 'Yes.', 'No.')))
    r, lo, hi, n = corr_ci(both['apx'], both['system'])
    rows.append(row('Correlation (block bootstrap)', 'How closely do the two prices move together, daily?', f'r = {f(r, 3)}', None, f'r = {f(r, 3)}', ci(lo, hi, 3), ''))
    finish('market_index', [
        table('Estimates with uncertainty', EST_COLS, [estimates(apx, 'APX market index (£/MWh)', 2), estimates(both['system'] - both['apx'], 'System − market price (£/MWh)', 2)]),
        table('Tests', COLS, rows + battery(apx, 'the daily APX price', '£/MWh', 2)[2:8], 'APX only, half-hours with trades. ' + NOTE)],
        calendar_charts(apx, 'APX price', '£/MWh', 2))
    return daily, apx


# --- 9. remit_outages -----------------------------------------------------------------------------------------

def remit():
    CURRENT[0] = 'remit_outages'
    r = load('remit_outages')
    cur = load('remit_outages_current')
    cur = cur[cur['unavailability_type'].isin(['Planned', 'Unplanned'])].copy()
    cur['hours'] = (cur['event_end_time'] - cur['event_start_time']).dt.total_seconds() / 3600
    rows = []
    top = cur['fuel_type'].value_counts().head(7).index
    tab = pd.crosstab(cur.loc[cur['fuel_type'].isin(top), 'fuel_type'], cur.loc[cur['fuel_type'].isin(top), 'unavailability_type'])
    chi2, p, dof, v = cramers_v(tab)
    rows.append(row('Chi-square independence', 'Does the planned/unplanned split depend on fuel type?', f'χ² = {f(chi2, 0)} ({dof} df)', p,
                    f'Cramér V = {f(v)}', '–', 'Yes, strongly: ' + ', '.join(f'{k} {100 * tab.loc[k, "Planned"] / tab.loc[k].sum():.0f}% planned' for k in tab.index[:3]) + '.'))
    weekday = np.where(r['publish_time'].dt.dayofweek >= 5, 'Weekend', 'Weekday')
    known = r['unavailability_type'].isin(['Planned', 'Unplanned'])
    tab = pd.crosstab(weekday[known], r.loc[known, 'unavailability_type'])
    chi2, p, dof, v = cramers_v(tab)
    rows.append(row('Chi-square independence', 'Are weekend notices more often unplanned?', f'χ² = {f(chi2, 0)} ({dof} df)', p,
                    f'Cramér V = {f(v)}; unplanned {100 * tab.loc["Weekend", "Unplanned"] / tab.loc["Weekend"].sum():.0f}% at weekends, '
                    f'{100 * tab.loc["Weekday", "Unplanned"] / tab.loc["Weekday"].sum():.0f}% on weekdays', '–',
                    verdict(p, 'Yes, but look at V for how much.', 'No.')))
    planned, unplanned = cur.loc[(cur['unavailability_type'] == 'Planned') & (cur['hours'] > 0), 'hours'], cur.loc[(cur['unavailability_type'] == 'Unplanned') & (cur['hours'] > 0), 'hours']
    mw = st.mannwhitneyu(planned, unplanned)
    boot = [np.median(RNG.choice(planned.to_numpy(), len(planned))) - np.median(RNG.choice(unplanned.to_numpy(), len(unplanned))) for _ in range(500)]
    rows.append(row('Mann–Whitney U', 'Do planned outages last longer than unplanned ones?', f'U = {f(mw.statistic, 0)}', mw.pvalue,
                    f'median {f(planned.median(), 1)} h vs {f(unplanned.median(), 1)} h; rank-biserial {f(1 - 2 * mw.statistic / (len(planned) * len(unplanned)))}',
                    ci(*np.percentile(boot, [2.5, 97.5]), 1, ' h'), verdict(mw.pvalue, 'Yes.', 'No difference found.')))
    kw = st.kruskal(*[g['unavailable_capacity_mw'].dropna() for _, g in cur[cur['fuel_type'].isin(top)].groupby('fuel_type')])
    rows.append(row('Kruskal–Wallis', 'Does the size of an outage depend on fuel type?', f'H = {f(kw.statistic, 0)}', kw.pvalue, '–', '–',
                    verdict(kw.pvalue, 'Yes.', 'No.')))
    hours = cur.loc[cur['hours'] > 0, 'hours']
    logs = np.log(hours)
    sample = RNG.choice(logs.to_numpy(), 2000, replace=False)
    ks = st.kstest(sample, st.norm(logs.mean(), logs.std()).cdf)
    rows.append(row('Kolmogorov–Smirnov', 'Is outage duration log-normal?', f'D = {f(ks.statistic, 3)}', ks.pvalue,
                    f'log-mean {f(logs.mean())}, log-sd {f(logs.std())} (median {f(math.exp(logs.mean()), 1)} h)', '–',
                    verdict(ks.pvalue, 'Not exactly: durations pile up on round numbers.', 'Consistent with log-normal.') + ' Sample of 2,000.'))
    late = (r['event_start_time'] < r['publish_time'])
    per_message = late.groupby(r['mrid']).mean()
    lo, hi = proportion_confint(int(late.sum()), int(late.notna().sum()), method='wilson')
    se = per_message.std() / math.sqrt(len(per_message))
    rows.append(row('Proportion (Wilson; by message)', 'What share of notices is published after the event began?', '–', None,
                    f'{100 * late.mean():.1f}% of notices', f'{100 * lo:.1f}% to {100 * hi:.1f}%',
                    f'Per message it is {100 * per_message.mean():.1f}% ± {196 * se:.1f}: revisions of one message are not independent.'))
    counts = r.groupby(r['publish_time'].dt.normalize()).size()
    dispersion = counts.var() / counts.mean()
    chi = (len(counts) - 1) * dispersion
    rows.append(row('Dispersion test', 'Do notices arrive at random (Poisson) from day to day?', f'χ² = {f(chi, 0)}', float(st.chi2.sf(chi, len(counts) - 1)),
                    f'variance ÷ mean = {f(dispersion, 1)}', '–', 'No: they come in bursts, far more variable than chance.'))
    wk = counts[counts.index.dayofweek < 5]
    we = counts[counts.index.dayofweek >= 5]
    t = st.ttest_ind(wk, we, equal_var=False)
    rows.append(row('Welch t', 'Are more notices published on weekdays than at weekends?', f't = {f(t.statistic)}', t.pvalue,
                    f'{f(wk.mean(), 0)} vs {f(we.mean(), 0)} a day; Cohen d {f(cohen_d(wk, we))}', '–', verdict(t.pvalue, 'Yes.', 'No.') + INDEP))
    labels, e, l, h = [], [], [], []
    for fuel in top:
        g = cur[cur['fuel_type'] == fuel]
        k, n = int((g['unavailability_type'] == 'Planned').sum()), len(g)
        lo, hi = proportion_confint(k, n, method='wilson')
        labels.append(fuel); e.append(100 * k / n); l.append(100 * lo); h.append(100 * hi)
    finish('remit_outages', [table('Tests', COLS, rows, 'Latest revision of each message unless the question is about notices. With tens of thousands of rows nearly '
                                   'every difference is significant; the effect sizes say whether it matters.')],
           [ci_chart('Share of messages that are planned, by fuel type, with 95% intervals', 'Wilson interval on the latest revision of each message.',
                     labels, e, l, h, '%', ref=50, wide=True, digits=1)])


# --- 11. carbon_intensity -------------------------------------------------------------------------------------

def carbon():
    CURRENT[0] = 'carbon_intensity'
    c = load('carbon_intensity')
    s = c.set_index('period_start')
    daily = s['actual_gco2_kwh'].resample('D').mean().dropna()
    rows = battery(daily, 'daily carbon intensity', 'gCO2/kWh')
    error = (s['forecast_gco2_kwh'] - s['actual_gco2_kwh']).resample('D').mean()
    r, _ = bias_row(error, 'the carbon intensity forecast', 'gCO2/kWh', 2)
    rows.append(r)
    rows.append(mincer_zarnowitz(s['forecast_gco2_kwh'].resample('D').mean(), daily, 'the carbon intensity forecast (daily)'))
    frame = pd.DataFrame({'carbon': daily, 'gas': GEN['gas_share']}).dropna()
    m = hac_ols(frame['carbon'], sm.add_constant(frame['gas']))
    lo, hi = m.conf_int().loc['gas']
    rows.append(row('Regression (HAC)', 'How much carbon intensity does a point of gas share add?', f't = {f(m.tvalues["gas"], 1)}', m.pvalues['gas'],
                    f'{f(m.params["gas"])} gCO2/kWh per point; R² = {f(m.rsquared, 3)}', ci(lo, hi),
                    f'With no gas the line predicts {f(m.params["const"], 0)} gCO2/kWh.'))
    tab = pd.crosstab(c['period_start'].dt.year, c['intensity_index']).loc[[2024, 2025, 2026]]
    chi2, p, dof, v = cramers_v(tab)
    rows.append(row('Chi-square independence', 'Did the mix of intensity bands change between years?', f'χ² = {f(chi2, 0)} ({dof} df)', p, f'Cramér V = {f(v)}', '–',
                    'Significant only because of the sample size: V shows the mix barely moved.' + INDEP))
    finish('carbon_intensity', [table('Estimates with uncertainty', EST_COLS, [estimates(daily, 'Carbon intensity (gCO2/kWh)'),
                                                                               estimates(error, 'Forecast error (gCO2/kWh)', 2)]),
                                table('Tests on daily carbon intensity', COLS, rows, NOTE)], calendar_charts(daily, 'Carbon intensity', 'gCO2/kWh'))
    return daily


# --- 12. pv_live and 13. neso_demand --------------------------------------------------------------------------

def solar_and_neso(demand):
    CURRENT[0] = 'pv_live'
    p = load('pv_live')
    s = pd.Series(p['generation_mw'].to_numpy(), index=p['datetime_gmt'] - pd.Timedelta(minutes=30))
    energy = (s.resample('D').sum() * 0.5 / 1000).iloc[:-1]
    rows = battery(energy, 'daily solar energy', 'GWh')
    positive = energy[energy > 0]
    X = sm.add_constant(harmonics(positive.index, pairs=3))
    m = hac_ols(np.log(positive), X)
    lo, hi = m.conf_int().loc['years']
    rows.append(row('Growth regression (HAC)', 'How fast is solar output growing, after the season?', f't = {f(m.tvalues["years"])}', m.pvalues['years'],
                    f'{100 * (math.exp(m.params["years"]) - 1):+.1f}% per year', f'{100 * (math.exp(lo) - 1):+.1f}% to {100 * (math.exp(hi) - 1):+.1f}%',
                    'Compound growth, from log daily energy on a trend and three annual harmonics.'))
    e24, e25 = energy[energy.index.year == 2024], energy[energy.index.year == 2025]
    e24.index, e25.index = e24.index.strftime('%m-%d'), e25.index.strftime('%m-%d')
    wx = st.wilcoxon((e25 - e24).dropna())
    rows.append(row('Wilcoxon signed-rank', 'Was solar output higher in 2025 than in 2024, same calendar day?', f'W = {f(wx.statistic, 0)}', wx.pvalue,
                    f'2025 higher on {100 * ((e25 - e24).dropna() > 0).mean():.0f}% of days', '–', verdict(wx.pvalue, 'Yes.', 'No difference found.') + INDEP))
    r, lo, hi, n = corr_ci(energy, WX.groupby(WX['time'].dt.normalize())['shortwave_radiation_wm2'].mean())
    rows.append(row('Correlation (block bootstrap)', 'Does daily solar energy follow radiation at the five weather points?', f'r = {f(r, 3)}', None,
                    f'r = {f(r, 3)}', ci(lo, hi, 3), ''))
    finish('pv_live', [table('Estimates with uncertainty', EST_COLS, [estimates(energy, 'Solar energy (GWh a day)')],
                             'Solar is almost all season, so its mean is known far less precisely than 1,100 days suggest.'),
                       table('Tests on daily solar energy', COLS, rows, NOTE)], calendar_charts(energy, 'Solar energy', 'GWh'))

    CURRENT[0] = 'neso_demand'
    n = load('neso_demand')
    nd = n.groupby('settlement_date')['nd_mw'].mean()
    rows = []
    diff = (nd - demand).dropna()
    hm = hac_mean(diff)
    rows.append(row('Paired test (HAC)', 'Is NESO national demand different from Elexon INDO on average?', f't = {f(hm["t"])}', hm['p'], f'{f(hm["mean"], 1)} MW',
                    ci(hm['lo'], hm['hi'], 1), verdict(hm['p'], 'Yes, statistically.', 'No.')))
    se = (hm['hi'] - hm['lo']) / (2 * 1.96)
    margin = 100
    p_tost = max(st.norm.sf((hm['mean'] + margin) / se), st.norm.cdf((hm['mean'] - margin) / se))
    rows.append(row('Equivalence test (TOST, HAC)', f'Are the two within ±{margin} MW of each other on average?', f'z = {f((margin - abs(hm["mean"])) / se, 1)}',
                    float(p_tost), f'{f(hm["mean"], 1)} MW against a margin of ±{margin} MW ({100 * margin / demand.mean():.1f}% of demand)', '–',
                    verdict(p_tost, 'Equivalent for practical purposes: the difference is real but tiny.', 'Equivalence not established.')))
    pv_daily = s.resample('D').mean()
    solar = n.groupby('settlement_date')['embedded_solar_generation_mw'].mean()
    hm = hac_mean((solar - pv_daily).dropna())
    rows.append(row('Paired test (HAC)', 'Is NESO embedded solar different from PV_Live on average?', f't = {f(hm["t"])}', hm['p'], f'{f(hm["mean"], 1)} MW',
                    ci(hm['lo'], hm['hi'], 1), verdict(hm['p'], 'A small systematic difference.', 'No: they are the same series.')))
    cap = n.groupby('settlement_date')[['embedded_solar_capacity_mw', 'embedded_wind_capacity_mw']].max()
    for col, name in (('embedded_solar_capacity_mw', 'embedded solar capacity'), ('embedded_wind_capacity_mw', 'embedded wind capacity')):
        t = (cap.index - cap.index[0]).days / 365.25
        m = hac_ols(cap[col], sm.add_constant(pd.Series(t, index=cap.index, name='years')))
        lo, hi = m.conf_int().loc['years']
        rows.append(row('Trend regression (HAC)', f'Is {name} growing?', f't = {f(m.tvalues["years"], 1)}', m.pvalues['years'],
                        f'{f(m.params["years"], 0)} MW per year', ci(lo, hi, 0), verdict(m.pvalues['years'], 'Yes.' if m.params['years'] > 0 else 'It is shrinking.', 'No trend distinguishable.')))
    emb = n.groupby('settlement_date')[['embedded_wind_generation_mw', 'embedded_solar_generation_mw']].mean()
    r, lo, hi, _ = corr_ci(emb['embedded_wind_generation_mw'], GEN['wind'])
    rows.append(row('Correlation (block bootstrap)', 'Does embedded wind move with transmission wind (FUELHH)?', f'r = {f(r, 3)}', None, f'r = {f(r, 3)}', ci(lo, hi, 3),
                    'Closely: FUELHH wind is a good proxy for the wind it cannot see.'))
    finish('neso_demand', [table('Estimates with uncertainty', EST_COLS, [estimates(nd, 'National demand, NESO (MW)', 0), estimates(diff, 'NESO ND − Elexon INDO (MW)', 1),
                                                                          estimates(emb.sum(axis=1), 'Embedded wind + solar (MW)', 0)]),
                           table('Tests', COLS, rows + battery(nd, 'NESO daily demand', 'MW', 0)[2:4], NOTE)],
           calendar_charts(emb.sum(axis=1), 'Embedded wind + solar', 'MW', 0))


# --- 14 and 15. weather ---------------------------------------------------------------------------------------

LOCATIONS = {'dogger_bank': 'Dogger Bank', 'east_anglia_offshore': 'East Anglia offshore', 'irish_sea': 'Irish Sea',
             'scottish_highlands': 'Scottish Highlands', 'southern_england': 'Southern England'}
VARIABLES = {'wind_speed_100m_kmh': ('wind at 100 m', 'km/h'), 'wind_speed_10m_kmh': ('wind at 10 m', 'km/h'),
             'shortwave_radiation_wm2': ('radiation', 'W/m²'), 'temperature_2m_c': ('temperature', '°C'), 'cloud_cover_pct': ('cloud cover', '%')}


def weather():
    CURRENT[0] = 'weather_historical'
    hist = WX.assign(place=WX['location'].map(LOCATIONS))
    wind = hist.pivot(index='time', columns='place', values='wind_speed_100m_kmh')[list(LOCATIONS.values())]
    daily = wind.resample('D').mean()
    est = [estimates(daily[c], f'Wind at 100 m, {c} (km/h)') for c in daily.columns] + [estimates(TEMP, 'Temperature, five-point mean (°C)')]
    rows = []
    kw = st.kruskal(*[daily[c] for c in daily.columns])
    an = st.f_oneway(*[daily[c] for c in daily.columns])
    grand = daily.to_numpy().mean()
    eta = sum(len(daily) * (daily[c].mean() - grand) ** 2 for c in daily.columns) / ((daily.to_numpy() - grand) ** 2).sum()
    rows.append(row('Kruskal–Wallis', 'Do the five locations have the same wind?', f'H = {f(kw.statistic, 0)}', kw.pvalue, f'η² = {f(eta)} (ANOVA F = {f(an.statistic, 0)})', '–',
                    f'No, but location explains only {100 * eta:.0f}% of the variance: the day matters more than the place.' + INDEP))
    fried = st.friedmanchisquare(*[daily[c] for c in daily.columns])
    rows.append(row('Friedman', 'Same question, comparing locations within each day.', f'χ² = {f(fried.statistic, 0)}', fried.pvalue, '–', '–',
                    verdict(fried.pvalue, 'The ranking of locations is consistent from day to day.', 'No consistent ranking.')))
    weibull_rows, labels, e, l, h = [], [], [], [], []
    for c in wind.columns:
        x = wind[c].dropna().to_numpy()
        x = x[x > 0]
        k, _, scale = st.weibull_min.fit(x, floc=0)
        boots = [st.weibull_min.fit(RNG.choice(x, 3000), floc=0)[0] for _ in range(120)]
        ks = st.kstest(RNG.choice(x, 2000, replace=False), st.weibull_min(k, 0, scale).cdf)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        weibull_rows.append([c, f(k, 3), ci(lo, hi, 3), f(scale, 1), f(ks.statistic, 3), pf(ks.pvalue), verdict(ks.pvalue, 'Weibull rejected', 'Consistent with Weibull')])
        PVALUES.append(('weather_historical', 'Kolmogorov–Smirnov (Weibull)', c, float(ks.pvalue)))
        labels.append(c); e.append(k); l.append(lo); h.append(hi)
    tukey = pairwise_tukeyhsd(daily.to_numpy().ravel(), np.tile(daily.columns.to_numpy(), len(daily)))
    groups = tukey.groupsunique
    pairs = [(groups[i], groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    tukey_rows = [[f'{b} − {a}', f(d), ci(lo, hi), pf(p), 'differ' if rej else 'not distinguishable']
                  for (a, b), d, (lo, hi), p, rej in zip(pairs, tukey.meandiffs, tukey.confint, tukey.pvalues, tukey.reject)]
    for (a, b), p in zip(pairs, tukey.pvalues):
        PVALUES.append(('weather_historical', 'Tukey HSD', f'{b} − {a}', float(p)))
    national = daily.mean(axis=1)
    rows += battery(national, 'daily mean wind speed', 'km/h')
    r, lo, hi, n = corr_ci(national, GEN['share'])
    rows.append(row('Correlation (block bootstrap)', 'Does mean wind speed track the renewable share?', f'r = {f(r, 3)}', None, f'r = {f(r, 3)}', ci(lo, hi, 3), 'Closely.'))
    finish('weather_historical', [
        table('Estimates with uncertainty', EST_COLS, est),
        table('Tests', COLS, rows, NOTE),
        table('Weibull fit to hourly wind at 100 m', ['Location', 'Shape k', '95% interval', 'Scale km/h', 'KS D', 'p-value', 'Reading'], weibull_rows,
              'Shape near 2 is the classic Rayleigh wind distribution; higher means steadier wind. Fit on all hours, tested on a sample of 2,000.'),
        table('Which locations differ: Tukey HSD on daily mean wind', ['Pair', 'Difference km/h', '95% interval', 'Adjusted p-value', 'Reading'], tukey_rows,
              'Adjusted for the ten comparisons. Assumes independent days.')],
        [ci_chart('Weibull shape by location, with 95% intervals', 'Higher is steadier. Bootstrap interval.', labels, e, l, h, '', ref=2, wide=True, digits=3)]
        + calendar_charts(national, 'Wind speed at 100 m', 'km/h'))

    CURRENT[0] = 'weather_day_ahead'
    da = load('weather_day_ahead')
    merged = da.merge(WX, on=['location', 'time'], suffixes=('', '_actual'))
    merged['day'] = merged['time'].dt.normalize()
    rows, labels, e, l, h = [], [], [], [], []
    for v, (name, unit) in VARIABLES.items():
        err = merged[v] - merged[f'{v}_actual']
        r, hm = bias_row(err.groupby(merged['day']).mean(), f'the day-ahead {name} forecast', unit, 3)
        rows.append(r)
    for loc, nice in LOCATIONS.items():
        g = merged[merged['location'] == loc]
        hm = hac_mean((g['wind_speed_100m_kmh'] - g['wind_speed_100m_kmh_actual']).groupby(g['day']).mean())
        labels.append(nice); e.append(hm['mean']); l.append(hm['lo']); h.append(hm['hi'])
    for v in ('wind_speed_100m_kmh', 'temperature_2m_c'):
        rows.append(mincer_zarnowitz(merged.groupby('day')[v].mean(), merged.groupby('day')[f'{v}_actual'].mean(), f'the day-ahead {VARIABLES[v][0]} forecast (daily)'))
    abs_err = (merged['wind_speed_100m_kmh'] - merged['wind_speed_100m_kmh_actual']).abs()
    per_place = abs_err.groupby([merged['day'], merged['location']]).mean().unstack().dropna()
    fried = st.friedmanchisquare(*[per_place[c] for c in per_place.columns])
    rows.append(row('Friedman', 'Is the wind forecast equally accurate at all five locations?', f'χ² = {f(fried.statistic, 0)}', fried.pvalue,
                    ', '.join(f'{LOCATIONS[c]} {per_place[c].mean():.1f}' for c in per_place.columns) + ' km/h', '–',
                    verdict(fried.pvalue, 'No: some locations are harder.', 'Yes.')))
    da_wind = merged.groupby('day')['wind_speed_100m_kmh'].mean()
    actual_wind = merged.groupby('day')['wind_speed_100m_kmh_actual'].mean()
    naive = (actual_wind.shift(2) - actual_wind).abs()
    rows.append(diebold_mariano((da_wind - actual_wind).abs(), naive, 'the day-ahead wind forecast', 'wind two days earlier', 'km/h', 2))
    r1, lo1, hi1, _ = corr_ci(da_wind, GEN['share'])
    r2, lo2, hi2, _ = corr_ci(actual_wind, GEN['share'])
    rows.append(row('Correlation (block bootstrap)', 'Does day-ahead wind predict the renewable share as well as the wind that happened?', f'r = {f(r1, 3)} vs {f(r2, 3)}',
                    None, f'day-ahead {f(r1, 3)}; short-range {f(r2, 3)}', f'{ci(lo1, hi1, 3)}; {ci(lo2, hi2, 3)}',
                    'The intervals overlap: little is lost by using the day-ahead forecast.' if hi1 > lo2 else 'The day-ahead forecast is measurably weaker.'))
    finish('weather_day_ahead', [table('Forecast evaluation', COLS, rows, 'Day-ahead minus the short-range forecast in weather_historical, averaged over the five points and to days. ' + NOTE)],
           [ci_chart('Day-ahead wind bias by location, with 95% intervals', 'Day-ahead − short-range, wind at 100 m. Newey–West interval.', labels, e, l, h, 'km/h',
                     ref=0, wide=True, digits=3)])
    return da.groupby(da['time'].dt.normalize())[['wind_speed_100m_kmh', 'temperature_2m_c', 'shortwave_radiation_wm2']].mean()


# --- Across tables: the target on its day-ahead features ------------------------------------------------------

def across(ndf, dgws, wx_da, price, carbon_daily, demand):
    CURRENT[0] = 'across'
    target = GEN['share']
    features = pd.DataFrame({
        'Forecast wind + solar': dgws, 'Forecast demand': ndf, 'Day-ahead wind speed': wx_da['wind_speed_100m_kmh'],
        'Day-ahead temperature': wx_da['temperature_2m_c'], 'Day-ahead radiation': wx_da['shortwave_radiation_wm2'],
        'Share 2 days earlier': target.shift(2, freq='D'), 'Share 7 days earlier': target.shift(7, freq='D'),
        'Weekend': pd.Series((target.index.dayofweek >= 5).astype(float), index=target.index)}).reindex(target.index)
    frame = features.assign(target=target).dropna()
    z = (frame - frame.mean()) / frame.std()
    m = hac_ols(z['target'], sm.add_constant(z[features.columns]))
    raw = hac_ols(frame['target'], sm.add_constant(frame[features.columns]))
    conf = m.conf_int()
    X = sm.add_constant(frame[features.columns])
    vif = {c: variance_inflation_factor(X.to_numpy(), i) for i, c in enumerate(X.columns) if c != 'const'}
    coef_rows, labels, e, l, h = [], [], [], [], []
    for c in features.columns:
        others = [o for o in features.columns if o != c]
        ry = sm.OLS(frame['target'], sm.add_constant(frame[others])).fit().resid
        rx = sm.OLS(frame[c], sm.add_constant(frame[others])).fit().resid
        coef_rows.append([c, f(m.params[c], 3), ci(conf.loc[c, 0], conf.loc[c, 1], 3), f(m.tvalues[c]), pf(m.pvalues[c]), f(frame[c].corr(frame['target'])),
                          f(np.corrcoef(rx, ry)[0, 1]), f(vif[c], 1)])
        PVALUES.append(('across', 'Target regression', c, float(m.pvalues[c])))
        labels.append(c); e.append(m.params[c]); l.append(conf.loc[c, 0]); h.append(conf.loc[c, 1])
    bp = het_breuschpagan(raw.resid, X)
    lb = acorr_ljungbox(raw.resid, lags=[7], return_df=True)
    jb = st.jarque_bera(raw.resid)
    rows = [
        row('F test', 'Do the day-ahead features explain the target at all?', f'F = {f(float(raw.fvalue), 0)}', float(raw.f_pvalue),
            f'R² = {f(raw.rsquared, 3)}, adjusted {f(raw.rsquared_adj, 3)}', '–', f'Yes: {100 * raw.rsquared:.0f}% of the variance, on {len(frame):,} days.'),
        row('Durbin–Watson', 'Are the regression residuals autocorrelated at one day?', f'DW = {f(durbin_watson(raw.resid))}', None, '2 means none', '–',
            'Positive autocorrelation remains.' if durbin_watson(raw.resid) < 1.7 else 'Little autocorrelation remains.'),
        row('Ljung–Box on residuals', 'Is anything predictable left in the residuals?', f'Q = {f(lb.loc[7, "lb_stat"], 0)}', lb.loc[7, 'lb_pvalue'],
            f'lag-1 r = {f(pd.Series(raw.resid).autocorr(1))}', '–', verdict(lb.loc[7, 'lb_pvalue'], 'Yes: yesterday\'s miss says something about today\'s.', 'No.')),
        row('Breusch–Pagan', 'Is the error equally large across the range of the features?', f'LM = {f(bp[0], 1)}', bp[1], '–', '–',
            verdict(bp[1], 'No: the error varies with the features (heteroscedastic).', 'Yes.')),
        row('Jarque–Bera on residuals', 'Are the residuals normal?', f'JB = {f(jb.statistic, 1)}', jb.pvalue, f'skew {f(st.skew(raw.resid))}, excess kurtosis {f(st.kurtosis(raw.resid))}',
            '–', verdict(jb.pvalue, 'Not quite: use robust intervals.', 'Yes.')),
    ]
    # Out of sample: fit on everything before 2026, score on 2026
    train, test = frame[frame.index < '2026-01-01'], frame[frame.index >= '2026-01-01']
    full = sm.OLS(train['target'], sm.add_constant(train[features.columns])).fit()
    one = sm.OLS(train['target'], sm.add_constant(train[['Forecast wind + solar', 'Forecast demand']])).fit()
    pred = {'All eight features': full.predict(sm.add_constant(test[features.columns])),
            'Forecast wind + solar and demand only': one.predict(sm.add_constant(test[['Forecast wind + solar', 'Forecast demand']])),
            'Training mean': pd.Series(train['target'].mean(), index=test.index),
            'Share 2 days earlier': test['Share 2 days earlier']}
    oos_rows = []
    for name, yhat in pred.items():
        err = test['target'] - yhat
        lo, hi = block_bootstrap(err.to_numpy(), lambda x: np.abs(x).mean())
        oos_rows.append([name, f(np.sqrt((err ** 2).mean())), f(err.abs().mean()), ci(lo, hi), f(1 - (err ** 2).sum() / ((test['target'] - test['target'].mean()) ** 2).sum(), 3)])
    rows.append(diebold_mariano((test['target'] - pred['All eight features']).abs(), (test['target'] - pred['Forecast wind + solar and demand only']).abs(),
                                'the eight-feature model', 'the two-forecast model', 'points', 2))
    rows.append(diebold_mariano((test['target'] - pred['All eight features']).abs(), (test['target'] - pred['Training mean']).abs(),
                                'the eight-feature model', 'the training mean', 'points', 2))
    pairs = [('Renewable share and wind speed', target, WIND_SPEED), ('Renewable share and system price', target, price),
             ('Renewable share and carbon intensity', target, carbon_daily), ('Demand and temperature', demand, TEMP),
             ('System price and gas output', price, GEN['gas']), ('Net imports and wind output', GEN['imports'], GEN['wind'])]
    corr_rows = []
    for name, a, b in pairs:
        r, lo, hi, n = corr_ci(a, b)
        z_ = math.atanh(r)
        naive = (math.tanh(z_ - 1.96 / math.sqrt(n - 3)), math.tanh(z_ + 1.96 / math.sqrt(n - 3)))
        rho = pd.DataFrame({'a': a, 'b': b}).dropna().corr(method='spearman').iloc[0, 1]
        corr_rows.append([name, f'{n:,}', f(r, 3), ci(*naive, 3), ci(lo, hi, 3), f(rho, 3), 'Yes' if lo * hi > 0 else 'No'])
    g = grangercausalitytests(pd.DataFrame({'share': target, 'wind': WIND_SPEED}).dropna()[['share', 'wind']], maxlag=[2], verbose=False)[2][0]['ssr_ftest']
    rows.append(row('Granger causality (2 lags)', 'Does past wind speed help predict the share beyond its own past?', f'F = {f(g[0], 1)}', g[1], '–', '–', verdict(g[1], 'Yes.', 'No.')))
    finish('across', [
        table('The target on its day-ahead features: regression with Newey–West errors',
              ['Feature', 'Standardised coefficient', '95% interval', 't', 'p-value', 'Correlation with target', 'Partial correlation', 'VIF'], coef_rows,
              f'{len(frame):,} days with every feature. Standardised: the change in the target, in standard deviations, for one standard deviation of the feature, '
              'holding the others fixed. Partial correlation: what the feature adds once the others are known. VIF above 5 signals collinearity.'),
        table('Regression diagnostics and model comparisons', COLS, rows, NOTE),
        table(f'Out of sample: fitted before 2026, scored on {len(test)} days of 2026', ['Model', 'RMSE (points)', 'MAE (points)', 'MAE 95% interval', 'R²'], oos_rows,
              'Time-ordered, so nothing from the test period leaks into the fit. Interval from a 14-day block bootstrap.'),
        table('Correlations with honest intervals', ['Pair', 'Days', 'Pearson r', 'Naive 95% interval', 'Block-bootstrap 95% interval', 'Spearman ρ', 'Excludes zero'], corr_rows,
              'The naive Fisher interval treats days as independent and is too narrow; the block-bootstrap one allows for autocorrelation.')],
        [ci_chart('Standardised effect of each day-ahead feature on the target, with 95% intervals', 'Holding the other features fixed. Newey–West interval.',
                  labels, e, l, h, 'std dev', ref=0, wide=True, digits=3)])
    DIGEST.append(f'  vif {vif}')


fuelhh()
DEMAND, OUTTURN = demand_and_holidays()
wind_forecast()
NDF = demand_forecast(OUTTURN)
DGWS = wind_solar()
PRICE, APX = prices()
remit()
CARBON = carbon()
solar_and_neso(DEMAND)
WX_DA = weather()
across(NDF, DGWS, WX_DA, PRICE, CARBON, DEMAND)

# How many results survive a correction for the number of tests run
p = np.array([x[3] for x in PVALUES])
reject, _, _, _ = multipletests(p, alpha=0.05, method='fdr_bh')
bonf = multipletests(p, alpha=0.05, method='bonferroni')[0]
lost = [f'{t}: {test} – {q}' for (t, test, q, pv), keep in zip(PVALUES, reject) if pv < 0.05 and not keep]
RESULT['summary'] = {'tests': len(p), 'raw': int((p < 0.05).sum()), 'fdr': int(reject.sum()), 'bonferroni': int(bonf.sum()), 'lost': lost}
DIGEST.append(f"\n===== summary {RESULT['summary']}")
(OUT / 'inference.json').write_text(json.dumps(sanitize(RESULT), allow_nan=False, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
(OUT / 'inference_digest.txt').write_text('\n'.join(DIGEST), encoding='utf-8')
print(f"{len(p)} tests, {int((p < 0.05).sum())} below 0.05, {int(reject.sum())} after FDR; "
      f"{sum(len(v['tables']) for k, v in RESULT.items() if k != 'summary')} tables, "
      f"{sum(len(v['charts']) for k, v in RESULT.items() if k != 'summary')} charts")
