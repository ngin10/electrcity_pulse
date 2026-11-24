"""Exploratory analysis of the 15 silver tables: statistics, data-quality checks and chart series.

Reads eda/out/silver (built by build_silver.py) and writes eda/out/report.json, which build_report.py turns
into the page. Prints a digest of the headline numbers.

    python eda/analyze.py
"""
import json
import warnings

import duckdb
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller

from lib import (DOW, MONTHS, OUT, PERIOD_LABELS, ROOT, bar, box, check, completeness, corr_heat, days, error_stats,
                 fmt_int, heat, hist, line, load, month_hour_heat, month_period_heat, num, period_start,
                 periods_expected, profile, range_check, sanitize, scatter, table, tile, uk_local)

warnings.filterwarnings('ignore')
BUILD = json.loads((OUT / 'build_stats.json').read_text())
SECTIONS, SHARED, DIGEST = [], {}, []
SEASONS = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
           6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}


def say(*parts):
    DIGEST.append(' '.join(str(p) for p in parts))


def section(table, title, source, grain, keys, df, tiles, checks, charts, tables=()):
    b = BUILD[table]
    removed = b['raw_records'] - b['rows']
    base = [tile('Rows', fmt_int(len(df)), f'one per {grain}'),
            tile('Raw records', fmt_int(b['raw_records']),
                 f'{removed:,} superseded by a later version' if removed else 'no duplicate keys'),
            tile('Landing', f"{b['files']} files", f"{b['responses']:,} responses, {b['raw_json_bytes'] / 1e6:.1f} MB of JSON")]
    SECTIONS.append({'id': table, 'title': title, 'source': source, 'grain': grain, 'keys': list(keys),
                     'rows': len(df), 'raw_records': b['raw_records'], 'duplicates': removed,
                     'tiles': base + list(tiles), 'profile': profile(df), 'checks': list(checks),
                     'charts': list(charts), 'tables': list(tables)})
    say(f'\n===== {table}: {len(df):,} rows, {removed:,} duplicates')
    for c in checks:
        say(f"  check [{c['level']}] {c['name']}: {c['count']:,}/{c['of']:,} {c['note']}")
    for t in tables:
        say(f"  table {t['title']}: {t['cols']}")
        for r in t['rows'][:14]:
            say('    ', r)


def rolling(series, window=28):
    return series.rolling(window, min_periods=window // 2).mean()


def acf_chart(series, lags, title, sub, step_label='lag (days)'):
    values = acf(series.interpolate().dropna(), nlags=lags, fft=True)[1:]
    return bar(title, sub, range(1, lags + 1), [('Autocorrelation', values)], digits=3)


def yearly(frame, columns, how):
    """A table of one statistic per year for several columns."""
    g = frame.groupby(frame.index.year)[columns].agg(how)
    return [[int(y), *[num(v, 1) for v in row]] for y, row in zip(g.index, g.to_numpy())]


def season_profile(df, value, period='settlement_period'):
    s = df[df[period].between(1, 48)]
    grid = s.groupby([s['settlement_date'].dt.month.map(SEASONS), period])[value].mean().unstack()
    return [(name, grid.loc[name].reindex(range(1, 49)).to_numpy()) for name in ('Winter', 'Spring', 'Summer', 'Autumn')]


# --- 1. fuelhh ----------------------------------------------------------------------------------------------

def fuelhh():
    f = load('fuelhh')
    W = f.pivot(index=['settlement_date', 'settlement_period'], columns='fuel_type', values='generation_mw')
    ints = [c for c in W.columns if c.startswith('INT')]
    dates = W.index.get_level_values(0)
    G = pd.DataFrame({'Wind': W['WIND'], 'Gas': W['CCGT'] + W['OCGT'], 'Nuclear': W['NUCLEAR'],
                      'Biomass': W['BIOMASS'], 'Net imports': W[ints].sum(axis=1), 'Hydro': W['NPSHYD'],
                      'Coal and oil': W['COAL'] + W['OIL'], 'Other and pumped storage': W['OTHER'] + W['PS']})
    domestic = W.drop(columns=ints).sum(axis=1)
    renewable = W[['WIND', 'NPSHYD', 'BIOMASS']].sum(axis=1)
    present = W['WIND'].groupby(dates).count()
    complete = present == periods_expected(present.index)
    daily = pd.DataFrame({'domestic_mwh': domestic.groupby(dates).sum() * 0.5,
                          'renewable_mwh': renewable.groupby(dates).sum() * 0.5})
    daily['share'] = 100 * daily['renewable_mwh'] / daily['domestic_mwh']
    daily = daily[complete]
    share = daily['share']
    weekly = (G.groupby(dates).mean().resample('W').mean() / 1000).iloc[1:-1]

    half = pd.DataFrame({'start_time': period_start(pd.Series(dates), pd.Series(W.index.get_level_values(1))).to_numpy(),
                         'wind': W['WIND'].to_numpy(), 'gas': G['Gas'].to_numpy(), 'domestic': domestic.to_numpy(),
                         'renewable': renewable.to_numpy(), 'imports': G['Net imports'].to_numpy(),
                         'nuclear': W['NUCLEAR'].to_numpy()})
    SHARED['gen_half'] = half.set_index('start_time')
    SHARED['gen_daily'] = pd.DataFrame({'renewable_share': share, 'wind_mw': W['WIND'].groupby(dates).mean(),
                                        'gas_mw': G['Gas'].groupby(dates).mean(),
                                        'nuclear_mw': W['NUCLEAR'].groupby(dates).mean(),
                                        'net_imports_mw': G['Net imports'].groupby(dates).mean(),
                                        'domestic_mw': domestic.groupby(dates).mean()})

    # Replays and revisions, from the raw messages
    landing = (ROOT / 'data' / 'landing' / 'fuelhh').as_posix()
    keys, repeated, revised = duckdb.connect().execute(f"""
        SELECT count(*), count(*) FILTER (n > 1), count(*) FILTER (v > 1) FROM (
            SELECT count(*) AS n, count(DISTINCT raw_value->'payload'->>'generation') AS v
            FROM read_parquet('{landing}/**/*.parquet', hive_partitioning = false)
            GROUP BY raw_value->'payload'->>'settlementDate', raw_value->'payload'->>'settlementPeriod',
                     raw_value->'payload'->>'fuelType')""").fetchone()
    lag = (f['publish_time'] - f['start_time']).dt.total_seconds() / 60
    non_flow = f[~f['fuel_type'].str.startswith('INT') & (f['fuel_type'] != 'PS')]
    negative = int((non_flow['generation_mw'] < 0).sum())
    coal = W['COAL']
    coal_periods = (coal > 0).groupby(dates).sum()
    last_coal = coal_periods[coal_periods >= 24].index.max()
    coal_blips = int(coal_periods[coal_periods.index > last_coal].sum())
    fuels_per_period = W.notna().sum(axis=1)

    energy = (G.groupby(dates.year).sum() * 0.5 / 1e6)
    total = energy.sum(axis=1)
    mix_rows = [[int(y), *[f'{100 * v / total[y]:.1f}%' for v in energy.loc[y]], f'{total[y]:.1f}'] for y in energy.index]
    by_fuel = f.groupby('fuel_type')['generation_mw']
    stats_rows = [[k, num(v.mean(), 0), num(v.median(), 0), num(v.std(), 0), num(v.min(), 0), num(v.max(), 0),
                   f'{100 * (v == 0).mean():.1f}%', f'{100 * (v < 0).mean():.1f}%']
                  for k, v in by_fuel]
    mean_by_fuel = by_fuel.mean().sort_values(ascending=False)
    import_share = (100 * (W[ints] > 0).sum() / W[ints].notna().sum()).sort_values(ascending=False)
    sp = W.index.get_level_values(1)
    profile_lines = [(name, G[name][sp <= 48].groupby(sp[sp <= 48]).mean().reindex(range(1, 49)) / 1000)
                     for name in ('Wind', 'Gas', 'Nuclear', 'Biomass', 'Net imports')]
    flat = W.reset_index()
    monthly_coal = (coal.groupby(dates.to_period('M')).sum() * 0.5 / 1000)
    month_share = {MONTHS[m - 1]: share[share.index.month == m] for m in range(1, 13)}
    dow_share = {DOW[d]: share[share.index.dayofweek == d] for d in range(7)}
    duration = np.sort(W['WIND'].dropna().to_numpy())[::-1]
    pct = np.linspace(0, 100, 101)
    corr = pd.DataFrame({'Wind': W['WIND'], 'Gas CCGT': W['CCGT'], 'Nuclear': W['NUCLEAR'], 'Biomass': W['BIOMASS'],
                         'Hydro': W['NPSHYD'], 'Pumped storage': W['PS'], 'Other': W['OTHER'],
                         'Net imports': G['Net imports'], 'Domestic total': domestic})

    charts = [
        line('Generation mix, weekly mean', 'Wind and gas trade places week to week; nuclear and biomass are the flat base.',
             days(weekly.index), [(c, weekly[c]) for c in weekly.columns], 'GW', area=True, stack=True, wide=True),
        line('Daily renewable share of GB generation', 'Wind + hydro + biomass over domestic generation, energy-weighted. '
             'This is the model target.', days(share.index),
             [('Daily share', share), ('28-day mean', rolling(share))], '%', wide=True),
        hist('Distribution of the daily renewable share', 'Complete days only.', share, '%', bins=40, clip=None),
        box('Renewable share by month of year', 'Higher and wider in winter, when wind is strongest.', month_share, '%'),
        box('Renewable share by day of week', 'Weekends run a little higher: demand falls, wind does not.', dow_share, '%'),
        acf_chart(share, 30, 'Autocorrelation of the daily renewable share',
                  'Yesterday explains a lot; a week back explains little.'),
        bar('Mean output by fuel type', 'Interconnectors (INT*) are flows: positive is import.',
            mean_by_fuel.index, [('Mean MW', mean_by_fuel.to_numpy())], 'MW', horizontal=True, digits=0),
        box('Spread of half-hourly output by fuel type', '', {k: v for k, v in by_fuel}, 'MW', digits=0, wide=True),
        line('Average day by source', 'Mean by settlement period (UK local time), all days.', PERIOD_LABELS,
             profile_lines, 'GW', digits=2),
        month_period_heat(flat.assign(v=flat['CCGT'] / 1000), 'v', 'Gas (CCGT) output by month and time of day',
                          'Gas follows demand: winter evenings are darkest.', 'GW', digits=1),
        month_period_heat(flat.assign(v=flat['WIND'] / 1000), 'v', 'Wind output by month and time of day',
                          'Seasonal far more than daily: rows differ, columns barely do.', 'GW', digits=1),
        line('Wind duration curve', 'Share of half-hours in which wind output was at least this high.',
             [f'{p:.0f}%' for p in pct], [('Wind output', np.percentile(duration, 100 - pct))], 'MW', area=True, digits=0),
        corr_heat(corr, 'Correlation between sources, half-hourly', 'Wind and gas are the strongest opposites.'),
        bar('How often each interconnector imports', 'Share of half-hours with a positive (import) flow.',
            import_share.index, [('Importing', import_share.to_numpy())], '%', horizontal=True),
        bar('Coal generation by month', f'Sustained coal generation ends on {last_coal:%d %B %Y}; after it only {coal_blips} isolated half-hours report any.',
            [str(p) for p in monthly_coal.index], [('Coal', monthly_coal.to_numpy())], 'GWh', digits=1, wide=True),
        hist('Delay between a period starting and Elexon publishing it', 'Minutes from start_time to publish_time.',
             lag, 'min', bins=40, clip=(0, 0.99), digits=0),
    ]
    checks = completeness(f[f['fuel_type'] == 'WIND'].assign(start_time=f['start_time'])) + [
        check('Periods where all listed fuel types report', int((fuels_per_period == fuels_per_period.max()).sum()),
              len(fuels_per_period), 'warn', f'{fuels_per_period.min()} to {fuels_per_period.max()} fuel types per period; '
              'INTGRNL and INTVKL start later'),
        check('Keys delivered more than once', repeated, keys, 'good', 'at-least-once delivery: replays collapse to one row'),
        check('Keys whose value was revised', revised, keys, 'good' if revised == 0 else 'warn',
              'different generation figures for the same period and fuel', bad=True),
        range_check(f, 'generation_mw', -10000, 40000, 'generation_implausible'),
        check('Negative output outside interconnectors and pumped storage', negative, len(non_flow),
              'good' if negative == 0 else 'warn', 'the soft rule negative_generation', bad=True),
        check('Pumped storage periods below zero (pumping)', int((W['PS'] < 0).sum()), int(W['PS'].notna().sum()), 'info',
              'expected: water pumped uphill'),
    ]
    tiles = [tile('Coverage', f'{dates.min():%b %Y} – {dates.max():%b %Y}', f'{dates.nunique():,} days'),
             tile('Renewable share', f'{100 * daily["renewable_mwh"].sum() / daily["domestic_mwh"].sum():.1f}%',
                  f'daily range {share.min():.0f}% to {share.max():.0f}%'),
             tile('Mean GB generation', f'{domestic.mean() / 1000:.1f} GW', f'peak {domestic.max() / 1000:.1f} GW'),
             tile('Fuel types', str(f['fuel_type'].nunique()), f'{len(ints)} of them interconnectors')]
    tables = [table('Energy mix by year', ['Year', *G.columns, 'Total TWh'], mix_rows,
                    'Share of GB supply including net imports. 2023 and 2026 are part years.'),
              table('Half-hourly statistics by fuel type',
                    ['Fuel', 'Mean MW', 'Median', 'Std dev', 'Min', 'Max', 'Zero', 'Negative'], stats_rows)]
    section('fuelhh', 'Generation by fuel type', 'Elexon FUELHH, via Kafka',
            'settlement date, period and fuel type', ('settlement_date', 'settlement_period', 'fuel_type'),
            f, tiles, checks, charts, tables)
    say(f'  share mean {share.mean():.1f} median {share.median():.1f} std {share.std():.1f} min {share.min():.1f} '
        f'({share.idxmin():%Y-%m-%d}) max {share.max():.1f} ({share.idxmax():%Y-%m-%d}); last coal {last_coal:%Y-%m-%d} blips {coal_blips}')
    say('  share by year', share.groupby(share.index.year).mean().round(1).to_dict())
    say('  share by month', share.groupby(share.index.month).mean().round(1).to_dict())
    say('  share by dow', share.groupby(share.index.dayofweek).mean().round(1).to_dict())
    say('  share acf', np.round(acf(share.interpolate().dropna(), nlags=8)[1:], 2))
    say('  corr wind-gas', round(corr['Wind'].corr(corr['Gas CCGT']), 2), 'wind-imports',
        round(corr['Wind'].corr(corr['Net imports']), 2), 'publish lag median', lag.median(), 'p99', lag.quantile(.99))
    say('  wind p10/p50/p90', np.percentile(duration, [10, 50, 90]).round(0), 'wind max', duration.max())
    say('  import share', import_share.round(0).to_dict())
    say('  incomplete days', (~complete).sum(), list(complete[~complete].index.strftime('%Y-%m-%d'))[:12])


# --- 2. demand_outturn --------------------------------------------------------------------------------------

def demand_outturn():
    d = load('demand_outturn')
    v = 'initial_demand_outturn_mw'
    s = d.set_index('start_time')[v]
    daily = d.groupby('settlement_date')[v].agg(['mean', 'max', 'min'])
    SHARED['demand_half'] = s
    SHARED['demand_daily'] = daily['mean']
    weekday = d['settlement_date'].dt.dayofweek
    in_range = d['settlement_period'].between(1, 48)
    kind = np.where(weekday < 5, 'Weekday', np.where(weekday == 5, 'Saturday', 'Sunday'))
    by_kind = d[in_range].groupby([kind[in_range], 'settlement_period'])[v].mean().unstack()
    gap = d['initial_transmission_demand_outturn_mw'] - d[v]
    lag = (d['publish_time'] - d['start_time']).dt.total_seconds() / 60
    peak_period = d.loc[d.groupby('settlement_date')[v].idxmax(), 'settlement_period']
    hourly = s.resample('30min').mean()
    ac = acf(hourly.interpolate().dropna(), nlags=336, fft=True)
    top = d.nlargest(5, v)
    low = d.nsmallest(5, v)
    extremes = [[f'{r.settlement_date:%Y-%m-%d}', PERIOD_LABELS[min(r.settlement_period, 48) - 1], f'{r[v]:,.0f}', 'highest']
                for _, r in top.iterrows()] + \
               [[f'{r.settlement_date:%Y-%m-%d}', PERIOD_LABELS[min(r.settlement_period, 48) - 1], f'{r[v]:,.0f}', 'lowest']
                for _, r in low.iterrows()]
    year_rows = [[int(y), num(g[v].mean(), 0), num(g[v].max(), 0), num(g[v].min(), 0), num(g[v].std(), 0),
                  num(g[v].sum() * 0.5 / 1e6, 1)] for y, g in d.groupby(d['settlement_date'].dt.year)]
    charts = [
        line('National demand, daily mean', 'A clear annual cycle: winter demand is about half as high again as summer.',
             days(daily.index), [('Daily mean', daily['mean']), ('28-day mean', rolling(daily['mean']))], 'MW',
             digits=0, wide=True),
        line('Daily peak and trough', 'The highest and lowest half-hour of each day.', days(daily.index),
             [('Daily peak', daily['max']), ('Daily minimum', daily['min'])], 'MW', digits=0, wide=True),
        hist('Distribution of half-hourly demand', 'Right-skewed: a long tail of cold winter evenings.', d[v], 'MW',
             bins=50, clip=None, digits=0),
        line('Average day by season', 'Mean by settlement period, UK local time.', PERIOD_LABELS, season_profile(d, v),
             'MW', digits=0),
        line('Average day: weekday, Saturday, Sunday', 'The morning ramp is a working-day feature.', PERIOD_LABELS,
             [(k, by_kind.loc[k].reindex(range(1, 49))) for k in ('Weekday', 'Saturday', 'Sunday')], 'MW', digits=0),
        month_period_heat(d.assign(v=d[v] / 1000), 'v', 'Demand by month and time of day',
                          'The evening peak moves earlier and grows as days shorten.', 'GW', digits=1),
        box('Demand by day of week', '', {DOW[i]: d.loc[weekday == i, v] for i in range(7)}, 'MW', digits=0),
        box('Demand by month of year', '', {MONTHS[m - 1]: d.loc[d['settlement_date'].dt.month == m, v] for m in range(1, 13)},
            'MW', digits=0),
        line('Autocorrelation of half-hourly demand, lags up to 7 days', 'Peaks at every 24 hours, strongest at one week.',
             [f'{lag_ / 2:g} h' for lag_ in range(1, 337)], [('Autocorrelation', ac[1:])], digits=3, wide=True),
        bar('When the daily peak falls', 'Settlement period of the highest half-hour of each day.', PERIOD_LABELS,
            [('Days', peak_period.value_counts().reindex(range(1, 49)).fillna(0).to_numpy())], 'days', digits=0),
        hist('Transmission demand minus national demand', 'ITSDO adds station load, pumping and interconnector exports to INDO.',
             gap, 'MW', bins=50, digits=0),
        hist('Delay between a period starting and its outturn being published', 'Minutes from start_time to publish_time.',
             lag, 'min', bins=40, clip=(0, 0.99), digits=0),
    ]
    checks = completeness(d) + [
        range_check(d, v, 0, 80000, 'demand_implausible'),
        check('Transmission demand below national demand', int((gap < 0).sum()), len(d),
              'good' if (gap < 0).sum() == 0 else 'warn', 'ITSDO should include INDO', bad=True),
        check('NULL transmission demand', int(d['initial_transmission_demand_outturn_mw'].isna().sum()), len(d), 'info', '', bad=True),
    ]
    tiles = [tile('Coverage', f'{d.settlement_date.min():%b %Y} – {d.settlement_date.max():%b %Y}', f'{len(daily):,} days'),
             tile('Mean demand', f'{d[v].mean() / 1000:.1f} GW', f'median {d[v].median() / 1000:.1f} GW'),
             tile('Peak', f'{d[v].max() / 1000:.1f} GW', f'{top.iloc[0].settlement_date:%d %b %Y}'),
             tile('Minimum', f'{d[v].min() / 1000:.1f} GW', f'{low.iloc[0].settlement_date:%d %b %Y}')]
    tables = [table('Demand by year', ['Year', 'Mean MW', 'Peak MW', 'Min MW', 'Std dev', 'Energy TWh'], year_rows,
                    '2023 and 2026 are part years.'),
              table('Extreme half-hours', ['Date', 'Local time', 'MW', ''], extremes)]
    section('demand_outturn', 'National demand outturn', 'Elexon INDO and ITSDO', 'settlement date and period',
            ('settlement_date', 'settlement_period'), d, tiles, checks, charts, tables)
    say('  acf 24h', round(ac[48], 3), '7d', round(ac[336], 3), 'gap mean', round(gap.mean()), 'lag median', lag.median())
    say('  weekday mean', round(d.loc[weekday < 5, v].mean()), 'sat', round(d.loc[weekday == 5, v].mean()), 'sun',
        round(d.loc[weekday == 6, v].mean()))
    say('  month mean', d.groupby(d.settlement_date.dt.month)[v].mean().round(0).to_dict())
    say('  peak period mode', PERIOD_LABELS[int(peak_period.mode()[0]) - 1], 'skew', round(d[v].skew(), 2))


# --- 3. wind_forecast ---------------------------------------------------------------------------------------

def horizon_table(frame, buckets, labels):
    frame = frame.assign(bucket=pd.cut(frame['horizon_h'], buckets, labels=labels, right=False))
    rows, stats = [], {}
    for label in labels:
        g = frame[frame['bucket'] == label]
        e = error_stats(g['forecast'], g['actual'])
        stats[label] = e
        if e['n']:
            rows.append([label, f"{e['n']:,}", e['bias'], e['mae'], e['cmae'], e['rmse'], f"{e['nmae_pct']}%", e['r']])
    return rows, stats


def wind_forecast():
    w = load('wind_forecast')
    w['horizon_h'] = (w['start_time'] - w['publish_time']).dt.total_seconds() / 3600
    wind = SHARED['gen_half']['wind']
    hourly_actual = wind.resample('1h').mean()
    w['actual'] = hourly_actual.reindex(w['start_time']).to_numpy()
    w['forecast'] = w['generation_mw']
    buckets = [-1e9, 0, 6, 12, 24, 36, 48, 60, 72, 1e9]
    labels = ['already started', '0–6 h', '6–12 h', '12–24 h', '24–36 h', '36–48 h', '48–60 h', '60–72 h', '72 h +']
    rows, stats = horizon_table(w, buckets, labels)
    per_publish = w.groupby('publish_time').size()
    publishes_per_day = per_publish.groupby(per_publish.index.date).size()
    ahead = w[w['horizon_h'].between(18, 42)].copy()
    ahead['abs_err'] = (ahead['forecast'] - ahead['actual']).abs()
    monthly = ahead.groupby(ahead['start_time'].dt.to_period('M')).agg(mae=('abs_err', 'mean'), actual=('actual', 'mean'))
    err = (ahead['forecast'] - ahead['actual']).dropna()
    hour_count = per_publish.groupby(per_publish.index.hour).size().reindex(range(24)).fillna(0)
    latest = w.sort_values('publish_time').groupby('start_time').tail(1).set_index('start_time')['generation_mw']
    daily_latest = latest.resample('D').mean()
    shown = [l for l in labels if stats[l]['n']]
    charts = [
        bar('Forecast error by lead time', 'Mean absolute error and bias (forecast − actual) against FUELHH wind, '
            'hourly mean of the two half-hours.', shown,
            [('Mean absolute error', [stats[l]['mae'] for l in shown]), ('Bias', [stats[l]['bias'] for l in shown]),
             ('Mean absolute error, bias removed', [stats[l]['cmae'] for l in shown])],
            'MW', digits=0, wide=True),
        scatter('Forecast against actual, 18–42 hours ahead', 'The day-ahead range a model at the 18:00 cutoff would use.',
                ahead['forecast'], ahead['actual'], 'Forecast wind (MW)', 'Actual wind (MW)', digits=0),
        hist('Day-ahead forecast error', 'Forecast − actual, 18–42 hours ahead.', err, 'MW', bins=60, digits=0),
        line('Day-ahead error by month', 'Mean absolute error, 18–42 hours ahead, with mean actual wind for scale.',
             [str(p) for p in monthly.index], [('Mean absolute error', monthly['mae']), ('Mean actual wind', monthly['actual'])],
             'MW', digits=0, wide=True),
        hist('Lead time of every forecast row', 'Hours from publish_time to start_time. Negative: the hour had already begun.',
             w['horizon_h'], 'h', bins=60, clip=None, digits=1),
        bar('Publishes by hour of day (UTC)', 'How many forecast runs were published in each hour.', range(24),
            [('Publishes', hour_count.to_numpy())], '', digits=0),
        hist('Rows per publish', 'How many hours ahead one forecast run covers.', per_publish, 'rows', bins=40, clip=None, digits=0),
        hist('Distribution of forecast wind', 'All forecast values.', w['generation_mw'], 'MW', bins=50, clip=None, digits=0),
        line('Latest forecast, daily mean', 'For each hour, the last forecast published.', days(daily_latest.index),
             [('Forecast wind', daily_latest), ('28-day mean', rolling(daily_latest))], 'MW', digits=0, wide=True),
    ]
    hours = pd.date_range(w['start_time'].min(), w['start_time'].max(), freq='1h')
    checks = [
        range_check(w, 'generation_mw', 0, 50000, 'generation_implausible'),
        check('Target hours with at least one forecast', w['start_time'].nunique(), len(hours),
              'good' if w['start_time'].nunique() == len(hours) else 'warn', 'hourly grid between first and last start_time'),
        check('Rows published after their hour began', int((w['horizon_h'] < 0).sum()), len(w), 'info',
              'not forecasts in the strict sense; a cutoff filter removes them'),
        check('Days with at least one publish', len(publishes_per_day),
              (per_publish.index.max().normalize() - per_publish.index.min().normalize()).days + 1,
              'good', f'median {publishes_per_day.median():.0f} publishes a day'),
        check('start_time not on the hour', int((w['start_time'].dt.minute != 0).sum()), len(w), 'good', '', bad=True),
    ]
    tiles = [tile('Coverage', f'{w.start_time.min():%b %Y} – {w.start_time.max():%b %Y}', 'by forecast start time'),
             tile('Forecast runs', fmt_int(len(per_publish)), f'median {per_publish.median():.0f} hours each'),
             tile('Day-ahead error', f"{error_stats(ahead['forecast'], ahead['actual'])['mae']:,.0f} MW",
                  f"{error_stats(ahead['forecast'], ahead['actual'])['nmae_pct']}% of mean wind"),
             tile('Longest lead time', f"{w['horizon_h'].max():.0f} h", f"median {w['horizon_h'].median():.0f} h")]
    tables = [table('Accuracy by lead time', ['Lead time', 'Rows', 'Bias MW', 'MAE MW', 'MAE MW, bias removed', 'RMSE MW', 'MAE / mean', 'r'], rows,
                    'Bias is forecast minus actual. Actual is FUELHH wind, which excludes embedded wind.')]
    section('wind_forecast', 'Wind generation forecast', 'Elexon WINDFOR', 'publish time and forecast start time',
            ('publish_time', 'start_time'), w.drop(columns=['actual', 'forecast']), tiles, checks, charts, tables)
    say('  publish hours', hour_count.astype(int).to_dict())
    say('  publishes/day', publishes_per_day.describe().round(1).to_dict(), 'horizon quantiles',
        w['horizon_h'].quantile([0, .01, .25, .5, .75, .99, 1]).round(1).to_dict())
    say('  day-ahead', error_stats(ahead['forecast'], ahead['actual']))
    SHARED['windfor'] = w


# --- 4. demand_forecast -------------------------------------------------------------------------------------

def demand_forecast():
    d = load('demand_forecast')
    d['horizon_h'] = (d['start_time'] - d['publish_time']).dt.total_seconds() / 3600
    d['actual'] = SHARED['demand_half'].reindex(d['start_time']).to_numpy()
    d['forecast'] = d['demand_mw']
    buckets = [-1e9, 0, 6, 12, 24, 36, 48, 1e9]
    labels = ['already started', '0–6 h', '6–12 h', '12–24 h', '24–36 h', '36–48 h', '48 h +']
    rows, stats = horizon_table(d, buckets, labels)
    per_publish = d.groupby('publish_time').size()
    publish_days = per_publish.groupby(per_publish.index.date).size()
    # What gold uses: for target day D, the latest forecast published before 18:00 UTC on D-1
    cutoff = d['settlement_date'] - pd.Timedelta(hours=6)
    known = d[d['publish_time'] < cutoff].sort_values('publish_time').groupby(['settlement_date', 'settlement_period']).tail(1)
    known = known.assign(err=known['forecast'] - known['actual'])
    known['ape'] = 100 * known['err'].abs() / known['actual']
    by_period = known[known['settlement_period'] <= 48].groupby('settlement_period')['err'].agg(['mean', lambda e: e.abs().mean()])
    monthly = known.groupby(known['settlement_date'].dt.to_period('M'))['ape'].mean()
    minute = per_publish.groupby(per_publish.index.hour * 60 + per_publish.index.minute).size()
    target_days = known.groupby('settlement_date')['settlement_period'].nunique()
    all_days = pd.date_range(d['settlement_date'].min(), d['settlement_date'].max())
    e = error_stats(known['forecast'], known['actual'])
    shown = [l for l in labels if stats[l]['n']]
    dow = known['settlement_date'].dt.dayofweek
    charts = [
        scatter('Day-ahead forecast against outturn', 'The latest forecast published before 18:00 UTC the day before.',
                known['forecast'], known['actual'], 'Forecast demand (MW)', 'Demand outturn (MW)', digits=0),
        hist('Day-ahead forecast error', 'Forecast − outturn.', known['err'], 'MW', bins=60, digits=0),
        line('Day-ahead error by time of day', 'Bias shows where the forecast runs high or low; UK local time.',
             PERIOD_LABELS, [('Mean absolute error', by_period.iloc[:, 1].reindex(range(1, 49))),
                             ('Bias', by_period['mean'].reindex(range(1, 49)))], 'MW', digits=0),
        line('Day-ahead percentage error by month', 'Mean absolute percentage error.', [str(p) for p in monthly.index],
             [('MAPE', monthly)], '%', digits=2, wide=True),
        bar('Error by lead time', 'Mean absolute error and bias (forecast − outturn).', shown,
            [('Mean absolute error', [stats[l]['mae'] for l in shown]), ('Bias', [stats[l]['bias'] for l in shown])],
            'MW', digits=0),
        box('Day-ahead error by day of week', 'Forecast − outturn.', {DOW[i]: known.loc[dow == i, 'err'] for i in range(7)},
            'MW', digits=0),
        hist('Lead time of every forecast row', 'Hours from publish_time to start_time.', d['horizon_h'], 'h', bins=60,
             clip=None, digits=1),
        bar('Publish time within the 17:00–18:00 UTC window', 'Minute of the hour at which each run was published.',
            [f'{m // 60:02d}:{m % 60:02d}' for m in minute.index], [('Publishes', minute.to_numpy())], '', digits=0),
        hist('Rows per publish', 'Settlement periods covered by one forecast run.', per_publish, 'rows', bins=40, clip=None, digits=0),
        hist('Distribution of forecast demand', '', d['demand_mw'], 'MW', bins=50, clip=None, digits=0),
    ]
    checks = [
        check('Target days with a full day-ahead forecast', int((target_days >= 46).sum()), len(all_days),
              'good' if (target_days >= 46).sum() >= len(all_days) - 5 else 'warn',
              f'{len(all_days) - len(target_days)} days have no forecast published before the cutoff'),
        check('Days with a publish in the window', len(publish_days),
              (per_publish.index.max().normalize() - per_publish.index.min().normalize()).days + 1,
              'good' if len(publish_days) >= 1095 else 'warn', f'median {publish_days.median():.0f} publishes a day'),
        range_check(d, 'demand_mw', 0, 80000, 'demand_implausible'),
        check('Rows with a boundary other than N', int((d['boundary'] != 'N').sum()), len(d), 'good', 'N = national', bad=True),
        check('start_time disagrees with date + period',
              int((d['start_time'] != period_start(d['settlement_date'], d['settlement_period'])).sum()), len(d),
              'good', 'the soft rule start_time_mismatch', bad=True),
        check('Rows published after their period began', int((d['horizon_h'] < 0).sum()), len(d), 'info', ''),
    ]
    tiles = [tile('Coverage', f'{d.settlement_date.min():%b %Y} – {d.settlement_date.max():%b %Y}', 'by settlement date'),
             tile('Forecast runs', fmt_int(len(per_publish)), f'median {per_publish.median():.0f} periods each'),
             tile('Day-ahead MAPE', f"{known['ape'].mean():.2f}%", f"MAE {e['mae']:,.0f} MW, bias {e['bias']:+,.0f} MW"),
             tile('Correlation with outturn', f"{e['r']:.3f}", 'day-ahead forecast')]
    tables = [table('Accuracy by lead time', ['Lead time', 'Rows', 'Bias MW', 'MAE MW', 'MAE MW, bias removed', 'RMSE MW', 'MAE / mean', 'r'], rows,
                    'Bias is forecast minus outturn (INDO).')]
    section('demand_forecast', 'National demand forecast', 'Elexon NDF, 17:00–18:00 UTC publishes',
            'publish time, settlement date, period and boundary',
            ('publish_time', 'settlement_date', 'settlement_period', 'boundary'),
            d.drop(columns=['actual', 'forecast']), tiles, checks, charts, tables)
    say('  day-ahead', e, 'mape', round(known['ape'].mean(), 2), 'minute', minute.to_dict())
    say('  rows per publish', per_publish.describe().round(1).to_dict(), 'publishes/day', publish_days.value_counts().to_dict())
    say('  bias by period min/max', by_period['mean'].idxmin(), round(by_period['mean'].min()), by_period['mean'].idxmax(),
        round(by_period['mean'].max()), 'horizon', d['horizon_h'].quantile([0, .5, 1]).round(1).to_dict())
    SHARED['ndf_known'] = known.groupby('settlement_date')['forecast'].mean()


# --- 5 and 6. wind and solar: forecast and actual ---------------------------------------------------------------

PSR = ('Solar', 'Wind Offshore', 'Wind Onshore')


def wind_solar_actual():
    a = load('wind_solar_actual')
    wide = a.pivot(index=['settlement_date', 'settlement_period'], columns='psr_type', values='quantity_mw')
    dates = wide.index.get_level_values(0)
    weekly = wide.groupby(dates).mean().resample('W').mean().iloc[1:-1]
    a['start'] = period_start(a['settlement_date'], a['settlement_period'])
    total_wind = (wide['Wind Offshore'] + wide['Wind Onshore'])
    start = period_start(pd.Series(dates), pd.Series(wide.index.get_level_values(1))).to_numpy()
    gen = SHARED['gen_half']
    compare = pd.DataFrame({'agws': total_wind.to_numpy(), 'solar': wide['Solar'].to_numpy()}, index=start)
    compare['fuelhh'] = gen['wind'].reindex(compare.index).to_numpy()
    SHARED['agws'] = compare
    SHARED['agws_keyed'] = a.set_index(['psr_type', 'settlement_date', 'settlement_period'])['quantity_mw']
    lag = (a['publish_time'] - a['start_time']).dt.total_seconds() / 3600
    year_rows = []
    for psr in PSR:
        for y, g in a[a['psr_type'] == psr].groupby(a['settlement_date'].dt.year):
            year_rows.append([psr, int(y), num(g['quantity_mw'].mean(), 0), num(g['quantity_mw'].max(), 0),
                              num(g['quantity_mw'].sum() * 0.5 / 1e6, 2)])
    flat = wide.reset_index()
    sp = wide.index.get_level_values(1)
    profile_lines = [(p, wide[p][sp <= 48].groupby(sp[sp <= 48]).mean().reindex(range(1, 49))) for p in PSR]
    monthly = wide.groupby(dates.to_period('M')).mean()
    ratio = (compare['agws'] / compare['fuelhh'])[compare['fuelhh'] > 1000]
    fc = load('wind_solar_forecast')
    fc = fc[(fc['process_type'] == 'Day ahead') & (fc['psr_type'] != 'Solar')]
    fc_total = fc.groupby(['settlement_date', 'settlement_period'])['quantity_mw'].sum()
    windfor = SHARED['windfor']
    views = pd.DataFrame({
        'FUELHH wind': gen['wind'].groupby(gen.index.to_period('M')).mean(),
        'AGWS onshore + offshore': compare['agws'].groupby(compare.index.to_period('M')).mean(),
        'DGWS day-ahead forecast': fc_total.groupby(fc_total.index.get_level_values(0).to_period('M')).mean(),
        'WINDFOR forecast': windfor.groupby(windfor['start_time'].dt.to_period('M'))['generation_mw'].mean()})
    views = views.loc['2023-11':'2026-10']
    monthly_ratio = views['AGWS onshore + offshore'] / views['FUELHH wind']
    bad_months = monthly_ratio[monthly_ratio < 1.0].index
    SHARED['agws_bad_months'] = bad_months
    good = ~compare.index.to_period('M').isin(bad_months)
    ratio_good = ratio[~ratio.index.to_period('M').isin(bad_months)]
    charts = [
        line('Four views of GB wind, monthly mean', 'AGWS falls far below the other three for part of 2024: the actuals feed '
             'under-reports there, mostly offshore.', [str(p) for p in views.index], [(c, views[c]) for c in views.columns], 'MW',
             digits=0, wide=True),
        bar('AGWS wind as a multiple of FUELHH wind, by month', 'About 1.2 when both feeds are healthy. Below 1.0 the AGWS '
            'figure cannot be right.', [str(p) for p in monthly_ratio.index], [('AGWS ÷ FUELHH', monthly_ratio.to_numpy())], '×',
            digits=2, wide=True),
        line('Actual output by technology, weekly mean', 'Solar peaks in summer as wind dips: partly complementary.',
             days(weekly.index), [(p, weekly[p]) for p in PSR], 'MW', digits=0, wide=True),
        bar('Monthly mean output by technology', '', [str(p) for p in monthly.index], [(p, monthly[p]) for p in PSR], 'MW',
            stack=True, digits=0, wide=True),
        box('Spread of half-hourly output by technology', '', {p: wide[p] for p in PSR}, 'MW', digits=0),
        line('Average day by technology', 'Mean by settlement period, UK local time.', PERIOD_LABELS, profile_lines, 'MW', digits=0),
        month_period_heat(flat, 'Solar', 'Solar output by month and time of day', 'The shape of daylight.', 'MW'),
        month_period_heat(flat.assign(v=flat['Wind Offshore'] + flat['Wind Onshore']), 'v',
                          'Wind output by month and time of day', 'Onshore plus offshore.', 'MW'),
        scatter('AGWS wind against FUELHH wind', 'Two Elexon feeds measuring the same transmission-connected fleet.',
                compare['agws'], compare['fuelhh'], 'AGWS onshore + offshore (MW)', 'FUELHH WIND (MW)', digits=0),
        hist('AGWS wind as a share of FUELHH wind', 'Half-hours with FUELHH wind above 1,000 MW.', 100 * ratio, '%', bins=50),
        hist('Delay between a period starting and AGWS publishing it', 'Hours from start_time to publish_time.', lag, 'h',
             bins=50, clip=(0, 0.99), digits=1),
        corr_heat(wide.rename(columns=str), 'Correlation between technologies, half-hourly',
                  'Onshore and offshore wind move together; solar is nearly independent of both.'),
    ]
    checks = completeness(a, group='psr_type') + [
        range_check(a, 'quantity_mw', -500, 50000, 'quantity_implausible'),
        check('Negative quantities', int((a['quantity_mw'] < 0).sum()), len(a), 'info',
              f"minimum {a['quantity_mw'].min():.1f} MW; the soft rule negative_quantity"),
        check('Months where AGWS wind falls below FUELHH wind', len(bad_months), len(monthly_ratio),
              'bad' if len(bad_months) else 'good',
              (f'{bad_months.min()} to {bad_months.max()}: the feed under-reports and no row rule can see it' if len(bad_months) else ''),
              bad=True),
        check('Periods with all three technologies', int(wide.notna().all(axis=1).sum()), len(wide),
              'good' if wide.notna().all(axis=1).all() else 'warn', ''),
        check('Solar above zero at night (22:00–03:00 local)',
              int(((wide['Solar'] > 0) & ((sp >= 45) | (sp <= 6))).sum()), int(((sp >= 45) | (sp <= 6)).sum()), 'info', '',
              bad=True),
    ]
    tiles = [tile('Coverage', f'{a.settlement_date.min():%b %Y} – {a.settlement_date.max():%b %Y}', f'{dates.nunique():,} days'),
             tile('Mean wind', f'{total_wind.mean() / 1000:.1f} GW', f'peak {total_wind.max() / 1000:.1f} GW'),
             tile('Mean solar', f"{wide['Solar'].mean() / 1000:.2f} GW", f"peak {wide['Solar'].max() / 1000:.1f} GW"),
             tile('AGWS vs FUELHH wind', f'{ratio_good.median():.2f}×', f'healthy months; {len(bad_months)} months fall below 1.0')]
    tables = [table('Output by technology and year', ['Technology', 'Year', 'Mean MW', 'Max MW', 'Energy TWh'], year_rows)]
    section('wind_solar_actual', 'Actual wind and solar generation', 'Elexon AGWS', 'technology, settlement date and period',
            ('psr_type', 'settlement_date', 'settlement_period'), a.drop(columns=['start']), tiles, checks, charts, tables)
    say('  monthly ratio', monthly_ratio.round(2).to_dict())
    say('  bad months', list(map(str, bad_months)), 'ratio good median', ratio_good.median().round(3), 'corr good',
        compare.loc[good, ['agws', 'fuelhh']].corr().iloc[0, 1].round(4))
    short_days = a[a.psr_type == 'Solar'].groupby('settlement_date').size().loc[lambda x: x < 46]
    say('  days short of periods by quarter', short_days.groupby(short_days.index.to_period('Q')).agg(['count', 'sum']).to_dict())
    say('  ratio median', ratio.median().round(3), 'corr', compare[['agws', 'fuelhh']].corr().iloc[0, 1].round(4), 'lag',
        lag.describe().round(2).to_dict())
    say('  psr corr', wide.corr().round(2).to_dict())


def wind_solar_forecast():
    f = load('wind_solar_forecast')
    f['actual'] = SHARED['agws_keyed'].reindex(pd.MultiIndex.from_frame(f[['psr_type', 'settlement_date', 'settlement_period']])).to_numpy()
    f['horizon_h'] = (f['start_time'] - f['publish_time']).dt.total_seconds() / 3600
    f['err'] = f['quantity_mw'] - f['actual']
    process = ('Day ahead', 'Intraday process', 'Intraday total')
    # Score only where the actuals can be trusted (see wind_solar_actual: AGWS under-reports for some months)
    bad_months = SHARED['agws_bad_months']
    f['scorable'] = ~f['settlement_date'].dt.to_period('M').isin(bad_months)
    gap_note = f'Months where AGWS under-reports ({bad_months.min()} to {bad_months.max()}) are left out.' if len(bad_months) else ''
    rows, grid = [], {}
    for p in process:
        for psr in PSR:
            g = f[(f['process_type'] == p) & (f['psr_type'] == psr)]
            every = error_stats(g['quantity_mw'], g['actual'])
            e = error_stats(g.loc[g['scorable'], 'quantity_mw'], g.loc[g['scorable'], 'actual'])
            grid[(p, psr)] = e
            rows.append([p, psr, f"{e['n']:,}", e.get('bias'), e.get('mae'), e.get('rmse'),
                         f"{e.get('nmae_pct')}%", e.get('r'), f"{every.get('nmae_pct')}%", f"{g['horizon_h'].median():.0f} h"])
    da = f[f['process_type'] == 'Day ahead']
    da_wide = da.pivot_table(index=['settlement_date', 'settlement_period'], columns='psr_type', values='quantity_mw', aggfunc='last')
    da_dates = da_wide.index.get_level_values(0)
    weekly = da_wide.groupby(da_dates).mean().resample('W').mean().iloc[1:-1]
    da = da.assign(abs_err=da['err'].abs())
    monthly = da.groupby([da['settlement_date'].dt.to_period('M'), 'psr_type']).agg(mae=('abs_err', 'mean'), act=('actual', 'mean'))
    nmae = (100 * monthly['mae'] / monthly['act']).unstack()
    wind = (da[(da['psr_type'] != 'Solar') & da['scorable']]
            .groupby(['settlement_date', 'settlement_period'])[['quantity_mw', 'actual']].sum(min_count=2))
    hours = f.drop_duplicates(['document_id', 'process_type'])
    hour_by_process = hours.groupby(['process_type', hours['publish_time'].dt.hour]).size().unstack(0).reindex(range(24)).fillna(0)
    periods = da.groupby(['settlement_date', 'psr_type'])['settlement_period'].nunique().unstack()
    expected = periods_expected(periods.index)
    full = (periods.eq(expected, axis=0)).all(axis=1)
    all_days = pd.date_range(da['settlement_date'].min(), da['settlement_date'].max())
    solar_day = da[(da['psr_type'] == 'Solar') & (da['actual'] > 100)]
    SHARED['dgws_daily'] = da_wide.groupby(da_dates).mean().sum(axis=1, min_count=3)
    charts = [
        bar('Error by process and technology', 'Mean absolute error as a share of mean actual output (AGWS). ' + gap_note,
            list(PSR), [(p, [grid[(p, psr)].get('nmae_pct') for psr in PSR]) for p in process], '%', wide=True),
        scatter('Day-ahead wind forecast against actual', 'Onshore plus offshore, per settlement period. ' + gap_note,
                wind['quantity_mw'], wind['actual'], 'Forecast wind (MW)', 'Actual wind, AGWS (MW)', digits=0),
        scatter('Day-ahead solar forecast against actual', 'Per settlement period, daylight only (actual above 100 MW).',
                solar_day['quantity_mw'], solar_day['actual'], 'Forecast solar (MW)', 'Actual solar, AGWS (MW)', digits=0),
        line('Day-ahead error by month and technology', 'Mean absolute error as a share of the month mean actual. '
             'The 2024 spike is the AGWS actuals failing, not the forecast.',
             [str(p) for p in nmae.index], [(p, nmae[p]) for p in PSR], '%', wide=True),
        box('Day-ahead error by technology', 'Forecast − actual. ' + gap_note,
            {p: da.loc[(da['psr_type'] == p) & da['scorable'], 'err'] for p in PSR}, 'MW', digits=0),
        line('Day-ahead forecast by technology, weekly mean', '', days(weekly.index), [(p, weekly[p]) for p in PSR], 'MW',
             digits=0, wide=True),
        bar('Publish hour by process (UTC)', 'Documents published in each hour.', range(24),
            [(p, hour_by_process[p].to_numpy()) for p in process], 'documents', digits=0),
        box('Lead time by process', 'Hours from publish_time to the period\'s start.',
            {p: f.loc[f['process_type'] == p, 'horizon_h'] for p in process}, 'h'),
        month_period_heat(da[da['psr_type'] == 'Solar'], 'quantity_mw', 'Day-ahead solar forecast by month and time of day', '', 'MW'),
        {**month_period_heat(da[da['psr_type'] == 'Solar'], 'err', 'Day-ahead solar bias by month and time of day',
                             'Forecast − actual: red where the forecast runs low, blue where it runs high.', 'MW'),
         'diverging': True},
    ]
    checks = [
        check('Days with a complete day-ahead forecast, all three technologies', int(full.sum()), len(all_days),
              'good' if full.sum() >= len(all_days) - 3 else 'warn', f'{len(all_days) - len(periods)} days have none at all'),
        check('Rows that match an AGWS actual', int(f['actual'].notna().sum()), len(f), 'info', 'the rest cannot be scored'),
        range_check(f, 'quantity_mw', -500, 50000, 'quantity_implausible'),
        check('Negative quantities', int((f['quantity_mw'] < 0).sum()), len(f), 'good', 'the soft rule negative_quantity', bad=True),
        check('Documents revised (revision above 1)', int((f['document_revision_number'] > 1).sum()), len(f), 'info', ''),
        check('Periods outside 1..46/48/50 for their day',
              int(((f['settlement_period'] < 1) | (f['settlement_period'] >
                                                     periods_expected(f['settlement_date'].unique()).reindex(f['settlement_date']).to_numpy())).sum()),
              len(f), 'good', 'the hard rule period_out_of_range', bad=True),
        check('start_time disagrees with date + period',
              int((f['start_time'] != period_start(f['settlement_date'], f['settlement_period'])).sum()), len(f), 'good',
              'the soft rule start_time_mismatch', bad=True),
    ]
    e = error_stats(wind['quantity_mw'], wind['actual'])
    tiles = [tile('Coverage', f'{f.settlement_date.min():%b %Y} – {f.settlement_date.max():%b %Y}', 'by settlement date'),
             tile('Documents', fmt_int(f['document_id'].nunique()), '3 processes × 3 technologies'),
             tile('Day-ahead wind error', f"{e['mae']:,.0f} MW", f"{e['nmae_pct']}% of mean wind, r {e['r']}"),
             tile('Day-ahead solar error', f"{grid[('Day ahead', 'Solar')]['mae']:,.0f} MW",
                  f"{grid[('Day ahead', 'Solar')]['nmae_pct']}% of mean solar")]
    tables = [table('Accuracy by process and technology',
                    ['Process', 'Technology', 'Rows scored', 'Bias MW', 'MAE MW', 'RMSE MW', 'MAE / mean', 'r',
                     'MAE / mean, all months', 'Median lead'], rows,
                    'Scored against AGWS actuals, forecast minus actual. ' + gap_note + ' The last-but-one column keeps them in.')]
    section('wind_solar_forecast', 'Wind and solar generation forecast', 'Elexon DGWS',
            'forecast document, process, technology and settlement period',
            ('document_id', 'process_type', 'psr_type', 'settlement_date', 'settlement_period'),
            f.drop(columns=['actual', 'err', 'scorable']), tiles, checks, charts, tables)
    say('  da wind total', e, 'periods per da doc', da.groupby('document_id').size().describe().round(0).to_dict())
    say('  horizon by process', f.groupby('process_type')['horizon_h'].describe().round(1).to_dict())
    say('  publish hours', hour_by_process.astype(int).loc[lambda x: x.sum(axis=1) > 0].to_dict())
    say('  missing da days', sorted(set(all_days) - set(periods.index))[:10], 'not full', list(full[~full].index.strftime('%Y-%m-%d'))[:10])


# --- 7. system_prices ---------------------------------------------------------------------------------------

def system_prices():
    p = load('system_prices')
    v = 'system_sell_price_gbp_mwh'
    daily = p.groupby('settlement_date')[v].agg(['mean', 'std', 'min', 'max'])
    SHARED['price_half'] = p.set_index('start_time')[v]
    SHARED['price_daily'] = daily['mean']
    SHARED['niv_half'] = p.set_index('start_time')['net_imbalance_volume_mwh']
    month = p['settlement_date'].dt.to_period('M')
    negative = 100 * (p[v] < 0).groupby(month).mean()
    in_range = p[p['settlement_period'] <= 48]
    q = in_range.groupby('settlement_period')[v].quantile([.05, .5, .95]).unstack()
    mean_p = in_range.groupby('settlement_period')[v].mean()
    codes = p['price_derivation_code'].value_counts()
    top = p.nlargest(8, v)
    low = p.nsmallest(4, v)
    spikes = [[f'{r.settlement_date:%Y-%m-%d}', PERIOD_LABELS[min(r.settlement_period, 48) - 1], f'{r[v]:,.2f}',
               f'{r.net_imbalance_volume_mwh:,.0f}'] for _, r in pd.concat([top, low]).iterrows()]
    year_rows = [[int(y), num(g[v].mean()), num(g[v].median()), num(g[v].std()), num(g[v].min()), num(g[v].max()),
                  f'{100 * (g[v] < 0).mean():.1f}%', f'{100 * (g[v] > 200).mean():.1f}%']
                 for y, g in p.groupby(p['settlement_date'].dt.year)]
    niv = p['net_imbalance_volume_mwh']
    short = p[niv > 0][v]
    long_ = p[niv < 0][v]
    ac = acf(p.set_index('start_time')[v].resample('30min').mean().interpolate().dropna(), nlags=96, fft=True)
    charts = [
        line('System price, daily mean', 'The single imbalance price paid or received for being out of balance.',
             days(daily.index), [('Daily mean', daily['mean']), ('28-day mean', rolling(daily['mean']))], '£/MWh', wide=True),
        hist('Distribution of the half-hourly system price', 'Two humps: one when the system is long, one when it is short.',
             p[v], '£/MWh', bins=70),
        box('Price when the system is short or long', 'Short: net imbalance volume above zero, the operator buys energy.',
            {'System short': short, 'System long': long_}, '£/MWh'),
        scatter('Price against net imbalance volume', 'Positive volume: the system was short.', niv, p[v],
                'Net imbalance volume (MWh)', 'System price (£/MWh)'),
        line('Price by time of day', 'Mean, median and the 5th–95th percentile range; UK local time.', PERIOD_LABELS,
             [('95th percentile', q[.95].reindex(range(1, 49))), ('Mean', mean_p.reindex(range(1, 49))),
              ('Median', q[.5].reindex(range(1, 49))), ('5th percentile', q[.05].reindex(range(1, 49)))], '£/MWh'),
        month_period_heat(p, v, 'System price by month and time of day', 'Winter evening peaks cost most.', '£/MWh'),
        bar('Share of half-hours with a negative price, by month', 'Negative prices cluster in windy, sunny, low-demand months.',
            [str(m) for m in negative.index], [('Negative price', negative.to_numpy())], '%', wide=True),
        line('Within-day price volatility', 'Standard deviation of the 48 prices of each day.', days(daily.index),
             [('Daily std dev', daily['std']), ('28-day mean', rolling(daily['std']))], '£/MWh', wide=True),
        hist('Distribution of net imbalance volume', 'Centred just below zero: the system is long slightly more often.', niv,
             'MWh', bins=60, digits=0),
        bar('Autocorrelation of the half-hourly price, lags up to 48 hours', 'Prices persist for a few hours, then decay.',
            [f'{lag_ / 2:g} h' for lag_ in range(1, 97)], [('Autocorrelation', ac[1:])], digits=3, wide=True),
        bar('Price derivation code', 'How Elexon derived each price.', codes.index, [('Half-hours', codes.to_numpy())], '',
            horizontal=True, digits=0),
        box('System price by year', '', {str(y): g[v] for y, g in p.groupby(p['settlement_date'].dt.year)}, '£/MWh'),
    ]
    checks = completeness(p) + [
        range_check(p, v, -10000, 20000, 'price_implausible'),
        check('Sell price differs from buy price', int((p[v] != p['system_buy_price_gbp_mwh']).sum()), len(p),
              'good', 'GB has had a single imbalance price since 2015', bad=True),
        check('Negative prices', int((p[v] < 0).sum()), len(p), 'info', f'lowest {p[v].min():,.2f} £/MWh'),
        check('Prices above £200/MWh', int((p[v] > 200).sum()), len(p), 'info', f'highest {p[v].max():,.2f} £/MWh'),
        check('Price exactly zero', int((p[v] == 0).sum()), len(p), 'info', ''),
        check('NULL reserve scarcity price', int(p['reserve_scarcity_price_gbp_mwh'].isna().sum()), len(p), 'info',
              'only set when reserve is scarce'),
        check('NULL replacement price', int(p['replacement_price_gbp_mwh'].isna().sum()), len(p), 'info', ''),
        check('BSAD defaulted', int(p['bsad_defaulted'].fillna(False).astype(bool).sum()), len(p), 'info', ''),
    ]
    tiles = [tile('Coverage', f'{p.settlement_date.min():%b %Y} – {p.settlement_date.max():%b %Y}', f'{len(daily):,} days'),
             tile('Mean price', f'£{p[v].mean():.2f}', f'median £{p[v].median():.2f} per MWh'),
             tile('Range', f'£{p[v].min():,.0f} to £{p[v].max():,.0f}', f'std dev £{p[v].std():.0f}'),
             tile('Negative prices', f'{100 * (p[v] < 0).mean():.1f}%', 'of half-hours')]
    tables = [table('Price by year', ['Year', 'Mean', 'Median', 'Std dev', 'Min', 'Max', 'Negative', 'Above £200'], year_rows,
                    '£/MWh. 2023 and 2026 are part years.'),
              table('Most extreme half-hours', ['Date', 'Local time', '£/MWh', 'Net imbalance MWh'], spikes)]
    section('system_prices', 'System (imbalance) prices', 'Elexon DISEBSP', 'settlement date and period',
            ('settlement_date', 'settlement_period'), p, tiles, checks, charts, tables)
    say('  short mean', round(short.mean(), 1), 'long mean', round(long_.mean(), 1), 'short share',
        round(100 * (niv > 0).mean(), 1), 'acf 30m/3h/24h', np.round(ac[[1, 6, 48]], 2), 'codes', codes.to_dict())
    say('  skew', round(p[v].skew(), 2), 'kurt', round(p[v].kurt(), 1), 'negative by month-of-year',
        (100 * (p[v] < 0).groupby(p.settlement_date.dt.month).mean()).round(1).to_dict())
    say('  by period mean min/max', mean_p.idxmin(), round(mean_p.min(), 1), mean_p.idxmax(), round(mean_p.max(), 1))


# --- 8. market_index ----------------------------------------------------------------------------------------

def market_index():
    m = load('market_index')
    apx = m[m['data_provider'] == 'APXMIDP']
    n2 = m[m['data_provider'] == 'N2EXMIDP']
    daily = apx.groupby('settlement_date').agg(price=('price_gbp_mwh', 'mean'), volume=('volume_mwh', 'sum'))
    SHARED['apx_daily'] = daily['price']
    system = SHARED['price_half']
    both = pd.DataFrame({'apx': apx.set_index('start_time')['price_gbp_mwh']})
    both['system'] = system.reindex(both.index)
    traded = both[both['apx'] != 0]
    spread = (traded['system'] - traded['apx']).dropna()
    monthly = both.replace({'apx': {0: np.nan}}).resample('MS').mean()
    in_range = apx[apx['settlement_period'] <= 48]
    volume_profile = in_range.groupby('settlement_period')['volume_mwh'].mean()
    price_profile = in_range[in_range['price_gbp_mwh'] != 0].groupby('settlement_period')['price_gbp_mwh'].mean()
    provider_rows = [[name, f'{len(g):,}', f"{100 * (g['price_gbp_mwh'] == 0).mean():.1f}%",
                      f"{100 * (g['volume_mwh'] == 0).mean():.1f}%", num(g['price_gbp_mwh'].mean()),
                      num(g.loc[g['price_gbp_mwh'] != 0, 'price_gbp_mwh'].mean()), num(g['volume_mwh'].mean(), 0),
                      num(g['price_gbp_mwh'].min()), num(g['price_gbp_mwh'].max())]
                     for name, g in m.groupby('data_provider')]
    apx_traded = apx[apx['price_gbp_mwh'] != 0]
    year_rows = [[int(y), num(g['price_gbp_mwh'].mean()), num(g['price_gbp_mwh'].median()), num(g['price_gbp_mwh'].std()),
                  num(g['price_gbp_mwh'].min()), num(g['price_gbp_mwh'].max()), num(g['volume_mwh'].mean(), 0)]
                 for y, g in apx_traded.groupby(apx_traded['settlement_date'].dt.year)]
    charts = [
        line('APX market index price, daily mean', 'The price of short-term wholesale trades before each half-hour.',
             days(daily.index), [('Daily mean', daily['price']), ('28-day mean', rolling(daily['price']))], '£/MWh', wide=True),
        line('Market index against system price, monthly mean', 'The two track each other; the system price swings wider.',
             [d.strftime('%Y-%m') for d in monthly.index], [('APX market index', monthly['apx']), ('System price', monthly['system'])],
             '£/MWh', wide=True),
        hist('Distribution of the APX price', 'Half-hours with trades.', apx_traded['price_gbp_mwh'], '£/MWh', bins=60),
        scatter('System price against market index', 'Each point is one half-hour.', traded['apx'], traded['system'],
                'APX market index (£/MWh)', 'System price (£/MWh)'),
        hist('System price minus market index', 'What being out of balance cost relative to the market.', spread, '£/MWh', bins=70),
        line('Average traded volume by time of day', 'APX, mean MWh per half-hour; UK local time.', PERIOD_LABELS,
             [('Volume', volume_profile.reindex(range(1, 49)))], 'MWh', area=True, digits=0),
        line('Average APX price by time of day', 'UK local time.', PERIOD_LABELS,
             [('Price', price_profile.reindex(range(1, 49)))], '£/MWh'),
        month_period_heat(apx_traded, 'price_gbp_mwh', 'APX price by month and time of day', '', '£/MWh'),
        line('APX traded volume, daily total', '', days(daily.index), [('Daily volume', daily['volume']),
                                                                       ('28-day mean', rolling(daily['volume']))], 'MWh',
             digits=0, wide=True),
        hist('Distribution of APX half-hourly volume', '', apx['volume_mwh'], 'MWh', bins=50, digits=0),
    ]
    checks = completeness(m, group='data_provider') + [
        range_check(m, 'price_gbp_mwh', -10000, 20000, 'price_implausible'),
        check('Negative volume', int((m['volume_mwh'] < 0).sum()), len(m), 'good', 'the hard rule negative_volume', bad=True),
        check('N2EX rows with zero price and volume', int(((n2['price_gbp_mwh'] == 0) & (n2['volume_mwh'] == 0)).sum()), len(n2),
              'warn', 'N2EX carries no information here'),
        check('APX rows with zero price', int((apx['price_gbp_mwh'] == 0).sum()), len(apx), 'warn',
              'no trades in the half-hour: a zero, not a price', bad=True),
        check('APX negative prices', int((apx['price_gbp_mwh'] < 0).sum()), len(apx), 'info', ''),
        check('Unknown providers', int((~m['data_provider'].isin(['APXMIDP', 'N2EXMIDP'])).sum()), len(m), 'good', '', bad=True),
    ]
    tiles = [tile('Coverage', f'{m.settlement_date.min():%b %Y} – {m.settlement_date.max():%b %Y}', '2 providers'),
             tile('Mean APX price', f"£{apx_traded['price_gbp_mwh'].mean():.2f}", 'per MWh, half-hours with trades'),
             tile('Mean APX volume', f"{apx['volume_mwh'].mean():,.0f} MWh", 'per half-hour'),
             tile('Correlation with system price', f"{traded['apx'].corr(traded['system']):.2f}", 'half-hourly')]
    tables = [table('By provider', ['Provider', 'Rows', 'Zero price', 'Zero volume', 'Mean price', 'Mean price when traded',
                                    'Mean volume', 'Min price', 'Max price'], provider_rows),
              table('APX price by year', ['Year', 'Mean', 'Median', 'Std dev', 'Min', 'Max', 'Mean volume MWh'], year_rows,
                    '£/MWh, half-hours with trades.')]
    section('market_index', 'Market index price', 'Elexon MID', 'data provider, settlement date and period',
            ('data_provider', 'settlement_date', 'settlement_period'), m, tiles, checks, charts, tables)
    say('  spread', spread.describe().round(2).to_dict(), 'apx std', round(apx_traded.price_gbp_mwh.std(), 1),
        'system std', round(system.std(), 1))
    say('  volume profile min/max', volume_profile.idxmin(), round(volume_profile.min()), volume_profile.idxmax(), round(volume_profile.max()))


# --- 9. remit_outages ---------------------------------------------------------------------------------------

def remit_outages():
    r = load('remit_outages')
    cur = load('remit_outages_current')
    per_message = r.groupby('mrid')['revision_number'].agg(['count', 'max'])
    month = r['publish_time'].dt.to_period('M')
    kind = r['unavailability_type'].fillna('Not stated')
    monthly = r.groupby([month, kind]).size().unstack().fillna(0)
    fuel = cur['fuel_type'].fillna('Not stated').value_counts()
    duration = (cur['event_end_time'] - cur['event_start_time']).dt.total_seconds() / 3600
    lead = (r['event_start_time'] - r['publish_time']).dt.total_seconds() / 3600
    planned = cur['unavailability_type'] == 'Planned'
    log_dur = np.log10(duration[duration > 0])
    revisions = per_message['count'].clip(upper=15).value_counts().sort_index()
    hour = r.groupby(r['publish_time'].dt.hour).size().reindex(range(24)).fillna(0)
    dow = r.groupby(r['publish_time'].dt.dayofweek).size().reindex(range(7)).fillna(0)
    assets = (cur.groupby('asset_id').agg(notices=('mrid', 'count'), fuel=('fuel_type', 'first'),
                                          normal=('normal_capacity_mw', 'max'), unavailable=('unavailable_capacity_mw', 'mean'))
              .sort_values('notices', ascending=False).head(12))
    asset_rows = [[i, a.fuel or '', f'{a.notices:,}', num(a.normal, 0), num(a.unavailable, 0)] for i, a in assets.iterrows()]
    participants = cur['participant_id'].value_counts().head(12)
    # Capacity out per day, from the latest revision of each active notice
    active = cur[(cur['event_status'] == 'Active') & cur['event_start_time'].notna() & cur['unavailable_capacity_mw'].notna()
                 & (cur['event_type'] == 'Production unavailability')].copy()
    active['cap'] = active[['unavailable_capacity_mw', 'normal_capacity_mw']].min(axis=1)
    group = np.where(active['fuel_type'] == 'Fossil Gas', 'Gas', np.where(active['fuel_type'] == 'Nuclear', 'Nuclear',
             np.where(active['fuel_type'].fillna('').str.startswith('Wind'), 'Wind', 'Other')))
    grid = pd.date_range(r['publish_time'].min().normalize(), r['publish_time'].max().normalize(), freq='D')
    active['group'] = group
    far = pd.Timestamp('2100-01-01')
    start_ns = active['event_start_time'].to_numpy()
    end_ns = active['event_end_time'].fillna(far).to_numpy()
    long_running = ((active['event_end_time'].fillna(far) - active['event_start_time']) > pd.Timedelta(days=365)).to_numpy()
    rows_out, naive_total, stale_total = [], [], []
    for day in grid:
        noon = np.datetime64(day + pd.Timedelta(hours=12))
        mask = (start_ns <= noon) & (end_ns > noon)
        live = active[mask]
        # Two notices on one unit at the same moment describe the same megawatts: count the larger
        per_asset = live.groupby(['group', 'asset_id'])['cap'].max().groupby(level=0).sum()
        rows_out.append(per_asset.reindex(['Gas', 'Nuclear', 'Wind', 'Other']).fillna(0).to_numpy())
        naive_total.append(live['cap'].sum())
        stale_total.append(active.loc[mask & long_running, 'cap'].sum())
    out_daily = pd.DataFrame(rows_out, index=grid, columns=['Gas', 'Nuclear', 'Wind', 'Other'])
    out_series = {g: out_daily[g].resample('W').mean() for g in out_daily.columns}
    naive_total, stale_total = pd.Series(naive_total, index=grid), pd.Series(stale_total, index=grid)
    SHARED['outage_daily'] = out_daily.sum(axis=1)
    cap = cur['unavailable_capacity_mw']
    status = r['event_status'].value_counts()
    event_type = r['event_type'].fillna('Not stated (other market information)').value_counts()
    by_fuel_stats = [[k, f'{len(g):,}', f"{100 * (g['unavailability_type'] == 'Planned').mean():.0f}%",
                      num(g['unavailable_capacity_mw'].median(), 0), num(g['unavailable_capacity_mw'].max(), 0),
                      num(((g['event_end_time'] - g['event_start_time']).dt.total_seconds() / 3600).median(), 1)]
                     for k, g in cur.assign(fuel_type=cur['fuel_type'].fillna('Not stated')).groupby('fuel_type')]
    by_fuel_stats.sort(key=lambda row: -int(row[1].replace(',', '')))
    charts = [
        bar('Notices published by month', 'Every revision counts as one notice.', [str(p) for p in monthly.index],
            [(c, monthly[c].to_numpy()) for c in monthly.columns], 'notices', stack=True, digits=0, wide=True),
        line('Generation capacity reported unavailable, weekly mean', 'At noon each day: the latest revision of each active '
             'production notice, capped at the unit normal capacity, one count per unit. Headline values, not the hourly profile.',
             days(out_series['Gas'].index), [(g, s / 1000) for g, s in out_series.items()], 'GW', area=True, stack=True,
             digits=2, wide=True),
        bar('Messages by fuel type', 'Latest revision of each message.', fuel.index, [('Messages', fuel.to_numpy())], '',
            horizontal=True, digits=0),
        bar('Revisions per message', 'How many versions a message went through; 15 means 15 or more.',
            revisions.index, [('Messages', revisions.to_numpy())], '', digits=0),
        hist('Outage duration', 'log10 of hours, latest revision: 0 is one hour, 1 ten hours, 2 about four days, 3 six weeks.',
             log_dur, 'log10 h', bins=50, clip=None, digits=2),
        box('Outage duration, planned against unplanned', 'Hours, latest revision.',
            {'Planned': duration[planned], 'Unplanned': duration[cur['unavailability_type'] == 'Unplanned']}, 'h'),
        hist('Unavailable capacity per message', 'Latest revision.', cap, 'MW', bins=50, clip=(0, 0.99), digits=0),
        hist('Notice lead time', 'Hours from publish_time to the event start. Negative: published after the outage began.',
             lead, 'h', bins=70, clip=(0.02, 0.98), digits=0),
        bar('Publish hour (UTC)', 'Notices follow the working day, with a floor of automated overnight messages.', range(24),
            [('Notices', hour.to_numpy())], '', digits=0),
        bar('Publish day of week', '', DOW, [('Notices', dow.to_numpy())], '', digits=0),
        bar('Event status', 'All revisions.', status.index, [('Notices', status.to_numpy())], '', horizontal=True, digits=0),
        bar('Event type', 'All revisions.', event_type.index, [('Notices', event_type.to_numpy())], '', horizontal=True, digits=0),
        bar('Most active participants', 'Messages, latest revision.', participants.index,
            [('Messages', participants.to_numpy())], '', horizontal=True, digits=0),
    ]
    over = int((r['unavailable_capacity_mw'] > r['normal_capacity_mw']).sum())
    checks = [
        check('Event ends before it starts', int((r['event_end_time'] < r['event_start_time']).sum()), len(r),
              'good' if (r['event_end_time'] < r['event_start_time']).sum() == 0 else 'warn',
              'the soft rule event_ends_before_start', bad=True),
        check('Unavailable capacity above normal capacity', over, len(r), 'warn' if over else 'good',
              'the soft rule unavailable_exceeds_normal; gold caps it', bad=True),
        check('Negative unavailable capacity', int((r['unavailable_capacity_mw'] < 0).sum()), len(r),
              'good' if (r['unavailable_capacity_mw'] < 0).sum() == 0 else 'bad',
              'the hard rule negative_unavailable_capacity', bad=True),
        check('No fuel type', int(r['fuel_type'].isna().sum()), len(r), 'warn', 'fall into gold\'s "other" group', bad=True),
        check('No unavailable capacity', int(r['unavailable_capacity_mw'].isna().sum()), len(r), 'warn',
              'cannot count toward capacity out', bad=True),
        check('No event end time', int(r['event_end_time'].isna().sum()), len(r), 'info', 'open-ended; gold treats as ongoing'),
        check('Unknown event status', int((~r['event_status'].isin(['Active', 'Inactive', 'Dismissed'])).sum()), len(r), 'good',
              '', bad=True),
        check('Revision numbers with gaps', int((per_message['count'] != per_message['max']).sum()), len(per_message), 'info',
              'earlier revisions fell outside the fetch window, or numbering starts above 1'),
        check('Notices published after the event started', int((lead < 0).sum()), int(lead.notna().sum()), 'info',
              'not knowable before the outage'),
        check('Active production notices running longer than a year', int(long_running.sum()), len(active), 'warn',
              f'they hold {stale_total.mean() / 1000:.1f} GW of the {out_daily.sum(axis=1).mean() / 1000:.1f} GW out on an average day: '
              'closed or mothballed units', bad=True),
        check('Double counting from overlapping notices on one unit', int(round((naive_total - out_daily.sum(axis=1)).mean())),
              int(round(naive_total.mean())), 'info', 'mean MW a day counted twice when notices are simply summed', bad=True),
    ]
    tiles = [tile('Coverage', f'{r.publish_time.min():%b %Y} – {r.publish_time.max():%b %Y}', 'by publish time'),
             tile('Messages', fmt_int(len(per_message)), f"{per_message['count'].mean():.1f} revisions each on average"),
             tile('Planned', f"{100 * planned.mean():.0f}%", 'of messages, latest revision'),
             tile('Median outage', f'{duration.median():.1f} h', f'median size {cap.median():.0f} MW')]
    tables = [table('By fuel type', ['Fuel type', 'Messages', 'Planned', 'Median MW out', 'Max MW out', 'Median hours'],
                    by_fuel_stats, 'Latest revision of each message.'),
              table('Assets with the most messages', ['Asset', 'Fuel type', 'Messages', 'Normal MW', 'Mean MW out'], asset_rows)]
    section('remit_outages', 'REMIT outage notices', 'Elexon REMIT', 'message (mrid) and revision', ('mrid', 'revision_number'),
            r, tiles, checks, charts, tables)
    say('  duration q', duration.quantile([.1, .25, .5, .75, .9, .99]).round(1).to_dict(), 'lead q',
        lead.quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
    say('  revisions', per_message['count'].describe().round(2).to_dict(), 'status', status.to_dict())
    say('  capacity out mean GW', {g: round(float(s.mean()) / 1000, 2) for g, s in out_series.items()}, 'stale GW',
        round(stale_total.mean() / 1000, 2), 'double MW', round((naive_total - out_daily.sum(axis=1)).mean()))
    stale = active[long_running].sort_values('cap', ascending=False)
    say('  stale', stale[['asset_id', 'fuel_type', 'cap', 'event_start_time', 'event_end_time']].head(14).to_string())
    say('  monthly notices min/max', monthly.sum(axis=1).min(), monthly.sum(axis=1).idxmax(), monthly.sum(axis=1).max(),
        'profile points', r['outage_profile_points'].describe().round(1).to_dict())


# --- 10. bank_holidays --------------------------------------------------------------------------------------

def bank_holidays():
    b = load('bank_holidays')
    names = {'england-and-wales': 'England and Wales', 'scotland': 'Scotland', 'northern-ireland': 'Northern Ireland'}
    b['nation'] = b['division'].map(names)
    by_year = b.groupby([b['holiday_date'].dt.year, 'nation']).size().unstack().fillna(0)
    weekday = b.groupby([b['holiday_date'].dt.dayofweek, 'nation']).size().unstack().reindex(range(7)).fillna(0)
    by_month = b.groupby([b['holiday_date'].dt.month, 'nation']).size().unstack().reindex(range(1, 13)).fillna(0)
    title = b['title']
    titles = title.value_counts()
    demand = SHARED['demand_daily']
    ew = set(b.loc[b['division'] == 'england-and-wales', 'holiday_date'])
    frame = pd.DataFrame({'demand': demand})
    frame['kind'] = np.where(frame.index.isin(ew), 'Bank holiday', np.where(frame.index.dayofweek >= 5, 'Weekend', 'Working day'))
    # Compare each holiday with working days of the same month, so the season doesn't decide the answer
    working = frame[frame['kind'] == 'Working day']
    month_mean = working.groupby([working.index.year, working.index.month])['demand'].mean()
    holiday = frame[frame['kind'] == 'Bank holiday']
    base = np.array([month_mean.get((d.year, d.month), np.nan) for d in holiday.index])
    drop = 100 * (holiday['demand'].to_numpy() / base - 1)
    weekend = frame[frame['kind'] == 'Weekend']
    wk_base = np.array([month_mean.get((d.year, d.month), np.nan) for d in weekend.index])
    wk_drop = 100 * (weekend['demand'].to_numpy() / wk_base - 1)
    in_window = b[b['holiday_date'].between(demand.index.min(), demand.index.max())]
    effect_rows = [[f'{d:%Y-%m-%d}', title[b.index[(b['division'] == 'england-and-wales') & (b['holiday_date'] == d)][0]],
                    f'{v:,.0f}', f'{x:+.1f}%'] for d, v, x in zip(holiday.index, holiday['demand'], drop)]
    order = list(names.values())
    charts = [
        bar('Bank holidays per year by nation', 'Scotland and Northern Ireland have more than England and Wales.',
            by_year.index, [(n, by_year[n].to_numpy()) for n in order], 'days', digits=0, wide=True),
        bar('Demand on bank holidays against working days of the same month', 'England and Wales holidays; national demand, '
            'daily mean. Weekends shown for comparison.', ['Bank holidays', 'Weekends'],
            [('Change from working days', [np.nanmean(drop), np.nanmean(wk_drop)])], '%'),
        box('Demand change by day type', 'Each day against working days of its own month.',
            {'Bank holidays': drop, 'Weekends': wk_drop}, '%'),
        bar('Day of week', 'Mondays dominate; fixed-date holidays land anywhere.', DOW,
            [(n, weekday[n].to_numpy()) for n in order], 'days', digits=0),
        bar('Month of year', '', MONTHS, [(n, by_month[n].to_numpy()) for n in order], 'days', digits=0),
        bar('Holiday names', 'All nations, 2019–2028.', titles.index, [('Days', titles.to_numpy())], '', horizontal=True, digits=0),
    ]
    checks = [
        check('Unknown divisions', int((~b['division'].isin(names)).sum()), len(b), 'good', '', bad=True),
        check('Duplicate division + date', int(b.duplicated(['division', 'holiday_date']).sum()), len(b), 'good', '', bad=True),
        check('Holidays on a weekend', int((b['holiday_date'].dt.dayofweek >= 5).sum()), len(b), 'info',
              'expected to be moved to a substitute day'),
        check('Substitute days', int(b['notes'].fillna('').str.contains('Substitute').sum()), len(b), 'info', 'noted in notes'),
        check('Holidays inside the demand data window', len(in_window), len(b), 'info',
              f'{demand.index.min():%Y-%m-%d} to {demand.index.max():%Y-%m-%d}'),
    ]
    tiles = [tile('Coverage', f'{b.holiday_date.min():%Y} – {b.holiday_date.max():%Y}', 'includes future years'),
             tile('Nations', '3', ', '.join(f'{n}: {int((b.nation == n).sum())}' for n in order)),
             tile('Demand on a bank holiday', f'{np.nanmean(drop):+.1f}%', 'against working days of the same month'),
             tile('With bunting', f"{100 * b['bunting'].astype(bool).mean():.0f}%", 'the GOV.UK celebration flag')]
    tables = [table('England and Wales bank holidays inside the data window', ['Date', 'Holiday', 'Mean demand MW', 'vs working days'],
                    effect_rows, 'Compared with the mean of working days in the same month and year.')]
    section('bank_holidays', 'UK bank holidays', 'GOV.UK', 'division (nation) and date', ('division', 'holiday_date'),
            b.drop(columns=['nation']), tiles, checks, charts, tables)
    say('  drop mean', np.nanmean(drop).round(1), 'weekend', np.nanmean(wk_drop).round(1), 'per nation',
        b.nation.value_counts().to_dict(), 'titles', titles.head(6).to_dict())
    SHARED['bank_ew'] = ew


# --- 11. carbon_intensity -----------------------------------------------------------------------------------

def carbon_intensity():
    c = load('carbon_intensity')
    s = c.set_index('period_start')
    local = uk_local(c['period_start'])
    daily = s['actual_gco2_kwh'].resample('D').mean()
    SHARED['carbon_daily'] = daily
    SHARED['carbon_half'] = s['actual_gco2_kwh']
    err = (c['forecast_gco2_kwh'] - c['actual_gco2_kwh']).dropna()
    e = error_stats(c['forecast_gco2_kwh'], c['actual_gco2_kwh'])
    order = ['very low', 'low', 'moderate', 'high', 'very high']
    index = c['intensity_index'].value_counts().reindex(order).fillna(0)
    index_year = (c.groupby([c['period_start'].dt.year, 'intensity_index']).size().unstack().reindex(columns=order).fillna(0))
    index_year = 100 * index_year.div(index_year.sum(axis=1), axis=0)
    grid = pd.date_range(c['period_start'].min(), c['period_start'].max(), freq='30min')
    monthly = s['actual_gco2_kwh'].resample('MS').mean()
    year_rows = [[int(y), num(g.mean(), 1), num(g.median(), 0), num(g.std(), 1), num(g.min(), 0), num(g.max(), 0)]
                 for y, g in s['actual_gco2_kwh'].groupby(s.index.year)]
    monthly_err = (c.assign(e=(c['forecast_gco2_kwh'] - c['actual_gco2_kwh']).abs())
                   .groupby(c['period_start'].dt.to_period('M'))['e'].mean())
    gen = SHARED['gen_half']
    share = (100 * gen['renewable'] / gen['domestic']).reindex(s.index)
    gas_share = (100 * gen['gas'] / gen['domestic']).reindex(s.index)
    charts = [
        line('Carbon intensity, daily mean', 'Grams of CO2 per kWh of electricity consumed.', days(daily.index),
             [('Daily mean', daily), ('28-day mean', rolling(daily))], 'gCO2/kWh', wide=True),
        hist('Distribution of half-hourly carbon intensity', '', c['actual_gco2_kwh'], 'gCO2/kWh', bins=60, clip=None, digits=0),
        scatter('Carbon intensity against the gas share of generation', 'Gas sets the carbon intensity almost alone.',
                gas_share, s['actual_gco2_kwh'], 'Gas share of GB generation (%)', 'Carbon intensity (gCO2/kWh)'),
        scatter('Carbon intensity against the renewable share', '', share, s['actual_gco2_kwh'],
                'Renewable share of GB generation (%)', 'Carbon intensity (gCO2/kWh)'),
        month_hour_heat(local, c['actual_gco2_kwh'], 'Carbon intensity by month and time of day',
                        'Cleanest on summer middays (solar); dirtiest on winter evenings (gas).', 'gCO2/kWh'),
        scatter('Forecast against actual', 'The forecast is the last one issued, so it is close to a nowcast.',
                c['forecast_gco2_kwh'], c['actual_gco2_kwh'], 'Forecast (gCO2/kWh)', 'Actual (gCO2/kWh)', digits=0),
        hist('Forecast error', 'Forecast − actual.', err, 'gCO2/kWh', bins=60, digits=0),
        line('Forecast error by month', 'Mean absolute error.', [str(p) for p in monthly_err.index],
             [('Mean absolute error', monthly_err)], 'gCO2/kWh', wide=True),
        bar('Intensity index', 'The API\'s five bands.', order, [('Half-hours', index.to_numpy())], '', digits=0),
        bar('Intensity index by year', 'Share of half-hours in each band.', index_year.index,
            [(o, index_year[o].to_numpy()) for o in order], '%', stack=True),
        bar('Monthly mean carbon intensity', '', [d.strftime('%Y-%m') for d in monthly.index], [('Monthly mean', monthly.to_numpy())],
            'gCO2/kWh', wide=True),
        acf_chart(daily, 30, 'Autocorrelation of daily carbon intensity', ''),
    ]
    checks = [
        check('Half-hours present on the grid', len(c), len(grid), 'good' if len(c) == len(grid) else 'warn',
              f'{len(grid) - len(c):,} half-hours missing between first and last'),
        check('Actual not published', int(c['actual_gco2_kwh'].isna().sum()), len(c), 'info',
              'the soft rule actual_not_yet_published', bad=True),
        check('Forecast missing', int(c['forecast_gco2_kwh'].isna().sum()), len(c), 'info', 'the soft rule forecast_missing', bad=True),
        check('Neither forecast nor actual', int((c['forecast_gco2_kwh'].isna() & c['actual_gco2_kwh'].isna()).sum()), len(c),
              'good' if (c['forecast_gco2_kwh'].isna() & c['actual_gco2_kwh'].isna()).sum() == 0 else 'bad',
              'the hard rule no_intensity_value', bad=True),
        check('Period not 30 minutes long', int(((c['period_end'] - c['period_start']) != pd.Timedelta(minutes=30)).sum()), len(c),
              'good', 'the hard rule period_not_30_min', bad=True),
        check('Intensity outside [0, 1000]', int((~c['actual_gco2_kwh'].dropna().between(0, 1000)).sum()), len(c), 'good',
              'the hard rule intensity_implausible', bad=True),
        check('Unknown index value', int((~c['intensity_index'].isin(order)).sum()), len(c), 'good', '', bad=True),
    ]
    tiles = [tile('Coverage', f'{c.period_start.min():%b %Y} – {c.period_start.max():%b %Y}', 'half-hourly, UTC'),
             tile('Mean intensity', f"{c['actual_gco2_kwh'].mean():.0f} g", f"median {c['actual_gco2_kwh'].median():.0f} gCO2/kWh"),
             tile('Range', f"{c['actual_gco2_kwh'].min():.0f} to {c['actual_gco2_kwh'].max():.0f}", 'gCO2/kWh'),
             tile('Forecast error', f"{e['mae']:.1f} g", f"bias {e['bias']:+.1f}, r {e['r']}")]
    tables = [table('Actual intensity by year', ['Year', 'Mean', 'Median', 'Std dev', 'Min', 'Max'], year_rows, 'gCO2/kWh.')]
    section('carbon_intensity', 'Carbon intensity', 'Carbon Intensity API (NESO)', 'half-hour (period_start)', ('period_start',),
            c, tiles, checks, charts, tables)
    say('  forecast', e, 'index', index.to_dict(), 'corr gas share', round(gas_share.corr(s['actual_gco2_kwh']), 3),
        'corr ren share', round(share.corr(s['actual_gco2_kwh']), 3))
    missing = grid.difference(pd.DatetimeIndex(c['period_start']))
    say('  missing half-hours', len(missing), list(missing[:6]), 'null actual dates',
        c.loc[c.actual_gco2_kwh.isna(), 'period_start'].dt.date.value_counts().head(5).to_dict())


# --- 12. pv_live --------------------------------------------------------------------------------------------

def pv_live():
    p = load('pv_live')
    # PV_Live stamps the end of each half-hour
    p['start'] = p['datetime_gmt'] - pd.Timedelta(minutes=30)
    s = p.set_index('start')['generation_mw']
    local = uk_local(p['start'])
    SHARED['pv_half'] = s
    daily_energy = s.resample('D').sum() * 0.5 / 1000
    daily_peak = s.resample('D').max()
    SHARED['pv_daily'] = s.resample('D').mean()
    monthly = s.resample('MS').mean()
    grid = pd.date_range(p['datetime_gmt'].min(), p['datetime_gmt'].max(), freq='30min')
    year_rows = [[int(y), num(g.mean(), 0), num(g.max(), 0), f'{g.idxmax():%Y-%m-%d %H:%M}', num(g.sum() * 0.5 / 1e6, 2)]
                 for y, g in s.groupby(s.index.year)]
    agws = SHARED['agws']['solar'].reindex(s.index)
    by_month = {MONTHS[m - 1]: daily_energy[daily_energy.index.month == m] for m in range(1, 13)}
    daylight = s[s > 0]
    hours_gen = (s > 0).resample('D').sum() / 2
    # NESO embedded solar is keyed by settlement period, so it settles which end the PV_Live stamp marks
    neso = load('neso_demand')
    neso_solar = pd.Series(neso['embedded_solar_generation_mw'].to_numpy(),
                           index=period_start(neso['settlement_date'], neso['settlement_period']))
    stamp_rows, stamp_r = [], {}
    for label, shift in (('End of the half-hour (what gold assumes)', -30), ('Start of the half-hour', 0)):
        shifted = pd.Series(p['generation_mw'].to_numpy(), index=p['datetime_gmt'] + pd.Timedelta(minutes=shift))
        both = pd.DataFrame({'neso': neso_solar, 'pv': shifted}).dropna()
        stamp_r[shift] = both['neso'].corr(both['pv'])
        stamp_rows.append([label, f"{stamp_r[shift]:.5f}", f"{(both['neso'] - both['pv']).abs().mean():.1f}",
                           f"{100 * ((both['neso'] - both['pv']).abs() < 1).mean():.1f}%"])
    charts = [
        line('Solar energy per day', 'National estimate, all PV including rooftop.', days(daily_energy.index),
             [('Daily energy', daily_energy), ('28-day mean', rolling(daily_energy))], 'GWh', wide=True),
        month_hour_heat(local, p['generation_mw'], 'Solar output by month and time of day',
                        'Mean MW by UK local hour.', 'MW'),
        line('Daily peak output', 'The record keeps rising as capacity is added.', days(daily_peak.index),
             [('Daily peak', daily_peak)], 'MW', digits=0, wide=True),
        box('Daily energy by month of year', 'A June day yields roughly ten times a December day.', by_month, 'GWh'),
        bar('Monthly mean output', '', [d.strftime('%Y-%m') for d in monthly.index], [('Monthly mean', monthly.to_numpy())], 'MW',
            digits=0, wide=True),
        hist('Distribution of half-hourly output in daylight', 'Half-hours with output above zero.', daylight, 'MW', bins=50,
             clip=None, digits=0),
        scatter('PV_Live against Elexon AGWS solar', 'AGWS sees only solar metered on the transmission system.',
                agws, s, 'AGWS solar (MW)', 'PV_Live solar (MW)', digits=0),
        line('Hours a day with any solar output', 'Day length, as the panels see it.', days(hours_gen.index),
             [('Hours with output', hours_gen)], 'h', wide=True),
        acf_chart(daily_energy, 30, 'Autocorrelation of daily solar energy', 'Slow decay: season dominates day-to-day weather.'),
    ]
    checks = [
        check('Half-hours present on the grid', len(p), len(grid), 'good' if len(p) == len(grid) else 'warn',
              f'{len(grid) - len(p):,} half-hours missing between first and last'),
        range_check(p, 'generation_mw', 0, 40000, 'generation_implausible'),
        check('Half-hours with zero output', int((p['generation_mw'] == 0).sum()), len(p), 'info', 'night'),
        check('GSP other than 0 (national)', int((p['gsp_id'] != 0).sum()), len(p), 'good', '', bad=True),
        check('Timestamps off the half-hour', int((~p['datetime_gmt'].dt.minute.isin([0, 30])).sum()), len(p), 'good', '', bad=True),
        check('Output above zero between 23:00 and 03:00 local', int(((p['generation_mw'] > 0) & ((local.dt.hour >= 23) | (local.dt.hour < 3))).sum()),
              int(((local.dt.hour >= 23) | (local.dt.hour < 3)).sum()), 'good', '', bad=True),
        check('Timestamp marks the end of the half-hour', 1, 1, 'good',
              f'r = {stamp_r[-30]:.5f} against NESO embedded solar read that way, {stamp_r[0]:.5f} read as the start'),
    ]
    tiles = [tile('Coverage', f'{p.datetime_gmt.min():%b %Y} – {p.datetime_gmt.max():%b %Y}', 'half-hourly, GMT'),
             tile('Mean output', f'{s.mean() / 1000:.2f} GW', f'{100 * (s > 0).mean():.0f}% of half-hours above zero'),
             tile('Record peak', f'{s.max() / 1000:.1f} GW', f'{s.idxmax():%d %b %Y}'),
             tile('Best day', f'{daily_energy.max():.0f} GWh', f'{daily_energy.idxmax():%d %b %Y}')]
    tables = [table('Solar by year', ['Year', 'Mean MW', 'Peak MW', 'Peak at (GMT, period start)', 'Energy TWh'], year_rows),
              table('Which end of the half-hour does datetime_gmt mark?',
                    ['Reading', 'r with NESO embedded solar', 'Mean absolute difference MW', 'Within 1 MW'], stamp_rows,
                    'NESO embedded solar is keyed by settlement period and derives from PV_Live, so the better match is the right reading.')]
    section('pv_live', 'National solar estimate', 'Sheffield Solar PV_Live', 'GSP and datetime_gmt (period end)',
            ('gsp_id', 'datetime_gmt'), p.drop(columns=['start']), tiles, checks, charts, tables)
    ratio = (agws / s)[s > 2000]
    say('  agws/pvlive ratio median', ratio.median().round(3), 'corr', round(agws.corr(s), 3), 'month mean energy',
        daily_energy.groupby(daily_energy.index.month).mean().round(1).to_dict())
    peak_hour = local[p['generation_mw'].to_numpy() > 0.9 * daily_peak.reindex(p['start'].dt.normalize()).to_numpy()].dt.hour
    say('  near-peak local hour distribution (period START)', peak_hour.value_counts().sort_index().to_dict())
    missing = grid.difference(pd.DatetimeIndex(p['datetime_gmt']))
    say('  missing', len(missing), list(missing[:5]))


# --- 13. neso_demand ----------------------------------------------------------------------------------------

def neso_demand():
    n = load('neso_demand')
    n['start_time'] = period_start(n['settlement_date'], n['settlement_period'])
    s = n.set_index('start_time')
    daily = n.groupby('settlement_date')[['nd_mw', 'tsd_mw', 'embedded_wind_generation_mw', 'embedded_solar_generation_mw']].mean()
    SHARED['neso_daily'] = daily
    SHARED['neso_half'] = s
    flows = [c for c in n.columns if c.endswith('_flow_mw')]
    flow_mean = n[flows].mean().sort_values(ascending=False)
    flow_mean.index = [c.replace('_flow_mw', '').replace('_', ' ').upper() for c in flow_mean.index]
    cap = n.groupby(n['settlement_date'].dt.to_period('M'))[['embedded_wind_capacity_mw', 'embedded_solar_capacity_mw']].max()
    monthly = n.groupby(n['settlement_date'].dt.to_period('M'))[['embedded_wind_generation_mw', 'embedded_solar_generation_mw']].mean()
    indo = SHARED['demand_half'].reindex(s.index)
    diff = (s['nd_mw'] - indo).dropna()
    pv = SHARED['pv_half'].reindex(s.index)
    in_range = n[n['settlement_period'] <= 48]
    pump = in_range.groupby('settlement_period')['pump_storage_pumping_mw'].mean()
    net_flow = n[flows].sum(axis=1)
    corr = n[['nd_mw', 'tsd_mw', 'embedded_wind_generation_mw', 'embedded_solar_generation_mw', 'pump_storage_pumping_mw',
              'scottish_transfer_mw']].assign(net_interconnector_flow_mw=net_flow)
    corr.columns = ['National demand', 'Transmission demand', 'Embedded wind', 'Embedded solar', 'Pumping', 'Scottish transfer',
                    'Net interconnector flow']
    solar_cf = 100 * n['embedded_solar_generation_mw'] / n['embedded_solar_capacity_mw']
    wind_cf = 100 * n['embedded_wind_generation_mw'] / n['embedded_wind_capacity_mw']
    year_rows = [[int(y), num(g['nd_mw'].mean(), 0), num(g['nd_mw'].max(), 0), num(g['embedded_wind_generation_mw'].mean(), 0),
                  num(g['embedded_solar_generation_mw'].mean(), 0), num(g['embedded_wind_capacity_mw'].max(), 0),
                  num(g['embedded_solar_capacity_mw'].max(), 0)] for y, g in n.groupby(n['settlement_date'].dt.year)]
    flow_rows = [[c.replace('_flow_mw', '').replace('_', ' ').upper(), num(n[c].mean(), 0), num(n[c].min(), 0), num(n[c].max(), 0),
                  f'{100 * (n[c] > 0).mean():.0f}%', f'{100 * (n[c] < 0).mean():.0f}%', f'{100 * (n[c] == 0).mean():.0f}%']
                 for c in flows]
    embedded_share = 100 * (daily['embedded_wind_generation_mw'] + daily['embedded_solar_generation_mw']) / \
        (daily['nd_mw'] + daily['embedded_wind_generation_mw'] + daily['embedded_solar_generation_mw'])
    charts = [
        line('National and transmission demand, daily mean', 'ND excludes, TSD includes, pumping and exports.',
             days(daily.index), [('National demand (ND)', daily['nd_mw']), ('Transmission demand (TSD)', daily['tsd_mw'])],
             'MW', digits=0, wide=True),
        line('Embedded capacity', 'Solar and wind connected to distribution networks, invisible to FUELHH.',
             [str(p) for p in cap.index], [('Embedded solar capacity', cap['embedded_solar_capacity_mw']),
                                           ('Embedded wind capacity', cap['embedded_wind_capacity_mw'])], 'MW', digits=0, wide=True),
        line('Embedded generation, monthly mean', 'NESO\'s estimate.', [str(p) for p in monthly.index],
             [('Embedded solar', monthly['embedded_solar_generation_mw']), ('Embedded wind', monthly['embedded_wind_generation_mw'])],
             'MW', digits=0, wide=True),
        line('Embedded wind and solar as a share of underlying demand', 'Daily; underlying demand = ND + embedded generation.',
             days(embedded_share.index), [('Daily share', embedded_share), ('28-day mean', rolling(embedded_share))], '%', wide=True),
        bar('Mean interconnector flow', 'Positive: import to GB.', flow_mean.index, [('Mean MW', flow_mean.to_numpy())], 'MW',
            horizontal=True, digits=0),
        hist('NESO national demand minus Elexon INDO', 'The same quantity from two publishers.', diff, 'MW', bins=60, digits=0),
        scatter('NESO embedded solar against PV_Live', 'PV_Live shifted to period start.', pv, s['embedded_solar_generation_mw'],
                'PV_Live (MW)', 'NESO embedded solar (MW)', digits=0),
        box('Embedded load factor', 'Generation as a share of capacity, half-hourly.', {'Embedded wind': wind_cf, 'Embedded solar': solar_cf}, '%'),
        line('Pumped-storage pumping by time of day', 'Pumping fills the reservoirs overnight and at the solar midday.',
             PERIOD_LABELS, [('Pumping', pump.reindex(range(1, 49)))], 'MW', area=True, digits=0),
        hist('Scottish transfer', 'Flow from Scotland to England: wind exported south.', n['scottish_transfer_mw'], 'MW', bins=50, digits=0),
        corr_heat(corr, 'Correlation between NESO measures, half-hourly', ''),
        month_period_heat(n, 'embedded_solar_generation_mw', 'Embedded solar by month and time of day', '', 'MW'),
    ]
    checks = completeness(n) + [
        range_check(n, 'nd_mw', 0, 80000, 'demand_implausible'),
        check('Rows without a forecast/actual indicator', int(n['forecast_actual_indicator'].isna().sum()), len(n), 'info',
              'older yearly files have no such column'),
        check('Forecast rows (indicator F)', int((n['forecast_actual_indicator'] == 'F').sum()), len(n), 'good',
              'the soft rule forecast_not_actual', bad=True),
        check('Days NESO ends before Elexon INDO', int((SHARED['demand_daily'].index.max() - n['settlement_date'].max()).days),
              0, 'warn', f"NESO ends {n['settlement_date'].max():%Y-%m-%d}; INDO ends {SHARED['demand_daily'].index.max():%Y-%m-%d}", bad=True),
        check('Embedded solar above zero at night (23:00–03:00)',
              int(((n['embedded_solar_generation_mw'] > 0) & ((n['settlement_period'] >= 47) | (n['settlement_period'] <= 6))).sum()),
              len(n), 'good', '', bad=True),
        check('Interconnector columns that are always zero', int((n[flows].abs().sum() == 0).sum()), len(flows), 'info',
              ', '.join(c for c in flows if n[c].abs().sum() == 0) or 'none', bad=True),
        check('Embedded generation above capacity', int((n['embedded_solar_generation_mw'] > n['embedded_solar_capacity_mw']).sum()
                                                         + (n['embedded_wind_generation_mw'] > n['embedded_wind_capacity_mw']).sum()),
              len(n), 'good', '', bad=True),
    ]
    tiles = [tile('Coverage', f'{n.settlement_date.min():%b %Y} – {n.settlement_date.max():%b %Y}', f'{len(daily):,} days'),
             tile('Mean national demand', f"{n['nd_mw'].mean() / 1000:.1f} GW", f"peak {n['nd_mw'].max() / 1000:.1f} GW"),
             tile('Embedded solar capacity', f"{n['embedded_solar_capacity_mw'].max() / 1000:.1f} GW",
                  f"up from {n['embedded_solar_capacity_mw'].min() / 1000:.1f} GW"),
             tile('ND vs INDO', f'{diff.abs().mean():.0f} MW', 'mean absolute difference')]
    tables = [table('By year', ['Year', 'Mean ND MW', 'Peak ND MW', 'Embedded wind MW', 'Embedded solar MW', 'Wind capacity MW',
                                'Solar capacity MW'], year_rows),
              table('Interconnector flows', ['Link', 'Mean MW', 'Min', 'Max', 'Importing', 'Exporting', 'Idle'], flow_rows,
                    'Share of half-hours.')]
    section('neso_demand', 'NESO historic demand', 'NESO data portal', 'settlement date and period',
            ('settlement_date', 'settlement_period'), n.drop(columns=['start_time']), tiles, checks, charts, tables)
    say('  nd-indo', diff.describe().round(1).to_dict(), 'corr', round(s['nd_mw'].corr(indo), 4))
    say('  solar vs pvlive corr', round(pv.corr(s['embedded_solar_generation_mw']), 4), 'ratio',
        round((s['embedded_solar_generation_mw'] / pv)[pv > 2000].median(), 3))
    say('  corr', corr.corr().round(2).to_dict()['National demand'], 'embedded share mean', round(embedded_share.mean(), 1),
        'max', round(embedded_share.max(), 1))
    say('  indicator by year', n.groupby(n.settlement_date.dt.year)['forecast_actual_indicator'].apply(lambda x: x.fillna('null').value_counts().to_dict()).to_dict())


# --- 14 and 15. weather -------------------------------------------------------------------------------------

LOCATIONS = ('dogger_bank', 'east_anglia_offshore', 'irish_sea', 'scottish_highlands', 'southern_england')
NICE = {'dogger_bank': 'Dogger Bank', 'east_anglia_offshore': 'East Anglia offshore', 'irish_sea': 'Irish Sea',
        'scottish_highlands': 'Scottish Highlands', 'southern_england': 'Southern England'}
VARS = {'wind_speed_100m_kmh': ('Wind at 100 m', 'km/h'), 'wind_speed_10m_kmh': ('Wind at 10 m', 'km/h'),
        'shortwave_radiation_wm2': ('Shortwave radiation', 'W/m²'), 'temperature_2m_c': ('Temperature at 2 m', '°C'),
        'cloud_cover_pct': ('Cloud cover', '%')}


def weather(table_name, title, source, other=None):
    w = load(table_name)
    w['place'] = w['location'].map(NICE)
    local = uk_local(w['time'])
    places = [NICE[l] for l in LOCATIONS]
    wide = {v: w.pivot(index='time', columns='place', values=v)[places] for v in VARS}
    SHARED[table_name] = wide
    monthly = {v: wide[v].resample('MS').mean() for v in VARS}
    months = [d.strftime('%Y-%m') for d in monthly['wind_speed_100m_kmh'].index]
    stats_rows = []
    for place in places:
        g = w[w['place'] == place]
        stats_rows.append([place, num(g['wind_speed_100m_kmh'].mean(), 1), num(g['wind_speed_100m_kmh'].quantile(.95), 1),
                           num(g['wind_speed_100m_kmh'].max(), 1), num(g['temperature_2m_c'].mean(), 1),
                           num(g['temperature_2m_c'].min(), 1), num(g['temperature_2m_c'].max(), 1),
                           num(g['shortwave_radiation_wm2'].mean(), 0), num(g['cloud_cover_pct'].mean(), 0)])
    shear = {p: (wide['wind_speed_100m_kmh'][p] / wide['wind_speed_10m_kmh'][p])[wide['wind_speed_10m_kmh'][p] > 5] for p in places}
    south = w[w['location'] == 'southern_england']
    grid = pd.date_range(w['time'].min(), w['time'].max(), freq='1h')
    null_rows = int(w[list(VARS)].isna().any(axis=1).sum())
    daily_wind = wide['wind_speed_100m_kmh'].resample('D').mean()
    charts = [
        line('Wind at 100 m by location, monthly mean', 'Offshore points are windiest; all share the winter peak.', months,
             [(p, monthly['wind_speed_100m_kmh'][p]) for p in places], 'km/h', wide=True),
        box('Hourly wind at 100 m by location', '', {p: wide['wind_speed_100m_kmh'][p] for p in places}, 'km/h'),
        hist('Distribution of hourly wind at 100 m, all locations', 'Weibull-shaped, as wind speeds are.',
             w['wind_speed_100m_kmh'], 'km/h', bins=50, clip=None),
        corr_heat(wide['wind_speed_100m_kmh'], 'Correlation of hourly wind between locations',
                  'Nearby sea areas move together; distant ones much less. Five points give real diversity.'),
        line('Temperature by location, monthly mean', '', months, [(p, monthly['temperature_2m_c'][p]) for p in places], '°C', wide=True),
        box('Hourly temperature by location', 'Sea points have a narrow range; land points swing wider.',
            {p: wide['temperature_2m_c'][p] for p in places}, '°C'),
        month_hour_heat(uk_local(south['time']), south['shortwave_radiation_wm2'], 'Shortwave radiation in Southern England, by month and hour',
                        'The solar resource; UK local time.', 'W/m²'),
        line('Shortwave radiation by location, monthly mean', '', months, [(p, monthly['shortwave_radiation_wm2'][p]) for p in places],
             'W/m²', digits=0, wide=True),
        line('Cloud cover by location, monthly mean', '', months, [(p, monthly['cloud_cover_pct'][p]) for p in places], '%', digits=0, wide=True),
        box('Wind shear: 100 m speed over 10 m speed', 'Hours with at least 5 km/h at 10 m. Land has more shear than sea.', shear, '×', digits=2),
        hist('Distribution of cloud cover', 'U-shaped: skies are mostly clear or mostly overcast.', w['cloud_cover_pct'], '%', bins=20, clip=None, digits=0),
        month_hour_heat(local, w['wind_speed_100m_kmh'], 'Wind at 100 m by month and hour, all locations', 'UK local time.', 'km/h', digits=1),
    ]
    tables = [table('By location', ['Location', 'Mean wind 100 m', 'p95 wind', 'Max wind', 'Mean °C', 'Min °C', 'Max °C',
                                    'Mean W/m²', 'Mean cloud %'], stats_rows, 'Wind in km/h.')]
    per_location = w.groupby('place')['time'].nunique()
    checks = [
        check('Hours present for every location', int((per_location == len(grid)).sum()), len(places),
              'good' if (per_location == len(grid)).all() else 'warn', f'{len(grid):,} hours on the grid'),
        check('Rows with a missing weather value', null_rows, len(w), 'warn' if null_rows else 'good',
              'the soft rule missing_weather_value', bad=True),
        *[check(f'{c} outside [{lo}, {hi}]', int((~w[c].dropna().between(lo, hi)).sum()), int(w[c].notna().sum()), 'good',
                'hard rule', bad=True)
          for c, lo, hi in (('wind_speed_100m_kmh', 0, 400), ('wind_speed_10m_kmh', 0, 400), ('shortwave_radiation_wm2', 0, 1500),
                            ('temperature_2m_c', -60, 60), ('cloud_cover_pct', 0, 100))],
        check('Radiation above zero at night (23:00–03:00 local)',
              int(((w['shortwave_radiation_wm2'] > 0) & ((local.dt.hour >= 23) | (local.dt.hour < 3))).sum()),
              int(((local.dt.hour >= 23) | (local.dt.hour < 3)).sum()), 'info', 'far-north summer nights are light', bad=True),
        check('Wind at 100 m below wind at 10 m', int((w['wind_speed_100m_kmh'] < w['wind_speed_10m_kmh']).sum()), len(w), 'info',
              'unusual but physical in unstable air', bad=True),
    ]
    tiles = [tile('Coverage', f'{w.time.min():%b %Y} – {w.time.max():%b %Y}', 'hourly, UTC'),
             tile('Locations', '5', 'three offshore, two on land'),
             tile('Mean wind at 100 m', f"{w['wind_speed_100m_kmh'].mean():.1f} km/h", f"max {w['wind_speed_100m_kmh'].max():.0f} km/h"),
             tile('Temperature range', f"{w['temperature_2m_c'].min():.0f} to {w['temperature_2m_c'].max():.0f} °C",
                  f"mean {w['temperature_2m_c'].mean():.1f} °C")]
    if other is not None:
        # Day-ahead forecast against the stitched short-range forecast (close to what happened)
        rows, mae_by_place = [], {}
        for v, (label, _unit) in VARS.items():
            for place in places:
                e = error_stats(wide[v][place], other[v][place].reindex(wide[v].index))
                rows.append([label, place, f"{e['n']:,}", e['bias'], e['mae'], e['rmse'], e['r']])
                mae_by_place[(v, place)] = e
        wind_f = wide['wind_speed_100m_kmh'].stack()
        wind_a = other['wind_speed_100m_kmh'].reindex(wide['wind_speed_100m_kmh'].index).stack().reindex(wind_f.index)
        temp_f = wide['temperature_2m_c'].stack()
        temp_a = other['temperature_2m_c'].reindex(wide['temperature_2m_c'].index).stack().reindex(temp_f.index)
        err = (wind_f - wind_a).dropna()
        monthly_mae = (wide['wind_speed_100m_kmh'] - other['wind_speed_100m_kmh'].reindex(wide['wind_speed_100m_kmh'].index)).abs().resample('MS').mean()
        charts = [
            bar('Day-ahead error by variable and location', 'Mean absolute error against the stitched short-range forecast, in each '
                'variable\'s own unit (km/h, km/h, W/m², °C, %).', [VARS[v][0] for v in VARS],
                [(p, [mae_by_place[(v, p)]['mae'] for v in VARS]) for p in places], '', wide=True),
            scatter('Day-ahead wind against the short-range forecast', 'Wind at 100 m, every location and hour.', wind_f, wind_a,
                    'Day-ahead forecast (km/h)', 'Short-range forecast (km/h)', digits=1),
            hist('Day-ahead wind error', 'Day-ahead − short-range, wind at 100 m.', err, 'km/h', bins=60),
            line('Day-ahead wind error by month and location', 'Mean absolute error, wind at 100 m.',
                 [d.strftime('%Y-%m') for d in monthly_mae.index], [(p, monthly_mae[p]) for p in places], 'km/h', digits=2, wide=True),
            scatter('Day-ahead temperature against the short-range forecast', 'Every location and hour.', temp_f, temp_a,
                    'Day-ahead forecast (°C)', 'Short-range forecast (°C)', digits=1),
        ] + charts
        tables.insert(0, table('Day-ahead accuracy by variable and location',
                               ['Variable', 'Location', 'Hours', 'Bias', 'MAE', 'RMSE', 'r'], rows,
                               'Against weather_historical. Bias is day-ahead minus short-range, in the variable\'s unit.'))
        e = error_stats(wind_f, wind_a)
        tiles[3] = tile('Day-ahead wind error', f"{e['mae']:.1f} km/h", f"r {e['r']} against the short-range forecast")
        say('  day-ahead wind', e, 'temp', error_stats(temp_f, temp_a))
    section(table_name, title, source, 'location and hour (UTC)', ('location', 'time'), w.drop(columns=['place']), tiles, checks,
            charts, tables)
    say('  wind corr', wide['wind_speed_100m_kmh'].corr().round(2).to_dict())
    nulls = w[w[list(VARS)].isna().any(axis=1)]
    say('  null rows', null_rows, nulls.groupby('place').size().to_dict(), 'dates', nulls['time'].dt.date.value_counts().head(8).to_dict())
    say('  shear median', {p: round(s.median(), 2) for p, s in shear.items()}, 'missing hours', len(grid) - per_location.min())
    SHARED[f'{table_name}_daily'] = pd.DataFrame({'wind100': daily_wind.mean(axis=1), 'temp': wide['temperature_2m_c'].resample('D').mean().mean(axis=1),
                                                  'radiation': wide['shortwave_radiation_wm2'].resample('D').mean().mean(axis=1),
                                                  'cloud': wide['cloud_cover_pct'].resample('D').mean().mean(axis=1)})
    return wide


# --- Across tables ------------------------------------------------------------------------------------------

def cross():
    gen, wx, wx_da = SHARED['gen_daily'], SHARED['weather_historical_daily'], SHARED['weather_day_ahead_daily']
    daily = pd.DataFrame({
        'Renewable share': gen['renewable_share'], 'Wind output': gen['wind_mw'], 'Gas output': gen['gas_mw'],
        'Net imports': gen['net_imports_mw'], 'Demand': SHARED['demand_daily'], 'Solar (PV_Live)': SHARED['pv_daily'],
        'System price': SHARED['price_daily'], 'Market price': SHARED['apx_daily'], 'Carbon intensity': SHARED['carbon_daily'],
        'Wind speed 100 m': wx['wind100'], 'Temperature': wx['temp'], 'Radiation': wx['radiation'], 'Cloud cover': wx['cloud'],
        'Capacity on outage': SHARED['outage_daily'].reindex(gen.index),
    }).dropna(subset=['Renewable share'])
    fc_share = 100 * SHARED['dgws_daily'] / SHARED['ndf_known']
    target = daily['Renewable share']
    features = pd.DataFrame({
        'Forecast wind + solar ÷ forecast demand': fc_share, 'Day-ahead forecast wind + solar': SHARED['dgws_daily'],
        'Day-ahead weather: wind at 100 m': wx_da['wind100'], 'Forecast demand': SHARED['ndf_known'],
        'Day-ahead weather: temperature': wx_da['temp'], 'Day-ahead weather: radiation': wx_da['radiation'],
        'Share 2 days earlier': target.shift(2, freq='D'), 'Share 7 days earlier': target.shift(7, freq='D'),
        'Capacity on outage': daily['Capacity on outage'],
        'Weekend': pd.Series((target.index.dayofweek >= 5).astype(float), index=target.index),
    }).reindex(target.index)
    feature_r = features.corrwith(target).sort_values(key=abs, ascending=False)
    feature_rho = features.corrwith(target, method='spearman').reindex(feature_r.index)
    # A one-variable baseline, to show how far the strongest feature goes alone
    best = features['Forecast wind + solar ÷ forecast demand']
    fit = pd.DataFrame({'x': best, 'y': target}).dropna()
    slope, intercept = np.polyfit(fit['x'], fit['y'], 1)
    resid = fit['y'] - (slope * fit['x'] + intercept)
    persistence = (target - target.shift(2, freq='D')).dropna()
    baseline_rows = [['Always the mean', num(target.std(), 2), num((target - target.mean()).abs().mean(), 2), '0.00'],
                     ['Share 2 days earlier', num(np.sqrt((persistence ** 2).mean()), 2), num(persistence.abs().mean(), 2),
                      num(1 - (persistence ** 2).sum() / ((target.loc[persistence.index] - target.mean()) ** 2).sum(), 2)],
                     ['Linear fit on forecast wind + solar ÷ forecast demand', num(np.sqrt((resid ** 2).mean()), 2),
                      num(resid.abs().mean(), 2), num(1 - (resid ** 2).sum() / ((fit['y'] - fit['y'].mean()) ** 2).sum(), 2)]]
    tests = []
    for name in ('Renewable share', 'Wind output', 'Demand', 'System price', 'Carbon intensity', 'Solar (PV_Live)', 'Temperature'):
        s = daily[name].dropna()
        adf = adfuller(s, autolag='AIC')
        a = acf(s, nlags=7)
        tests.append([name, f'{len(s):,}', num(s.mean(), 1), num(s.std(), 1), num(s.skew(), 2), num(s.kurt(), 2), num(a[1], 2),
                      num(a[7], 2), num(adf[0], 2), f'{adf[1]:.3f}', 'stationary' if adf[1] < 0.05 else 'unit root not rejected'])
    half = pd.DataFrame({'price': SHARED['price_half']})
    g = SHARED['gen_half']
    half['wind'] = g['wind'].reindex(half.index)
    half['demand'] = SHARED['demand_half'].reindex(half.index)
    half['residual'] = half['demand'] - half['wind'] - SHARED['pv_half'].reindex(half.index).fillna(0)
    # Coverage: rows per month relative to each table's typical month
    cover_spec = [('fuelhh', 'settlement_date'), ('demand_outturn', 'settlement_date'), ('wind_forecast', 'start_time'),
                  ('demand_forecast', 'settlement_date'), ('wind_solar_forecast', 'settlement_date'),
                  ('wind_solar_actual', 'settlement_date'), ('system_prices', 'settlement_date'), ('market_index', 'settlement_date'),
                  ('remit_outages', 'publish_time'), ('carbon_intensity', 'period_start'), ('pv_live', 'datetime_gmt'),
                  ('neso_demand', 'settlement_date'), ('weather_day_ahead', 'time'), ('weather_historical', 'time')]
    months = pd.period_range('2023-11', '2026-11', freq='M')
    cover, names = [], []
    for t, col in cover_spec:
        s = pd.to_datetime(pd.read_parquet(OUT / 'silver' / f'{t}.parquet', columns=[col])[col])
        per_day = s.groupby(s.dt.to_period('M')).size().reindex(months).fillna(0) / months.days_in_month
        cover.append((100 * per_day / per_day[per_day > 0].median()).clip(upper=150).to_numpy())
        names.append(t)
    lagged = pd.DataFrame({f'{k} days': [target.corr(target.shift(k, freq='D'))] for k in range(1, 15)}).iloc[0]
    wind_power = pd.DataFrame({'speed': wx['wind100'], 'wind': gen['wind_mw']}).dropna()
    speed_bins = wind_power.groupby(pd.cut(wind_power['speed'], np.arange(5, 75, 5)))['wind'].agg(['mean', 'count'])
    speed_bins = speed_bins[speed_bins['count'] >= 5]
    charts = [
        heat('Data coverage by month', 'Rows per day in each month, as a percentage of that table\'s typical month. Cells that stand out from '
             'their row are thin or missing months. Bank holidays is a calendar and is not shown.', [str(m) for m in months], names, cover, '%',
             digits=0, wide=True),
        corr_heat(daily, 'Correlation between daily measures', 'Pearson r on daily means. Blue: move together; red: move opposite.'),
        bar('Correlation of each day-ahead feature with the target', 'Pearson r with the daily renewable share. Every feature is '
            'known at 18:00 UTC the day before.', feature_r.index, [('Pearson r', feature_r.to_numpy()), ('Spearman ρ', feature_rho.to_numpy())],
            '', horizontal=True, digits=2, wide=True),
        scatter('The strongest feature: forecast wind + solar over forecast demand', 'Each point is one target day.', best, target,
                'Forecast wind + solar ÷ forecast demand (%)', 'Renewable share (%)'),
        scatter('Renewable share against wind speed', 'Daily mean wind at 100 m across the five points.', daily['Wind speed 100 m'],
                target, 'Wind speed at 100 m (km/h)', 'Renewable share (%)'),
        line('The fleet power curve', 'Mean FUELHH wind output by daily wind-speed band (5 km/h bands with at least 5 days).',
             [f'{int(i.left)}–{int(i.right)}' for i in speed_bins.index], [('Mean wind output', speed_bins['mean'])], 'MW',
             area=True, digits=0),
        scatter('Demand against temperature', 'Daily means. Heating drives demand; it flattens above about 15 °C.',
                daily['Temperature'], daily['Demand'], 'Temperature (°C)', 'National demand (MW)', digits=1),
        scatter('System price against residual demand', 'Half-hourly. Residual demand = demand − wind − solar: what gas must cover.',
                half['residual'], half['price'], 'Residual demand (MW)', 'System price (£/MWh)', digits=1),
        scatter('Market price against gas output', 'Daily means.', daily['Gas output'], daily['Market price'],
                'Gas output (MW)', 'APX market index (£/MWh)', digits=1),
        scatter('Carbon intensity against renewable share', 'Daily means.', target, daily['Carbon intensity'],
                'Renewable share (%)', 'Carbon intensity (gCO2/kWh)', digits=1),
        scatter('Net imports against wind output', 'Daily means. Britain imports less, and exports, when it is windy.',
                daily['Wind output'], daily['Net imports'], 'Wind output (MW)', 'Net imports (MW)', digits=0),
        bar('How far back the target remembers', 'Correlation of the daily renewable share with itself k days earlier.',
            lagged.index, [('Correlation', lagged.to_numpy())], '', digits=2),
    ]
    tables = [table('Simple baselines for the daily renewable share', ['Predictor', 'RMSE (points)', 'MAE (points)', 'R²'],
                    baseline_rows, 'In percentage points of renewable share, in-sample. A bar any model should clear.'),
              table('Distribution and stationarity of the daily series',
                    ['Series', 'Days', 'Mean', 'Std dev', 'Skew', 'Excess kurtosis', 'ACF lag 1', 'ACF lag 7', 'ADF statistic',
                     'ADF p-value', 'Reading'], tests,
                    'Augmented Dickey–Fuller test, lag chosen by AIC. A p-value below 0.05 rejects a unit root.'),
              table('Day-ahead features against the target', ['Feature', 'Pearson r', 'Spearman ρ', 'Days'],
                    [[k, num(feature_r[k], 3), num(feature_rho[k], 3), int(features[k].notna().sum())] for k in feature_r.index])]
    say('\n===== cross')
    say('  feature r', feature_r.round(3).to_dict())
    say('  baselines', baseline_rows)
    say('  tests', tests)
    say('  daily corr with share', daily.corr()['Renewable share'].round(2).to_dict())
    say('  daily corr with demand', daily.corr()['Demand'].round(2).to_dict())
    say('  daily corr with price', daily.corr()['System price'].round(2).to_dict())
    say('  price-residual r', round(half['price'].corr(half['residual']), 3), 'price-wind', round(half['price'].corr(half['wind']), 3),
        'price-demand', round(half['price'].corr(half['demand']), 3))
    say('  lagged', lagged.round(2).to_dict())
    say('  trainable days', int(features[['Forecast wind + solar ÷ forecast demand', 'Day-ahead weather: wind at 100 m',
                                         'Share 2 days earlier']].notna().all(axis=1).sum()), 'of', len(target))
    return {'charts': charts, 'tables': tables}


fuelhh()
demand_outturn()
wind_forecast()
demand_forecast()
wind_solar_actual()
wind_solar_forecast()
system_prices()
market_index()
remit_outages()
bank_holidays()
carbon_intensity()
pv_live()
neso_demand()
historical = weather('weather_historical', 'Weather: short-range forecasts', 'Open-Meteo historical forecast API')
weather('weather_day_ahead', 'Weather: day-ahead forecasts', 'Open-Meteo previous-runs API (previous_day1)', other=historical)
CROSS = cross()

ORDER = ['fuelhh', 'demand_outturn', 'wind_forecast', 'demand_forecast', 'wind_solar_forecast', 'wind_solar_actual',
         'system_prices', 'market_index', 'remit_outages', 'bank_holidays', 'carbon_intensity', 'pv_live', 'neso_demand',
         'weather_day_ahead', 'weather_historical']
SECTIONS.sort(key=lambda s: ORDER.index(s['id']))
report = sanitize({'generated': '2026-11-16', 'sections': SECTIONS, 'cross': CROSS})
(OUT / 'report.json').write_text(json.dumps(report, allow_nan=False, separators=(',', ':')), encoding='utf-8')
(OUT / 'digest.txt').write_text('\n'.join(DIGEST), encoding='utf-8')
print(f"report.json {(OUT / 'report.json').stat().st_size / 1e6:.2f} MB, "
      f"{sum(len(s['charts']) for s in SECTIONS) + len(CROSS['charts'])} charts")
