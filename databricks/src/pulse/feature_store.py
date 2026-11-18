"""The day-ahead feature store: one row per target day D, using only what was known at the forecast cutoff
(18:00 UTC on D-1). The target is D's renewable share of GB generation.

Plain pandas, so the same code runs in the EDA (eda/features.py, on a local copy of silver) and in the Databricks
job (notebooks/feature_store.py, on the silver tables). Nothing here reads or writes storage.

- availability(): when each silver row could first have been read. This is the whole leakage contract:
  build() uses a value only if it was available at the cutoff.
- build(): every feature for every target day, plus metadata (definition, source table, when it is known).
- prepare(): silver tables as Spark hands them to pandas -> the types build() expects.
- feature_hash(): a fingerprint of a row's values, so a rerun only rewrites days whose features changed.

Change a feature's definition -> bump FEATURE_SET_VERSION, so every row says which logic produced it.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from pulse.settings import UK_TZ

UK = UK_TZ
# Bump when any feature's definition, or the set of features, changes
FEATURE_SET_VERSION = 'v1'

H = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
CUTOFF_HOUR = 18
TABLES = ('fuelhh', 'demand_outturn', 'wind_forecast', 'demand_forecast', 'wind_solar_forecast', 'wind_solar_actual',
          'system_prices', 'market_index', 'remit_outages', 'bank_holidays', 'carbon_intensity', 'pv_live', 'neso_demand',
          'weather_day_ahead', 'weather_historical')
LOCATIONS = ('dogger_bank', 'east_anglia_offshore', 'irish_sea', 'scottish_highlands', 'southern_england')
OFFSHORE = ('dogger_bank', 'east_anglia_offshore', 'irish_sea')


# --- UK settlement time ----------------------------------------------------------------------------------------

def period_start(dates, periods):
    """The UTC instant each settlement period starts."""
    local = pd.to_datetime(dates).dt.tz_localize(UK)
    return (local.dt.tz_convert('UTC') + pd.to_timedelta((periods - 1) * 30, unit='m')).dt.tz_localize(None)


def periods_expected(dates):
    """Settlement periods in each UK day: 46, 48 or 50."""
    d = pd.DatetimeIndex(dates)
    a, b = d.tz_localize(UK), (d + pd.Timedelta(days=1)).tz_localize(UK)
    return pd.Series(((b - a) / pd.Timedelta(minutes=30)).astype(int), index=d)


def uk_date(ts):
    """The UK calendar date (as a naive midnight timestamp) of UTC-naive instants."""
    return ts.dt.tz_localize('UTC').dt.tz_convert(UK).dt.tz_localize(None).dt.normalize()


# --- When each row could first have been read -----------------------------------------------------------------

def availability(raw):
    """table -> Series of the instant each row became available. This is the whole leakage contract."""
    return {
        'fuelhh': raw['fuelhh']['publish_time'],
        'demand_outturn': raw['demand_outturn']['publish_time'],
        'wind_forecast': raw['wind_forecast']['publish_time'],
        'demand_forecast': raw['demand_forecast']['publish_time'],
        'wind_solar_forecast': raw['wind_solar_forecast']['publish_time'],
        'wind_solar_actual': raw['wind_solar_actual']['publish_time'],
        # created_at is the latest settlement run, days later for most rows: assume the first price an hour after the period starts
        'system_prices': raw['system_prices']['start_time'] + H,
        # No publish time in the feed: assume the end of the half-hour
        'market_index': raw['market_index']['start_time'] + 0.5 * H,
        'remit_outages': raw['remit_outages']['publish_time'],
        # Known years ahead
        'bank_holidays': pd.Series(pd.Timestamp('2000-01-01'), index=raw['bank_holidays'].index),
        # Assumed: half an hour after the half-hour ends
        'carbon_intensity': raw['carbon_intensity']['period_end'] + 0.5 * H,
        'pv_live': raw['pv_live']['datetime_gmt'] + 0.5 * H,
        # NESO publishes a settlement day late: assume usable from the end of the following day
        'neso_demand': raw['neso_demand']['settlement_date'] + 2 * DAY,
        # previous_day1: the forecast issued about 24 hours before each hour
        'weather_day_ahead': raw['weather_day_ahead']['time'] - 24 * H,
        # Stitched short-range forecasts: assume an hour after the hour
        'weather_historical': raw['weather_historical']['time'] + H,
    }


AVAILABILITY_NOTE = {
    'fuelhh': 'publish_time', 'demand_outturn': 'publish_time', 'wind_forecast': 'publish_time', 'demand_forecast': 'publish_time',
    'wind_solar_forecast': 'publish_time', 'wind_solar_actual': 'publish_time', 'system_prices': 'one hour after the period starts (assumed; the table keeps the latest settlement run, so a value may be a later revision)',
    'market_index': 'end of the half-hour (assumed)', 'remit_outages': 'publish_time', 'bank_holidays': 'known in advance',
    'carbon_intensity': '30 minutes after the half-hour ends (assumed)', 'pv_live': '30 minutes after the half-hour ends (assumed)',
    'neso_demand': 'end of the day after the settlement day (assumed)',
    'weather_day_ahead': '24 hours before each hour (previous_day1); hours after 18:00 UTC of the target day were issued after the cutoff',
    'weather_historical': 'one hour after each hour (assumed)',
}


def truncate(raw, cutoff):
    """The tables as they stood at the cutoff. Day-ahead weather keeps the target day: see the note on its features."""
    avail = availability(raw)
    out = {}
    for name, df in raw.items():
        limit = cutoff + 30 * H if name == 'weather_day_ahead' else cutoff
        out[name] = df[avail[name] <= limit].copy()
    return out


# --- Building blocks ------------------------------------------------------------------------------------------

def before_cutoff(values, available, first_hour=0, last_hour=CUTOFF_HOUR):
    """Half-hourly or hourly values (indexed by UTC start) of the day before each target day, between two UTC hours,
    that were available at the cutoff. Returns a groupby keyed by target day."""
    t = values.index
    ok = (t.hour >= first_hour) & (t.hour < last_hour) & (available.to_numpy() <= (t.normalize() + CUTOFF_HOUR * H).to_numpy())
    kept = values[ok]
    return kept.groupby(kept.index.normalize() + DAY)


def power_curve(kmh):
    """Share of rated output of a generic turbine at a hub-height wind speed: cubic from 3 to 12 m/s, flat to 25, then off."""
    v = kmh / 3.6
    p = ((v ** 3 - 27) / (1728 - 27)).clip(0, 1)
    return p.where(v <= 25, 0.0)


def daylight_hours(dates, latitude=54.0):
    doy = dates.dayofyear.to_numpy()
    declination = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365.25)
    return 24 / np.pi * np.arccos(np.clip(-np.tan(np.radians(latitude)) * np.tan(declination), -1, 1))


class Features:
    def __init__(self, index):
        self.frame = pd.DataFrame(index=index)
        self.meta = []

    def add(self, name, values, source, definition, known='at the cutoff'):
        aligned = values.reindex(self.frame.index) if isinstance(values, pd.Series) else (np.nan if values is None else np.asarray(values))
        self.frame[name] = aligned
        self.frame[name] = self.frame[name].astype(float)
        self.meta.append({'feature': name, 'source': source, 'definition': definition, 'known': known})


# --- The feature table ----------------------------------------------------------------------------------------

def build(raw, remit_days=None):
    """All features for every target day the data can support. remit_days limits the (slow) outage features."""
    fuel = raw['fuelhh']
    wide = fuel.pivot(index=['settlement_date', 'settlement_period'], columns='fuel_type', values='generation_mw')
    ints = [c for c in wide.columns if c.startswith('INT')]
    dates = wide.index.get_level_values(0)
    start = period_start(pd.Series(dates), pd.Series(wide.index.get_level_values(1))).to_numpy()
    published = fuel.groupby(['settlement_date', 'settlement_period'])['publish_time'].max().reindex(wide.index).to_numpy()
    half = pd.DataFrame({'wind': wide['WIND'].to_numpy(), 'gas': (wide['CCGT'] + wide['OCGT']).to_numpy(),
                         'nuclear': wide['NUCLEAR'].to_numpy(), 'biomass': wide['BIOMASS'].to_numpy(), 'hydro': wide['NPSHYD'].to_numpy(),
                         'imports': wide[ints].sum(axis=1).to_numpy(), 'domestic': wide.drop(columns=ints).sum(axis=1).to_numpy(),
                         'renewable': wide[['WIND', 'NPSHYD', 'BIOMASS']].sum(axis=1).to_numpy(), 'published': published}, index=start)
    present = wide['WIND'].groupby(dates).count()
    complete = present == periods_expected(present.index)
    daily = half.drop(columns='published').groupby(dates.to_numpy()).mean()[complete.to_numpy()]
    daily.index = pd.DatetimeIndex(daily.index)
    share = 100 * daily['renewable'] / daily['domestic']
    index = pd.date_range(dates.min(), dates.max() + 2 * DAY, freq='D')
    F = Features(index)

    def lag(daily, days):
        """A daily value from `days` days before each target day."""
        return daily.reindex(index).shift(days)

    def rolling(daily, window, how='mean', days=2):
        """A rolling statistic over the days ending `days` days before the target day."""
        return getattr(lag(daily, days).rolling(window, min_periods=max(2, window // 2)), how)()

    doy = index.dayofyear.to_numpy()

    # Calendar (and bank_holidays)
    F.add('cal_dow_sin', np.sin(2 * np.pi * index.dayofweek / 7), 'calendar', 'Day of week, sine')
    F.add('cal_dow_cos', np.cos(2 * np.pi * index.dayofweek / 7), 'calendar', 'Day of week, cosine')
    F.add('cal_is_weekend', (index.dayofweek >= 5).astype(float), 'calendar', 'Saturday or Sunday')
    F.add('cal_doy_sin', np.sin(2 * np.pi * doy / 365.25), 'calendar', 'Day of year, sine')
    F.add('cal_doy_cos', np.cos(2 * np.pi * doy / 365.25), 'calendar', 'Day of year, cosine')
    F.add('cal_daylight_hours', daylight_hours(index), 'calendar', 'Astronomical day length at 54°N')
    F.add('cal_periods_in_day', periods_expected(index).to_numpy(), 'calendar', 'Settlement periods in the day: 46, 48 or 50')
    bank = raw['bank_holidays']
    for division, tag in (('england-and-wales', 'ew'), ('scotland', 'scotland'), ('northern-ireland', 'ni')):
        days = pd.DatetimeIndex(bank.loc[bank['division'] == division, 'holiday_date'])
        F.add(f'hol_is_bank_holiday_{tag}', pd.Series(index.isin(days).astype(float), index=index), 'bank_holidays', f'Bank holiday in {division}')
    ew = pd.DatetimeIndex(bank.loc[bank['division'] == 'england-and-wales', 'holiday_date'])
    F.add('hol_adjacent_ew', pd.Series((index.isin(ew + DAY) | index.isin(ew - DAY)).astype(float), index=index), 'bank_holidays',
          'The day before or after an England and Wales bank holiday')
    F.add('hol_christmas_week', pd.Series((((index.month == 12) & (index.day >= 24)) | ((index.month == 1) & (index.day <= 1))).astype(float), index=index),
          'bank_holidays', '24 December to 1 January')
    nxt = np.searchsorted(ew.sort_values().to_numpy(), index.to_numpy())
    ew_sorted = ew.sort_values()
    days_to = [(ew_sorted[i] - d).days if i < len(ew_sorted) else np.nan for i, d in zip(nxt, index)]
    F.add('hol_days_to_next_ew', pd.Series(np.minimum(days_to, 30), index=index), 'bank_holidays', 'Days to the next England and Wales bank holiday, capped at 30')

    # FUELHH: yesterday so far, and complete days further back
    pub = half['published']
    w = lambda col, **k: before_cutoff(half[col], pub, **k)
    so_far = 100 * w('renewable').sum() / w('domestic').sum()
    F.add('fuel_share_d1_until_cutoff', so_far, 'fuelhh', 'Renewable share over yesterday 00:00–18:00 UTC')
    for col, label in (('wind', 'Wind'), ('gas', 'Gas'), ('nuclear', 'Nuclear'), ('biomass', 'Biomass'), ('hydro', 'Hydro'), ('imports', 'Net imports'),
                       ('domestic', 'GB generation')):
        F.add(f'fuel_{col}_d1_mw', w(col).mean(), 'fuelhh', f'{label}, mean MW over yesterday 00:00–18:00 UTC')
    last3 = w('wind', first_hour=15).mean()
    F.add('fuel_wind_last3h_mw', last3, 'fuelhh', 'Wind, mean MW over the last three hours before the cutoff')
    F.add('fuel_wind_ramp_3h_mw', last3 - w('wind', first_hour=12, last_hour=15).mean(), 'fuelhh', 'Wind in the last three hours minus the three before: is it rising or falling')
    F.add('fuel_share_last3h', 100 * w('renewable', first_hour=15).sum() / w('domestic', first_hour=15).sum(), 'fuelhh', 'Renewable share over the last three hours before the cutoff')
    F.add('fuel_baseload_d1_mw', w('nuclear').mean() + w('biomass').mean() + w('hydro').mean(), 'fuelhh', 'Nuclear + biomass + hydro yesterday: the steady low-carbon base')
    F.add('fuel_share_lag2', lag(share, 2), 'fuelhh', 'Renewable share two days earlier (complete day)', 'two days before')
    F.add('fuel_share_lag7', lag(share, 7), 'fuelhh', 'Renewable share seven days earlier', 'a week before')
    F.add('fuel_share_roll7_mean', rolling(share, 7), 'fuelhh', 'Mean share over the 7 days ending two days earlier', 'two days before')
    F.add('fuel_share_roll7_std', rolling(share, 7, 'std'), 'fuelhh', 'Std dev of the share over the same 7 days: how changeable the weather is', 'two days before')
    F.add('fuel_share_roll28_mean', rolling(share, 28), 'fuelhh', 'Mean share over the 28 days ending two days earlier', 'two days before')
    F.add('fuel_wind_lag2_mw', lag(daily['wind'], 2), 'fuelhh', 'Wind two days earlier, daily mean MW', 'two days before')

    # Demand outturn
    d = raw['demand_outturn']
    dem = d.set_index('start_time')['initial_demand_outturn_mw']
    dem_pub = d.set_index('start_time')['publish_time']
    dem_daily = d.groupby('settlement_date')['initial_demand_outturn_mw'].agg(['mean', 'count'])
    dem_daily = dem_daily.loc[dem_daily['count'] >= 46, 'mean']
    F.add('dem_d1_until_cutoff_mw', before_cutoff(dem, dem_pub).mean(), 'demand_outturn', 'Demand, mean MW over yesterday 00:00–18:00 UTC')
    F.add('dem_lag2_mw', lag(dem_daily, 2), 'demand_outturn', 'Demand two days earlier, daily mean', 'two days before')
    F.add('dem_lag7_mw', lag(dem_daily, 7), 'demand_outturn', 'Demand seven days earlier: the same weekday', 'a week before')
    F.add('dem_roll7_mw', rolling(dem_daily, 7), 'demand_outturn', 'Mean demand over the 7 days ending two days earlier', 'two days before')

    # WINDFOR: the latest run before the cutoff, for the hours of the target day
    wf = raw['wind_forecast'].copy()
    wf['day'] = uk_date(wf['start_time'])
    wf['cutoff'] = wf['day'] - (24 - CUTOFF_HOUR) * H

    def latest_run(frame, by):
        return frame.sort_values('publish_time').groupby(by).tail(1)

    known = latest_run(wf[wf['publish_time'] <= wf['cutoff']], ['day', 'start_time'])
    g = known.groupby('day')['generation_mw']
    F.add('wf_mean_mw', g.mean(), 'wind_forecast', 'WINDFOR wind forecast for the target day, mean MW, latest run before the cutoff')
    F.add('wf_max_mw', g.max(), 'wind_forecast', 'The same forecast, highest hour')
    F.add('wf_min_mw', g.min(), 'wind_forecast', 'The same forecast, lowest hour')
    F.add('wf_std_mw', g.std(), 'wind_forecast', 'Std dev across the hours: how much wind changes within the day')
    F.add('wf_hours', g.count(), 'wind_forecast', 'Hours of the target day the forecast covers')
    first6 = known[known['start_time'].dt.hour < 6].groupby('day')['generation_mw'].mean()
    last6 = known[known['start_time'].dt.hour >= 18].groupby('day')['generation_mw'].mean()
    F.add('wf_ramp_mw', last6 - first6, 'wind_forecast', 'Forecast for the last six hours minus the first six: wind building or dying')
    earlier = latest_run(wf[wf['publish_time'] <= wf['cutoff'] - 24 * H], ['day', 'start_time']).groupby('day')['generation_mw'].mean()
    F.add('wf_revision_24h_mw', g.mean() - earlier, 'wind_forecast', 'Latest forecast minus the one made 24 hours before: which way the forecast is moving')
    # How the forecast has been running against FUELHH wind: the two cover different fleets, so calibrate
    hourly_wind = half['wind'].resample('1h').mean()
    view = known.set_index('start_time')['generation_mw']
    pair = pd.DataFrame({'forecast': view, 'actual': hourly_wind.reindex(view.index)}).dropna()
    pair_pub = pd.Series(pair.index + H + 0.5 * H, index=pair.index)
    ratio_d1 = before_cutoff(pair['actual'], pair_pub).sum() / before_cutoff(pair['forecast'], pair_pub).sum()
    F.add('wf_calibration_d1', ratio_d1, 'wind_forecast', 'FUELHH wind ÷ WINDFOR forecast over yesterday 00:00–18:00 UTC')
    daily_ratio = (pair['actual'].groupby(pair.index.normalize()).sum() / pair['forecast'].groupby(pair.index.normalize()).sum())
    wf_cal = rolling(daily_ratio, 7)
    F.add('wf_calibration_7d', wf_cal, 'wind_forecast', 'The same ratio over the 7 days ending two days earlier', 'two days before')
    F.add('wf_calibrated_mw', F.frame['wf_mean_mw'] * wf_cal.reindex(index), 'wind_forecast', 'WINDFOR forecast × its 7-day calibration: an estimate of FUELHH wind')

    # NDF demand forecast
    nd = raw['demand_forecast']
    nd = nd[nd['publish_time'] <= nd['settlement_date'] - (24 - CUTOFF_HOUR) * H]
    nd = latest_run(nd, ['settlement_date', 'settlement_period'])
    g = nd.groupby('settlement_date')['demand_mw']
    F.add('ndf_mean_mw', g.mean(), 'demand_forecast', 'National demand forecast for the target day, mean MW, latest run before the cutoff')
    F.add('ndf_peak_mw', g.max(), 'demand_forecast', 'The same forecast, peak half-hour')
    F.add('ndf_min_mw', g.min(), 'demand_forecast', 'The same forecast, lowest half-hour')
    F.add('ndf_periods', g.count(), 'demand_forecast', 'Settlement periods of the target day the forecast covers')
    nd_view = nd.set_index('start_time')['demand_mw']
    err = pd.DataFrame({'forecast': nd_view, 'actual': dem.reindex(nd_view.index), 'pub': dem_pub.reindex(nd_view.index)}).dropna()
    F.add('ndf_error_d1_mw', before_cutoff(err['forecast'] - err['actual'], err['pub']).mean(), 'demand_forecast',
          'Yesterday\'s demand forecast minus outturn, 00:00–18:00 UTC: the errors are autocorrelated')

    # DGWS day-ahead wind and solar forecast
    ws = raw['wind_solar_forecast']
    ws = ws[(ws['process_type'] == 'Day ahead') & (ws['publish_time'] <= ws['settlement_date'] - (24 - CUTOFF_HOUR) * H)]
    ws = latest_run(ws, ['settlement_date', 'settlement_period', 'psr_type'])
    per = ws.pivot(index=['settlement_date', 'settlement_period'], columns='psr_type', values='quantity_mw')
    per['wind'] = per['Wind Offshore'] + per['Wind Onshore']
    per['demand'] = nd.set_index(['settlement_date', 'settlement_period'])['demand_mw'].reindex(per.index)
    per['residual'] = per['demand'] - per['wind'] - per['Solar']
    g = per.groupby(level=0)
    F.add('fc_solar_mean_mw', g['Solar'].mean(), 'wind_solar_forecast', 'Day-ahead solar forecast, mean MW')
    F.add('fc_solar_peak_mw', g['Solar'].max(), 'wind_solar_forecast', 'Day-ahead solar forecast, peak half-hour')
    F.add('fc_wind_onshore_mean_mw', g['Wind Onshore'].mean(), 'wind_solar_forecast', 'Day-ahead onshore wind forecast, mean MW')
    F.add('fc_wind_offshore_mean_mw', g['Wind Offshore'].mean(), 'wind_solar_forecast', 'Day-ahead offshore wind forecast, mean MW')
    F.add('fc_wind_mean_mw', g['wind'].mean(), 'wind_solar_forecast', 'Day-ahead wind forecast, onshore + offshore, mean MW')
    F.add('fc_wind_min_mw', g['wind'].min(), 'wind_solar_forecast', 'The same forecast, lowest half-hour')
    F.add('fc_wind_max_mw', g['wind'].max(), 'wind_solar_forecast', 'The same forecast, highest half-hour')
    F.add('fc_wind_std_mw', g['wind'].std(), 'wind_solar_forecast', 'Std dev across the half-hours')
    F.add('fc_wind_solar_mean_mw', g['wind'].mean() + g['Solar'].mean(), 'wind_solar_forecast', 'Day-ahead wind + solar forecast, mean MW (the gold feature)')
    F.add('fc_offshore_share_of_wind', g['Wind Offshore'].mean() / g['wind'].mean(), 'wind_solar_forecast', 'Offshore share of forecast wind')
    F.add('fc_wind_share_of_demand', 100 * g['wind'].sum() / g['demand'].sum(), 'wind_solar_forecast', 'Forecast wind as a share of forecast demand')
    F.add('fc_wind_solar_share_of_demand', 100 * (g['wind'].sum() + g['Solar'].sum()) / g['demand'].sum(), 'wind_solar_forecast',
          'Forecast wind + solar as a share of forecast demand')
    F.add('fc_residual_demand_mean_mw', g['residual'].mean(), 'wind_solar_forecast', 'Forecast demand − wind − solar, mean: what gas and imports must cover')
    F.add('fc_residual_demand_max_mw', g['residual'].max(), 'wind_solar_forecast', 'The same, tightest half-hour')
    F.add('fc_residual_demand_min_mw', g['residual'].min(), 'wind_solar_forecast', 'The same, slackest half-hour: low values mean curtailment or exports')
    # Calibration against FUELHH wind, as for WINDFOR
    fc_start = period_start(pd.Series(per.index.get_level_values(0)), pd.Series(per.index.get_level_values(1))).to_numpy()
    pair = pd.DataFrame({'forecast': per['wind'].to_numpy()}, index=fc_start)
    pair['actual'] = half['wind'].reindex(pair.index)
    pair['pub'] = half['published'].reindex(pair.index)
    pair = pair.dropna()
    F.add('fc_calibration_d1', before_cutoff(pair['actual'], pair['pub']).sum() / before_cutoff(pair['forecast'], pair['pub']).sum(), 'wind_solar_forecast',
          'FUELHH wind ÷ day-ahead wind forecast over yesterday 00:00–18:00 UTC')
    daily_ratio = pair['actual'].groupby(pair.index.normalize()).sum() / pair['forecast'].groupby(pair.index.normalize()).sum()
    fc_cal = rolling(daily_ratio, 7)
    F.add('fc_calibration_7d', fc_cal, 'wind_solar_forecast', 'The same ratio over the 7 days ending two days earlier', 'two days before')
    calibrated = F.frame['fc_wind_mean_mw'] * fc_cal.reindex(index)
    F.add('fc_wind_calibrated_mw', calibrated, 'wind_solar_forecast', 'Day-ahead wind forecast × its 7-day calibration: an estimate of FUELHH wind')
    # A physical estimate of the target: calibrated wind plus yesterday's steady renewables, over forecast demand less imports
    renewables = calibrated + F.frame['fuel_biomass_d1_mw'] + F.frame['fuel_hydro_d1_mw']
    domestic = (F.frame['ndf_mean_mw'] - F.frame['fuel_imports_d1_mw']).clip(lower=1)
    F.add('fc_share_estimate', (100 * renewables / domestic).clip(0, 100), 'wind_solar_forecast',
          '(calibrated wind forecast + yesterday\'s biomass and hydro) ÷ (forecast demand − yesterday\'s net imports): the target, estimated from first principles')

    # AGWS actuals: solar is the useful part; the wind ratio is a health check on the feed
    ag = raw['wind_solar_actual']
    ag_wide = ag.pivot(index=['settlement_date', 'settlement_period'], columns='psr_type', values='quantity_mw')
    ag_start = period_start(pd.Series(ag_wide.index.get_level_values(0)), pd.Series(ag_wide.index.get_level_values(1))).to_numpy()
    ag_pub = ag.groupby(['settlement_date', 'settlement_period'])['publish_time'].max().reindex(ag_wide.index).to_numpy()
    agh = pd.DataFrame({'solar': ag_wide['Solar'].to_numpy(), 'wind': (ag_wide['Wind Offshore'] + ag_wide['Wind Onshore']).to_numpy(), 'pub': ag_pub}, index=ag_start)
    agh = agh[agh['pub'].notna()]
    F.add('agws_solar_d1_mw', before_cutoff(agh['solar'], agh['pub']).mean(), 'wind_solar_actual', 'AGWS solar, mean MW over yesterday 00:00–18:00 UTC (as published by the cutoff)')
    F.add('agws_wind_d1_mw', before_cutoff(agh['wind'], agh['pub']).mean(), 'wind_solar_actual', 'AGWS onshore + offshore wind, same window')
    # AGWS republishes old periods; a republished value is later knowledge, so the health ratio uses prompt publishes only
    prompt = agh[agh['pub'].to_numpy() <= (agh.index + 12 * H).to_numpy()]
    ag_ratio = rolling(prompt['wind'].groupby(prompt.index.normalize()).mean() / half['wind'].groupby(half.index.normalize()).mean(), 7, days=3)
    F.add('agws_to_fuelhh_7d', ag_ratio, 'wind_solar_actual', 'AGWS wind ÷ FUELHH wind over the 7 days ending three days earlier (AGWS can publish late)', 'three days before')
    F.add('agws_is_healthy', (ag_ratio >= 1.0).astype(float).where(ag_ratio.notna()), 'wind_solar_actual', 'That ratio is at least 1: the feed is not under-reporting', 'three days before')

    # System prices and the market index
    sp = raw['system_prices'].set_index('start_time')
    price, created = sp['system_sell_price_gbp_mwh'], pd.Series(sp.index + H, index=sp.index)
    F.add('px_system_d1', before_cutoff(price, created).mean(), 'system_prices', 'System price, mean £/MWh over yesterday until the cutoff')
    F.add('px_system_d1_min', before_cutoff(price, created).min(), 'system_prices', 'System price, lowest half-hour yesterday')
    F.add('px_negative_share_d1', 100 * before_cutoff((price < 0).astype(float), created).mean(), 'system_prices', 'Share of yesterday\'s half-hours with a negative price')
    F.add('px_imbalance_d1_mwh', before_cutoff(sp['net_imbalance_volume_mwh'], created).mean(), 'system_prices', 'Net imbalance volume, mean: negative when the system is long')
    F.add('px_long_share_d1', 100 * before_cutoff((sp['net_imbalance_volume_mwh'] < 0).astype(float), created).mean(), 'system_prices', 'Share of yesterday\'s half-hours with the system long')
    sp_daily = raw['system_prices'].groupby('settlement_date')['system_sell_price_gbp_mwh'].median()
    F.add('px_system_roll7_median', rolling(sp_daily, 7, 'median'), 'system_prices', 'Median of daily median prices over the 7 days ending two days earlier', 'two days before')
    mi = raw['market_index']
    apx = mi[(mi['data_provider'] == 'APXMIDP') & (mi['price_gbp_mwh'] != 0)].set_index('start_time')
    apx_pub = pd.Series(apx.index + 0.5 * H, index=apx.index)
    F.add('mid_apx_d1', before_cutoff(apx['price_gbp_mwh'], apx_pub).mean(), 'market_index', 'APX market index, mean £/MWh over yesterday until the cutoff')
    F.add('mid_apx_volume_d1_mwh', before_cutoff(apx['volume_mwh'], apx_pub).mean(), 'market_index', 'APX traded volume, mean MWh per half-hour yesterday')
    apx_daily = apx.groupby('settlement_date')['price_gbp_mwh'].mean()
    F.add('mid_apx_roll7', rolling(apx_daily, 7), 'market_index', 'Mean APX price over the 7 days ending two days earlier', 'two days before')
    F.add('mid_apx_vs_week', F.frame['mid_apx_d1'] - rolling(apx_daily, 7).reindex(index), 'market_index', 'Yesterday\'s APX price minus its 7-day mean: cheap days are windy days')

    # Carbon intensity, PV_Live
    ci = raw['carbon_intensity'].set_index('period_start')
    ci_pub = ci['period_end'] + 0.5 * H
    F.add('co2_d1_until_cutoff', before_cutoff(ci['actual_gco2_kwh'].dropna(), ci_pub[ci['actual_gco2_kwh'].notna()]).mean(), 'carbon_intensity',
          'Carbon intensity, mean gCO2/kWh over yesterday until the cutoff')
    ci_daily = ci['actual_gco2_kwh'].resample('D').mean()
    F.add('co2_lag2', lag(ci_daily, 2), 'carbon_intensity', 'Carbon intensity two days earlier, daily mean', 'two days before')
    F.add('co2_roll7', rolling(ci_daily, 7), 'carbon_intensity', 'Mean carbon intensity over the 7 days ending two days earlier', 'two days before')
    pv = raw['pv_live']
    pvs = pd.Series(pv['generation_mw'].to_numpy(), index=pv['datetime_gmt'] - 0.5 * H).sort_index()
    pv_pub = pd.Series(pvs.index + H, index=pvs.index)
    F.add('pv_d1_energy_gwh', before_cutoff(pvs, pv_pub).sum() * 0.5 / 1000, 'pv_live', 'Solar energy yesterday until the cutoff, GWh')
    F.add('pv_d1_peak_mw', before_cutoff(pvs, pv_pub).max(), 'pv_live', 'Solar peak yesterday')
    pv_daily = pvs.resample('D').sum() * 0.5 / 1000
    F.add('pv_lag2_energy_gwh', lag(pv_daily, 2), 'pv_live', 'Solar energy two days earlier, GWh', 'two days before')
    F.add('pv_roll7_energy_gwh', rolling(pv_daily, 7), 'pv_live', 'Mean daily solar energy over the 7 days ending two days earlier', 'two days before')
    F.add('pv_clearness', (F.frame['pv_d1_energy_gwh'] / before_cutoff(pvs, pv_pub).sum().rolling(30, min_periods=10).max().mul(0.5 / 1000).reindex(index)).clip(0, 1),
          'pv_live', 'Yesterday\'s solar energy ÷ the best of the last 30 days: how clear the sky was')

    # NESO: embedded generation, which FUELHH cannot see
    ne = raw['neso_demand']
    ne_daily = ne.groupby('settlement_date')[['embedded_wind_generation_mw', 'embedded_solar_generation_mw', 'embedded_wind_capacity_mw',
                                              'embedded_solar_capacity_mw']].mean()
    F.add('emb_wind_lag2_mw', lag(ne_daily['embedded_wind_generation_mw'], 3), 'neso_demand', 'Embedded wind three days earlier, daily mean MW (NESO publishes a day late)', 'three days before')
    F.add('emb_solar_lag2_mw', lag(ne_daily['embedded_solar_generation_mw'], 3), 'neso_demand', 'Embedded solar three days earlier, daily mean MW', 'three days before')
    solar_capacity = lag(ne_daily['embedded_solar_capacity_mw'], 3).ffill()
    F.add('emb_solar_capacity_mw', solar_capacity, 'neso_demand', 'Embedded solar capacity, latest known (carried forward)', 'three days before')
    F.add('emb_wind_capacity_mw', lag(ne_daily['embedded_wind_capacity_mw'], 3).ffill(), 'neso_demand', 'Embedded wind capacity, latest known (carried forward)', 'three days before')

    # Day-ahead weather for the target day
    wx = raw['weather_day_ahead'].copy()
    wx['day'] = uk_date(wx['time'])
    wx['power'] = power_curve(wx['wind_speed_100m_kmh'])
    note = 'forecast issued about 24 hours before each hour'
    by_place = wx.groupby(['day', 'location']).agg(wind=('wind_speed_100m_kmh', 'mean'), power=('power', 'mean'), radiation=('shortwave_radiation_wm2', 'mean'),
                                                   temp=('temperature_2m_c', 'mean'))
    power_by_place = by_place['power'].unstack()
    for loc in LOCATIONS:
        F.add(f'wx_power_{loc}', power_by_place.get(loc), 'weather_day_ahead', f'Turbine power curve applied to hourly wind at 100 m, {loc}: mean share of rated output', note)
    F.add('wx_power_mean', by_place['power'].groupby(level=0).mean(), 'weather_day_ahead', 'The same, mean of the five points', note)
    F.add('wx_power_offshore', by_place['power'][by_place.index.get_level_values(1).isin(OFFSHORE)].groupby(level=0).mean(), 'weather_day_ahead',
          'The same, mean of the three sea points', note)
    F.add('wx_power_min_location', by_place['power'].groupby(level=0).min(), 'weather_day_ahead', 'The same, calmest point: low when it is still everywhere', note)
    F.add('wx_wind_spread_locations', by_place['wind'].groupby(level=0).std(), 'weather_day_ahead', 'Std dev of mean wind across the five points', note)
    g = wx.groupby('day')
    F.add('wx_wind_100m_mean_kmh', g['wind_speed_100m_kmh'].mean(), 'weather_day_ahead', 'Wind at 100 m, mean of points and hours (the gold feature)', note)
    F.add('wx_wind_100m_max_kmh', g['wind_speed_100m_kmh'].max(), 'weather_day_ahead', 'Wind at 100 m, highest hour at any point', note)
    hourly_power = wx.groupby(['day', 'time'])['power'].mean()
    F.add('wx_power_min_hour', hourly_power.groupby(level=0).min(), 'weather_day_ahead', 'Five-point power in the calmest hour', note)
    F.add('wx_power_std_hours', hourly_power.groupby(level=0).std(), 'weather_day_ahead', 'Std dev of five-point power across the hours', note)
    hp = hourly_power.reset_index()
    F.add('wx_power_ramp', hp[hp['time'].dt.hour >= 18].groupby('day')['power'].mean() - hp[hp['time'].dt.hour < 6].groupby('day')['power'].mean(), 'weather_day_ahead',
          'Five-point power in the last six hours minus the first six', note)
    F.add('wx_wind_10m_mean_kmh', g['wind_speed_10m_kmh'].mean(), 'weather_day_ahead', 'Wind at 10 m, mean', note)
    F.add('wx_radiation_mean_wm2', g['shortwave_radiation_wm2'].mean(), 'weather_day_ahead', 'Shortwave radiation, mean of points and hours', note)
    F.add('wx_temperature_mean_c', g['temperature_2m_c'].mean(), 'weather_day_ahead', 'Temperature, mean', note)
    F.add('wx_temperature_min_c', g['temperature_2m_c'].min(), 'weather_day_ahead', 'Temperature, coldest hour at any point', note)
    heating = (15.5 - by_place['temp']).clip(lower=0).groupby(level=0).mean()
    F.add('wx_heating_degrees', heating, 'weather_day_ahead', 'Degrees below 15.5 °C, mean of the points: the demand driver', note)
    F.add('wx_cloud_cover_mean_pct', g['cloud_cover_pct'].mean(), 'weather_day_ahead', 'Cloud cover, mean', note)
    F.add('wx_hours', g['time'].nunique(), 'weather_day_ahead', 'Hours of the target day present', note)
    F.add('emb_solar_expected_mw', solar_capacity * F.frame['wx_radiation_mean_wm2'] / 1000, 'neso_demand',
          'Embedded solar capacity × forecast radiation ÷ 1000: expected embedded solar, which lowers demand on the transmission system')
    F.add('wx_power_x_calibration', F.frame['wx_power_mean'] * fc_cal.reindex(index), 'weather_day_ahead', 'Five-point power × the day-ahead forecast\'s 7-day calibration')

    # Short-range weather: only what had already happened
    wh = raw['weather_historical'].copy()
    wh['power'] = power_curve(wh['wind_speed_100m_kmh'])
    whh = wh.groupby('time')[['power', 'wind_speed_100m_kmh', 'temperature_2m_c']].mean()
    wh_pub = pd.Series(whh.index + H, index=whh.index)
    F.add('wxh_power_d1_until_cutoff', before_cutoff(whh['power'], wh_pub).mean(), 'weather_historical', 'Five-point power over yesterday until the cutoff, from short-range forecasts')
    F.add('wxh_power_last3h', before_cutoff(whh['power'], wh_pub, first_hour=14, last_hour=17).mean(), 'weather_historical', 'Five-point power over the last three available hours')
    F.add('wxh_power_lag2', lag(whh['power'].resample('D').mean(), 2), 'weather_historical', 'Five-point power two days earlier', 'two days before')
    da_hourly = raw['weather_day_ahead'].groupby('time')['wind_speed_100m_kmh'].mean()
    wx_err = (da_hourly - whh['wind_speed_100m_kmh'].reindex(da_hourly.index)).dropna()
    F.add('wxh_day_ahead_error_d1', before_cutoff(wx_err, pd.Series(wx_err.index + H, index=wx_err.index)).mean(), 'weather_historical',
          'Day-ahead wind minus short-range wind over yesterday until the cutoff: was the forecast running high or low')

    # REMIT: capacity out during the target day, as the notices stood at the cutoff
    remit = raw['remit_outages'].sort_values('publish_time')
    remit = remit[(remit['event_type'] == 'Production unavailability') & remit['event_start_time'].notna()]
    r_pub, r_start = remit['publish_time'].to_numpy(), remit['event_start_time'].to_numpy()
    r_end = remit['event_end_time'].fillna(pd.Timestamp('2100-01-01')).to_numpy()
    fuel_type = remit['fuel_type'].fillna('')
    group = np.where(fuel_type == 'Fossil Gas', 'gas', np.where(fuel_type == 'Nuclear', 'nuclear', np.where(fuel_type.str.startswith('Wind'), 'wind',
             np.where(fuel_type == 'Biomass', 'biomass', 'other'))))
    cap = remit[['unavailable_capacity_mw', 'normal_capacity_mw']].min(axis=1).to_numpy()
    mrid, revision = pd.factorize(remit['mrid'])[0], remit['revision_number'].to_numpy()
    status, planned = remit['event_status'].to_numpy(), (remit['unavailability_type'] == 'Planned').to_numpy()
    rows = {}
    for day in (index if remit_days is None else remit_days):
        cutoff = np.datetime64(day - (24 - CUTOFF_HOUR) * H)
        day_start = np.datetime64(pd.Timestamp(day).tz_localize(UK).tz_convert('UTC').tz_localize(None))
        day_end = np.datetime64((pd.Timestamp(day) + DAY).tz_localize(UK).tz_convert('UTC').tz_localize(None))
        n = np.searchsorted(r_pub, cutoff, side='right')
        about = np.unique(mrid[:n][(r_start[:n] < day_end) & (r_end[:n] > day_start)])
        if not len(about):
            rows[day] = {} if n == 0 else {f'out_{k}_mw': 0.0 for k in ('gas', 'nuclear', 'wind', 'biomass', 'other', 'stale')} | {'out_notices': 0.0, 'out_unplanned_mw': 0.0}
            continue
        sel = np.flatnonzero(np.isin(mrid[:n], about))
        latest = pd.DataFrame({'mrid': mrid[sel], 'rev': revision[sel], 'i': sel}).sort_values(['rev', 'i']).groupby('mrid')['i'].last().to_numpy()
        s, e = np.maximum(r_start[latest], day_start), np.minimum(r_end[latest], day_end)
        hours = np.clip((e - s) / np.timedelta64(1, 'h'), 0, None) / ((day_end - day_start) / np.timedelta64(1, 'h'))
        live = (status[latest] == 'Active') & (hours > 0) & ~np.isnan(cap[latest])
        stale = (r_end[latest] - r_start[latest]) > np.timedelta64(365, 'D')
        mw = cap[latest] * hours
        out = {f'out_{k}_mw': float(mw[live & ~stale & (group[latest] == k)].sum()) for k in ('gas', 'nuclear', 'wind', 'biomass', 'other')}
        out['out_stale_mw'] = float(mw[live & stale].sum())
        out['out_notices'] = float((live & ~stale).sum())
        out['out_unplanned_mw'] = float(mw[live & ~stale & ~planned[latest]].sum())
        rows[day] = out
    out = pd.DataFrame.from_dict(rows, orient='index')
    weighted = 'MW unavailable during the target day, weighted by the hours it is out, from the latest revision of each notice before the cutoff; notices longer than a year excluded'
    for key, label in (('gas', 'Gas'), ('nuclear', 'Nuclear'), ('wind', 'Wind'), ('biomass', 'Biomass'), ('other', 'Other')):
        F.add(f'out_{key}_mw', out.get(f'out_{key}_mw'), 'remit_outages', f'{label}: {weighted}' if key == 'gas' else f'{label}, same measure')
    F.add('out_stale_mw', out.get('out_stale_mw'), 'remit_outages', 'MW held by notices running longer than a year: closed stations, kept apart')
    F.add('out_notices', out.get('out_notices'), 'remit_outages', 'Active notices touching the target day')
    F.add('out_unplanned_mw', out.get('out_unplanned_mw'), 'remit_outages', 'Unplanned MW out during the target day')
    F.add('out_low_carbon_mw', F.frame[['out_nuclear_mw', 'out_wind_mw', 'out_biomass_mw']].sum(axis=1, min_count=3), 'remit_outages',
          'Nuclear + wind + biomass out: capacity the target\'s numerator and base lose')
    return F.frame.assign(target=share.reindex(index)), F.meta


# --- Reading silver, writing the store --------------------------------------------------------------------------

# The silver columns build() reads, per table: the job selects only these before handing them to pandas
SILVER_COLUMNS = {
    'fuelhh': ('settlement_date', 'settlement_period', 'fuel_type', 'generation_mw', 'publish_time'),
    'demand_outturn': ('settlement_date', 'settlement_period', 'start_time', 'publish_time', 'initial_demand_outturn_mw'),
    'wind_forecast': ('publish_time', 'start_time', 'generation_mw'),
    'demand_forecast': ('publish_time', 'settlement_date', 'settlement_period', 'start_time', 'demand_mw'),
    'wind_solar_forecast': ('process_type', 'psr_type', 'publish_time', 'settlement_date', 'settlement_period', 'quantity_mw'),
    'wind_solar_actual': ('psr_type', 'settlement_date', 'settlement_period', 'publish_time', 'quantity_mw'),
    'system_prices': ('settlement_date', 'start_time', 'created_at', 'system_sell_price_gbp_mwh', 'net_imbalance_volume_mwh'),
    'market_index': ('data_provider', 'settlement_date', 'start_time', 'price_gbp_mwh', 'volume_mwh'),
    'remit_outages': ('mrid', 'revision_number', 'publish_time', 'event_type', 'event_status', 'unavailability_type', 'fuel_type',
                      'event_start_time', 'event_end_time', 'unavailable_capacity_mw', 'normal_capacity_mw'),
    'bank_holidays': ('division', 'holiday_date'),
    'carbon_intensity': ('period_start', 'period_end', 'actual_gco2_kwh'),
    'pv_live': ('datetime_gmt', 'generation_mw'),
    'neso_demand': ('settlement_date', 'embedded_wind_generation_mw', 'embedded_solar_generation_mw', 'embedded_wind_capacity_mw',
                    'embedded_solar_capacity_mw'),
    'weather_day_ahead': ('location', 'time', 'wind_speed_100m_kmh', 'wind_speed_10m_kmh', 'shortwave_radiation_wm2', 'temperature_2m_c',
                          'cloud_cover_pct'),
    'weather_historical': ('location', 'time', 'wind_speed_100m_kmh', 'temperature_2m_c'),
}
assert set(SILVER_COLUMNS) == set(TABLES)


def prepare(raw):
    """Silver tables as pandas DataFrames -> the types build() expects: *_date columns as midnight timestamps,
    every timestamp naive UTC (Spark hands dates over as datetime.date and may attach a time zone)."""
    out = {}
    for name, df in raw.items():
        df = df.copy()
        for column in df.columns:
            if column.endswith('_date'):
                # pandas 2+ may give date objects second resolution; keep one resolution throughout
                df[column] = pd.to_datetime(df[column]).astype('datetime64[ns]')
            elif isinstance(df[column].dtype, pd.DatetimeTZDtype):
                df[column] = df[column].dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]')
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                df[column] = df[column].astype('datetime64[ns]')
        out[name] = df
    return out


def feature_hash(frame, columns, digits=6):
    """A fingerprint per row of the rounded feature values. Rounding keeps last-bit float noise from looking like a
    change, so a rerun with the same inputs rewrites nothing."""
    values = frame[columns].astype(float).round(digits).to_numpy()
    return [hashlib.sha256(json.dumps([None if np.isnan(v) else float(v) for v in row]).encode()).hexdigest()[:16]
            for row in values]


def catalogue(meta):
    """The feature catalogue: one row per feature, with where it comes from and when it is known."""
    rows = [{'feature': m['feature'], 'source_table': m['source'], 'definition': m['definition'], 'known': m['known'],
             'available_from': AVAILABILITY_NOTE.get(m['source'], 'known in advance: the calendar'),
             'data_type': 'DOUBLE', 'feature_set_version': FEATURE_SET_VERSION} for m in meta]
    return pd.DataFrame(rows)
