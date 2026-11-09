"""Gold: business-ready tables built from silver, as plain DataFrame functions.

- dim_date: one row per UK date, with calendar facts (bank holidays, clock changes).
- grid_half_hourly: one row per settlement period, with generation by fuel and category, demand, prices,
  carbon intensity and embedded solar and wind side by side.
- generation_daily: one row per date: energy by category (MWh) and the renewable share, the model's target.
- weather_daily: one row per kind (day_ahead / historical), location and UK date.
- outages_day_ahead: one row per target date: capacity unavailable, as known at the forecast cutoff.
- features_day_ahead: one row per target date: everything known at the cutoff on the day before, plus the
  target. Built so a model trained on it can't see the future (see FORECAST_CUTOFF_HOUR_UTC).

Shares are energy-weighted (total renewable MWh / total MWh), not an average of half-hourly shares.
"""
from pyspark.sql import Window, functions as F

from pulse.quality import period_start_utc, periods_in_day
from pulse.settings import FORECAST_CUTOFF_HOUR_UTC, FUEL_CATEGORIES, KNOWN_FUEL_TYPES, UK_TZ, WEATHER_POINTS

PERIOD_KEYS = ['settlement_date', 'settlement_period']
# For events with no end time yet
OPEN_ENDED = "TIMESTAMP'9999-12-31 00:00:00'"


def divide(numerator_sql, denominator_sql):
    """NULL instead of an error when the denominator is 0 (ANSI SQL raises on division by zero)."""
    return F.expr(f'try_divide({numerator_sql}, {denominator_sql})')


def uk_date(ts_column):
    """The UK calendar date of a UTC instant."""
    return F.to_date(F.from_utc_timestamp(ts_column, UK_TZ))


def cutoff(date_column):
    """When a day-ahead prediction for this date is made: FORECAST_CUTOFF_HOUR_UTC on the day before."""
    return F.expr(f'timestampadd(HOUR, {FORECAST_CUTOFF_HOUR_UTC}, CAST(date_sub({date_column}, 1) AS TIMESTAMP))')


def latest(df, partition, order):
    """The first row per partition by `order` (Column expressions, newest first)."""
    window = Window.partitionBy(*partition).orderBy(*order)
    return df.withColumn('_rank', F.row_number().over(window)).where('_rank = 1').drop('_rank')


# --- Calendar -----------------------------------------------------------------------------------------------

def dim_date(fuelhh, bank_holidays):
    """Every date FUELHH covers, plus two days ahead so tomorrow's features have a calendar row."""
    dates = (fuelhh.agg(F.min('settlement_date').alias('first'), F.max('settlement_date').alias('last'))
             .select(F.explode(F.sequence('first', F.date_add('last', 2))).alias('date')))
    divisions = {'england-and-wales': 'england_wales', 'scotland': 'scotland', 'northern-ireland': 'northern_ireland'}
    holidays = bank_holidays.groupBy(F.col('holiday_date').alias('date')).agg(
        *[F.max(F.col('division') == d).alias(f'is_bank_holiday_{name}') for d, name in divisions.items()],
        F.max(F.when(F.col('division') == 'england-and-wales', F.col('title'))).alias('bank_holiday_name'))
    flags = [f'is_bank_holiday_{name}' for name in divisions.values()]
    return (dates.join(holidays, 'date', 'left')
            .select('date',
                    F.year('date').alias('year'), F.month('date').alias('month'),
                    F.dayofmonth('date').alias('day_of_month'), F.dayofyear('date').alias('day_of_year'),
                    F.weekofyear('date').alias('week_of_year'),
                    # ISO weekday: 1 = Monday ... 7 = Sunday
                    (F.expr('weekday(date)') + 1).alias('day_of_week'),
                    (F.expr('weekday(date)') >= 5).alias('is_weekend'),
                    F.expr(periods_in_day('date')).alias('periods_in_day'),
                    (F.expr(periods_in_day('date')) != 48).alias('is_clock_change'),
                    *[F.coalesce(F.col(flag), F.lit(False)).alias(flag) for flag in flags],
                    'bank_holiday_name'))


# --- Half-hourly grid ---------------------------------------------------------------------------------------

def generation_by_period(fuelhh):
    """FUELHH (one row per fuel) -> one row per period with a column per fuel and per category."""
    generation, fuel = F.col('generation_mw'), F.col('fuel_type')
    interconnector = fuel.startswith('INT')
    return (fuelhh.groupBy(*PERIOD_KEYS).agg(
        # GB generation excludes interconnectors, which are flows to and from other countries
        F.sum(F.when(~interconnector, generation)).alias('domestic_generation_mw'),
        F.sum(F.when(interconnector, generation)).alias('net_imports_mw'),
        *[F.sum(F.when(fuel.isin(*fuels), generation)).alias(f'{category}_mw')
          for category, fuels in FUEL_CATEGORIES.items()],
        # One column per fuel type, prefixed so they can't clash with the categories (nuclear_mw, other_mw)
        *[F.sum(F.when(fuel == f, generation)).alias(f'fuel_{f.lower()}_mw') for f in KNOWN_FUEL_TYPES],
        # A fuel type gold doesn't know yet still counts in domestic_generation_mw; this shows how much
        F.sum(F.when(~fuel.isin(*KNOWN_FUEL_TYPES) & ~interconnector, generation)).alias('unlisted_fuels_mw'),
        F.count('*').alias('fuel_types_reported'))
        .withColumn('start_time', F.expr(period_start_utc('settlement_date', 'settlement_period')))
        .withColumn('renewable_share', divide('renewable_mw', 'domestic_generation_mw'))
        .withColumn('low_carbon_share',
                    divide('coalesce(renewable_mw, 0) + coalesce(nuclear_mw, 0)', 'domestic_generation_mw')))


def grid_half_hourly(fuelhh, demand_outturn, neso_demand, system_prices, market_index, carbon_intensity, pv_live):
    """One row per settlement period FUELHH covers, with every half-hourly source joined on."""
    demand = demand_outturn.select(*PERIOD_KEYS, 'initial_demand_outturn_mw', 'initial_transmission_demand_outturn_mw')
    neso = neso_demand.select(*PERIOD_KEYS,
                              F.col('nd_mw').alias('neso_national_demand_mw'),
                              F.col('tsd_mw').alias('neso_transmission_demand_mw'),
                              'embedded_wind_generation_mw', 'embedded_solar_generation_mw',
                              'embedded_wind_capacity_mw', 'embedded_solar_capacity_mw')
    prices = system_prices.select(*PERIOD_KEYS, F.col('system_sell_price_gbp_mwh').alias('system_price_gbp_mwh'),
                                  'net_imbalance_volume_mwh')
    # N2EX reports mostly zeros, so the market index price comes from APX
    mid = (market_index.where("data_provider = 'APXMIDP'")
           .select(*PERIOD_KEYS, F.col('price_gbp_mwh').alias('market_index_price_gbp_mwh'),
                   F.col('volume_mwh').alias('market_index_volume_mwh')))
    carbon = carbon_intensity.select(F.col('period_start').alias('start_time'),
                                     F.col('forecast_gco2_kwh').alias('carbon_intensity_forecast_gco2_kwh'),
                                     F.col('actual_gco2_kwh').alias('carbon_intensity_actual_gco2_kwh'))
    # PV_Live stamps each half-hour with its end; GSP 0 is the national total
    pv = (pv_live.where('gsp_id = 0')
          .select(F.expr('datetime_gmt - INTERVAL 30 MINUTES').alias('start_time'),
                  F.col('generation_mw').alias('pv_live_solar_mw')))
    embedded = 'embedded_wind_generation_mw + embedded_solar_generation_mw'
    return (generation_by_period(fuelhh)
            .join(demand, PERIOD_KEYS, 'left').join(neso, PERIOD_KEYS, 'left')
            .join(prices, PERIOD_KEYS, 'left').join(mid, PERIOD_KEYS, 'left')
            .join(carbon, 'start_time', 'left').join(pv, 'start_time', 'left')
            # Embedded wind and solar are invisible to FUELHH (they reduce demand instead); NESO estimates them
            .withColumn('renewable_share_incl_embedded',
                        divide(f'renewable_mw + {embedded}', f'domestic_generation_mw + {embedded}')))


# --- Daily generation: the target ---------------------------------------------------------------------------

def generation_daily(grid):
    energy = ['domestic_generation', 'net_imports', *FUEL_CATEGORIES,
              'embedded_wind_generation', 'embedded_solar_generation']
    return (grid.groupBy('settlement_date').agg(
        F.count('*').alias('periods_present'),
        # Each period is half an hour: MWh = MW x 0.5
        *[(F.sum(f'{name}_mw') * 0.5).alias(f'{name}_mwh') for name in energy],
        F.count('embedded_solar_generation_mw').alias('periods_with_embedded'))
        .withColumn('periods_expected', F.expr(periods_in_day('settlement_date')))
        .withColumn('is_complete', F.col('periods_present') == F.col('periods_expected'))
        .withColumn('renewable_share', divide('renewable_mwh', 'domestic_generation_mwh'))
        .withColumn('low_carbon_share', divide('coalesce(renewable_mwh, 0) + coalesce(nuclear_mwh, 0)',
                                               'domestic_generation_mwh'))
        .withColumn('renewable_share_incl_embedded', F.when(
            F.col('periods_with_embedded') == F.col('periods_expected'),
            divide('renewable_mwh + embedded_wind_generation_mwh + embedded_solar_generation_mwh',
                   'domestic_generation_mwh + embedded_wind_generation_mwh + embedded_solar_generation_mwh'))))


# --- Weather --------------------------------------------------------------------------------------------------

def _weather_daily(weather, kind):
    return (weather.withColumn('date', uk_date('time'))
            .groupBy('date', 'location').agg(
                F.count('*').alias('hours_present'),
                F.avg('wind_speed_100m_kmh').alias('wind_speed_100m_mean_kmh'),
                F.max('wind_speed_100m_kmh').alias('wind_speed_100m_max_kmh'),
                F.avg('wind_speed_10m_kmh').alias('wind_speed_10m_mean_kmh'),
                F.avg('shortwave_radiation_wm2').alias('shortwave_radiation_mean_wm2'),
                F.avg('temperature_2m_c').alias('temperature_mean_c'),
                F.min('temperature_2m_c').alias('temperature_min_c'),
                F.max('temperature_2m_c').alias('temperature_max_c'),
                F.avg('cloud_cover_pct').alias('cloud_cover_mean_pct'))
            .withColumn('kind', F.lit(kind)))


def weather_daily(weather_day_ahead, weather_historical):
    return _weather_daily(weather_day_ahead, 'day_ahead').unionByName(_weather_daily(weather_historical, 'historical'))


# --- Outages known before the cutoff ------------------------------------------------------------------------

def _targets(dates):
    return dates.select(F.col('date').alias('target_date')).select(
        'target_date', cutoff('target_date').alias('cutoff'),
        F.expr(f"to_utc_timestamp(target_date, '{UK_TZ}')").alias('day_start'),
        F.expr(f"to_utc_timestamp(date_add(target_date, 1), '{UK_TZ}')").alias('day_end'))


def outages_day_ahead(remit_outages, dates):
    """Per target date: MW unavailable during that day, by fuel group, as REMIT notices said at the cutoff.

    For each date and message, only revisions published before the cutoff count, and the latest of those wins.
    A later revision that moved or cancelled an outage is therefore not seen early.
    """
    targets = _targets(dates)
    notices = remit_outages.select(
        'mrid', 'revision_number', 'publish_time', 'event_status', 'fuel_type', 'event_start_time',
        F.coalesce('event_end_time', F.expr(OPEN_ENDED)).alias('event_end_time'),
        # A fully unavailable unit can report more than its normal capacity (412 of 390 MW); cap it
        F.least('unavailable_capacity_mw', 'normal_capacity_mw').alias('unavailable_mw'))
    overlaps = (F.col('event_start_time') < F.col('day_end')) & (F.col('event_end_time') > F.col('day_start'))
    # Messages with any revision about the day, known before the cutoff...
    candidates = (F.broadcast(targets).join(notices, (F.col('publish_time') < F.col('cutoff')) & overlaps)
                  .select('target_date', 'mrid').distinct())
    # ...then each message's latest revision before the cutoff, which must still be active and overlapping
    known = latest(candidates.join(F.broadcast(targets), 'target_date').join(notices, 'mrid')
                   .where(F.col('publish_time') < F.col('cutoff')),
                   ['target_date', 'mrid'], [F.desc('revision_number'), F.desc('publish_time')])
    active = known.where(overlaps & (F.col('event_status') == 'Active'))
    group = (F.when(F.col('fuel_type') == 'Fossil Gas', 'gas').when(F.col('fuel_type') == 'Nuclear', 'nuclear')
             .when(F.col('fuel_type').startswith('Wind'), 'wind').otherwise('other'))
    totals = active.groupBy('target_date').agg(
        *[F.sum(F.when(group == g, F.col('unavailable_mw'))).alias(f'unavailable_{g}_mw')
          for g in ('gas', 'nuclear', 'wind', 'other')],
        F.countDistinct('mrid').alias('active_outage_notices'))
    # Zero where REMIT data exists but nothing was out; NULL before the REMIT history starts
    first_known = remit_outages.agg(F.min(uk_date('publish_time')).alias('first_known'))
    covered = F.col('target_date') > F.col('first_known')
    columns = [c for c in totals.columns if c != 'target_date']
    return (targets.select('target_date').join(totals, 'target_date', 'left').crossJoin(first_known)
            .select('target_date', *[F.when(covered, F.coalesce(c, F.lit(0))).alias(c) for c in columns]))


# --- Features for a day-ahead model -------------------------------------------------------------------------

def _wind_solar_day_ahead(wind_solar_forecast, targets):
    day_ahead = (wind_solar_forecast.where("process_type = 'Day ahead'")
                 .join(targets, F.col('settlement_date') == F.col('target_date'))
                 .where(F.col('publish_time') < F.col('cutoff')))
    newest = latest(day_ahead, ['target_date', 'psr_type', 'settlement_period'],
                    [F.desc('publish_time'), F.desc('document_revision_number')])
    by_type = {'Solar': 'fc_solar_mean_mw', 'Wind Offshore': 'fc_wind_offshore_mean_mw',
               'Wind Onshore': 'fc_wind_onshore_mean_mw'}
    return (newest.groupBy('target_date').agg(
        *[F.avg(F.when(F.col('psr_type') == psr, F.col('quantity_mw'))).alias(name) for psr, name in by_type.items()],
        F.countDistinct('settlement_period').alias('fc_wind_solar_periods'))
        .withColumn('fc_wind_solar_mean_mw', F.expr(' + '.join(by_type.values()))))


def _windfor(wind_forecast, targets):
    forecasts = (wind_forecast.withColumn('target_date', uk_date('start_time')).join(targets, 'target_date')
                 .where(F.col('publish_time') < F.col('cutoff')))
    return (latest(forecasts, ['target_date', 'start_time'], [F.desc('publish_time')])
            .groupBy('target_date').agg(F.avg('generation_mw').alias('windfor_wind_mean_mw'),
                                        F.max('generation_mw').alias('windfor_wind_max_mw')))


def _demand_forecast(demand_forecast, targets):
    forecasts = (demand_forecast.where("boundary = 'N'")
                 .join(targets, F.col('settlement_date') == F.col('target_date'))
                 .where(F.col('publish_time') < F.col('cutoff')))
    return (latest(forecasts, ['target_date', 'settlement_period'], [F.desc('publish_time')])
            .groupBy('target_date').agg(F.avg('demand_mw').alias('fc_demand_mean_mw'),
                                        F.max('demand_mw').alias('fc_demand_peak_mw'),
                                        F.countDistinct('settlement_period').alias('fc_demand_periods')))


def _weather_features(weather_daily_df):
    day_ahead = weather_daily_df.where("kind = 'day_ahead'").withColumnRenamed('date', 'target_date')
    per_location = (day_ahead.groupBy('target_date')
                    .pivot('location', [name for name, _, _ in WEATHER_POINTS])
                    .agg(F.first('wind_speed_100m_mean_kmh').alias('wx_wind_100m_kmh'),
                         F.first('shortwave_radiation_mean_wm2').alias('wx_radiation_wm2')))
    national = day_ahead.groupBy('target_date').agg(
        F.avg('wind_speed_100m_mean_kmh').alias('wx_wind_100m_mean_kmh'),
        F.avg('temperature_mean_c').alias('wx_temperature_mean_c'),
        F.avg('cloud_cover_mean_pct').alias('wx_cloud_cover_mean_pct'),
        F.avg('shortwave_radiation_mean_wm2').alias('wx_radiation_mean_wm2'))
    return national.join(per_location, 'target_date', 'left')


def _recent_shares(grid, generation, targets):
    """What the renewable share looked like before the cutoff: the day before (so far), 2 and 7 days back."""
    partial = (grid.join(targets, F.col('settlement_date') == F.date_sub('target_date', 1))
               # A period's figures exist once it has ended
               .where(F.expr('start_time + INTERVAL 30 MINUTES') <= F.col('cutoff'))
               .groupBy('target_date').agg(
                   divide('sum(renewable_mw)', 'sum(domestic_generation_mw)').alias('renewable_share_d1_until_cutoff'),
                   F.count('*').alias('d1_periods_until_cutoff')))
    daily = generation.select('settlement_date', 'renewable_share', 'is_complete')

    def lag(days):
        return daily.select(F.date_add('settlement_date', days).alias('target_date'),
                            F.when(F.col('is_complete'), F.col('renewable_share')).alias(f'renewable_share_lag_{days}d'))

    return partial.join(lag(2), 'target_date', 'full').join(lag(7), 'target_date', 'full')


def features_day_ahead(dates, generation, grid, wind_solar_forecast, wind_forecast, demand_forecast,
                       weather_daily_df, outages):
    """One row per target date D: features known at the cutoff on D-1, and D's renewable share (the target).

    Leakage rules: forecasts only if published before the cutoff (the latest of those); actuals only from
    periods that ended before it; REMIT as known at it. Weather is Open-Meteo's previous-day forecast: for the
    last hours of D it was issued a few hours after the cutoff, a small approximation noted in the README.
    """
    targets = _targets(dates).select('target_date', 'cutoff')
    calendar = dates.select(F.col('date').alias('target_date'), 'day_of_week', 'is_weekend', 'month', 'day_of_year',
                            'is_bank_holiday_england_wales', 'is_clock_change', 'periods_in_day')
    target = generation.select(F.col('settlement_date').alias('target_date'),
                               F.col('renewable_share').alias('target_renewable_share'),
                               F.col('renewable_share_incl_embedded').alias('target_renewable_share_incl_embedded'),
                               F.col('is_complete').alias('target_is_complete'))
    features = (targets.join(calendar, 'target_date', 'left')
                .join(_wind_solar_day_ahead(wind_solar_forecast, targets), 'target_date', 'left')
                .join(_windfor(wind_forecast, targets), 'target_date', 'left')
                .join(_demand_forecast(demand_forecast, targets), 'target_date', 'left')
                .join(_weather_features(weather_daily_df), 'target_date', 'left')
                .join(outages, 'target_date', 'left')
                .join(_recent_shares(grid, generation, targets), 'target_date', 'left')
                .join(target, 'target_date', 'left')
                # How much of the forecast demand forecast wind and solar could cover
                .withColumn('fc_wind_solar_share_of_demand', divide('fc_wind_solar_mean_mw', 'fc_demand_mean_mw')))
    required = ['fc_wind_solar_mean_mw', 'fc_demand_mean_mw', 'wx_wind_100m_mean_kmh', 'renewable_share_lag_2d']
    trainable = F.coalesce(F.col('target_is_complete'), F.lit(False))
    for column in required:
        trainable = trainable & F.col(column).isNotNull()
    return features.withColumn('is_trainable', trainable)
