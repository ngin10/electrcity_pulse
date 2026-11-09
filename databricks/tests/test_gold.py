"""Gold logic on small hand-made silver tables: shares, completeness, and no leakage past the cutoff."""
import pytest

from bronze_rows import as_strings, rows_frame
from pulse import gold

FUELHH_SCHEMA = 'settlement_date DATE, settlement_period INT, fuel_type STRING, generation_mw DOUBLE'


def fuelhh(spark, *rows):
    return rows_frame(spark, rows, FUELHH_SCHEMA)


@pytest.fixture
def two_periods(spark):
    # Period 1: WIND 100, CCGT 300, an export of 50; period 2: WIND 300, CCGT 100
    return fuelhh(spark,
                  ('2026-11-01', 1, 'WIND', 100.0), ('2026-11-01', 1, 'CCGT', 300.0), ('2026-11-01', 1, 'INTNED', -50.0),
                  ('2026-11-01', 2, 'WIND', 300.0), ('2026-11-01', 2, 'CCGT', 100.0))


def test_interconnectors_are_flows_not_generation(two_periods):
    [period1] = gold.generation_by_period(two_periods).where('settlement_period = 1').collect()
    assert period1.domestic_generation_mw == 400.0
    assert period1.net_imports_mw == -50.0
    assert period1.renewable_share == 0.25
    assert period1.fuel_types_reported == 3


def test_daily_share_is_energy_weighted_and_completeness_is_checked(spark, two_periods):
    grid = gold.generation_by_period(two_periods).withColumns(
        {'embedded_wind_generation_mw': gold.F.lit(None).cast('DOUBLE'),
         'embedded_solar_generation_mw': gold.F.lit(None).cast('DOUBLE')})
    [day] = gold.generation_daily(grid).collect()
    # (100 + 300) / (400 + 400) = 0.5; the mean of the two half-hourly shares would be (0.25 + 0.75) / 2 too,
    # so check the energy: each period is half an hour
    assert day.renewable_mwh == 200.0 and day.domestic_generation_mwh == 400.0
    assert day.renewable_share == 0.5
    assert (day.periods_present, day.periods_expected, day.is_complete) == (2, 48, False)
    assert day.renewable_share_incl_embedded is None


def test_calendar_knows_clock_changes_and_bank_holidays(spark):
    bounds = fuelhh(spark, ('2026-10-24', 1, 'WIND', 1.0), ('2026-12-25', 1, 'WIND', 1.0))
    holidays = rows_frame(spark, [('england-and-wales', '2026-12-25', 'Christmas Day')],
                          'division STRING, holiday_date DATE, title STRING')
    dates = gold.dim_date(bounds, holidays)
    rows = {r[0]: r[1:] for r in as_strings(dates, 'date', 'periods_in_day', 'is_clock_change',
                                             'is_bank_holiday_england_wales', 'is_bank_holiday_scotland')}
    assert rows['2026-10-25'] == ('50', 'true', 'false', 'false')
    assert rows['2026-12-25'] == ('48', 'false', 'true', 'false')
    assert max(rows) == '2026-12-27'   # two days past the last FUELHH date, for tomorrow's features


DGWS_SCHEMA = ('process_type STRING, psr_type STRING, settlement_date DATE, settlement_period INT, '
               'publish_time TIMESTAMP, document_revision_number INT, quantity_mw DOUBLE')


def test_forecasts_published_after_the_cutoff_are_not_used(spark):
    dates = rows_frame(spark, [('2026-11-02',)], 'date DATE')
    targets = gold._targets(dates).select('target_date', 'cutoff')
    # Cutoff for 2026-11-02 is 2026-11-01 18:00 UTC
    dgws = rows_frame(spark, [
        ('Day ahead', 'Solar', '2026-11-02', 30, '2026-11-01 16:45:00', 1, 100.0),
        ('Day ahead', 'Solar', '2026-11-02', 30, '2026-11-01 19:00:00', 1, 999.0),   # after the cutoff
        ('Intraday process', 'Solar', '2026-11-02', 30, '2026-11-01 17:00:00', 1, 555.0),  # not day-ahead
    ], DGWS_SCHEMA)
    [row] = gold._wind_solar_day_ahead(dgws, targets).collect()
    assert row.fc_solar_mean_mw == 100.0

    ndf = rows_frame(spark, [
        ('N', '2026-11-02', 30, '2026-11-01 17:17:00', 30000.0),
        ('N', '2026-11-02', 30, '2026-11-01 17:47:00', 31000.0),   # the latest before the cutoff wins
        ('N', '2026-11-02', 30, '2026-11-01 18:17:00', 99999.0),
    ], 'boundary STRING, settlement_date DATE, settlement_period INT, publish_time TIMESTAMP, demand_mw DOUBLE')
    [row] = gold._demand_forecast(ndf, targets).collect()
    assert row.fc_demand_mean_mw == 31000.0


REMIT_SCHEMA = ('mrid STRING, revision_number INT, publish_time TIMESTAMP, event_status STRING, fuel_type STRING, '
                'event_start_time TIMESTAMP, event_end_time TIMESTAMP, unavailable_capacity_mw DOUBLE, '
                'normal_capacity_mw DOUBLE')


def test_outages_use_only_revisions_known_at_the_cutoff(spark):
    dates = rows_frame(spark, [('2026-11-01',), ('2026-11-02',)], 'date DATE')
    remit = rows_frame(spark, [
        # A: active for the 2nd when the forecast is made; cancelled only after the cutoff
        ('A', 1, '2026-11-01 09:00:00', 'Active', 'Fossil Gas', '2026-11-02 06:00:00', '2026-11-02 20:00:00', 412.0, 390.0),
        ('A', 2, '2026-11-01 21:00:00', 'Dismissed', 'Fossil Gas', '2026-11-02 06:00:00', '2026-11-02 20:00:00', 412.0, 390.0),
        # B: cancelled before the cutoff, so it must not count
        ('B', 1, '2026-11-01 08:00:00', 'Active', 'Nuclear', '2026-11-02 00:00:00', None, 600.0, 600.0),
        ('B', 2, '2026-11-01 12:00:00', 'Dismissed', 'Nuclear', '2026-11-02 00:00:00', None, 600.0, 600.0),
        # C: published after the cutoff
        ('C', 1, '2026-11-01 19:00:00', 'Active', 'Wind Offshore', '2026-11-02 00:00:00', None, 50.0, 100.0),
    ], REMIT_SCHEMA)
    rows = {r.target_date.isoformat(): r for r in gold.outages_day_ahead(remit, dates).collect()}
    second = rows['2026-11-02']
    # A counts, capped at its normal capacity; B and C don't
    assert (second.unavailable_gas_mw, second.unavailable_nuclear_mw, second.unavailable_wind_mw) == (390.0, 0, 0)
    assert second.active_outage_notices == 1
    # Before REMIT data starts, "no outages" isn't known: NULL, not 0
    assert rows['2026-11-01'].unavailable_gas_mw is None


def test_whole_gold_layer_runs_on_real_responses(spark):
    """Real fixture responses -> silver -> every gold table: catches any column that doesn't line up."""
    from bronze_rows import batch_bronze, envelope, fixture_json, fixture_text, kafka_bronze
    from pulse import sources
    from pulse.quality import check, valid_rows

    def silver(source, fixture):
        return valid_rows(source, check(source, batch_bronze(spark, fixture_text(fixture)))).cache()

    fuel = valid_rows(sources.FUELHH, check(sources.FUELHH, kafka_bronze(
        spark, *[envelope(r) for r in fixture_json('fuelhh_records')])))
    holidays = silver(sources.BANK_HOLIDAYS, 'uk_bank_holidays')
    dates = gold.dim_date(fuel, holidays)
    grid = gold.grid_half_hourly(fuel, silver(sources.DEMAND_OUTTURN, 'elexon_demand_outturn'),
                                 silver(sources.NESO_DEMAND, 'neso_historic_demand'),
                                 silver(sources.SYSTEM_PRICES, 'elexon_system_prices'),
                                 silver(sources.MARKET_INDEX, 'elexon_market_index'),
                                 silver(sources.CARBON_INTENSITY, 'carbon_intensity'),
                                 silver(sources.PV_LIVE, 'pv_live'))
    generation = gold.generation_daily(grid)
    weather = gold.weather_daily(silver(sources.WEATHER_DAY_AHEAD, 'weather_forecast_day_ahead'),
                                 silver(sources.WEATHER_HISTORICAL, 'weather_historical_forecast'))
    outages = gold.outages_day_ahead(silver(sources.REMIT_OUTAGES, 'elexon_remit_outages'), dates)
    features = gold.features_day_ahead(dates, generation, grid,
                                       silver(sources.WIND_SOLAR_FORECAST, 'elexon_wind_solar_forecast'),
                                       silver(sources.WIND_FORECAST, 'elexon_wind_forecast'),
                                       silver(sources.DEMAND_FORECAST, 'elexon_demand_forecast'), weather, outages)

    # FUELHH fixture: period 48 of 2026-11-01, with the MID fixture's APX price for that period
    [period] = grid.collect()
    assert (period.settlement_period, period.market_index_price_gbp_mwh) == (48, 203.83)
    assert as_strings(grid, 'start_time') == [('2026-11-01 23:30:00',)]
    assert generation.count() == 1 and weather.count() == 10   # 5 points x 1 day x 2 kinds
    rows = {r.target_date.isoformat(): r for r in features.collect()}
    assert set(rows) == {'2026-11-01', '2026-11-02', '2026-11-03'}
    # The DGWS fixture's day-ahead solar forecast for 2026-11-02 was published 16:45 on the 1st: before the cutoff
    assert rows['2026-11-02'].fc_solar_mean_mw == 0.0
    assert rows['2026-11-02'].is_trainable is False   # one period of FUELHH is not a complete day
