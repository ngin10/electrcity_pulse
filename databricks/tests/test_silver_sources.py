"""Silver parsing and rules, run on real API responses (trimmed copies in fixtures/)."""
import copy
import json

import pytest

from bronze_rows import as_strings, batch_bronze, envelope, fixture_json, fixture_text, kafka_bronze
from pulse import sources
from pulse.quality import check, rejected_rows, valid_rows

BATCH_FIXTURES = [
    (sources.DEMAND_OUTTURN, 'elexon_demand_outturn', 4),
    (sources.WIND_FORECAST, 'elexon_wind_forecast', 4),
    (sources.DEMAND_FORECAST, 'elexon_demand_forecast', 4),
    (sources.WIND_SOLAR_FORECAST, 'elexon_wind_solar_forecast', 4),
    (sources.WIND_SOLAR_ACTUAL, 'elexon_wind_solar_actual', 4),
    (sources.SYSTEM_PRICES, 'elexon_system_prices', 4),
    (sources.MARKET_INDEX, 'elexon_market_index', 4),
    (sources.REMIT_OUTAGES, 'elexon_remit_outages', 4),
    (sources.BANK_HOLIDAYS, 'uk_bank_holidays', 6),           # 3 divisions x 2 events
    (sources.CARBON_INTENSITY, 'carbon_intensity', 4),
    (sources.PV_LIVE, 'pv_live', 4),
    (sources.NESO_DEMAND, 'neso_historic_demand', 2),
    (sources.WEATHER_DAY_AHEAD, 'weather_forecast_day_ahead', 10),   # 5 points x 2 hours
    (sources.WEATHER_HISTORICAL, 'weather_historical_forecast', 10),
]


# Warnings real data is known to raise: a fully unavailable unit reports more MW unavailable than its normal
# capacity (412 of 390), so the row is kept and gold caps it
KNOWN_REAL_WARNINGS = {'remit_outages': {'unavailable_exceeds_normal'}}


def checked(spark, source, bronze):
    return check(source, bronze).cache()


def reasons_of(df):
    return [(list(r.reject_reasons), list(r.warnings)) for r in df.select('reject_reasons', 'warnings').collect()]


# --- Every source reads its real response cleanly ---------------------------------------------------------

@pytest.mark.parametrize('source,fixture,expected_rows', BATCH_FIXTURES, ids=[f[1] for f in BATCH_FIXTURES])
def test_real_response_is_fully_valid(spark, source, fixture, expected_rows):
    result = checked(spark, source, batch_bronze(spark, fixture_text(fixture)))
    assert result.count() == expected_rows
    allowed = KNOWN_REAL_WARNINGS.get(source.table, set())
    problems = [r for r in reasons_of(result) if r[0] or set(r[1]) - allowed]
    assert problems == []
    # Every column the silver table promises exists, typed
    assert valid_rows(source, result).count() == expected_rows


def test_fuelhh_messages_are_fully_valid(spark):
    records = fixture_json('fuelhh_records')
    result = checked(spark, sources.FUELHH, kafka_bronze(spark, *[envelope(r) for r in records]))
    assert reasons_of(result) == [([], [])] * len(records)
    # An interconnector exporting (INTNED -372) is normal, so no negative_generation warning
    assert ('INTNED', '-372.0') in as_strings(result, 'fuel_type', 'generation_mw')
    assert as_strings(result.limit(1), 'settlement_date', 'start_time', 'publish_time') == [
        ('2026-11-01', '2026-11-01 23:30:00', '2026-11-02 00:00:00')]


def test_values_are_typed(spark):
    ci = checked(spark, sources.CARBON_INTENSITY, batch_bronze(spark, fixture_text('carbon_intensity')))
    assert as_strings(ci.limit(1), 'period_start', 'period_end', 'forecast_gco2_kwh', 'actual_gco2_kwh',
                      'intensity_index') == [('2026-10-31 23:30:00', '2026-11-01 00:00:00', '197', '190', 'high')]

    pv = checked(spark, sources.PV_LIVE, batch_bronze(spark, fixture_text('pv_live')))
    assert as_strings(pv.limit(1), 'gsp_id', 'datetime_gmt', 'generation_mw') == [('0', '2026-11-02 00:00:00', '0.0')]

    remit = checked(spark, sources.REMIT_OUTAGES, batch_bronze(spark, fixture_text('elexon_remit_outages')))
    assert as_strings(remit.limit(1), 'mrid', 'revision_number', 'unavailable_capacity_mw', 'event_start_time') == [
        ('11XINNOGY------2-NGET-RMT-00219922', '6', '26.0', '2026-11-01 11:42:49')]

    prices = checked(spark, sources.SYSTEM_PRICES, batch_bronze(spark, fixture_text('elexon_system_prices')))
    assert as_strings(prices.limit(1), 'system_sell_price_gbp_mwh', 'bsad_defaulted', 'created_at') == [
        ('250.0', 'false', '2026-11-02 00:44:24')]
    assert as_strings(prices.limit(1), 'fetched_at') == [('2026-11-06 10:00:00.123456',)]


def test_neso_reads_all_three_date_formats(spark):
    bodies = [fixture_text(f'neso_historic_demand_{year}') for year in (2021, 2023, 2024)]
    result = checked(spark, sources.NESO_DEMAND, batch_bronze(spark, *bodies))
    assert all(r == ([], []) for r in reasons_of(result))
    assert sorted({d for (d,) in as_strings(result, 'settlement_date')}) == ['2021-11-01', '2023-11-01', '2024-11-01']


def test_weather_points_are_named_by_position(spark):
    result = checked(spark, sources.WEATHER_DAY_AHEAD, batch_bronze(spark, fixture_text('weather_forecast_day_ahead')))
    rows = as_strings(result.orderBy('record_index'), 'location', 'time', 'wind_speed_100m_kmh')
    assert rows[:2] == [('dogger_bank', '2026-11-01 00:00:00', '24.6'), ('dogger_bank', '2026-11-01 01:00:00', '26.2')]
    assert [r[0] for r in rows[::2]] == ['dogger_bank', 'east_anglia_offshore', 'irish_sea', 'scottish_highlands',
                                         'southern_england']


def test_weather_in_another_order_is_rejected(spark):
    points = fixture_json('weather_forecast_day_ahead')
    points[0], points[1] = points[1], points[0]
    result = checked(spark, sources.WEATHER_DAY_AHEAD, batch_bronze(spark, json.dumps(points)))
    assert 'unexpected_location' in result.orderBy('record_index').first().reject_reasons


# --- FUELHH rules -------------------------------------------------------------------------------------------

FUELHH_RECORD = {'dataset': 'FUELHH', 'publishTime': '2026-10-26T00:30:00Z', 'startTime': '2026-10-25T23:30:00Z',
                 'settlementDate': '2026-10-25', 'settlementPeriod': 49, 'fuelType': 'WIND', 'generation': 5000}


def fuelhh_reasons(spark, *messages):
    return reasons_of(checked(spark, sources.FUELHH, kafka_bronze(spark, *messages)).orderBy('kafka_offset'))


def with_(**changes):
    record = copy.deepcopy(FUELHH_RECORD)
    record.update(changes)
    return record


def test_period_49_is_valid_only_on_the_autumn_clock_change(spark):
    # 2026-10-25 has 50 periods; period 49 starts at 23:00 UTC
    autumn = with_(startTime='2026-10-25T23:00:00Z')
    ordinary_day = with_(settlementDate='2026-10-26', startTime='2026-10-27T00:00:00Z')
    assert fuelhh_reasons(spark, envelope(autumn), envelope(ordinary_day)) == [([], []), (['period_out_of_range'], [])]


def test_start_time_must_match_date_and_period(spark):
    # Period 48 of an ordinary GMT day starts at 23:30 UTC, not 23:00
    [(reject, warnings)] = fuelhh_reasons(spark, envelope(with_(settlementDate='2026-11-01', settlementPeriod=48,
                                                                startTime='2026-11-01T23:00:00Z')))
    assert reject == [] and warnings == ['start_time_mismatch']


def test_missing_and_wrong_type_are_told_apart(spark):
    missing = with_(startTime='2026-10-25T23:00:00Z')
    del missing['fuelType']
    wrong_type = with_(startTime='2026-10-25T23:00:00Z', settlementPeriod='forty-nine')
    [(missing_reasons, _), (wrong_reasons, _)] = fuelhh_reasons(spark, envelope(missing), envelope(wrong_type))
    assert missing_reasons == ['missing_fuel_type']
    assert wrong_reasons == ['bad_type_settlement_period']


def test_nan_generation_is_rejected(spark):
    # Python's json.dumps writes NaN; it is a valid double, so only the finiteness rule catches it
    message = envelope(with_(startTime='2026-10-25T23:00:00Z')).replace('"generation": 5000', '"generation": NaN')
    [(reject, _)] = fuelhh_reasons(spark, message)
    assert 'not_finite_generation_mw' in reject


def test_broken_messages_get_one_root_cause(spark):
    assert fuelhh_reasons(spark, 'maintenance') == [(['invalid_json'], [])]
    assert fuelhh_reasons(spark, envelope('maintenance')) == [(['record_not_object'], [])]
    assert fuelhh_reasons(spark, envelope(None)) == [(['no_records'], [])]
    assert fuelhh_reasons(spark, envelope(with_(startTime='2026-10-25T23:00:00Z'), schema_version=2))[0][0] == [
        'unsupported_schema_version']


def test_unknown_things_are_kept_and_flagged(spark):
    new_interconnector = with_(startTime='2026-10-25T23:00:00Z', fuelType='intnew ', generation=-50, unit='MW')
    [(reject, warnings)] = fuelhh_reasons(spark, envelope(new_interconnector))
    assert reject == []
    assert set(warnings) == {'unexpected_fields', 'unknown_fuel_type'}


# --- Batch response rules -----------------------------------------------------------------------------------

def test_bad_responses_are_rejected_with_the_response_kept(spark):
    bronze = batch_bronze(spark, '{"data": []}', '<html>Service Unavailable</html>')
    result = checked(spark, sources.DEMAND_OUTTURN, bronze).orderBy('request_url')
    assert reasons_of(result) == [(['no_records'], []), (['invalid_json'], [])]
    rejected = rejected_rows(sources.DEMAND_OUTTURN, result).orderBy('request_url').collect()
    assert [r.response_json for r in rejected] == ['{"data": []}', '<html>Service Unavailable</html>']
    assert {r.source_table for r in rejected} == {'demand_outturn'}


def test_http_errors_are_rejected(spark):
    result = checked(spark, sources.CARBON_INTENSITY, batch_bronze(spark, fixture_text('carbon_intensity'),
                                                                   http_status=500))
    assert all('http_not_ok' in reject for reject, _ in reasons_of(result))


def test_reordered_pv_columns_are_rejected(spark):
    body = fixture_json('pv_live')
    body['meta'] = ['datetime_gmt', 'gsp_id', 'generation_mw']
    result = checked(spark, sources.PV_LIVE, batch_bronze(spark, json.dumps(body)))
    assert all('unexpected_columns' in reject for reject, _ in reasons_of(result))


def test_rejected_record_keeps_its_own_text(spark):
    body = fixture_json('carbon_intensity')
    body['data'][1]['intensity']['forecast'] = 'high'
    result = checked(spark, sources.CARBON_INTENSITY, batch_bronze(spark, json.dumps(body)))
    [rejected] = rejected_rows(sources.CARBON_INTENSITY, result).collect()
    assert rejected.reject_reasons == ['bad_type_forecast_gco2_kwh']
    assert json.loads(rejected.record_json)['intensity']['forecast'] == 'high'
    assert rejected.response_json is None   # the record is enough; the whole response isn't copied
    assert valid_rows(sources.CARBON_INTENSITY, result).count() == 3


def test_every_source_has_a_unique_table_and_final_columns_cover_its_keys():
    tables = [s.table for s in sources.SOURCES]
    assert len(tables) == len(set(tables)) == 15
    for source in sources.SOURCES:
        columns = source.final_columns
        assert set(source.keys) <= set(columns), source.table
        assert set(source.sequence) <= set(columns), source.table
        assert set(source.cluster_by) <= set(columns), source.table
