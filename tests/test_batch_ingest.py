import json
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_ingest.__main__ import BatchSettings, compute_range, run_source
from batch_ingest.sources import SOURCES, WEATHER_POINTS, Request, neso_resource_ids
from checkpoint import load_checkpoint
from landing import table_dir
from source import FetchError

NESO_PACKAGE = {'result': {'resources': [
    {'name': 'Historic Demand Data 2025', 'id': 'res-2025', 'datastore_active': True},
    {'name': 'Historic Demand Data 2026', 'id': 'res-2026', 'datastore_active': True},
    {'name': 'Historic Demand Data 2026 (archived copy)', 'id': 'ignored', 'datastore_active': True},
    {'name': 'Data dictionary', 'id': 'dictionary', 'datastore_active': False},
]}}


class FakeResponse:
    def __init__(self, status_code, text, url):
        self.status_code, self.text, self.url = status_code, text, url
        self.ok = status_code < 400

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.status_code)


class FakeSession:
    """Answers every request with a body naming it; fail_on picks requests to answer with a 500."""

    def __init__(self, fail_on=lambda url, params: False):
        self.calls = []
        self.fail_on = fail_on

    def get(self, url, params=None, timeout=None):
        params = params or {}
        self.calls.append((url, params))
        full_url = f'{url}?{urlencode(params)}' if params else url
        if url.endswith('/package_show'):
            return FakeResponse(200, json.dumps(NESO_PACKAGE), full_url)
        if self.fail_on(url, params):
            return FakeResponse(500, 'upstream error', full_url)
        return FakeResponse(200, json.dumps({'echo': full_url}), full_url)


@pytest.fixture
def settings(tmp_path):
    return BatchSettings(landing_path=str(tmp_path / 'landing'), state_dir=str(tmp_path / 'state'),
                         backfill_start=date(2026, 1, 1), initial_lookback_days=7, log_level='INFO')


def read_landed(settings, name):
    return pd.read_parquet(table_dir(settings.landing_path, name))


# --- request building -------------------------------------------------------

def test_chunks_respect_api_limits():
    # Checked against the live APIs: Elexon rejects ranges over 7 days, Carbon Intensity over 31
    for name in ('elexon_demand_outturn', 'elexon_wind_forecast', 'elexon_wind_solar_forecast',
                 'elexon_wind_solar_actual', 'elexon_market_index'):
        assert SOURCES[name].chunk_days <= 7
    assert SOURCES['carbon_intensity'].chunk_days <= 31


@pytest.mark.parametrize('name', ['elexon_demand_forecast', 'elexon_remit_outages', 'elexon_system_prices'])
def test_one_day_apis_get_one_request_per_day(name):
    # NDF and REMIT allow at most 1 day per request; system prices take a single date in the path
    requests = SOURCES[name].build(None, date(2026, 11, 1), date(2026, 11, 7))
    assert len(requests) == 7


def test_demand_forecast_takes_the_evening_publishes():
    requests = SOURCES['elexon_demand_forecast'].build(None, date(2026, 11, 1), date(2026, 11, 2))
    assert [(r.params['publishDateTimeFrom'], r.params['publishDateTimeTo']) for r in requests] == [
        ('2026-11-01T17:00Z', '2026-11-01T17:59Z'), ('2026-11-02T17:00Z', '2026-11-02T17:59Z')]


def test_system_prices_put_the_date_in_the_path():
    first, *_, last = SOURCES['elexon_system_prices'].build(None, date(2026, 11, 1), date(2026, 11, 7))
    assert first.url.endswith('/balancing/settlement/system-prices/2026-11-01')
    assert last.url.endswith('/balancing/settlement/system-prices/2026-11-07')


def test_market_index_ends_at_the_last_half_hour():
    # MID's 'to' is inclusive: 00:00 of the next day would overlap the next window
    [request] = SOURCES['elexon_market_index'].build(None, date(2026, 11, 1), date(2026, 11, 7))
    assert (request.params['from'], request.params['to']) == ('2026-11-01T00:00Z', '2026-11-07T23:30Z')


def test_wind_solar_sources_use_their_datasets():
    for name, dataset in (('elexon_wind_solar_forecast', 'DGWS'), ('elexon_wind_solar_actual', 'AGWS')):
        [request] = SOURCES[name].build(None, date(2026, 11, 1), date(2026, 11, 7))
        assert request.url.endswith(f'/datasets/{dataset}')
        assert request.params['publishDateTimeTo'] == '2026-11-08T00:00Z'


def test_bank_holidays_backfill_is_a_single_request(settings):
    session = FakeSession()
    assert run_source(session, SOURCES['uk_bank_holidays'], settings, today=date(2026, 11, 6)) == 1
    assert session.calls == [('https://www.gov.uk/bank-holidays.json', {})]


def test_elexon_demand_request():
    [request] = SOURCES['elexon_demand_outturn'].build(None, date(2026, 11, 1), date(2026, 11, 7))
    assert request == Request('https://data.elexon.co.uk/bmrs/api/v1/demand/outturn', {
        'settlementDateFrom': '2026-11-01', 'settlementDateTo': '2026-11-07', 'format': 'json'})


def test_end_exclusive_apis_get_the_day_after_the_window():
    [windfor] = SOURCES['elexon_wind_forecast'].build(None, date(2026, 11, 1), date(2026, 11, 7))
    assert (windfor.params['publishDateTimeFrom'], windfor.params['publishDateTimeTo']) == \
        ('2026-11-01T00:00Z', '2026-11-08T00:00Z')
    [carbon] = SOURCES['carbon_intensity'].build(None, date(2026, 11, 1), date(2026, 11, 14))
    assert carbon.url.endswith('/intensity/2026-11-01T00:00Z/2026-11-15T00:00Z')
    [pv] = SOURCES['pv_live'].build(None, date(2026, 12, 1), date(2026, 12, 31))
    assert (pv.params['start'], pv.params['end']) == ('2026-12-01T00:00:00Z', '2027-01-01T00:00:00Z')


def test_weather_requests_all_points_at_once():
    [day_ahead] = SOURCES['weather_forecast_day_ahead'].build(None, date(2026, 11, 1), date(2026, 11, 30))
    assert len(day_ahead.params['latitude'].split(',')) == len(WEATHER_POINTS)
    assert 'wind_speed_100m_previous_day1' in day_ahead.params['hourly'].split(',')
    [stitched] = SOURCES['weather_historical_forecast'].build(None, date(2026, 11, 1), date(2026, 11, 30))
    assert 'wind_speed_100m' in stitched.params['hourly'].split(',')


def test_neso_resource_ids_by_year():
    assert neso_resource_ids(FakeSession()) == {2025: 'res-2025', 2026: 'res-2026'}


def test_neso_window_across_new_year_queries_both_tables():
    requests = SOURCES['neso_historic_demand'].build(FakeSession(), date(2025, 12, 20), date(2026, 1, 10))
    sqls = [r.params['sql'] for r in requests]
    assert '"res-2025"' in sqls[0] and "BETWEEN '2025-12-20' AND '2025-12-31'" in sqls[0]
    assert '"res-2026"' in sqls[1] and "BETWEEN '2026-01-01' AND '2026-01-10'" in sqls[1]
    # Every year's date format is read, not just real dates
    assert all("'DD-Mon-YY'" in sql and "'DD-Mon-YYYY'" in sql for sql in sqls)


def test_neso_refetches_past_its_publishing_delay():
    assert SOURCES['neso_historic_demand'].revision_days >= 28


# --- ranges -----------------------------------------------------------------

def test_first_run_backfills_then_resumes_with_revision_window():
    source = SOURCES['carbon_intensity']
    today = date(2026, 11, 6)
    assert compute_range(source, None, today, date(2023, 11, 1), 7) == (date(2023, 11, 1), today)
    assert compute_range(source, None, today, None, 7) == (date(2026, 10, 30), today)
    assert compute_range(source, date(2026, 11, 6), today, None, 7) == (date(2026, 11, 4), today)


def test_backfill_never_requests_before_the_source_has_data():
    source = SOURCES['weather_forecast_day_ahead']
    start, _ = compute_range(source, None, date(2026, 11, 6), date(2023, 11, 1), 7)
    assert start == date(2024, 3, 1)


# --- running ----------------------------------------------------------------

def test_run_lands_responses_unchanged_and_checkpoints(settings):
    session = FakeSession()
    landed = run_source(session, SOURCES['elexon_demand_outturn'], settings, today=date(2026, 1, 20))

    assert landed == 3  # 2026-01-01..20 in 7-day windows
    df = read_landed(settings, 'elexon_demand_outturn').sort_values('window_from')
    assert list(df['window_from']) == ['2026-01-01', '2026-01-08', '2026-01-15']
    assert list(df['window_to']) == ['2026-01-07', '2026-01-14', '2026-01-20']
    assert json.loads(df.iloc[0]['raw_value']) == {'echo': df.iloc[0]['request_url']}
    assert set(df['http_status']) == {200}
    assert load_checkpoint(Path(settings.state_dir) / 'elexon_demand_outturn.json') == date(2026, 1, 20)


def test_failure_keeps_checkpoint_at_last_stored_window(settings):
    session = FakeSession(fail_on=lambda url, params: params.get('settlementDateFrom') == '2026-01-15')
    with pytest.raises(FetchError):
        run_source(session, SOURCES['elexon_demand_outturn'], settings, today=date(2026, 1, 20))

    # The failed window lands nothing and is retried next run
    assert list(read_landed(settings, 'elexon_demand_outturn')['window_from'].sort_values()) == \
        ['2026-01-01', '2026-01-08']
    assert load_checkpoint(Path(settings.state_dir) / 'elexon_demand_outturn.json') == date(2026, 1, 14)
