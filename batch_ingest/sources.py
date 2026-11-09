"""The batch data sources: what to request for a range of days. Responses are landed as received.

Each source's chunk_days is within its API's limit per request, checked against the live APIs:
Elexon allows at most 7 days (1 day for NDF and REMIT), Carbon Intensity at most 31.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

ELEXON = 'https://data.elexon.co.uk/bmrs/api/v1'
NESO = 'https://api.neso.energy/api/3/action'
NESO_DEMAND_PACKAGE = 'historic-demand-data'

# Points averaged downstream into national weather features: offshore wind clusters, Scottish onshore wind,
# and southern England for solar
WEATHER_POINTS = {
    'dogger_bank': (54.8, 1.9),
    'east_anglia_offshore': (52.6, 2.2),
    'irish_sea': (53.8, -3.6),
    'scottish_highlands': (57.0, -4.0),
    'southern_england': (51.5, -1.0),
}
WEATHER_VARIABLES = ('wind_speed_100m', 'wind_speed_10m', 'shortwave_radiation', 'temperature_2m', 'cloud_cover')


@dataclass(frozen=True)
class Request:
    """One HTTP GET: the URL and its query parameters."""
    url: str
    params: dict


@dataclass(frozen=True)
class Source:
    """How to fetch one source; the runner in __main__.py does the fetching, storing and checkpointing."""
    # Also the landing folder: <LANDING_PATH>/<name>/
    name: str
    description: str
    chunk_days: int
    # Recent days re-fetched on every run, because the source fills them in or revises them later
    revision_days: int
    # No data before this date, so a backfill never requests it
    earliest: date | None
    # (session, first day, last day) -> the requests covering those days, inclusive
    build: Callable[..., list[Request]]


def _next_day(day):
    # For APIs whose end time is exclusive: ask until midnight after the last day we want
    return day + timedelta(days=1)


def _days(start, end):
    """Every day from start to end, inclusive: for APIs that take one day per request."""
    day = start
    while day <= end:
        yield day
        day = _next_day(day)


# --- One build function per source: (session, first day, last day) -> requests --------------------------
# They only build URLs and parameters; nothing here changes the data that comes back.

def _elexon_demand(session, start, end):
    # Settlement-date range, both ends included, like FUELHH
    return [Request(f'{ELEXON}/demand/outturn', {
        'settlementDateFrom': start.isoformat(), 'settlementDateTo': end.isoformat(), 'format': 'json'})]


def _elexon_wind_forecast(session, start, end):
    # Every forecast published in the window: each publish covers the next few days, so a day-ahead value
    # can be picked downstream without using anything published later
    return [Request(f'{ELEXON}/datasets/WINDFOR', {
        'publishDateTimeFrom': f'{start.isoformat()}T00:00Z',
        'publishDateTimeTo': f'{_next_day(end).isoformat()}T00:00Z',
        'format': 'json'})]


def _elexon_publish_window(dataset):
    """A build function for an Elexon dataset filtered by publish time; 7 days per request at most."""
    def build(session, start, end):
        return [Request(f'{ELEXON}/datasets/{dataset}', {
            'publishDateTimeFrom': f'{start.isoformat()}T00:00Z',
            'publishDateTimeTo': f'{_next_day(end).isoformat()}T00:00Z',
            'format': 'json'})]
    return build


def _elexon_demand_forecast(session, start, end):
    # NDF is republished every 30 minutes and allows one day per request. Only the publishes between
    # 17:00 and 18:00 UTC are fetched: from about 11:00 UTC each publish covers all of the next day, so
    # these are the day-ahead forecast as known on the day, at a twentieth of the size of every publish
    return [Request(f'{ELEXON}/datasets/NDF', {
        'publishDateTimeFrom': f'{day.isoformat()}T17:00Z',
        'publishDateTimeTo': f'{day.isoformat()}T17:59Z',
        'format': 'json'}) for day in _days(start, end)]


def _elexon_system_prices(session, start, end):
    # The settlement date goes in the URL path, and there's no range form, so one request per day
    return [Request(f'{ELEXON}/balancing/settlement/system-prices/{day.isoformat()}', {'format': 'json'})
            for day in _days(start, end)]


def _elexon_market_index(session, start, end):
    # UTC timestamps, and 'to' is inclusive: ending at 23:30 gives whole days without overlapping the next
    # window. Two providers per half-hour; N2EXMIDP is mostly zero, so silver keeps APXMIDP
    return [Request(f'{ELEXON}/datasets/MID', {
        'from': f'{start.isoformat()}T00:00Z', 'to': f'{end.isoformat()}T23:30Z', 'format': 'json'})]


def _elexon_remit(session, start, end):
    # Outage notices with full details (the /remit/list endpoints give only IDs); one day per request
    return [Request(f'{ELEXON}/datasets/REMIT', {
        'publishDateTimeFrom': f'{day.isoformat()}T00:00Z',
        'publishDateTimeTo': f'{_next_day(day).isoformat()}T00:00Z',
        'format': 'json'}) for day in _days(start, end)]


def _uk_bank_holidays(session, start, end):
    # The whole list (past and upcoming, all UK nations) in one response: the dates aren't used
    return [Request('https://www.gov.uk/bank-holidays.json', {})]


def _carbon_intensity(session, start, end):
    return [Request(
        f'https://api.carbonintensity.org.uk/intensity/{start.isoformat()}T00:00Z/{_next_day(end).isoformat()}T00:00Z',
        {})]


def _pv_live(session, start, end):
    # GSP 0 is the national total
    return [Request('https://api.pvlive.uk/pvlive/api/v4/gsp/0', {
        'start': f'{start.isoformat()}T00:00:00Z', 'end': f'{_next_day(end).isoformat()}T00:00:00Z'})]


def _open_meteo(url, variables):
    """A build function for an Open-Meteo endpoint; both weather sources share it, with different variables."""
    def build(session, start, end):
        lats, lons = zip(*WEATHER_POINTS.values())
        return [Request(url, {
            # One request for all points; the response is a list in this order
            'latitude': ','.join(str(lat) for lat in lats),
            'longitude': ','.join(str(lon) for lon in lons),
            'hourly': ','.join(variables),
            'start_date': start.isoformat(), 'end_date': end.isoformat(),
            'timezone': 'UTC',
        })]
    return build


def neso_resource_ids(session):
    """Map year -> datastore resource id: NESO publishes historic demand as one table per year."""
    # The dataset's catalogue entry lists its tables ("resources")
    response = session.get(f'{NESO}/package_show', params={'id': NESO_DEMAND_PACKAGE}, timeout=(5, 60))
    response.raise_for_status()
    ids = {}
    for resource in response.json()['result']['resources']:
        # Keep only queryable tables named exactly "Historic Demand Data <year>"
        name = resource.get('name', '')
        if resource.get('datastore_active') and name.startswith('Historic Demand Data '):
            year = name.removeprefix('Historic Demand Data ').strip()
            if year.isdigit():
                ids[int(year)] = resource['id']
    return ids


# The yearly tables store the date differently: text '01-JAN-2019' up to 2022, text '01-Jan-23' in 2023,
# a real date from 2024; one expression reads all three
_NESO_DATE = '"SETTLEMENT_DATE"::text'
NESO_SETTLEMENT_DATE = (f"(CASE length({_NESO_DATE}) WHEN 9 THEN to_date({_NESO_DATE}, 'DD-Mon-YY') "
                        f"WHEN 11 THEN to_date({_NESO_DATE}, 'DD-Mon-YYYY') ELSE {_NESO_DATE}::date END)")


def _neso_demand(session, start, end):
    # NESO's API runs read-only SQL on its tables; the date expression only selects which rows to return,
    # and they come back exactly as stored (2023's dates still read '01-Nov-23')
    ids = neso_resource_ids(session)
    requests = []
    # A window crossing New Year needs a query on each year's table
    for year in range(start.year, end.year + 1):
        if year not in ids:
            continue
        first, last = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        sql = (f'SELECT * FROM "{ids[year]}" WHERE {NESO_SETTLEMENT_DATE} '
               f"BETWEEN '{first.isoformat()}' AND '{last.isoformat()}' ORDER BY \"_id\"")
        requests.append(Request(f'{NESO}/datastore_search_sql', {'sql': sql}))
    return requests


# --- The registry: every source the runner fetches, by name ------------------------------------------------
# To add a source: write a build function above and add one line here.

SOURCES = {source.name: source for source in [
    Source('elexon_demand_outturn', 'Elexon national demand outturn, half-hourly',
           chunk_days=7, revision_days=2, earliest=None, build=_elexon_demand),
    Source('elexon_wind_forecast', 'Elexon wind generation forecasts (WINDFOR), every publish',
           chunk_days=7, revision_days=1, earliest=None, build=_elexon_wind_forecast),
    Source('elexon_demand_forecast', 'Elexon national demand forecast (NDF), publishes 17:00-18:00 UTC daily',
           chunk_days=7, revision_days=1, earliest=None, build=_elexon_demand_forecast),
    Source('elexon_wind_solar_forecast', 'Elexon day-ahead wind and solar generation forecasts (DGWS)',
           chunk_days=7, revision_days=1, earliest=None, build=_elexon_publish_window('DGWS')),
    Source('elexon_wind_solar_actual', 'Elexon actual wind and solar generation (AGWS)',
           chunk_days=7, revision_days=2, earliest=None, build=_elexon_publish_window('AGWS')),
    Source('elexon_system_prices', 'Elexon system (imbalance) prices and volumes, half-hourly',
           chunk_days=7, revision_days=3, earliest=None, build=_elexon_system_prices),
    Source('elexon_market_index', 'Elexon market index price and volume (MID), half-hourly',
           chunk_days=7, revision_days=2, earliest=None, build=_elexon_market_index),
    Source('elexon_remit_outages', 'Elexon REMIT outage and unavailability notices, with details',
           chunk_days=7, revision_days=1, earliest=None, build=_elexon_remit),
    # One snapshot per run; the huge chunk makes even a backfill a single request
    Source('uk_bank_holidays', 'GOV.UK bank holidays for England & Wales, Scotland and Northern Ireland',
           chunk_days=36500, revision_days=0, earliest=None, build=_uk_bank_holidays),
    Source('carbon_intensity', 'National carbon intensity, forecast and actual, half-hourly',
           chunk_days=14, revision_days=2, earliest=None, build=_carbon_intensity),
    Source('pv_live', 'Sheffield Solar PV_Live national solar generation, half-hourly',
           chunk_days=31, revision_days=3, earliest=None, build=_pv_live),
    # NESO publishes about three weeks behind, so recent weeks are re-fetched until they appear
    Source('neso_historic_demand', 'NESO demand with embedded wind and solar estimates, half-hourly',
           chunk_days=31, revision_days=45, earliest=None, build=_neso_demand),
    # As forecast the day before: the leakage-free weather input for a day-ahead model
    Source('weather_forecast_day_ahead', 'Open-Meteo day-ahead weather forecasts at UK wind and solar points',
           chunk_days=31, revision_days=2, earliest=date(2024, 3, 1),
           build=_open_meteo('https://previous-runs-api.open-meteo.com/v1/forecast',
                             [f'{v}_previous_day1' for v in WEATHER_VARIABLES])),
    # Stitched short-range forecasts: longer history, closer to actual weather; for exploration, not training
    Source('weather_historical_forecast', 'Open-Meteo historical short-range weather forecasts, same points',
           chunk_days=31, revision_days=2, earliest=date(2021, 1, 1),
           build=_open_meteo('https://historical-forecast-api.open-meteo.com/v1/forecast', WEATHER_VARIABLES)),
]}
