import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# Settlement dates are defined in UK local time, not UTC
UK_TZ = ZoneInfo('Europe/London')


class FetchError(Exception):
    pass


def session_creation():
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504, 429],
        allowed_methods=['GET'],
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session.mount('https://', HTTPAdapter(max_retries=retry))
    return session


def uk_today():
    return datetime.now(UK_TZ).date()


def date_windows(start: date, end: date, chunk_days: int):
    """Split [start, end] (inclusive) into consecutive inclusive windows of at most chunk_days."""
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=chunk_days - 1), end)
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)


def fetch_data(session, url, start_date: date, end_date: date):
    params = {
        'settlementDateFrom': start_date.isoformat(),
        'settlementDateTo': end_date.isoformat(),
        'format': 'json',
    }
    response = session.get(url, params=params, timeout=(5, 30))
    if not response.ok:
        # 4xx bodies explain what was wrong with the request (e.g. range too large)
        raise FetchError(f'{response.status_code} from API for {start_date}..{end_date}: {response.text[:500]}')

    try:
        payload = response.json()
    except ValueError as exc:
        raise FetchError(f'Non-JSON response for {start_date}..{end_date}') from exc

    records = payload.get('data', []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise FetchError(f'Unexpected payload shape: {type(records).__name__}')

    log.info('Fetched %d records for %s..%s', len(records), start_date, end_date)
    return records
