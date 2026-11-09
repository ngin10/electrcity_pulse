"""Producer step 1: fetch FUELHH records from the Elexon API.

Provides the HTTP session (with retries), the date-window splitter and the fetch itself. The batch sources
in batch_ingest/ reuse the session, the window splitter and FetchError.
"""
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
    """An API request that failed for good: an error status after retries, or a response we can't read."""


def session_creation():
    """One HTTP session for every request, retrying temporary failures with growing waits."""
    session = requests.Session()
    retry = Retry(
        # Up to 5 retries in total, including connection and read failures
        total=5,
        connect=5,
        read=5,
        # Waits 0.5 s, 1 s, 2 s, ... between tries, so a struggling server gets time to recover
        backoff_factor=0.5,
        # Only temporary errors; a 400/401/403/404 won't be fixed by asking again
        status_forcelist=[500, 502, 503, 504, 429],
        allowed_methods=['GET'],
        # After a 429 (rate limited), wait as long as the server asks
        respect_retry_after_header=True,
        # After the last retry, return the error response so the caller can report its body
        raise_on_status=False,
    )
    session.mount('https://', HTTPAdapter(max_retries=retry))
    return session


def uk_today():
    """Today's date in the UK, which is what a settlement date means."""
    return datetime.now(UK_TZ).date()


def date_windows(start: date, end: date, chunk_days: int):
    """Split [start, end] (inclusive) into consecutive inclusive windows of at most chunk_days.

    e.g. 1..20 Jan in 7-day chunks -> (1, 7), (8, 14), (15, 20). APIs limit the range per request
    (Elexon: 7 days), so a long range is fetched one window at a time.

    A generator: it hands out one window at a time instead of building the whole list first, so the caller
    fetches, stores and checkpoints each window before asking for the next, and can stop between windows.
    Edge cases: start == end gives one single-day window; start after end gives none.
    """
    # The first day of the next window to hand out
    cursor = start
    while cursor <= end:
        # A window of chunk_days days, counting both ends: 1 Jan + 6 days = 7 Jan, which is 7 days (1..7).
        # Adding chunk_days instead would give 8 days and break Elexon's 7-day limit.
        # min(..., end): the last window stops at end, so it can be shorter (15..20 is 6 days)
        window_end = min(cursor + timedelta(days=chunk_days - 1), end)
        # Hand this window to the caller; the loop resumes here when the caller asks for the next one
        yield cursor, window_end
        # The next window starts the day after this one ends: no day skipped, no day fetched twice
        cursor = window_end + timedelta(days=1)

    # Walkthrough for 1..20 Jan, chunk_days=7:
    #   cursor  1 Jan -> window_end = min(7 Jan, 20 Jan)  =  7 Jan -> yield (1, 7)   -> cursor  8 Jan
    #   cursor  8 Jan -> window_end = min(14 Jan, 20 Jan) = 14 Jan -> yield (8, 14)  -> cursor 15 Jan
    #   cursor 15 Jan -> window_end = min(21 Jan, 20 Jan) = 20 Jan -> yield (15, 20) -> cursor 21 Jan
    #   cursor 21 Jan is after 20 Jan -> loop ends


def fetch_data(session, url, start_date: date, end_date: date):
    """Fetch the FUELHH records for one window of settlement dates; returns them as a list, unchanged."""
    # Elexon's settlement-date range includes both ends
    params = {
        'settlementDateFrom': start_date.isoformat(),
        'settlementDateTo': end_date.isoformat(),
        'format': 'json',
    }
    # (connect, read) timeouts in seconds, so a stuck connection can't hang the run
    response = session.get(url, params=params, timeout=(5, 30))
    if not response.ok:
        # 4xx bodies explain what was wrong with the request (e.g. range too large)
        raise FetchError(f'{response.status_code} from API for {start_date}..{end_date}: {response.text[:500]}')

    # Check 1 of 2: is the body valid JSON at all?
    try:
        payload = response.json()
    except ValueError as exc:
        # requests' JSONDecodeError is a ValueError. It covers a body that isn't JSON even though the status
        # was OK: an empty body, an HTML maintenance or Wi-Fi login page, a plain-text error ("Rate limit
        # exceeded"), CSV, or JSON cut off when the connection dropped. Bytes that aren't valid in the
        # charset don't raise; requests replaces them with U+FFFD.
        raise FetchError(f'Non-JSON response for {start_date}..{end_date}') from exc

    # Check 2 of 2: is that JSON a list of records? Valid JSON doesn't have to be: "maintenance", 42, true,
    # null and {"data": "maintenance"} all pass check 1. This matters because publish_to_kafka loops over
    # the list and publishes each item: given the string "maintenance" it would publish its letters "m", "a",
    # "i", ... as 11 records, and given a dict, its key names, with nothing reported as wrong.
    # This checks only the response's structure, needed here to split it into messages; what is inside each
    # record is not checked and goes to the lakehouse as it is.
    #
    # Elexon wraps the records as {"data": [...]}; a bare list [...] is accepted too.
    records = payload.get('data', []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise FetchError(f'Unexpected payload shape: {type(records).__name__}')

    # Two cases pass both checks quietly: {} gives zero records (watch the "Fetched 0 records" log line),
    # and NaN or duplicate keys, which Python's parser accepts, are published as they are for the
    # lakehouse checks to catch.
    log.info('Fetched %d records for %s..%s', len(records), start_date, end_date)
    return records
