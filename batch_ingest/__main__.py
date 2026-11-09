"""Fetch the batch sources and land every API response unchanged: python -m batch_ingest [--source NAME ...]

One run catches each source up to today and exits, so schedule it (cron, Task Scheduler, a job). Each source
keeps its own checkpoint, advanced only after a window's responses are stored, so an interrupted backfill
resumes where it stopped.
"""
import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import landing
from batch_ingest.sources import SOURCES
from checkpoint import load_checkpoint, save_checkpoint
from source import FetchError, date_windows, session_creation, uk_today

log = logging.getLogger('pulse.batch')

# One row per API response. Fixed types for the columns around the body; raw_value is the body as received
DTYPES = {
    'raw_value': 'string', 'source': 'string', 'request_url': 'string', 'window_from': 'string',
    'window_to': 'string', 'http_status': 'int64', 'fetched_at': 'string',
}


def _env(name, default=None):
    """An environment variable's value, treating unset and empty ('') the same: use the default."""
    value = os.getenv(name)
    return value if value not in (None, '') else default


@dataclass(frozen=True)
class BatchSettings:
    """Settings shared by every source, read from environment variables."""
    landing_path: str
    state_dir: str
    backfill_start: date | None
    initial_lookback_days: int
    log_level: str


def load_batch_settings():
    backfill = _env('BACKFILL_START')
    return BatchSettings(
        landing_path=_env('LANDING_PATH', 'data/landing'),
        state_dir=_env('BATCH_STATE_DIR', 'state/batch'),
        backfill_start=date.fromisoformat(backfill) if backfill else None,
        initial_lookback_days=int(_env('INITIAL_LOOKBACK_DAYS', 7)),
        log_level=_env('LOG_LEVEL', 'INFO'),
    )


def compute_range(source, checkpoint, today, backfill_start, initial_lookback_days):
    """Inclusive range of days to fetch for one source on this run."""
    if checkpoint is None:
        # First run for this source: from BACKFILL_START, or the last INITIAL_LOOKBACK_DAYS days
        start = backfill_start or today - timedelta(days=initial_lookback_days)
    else:
        # Step back to re-fetch the days the source may still revise
        start = min(checkpoint, today) - timedelta(days=source.revision_days)
    if source.earliest:
        # Never ask for days before the source has data
        start = max(start, source.earliest)
    return start, today


def fetch_window(session, source, window_start, window_end):
    """Fetch every request for one window; returns landing rows, or raises FetchError on any failure."""
    rows = []
    # Usually one request per window; NESO needs two when a window crosses New Year
    for request in source.build(session, window_start, window_end):
        # The session retries temporary failures (see source.session_creation); (connect, read) timeouts
        response = session.get(request.url, params=request.params, timeout=(5, 60))
        if not response.ok:
            # An error response is never stored as if it were data; the window is retried next run
            raise FetchError(f'{source.name}: {response.status_code} for {window_start}..{window_end}: '
                             f'{response.text[:300]}')
        rows.append({
            # The body exactly as received; parsing and checks happen in the lakehouse. Nothing here parses
            # it, so a non-JSON body sent with an OK status (an HTML maintenance page, say) lands too:
            # silver should reject rows whose raw_value doesn't parse as JSON
            'raw_value': response.text,
            'source': source.name,
            # The exact URL requested, query included: enough to repeat or trace the call
            'request_url': response.url,
            # The days this response covers
            'window_from': window_start.isoformat(),
            'window_to': window_end.isoformat(),
            'http_status': response.status_code,
            # When we fetched it (UTC): readers keep the latest fetch of re-fetched days; also sets ingest_date
            'fetched_at': datetime.now(timezone.utc).isoformat(),
        })
    return rows


def run_source(session, source, settings, today):
    """Catch one source up to today; returns the number of responses landed."""
    # Each source has its own checkpoint, so sources progress (and fail) independently
    checkpoint_path = os.path.join(settings.state_dir, f'{source.name}.json')
    checkpoint = load_checkpoint(checkpoint_path)
    start, end = compute_range(source, checkpoint, today, settings.backfill_start, settings.initial_lookback_days)
    log.info('%s: fetching %s..%s (checkpoint %s)', source.name, start, end, checkpoint)

    landed = 0
    # Fetch -> store -> checkpoint, one window at a time: a failure loses at most the current window's work
    for window_start, window_end in date_windows(start, end, source.chunk_days):
        rows = fetch_window(session, source, window_start, window_end)
        if rows:
            landing.append_rows(settings.landing_path, source.name, rows, DTYPES, 'fetched_at')
            landed += len(rows)
        # Only after the window is stored; never moved backwards by the revision re-fetch
        checkpoint = max(window_end, checkpoint) if checkpoint else window_end
        save_checkpoint(checkpoint_path, checkpoint)
    log.info('%s: landed %d responses', source.name, landed)
    return landed


def main():
    parser = argparse.ArgumentParser(description='Fetch the batch data sources into the landing zone')
    parser.add_argument('--source', action='append', choices=sorted(SOURCES),
                        help='only this source (repeatable); default: all')
    parser.add_argument('--list', action='store_true', help='list the sources and exit')
    args = parser.parse_args()

    if args.list:
        for source in SOURCES.values():
            print(f'{source.name:30} {source.description}')
        return 0

    settings = load_batch_settings()
    logging.basicConfig(level=settings.log_level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    # One session with retries for every request; the User-Agent tells API providers who is calling
    session = session_creation()
    session.headers['User-Agent'] = 'electricity-pulse/1.0 (data engineering portfolio project)'

    exit_code = 0
    # One "today" for the whole run, so every source stops at the same date
    today = uk_today()
    try:
        for name in args.source or SOURCES:
            try:
                run_source(session, SOURCES[name], settings, today)
            except Exception:
                # Its checkpoint stayed at the last stored window; the other sources still run
                log.exception('%s failed; it resumes from its checkpoint next run', name)
                exit_code = 1
    finally:
        session.close()
    # Non-zero tells a scheduler that at least one source failed
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
