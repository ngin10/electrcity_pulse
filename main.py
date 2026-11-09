"""Producer entry point: python main.py [--once]

Each run fetches FUELHH from Elexon one date window at a time and publishes every record to Kafka. The
checkpoint (last settlement date fully delivered) moves only after Kafka has acknowledged a whole window,
so a failed window is fetched again on the next run.
"""
import argparse
import logging
import signal
import sys
import threading
from datetime import timedelta

import requests

from checkpoint import load_checkpoint, save_checkpoint
from config import load_settings
from sink import PublishError, ensure_topics, make_admin, make_producer, publish_to_kafka
from source import FetchError, date_windows, fetch_data, session_creation, uk_today

log = logging.getLogger('pulse')


def compute_range(settings, checkpoint, today):
    """Pick the inclusive settlement-date range for this run."""
    if checkpoint is None:
        # First run: from BACKFILL_START, or the last INITIAL_LOOKBACK_DAYS days
        start = settings.backfill_start or today - timedelta(days=settings.initial_lookback_days)
    else:
        # Step back to re-fetch recent days, because Elexon revises figures after first publication
        start = min(checkpoint, today) - timedelta(days=settings.revision_lookback_days)
    return start, today


def run_once(settings, session, producer, stop_event):
    """One pass: fetch and publish every window from the checkpoint up to today."""
    checkpoint = load_checkpoint(settings.checkpoint_path)
    start, end = compute_range(settings, checkpoint, uk_today())
    log.info('Run starting: checkpoint=%s range=%s..%s', checkpoint, start, end)

    for window_start, window_end in date_windows(start, end, settings.chunk_days):
        # Ctrl+C stops between windows, never halfway through one
        if stop_event.is_set():
            log.info('Stop requested, ending run before %s', window_start)
            return
        records = fetch_data(session, settings.api_url, window_start, window_end)
        # Raises if any message isn't acknowledged, which skips the checkpoint below
        publish_to_kafka(producer, settings.topic, records, window_start, window_end)

        # Only advance after Kafka has acknowledged everything; never move it backwards on re-fetches
        checkpoint = max(window_end, checkpoint) if checkpoint else window_end
        save_checkpoint(settings.checkpoint_path, checkpoint)


def main():
    parser = argparse.ArgumentParser(description='Ingest Elexon FUELHH into Kafka')
    parser.add_argument('--once', action='store_true', help='run a single pass and exit (for cron/Airflow)')
    args = parser.parse_args()

    settings = load_settings()
    logging.basicConfig(
        level=settings.log_level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    # Ctrl+C or a kill request sets this flag, so the run stops cleanly after the current window
    stop_event = threading.Event()

    def request_stop(signum, _frame):
        log.info('Received signal %s, shutting down after current window', signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # Create the topic if it doesn't exist yet, then open the connections reused for every window
    admin = make_admin(settings.kafka_bootstrap)
    ensure_topics(admin, settings)
    session = session_creation()
    producer = make_producer(settings.kafka_bootstrap)

    exit_code = 0
    try:
        # Without --once, run a pass every POLL_INTERVAL_SECONDS until stopped
        while not stop_event.is_set():
            try:
                run_once(settings, session, producer, stop_event)
            except (FetchError, PublishError, requests.RequestException) as exc:
                # Checkpoint was not advanced, so the failed window is retried on the next run
                log.error('Run failed: %s', exc)
                exit_code = 1
            if args.once:
                break
            # Sleeps, but wakes at once if a stop is requested
            stop_event.wait(settings.poll_interval_seconds)
    finally:
        # Send anything still queued before exiting
        remaining = producer.flush(10)
        if remaining:
            log.error('%d messages undelivered at shutdown', remaining)
            exit_code = 1
        session.close()

    # Non-zero tells a scheduler the run failed
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
