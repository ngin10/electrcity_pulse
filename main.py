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
        start = settings.backfill_start or today - timedelta(days=settings.initial_lookback_days)
    else:
        # Step back to re-fetch recent days, because Elexon revises figures after first publication
        start = min(checkpoint, today) - timedelta(days=settings.revision_lookback_days)
    return start, today


def run_once(settings, session, producer, stop_event):
    checkpoint = load_checkpoint(settings.checkpoint_path)
    start, end = compute_range(settings, checkpoint, uk_today())
    log.info('Run starting: checkpoint=%s range=%s..%s', checkpoint, start, end)

    for window_start, window_end in date_windows(start, end, settings.chunk_days):
        if stop_event.is_set():
            log.info('Stop requested, ending run before %s', window_start)
            return
        records = fetch_data(session, settings.api_url, window_start, window_end)
        publish_to_kafka(producer, settings.topic, settings.dlq_topic, records, window_start, window_end)

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

    stop_event = threading.Event()

    def request_stop(signum, _frame):
        log.info('Received signal %s, shutting down after current window', signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    admin = make_admin(settings.kafka_bootstrap)
    ensure_topics(admin, settings)
    session = session_creation()
    producer = make_producer(settings.kafka_bootstrap)

    exit_code = 0
    try:
        while not stop_event.is_set():
            try:
                run_once(settings, session, producer, stop_event)
            except (FetchError, PublishError, requests.RequestException) as exc:
                # Checkpoint was not advanced, so the failed window is retried on the next run
                log.error('Run failed: %s', exc)
                exit_code = 1
            if args.once:
                break
            stop_event.wait(settings.poll_interval_seconds)
    finally:
        remaining = producer.flush(10)
        if remaining:
            log.error('%d messages undelivered at shutdown', remaining)
            exit_code = 1
        session.close()

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
