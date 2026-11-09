"""Consumer entry point: python -m consumer [--once]

Reads the FUELHH topic in batches and lands every message, unchanged, as Parquet under LANDING_PATH.
Kafka offsets are committed only after a batch's file is stored, so a crash means the batch is read
again (possibly landing twice, which readers deduplicate) but never lost.
"""
import argparse
import logging
import signal
import sys
import threading
from datetime import datetime, timezone

from consumer.config import load_consumer_settings
from consumer.source import landing_row, make_consumer
from consumer.store import LANDING_TABLE, append_rows

log = logging.getLogger('pulse.consumer')


class ConsumerError(Exception):
    """An unrecoverable Kafka error: stop without committing."""


def process_batch(consumer, settings, messages):
    """Append one batch to the landing zone and commit its offsets; returns the number of rows stored."""
    # One timestamp for the whole batch: when it was read
    consumed_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for msg in messages:
        err = msg.error()
        if err is not None:
            if err.fatal():
                raise ConsumerError(str(err))
            # A temporary notice from the broker (e.g. end of partition), not a message
            log.warning('Consumer error: %s', err)
            continue
        if msg.value() is None:
            # Tombstone on the compacted topic; nothing to store
            continue
        # Every message lands, valid or not; data-quality checks run downstream
        rows.append(landing_row(msg, consumed_at))

    if rows:
        append_rows(settings.landing_path, LANDING_TABLE, rows)
        log.info('Appended %d rows to landing', len(rows))

    # Commit only after files are written: a crash before this replays the batch, and readers drop the duplicates.
    # asynchronous=False waits for Kafka to confirm the commit before reading the next batch
    consumer.commit(asynchronous=False)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description='Consume Elexon FUELHH from Kafka into the landing zone of the lake')
    parser.add_argument('--once', action='store_true', help='exit after the first empty poll (for cron/Airflow)')
    args = parser.parse_args()

    settings = load_consumer_settings()
    logging.basicConfig(
        level=settings.log_level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    # Ctrl+C or a kill request sets this flag, so the consumer stops cleanly after the current batch
    stop_event = threading.Event()

    def request_stop(signum, _frame):
        log.info('Received signal %s, shutting down after current batch', signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    consumer = make_consumer(settings)
    consumer.subscribe([settings.topic])
    log.info('Consuming %s as group %s into %s',
             settings.topic, settings.group_id, settings.landing_path)

    exit_code = 0
    try:
        while not stop_event.is_set():
            # Up to batch_max_messages, or whatever arrived within batch_timeout_seconds
            messages = consumer.consume(settings.batch_max_messages, settings.batch_timeout_seconds)
            if not messages:
                # An empty poll before partitions are assigned doesn't mean the topic is drained
                if args.once and consumer.assignment():
                    break
                continue
            process_batch(consumer, settings, messages)
    except Exception:
        # Whatever failed, the current batch wasn't committed, so nothing is lost
        log.exception('Consumer stopped on error; uncommitted messages will be re-read on restart')
        exit_code = 1
    finally:
        # Leave the consumer group cleanly, so Kafka hands the partitions over without waiting
        consumer.close()

    # Non-zero tells a scheduler the run failed
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
