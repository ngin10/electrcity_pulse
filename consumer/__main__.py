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
    pass


def process_batch(consumer, settings, messages):
    """Append one batch to the landing zone and commit its offsets; returns the number of rows stored."""
    consumed_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for msg in messages:
        err = msg.error()
        if err is not None:
            if err.fatal():
                raise ConsumerError(str(err))
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

    # Commit only after files are written: a crash before this replays the batch, and readers drop the duplicates
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
            messages = consumer.consume(settings.batch_max_messages, settings.batch_timeout_seconds)
            if not messages:
                # An empty poll before partitions are assigned doesn't mean the topic is drained
                if args.once and consumer.assignment():
                    break
                continue
            process_batch(consumer, settings, messages)
    except Exception:
        log.exception('Consumer stopped on error; uncommitted messages will be re-read on restart')
        exit_code = 1
    finally:
        consumer.close()

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
