"""Producer step 3: publish each record to Kafka and wait until Kafka confirms every one.

publish_to_kafka raises if any message isn't confirmed, so main.py never moves the checkpoint past
a window that didn't fully arrive.
"""
import hashlib
import json
import logging

from envelope import build_envelope

log = logging.getLogger(__name__)

# How long to wait for Kafka to confirm a window's messages before giving up
FLUSH_TIMEOUT_SECONDS = 60
# Together these identify one record: a fuel type's generation in one half-hour
KEY_FIELDS = ('settlementDate', 'settlementPeriod', 'fuelType')


class PublishError(Exception):
    """Kafka didn't confirm every message of a window."""


def make_producer(bootstrap_servers):
    # Imported here so the rest of the module (and its tests) work without the Kafka client installed
    from confluent_kafka import Producer

    return Producer({
        'bootstrap.servers': bootstrap_servers,
        'client.id': 'pulse-fuelhh-ingest',
        # A message counts as delivered only once every replica of the partition has it
        'acks': 'all',
        # Kafka drops duplicates caused by the producer's own retries, and keeps messages in order
        'enable.idempotence': True,
        # With idempotence, let librdkafka retry until delivery.timeout.ms instead of capping retries
        'delivery.timeout.ms': 120000,
        # Wait up to 10 ms to group messages into batches of up to 16 KB: fewer, larger requests
        'linger.ms': 10,
        'batch.size': 16384,
        'max.in.flight.requests.per.connection': 5,
        'compression.type': 'gzip',
    })


def make_admin(bootstrap_servers):
    from confluent_kafka.admin import AdminClient

    return AdminClient({'bootstrap.servers': bootstrap_servers})


def ensure_topics(admin, settings):
    """Create the topic on first run; an existing topic is left as it is."""
    from confluent_kafka import KafkaError, KafkaException
    from confluent_kafka.admin import NewTopic

    topics = [
        # Compacted: Kafka keeps the latest value per key, so re-fetched/revised records replace older ones.
        # min.compaction.lag.ms keeps every message for at least an hour first, so the consumer sees it
        NewTopic(settings.topic, num_partitions=settings.topic_partitions,
                 replication_factor=settings.topic_replication,
                 config={'cleanup.policy': 'compact', 'min.compaction.lag.ms': '3600000'}),
    ]
    for name, future in admin.create_topics(topics).items():
        try:
            future.result()
            log.info('Created topic %s', name)
        except KafkaException as exc:
            if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise
            log.debug('Topic %s already exists', name)


def event_key(record):
    """Kafka key: the record's identity, so compaction keeps only the latest revision of each record.

    A record missing part of that identity is keyed by its whole content instead; otherwise unrelated broken
    records would share a key and compaction would delete all but one before the lakehouse could reject them.
    """
    if not isinstance(record, dict) or any(record.get(field) is None for field in KEY_FIELDS):
        identity = record
    else:
        identity = {'dataset': record.get('dataset', 'FUELHH'), **{field: record[field] for field in KEY_FIELDS}}
    # sort_keys and fixed separators: the same identity always gives the same text, hence the same hash
    raw = json.dumps(identity, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class DeliveryTracker:
    """Counts Kafka's delivery confirmations; Kafka calls it once per message."""

    def __init__(self):
        self.delivered = 0
        self.failed = 0

    def __call__(self, err, msg):
        if err is not None:
            self.failed += 1
            log.error('Delivery failed for key %s on %s: %s', msg.key(), msg.topic(), err)
        else:
            self.delivered += 1


def _produce(producer, topic, key, value, callback):
    """Queue one message; produce() only queues it, and Kafka confirms it later through the callback."""
    while True:
        try:
            producer.produce(topic, key=key, value=value, callback=callback)
            break
        except BufferError:
            # Local queue full: serve delivery callbacks to drain it, then retry
            producer.poll(1)
    # Handle any confirmations that have already arrived
    producer.poll(0)


def publish_to_kafka(producer, topic, records, window_start, window_end):
    """Publish every record raw and block until each is acknowledged; returns the count, raises if any were lost.

    Nothing is filtered here: the lakehouse runs the data-quality checks, so bad records still reach it.
    """
    tracker = DeliveryTracker()

    for record in records:
        value = build_envelope(record, window_start, window_end)
        # Compact JSON, as UTF-8 bytes
        _produce(producer, topic, event_key(record), json.dumps(value, separators=(',', ':')).encode('utf-8'),
                 tracker)

    # Wait until Kafka has confirmed (or rejected) every queued message
    remaining = producer.flush(FLUSH_TIMEOUT_SECONDS)
    if remaining > 0:
        raise PublishError(f'{remaining} messages still undelivered after {FLUSH_TIMEOUT_SECONDS}s flush')
    if tracker.failed > 0:
        raise PublishError(f'{tracker.failed} messages failed delivery')

    log.info('Window %s..%s: %d published, %d acknowledged',
             window_start, window_end, len(records), tracker.delivered)
    return len(records)
