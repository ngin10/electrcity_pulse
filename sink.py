import hashlib
import json
import logging

from validation import build_dead_letter, build_envelope, validate_record

log = logging.getLogger(__name__)

FLUSH_TIMEOUT_SECONDS = 60


class PublishError(Exception):
    pass


def make_producer(bootstrap_servers):
    from confluent_kafka import Producer

    return Producer({
        'bootstrap.servers': bootstrap_servers,
        'client.id': 'pulse-fuelhh-ingest',
        'acks': 'all',
        'enable.idempotence': True,
        # With idempotence, let librdkafka retry until delivery.timeout.ms instead of capping retries
        'delivery.timeout.ms': 120000,
        'linger.ms': 10,
        'batch.size': 16384,
        'max.in.flight.requests.per.connection': 5,
        'compression.type': 'gzip',
    })


def make_admin(bootstrap_servers):
    from confluent_kafka.admin import AdminClient

    return AdminClient({'bootstrap.servers': bootstrap_servers})


def ensure_topics(admin, settings):
    from confluent_kafka import KafkaError, KafkaException
    from confluent_kafka.admin import NewTopic

    topics = [
        # Compacted: Kafka keeps the latest value per key, so re-fetched/revised records replace older ones
        NewTopic(settings.topic, num_partitions=settings.topic_partitions,
                 replication_factor=settings.topic_replication,
                 config={'cleanup.policy': 'compact', 'min.compaction.lag.ms': '3600000'}),
        NewTopic(settings.dlq_topic, num_partitions=1,
                 replication_factor=settings.topic_replication,
                 config={'cleanup.policy': 'delete', 'retention.ms': str(14 * 24 * 3600 * 1000)}),
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
    key_fields = {
        'dataset': record.get('dataset', 'FUELHH'),
        'settlementDate': record.get('settlementDate'),
        'settlementPeriod': record.get('settlementPeriod'),
        'fuelType': record.get('fuelType'),
    }
    raw = json.dumps(key_fields, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class DeliveryTracker:
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
    while True:
        try:
            producer.produce(topic, key=key, value=value, callback=callback)
            break
        except BufferError:
            # Local queue full: serve delivery callbacks to drain it, then retry
            producer.poll(1)
    producer.poll(0)


def publish_to_kafka(producer, topic, dlq_topic, records, window_start, window_end):
    """Publish records and block until every message is acknowledged; raise if any were lost."""
    tracker = DeliveryTracker()
    published = invalid = 0

    for record in records:
        errors = validate_record(record)
        if errors:
            invalid += 1
            log.warning('Invalid record routed to DLQ: %s', errors)
            value = build_dead_letter(record, errors, window_start, window_end)
            key = None
            target = dlq_topic
        else:
            published += 1
            value = build_envelope(record, window_start, window_end)
            key = event_key(record)
            target = topic
        _produce(producer, target, key, json.dumps(value, separators=(',', ':')).encode('utf-8'), tracker)

    remaining = producer.flush(FLUSH_TIMEOUT_SECONDS)
    if remaining > 0:
        raise PublishError(f'{remaining} messages still undelivered after {FLUSH_TIMEOUT_SECONDS}s flush')
    if tracker.failed > 0:
        raise PublishError(f'{tracker.failed} messages failed delivery')

    log.info('Window %s..%s: %d published, %d to DLQ, %d acknowledged',
             window_start, window_end, published, invalid, tracker.delivered)
    return published, invalid
