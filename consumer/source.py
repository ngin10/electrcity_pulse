import logging

log = logging.getLogger(__name__)


def make_consumer(settings):
    from confluent_kafka import Consumer

    return Consumer({
        'bootstrap.servers': settings.kafka_bootstrap,
        'group.id': settings.group_id,
        'client.id': 'pulse-fuelhh-landing',
        # A new group reads the whole compacted topic, i.e. the latest value of every record
        'auto.offset.reset': 'earliest',
        # Offsets are committed only after the batch is safely in storage
        'enable.auto.commit': False,
        'partition.assignment.strategy': 'cooperative-sticky',
    })


def landing_row(msg, consumed_at):
    """Landing row for a Kafka message: the value exactly as received, plus where it came from in Kafka.

    Nothing is parsed or checked here; the lakehouse runs the data-quality checks, so a rule change
    never needs a consumer redeploy and can be re-applied to everything already landed.
    """
    key = msg.key()
    value = msg.value()
    timestamp_type, timestamp_ms = msg.timestamp()
    return {
        # Bytes that are not UTF-8 are kept as \xNN escapes, so the row still lands and fails the JSON check
        'raw_value': value.decode('utf-8', errors='backslashreplace'),
        'kafka_topic': msg.topic(),
        'kafka_partition': msg.partition(),
        'kafka_offset': msg.offset(),
        'kafka_key': key.decode('utf-8', errors='backslashreplace') if key is not None else None,
        # Type 0 means the broker gave no timestamp
        'kafka_timestamp_ms': timestamp_ms if timestamp_type else None,
        'consumed_at': consumed_at,
    }
