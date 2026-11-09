"""Consumer: the landing table's column types, and the hand-off to the shared writer in landing.py."""
import landing
from landing import table_dir  # noqa: F401  (re-exported for callers of the consumer)

# The landing folder: <LANDING_PATH>/fuelhh/
LANDING_TABLE = 'fuelhh'

# Fixed types so every file has the same schema, whatever values a given batch happens to contain.
# These are the columns around the data; raw_value's content is never touched.
# 'Int64' (capital I) allows missing values: not every message has a Kafka timestamp.
DTYPES = {
    LANDING_TABLE: {
        'raw_value': 'string', 'kafka_topic': 'string', 'kafka_partition': 'int64', 'kafka_offset': 'int64',
        'kafka_key': 'string', 'kafka_timestamp_ms': 'Int64', 'consumed_at': 'string',
    },
}


def append_rows(landing_path, table, rows):
    """Append Kafka rows to a landing table, in folders by the day they were consumed; returns the paths.

    A batch replayed after a crash is appended again, so readers deduplicate on the record key.
    """
    return landing.append_rows(landing_path, table, rows, DTYPES[table], 'consumed_at')
