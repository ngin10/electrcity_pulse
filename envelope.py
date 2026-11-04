from datetime import datetime, timezone

SCHEMA_VERSION = 1
SOURCE = 'elexon-bmrs'


def build_envelope(record, window_start, window_end):
    """Wrap an API record as it is, unchecked; data-quality checks run in the lakehouse."""
    return {
        'schema_version': SCHEMA_VERSION,
        'source': SOURCE,
        'dataset': record.get('dataset', 'FUELHH') if isinstance(record, dict) else 'FUELHH',
        'ingested_at': datetime.now(timezone.utc).isoformat(),
        'fetch_window': {'from': window_start.isoformat(), 'to': window_end.isoformat()},
        'payload': record,
    }
