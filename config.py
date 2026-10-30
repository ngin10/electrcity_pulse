import os
from dataclasses import dataclass
from datetime import date


def _env(name, default=None):
    value = os.getenv(name)
    return value if value not in (None, '') else default


@dataclass(frozen=True)
class Settings:
    api_url: str
    kafka_bootstrap: str
    topic: str
    dlq_topic: str
    topic_partitions: int
    topic_replication: int
    checkpoint_path: str
    backfill_start: date | None
    initial_lookback_days: int
    revision_lookback_days: int
    chunk_days: int
    poll_interval_seconds: int
    log_level: str


def load_settings():
    backfill = _env('BACKFILL_START')
    chunk_days = int(_env('CHUNK_DAYS', 7))
    if not 1 <= chunk_days <= 7:
        # Elexon rejects ranges over 7 days (inclusive of both ends)
        raise ValueError('CHUNK_DAYS must be between 1 and 7')

    return Settings(
        api_url=_env('API_URL', 'https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH'),
        kafka_bootstrap=_env('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        topic=_env('KAFKA_TOPIC', 'elexon.fuelhh'),
        dlq_topic=_env('KAFKA_DLQ_TOPIC', 'elexon.fuelhh.dlq'),
        topic_partitions=int(_env('KAFKA_TOPIC_PARTITIONS', 3)),
        topic_replication=int(_env('KAFKA_TOPIC_REPLICATION', 1)),
        checkpoint_path=_env('CHECKPOINT_PATH', 'state/checkpoint.json'),
        backfill_start=date.fromisoformat(backfill) if backfill else None,
        initial_lookback_days=int(_env('INITIAL_LOOKBACK_DAYS', 7)),
        revision_lookback_days=int(_env('REVISION_LOOKBACK_DAYS', 2)),
        chunk_days=chunk_days,
        poll_interval_seconds=int(_env('POLL_INTERVAL_SECONDS', 1800)),
        log_level=_env('LOG_LEVEL', 'INFO'),
    )
