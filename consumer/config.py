import os
from dataclasses import dataclass


def _env(name, default=None):
    value = os.getenv(name)
    return value if value not in (None, '') else default


@dataclass(frozen=True)
class ConsumerSettings:
    kafka_bootstrap: str
    topic: str
    group_id: str
    landing_path: str
    batch_max_messages: int
    batch_timeout_seconds: float
    log_level: str


def load_consumer_settings():
    return ConsumerSettings(
        kafka_bootstrap=_env('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        topic=_env('KAFKA_TOPIC', 'elexon.fuelhh'),
        group_id=_env('CONSUMER_GROUP_ID', 'pulse-fuelhh-bronze'),
        landing_path=_env('LANDING_PATH', 'data/landing'),
        batch_max_messages=int(_env('BATCH_MAX_MESSAGES', 500)),
        batch_timeout_seconds=float(_env('BATCH_TIMEOUT_SECONDS', 5)),
        log_level=_env('LOG_LEVEL', 'INFO'),
    )
