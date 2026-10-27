import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checkpoint import load_checkpoint, save_checkpoint
from config import load_settings
from main import compute_range
from sink import PublishError, event_key, publish_to_kafka
from source import FetchError, date_windows, fetch_data
from validation import validate_record

GOOD = {
    'dataset': 'FUELHH', 'publishTime': '2026-10-21T23:00:00Z', 'startTime': '2026-10-21T22:30:00Z',
    'settlementDate': '2026-10-21', 'settlementPeriod': 48, 'fuelType': 'BIOMASS', 'generation': 1644,
}


# --- keys -------------------------------------------------------------------

def test_event_key_is_stable_and_ignores_non_key_fields():
    revised = {**GOOD, 'generation': 9999, 'publishTime': '2026-10-22T10:00:00Z'}
    assert event_key(GOOD) == event_key(revised)


def test_event_key_differs_by_fuel_type():
    assert event_key(GOOD) != event_key({**GOOD, 'fuelType': 'CCGT'})


# --- windowing --------------------------------------------------------------

def test_date_windows_respect_seven_day_inclusive_limit():
    windows = list(date_windows(date(2026, 10, 1), date(2026, 10, 15), 7))
    assert windows == [
        (date(2026, 10, 1), date(2026, 10, 7)),
        (date(2026, 10, 8), date(2026, 10, 14)),
        (date(2026, 10, 15), date(2026, 10, 15)),
    ]


def test_compute_range_first_run_and_resume(monkeypatch):
    monkeypatch.delenv('BACKFILL_START', raising=False)
    settings = load_settings()
    today = date(2026, 10, 25)
    assert compute_range(settings, None, today) == (date(2026, 10, 18), today)
    assert compute_range(settings, date(2026, 10, 25), today) == (date(2026, 10, 23), today)
    # After downtime, resume from the checkpoint rather than skipping the gap
    assert compute_range(settings, date(2026, 10, 10), today) == (date(2026, 10, 8), today)


def test_chunk_days_over_limit_rejected(monkeypatch):
    monkeypatch.setenv('CHUNK_DAYS', '8')
    with pytest.raises(ValueError):
        load_settings()


# --- fetching ---------------------------------------------------------------

class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError('not json')
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.params = None

    def get(self, url, params, timeout):
        self.params = params
        return self.response


def test_fetch_data_unwraps_data_and_sends_plain_dates():
    session = FakeSession(FakeResponse(200, {'data': [GOOD]}))
    records = fetch_data(session, 'http://x', date(2026, 10, 20), date(2026, 10, 21))
    assert records == [GOOD]
    assert session.params['settlementDateFrom'] == '2026-10-20'
    assert session.params['settlementDateTo'] == '2026-10-21'


def test_fetch_data_surfaces_4xx_body():
    session = FakeSession(FakeResponse(400, 'range must not exceed 7 days'))
    with pytest.raises(FetchError, match='7 days'):
        fetch_data(session, 'http://x', date(2026, 10, 1), date(2026, 10, 20))


def test_fetch_data_rejects_non_json():
    with pytest.raises(FetchError):
        fetch_data(FakeSession(FakeResponse(200, '<html>')), 'http://x', date(2026, 10, 1), date(2026, 10, 1))


# --- validation -------------------------------------------------------------

def test_valid_record_passes():
    assert validate_record(GOOD) == []
    assert validate_record({**GOOD, 'generation': -500}) == []


@pytest.mark.parametrize('field,value', [
    ('settlementDate', '21/10/2026'),
    ('settlementPeriod', 0),
    ('settlementPeriod', 51),
    ('settlementPeriod', True),
    ('fuelType', ''),
    ('generation', None),
    ('generation', '100'),
])
def test_invalid_fields_rejected(field, value):
    assert validate_record({**GOOD, field: value})


# --- checkpoint -------------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / 'state' / 'cp.json'
    assert load_checkpoint(path) is None
    save_checkpoint(str(path), date(2026, 10, 25))
    assert load_checkpoint(path) == date(2026, 10, 25)


# --- publishing -------------------------------------------------------------

class FakeMessage:
    def __init__(self, topic, key):
        self._topic, self._key = topic, key

    def topic(self):
        return self._topic

    def key(self):
        return self._key


class FakeProducer:
    def __init__(self, fail_keys=(), buffer_errors=0, leftover=0):
        self.sent = []
        self.pending = []
        self.fail_keys = set(fail_keys)
        self.buffer_errors = buffer_errors
        self.leftover = leftover

    def produce(self, topic, key, value, callback):
        if self.buffer_errors:
            self.buffer_errors -= 1
            raise BufferError
        self.sent.append((topic, key, json.loads(value)))
        self.pending.append((topic, key, callback))

    def poll(self, timeout):
        while self.pending:
            topic, key, callback = self.pending.pop(0)
            callback('boom' if key in self.fail_keys else None, FakeMessage(topic, key))

    def flush(self, timeout):
        self.poll(0)
        return self.leftover


W = (date(2026, 10, 21), date(2026, 10, 21))


def test_publish_routes_valid_and_invalid():
    producer = FakeProducer()
    bad = {**GOOD, 'fuelType': None}
    published, invalid = publish_to_kafka(producer, 't', 'dlq', [GOOD, bad], *W)
    assert (published, invalid) == (1, 1)
    topic, key, value = producer.sent[0]
    assert topic == 't' and key == event_key(GOOD)
    assert value['payload'] == GOOD and value['schema_version'] == 1
    assert value['fetch_window'] == {'from': '2026-10-21', 'to': '2026-10-21'}
    assert producer.sent[1][0] == 'dlq' and producer.sent[1][2]['raw'] == bad


def test_publish_retries_on_buffer_error():
    producer = FakeProducer(buffer_errors=2)
    publish_to_kafka(producer, 't', 'dlq', [GOOD], *W)
    assert len(producer.sent) == 1


def test_publish_raises_on_delivery_failure():
    with pytest.raises(PublishError, match='failed delivery'):
        publish_to_kafka(FakeProducer(fail_keys={event_key(GOOD)}), 't', 'dlq', [GOOD], *W)


def test_publish_raises_when_flush_leaves_messages():
    with pytest.raises(PublishError, match='undelivered'):
        publish_to_kafka(FakeProducer(leftover=3), 't', 'dlq', [GOOD], *W)
