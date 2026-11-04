import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.memory import MemoryFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consumer.__main__ import ConsumerError, process_batch
from consumer.config import load_consumer_settings
from consumer.source import landing_row
from consumer.store import LANDING_TABLE, append_rows, table_dir

GOOD = {
    'dataset': 'FUELHH', 'publishTime': '2026-10-21T23:00:00Z', 'startTime': '2026-10-21T22:30:00Z',
    'settlementDate': '2026-10-21', 'settlementPeriod': 48, 'fuelType': 'BIOMASS', 'generation': 1644,
}
CONSUMED_AT = '2026-10-22T01:00:00+00:00'


def envelope(record, schema_version=1, ingested_at='2026-10-22T00:00:00+00:00'):
    return json.dumps({
        'schema_version': schema_version, 'source': 'elexon-bmrs', 'dataset': 'FUELHH',
        'ingested_at': ingested_at,
        'fetch_window': {'from': '2026-10-21', 'to': '2026-10-21'}, 'payload': record,
    }).encode('utf-8')


class FakeError:
    def __init__(self, fatal):
        self._fatal = fatal

    def fatal(self):
        return self._fatal

    def __str__(self):
        return 'broker error'


class FakeMessage:
    def __init__(self, value, offset=0, partition=0, error=None, key=b'k'):
        self._value = value
        self._offset = offset
        self._partition = partition
        self._error = error
        self._key = key

    def value(self):
        return self._value

    def key(self):
        return self._key

    def timestamp(self):
        return 1, 1761094800000

    def offset(self):
        return self._offset

    def partition(self):
        return self._partition

    def topic(self):
        return 'elexon.fuelhh'

    def error(self):
        return self._error


class FakeConsumer:
    def __init__(self):
        self.commits = 0

    def commit(self, asynchronous=True):
        assert asynchronous is False
        self.commits += 1


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv('LANDING_PATH', str(tmp_path))
    return load_consumer_settings()


def read_table(landing_path, table):
    return pd.read_parquet(table_dir(landing_path, table))


# --- landing rows -----------------------------------------------------------

def test_landing_row_keeps_raw_value_and_kafka_metadata():
    value = envelope(GOOD)
    row = landing_row(FakeMessage(value, offset=7, partition=2), CONSUMED_AT)
    assert row['raw_value'] == value.decode('utf-8')
    assert (row['kafka_topic'], row['kafka_partition'], row['kafka_offset']) == ('elexon.fuelhh', 2, 7)
    assert row['kafka_key'] == 'k'
    assert row['kafka_timestamp_ms'] == 1761094800000
    assert row['consumed_at'] == CONSUMED_AT


def test_landing_row_does_not_judge_the_value():
    # Checking is the lakehouse's job: garbage lands as-is, and undecodable bytes become escapes
    assert landing_row(FakeMessage(b'garbage'), CONSUMED_AT)['raw_value'] == 'garbage'
    assert landing_row(FakeMessage(b'\xff{'), CONSUMED_AT)['raw_value'] == '\\xff{'


# --- storage ----------------------------------------------------------------

def test_landing_is_append_only(tmp_path):
    first = landing_row(FakeMessage(envelope(GOOD), offset=1), CONSUMED_AT)
    revised = landing_row(FakeMessage(envelope({**GOOD, 'generation': 1700}), offset=2), CONSUMED_AT)
    append_rows(str(tmp_path), LANDING_TABLE, [first])
    append_rows(str(tmp_path), LANDING_TABLE, [revised])

    df = read_table(str(tmp_path), LANDING_TABLE)
    assert sorted(df['kafka_offset']) == [1, 2]


def test_landing_partitioned_by_ingest_date(tmp_path):
    rows = [landing_row(FakeMessage(envelope(GOOD)), consumed_at)
            for consumed_at in (CONSUMED_AT, '2026-10-23T09:00:00+00:00')]
    assert len(append_rows(str(tmp_path), LANDING_TABLE, rows)) == 2

    df = read_table(str(tmp_path), LANDING_TABLE)
    assert sorted(df['ingest_date'].astype(str)) == ['2026-10-22', '2026-10-23']


def test_every_file_has_the_same_schema(tmp_path):
    with_ts = landing_row(FakeMessage(envelope(GOOD), offset=1), CONSUMED_AT)
    # A message with no key or timestamp must not change the column types of its file
    without = {**landing_row(FakeMessage(envelope(GOOD), offset=2, key=None), CONSUMED_AT),
               'kafka_timestamp_ms': None}
    paths = append_rows(str(tmp_path), LANDING_TABLE, [with_ts]) + append_rows(str(tmp_path), LANDING_TABLE, [without])
    schemas = [pq.read_schema(path) for path in paths]
    assert schemas[0].equals(schemas[1])


def test_bucket_urls_use_forward_slashes():
    assert table_dir('gs://pulse-lake/landing', LANDING_TABLE) == 'gs://pulse-lake/landing/fuelhh'
    assert table_dir('gs://pulse-lake/landing/', LANDING_TABLE) == 'gs://pulse-lake/landing/fuelhh'


class FakeGCS(MemoryFileSystem):
    """In-memory stand-in for gcsfs, registered for gs:// paths, with create-only access like the consumer's."""
    protocol = ('gs',)
    store = {}
    pseudo_dirs = ['']

    def makedirs(self, path, exist_ok=False):
        # gcsfs lists the bucket to make a folder, which Storage Object Creator doesn't allow
        raise OSError(f'Forbidden: {path} (storage.objects.list denied)')


def test_writes_gs_paths_through_fsspec_without_a_temp_file(monkeypatch):
    # The write must not depend on pyarrow's own GCS support, which many builds (Anaconda's) lack: given a
    # gs:// path, pandas would try it first and fail with "Arrow compiled without GCS support"
    monkeypatch.setitem(importlib.import_module('fsspec.registry')._registry, 'gs', FakeGCS)
    FakeGCS.store.clear()
    [path] = append_rows('gs://pulse-lake/landing', LANDING_TABLE,
                         [landing_row(FakeMessage(envelope(GOOD)), CONSUMED_AT)])

    assert path.startswith('gs://pulse-lake/landing/fuelhh/ingest_date=2026-10-22/part-')
    fs = FakeGCS()
    # Exactly the one file: no temp file left beside it
    assert fs.find('gs://pulse-lake') == [path]
    with fs.open(path, 'rb') as f:
        assert len(pd.read_parquet(f)) == 1


# --- batches ----------------------------------------------------------------

def test_process_batch_lands_everything_and_commits(settings):
    consumer = FakeConsumer()
    messages = [
        FakeMessage(envelope(GOOD), offset=1),
        FakeMessage(b'garbage', offset=2),
        FakeMessage(None, offset=3),  # tombstone
        FakeMessage(None, error=FakeError(fatal=False)),
    ]
    assert process_batch(consumer, settings, messages) == 2
    assert consumer.commits == 1
    landed = read_table(settings.landing_path, LANDING_TABLE).sort_values('kafka_offset')
    assert list(landed['kafka_offset']) == [1, 2]


def test_fatal_error_does_not_commit(settings):
    consumer = FakeConsumer()
    with pytest.raises(ConsumerError):
        process_batch(consumer, settings, [FakeMessage(None, error=FakeError(fatal=True))])
    assert consumer.commits == 0
