"""Build bronze-shaped DataFrames from fixture responses, the way the landing files hold them."""
import json
from datetime import date
from functools import reduce
from pathlib import Path

from pyspark.sql import functions as F

FIXTURES = Path(__file__).parent / 'fixtures'

BATCH_SCHEMA = ('raw_value STRING, http_status BIGINT, request_url STRING, fetched_at STRING, '
                'source_file STRING, ingest_date DATE')
KAFKA_SCHEMA = ('raw_value STRING, kafka_partition BIGINT, kafka_offset BIGINT, kafka_key STRING, '
                'source_file STRING, ingest_date DATE')


def rows_frame(spark, rows, schema):
    """A DataFrame of rows built from literals in the JVM.

    spark.createDataFrame(list) needs a Python worker process, which can't start on some Windows setups; plain
    literals don't, and give exactly the same DataFrame.
    """
    names = [part.strip().split(' ')[0] for part in schema.split(',')]
    types = [part.strip().split(' ')[1] for part in schema.split(',')]
    frames = [spark.range(1).select(*[F.lit(value).cast(t).alias(n) for value, n, t in zip(row, names, types)])
              for row in rows]
    return reduce(lambda a, b: a.unionByName(b), frames)


def fixture_text(name):
    return (FIXTURES / f'{name}.json').read_text(encoding='utf-8')


def fixture_json(name):
    return json.loads(fixture_text(name))


def batch_bronze(spark, *bodies, http_status=200, fetched_at='2026-11-06T10:00:00.123456+00:00'):
    """One bronze row per response body, as batch ingest lands them."""
    rows = [(body, http_status, f'https://api.example/{i}', fetched_at, f'gs://landing/t/part-{i}.parquet',
             date(2026, 11, 6)) for i, body in enumerate(bodies)]
    return rows_frame(spark, rows, BATCH_SCHEMA)


def envelope(record, schema_version=1):
    """A Kafka message as the producer publishes it (envelope.py)."""
    return json.dumps({'schema_version': schema_version, 'source': 'elexon-bmrs', 'dataset': 'FUELHH',
                       'ingested_at': '2026-11-06T10:00:00+00:00',
                       'fetch_window': {'from': '2026-11-01', 'to': '2026-11-01'}, 'payload': record})


def kafka_bronze(spark, *messages):
    rows = [(message, 0, offset, f'key-{offset}', 'gs://landing/fuelhh/part-0.parquet', date(2026, 11, 6))
            for offset, message in enumerate(messages)]
    return rows_frame(spark, rows, KAFKA_SCHEMA)


def as_strings(df, *columns):
    """Rows as tuples of strings, formatted by Spark in UTC (collect() would convert to the machine's zone)."""
    return [tuple(row) for row in df.selectExpr(*[f'CAST({c} AS STRING)' for c in columns]).collect()]
