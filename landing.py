"""Writes landing files: append-only Parquet under <landing_path>/<table>/ingest_date=YYYY-MM-DD/.

Shared by the Kafka consumer and the batch sources. landing_path is a local folder or a bucket URL such as
gs://pulse-lake/landing.
"""
import logging
import os
import uuid
from collections import defaultdict

import fsspec
import pandas as pd

log = logging.getLogger(__name__)

PARTITION_COLUMN = 'ingest_date'

# gcsfs (loaded on the first gs:// write) defaults to experimental support for hierarchical-namespace buckets,
# which probes every bucket with a list call first; a create-only role can't list, so each write would log a
# 403 before falling back to the standard client. Set it to true yourself to opt back in.
os.environ.setdefault('GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT', 'false')


def _is_url(path):
    # gs://bucket/... (or s3://, memory://) rather than a local folder
    return '://' in path


def _join(base, *parts):
    # os.path.join would put backslashes into gs:// paths on Windows
    if _is_url(base):
        return '/'.join([base.rstrip('/'), *parts])
    return os.path.join(base, *parts)


def table_dir(landing_path, table):
    """Folder of a table; landing_path is a local folder or a bucket URL such as gs://pulse-lake/landing."""
    return _join(landing_path, table)


def _write_atomic(df, path):
    if _is_url(path):
        # An object in GCS (or S3) only becomes visible once its upload completes, so readers never see
        # a partial file; there is no rename to do, and object stores have no atomic rename anyway.
        # Opened through fsspec (gcsfs for gs://): given a URL, pandas would try pyarrow's own GCS support
        # first, and many pyarrow builds (Anaconda's included) are compiled without it.
        # No auto_mkdir: folders in a bucket are just name prefixes, and creating them would need list
        # access, more than a create-only role (Storage Object Creator) allows
        with fsspec.open(path, 'wb', auto_mkdir=False) as f:
            df.to_parquet(f, index=False)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f'{path}.tmp'
    df.to_parquet(tmp_path, index=False)
    # Readers never see a half-written file
    os.replace(tmp_path, path)


def append_rows(landing_path, table, rows, dtypes, timestamp_column):
    """Append rows to a landing table as new Parquet files, one per ingest date; returns the paths written.

    dtypes fixes every column's type, so every file has the same schema whatever values a batch happens to
    contain (Auto Loader reads these files, and a column that changes type between files breaks ingestion).
    timestamp_column holds an ISO UTC timestamp whose date picks the ingest_date folder.
    Landing is append-only: nothing is updated or deleted. Readers deduplicate anything appended twice.
    """
    by_date = defaultdict(list)
    for row in rows:
        # Hive-style folders by the UTC day the row was stored, so readers can prune by date
        by_date[row[timestamp_column][:10]].append(row)

    paths = []
    for ingest_date, date_rows in sorted(by_date.items()):
        # columns= keeps exactly the expected columns, in order; astype fixes their types.
        # The partition column lives only in the folder name, as Hive-style readers expect
        df = pd.DataFrame(date_rows, columns=list(dtypes)).astype(dtypes)
        # A new, randomly named file every time: nothing existing is ever overwritten
        path = _join(table_dir(landing_path, table), f'{PARTITION_COLUMN}={ingest_date}',
                     f'part-{uuid.uuid4().hex}.parquet')
        _write_atomic(df, path)
        log.debug('Wrote %d rows to %s', len(df), path)
        paths.append(path)
    return paths
