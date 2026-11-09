"""Producer step 2: wrap each Elexon record in an envelope before it goes to Kafka.

The record itself is not changed or checked. The envelope only adds facts known at fetch time, which can't
be recovered later: the format version, where the data came from, when it was fetched and which window.

One message on the topic looks like:
    {"schema_version": 1, "source": "elexon-bmrs", "dataset": "FUELHH",
     "ingested_at": "2026-11-04T18:51:27+00:00", "fetch_window": {"from": "2026-10-23", "to": "2026-10-29"},
     "payload": {"settlementDate": "2026-10-23", "settlementPeriod": 1, "fuelType": "WIND", ...}}
"""
from datetime import datetime, timezone

# Bump when the envelope's own layout changes, so readers know how to parse each message
SCHEMA_VERSION = 1
# Which API the data came from, for when several sources share a table downstream
SOURCE = 'elexon-bmrs'


def build_envelope(record, window_start, window_end):
    """Wrap an API record as it is, unchecked; data-quality checks run in the lakehouse."""
    return {
        'schema_version': SCHEMA_VERSION,
        'source': SOURCE,
        # The dataset name, copied from the record when it has one.
        #
        # Why check isinstance(record, dict) when source.py already checked the response? source.py only
        # checks that the response is a list; it doesn't look at the items in it. A list can hold anything,
        # e.g. [{...}, "maintenance", null], and record.get() exists only on a dict: on a string or None it
        # raises AttributeError. main.py doesn't catch that error, so the producer would crash, and since the
        # checkpoint wouldn't move, every later run would fetch the same window, hit the same item and crash
        # again: one bad record would stop the pipeline for good.
        # With the check, a non-dict record gets the default dataset and is published like any other, and
        # silver rejects it. This only avoids a crash; the record itself isn't changed or filtered.
        'dataset': record.get('dataset', 'FUELHH') if isinstance(record, dict) else 'FUELHH',
        # When we fetched it (UTC); lets readers keep the most recent revision of a record
        'ingested_at': datetime.now(timezone.utc).isoformat(),
        # The settlement dates requested, for tracing a message back to its API call
        'fetch_window': {'from': window_start.isoformat(), 'to': window_end.isoformat()},
        # The Elexon record exactly as received, whatever it is
        'payload': record,
    }
