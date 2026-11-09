"""Checkpoints: the last date fully stored, so the next run knows where to continue.

A small JSON file such as {"last_settlement_date": "2026-11-04"}. Used by the producer and, one file per
source, by batch ingest. Callers save it only after a window's data is safely stored.
"""
import json
import os
from datetime import date


def load_checkpoint(path):
    """Return the last settlement date that was fully delivered, or None on first run."""
    try:
        with open(path, encoding='utf-8') as f:
            return date.fromisoformat(json.load(f)['last_settlement_date'])
    except FileNotFoundError:
        return None


def save_checkpoint(path, last_settlement_date: date):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # Write a temporary file first, and force it onto the disk...
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump({'last_settlement_date': last_settlement_date.isoformat()}, f)
        f.flush()
        os.fsync(f.fileno())
    # ...then swap it in: a crash mid-write never leaves a corrupt checkpoint
    os.replace(tmp_path, path)
