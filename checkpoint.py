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
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump({'last_settlement_date': last_settlement_date.isoformat()}, f)
        f.flush()
        os.fsync(f.fileno())
    # Atomic swap so a crash mid-write never leaves a corrupt checkpoint
    os.replace(tmp_path, path)
