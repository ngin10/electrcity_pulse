from datetime import date, datetime, timezone

SCHEMA_VERSION = 1
SOURCE = 'elexon-bmrs'

# 46 periods on the spring clock change, 50 in autumn
MAX_SETTLEMENT_PERIOD = 50


def validate_record(record):
    """Return a list of problems; empty means the record is publishable."""
    if not isinstance(record, dict):
        return ['record is not an object']

    errors = []

    settlement_date = record.get('settlementDate')
    try:
        date.fromisoformat(settlement_date)
    except (TypeError, ValueError):
        errors.append(f'invalid settlementDate: {settlement_date!r}')

    period = record.get('settlementPeriod')
    if not isinstance(period, int) or isinstance(period, bool) or not 1 <= period <= MAX_SETTLEMENT_PERIOD:
        errors.append(f'invalid settlementPeriod: {period!r}')

    fuel_type = record.get('fuelType')
    if not isinstance(fuel_type, str) or not fuel_type.strip():
        errors.append(f'invalid fuelType: {fuel_type!r}')

    # Negative generation is legitimate (e.g. interconnector exports)
    generation = record.get('generation')
    if not isinstance(generation, (int, float)) or isinstance(generation, bool):
        errors.append(f'invalid generation: {generation!r}')

    return errors


def build_envelope(record, window_start, window_end):
    return {
        'schema_version': SCHEMA_VERSION,
        'source': SOURCE,
        'dataset': record.get('dataset', 'FUELHH'),
        'ingested_at': datetime.now(timezone.utc).isoformat(),
        'fetch_window': {'from': window_start.isoformat(), 'to': window_end.isoformat()},
        'payload': record,
    }


def build_dead_letter(record, errors, window_start, window_end):
    return {
        'schema_version': SCHEMA_VERSION,
        'source': SOURCE,
        'failed_at': datetime.now(timezone.utc).isoformat(),
        'fetch_window': {'from': window_start.isoformat(), 'to': window_end.isoformat()},
        'errors': errors,
        'raw': record,
    }
