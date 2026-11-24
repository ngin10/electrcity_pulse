"""The feature store's building blocks: the leakage rules, the type preparation and the change fingerprint.

pulse/feature_store.py is plain pandas, so these run without Spark.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'databricks' / 'src'))

from pulse import feature_store as fs  # noqa: E402

TIME_COLUMNS = ('publish_time', 'start_time', 'created_at', 'event_start_time', 'event_end_time', 'period_start', 'period_end',
                'datetime_gmt', 'time')


def empty_silver():
    """Every table build() reads, empty, with the types the job hands it."""
    raw = {}
    for table, columns in fs.SILVER_COLUMNS.items():
        dtypes = {c: 'datetime64[ns]' if c.endswith('_date') or c in TIME_COLUMNS else 'float64' for c in columns}
        raw[table] = pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})
    return raw


def test_every_table_has_a_leakage_rule_and_a_column_list():
    assert set(fs.availability(empty_silver())) == set(fs.TABLES) == set(fs.SILVER_COLUMNS)
    assert set(fs.AVAILABILITY_NOTE) == set(fs.TABLES)


def test_before_cutoff_keeps_only_what_was_published_by_18_00():
    hours = pd.date_range('2026-11-01 00:00', '2026-11-01 23:00', freq='h')
    values = pd.Series(np.arange(24, dtype=float), index=hours)
    available = pd.Series(hours + pd.Timedelta(minutes=30), index=hours)
    # The 10:00 value was only published the next morning, after the cutoff
    available[pd.Timestamp('2026-11-01 10:00')] = pd.Timestamp('2026-11-02 09:00')

    means = fs.before_cutoff(values, available).mean()

    # Hours 0-17 count towards the next day's features, except the late 10:00 value; 18:00 onwards never does
    assert list(means.index) == [pd.Timestamp('2026-11-02')]
    assert means.iloc[0] == np.mean([h for h in range(18) if h != 10])


def test_truncate_drops_rows_published_after_the_cutoff():
    raw = empty_silver()
    raw['fuelhh'] = pd.DataFrame({
        'settlement_date': pd.to_datetime(['2026-11-01', '2026-11-01']), 'settlement_period': [30, 40],
        'fuel_type': ['WIND', 'WIND'], 'generation_mw': [5000.0, 6000.0],
        'publish_time': pd.to_datetime(['2026-11-01 15:30', '2026-11-01 20:30'])})
    as_known = fs.truncate(raw, pd.Timestamp('2026-11-01 18:00'))
    assert list(as_known['fuelhh']['settlement_period']) == [30]


def test_prepare_turns_spark_types_into_naive_utc_timestamps():
    spark_like = {'demand_outturn': pd.DataFrame({
        'settlement_date': [date(2026, 3, 29)],
        'publish_time': pd.Series(pd.to_datetime(['2026-03-29 01:30']), dtype='datetime64[us]').dt.tz_localize('UTC'),
        'initial_demand_outturn_mw': [25000.0]})}
    df = fs.prepare(spark_like)['demand_outturn']
    assert df['settlement_date'].dtype == 'datetime64[ns]'
    assert df['publish_time'].dtype == 'datetime64[ns]'
    assert df['publish_time'].iloc[0] == pd.Timestamp('2026-03-29 01:30')


def test_feature_hash_ignores_float_noise_but_not_real_changes():
    rows = pd.DataFrame({'a': [1.0, 2.0], 'b': [np.nan, 3.5]})
    base = fs.feature_hash(rows, ['a', 'b'])
    assert fs.feature_hash(rows + 1e-9, ['a', 'b']) == base          # last-bit noise is not a change
    changed = rows.copy()
    changed.loc[1, 'b'] = 3.6
    assert fs.feature_hash(changed, ['a', 'b'])[0] == base[0]
    assert fs.feature_hash(changed, ['a', 'b'])[1] != base[1]
    filled = rows.fillna(0.0)                                        # a missing value becoming known is a change
    assert fs.feature_hash(filled, ['a', 'b'])[0] != base[0]


def test_catalogue_has_one_row_per_feature_with_its_leakage_rule():
    meta = [{'feature': 'fc_wind_mean_mw', 'source': 'wind_solar_forecast', 'definition': 'Day-ahead wind', 'known': 'at the cutoff'},
            {'feature': 'cal_is_weekend', 'source': 'calendar', 'definition': 'Saturday or Sunday', 'known': 'at the cutoff'}]
    cat = fs.catalogue(meta)
    assert list(cat['feature']) == ['fc_wind_mean_mw', 'cal_is_weekend']
    assert cat.loc[0, 'available_from'] == fs.AVAILABILITY_NOTE['wind_solar_forecast']
    assert set(cat['feature_set_version']) == {fs.FEATURE_SET_VERSION}


def test_power_curve_is_a_share_of_rated_output():
    kmh = pd.Series([0.0, 10.8, 25.0, 43.2, 60.0, 90.0, 100.0])   # 0, 3, ~7, 12, ~17, 25, ~28 m/s
    p = fs.power_curve(kmh)
    assert p.between(0, 1).all()
    assert p.iloc[1] == 0.0 and p.iloc[3] == 1.0 and p.iloc[5] == 1.0   # cut-in, rated, still running at 25 m/s
    assert p.iloc[6] == 0.0                                              # cut-out above 25 m/s
