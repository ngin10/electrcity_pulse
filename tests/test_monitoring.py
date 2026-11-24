"""The monitoring statistics and grading rules (pulse/monitoring.py): plain pandas, no Spark."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'databricks' / 'src'))

from pulse import monitoring as mon  # noqa: E402

RNG = np.random.default_rng(3)
DAYS = pd.date_range('2026-10-01', periods=30, freq='D')


@pytest.fixture(autouse=True)
def fresh_rng():
    """Each test gets the same random draws whatever runs before it."""
    global RNG
    RNG = np.random.default_rng(3)


def test_grade_both_directions():
    assert [mon.grade(v, (0.1, 0.25)) for v in (0.05, 0.1, 0.3)] == ['ok', 'warn', 'alert']
    assert [mon.grade(p, (0.10, 0.05), higher_is_worse=False) for p in (0.5, 0.08, 0.01)] == ['ok', 'warn', 'alert']
    assert mon.grade(float('nan'), (1, 2)) == 'no_data'


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    ref = RNG.normal(0, 1, 2000)
    assert mon.psi(ref, RNG.normal(0, 1, 2000)) < 0.02
    assert mon.psi(ref, RNG.normal(1.5, 1, 2000)) > 0.25


def test_psi_counts_newly_missing_values_as_drift():
    ref = pd.Series(RNG.normal(size=1000))
    cur = pd.Series(RNG.normal(size=100))
    cur.iloc[:60] = np.nan  # a source stopped arriving
    assert mon.psi(ref, cur) > 0.25


def test_ks_matches_known_behaviour():
    a = RNG.normal(size=500)
    d_same, p_same = mon.ks_2samp(a, RNG.normal(size=500))
    d_diff, p_diff = mon.ks_2samp(a, RNG.normal(0.5, 1, 500))
    assert p_same > 0.01 and p_diff < 0.001 and d_diff > d_same
    d, p = mon.ks_2samp([1, 2, 3], [1, 2, 3])
    assert d == 0 and p == 1


def test_kupiec_accepts_nominal_coverage_and_rejects_a_bad_interval():
    assert mon.kupiec(80, 100)[1] == pytest.approx(1.0)
    lr, p = mon.kupiec(60, 100)  # 60% coverage for a nominal 80%
    assert p < 0.001
    # LR for 60 hits of 100: 2 * [40 ln(0.4/0.2) + 60 ln(0.6/0.8)]
    assert lr == pytest.approx(2 * (40 * math.log(2) + 60 * math.log(0.75)))
    assert mon.kupiec(100, 100)[1] < 0.001  # never missing is also wrong


def scored(err, width=5.36):
    pred = np.full(len(err), 40.0)
    return pd.DataFrame({'actual': pred + err, 'predicted_share': pred, 'lower_80': pred - width, 'upper_80': pred + width},
                        index=DAYS[:len(err)])


def test_accuracy_ok_for_a_model_performing_as_validated():
    rows = {r['metric']: r for r in mon.accuracy(scored(RNG.normal(0, 4.0, 30)))}
    assert rows['mae']['status'] == 'ok' and rows['mae']['n'] == 30
    assert 0.6 < rows['coverage_80']['value'] < 1.0


def test_accuracy_alerts_on_a_degraded_model():
    rows = {r['metric']: r for r in mon.accuracy(scored(RNG.normal(8, 3, 30)))}
    assert rows['mae']['status'] == 'alert'
    assert rows['bias']['status'] == 'alert'
    assert rows['kupiec_p']['status'] == 'alert'


def test_coverage_is_graded_against_a_band_on_both_sides():
    assert [mon.grade_band(v) for v in (0.80, 0.68, 0.60, 0.91, 0.95)] == ['ok', 'warn', 'alert', 'warn', 'alert']
    too_wide = {r['metric']: r for r in mon.accuracy(scored(RNG.normal(0, 1, 30), width=20))}
    assert too_wide['coverage_80']['status'] == 'alert' and too_wide['coverage_80']['value'] == 1.0


def test_accuracy_waits_for_enough_days():
    rows = mon.accuracy(scored(RNG.normal(20, 1, 5)))
    assert {r['status'] for r in rows} == {'insufficient_data'}


def test_accuracy_ignores_days_without_an_outcome():
    s = scored(RNG.normal(0, 3, 30))
    s.iloc[-5:, 0] = np.nan
    assert mon.accuracy(s)[0]['n'] == 25


def test_feature_drift_flags_only_the_feature_that_moved():
    ref = pd.DataFrame({'stable': RNG.normal(size=365), 'moved': RNG.normal(size=365)})
    cur = pd.DataFrame({'stable': RNG.normal(size=30), 'moved': RNG.normal(3, 1, 30)}, index=DAYS)
    rows = {(r['subject'], r['metric']): r for r in mon.feature_drift(ref, cur, ['stable', 'moved'])}
    assert rows[('moved', 'psi')]['status'] == 'alert' and rows[('moved', 'ks_stat')]['status'] == 'alert'
    assert rows[('stable', 'psi')]['status'] in ('ok', 'warn')
    assert rows[('stable', 'psi')]['window_start'] == DAYS[0]


def test_prediction_drift():
    rng = np.random.default_rng(11)
    rows = mon.prediction_drift(rng.normal(45, 14, 365), pd.Series(rng.normal(70, 3, 30), index=DAYS))
    assert rows[0]['status'] == 'alert' and rows[0]['details'].startswith('mean 7') and 'now vs 4' in rows[0]['details']


def test_drift_thresholds_allow_for_small_samples():
    rng = np.random.default_rng(5)
    ref = rng.normal(size=365)
    warn, alert = mon.drift_thresholds(ref, 30)['psi']
    assert alert > mon.PSI[1]  # 30 days are noisier than the textbook rule assumes
    assert mon.drift_thresholds(ref, 5000, reps=50)['psi'][1] == pytest.approx(mon.PSI[1])  # big samples fall back to it
    # Thirty new days from the same distribution rarely alert. Not quite 1%: the noise is resampled from the
    # reference itself, which leaves out the reference's own sampling error
    alerts = sum(mon.psi(ref, rng.normal(size=30)) >= alert for _ in range(200))
    assert alerts <= 10


def test_drift_thresholds_allow_for_runs_of_similar_days():
    # A slowly wandering series (like a rolling mean): 30 consecutive days are far from a random sample of the year
    walk = pd.Series(np.cumsum(RNG.normal(size=365)), index=pd.date_range('2025-01-01', periods=365, freq='D'))
    blocked = mon.drift_thresholds(walk, 30, block=7)
    independent = mon.drift_thresholds(walk, 30, block=1)
    assert blocked['psi'][1] > independent['psi'][1] and blocked['ks_stat'][1] > independent['ks_stat'][1]


def test_freshness_alerts_when_a_source_misses_the_cutoff():
    served = pd.DataFrame({'wx_a': [1.0, 1.0, 1.0, np.nan], 'wx_b': [1.0, 1.0, 1.0, np.nan], 'fc_a': [1.0, 2.0, 3.0, 4.0]},
                          index=DAYS[:4])
    rows = {r['subject']: r for r in mon.freshness(served, {'wx_a': 'weather', 'wx_b': 'weather', 'fc_a': 'wind_forecast',
                                                             'unused': 'other'})}
    assert rows['weather']['status'] == 'alert' and rows['weather']['value'] == 0
    assert rows['weather']['window_start'] == DAYS[3]  # judged on the latest day
    assert rows['wind_forecast']['status'] == 'ok'
    assert 'other' not in rows


def test_freshness_accepts_a_source_that_is_always_partly_missing():
    # NESO publishes weeks behind: 3 of its 5 features are present every day, and that is normal
    served = pd.DataFrame({f'neso_{i}': [1.0] * 10 if i < 3 else [np.nan] * 10 for i in range(5)}, index=DAYS[:10])
    (row,) = mon.freshness(served, {c: 'neso_demand' for c in served})
    assert row['status'] == 'ok' and row['value'] == 1.0 and '60% present' in row['details']


def test_seasonal_reference_takes_the_same_time_of_year_across_new_year():
    reference = pd.DataFrame({'x': range(730)}, index=pd.date_range('2024-01-01', periods=730, freq='D'))
    current = pd.date_range('2026-12-20', periods=20, freq='D')  # 20 Dec to 8 Jan
    days = mon.seasonal_reference(reference, current, margin_days=5).index
    assert set(days.month) == {12, 1}
    assert days.min() >= pd.Timestamp('2024-01-01') and pd.Timestamp('2024-07-01') not in days
    # 15 Dec to 13 Jan, give or take a day in leap years (day of year shifts by one after February)
    assert all((d.month == 12 and d.day >= 14) or (d.month == 1 and d.day <= 14) for d in days)


def test_feature_drift_skips_calendar_features():
    ref = pd.DataFrame({'cal_doy_sin': RNG.normal(size=100), 'x': RNG.normal(size=100)})
    cur = pd.DataFrame({'cal_doy_sin': RNG.normal(5, 1, 30), 'x': RNG.normal(size=30)}, index=DAYS)
    assert {r['subject'] for r in mon.feature_drift(ref, cur, ['cal_doy_sin', 'x'])} == {'x'}


def test_weather_quality_flags():
    X = pd.DataFrame({'wx_a': [1.0, np.nan, np.nan], 'wx_b': [1.0, 2.0, np.nan], 'fc': [1.0, 1.0, 1.0]})
    assert mon.weather_quality(X).tolist() == ['ok', 'weather_partial', 'weather_missing']
    assert mon.weather_quality(X[['fc']]).tolist() == ['ok'] * 3


def test_rows_have_one_shape_and_summary_picks_the_worst():
    rows = mon.accuracy(scored(RNG.normal(8, 3, 30))) + mon.prediction_drift(RNG.normal(45, 14, 365), RNG.normal(45, 14, 30))
    assert all(list(r) == mon.ROW_COLUMNS for r in rows)
    s = mon.summary(rows)
    assert s['accuracy']['worst'] == 'alert'
    assert s['prediction_drift']['worst'] in ('ok', 'warn')
