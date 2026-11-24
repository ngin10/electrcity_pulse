"""The status page's logic: honest intervals, live-over-backtest precedence, gaps kept, badge grading."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'databricks' / 'tools'))
import model_status as ms  # noqa: E402


def frame(start, n, **cols):
    return pd.DataFrame(cols, index=pd.date_range(start, periods=n, freq='D'))


def test_band_uses_only_earlier_months():
    rng = np.random.default_rng(0)
    wf = frame('2025-01-01', 120, actual=rng.normal(50, 5, 120), pred=np.full(120, 50.0))
    banded = ms.monthly_conformal_band(wf, min_history=50)
    # January and February have under 50 earlier days: no interval rather than an in-sample one
    assert banded.loc[:'2025-02-28', 'lower'].isna().all()
    march = banded.loc['2025-03-01':'2025-03-31']
    assert march['lower'].notna().all()
    earlier = (wf['actual'] - wf['pred']).abs().loc[:'2025-02-28']
    q = ms.conformal_half_width(earlier)
    assert march['upper'].iloc[0] == pytest.approx(50 + q)
    # Changing a later month's errors leaves March's interval unchanged
    later = wf.copy()
    later.loc['2025-04-01':, 'actual'] += 40
    assert ms.monthly_conformal_band(later, min_history=50).loc['2025-03-15', 'upper'] == pytest.approx(march['upper'].iloc[14])


def test_band_is_clipped_to_0_100():
    wf = frame('2025-01-01', 100, actual=np.r_[np.full(50, 0.0), np.full(50, 100.0)], pred=np.r_[np.full(50, 99.0), np.full(50, 1.0)])
    banded = ms.monthly_conformal_band(wf, min_history=30).dropna()
    assert banded['lower'].min() >= 0 and banded['upper'].max() <= 100


def test_combine_prefers_live_and_keeps_gaps():
    wf = frame('2026-11-01', 5, pred=[10.0] * 5, lower=[5.0] * 5, upper=[15.0] * 5)
    live = frame('2026-11-04', 1, pred=[30.0], lower=[25.0], upper=[35.0])
    live = pd.concat([live, frame('2026-11-10', 1, pred=[40.0], lower=[35.0], upper=[45.0])])
    outcomes = pd.Series(20.0, index=pd.date_range('2026-11-01', '2026-11-09'))
    h = ms.combine(wf, live, outcomes)
    assert h.loc['2026-11-04', 'pred'] == 30 and h.loc['2026-11-04', 'source'] == 'live'
    assert h.loc['2026-11-03', 'source'] == 'walk-forward'
    # Outcome but no forecast: kept, forecast left empty (not filled)
    assert np.isnan(h.loc['2026-11-07', 'pred']) and h.loc['2026-11-07', 'actual'] == 20
    # Forecast with no outcome yet: kept
    assert h.loc['2026-11-10', 'pred'] == 40 and np.isnan(h.loc['2026-11-10', 'actual'])


def test_rolling_waits_for_enough_days():
    s = pd.Series(1.0, index=pd.date_range('2026-01-01', periods=25))
    r = ms.rolling(s, days=30, min_periods=20)
    assert r.iloc[:19].isna().all() and r.iloc[19:].eq(1.0).all()


@pytest.mark.parametrize('statuses, expected', [
    (['ok', 'warn', 'ok'], 'warn'),
    (['ok', 'alert', 'warn'], 'alert'),
    (['insufficient_data', 'ok'], 'ok'),
    (['insufficient_data'], None),
    ([], None),
])
def test_worst_status(statuses, expected):
    assert ms.worst_status(statuses) == expected


def test_mae_status_uses_monitoring_levels():
    warn, alert = ms.MAE_POINTS
    assert ms.mae_status(warn) == 'ok'
    assert ms.mae_status((warn + alert) / 2) == 'warn'
    assert ms.mae_status(alert + 0.01) == 'alert'
    assert ms.mae_status(None) is None and ms.mae_status(float('nan')) is None


def test_summary_and_badges():
    days = pd.date_range('2026-10-01', periods=40)
    history = pd.DataFrame({'pred': 50.0, 'lower': 45.0, 'upper': 55.0, 'actual': np.r_[np.full(30, 52.0), np.full(10, 60.0)],
                            'source': ['walk-forward'] * 35 + ['live'] * 5}, index=days)
    monitoring = pd.DataFrame({'run_date': ['2026-11-24'] * 3, 'check_name': ['accuracy', 'freshness', 'freshness'],
                               'status': ['insufficient_data', 'ok', 'warn']})
    latest = {'target_date': '2026-11-10', 'predicted_share': 55.04, 'lower_80': 49.6, 'upper_80': 60.4}
    s = ms.summarise(history, monitoring, {'aliases': {'champion': '1', 'challenger': '2'}}, None, latest)
    # The last 30 days: 20 days off by 2, 10 days off by 10
    assert s['mae_30d'] == pytest.approx((20 * 2 + 10 * 10) / 30)
    assert s['mae_30d_days'] == {'live': 5, 'walk_forward': 25}
    assert s['monitoring'] == {'accuracy': None, 'freshness': 'warn'} and s['monitoring_worst'] == 'warn'
    b = ms.badges(s)
    assert b['model']['message'] == 'champion v1 · challenger v2'
    assert b['mae']['color'] == 'red'                       # 4.67 > the 4.5 alert level
    assert b['monitoring'] == {'schemaVersion': 1, 'label': 'monitoring', 'message': 'warn', 'color': 'yellow'}
    assert b['forecast']['message'] == '2026-11-10: 55.0%'


def test_badges_before_any_data():
    s = {'aliases': {'champion': '1'}, 'mae_30d': None, 'monitoring_worst': None, 'latest_forecast': None}
    b = ms.badges(s)
    assert b['model']['message'] == 'champion v1'
    assert b['mae']['message'] == 'n/a' and b['mae']['color'] == 'lightgrey'
    assert b['monitoring']['message'] == 'collecting data'
