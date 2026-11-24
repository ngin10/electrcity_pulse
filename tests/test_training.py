"""The retraining gate and interval calibration (pulse/training.py): plain pandas, no Spark or scikit-learn."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'databricks' / 'src'))

from pulse import training as tr  # noqa: E402

DAYS = pd.date_range('2024-01-01', '2026-10-31', freq='D')


def walk(err_sd=4.0, shift=0.0, seed=0, days=DAYS):
    rng = np.random.default_rng(seed)
    actual = rng.uniform(10, 75, len(days))
    wf = pd.DataFrame({'actual': actual, 'pred': actual - rng.normal(shift, err_sd, len(days))}, index=days)
    return wf[wf.index >= tr.WALK_START]


def test_conformal_half_width_matches_the_champions_calibration():
    # The same formula as eda/calibrate_intervals.py, which set the champion's 5.36: the 'higher' quantile at
    # ceil((n + 1) * 0.8) / n. On 10 scores that is the largest, one rank above the textbook ceil((n + 1) * 0.8)-th
    errors = np.arange(1, 11, dtype=float)
    assert tr.conformal_half_width(errors) == 10.0
    assert tr.conformal_half_width(np.append(errors, np.nan)) == 10.0
    # On a realistic sample the two agree to within one rank
    scores = np.random.default_rng(0).exponential(3, 674)
    textbook = np.sort(scores)[int(np.ceil(675 * 0.8)) - 1]
    ranked = np.sort(scores)
    assert abs(np.searchsorted(ranked, tr.conformal_half_width(scores)) - np.searchsorted(ranked, textbook)) <= 1


def test_rolling_coverage_uses_only_earlier_months():
    wf = walk()
    hits, days = tr.rolling_conformal_coverage(wf)
    assert days < len(wf)  # the first months lack 60 earlier errors
    assert 0.74 < hits / days < 0.86


def test_kupiec():
    assert tr.kupiec_p(80, 100) == pytest.approx(1.0)
    assert tr.kupiec_p(60, 100) < 0.001


def test_gate_passes_a_model_performing_as_validated():
    wf = walk(err_sd=3.6)  # MAE about 2.9
    latest = np.linspace(20, 60, 60)
    passed, checks = tr.gate(wf, 1000, latest, latest + 0.5)
    assert passed, [c for c in checks if not c['passed']]
    assert [c['check'] for c in checks] == ['training_days', 'walk_forward_mae', 'recent_mae', 'recent_bias', 'coverage', 'agreement',
                                            'valid_output']


@pytest.mark.parametrize('case, failing', [
    ('few_days', 'training_days'),
    ('worse', 'walk_forward_mae'),
    ('biased', 'recent_bias'),
    ('jumped', 'agreement'),
    ('out_of_range', 'valid_output'),
])
def test_gate_fails_for_the_right_reason(case, failing):
    wf, n, latest, champion = walk(err_sd=3.6), 1000, np.linspace(20, 60, 60), np.linspace(20, 60, 60)
    if case == 'few_days':
        n = 400
    elif case == 'worse':
        wf = walk(err_sd=6.0)
    elif case == 'biased':
        recent = wf.index > wf.index.max() - pd.Timedelta(days=tr.RECENT_DAYS)
        wf.loc[recent, 'pred'] -= 3.0
    elif case == 'jumped':
        latest = latest + 5.0
    elif case == 'out_of_range':
        latest = latest.copy()
        latest[0] = 104.0
    passed, checks = tr.gate(wf, n, latest, champion)
    assert not passed
    assert failing in {c['check'] for c in checks if not c['passed']}


def test_gate_catches_a_miscalibrated_interval():
    # Errors that grow over time: an interval calibrated on the calm past covers too little of the stormy present
    wf = walk(err_sd=1.0)
    wf['pred'] = wf['actual'] - np.random.default_rng(1).normal(0, np.linspace(1, 6, len(wf)))
    passed, checks = tr.gate(wf, 1000, np.full(60, 40.0), np.full(60, 40.0))
    assert {c['check'] for c in checks if not c['passed']} >= {'coverage'}
