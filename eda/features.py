"""Feature engineering: one row per target day D, built from all 15 silver tables, using only what was known
at the forecast cutoff (18:00 UTC on D-1). The target is D's renewable share of GB generation.

Every source has an availability rule (when a row could first have been read). build() uses a value only
if it was available at the cutoff, and the audit proves it: it rebuilds a sample of days from tables with
everything after the cutoff deleted, and the feature rows must not change.

The feature logic lives in databricks/src/pulse/feature_store.py, shared with the
Databricks job. Writes eda/out/features_day_ahead.parquet (the feature table) and eda/out/features.json (catalogue,
screening and a time-ordered evaluation, for the report).

    python eda/features.py
"""
import json
import math
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib import OUT, ROOT, bar, heat, line, load, num, sanitize, scatter, table

sys.path.insert(0, str(ROOT / 'databricks' / 'src'))
from pulse.feature_store import AVAILABILITY_NOTE, CUTOFF_HOUR, H, TABLES, build, truncate  # noqa: E402

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(7)
# --- Run: build, audit, screen, evaluate ----------------------------------------------------------------------

import time
T0 = time.time()
RAW = {name: load(name) for name in TABLES}
FRAME, META = build(RAW)
print(f'built in {time.time() - T0:.0f}s')
META_BY = {m['feature']: m for m in META}
FEATURES = [m['feature'] for m in META]
print(f'{len(FEATURES)} features, {int(FRAME["target"].notna().sum())} target days')

# Audit: delete everything after the cutoff, rebuild, and compare the row
scored = FRAME.index[FRAME['target'].notna()]
audit_days = sorted(RNG.choice(scored[150:], 12, replace=False))
audit_rows, leaks = [], {}
for day in audit_days:
    day = pd.Timestamp(day)
    rebuilt, _ = build(truncate(RAW, day - (24 - CUTOFF_HOUR) * H), remit_days=[day])
    a, b = FRAME.loc[day, FEATURES].astype(float), rebuilt.loc[day, FEATURES].astype(float)
    differs = ~((a - b).abs() <= 1e-6 * (1 + a.abs())) & ~(a.isna() & b.isna())
    for name in a.index[differs]:
        leaks.setdefault(name, []).append(f'{day:%Y-%m-%d}: {a[name]:.4g} vs {b[name]:.4g}')
    audit_rows.append([f'{day:%Y-%m-%d}', len(FEATURES), int(differs.sum()), ', '.join(a.index[differs][:6]) or '–'])
print('audit:', {k: v[:2] for k, v in leaks.items()} or 'no feature changed')

data = FRAME[FRAME['target'].notna()]
print('coverage below 50%:', (100 * data[FEATURES].notna().mean()).loc[lambda c: c < 50].round(1).to_dict(), 'constant:', [f for f in FEATURES if data[f].nunique() <= 1])
train, test = data[data.index < '2026-01-01'], data[data.index >= '2026-01-01']
fit_part, valid_part = train.iloc[:int(len(train) * 0.8)], train.iloc[int(len(train) * 0.8):]


def gbm(seed=7):
    return HistGradientBoostingRegressor(max_iter=350, learning_rate=0.04, max_depth=4, min_samples_leaf=15, l2_regularization=1.0, random_state=seed)


def ridge():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def permutation(model, X, y, columns_sets, repeats=8):
    base = np.abs(y - model.predict(X)).mean()
    out = {}
    for name, cols in columns_sets.items():
        gains = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[cols] = shuffled[cols].to_numpy()[RNG.permutation(len(X))]
            gains.append(np.abs(y - model.predict(shuffled)).mean() - base)
        out[name] = float(np.mean(gains))
    return out


# Screening on the training years only
imputed = SimpleImputer(strategy='median').fit_transform(train[FEATURES])
mi = mutual_info_regression(imputed, train['target'], random_state=7)
screen = gbm().fit(fit_part[FEATURES], fit_part['target'])
valid_importance = permutation(screen, valid_part[FEATURES], valid_part['target'], {f: [f] for f in FEATURES})
ranked = sorted(FEATURES, key=lambda f: -valid_importance[f])
# Selected set: the features that help on the validation months, dropping near-duplicates of a better one
corr = train[FEATURES].corr().abs()
selected = []
for f in ranked:
    if valid_importance[f] <= 0.01 or len(selected) >= 30:
        break
    if all(corr.loc[f, s] < 0.97 for s in selected):
        selected.append(f)

GOLD = ['fc_wind_solar_mean_mw', 'ndf_mean_mw', 'wx_wind_100m_mean_kmh', 'wx_temperature_mean_c', 'wx_radiation_mean_wm2', 'fuel_share_lag2',
        'fuel_share_lag7', 'cal_is_weekend']
MODELS = {
    'Training mean': None,
    'Share 2 days earlier': 'lag',
    'Physical estimate alone (fc_share_estimate)': (make_pipeline(SimpleImputer(strategy='median'), LinearRegression()), ['fc_share_estimate']),
    'Gold today: 8 features, linear': (make_pipeline(SimpleImputer(strategy='median'), LinearRegression()), GOLD),
    'Gold today: 8 features, gradient boosting': (gbm(), GOLD),
    f'All {len(FEATURES)} features, ridge': (ridge(), FEATURES),
    f'All {len(FEATURES)} features, gradient boosting': (gbm(), FEATURES),
    f'Selected {len(selected)} features, ridge': (ridge(), selected),
    f'Selected {len(selected)} features, gradient boosting': (gbm(), selected),
}
y = test['target']
predictions, fitted = {}, {}
for name, spec in MODELS.items():
    if spec is None:
        predictions[name] = pd.Series(train['target'].mean(), index=test.index)
    elif spec == 'lag':
        predictions[name] = test['fuel_share_lag2'].fillna(train['target'].mean())
    else:
        model, cols = spec
        # The plain linear models are fitted on complete rows, as gold's is_trainable would
        rows = train.dropna(subset=cols) if isinstance(model[-1] if hasattr(model, 'steps') else model, LinearRegression) else train
        model.fit(rows[cols], rows['target'])
        fitted[name] = (model, cols)
        predictions[name] = pd.Series(model.predict(test[cols]), index=test.index)
BEST = min((n for n in predictions if 'Selected' in n or 'All' in n), key=lambda n: np.abs(y - predictions[n]).mean())
BASE = 'Gold today: 8 features, linear'


def block_ci(values, fn=np.mean, block=14, reps=1000):
    values = np.asarray(values, dtype=float)
    n, blocks = len(values), math.ceil(len(values) / block)
    starts = RNG.integers(0, n - block + 1, size=(reps, blocks))
    return np.percentile([fn(values[(s[:, None] + np.arange(block)).ravel()[:n]]) for s in starts], [2.5, 97.5])


def dm_p(loss_a, loss_b):
    import statsmodels.api as sm
    d = (loss_a - loss_b).to_numpy()
    m = sm.OLS(d, np.ones(len(d))).fit(cov_type='HAC', cov_kwds={'maxlags': round(len(d) ** (1 / 3))})
    return float(m.pvalues[0])


model_rows, ci_labels, ci_est, ci_lo, ci_hi = [], [], [], [], []
for name, pred in predictions.items():
    err = y - pred
    lo, hi = block_ci(err.abs())
    r2 = 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()
    gain = np.abs(y - predictions[BASE]).mean() - err.abs().mean()
    p = dm_p(err.abs(), (y - predictions[BASE]).abs()) if name != BASE else None
    model_rows.append([name, f'{err.abs().mean():.2f}', f'{lo:.2f} to {hi:.2f}', f'{np.sqrt((err ** 2).mean()):.2f}', f'{r2:.3f}', f'{err.mean():+.2f}',
                       '–' if name == BASE else f'{gain:+.2f}', '–' if p is None else ('< 0.001' if p < 0.001 else f'{p:.3f}')])
    ci_labels.append(name); ci_est.append(err.abs().mean()); ci_lo.append(lo); ci_hi.append(hi)
    print(f'  {name:55s} MAE {err.abs().mean():.2f}  R2 {r2:.3f}')

best_model, best_cols = fitted[BEST]
test_importance = permutation(best_model, test[best_cols], y, {f: [f] for f in best_cols})
all_model, _ = fitted[f'All {len(FEATURES)} features, gradient boosting']
sources = sorted({m['source'] for m in META}, key=lambda s: (TABLES + ('calendar',)).index(s))
source_importance = permutation(all_model, test[FEATURES], y, {s: [m['feature'] for m in META if m['source'] == s] for s in sources})
# Drop one source at a time and refit: what the model loses without each table
ablation = {}
full_mae = np.abs(y - predictions[f'All {len(FEATURES)} features, gradient boosting']).mean()
for s in sources:
    cols = [m['feature'] for m in META if m['source'] != s]
    ablation[s] = float(np.abs(y - gbm().fit(train[cols], train['target']).predict(test[cols])).mean() - full_mae)

r_all = data[FEATURES].corrwith(data['target'])
rho_all = data[FEATURES].corrwith(data['target'], method='spearman')
coverage = 100 * data[FEATURES].notna().mean()
catalogue = []
for i, f in enumerate(FEATURES):
    m = META_BY[f]
    catalogue.append({'feature': f, 'source': m['source'], 'definition': m['definition'], 'known': m['known'], 'coverage': num(coverage[f], 1),
                      'r': num(r_all[f], 3), 'rho': num(rho_all[f], 3), 'mi': num(mi[i], 3), 'importance': num(valid_importance[f], 3),
                      'selected': f in selected, 'gold': f in GOLD, 'mean': num(data[f].mean(), 3), 'std': num(data[f].std(), 3)})
pairs = []
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().sort_values(ascending=False)
for (a, b), v in upper[upper > 0.97].head(25).items():
    pairs.append([a, b, f'{v:.3f}', a if valid_importance[a] >= valid_importance[b] else b])

top_r = r_all.abs().sort_values(ascending=False).head(25).index
top_imp = sorted(best_cols, key=lambda f: -test_importance[f])[:25]
top_corr = list(r_all.abs().sort_values(ascending=False).head(14).index)
best_pred = predictions[BEST]
roll = pd.DataFrame({'Gold today, linear': (y - predictions[BASE]).abs(), BEST: (y - best_pred).abs()}).rolling(30, min_periods=15).mean()
per_source = pd.DataFrame(catalogue).groupby('source').agg(n=('feature', 'count'), coverage=('coverage', 'mean'), selected=('selected', 'sum')).reindex(sources)
ci_chart = {'t': 'ci', 'title': 'Error on 2026 by model, with 95% intervals', 'sub': 'Mean absolute error in points of renewable share. Fitted on data before 2026; 14-day block bootstrap.',
            'x': ci_labels, 'est': [num(v) for v in ci_est], 'lo': [num(v) for v in ci_lo], 'hi': [num(v) for v in ci_hi], 'unit': 'points', 'ref': None, 'wide': True}
charts = [
    ci_chart,
    line('Actual and predicted renewable share, 2026', f'{BEST}. Every prediction uses only what was known at 18:00 UTC the day before.',
         [d.strftime('%Y-%m-%d') for d in y.index], [('Actual', y), ('Predicted', best_pred)], '%', wide=True),
    scatter('Predicted against actual, 2026', BEST + '.', best_pred, y, 'Predicted share (%)', 'Actual share (%)'),
    line('Error through 2026, 30-day rolling mean', 'Mean absolute error. The gap between the lines is what the new features buy.',
         [d.strftime('%Y-%m-%d') for d in roll.index], [(c, roll[c]) for c in roll.columns], 'points', digits=2),
    bar('What each source table contributes', 'Rise in 2026 error when one table\'s features are shuffled together (all-feature model).', sources,
        [('Error added when shuffled', [source_importance[s] for s in sources])], 'points', horizontal=True, digits=3),
    bar('What the model loses without each table', 'Change in 2026 error when the table\'s features are dropped and the model refitted. Near zero: other tables '
        'carry the same information.', sources, [('Error added when dropped', [ablation[s] for s in sources])], 'points', horizontal=True, digits=3),
    bar('The 25 features that matter most to the model', f'Rise in 2026 error when the feature is shuffled ({BEST}).', top_imp,
        [('Error added when shuffled', [test_importance[f] for f in top_imp])], 'points', horizontal=True, digits=3, wide=True),
    bar('The 25 features most correlated with the target', 'Pearson and Spearman, all scored days. Correlation alone rewards near-duplicates.', top_r,
        [('Pearson r', r_all[top_r].to_numpy()), ('Spearman ρ', rho_all[top_r].to_numpy())], '', horizontal=True, digits=3, wide=True),
    heat('Correlation among the 14 strongest features', 'They are largely one signal measured many ways.', top_corr, top_corr,
         data[top_corr].corr().to_numpy(), '', diverging=True, digits=2, wide=True),
    scatter('The physical estimate against the target', 'fc_share_estimate: built from forecasts and yesterday\'s output with no fitting.',
            data['fc_share_estimate'], data['target'], 'Estimated share (%)', 'Actual share (%)'),
    bar('Features per source table', 'How many each table yields, and how many reach the selected set.', sources,
        [('Engineered', per_source['n'].to_numpy()), ('Selected', per_source['selected'].to_numpy())], 'features', horizontal=True, digits=0),
    bar('Coverage by source table', 'Mean share of scored days on which the table\'s features have a value.', sources,
        [('Coverage', per_source['coverage'].to_numpy())], '%', horizontal=True, digits=1),
]
tables = [
    table('Time-ordered evaluation: fitted before 2026, scored on 2026', ['Model', 'MAE (points)', 'MAE 95% interval', 'RMSE', 'R²', 'Bias', 'MAE gain over gold today',
                                                                          'Diebold–Mariano p'], model_rows,
          f'{len(train):,} training days, {len(test)} test days. Gain and p-value compare each model with the eight gold features in a linear model. '
          'Features were screened and selected on the training years only.'),
    table('Leakage audit', ['Target day', 'Features compared', 'Features that changed', 'Which'], audit_rows,
          'Each day\'s features were rebuilt from tables with every row that became available after the cutoff deleted. A feature that changed would be using the '
          'future. Day-ahead weather is the one stated exception: its hours after 18:00 UTC of the target day come from a run issued after the cutoff.'),
    table('When each table\'s rows count as known', ['Table', 'Available from'], [[t, AVAILABILITY_NOTE[t]] for t in TABLES],
          'Rules marked "assumed" have no publish time in the data; confirm them against the live feeds before relying on same-day values.'),
    table('Near-duplicate features (|r| above 0.97)', ['Feature', 'Feature', '|r|', 'Keep'], pairs, 'Kept: the one that helped more on the validation months.'),
]
summary = {'features': len(FEATURES), 'selected': len(selected), 'train': len(train), 'test': len(test), 'best': BEST,
           'best_mae': num(np.abs(y - best_pred).mean()), 'gold_mae': num(np.abs(y - predictions[BASE]).mean()), 'leaks': len(leaks), 'audit_days': len(audit_days),
           'selected_list': selected}
FRAME.reset_index(names='target_date').to_parquet(OUT / 'features_day_ahead.parquet', index=False)
(OUT / 'features.json').write_text(json.dumps(sanitize({'summary': summary, 'catalogue': catalogue, 'charts': charts, 'tables': tables}), allow_nan=False,
                                              ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
digest = [f'summary {summary}', 'leaks ' + json.dumps(leaks), '', 'MODELS'] + [' | '.join(r) for r in model_rows] + ['', 'SOURCE importance / ablation'] + \
    [f'{s:22s} {source_importance[s]:+.3f} {ablation[s]:+.3f} n={int(per_source.loc[s, "n"])} cov={per_source.loc[s, "coverage"]:.1f}' for s in sources] + \
    ['', 'TOP test importance'] + [f'{f:32s} {test_importance[f]:+.3f} r={r_all[f]:+.3f}' for f in top_imp] + ['', 'CATALOGUE'] + \
    [f"{c['feature']:32s} {c['source']:20s} cov={c['coverage']} r={c['r']} mi={c['mi']} imp={c['importance']} sel={c['selected']}" for c in catalogue] + \
    ['', 'PAIRS'] + [' | '.join(p) for p in pairs]
(OUT / 'features_digest.txt').write_text('\n'.join(digest), encoding='utf-8')
print(f'best: {BEST}; selected {len(selected)}')
