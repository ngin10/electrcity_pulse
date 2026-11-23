"""Validation evidence for the champion: stability, stress, sensitivity and explainability, tracked in MLflow.

Uses the champion's walk-forward predictions (out of sample) and the packaged champion (@champion). Writes
eda/out/validation.json, from which docs/model_validation_report.md quotes its numbers.

    python eda/validate_champion.py
"""
import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

import experiment as ex
from train_ensembles import ensemble_walk_forward
from train_linear import LINEAR
from train_trees import TREES

wf, _ = ensemble_walk_forward('blend_bayes_lgbm')
err = wf['actual'] - wf['pred']
members = {name: ex.walk_forward(family, name)[0]['pred'] for name, family in (('bayesian_ridge', LINEAR), ('lightgbm', TREES))}
champion = mlflow.pyfunc.load_model(f'models:/{ex.REGISTERED_MODEL}@champion')
blend = champion.unwrap_python_model().point_model
X_all = ex.data[ex.FEATURES]
X26 = X_all[X_all.index >= ex.HOLDOUT_START]
out = {}

# Stability over time
quarter = err.groupby(err.index.to_period('Q'))
out['by_quarter'] = {str(q): {'days': len(e), 'mae': round(e.abs().mean(), 3), 'bias': round(e.mean(), 3)} for q, e in quarter}
out['by_season'] = {s: round(v, 3) for s, v in err.abs().groupby(err.index.month.map(ex.SEASONS)).mean().items()}
labels = pd.cut(wf['actual'], [b[0] for b in ex.BANDS] + [101], labels=[b[2] for b in ex.BANDS], right=False)
out['by_share_band'] = {str(b): {'days': int((labels == b).sum()), 'mae': round(v, 3), 'bias': round(err[labels == b].mean(), 3)}
                        for b, v in err.abs().groupby(labels, observed=True).mean().items()}
monthly = err.abs().groupby(err.index.to_period('M')).mean()
out['monthly_mae_range'] = [round(monthly.min(), 3), round(monthly.max(), 3), str(monthly.idxmin()), str(monthly.idxmax())]

# Stress: the worst days, and whether the members disagreed on them
disagree = (members['bayesian_ridge'] - members['lightgbm']).abs()
worst = err.abs().sort_values(ascending=False).head(10).index
out['worst_days'] = [{'date': f'{d:%Y-%m-%d}', 'actual': round(wf.loc[d, 'actual'], 1), 'predicted': round(wf.loc[d, 'pred'], 1),
                      'error': round(err[d], 1), 'ridge': round(members['bayesian_ridge'][d], 1), 'lightgbm': round(members['lightgbm'][d], 1),
                      'forecast_wind_share_of_demand': round(ex.data.loc[d, 'fc_wind_share_of_demand'], 1)} for d in worst]
out['error_vs_member_disagreement_r'] = round(float(np.corrcoef(disagree, err.abs())[0, 1]), 3)
out['member_correlation'] = round(float(np.corrcoef(members['bayesian_ridge'], members['lightgbm'])[0, 1]), 3)
big = err.abs() > 8
out['days_error_over_8'] = int(big.sum())
out['share_of_big_misses_under_predicted'] = round(float((err[big] > 0).mean()), 3)

# Sensitivity of the packaged model, over the 2026 days: nudge one input and see how far predictions move
base = champion.predict(X26)['predicted_share']
scenarios = {
    'forecast wind share of demand +10%': {'fc_wind_share_of_demand': 1.10},
    'forecast wind share of demand -10%': {'fc_wind_share_of_demand': 0.90},
    'all wind forecasts +10%': {c: 1.10 for c in ex.FEATURES if c.startswith(('fc_wind', 'wf_')) and c.endswith(('_mw', '_of_demand'))},
    'demand forecast +5%': {'ndf_mean_mw': 1.05, 'ndf_peak_mw': 1.05, 'ndf_min_mw': 1.05},
    'temperature forecast -3 C': {'wx_temperature_mean_c': None, 'wx_temperature_min_c': None},
}
sens = {}
for name, changes in scenarios.items():
    Xs = X26.copy()
    for col, factor in changes.items():
        Xs[col] = Xs[col] - 3 if factor is None else Xs[col] * factor
    moved = champion.predict(Xs)['predicted_share'] - base
    sens[name] = {'mean_change': round(moved.mean(), 3), 'mean_abs_change': round(moved.abs().mean(), 3), 'max_abs_change': round(moved.abs().max(), 3)}
# Missing inputs: the day-ahead weather absent (as on 7 November 2026)
Xm = X26.copy()
Xm[[c for c in ex.FEATURES if c.startswith('wx_')]] = np.nan
moved = champion.predict(Xm)['predicted_share'] - base
actual26 = ex.data.loc[X26.index, 'target']
sens['day-ahead weather missing'] = {'mean_change': round(moved.mean(), 3), 'mean_abs_change': round(moved.abs().mean(), 3),
                                     'max_abs_change': round(moved.abs().max(), 3),
                                     'mae_in_sample_with_weather': round((actual26 - base).abs().mean(), 3),
                                     'mae_in_sample_without_weather': round((actual26 - base - moved).abs().mean(), 3)}
out['sensitivity'] = sens
# How much wider the 80% interval must be on a day without weather: the ratio of the 80th percentiles of |error|
# without and with weather, on the same 2026 days. Both are in sample, so the ratio (not either level) is used: the
# scoring job multiplies the out-of-sample conformal half-width by it for days flagged weather_missing
q80_with, q80_without = (actual26 - base).abs().quantile(0.8), (actual26 - base - moved).abs().quantile(0.8)
out['weather_missing_interval'] = {'q80_abs_error_with_weather': round(q80_with, 3), 'q80_abs_error_without_weather': round(q80_without, 3),
                                   'widening_factor': round(q80_without / q80_with, 3)}

# Monotonic sanity: does more forecast wind (all else equal) ever lower the prediction?
grid = np.linspace(X26['fc_wind_share_of_demand'].quantile(0.05), X26['fc_wind_share_of_demand'].quantile(0.95), 25)
curves = []
for _, row in X26.sample(60, random_state=7).iterrows():
    Xg = pd.DataFrame([row] * len(grid), columns=X26.columns)
    Xg['fc_wind_share_of_demand'] = grid
    curves.append(champion.predict(Xg.reset_index(drop=True))['predicted_share'].to_numpy())
curves = np.array(curves)
steps = np.diff(curves, axis=1)
out['wind_share_response'] = {'grid_from': round(grid[0], 1), 'grid_to': round(grid[-1], 1),
                              'mean_rise_points': round(float((curves[:, -1] - curves[:, 0]).mean()), 2),
                              'share_of_steps_decreasing': round(float((steps < -1e-6).mean()), 3),
                              'largest_decrease_points': round(float(-steps.min()), 3)}

# What drives the predictions: exact decomposition, mean |contribution| over 2026
fitted = {name: (est, cols) for name, est, cols in blend.fitted_}
weights = dict(zip([n for n, _, _ in blend.fitted_], blend.weights))
ridge, ridge_cols = fitted['bayesian_ridge']
linear = pd.DataFrame(ridge[:-1].transform(X26[ridge_cols]) * ridge[-1].coef_, index=X26.index, columns=ridge[:-1].get_feature_names_out())
lgbm, lgbm_cols = fitted['lightgbm']
shap = lgbm.contributions(X26[lgbm_cols])
contrib = (weights['bayesian_ridge'] * linear.reindex(columns=ex.FEATURES, fill_value=0.0)
           + weights['lightgbm'] * shap.drop(columns='base_value').reindex(columns=ex.FEATURES, fill_value=0.0))
importance = contrib.abs().mean().sort_values(ascending=False)
out['top_contributions_2026'] = {k: round(v, 3) for k, v in importance.head(12).items()}
by_source = json.loads((ex.OUT / 'features.json').read_text(encoding='utf-8'))['catalogue']
source_of = {c['feature']: c['source'] for c in by_source}
out['contribution_by_source_2026'] = {k: round(v, 3) for k, v in importance.groupby(importance.index.map(source_of)).sum()
                                      .sort_values(ascending=False).items()}
recon = (weights['bayesian_ridge'] * ridge[-1].intercept_ + weights['lightgbm'] * shap['base_value'] + contrib.sum(axis=1) - base).abs().max()
out['decomposition_max_gap'] = float(recon)

(ex.OUT / 'validation.json').write_text(json.dumps(out, indent=1))
folder = ex.TRACKING / 'tmp' / 'validation'
folder.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
for c in curves[:25]:
    ax.plot(grid, c, color=ex.BLUE, alpha=0.35, lw=0.8)
ax.plot(grid, curves.mean(axis=0), color=ex.ORANGE, lw=2, label='Average')
ax.set(title='Prediction as forecast wind share of demand varies (60 days of 2026)', xlabel='forecast wind ÷ demand (%)', ylabel='predicted share (%)')
ax.legend(frameon=False)
ex.save(fig, folder, 'wind_share_response.png')
fig, ax = plt.subplots(figsize=(7, 5))
top = importance.head(15)[::-1]
ax.barh(top.index, top.values, color=ex.BLUE)
ax.set(title='What drives the champion (2026, mean |contribution|)', xlabel='points of share')
ex.save(fig, folder, 'contributions_2026.png')

with mlflow.start_run(run_name='champion-validation', description='Validation evidence for the champion') as run:
    mlflow.set_tags({'model_family': 'validation', 'stage': 'validation', 'git_commit': ex.COMMIT, 'champion_local_version':
                     mlflow.tracking.MlflowClient().get_model_version_by_alias(ex.REGISTERED_MODEL, 'champion').version})
    mlflow.log_dict(out, 'validation.json')
    mlflow.log_artifacts(str(folder), artifact_path='charts')
    mlflow.log_metrics({'member_correlation': out['member_correlation'], 'error_vs_disagreement_r': out['error_vs_member_disagreement_r'],
                        'wind_share_steps_decreasing': out['wind_share_response']['share_of_steps_decreasing'],
                        **{f"sens_{k.replace(' ', '_').replace('%', 'pct').replace('+', 'plus').replace('-', 'minus')}": v['mean_abs_change']
                           for k, v in sens.items()}})
print(json.dumps(out, indent=1))
print('run', run.info.run_id)
