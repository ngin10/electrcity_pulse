"""Build the public model-status page: charts, badges and tables from the live Databricks deployment.

    python databricks/tools/model_status.py                 # -> site/ (open site/index.html)
    python databricks/tools/model_status.py --out other/

Run daily by .github/workflows/model-status.yml, which publishes site/ to GitHub Pages; the README embeds the
charts and badges from there, so the repository shows the model as it is today, not as it was when committed.

What it reads (read-only):
  registry      pulse.ml.renewable_share_day_ahead: versions and aliases (@champion, @challenger, ...)
  MLflow        the latest monthly-retrain run: gate.json and the walk-forward predictions
  SQL           pulse.ml.predictions and pulse.ml.monitoring, and the outcomes in the feature store

What it writes (site/):
  index.html                        the status page
  forecast-{light,dark}.svg         forecast vs outcome with the 80% interval, last 120 days
  accuracy-{light,dark}.svg         30-day rolling MAE against the monitoring warn/alert levels
  coverage-{light,dark}.svg         30-day rolling interval coverage against its 80% target
  badges/*.json                     shields.io endpoint badges
  status.json                       every number on the page

The history before live scoring began is the retraining job's walk-forward: each day predicted by a model refitted
monthly on earlier days only, with each month's interval calibrated on earlier errors only, so it is out of sample.

Auth: DATABRICKS_HOST and DATABRICKS_TOKEN (CI), or your `databricks auth login` profile (locally).
DATABRICKS_WAREHOUSE_ID picks the SQL warehouse; otherwise the first one is used.
"""
import argparse
import html
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pulse.monitoring import COVERAGE_BAND, MAE_POINTS  # noqa: E402  (one source of truth for the alert levels)
from pulse.training import COVERAGE, conformal_half_width  # noqa: E402

CATALOG = os.environ.get('PULSE_CATALOG', 'pulse')
MODEL = f'{CATALOG}.ml.renewable_share_day_ahead'
REPO = 'https://github.com/ngin10/electrcity_pulse'
VALIDATED_MAE = 3.30
ROLLING_DAYS = 30
FORECAST_WINDOW_DAYS = 120

# Chart tokens (light and dark are each selected, not flipped): surfaces, ink, two series, status
THEMES = {
    'light': {'surface': '#fcfcfb', 'ink': '#0b0b0b', 'ink2': '#52514e', 'muted': '#898781', 'grid': '#e1e0d9',
              'axis': '#c3c2b7', 'actual': '#2a78d6', 'forecast': '#eb6834'},
    'dark': {'surface': '#1a1a19', 'ink': '#f0efec', 'ink2': '#c3c2b7', 'muted': '#898781', 'grid': '#2c2c2a',
             'axis': '#383835', 'actual': '#3987e5', 'forecast': '#d95926'},
}
STATUS_ORDER = ['ok', 'warn', 'alert']          # worst last; insufficient_data is "no reading", not a grade
BADGE_COLOUR = {'ok': 'brightgreen', 'warn': 'yellow', 'alert': 'red', None: 'lightgrey'}


# --- pure logic (unit-tested) ----------------------------------------------------------------------------------------

def monthly_conformal_band(wf, level=COVERAGE, min_history=60):
    """Each walk-forward day's interval, calibrated on errors from earlier months only (NaN until enough history)."""
    abs_err = (wf['actual'] - wf['pred']).abs()
    q = pd.Series(np.nan, index=wf.index)
    months = wf.index.to_period('M')
    for month in sorted(set(months)):
        earlier = abs_err[abs_err.index < month.start_time]
        if len(earlier) >= min_history:
            q[months == month] = conformal_half_width(earlier, level)
    return wf.assign(lower=(wf['pred'] - q).clip(lower=0), upper=(wf['pred'] + q).clip(upper=100))


def combine(walk_forward, live, outcomes):
    """One daily history: live forecasts where they exist, the walk-forward elsewhere; outcomes from the feature store.

    walk_forward: index target_date, columns pred, lower, upper.  live: the same.  outcomes: Series of actuals.
    Returns daily rows with pred, lower, upper, actual and source ('walk-forward' / 'live'); days with neither a
    forecast nor an outcome are dropped, days with an outcome but no forecast keep NaN forecasts (gaps stay gaps).
    """
    wf = walk_forward[['pred', 'lower', 'upper']].assign(source='walk-forward')
    lv = live[['pred', 'lower', 'upper']].assign(source='live')
    history = pd.concat([wf[~wf.index.isin(lv.index)], lv]).sort_index()
    days = history.index.union(outcomes.index)
    out = history.reindex(days)
    out['actual'] = outcomes.reindex(days)
    return out.dropna(subset=['pred', 'actual'], how='all')


def rolling(series, days=ROLLING_DAYS, min_periods=20):
    """A calendar-day rolling mean that stays empty until enough days are in the window."""
    return series.dropna().rolling(f'{days}D', min_periods=min_periods).mean()


def worst_status(statuses):
    """The worst graded status (ok < warn < alert); None if nothing was graded (all insufficient_data)."""
    graded = [s for s in statuses if s in STATUS_ORDER]
    return max(graded, key=STATUS_ORDER.index) if graded else None


def mae_status(mae):
    warn, alert = MAE_POINTS
    return None if mae is None or np.isnan(mae) else 'ok' if mae <= warn else 'warn' if mae <= alert else 'alert'


def badge(label, message, colour):
    return {'schemaVersion': 1, 'label': label, 'message': message, 'color': colour}


def summarise(history, monitoring, registry, gate, latest_forecast):
    """Every headline number on the page, as plain data."""
    scored = history.dropna(subset=['pred', 'actual'])
    last30 = scored[scored.index > scored.index.max() - pd.Timedelta(days=ROLLING_DAYS)] if len(scored) else scored
    last90 = scored[scored.index > scored.index.max() - pd.Timedelta(days=90)] if len(scored) else scored
    banded = last90.dropna(subset=['lower', 'upper'])
    inside = ((banded['actual'] >= banded['lower']) & (banded['actual'] <= banded['upper']))
    by_check = {c: worst_status(g['status']) for c, g in monitoring.groupby('check_name')} if len(monitoring) else {}
    return {
        'model': MODEL,
        'aliases': registry['aliases'],
        'mae_30d': float((last30['actual'] - last30['pred']).abs().mean()) if len(last30) else None,
        'mae_30d_days': {'live': int((last30['source'] == 'live').sum()), 'walk_forward': int((last30['source'] == 'walk-forward').sum())},
        'coverage_90d': float(inside.mean()) if len(banded) else None,
        'coverage_90d_days': int(len(banded)),
        'monitoring_run': str(monitoring['run_date'].max()) if len(monitoring) else None,
        'monitoring': by_check,
        'monitoring_worst': worst_status(by_check.values()),
        'latest_forecast': latest_forecast,
        'gate': gate,
    }


def badges(summary):
    champion = summary['aliases'].get('champion')
    challenger = summary['aliases'].get('challenger')
    model_msg = f'champion v{champion}' + (f' · challenger v{challenger}' if challenger and challenger != champion else '')
    mae = summary['mae_30d']
    lf = summary['latest_forecast']
    worst = summary['monitoring_worst']
    return {
        'model': badge('model', model_msg, 'blue'),
        'mae': badge('30-day MAE', f'{mae:.2f} pts' if mae is not None else 'n/a', BADGE_COLOUR[mae_status(mae)]),
        'monitoring': badge('monitoring', worst or 'collecting data', BADGE_COLOUR[worst]),
        'forecast': badge('latest forecast', f"{lf['target_date']}: {lf['predicted_share']:.1f}%" if lf else 'none', 'blue'),
    }


# --- Databricks ------------------------------------------------------------------------------------------------------

def fetch():
    """Everything the page needs, read-only, from the workspace."""
    import mlflow
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState
    from mlflow.tracking import MlflowClient

    w = WorkspaceClient()
    warehouse = os.environ.get('DATABRICKS_WAREHOUSE_ID') or next(iter(w.warehouses.list())).id

    def sql(statement):
        res = w.statement_execution.execute_statement(statement=statement, warehouse_id=warehouse, wait_timeout='50s')
        while res.status.state in (StatementState.PENDING, StatementState.RUNNING):
            time.sleep(3)
            res = w.statement_execution.get_statement(res.statement_id)
        if res.status.state != StatementState.SUCCEEDED:
            raise RuntimeError(f'{statement[:80]}...: {res.status.error.message if res.status.error else res.status.state}')
        cols = [c.name for c in res.manifest.schema.columns]
        return pd.DataFrame((res.result.data_array if res.result else None) or [], columns=cols)

    mlflow.set_tracking_uri('databricks')
    mlflow.set_registry_uri('databricks-uc')
    client = MlflowClient()
    model = client.get_registered_model(MODEL)
    aliases = dict(model.aliases)
    versions = []
    for v in sorted(client.search_model_versions(f"name='{MODEL}'"), key=lambda v: int(v.version)):
        tags = client.get_model_version(MODEL, v.version).tags
        versions.append({'version': v.version, 'created': pd.Timestamp(v.creation_timestamp, unit='ms').strftime('%Y-%m-%d'),
                         'aliases': [a for a, n in aliases.items() if n == v.version], 'gate': tags.get('gate', ''),
                         'wf_mae': tags.get('wf_mae', ''), 'trained_through': tags.get('trained_through', '')})

    runs = mlflow.search_runs(search_all_experiments=True, filter_string="tags.stage = 'retraining'",
                              order_by=['start_time DESC'], max_results=1)
    gate, wf = None, pd.DataFrame(columns=['actual', 'pred'])
    if len(runs):
        run = runs.iloc[0]
        gate = json.load(open(mlflow.artifacts.download_artifacts(run_id=run.run_id, artifact_path='gate.json')))
        gate.update(run_id=run.run_id, date=pd.Timestamp(run.start_time).strftime('%Y-%m-%d'))
        table = json.load(open(mlflow.artifacts.download_artifacts(run_id=run.run_id, artifact_path='walk_forward_predictions.json')))
        wf = pd.DataFrame(table['data'], columns=table['columns'])
        wf = wf.assign(target_date=pd.to_datetime(wf['target_date'])).set_index('target_date').astype(float)

    live = sql(f"""SELECT target_date, model_version, predicted_share AS pred, lower_80 AS lower, upper_80 AS upper, data_quality
                   FROM {CATALOG}.ml.predictions WHERE model_name = '{MODEL}'""")
    outcomes = sql(f"""SELECT target_date, target_renewable_share AS actual FROM {CATALOG}.gold.feature_store_day_ahead
                       WHERE target_renewable_share IS NOT NULL AND target_date >= '2025-01-01'""")
    monitoring = sql(f"""SELECT run_date, model_version, check_name, subject, metric, value, warn_at, alert_at, status, n, details
                         FROM {CATALOG}.ml.monitoring
                         WHERE run_date = (SELECT max(run_date) FROM {CATALOG}.ml.monitoring)""")
    return {'aliases': aliases, 'versions': versions, 'gate': gate, 'wf': wf, 'live': live, 'outcomes': outcomes,
            'monitoring': monitoring}


def prepare(raw):
    """Typed frames and the combined daily history."""
    live = raw['live'].copy()
    live['target_date'] = pd.to_datetime(live['target_date'])
    live = live.set_index('target_date').sort_index()
    for c in ('pred', 'lower', 'upper'):
        live[c] = live[c].astype(float)
    outcomes = raw['outcomes'].assign(target_date=lambda d: pd.to_datetime(d['target_date']))
    outcomes = outcomes.set_index('target_date')['actual'].astype(float).sort_index()
    wf = monthly_conformal_band(raw['wf'].sort_index()) if len(raw['wf']) else pd.DataFrame(columns=['pred', 'lower', 'upper'])
    history = combine(wf, live, outcomes)
    monitoring = raw['monitoring'].copy()
    if len(monitoring):
        monitoring['value'] = pd.to_numeric(monitoring['value'], errors='coerce')
    latest = None
    if len(live):
        row = live.iloc[-1]
        latest = {'target_date': f'{live.index[-1]:%Y-%m-%d}', 'predicted_share': float(row['pred']), 'lower_80': float(row['lower']),
                  'upper_80': float(row['upper']), 'data_quality': row.get('data_quality'), 'model_version': row.get('model_version')}
    return history, live, monitoring, latest


# --- charts ----------------------------------------------------------------------------------------------------------

def _axes(theme, ylabel):
    import matplotlib.pyplot as plt
    t = THEMES[theme]
    plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'sans-serif',
                         'font.sans-serif': ['Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans'], 'font.size': 10})
    fig, ax = plt.subplots(figsize=(9, 3.3))
    fig.patch.set_facecolor(t['surface'])
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(t['axis'])
    ax.grid(axis='y', color=t['grid'], linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=t['muted'], length=0, labelsize=9)
    ax.set_ylabel(ylabel, color=t['ink2'], fontsize=9)
    return fig, ax, t


def _reference(ax, t, y, label):
    """A labelled hairline: a threshold, never a series."""
    ax.axhline(y, color=t['muted'], linewidth=1)
    ax.annotate(label, xy=(1, y), xycoords=('axes fraction', 'data'), xytext=(4, 0), textcoords='offset points',
                va='center', ha='left', fontsize=8.5, color=t['ink2'])


def _live_marker(ax, t, history):
    live_days = history.index[history['source'] == 'live']
    if len(live_days) and live_days.min() > history.index.min():  # only when the switch to live is inside the chart
        x = live_days.min()
        ax.axvline(x, color=t['axis'], linewidth=1)
        ax.annotate('live scoring', xy=(x, 1), xycoords=('data', 'axes fraction'), xytext=(4, -2), textcoords='offset points',
                    va='top', fontsize=8.5, color=t['ink2'])


def _date_axis(ax):
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _save(fig, path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, format='svg', facecolor=fig.get_facecolor())
    plt.close(fig)


def chart_forecast(history, path, theme):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    end = history.index.max()
    h = history[history.index > end - pd.Timedelta(days=FORECAST_WINDOW_DAYS)].asfreq('D')  # missing days break the lines
    fig, ax, t = _axes(theme, 'Renewable share, %')
    ax.fill_between(h.index, h['lower'], h['upper'], color=t['forecast'], alpha=0.12, linewidth=0)
    ax.plot(h.index, h['actual'], color=t['actual'], linewidth=2, solid_capstyle='round', solid_joinstyle='round')
    ax.plot(h.index, h['pred'], color=t['forecast'], linewidth=2, solid_capstyle='round', solid_joinstyle='round')
    lone = h['pred'].notna() & h['pred'].shift(1).isna() & h['pred'].shift(-1).isna()  # isolated days draw no line: mark them
    ax.plot(h.index[lone], h['pred'][lone], 'o', color=t['forecast'], markersize=8, markeredgecolor=t['surface'], markeredgewidth=2)
    last = h['pred'].last_valid_index()
    if last is not None:
        ax.annotate(f"{h.loc[last, 'pred']:.1f}%", xy=(last, h.loc[last, 'pred']), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=9, color=t['ink'])
    ax.set_ylim(0, max(100, float(np.nanmax(h[['actual', 'upper']].to_numpy())) + 2) if len(h) else 100)
    _date_axis(ax)
    _live_marker(ax, t, h.dropna(subset=['source']))
    ax.legend(handles=[Line2D([], [], color=t['actual'], lw=2, label='Outcome'), Line2D([], [], color=t['forecast'], lw=2, label='Forecast'),
                       Patch(color=t['forecast'], alpha=0.12, label='80% interval')],
              loc='upper left', ncols=3, frameon=False, fontsize=9, labelcolor=t['ink2'], bbox_to_anchor=(0, 1.13))
    _save(fig, path)


def chart_accuracy(history, path, theme):
    scored = history.dropna(subset=['pred', 'actual'])
    mae = rolling((scored['actual'] - scored['pred']).abs())
    fig, ax, t = _axes(theme, f'MAE, {ROLLING_DAYS}-day rolling, points')
    ax.plot(mae.index, mae, color=t['actual'], linewidth=2, solid_capstyle='round')
    warn, alert = MAE_POINTS
    for y, label in ((VALIDATED_MAE, f'validated {VALIDATED_MAE:.2f}'), (warn, f'warn {warn:.1f}'), (alert, f'alert {alert:.1f}')):
        _reference(ax, t, y, label)
    ax.set_ylim(0, max(alert + 1, float(mae.max()) + 0.5) if len(mae) else alert + 1)
    _date_axis(ax)
    _live_marker(ax, t, scored)
    _save(fig, path)


def chart_coverage(history, path, theme):
    banded = history.dropna(subset=['actual', 'lower', 'upper'])
    inside = ((banded['actual'] >= banded['lower']) & (banded['actual'] <= banded['upper'])).astype(float)
    cov = rolling(inside) * 100
    fig, ax, t = _axes(theme, f'Inside the 80% interval, {ROLLING_DAYS}-day rolling, %')
    ax.plot(cov.index, cov, color=t['actual'], linewidth=2, solid_capstyle='round')
    (_, _), (alert_lo, alert_hi) = COVERAGE_BAND
    for y, label in ((alert_hi * 100, f'alert above {alert_hi:.0%}'), (COVERAGE * 100, f'target {COVERAGE:.0%}'),
                     (alert_lo * 100, f'alert below {alert_lo:.0%}')):
        _reference(ax, t, y, label)
    ax.set_ylim(40, 100)
    _date_axis(ax)
    _live_marker(ax, t, banded)
    _save(fig, path)


# --- page ------------------------------------------------------------------------------------------------------------

STATUS_LABEL = {'ok': ('✓', 'ok', 'good'), 'warn': ('▲', 'warn', 'warning'), 'alert': ('✕', 'alert', 'critical'),
                'insufficient_data': ('–', 'not enough data yet', 'muted'), None: ('–', 'not enough data yet', 'muted')}


def _status(s):
    icon, label, cls = STATUS_LABEL.get(s, ('?', str(s), 'muted'))
    return f'<span class="st {cls}"><span aria-hidden="true">{icon}</span> {html.escape(label)}</span>'


def _picture(name, alt):
    return (f'<picture><source srcset="{name}-dark.svg" media="(prefers-color-scheme: dark)">'
            f'<img src="{name}-light.svg" alt="{html.escape(alt)}" width="900"></picture>')


def _num(x, fmt='{:.3f}'):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def page(summary, history, monitoring, versions, generated):
    s = summary
    lf = s['latest_forecast']
    champion = s['aliases'].get('champion', '?')
    challenger = s['aliases'].get('challenger')
    days = s['mae_30d_days']
    mae_note = (f"{days['live']} live days" + (f" + {days['walk_forward']} walk-forward" if days['walk_forward'] else '')) if s['mae_30d'] else ''
    tiles = [
        ('Champion', f'v{champion}', f'challenger v{challenger} awaiting review' if challenger and challenger != champion else 'no challenger'),
        ('MAE, last 30 days', _num(s['mae_30d'], '{:.2f}') + ' pts', f"{mae_note}; alert above {MAE_POINTS[1]}"),
        ('Interval coverage, 90 days', _num((s['coverage_90d'] or np.nan) * 100, '{:.0f}') + '%', f"target 80%; {s['coverage_90d_days']} days"),
        ('Latest forecast', f"{lf['predicted_share']:.1f}%" if lf else '–',
         f"{lf['target_date']}, 80% interval {lf['lower_80']:.1f}–{lf['upper_80']:.1f}" if lf else ''),
    ]
    tiles_html = ''.join(f'<div class="tile"><div class="label">{a}</div><div class="value">{html.escape(b)}</div>'
                         f'<div class="sub">{html.escape(c)}</div></div>' for a, b, c in tiles)

    gate = s['gate']
    gate_html = '<p class="muted">No retraining run yet.</p>'
    if gate:
        rows = ''.join(f"<tr><td>{html.escape(c['check'].replace('_', ' '))}</td><td class='n'>{_num(c['value'])}</td>"
                       f"<td class='n'>{_num(c['limit'], '{:g}')}</td><td>{_status('ok' if c['passed'] else 'alert')}</td>"
                       f"<td class='muted'>{html.escape(c.get('detail', ''))}</td></tr>" for c in gate['checks'])
        verdict = 'passed: registered as @challenger' if gate['passed'] else 'failed: nothing registered'
        gate_html = (f"<p>Run of {gate['date']}: <strong>{verdict}</strong>. Run id <code>{gate['run_id']}</code>.</p>"
                     f"<div class='scroll'><table><thead><tr><th>Check</th><th class='n'>Value</th><th class='n'>Limit</th><th>Result</th>"
                     f"<th>Detail</th></tr></thead><tbody>{rows}</tbody></table></div>")

    mon_rows = ''.join(f"<tr><td>{html.escape(c.replace('_', ' '))}</td><td>{_status(st)}</td></tr>" for c, st in sorted(s['monitoring'].items()))
    flagged = monitoring[monitoring['status'].isin(['warn', 'alert'])] if len(monitoring) else monitoring
    flag_html = ''
    if len(flagged):
        flag_html = ('<div class="scroll"><table><thead><tr><th>Check</th><th>Subject</th><th>Metric</th><th class="n">Value</th><th>Status</th>'
                     '</tr></thead><tbody>' + ''.join(
                         f"<tr><td>{html.escape(r.check_name)}</td><td>{html.escape(str(r.subject))}</td><td>{html.escape(r.metric)}</td>"
                         f"<td class='n'>{_num(r.value)}</td><td>{_status(r.status)}</td></tr>" for r in flagged.itertuples())
                     + '</tbody></table></div>')
    fresh = monitoring[monitoring['check_name'] == 'freshness'] if len(monitoring) else monitoring
    fresh_html = ''.join(f"<tr><td>{html.escape(str(r.subject))}</td><td class='n'>{_num(r.value, '{:.2f}')}</td><td>{_status(r.status)}</td></tr>"
                         for r in fresh.sort_values('subject').itertuples()) if len(fresh) else ''

    ver_rows = ''.join(f"<tr><td>v{v['version']}</td><td>{', '.join('@' + a for a in v['aliases']) or '–'}</td><td>{v['created']}</td>"
                       f"<td>{html.escape(v['gate'] or '–')}</td><td class='n'>{html.escape(v['wf_mae'] or '–')}</td>"
                       f"<td>{html.escape(v['trained_through'] or '–')}</td></tr>" for v in reversed(versions))

    recent = history.dropna(subset=['pred']).tail(14).iloc[::-1]
    table_rows = ''.join(f"<tr><td>{d:%Y-%m-%d}</td><td>{html.escape(str(r.source))}</td><td class='n'>{_num(r.pred, '{:.1f}')}</td>"
                         f"<td class='n'>{_num(r.lower, '{:.1f}')}–{_num(r.upper, '{:.1f}')}</td><td class='n'>{_num(r.actual, '{:.1f}')}</td></tr>"
                         for d, r in zip(recent.index, recent.itertuples()))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pulse Model Status</title>
<meta name="description" content="Live status of the GB day-ahead renewable-share model: forecasts, accuracy, monitoring, retraining gate and registry.">
<style>
:root {{ color-scheme: light; --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --line: #e1e0d9; --border: rgba(11,11,11,0.10); --good: #006300; --warning: #9a6a00; --critical: #d03b3b; --link: #256abf; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --page: #0d0d0d; --surface: #1a1a19;
  --ink: #f0efec; --ink2: #c3c2b7; --line: #2c2c2a; --border: rgba(255,255,255,0.10); --good: #0ca30c; --warning: #fab219;
  --critical: #e66767; --link: #6da7ec; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --line: #2c2c2a;
  --border: rgba(255,255,255,0.10); --good: #0ca30c; --warning: #fab219; --critical: #e66767; --link: #6da7ec; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 40px 0 8px; }}
p {{ margin: 6px 0; }} a {{ color: var(--link); }} code {{ font-size: 13px; }}
.muted, .sub {{ color: var(--ink2); }} .sub {{ font-size: 13px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 24px 0 8px; }}
.tile {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
.tile .label {{ color: var(--ink2); font-size: 13px; }} .tile .value {{ font-size: 28px; font-weight: 600; margin: 2px 0; }}
figure {{ margin: 12px 0 0; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }}
figure img {{ width: 100%; height: auto; display: block; }} figcaption {{ color: var(--ink2); font-size: 13px; margin-top: 6px; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; background: var(--surface); }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ color: var(--ink2); font-weight: 600; }} .n {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.st {{ white-space: nowrap; font-weight: 600; }} .st.good {{ color: var(--good); }} .st.warning {{ color: var(--warning); }}
.st.critical {{ color: var(--critical); }} .st.muted {{ color: var(--muted); font-weight: 400; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
footer {{ margin-top: 48px; color: var(--muted); font-size: 13px; }}
</style></head>
<body><main>
<h1>Pulse model status</h1>
<p class="muted">Day-ahead forecast of Great Britain's renewable share (wind, hydro and biomass), {html.escape(MODEL)}.
Updated daily from the Databricks deployment. Generated {generated}.</p>
<div class="tiles">{tiles_html}</div>

<h2>Forecast and outcome</h2>
<figure>{_picture('forecast', 'Forecast against outcome with the 80% interval, last 120 days')}
<figcaption>Before the "live scoring" line: the retraining job's walk-forward, each day predicted by a model refitted monthly on earlier
days only, with each month's interval calibrated on earlier errors only. After it: forecasts written by the 18:45 UTC scoring job.
Gaps are days that were not forecast in real time.</figcaption></figure>

<h2>Accuracy</h2>
<figure>{_picture('accuracy', 'Thirty-day rolling mean absolute error against the warn and alert levels')}
<figcaption>Warn and alert levels are the ones the monitoring job uses. Validated: the champion's out-of-sample MAE at selection.</figcaption></figure>
<figure>{_picture('coverage', 'Thirty-day rolling share of outcomes inside the 80% interval')}
<figcaption>An honest 80% interval contains about 80% of outcomes; misses come in runs during turbulent weeks.</figcaption></figure>

<h2>Recent days</h2>
<div class="scroll"><table><thead><tr><th>Day</th><th>Source</th><th class="n">Forecast %</th><th class="n">80% interval</th><th class="n">Outcome %</th>
</tr></thead><tbody>{table_rows}</tbody></table></div>

<h2>Monitoring</h2>
<p class="muted">Latest run {s['monitoring_run'] or '–'}. Accuracy and drift need about 30 scored days with outcomes before they grade.</p>
<div class="two"><div class="scroll"><table><thead><tr><th>Check</th><th>Worst</th></tr></thead><tbody>{mon_rows}</tbody></table></div>
<div class="scroll"><table><thead><tr><th>Source</th><th class="n">Present vs usual</th><th>Freshness</th></tr></thead>
<tbody>{fresh_html}</tbody></table></div></div>
{flag_html}

<h2>Retraining gate</h2>
<p class="muted">The monthly job refits the champion's own specification and must pass every check to become @challenger.
A person promotes it to @champion.</p>
{gate_html}

<h2>Registry</h2>
<div class="scroll"><table><thead><tr><th>Version</th><th>Aliases</th><th>Registered</th><th>Gate</th><th class="n">Walk-forward MAE</th>
<th>Trained through</th></tr></thead><tbody>{ver_rows}</tbody></table></div>

<footer>Source: <a href="{REPO}">{REPO.removeprefix('https://')}</a> ·
<a href="{REPO}/blob/main/docs/model_card.md">model card</a> ·
<a href="{REPO}/blob/main/docs/model_validation_report.md">validation report</a> ·
built by <code>databricks/tools/model_status.py</code></footer>
</main></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='site', help='output folder (default: site/)')
    args = parser.parse_args()
    out = Path(args.out)
    (out / 'badges').mkdir(parents=True, exist_ok=True)

    raw = fetch()
    history, live, monitoring, latest = prepare(raw)
    summary = summarise(history, monitoring, {'aliases': raw['aliases']}, raw['gate'], latest)
    generated = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')

    for theme in THEMES:
        chart_forecast(history, out / f'forecast-{theme}.svg', theme)
        chart_accuracy(history, out / f'accuracy-{theme}.svg', theme)
        chart_coverage(history, out / f'coverage-{theme}.svg', theme)
    for name, b in badges(summary).items():
        (out / 'badges' / f'{name}.json').write_text(json.dumps(b), encoding='utf-8')
    (out / 'status.json').write_text(json.dumps({**summary, 'generated': generated, 'versions': raw['versions']}, indent=2, default=str),
                                     encoding='utf-8')
    (out / 'index.html').write_text(page(summary, history, monitoring, raw['versions'], generated), encoding='utf-8')
    (out / '.nojekyll').write_text('', encoding='utf-8')
    print(f"wrote {out}/: champion v{summary['aliases'].get('champion')}, 30-day MAE {_num(summary['mae_30d'], '{:.2f}')}, "
          f"monitoring {summary['monitoring_worst'] or 'collecting data'}, {len(history)} days of history")


if __name__ == '__main__':
    main()
