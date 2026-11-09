# Electricity Pulse on Databricks: bronze, silver, gold

The lakehouse half of the project. The ingestion code in the repo root lands raw Kafka messages and API
responses in GCS; everything here runs in Databricks and turns them into a day-ahead feature table for
predicting the GB renewable share.

```
gs://.../landing/<15 folders>                      written by consumer/ and batch_ingest/ (repo root)
        │  (Free Edition: tools/sync_landing.py copies new files into a volume first)
        │  Auto Loader, exactly once per file
        ▼
pulse.bronze.<15 tables>        raw, append-only, + source_file, loaded_at          pipelines/bronze.py
        │  parse → type → check → deduplicate
        ▼
pulse.silver.<15 tables>        typed, one row per key, latest revision              pipelines/silver.py
pulse.silver.rejected           every rejected record, with all its reasons          pulse/sources.py
pulse.silver.fuelhh_history     every FUELHH revision (SCD type 2)                   pulse/quality.py
pulse.silver.completeness       periods present vs expected, per day
        │  joins, daily energy, leakage-safe features
        ▼
pulse.gold.grid_half_hourly     generation, demand, prices, carbon per period        pipelines/gold.py
pulse.gold.generation_daily     energy by category, renewable share (the target)     pulse/gold.py
pulse.gold.features_day_ahead   features known at 18:00 UTC the day before + target
        ▼
notebooks/pipeline_health.py    fails the job on missing files, rejections, gaps
```

## Reading order

| # | File | What to look for |
|---|---|---|
| 1 | [src/pulse/settings.py](src/pulse/settings.py) | Every name and constant: tables, schema hints, fuel categories, the forecast cutoff |
| 2 | [src/pipelines/bronze.py](src/pipelines/bronze.py) | Auto Loader options, lineage columns, why each table is defined in its own function call |
| 3 | [src/pulse/quality.py](src/pulse/quality.py) | The silver engine: the four steps in `check`, how rules and NULLs work, quarantine |
| 4 | [src/pulse/sources.py](src/pulse/sources.py) | The 15 sources: schemas, fields, rules, keys (the grain) and sequences (which version wins) |
| 5 | [src/pipelines/silver.py](src/pipelines/silver.py) | How each source becomes views, a rejected flow and AUTO CDC tables |
| 6 | [src/pulse/gold.py](src/pulse/gold.py) | Shares, completeness, and how every feature avoids looking past the cutoff |
| 7 | [src/pipelines/gold.py](src/pipelines/gold.py) | The gold materialized views |
| 8 | [src/notebooks/pipeline_health.py](src/notebooks/pipeline_health.py) | What fails the daily job |
| 9 | [tools/sync_landing.py](tools/sync_landing.py) | Free Edition only: copying new landing files into the volume, never overwriting |
| 10 | [tests/](tests/) | Each rule shown on a real response; the gold leakage rules on small tables |

The pipeline files only declare tables. All the logic is plain DataFrame functions in `src/pulse/`, so it runs
and is tested on a laptop.

## Design decisions

**Bronze**
- One table per landing folder, raw, append-only. Nothing is parsed, filtered or deduplicated, so silver can
  always be rebuilt from bronze.
- Auto Loader records which files are loaded, so each file loads exactly once, even after a crash or a rerun.
- Schema hints pin the column types to what the ingestion code writes. A new column stops the update once,
  then is added (the job retries the task). Values that don't fit go to `_rescued_data`, which is monitored.

**Silver**
- **An explicit contract.** Each source declares the record fields it expects. A field it doesn't know raises
  the `unexpected_fields` warning instead of disappearing silently.
- **Text first, then types.** Every value is parsed as text and converted with `try_cast`, so "missing" and
  "wrong type" are different reasons. One bad value never fails a batch (no poison pill).
- **Hard rules reject; soft rules flag.** Rejected records go to `silver.rejected` with every reason and the
  record as received. Unusual but usable records are kept with `warnings`: a new interconnector code, a
  fully unavailable REMIT unit reporting 412 MW out of 390.
- **A key contract that fails.** If a NULL key ever reaches the final step, a rule is broken, so the update
  stops instead of writing bad data.
- **Deduplication by business key.** At-least-once delivery means replays and republished copies; Elexon
  revises figures. AUTO CDC keeps one row per key, and the highest `sequence` wins (publish time, then
  fetch time, then a unique tiebreaker), so a late replay can't overwrite a newer revision.
- **Forecasts keep their publish time in the key.** A day-ahead model may only use forecasts made before
  its cutoff, so silver keeps every forecast, not just the newest.
- **UK time.** A UK day has 46, 48 or 50 settlement periods. The valid range is computed from the Europe/London
  time zone, and instants are stored in UTC (local times repeat when the clocks go back).

**Gold**
- Interconnectors are flows, not generation. The renewable share is renewable (wind, hydro, biomass) over
  GB generation, energy-weighted over the day.
- `features_day_ahead` has one row per target day D, using only what was known at 18:00 UTC on D-1:
  - the latest day-ahead wind and solar, wind, and demand forecasts published before then;
  - REMIT outages as their latest revision before then said;
  - actuals only from periods that had ended;
  - the calendar.
- `is_trainable` marks complete days with all the key features.

## Deploying

The same code runs two ways; only where bronze reads the landing files differs.

| | **Free Edition** (default, `free` target) | **Databricks on Google Cloud** (`gcp` target) |
|---|---|---|
| Cost | Nothing at all, no time limit | 14-day trial, then pay-as-you-go; GCS storage |
| Landing zone | `data/landing` on your machine (the ingestion code's default); GCS optional | The GCS bucket |
| Bronze reads | A volume, `/Volumes/pulse/bronze/landing`, which `tools/sync_landing.py` fills | The GCS landing zone directly, through a Unity Catalog external location |
| One-time setup | [00_setup_free.sql](src/setup/00_setup_free.sql) | [00_setup_gcp.sql](src/setup/00_setup_gcp.sql): storage credential with a read-only service account |

Reading GCS directly is the production design: no copy, and the writer (create-only) and reader (read-only)
are separate identities. The free target keeps the whole pipeline at zero cost instead: ingestion lands on
your machine (leave `LANDING_PATH` unset), and one copy step uploads the new files to Databricks.

### Free Edition

Once only:
1. Sign up for Databricks Free Edition, then in the workspace run
   [src/setup/00_setup_free.sql](src/setup/00_setup_free.sql). It creates the catalog, the three schemas and
   the landing volume. If it can't create the catalog, it says how to use the built-in `workspace` one.
2. On your machine, install the tools and sign in:
   ```
   winget install Databricks.DatabricksCLI        # the new CLI; `pip install databricks-cli` is the old one, without bundles
   pip install -r databricks/tools/requirements.txt
   databricks auth login --host https://<your-workspace-url>
   ```
3. Deploy the pipelines and job, from this folder:
   ```
   databricks bundle validate
   databricks bundle deploy
   ```
   This creates `pulse_bronze`, `pulse_silver` and `pulse_gold` (serverless, triggered) and the `pulse_daily`
   job: bronze → silver → gold → health, daily at 07:00 UK time.

Each run, from the repo root, after the ingestion code has landed new files in `data/landing`:
```
python databricks/tools/sync_landing.py          # copies only files the volume doesn't have yet
cd databricks; databricks bundle run pulse_daily # or let the 07:00 schedule run it
```
The first sync copies all of the landing zone, about 200 MB; later ones copy only the day's new files.
`--dry-run` lists what would be copied without copying; `--source gs://...` copies from a bucket instead.

### Databricks on Google Cloud

Create the storage credential in the UI and run [src/setup/00_setup_gcp.sql](src/setup/00_setup_gcp.sql)
(steps at its top), then `databricks bundle deploy -t gcp`. There's no sync step: bronze reads `gs://` itself.

### Without the CLI

Add the repo as a Git folder in the workspace. Create three ETL pipelines, each with one source file from
`src/pipelines/`, catalog `pulse`, and schema `bronze`, `silver` or `gold`. Give each these configuration
settings:
- `pulse.src_path`: the Git folder's `databricks/src` path;
- `pulse.catalog`: `pulse` (or `workspace`);
- `spark.sql.session.timeZone`: `UTC`;
- `pulse.landing`: the landing path (bronze only); left out, it's the catalog's landing volume.

**Full refresh.** After changing a silver rule, full-refresh `pulse_silver`; it rebuilds from bronze, and
previously rejected rows come back if they now pass. Full-refreshing bronze reloads every landing file, which
is safe because landing keeps every file.

## Testing

```
pip install -r databricks/requirements-dev.txt    # PySpark; needs Java 17
pytest databricks/tests
```
The tests run each source's real API response (trimmed copies in `tests/fixtures/`, fetched November 2026)
through silver. They also check the FUELHH rules: clock-change periods, missing vs wrong type, NaN, broken
messages. The gold tests check shares, the calendar, the forecast cutoff and REMIT revisions. One test runs
every gold table end to end on the fixtures. CI runs them in their own job.

The pipeline files themselves need Databricks. They have been run against a stand-in for the pipeline API on
local Spark, which checked that every view, flow and table connects, but not against a real workspace.

## Assumptions to check

- **PV_Live timestamps** are taken as the *end* of each half-hour (gold shifts them back 30 minutes). Confirm
  this against Sheffield Solar's documentation before relying on `pv_live_solar_mw` per period.
- **Day-ahead weather** is Open-Meteo's `previous_day1`: the forecast from about 24 hours before each hour.
  For the last hours of a target day, that run is a few hours after the 18:00 cutoff. It's a small leak; use
  `previous_day2` in batch ingest if you need it strictly clean.
- **REMIT** capacities are each notice's headline values, not its hour-by-hour `outage_profile`. They
  measure what's out during the day, not exact MWh.
- **Biomass counts as renewable**, as in UK government statistics. Change `FUEL_CATEGORIES` in
  [settings.py](src/pulse/settings.py) to use another definition.
- The `pyspark.pipelines` API is used, with a fallback to the older `dlt` module
  ([pipeline_api.py](src/pulse/pipeline_api.py)), so the code runs on either kind of runtime.
