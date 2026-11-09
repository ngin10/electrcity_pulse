# Electricity Pulse

**UK electricity generation by fuel type, streamed from the Elexon API through Kafka into a Google Cloud
Storage landing zone.** The medallion lakehouse (bronze, silver, gold) is built in Databricks on top of the
landing files: see [databricks/](databricks/README.md).

```
┌────────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────────────┐   ┌──────────────────────────┐
│ Elexon     │──►│ producer     │──►│  Kafka   │──►│ consumer           │──►│ gs://<bucket>/landing/   │──► Databricks
│ FUELHH API │   │ python main  │   │ (Docker) │   │ python -m consumer │   │   fuelhh/                │
└────────────┘   └──────────────┘   └──────────┘   └────────────────────┘   │   <batch source>/        │
┌────────────────────────────────────────┐   ┌───────────────────────┐      │                          │
│ demand, wind forecasts, carbon, solar, │──►│ batch ingest          │─────►│                          │
│ NESO embedded generation, weather      │   │ python -m batch_ingest│      └──────────────────────────┘
└────────────────────────────────────────┘   └───────────────────────┘
```

Every half-hour, Elexon publishes how much electricity each fuel type (wind, gas, nuclear, interconnectors
and so on) supplied to the GB grid, and revises those figures later. This repo gets every record, including
every revision, into GCS unchanged, along with the demand, solar, carbon and weather data that explain it.

## Batch sources

Fetched by date range and landed as received, one row per API response. Every endpoint, request limit and
history depth here was checked against the live APIs.

| Landing table | What it holds | Per request | History |
|---|---|---|---|
| `elexon_demand_outturn` | National demand (INDO, ITSDO), half-hourly | 7 days | 2023 and earlier |
| `elexon_wind_forecast` | Elexon wind generation forecasts (WINDFOR), every publish | 7 days | 2023 and earlier |
| `carbon_intensity` | Carbon intensity, forecast and actual, half-hourly | 14 days (API max 31) | 2023 and earlier |
| `pv_live` | Sheffield Solar PV_Live national solar estimate, half-hourly | 31 days | 2023 and earlier |
| `neso_historic_demand` | NESO demand, embedded wind and solar, interconnector flows | 31 days | 2004 onward; published ~3 weeks behind |
| `weather_forecast_day_ahead` | Open-Meteo forecasts as made the day before, at 5 UK wind/solar points | 31 days | from 2024-03 |
| `weather_historical_forecast` | Open-Meteo stitched short-range forecasts, same points | 31 days | 2021 onward |
| `elexon_demand_forecast` | National demand forecast (NDF), only the publishes 17:00-18:00 UTC: each covers all of the next day | 1 day (API max) | 2023 and earlier |
| `elexon_wind_solar_forecast` | Day-ahead wind and solar generation forecasts (DGWS) | 7 days (API max) | 2023 and earlier |
| `elexon_wind_solar_actual` | Actual wind and solar generation (AGWS) | 7 days (API max) | 2023 and earlier |
| `elexon_system_prices` | System (imbalance) prices and volumes | 1 day, date in the URL path | 2016 and earlier |
| `elexon_market_index` | Wholesale market index price and volume (MID); `APXMIDP` has the prices, `N2EXMIDP` is mostly zero | 7 days (API max) | 2023 and earlier |
| `elexon_remit_outages` | Power-station outage and unavailability notices, with details | 1 day (API max) | 2023 and earlier |
| `uk_bank_holidays` | Bank holidays for England & Wales, Scotland and Northern Ireland | one snapshot per run | 2019 onward |

For a day-ahead model, train on what was known the day before: `weather_forecast_day_ahead`,
`elexon_demand_forecast` and `elexon_wind_solar_forecast`. The stitched weather series and the actuals are
what really happened, which a model won't have when it predicts, so they suit exploration and targets rather
than features.

Columns: `raw_value` (the response body), `source`, `request_url`, `window_from`, `window_to` (the days
requested), `http_status`, `fetched_at` (ISO UTC; also sets `ingest_date`). Each run re-fetches the last
few days of every source, since sources revise recent data, so readers deduplicate on the record's own key
and keep the latest `fetched_at`.

## Landing files

`<LANDING_PATH>/fuelhh/ingest_date=YYYY-MM-DD/part-<uuid>.parquet`, append-only. One row per Kafka message:

| Column | Type | Meaning |
|---|---|---|
| `raw_value` | string | The message exactly as published: an envelope with `schema_version`, `source`, `dataset`, `ingested_at`, `fetch_window` and the Elexon record in `payload` |
| `kafka_topic`, `kafka_partition`, `kafka_offset` | string, int64, int64 | Where the message sits in Kafka; together they identify it uniquely |
| `kafka_key` | string | The record's identity (hash of date, period and fuel type) |
| `kafka_timestamp_ms` | int64, nullable | When Kafka received the message |
| `consumed_at` | string (ISO, UTC) | When the consumer read the batch; also sets `ingest_date` |

## Design decisions

- **Raw all the way to GCS.** Neither the producer nor the consumer filters or changes records: every record
  from Elexon is published and landed as-is. Data-quality checks belong downstream, so a rule change needs no
  redeploy here and can be re-applied to everything already landed.
- **A landing zone between Kafka and the lakehouse.** Ingestion and processing are decoupled: the consumer is
  small and cheap, Kafka never has to be reachable from the cloud, and the lakehouse can be rebuilt from the
  raw files at any time.
- **Kafka keys that survive compaction.** The topic is compacted (Kafka keeps the latest value per key), and
  a record's key is its identity: date, period and fuel type, so a revision replaces the older value. A broken
  record missing part of that identity is keyed by its whole content instead, so compaction can't collapse
  unrelated bad records into one before they are landed.
- **At-least-once delivery.** Kafka offsets are committed only after a file is safely stored. A crash replays
  the batch into a new file, so downstream deduplicates on the record key (or on topic, partition and offset).
- **Fixed file schema.** Every landing file has the same column types, whatever the batch contains, so a
  reader such as Auto Loader never sees a column change type between files.
- **No partial files.** Locally, files are written to a temp name and renamed; in GCS, an object only becomes
  visible once its upload completes.

## Running it

```
pip install -r requirements.txt
docker compose up -d          # Kafka
python main.py                # Elexon -> Kafka
python -m consumer --once     # Kafka -> LANDING_PATH (default: data/landing)
```

### Landing in GCS

The consumer writes as a dedicated service account that can **only create objects** in the bucket: it can't
read, list, overwrite or delete them, so append-only is enforced by GCS, not just by the code. There is no key
file: you sign in once and the consumer impersonates the service account.

One-time setup (`<project>`, `<bucket>` and the account name are yours):

```
gcloud iam service-accounts create pulse-consumer

# The consumer's only permission: create objects in this bucket
gcloud storage buckets add-iam-policy-binding gs://<bucket> --member="serviceAccount:pulse-consumer@<project>.iam.gserviceaccount.com" --role="roles/storage.objectCreator"

# Let your own account act as the service account, then sign in as it
gcloud iam service-accounts add-iam-policy-binding pulse-consumer@<project>.iam.gserviceaccount.com --member="user:<your-email>" --role="roles/iam.serviceAccountTokenCreator"
gcloud services enable iamcredentials.googleapis.com
gcloud auth application-default login --impersonate-service-account=pulse-consumer@<project>.iam.gserviceaccount.com
```

Each run (PowerShell):

```
$env:LANDING_PATH = "gs://<bucket>/landing"
python -m consumer --once
```

Impersonation is used instead of a key file because downloadable keys are a common source of leaks, and new
Google Cloud organisations block them by default. A key file still works: set `GOOGLE_APPLICATION_CREDENTIALS`.

### Batch sources

Same `LANDING_PATH` and the same service account; no API keys are needed.

```
python -m batch_ingest --list                       # the sources
python -m batch_ingest                              # every source, caught up to today
python -m batch_ingest --source carbon_intensity    # just one (repeatable)
```

The first run fetches the last `INITIAL_LOOKBACK_DAYS` (7), or from `BACKFILL_START` if set; later runs
continue from each source's checkpoint in `state/batch/`. A source that fails keeps its checkpoint and the
others still run. To backfill three years, set `$env:BACKFILL_START = "2023-11-01"` before the first run:
about 4,600 requests in total and roughly 600 MB of raw responses (far less once compressed in Parquet),
mostly REMIT notices and wind forecasts. The one-day APIs (NDF, system prices, REMIT) make most of the
requests, so a full backfill takes around an hour.

All settings, with defaults, are in [.env.example](.env.example).

## Project layout

```
main.py, source.py, sink.py     producer: Elexon API -> Kafka
envelope.py                     the envelope the producer wraps each record in
checkpoint.py, config.py        producer state and settings
consumer/                       Kafka -> landing zone (local folder or gs://)
batch_ingest/                   batch sources -> landing zone; sources.py defines what each one requests
landing.py                      the landing-file writer shared by the consumer and batch ingest
tests/                          pytest suite, run by GitHub Actions on every push
databricks/                     bronze, silver and gold pipelines, the daily job, and their tests
```

## Testing

`pytest` covers the producer (API paging, raw publishing, compaction-safe keys, delivery failures,
checkpoints), the consumer (landing rows, fixed file schema, append-only storage, object-storage writes,
offset commits) and batch ingest (each source's requests and API limits, NESO's per-year tables, backfill
ranges, checkpoints that only advance after a window is stored).

## What I'd do next

- Check at startup that the consumer can write to the bucket, so a bad key fails immediately.
- Move the consumer onto a small cloud VM or Cloud Run so ingestion runs without a laptop.
