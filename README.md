# Electricity Pulse

**UK electricity generation by fuel type, streamed from the Elexon API through Kafka into a Google Cloud
Storage landing zone.** The medallion lakehouse (bronze, silver, gold) is built in Databricks on top of the
landing files.

```
┌────────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────────────┐   ┌──────────────────────────┐
│ Elexon     │──►│ producer     │──►│  Kafka   │──►│ consumer           │──►│ gs://<bucket>/landing/   │──► Databricks
│ FUELHH API │   │ python main  │   │ (Docker) │   │ python -m consumer │   │ fuelhh/ingest_date=…/    │
└────────────┘   └──────────────┘   └──────────┘   └────────────────────┘   └──────────────────────────┘
```

Every half-hour, Elexon publishes how much electricity each fuel type (wind, gas, nuclear, interconnectors
and so on) supplied to the GB grid, and revises those figures later. This repo gets every record, including
every revision, into GCS unchanged.

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

All settings, with defaults, are in [.env.example](.env.example).

## Project layout

```
main.py, source.py, sink.py     producer: Elexon API -> Kafka
envelope.py                     the envelope the producer wraps each record in
checkpoint.py, config.py        producer state and settings
consumer/                       Kafka -> landing zone (local folder or gs://)
tests/                          pytest suite, run by GitHub Actions on every push
```

## Testing

`pytest` covers the producer (API paging, raw publishing, compaction-safe keys, delivery failures,
checkpoints) and the consumer (landing rows, fixed file schema, append-only storage, object-storage writes,
offset commits).

## What I'd do next

- Check at startup that the consumer can write to the bucket, so a bad key fails immediately.
- Move the consumer onto a small cloud VM or Cloud Run so ingestion runs without a laptop.
