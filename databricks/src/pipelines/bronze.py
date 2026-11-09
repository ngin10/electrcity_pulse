# Bronze pipeline: the GCS landing zone -> one raw, append-only Delta table per landing folder.
#
# Bronze keeps every row exactly as landed, duplicates included, plus where it came from (source_file) and
# when Databricks loaded it (loaded_at). It parses nothing and drops nothing: silver can always be rebuilt
# from it. Auto Loader tracks which files are loaded, so each file is loaded exactly once; the pipeline
# keeps its checkpoints and schema history itself.
#
# Pipeline configuration: pulse.src_path (the src/ folder, for imports), pulse.catalog, and pulse.landing (the
# GCS landing zone, or the volume tools/sync_landing.py copies it into; default: that volume).
import sys

sys.path.append(spark.conf.get('pulse.src_path'))  # noqa: F821 (spark is provided by the pipeline)

from pyspark.sql import functions as F  # noqa: E402

from pulse import pipeline_api as dp  # noqa: E402
from pulse.settings import (BATCH_HINTS, BRONZE_DESCRIPTIONS, BRONZE_TABLES, CATALOG_DEFAULT,  # noqa: E402
                            KAFKA_HINTS, KAFKA_TABLE, landing_volume)

CATALOG = spark.conf.get('pulse.catalog', CATALOG_DEFAULT)  # noqa: F821
LANDING = spark.conf.get('pulse.landing', None) or landing_volume(CATALOG)  # noqa: F821

# Observed, never enforced: bronze keeps bad rows so silver can report them
COMMON_EXPECTATIONS = {'no_rescued_data': '_rescued_data IS NULL', 'has_raw_value': 'raw_value IS NOT NULL'}
EXPECTATIONS = {
    KAFKA_TABLE: {**COMMON_EXPECTATIONS, 'has_offset': 'kafka_offset IS NOT NULL'},
    'batch': {**COMMON_EXPECTATIONS, 'http_ok': 'http_status = 200'},
}


def define_bronze_table(table, hints, expectations, description):
    # A function per table: each call keeps its own `table`, so the loop below can't leave every table
    # reading the last folder (Python looks up loop variables when the inner function runs, not when it's made)

    @dp.table(name=table, comment=f'{description}. Raw, append-only, not deduplicated.',
              cluster_by=['ingest_date'], table_properties={'quality': 'bronze'})
    @dp.expect_all(expectations)
    def bronze():
        return (spark.readStream.format('cloudFiles')  # noqa: F821
                .option('cloudFiles.format', 'parquet')
                # The landing folders are ingest_date=YYYY-MM-DD; the hints make that a DATE
                .option('cloudFiles.partitionColumns', 'ingest_date')
                .option('cloudFiles.schemaHints', hints)
                # A new column fails the update once, then is added (the job retries the task)
                .option('cloudFiles.schemaEvolutionMode', 'addNewColumns')
                .option('pathGlobFilter', '*.parquet')
                .load(f'{LANDING}/{table}')
                .select('*',
                        F.col('_metadata.file_path').alias('source_file'),
                        F.col('_metadata.file_modification_time').alias('source_file_modified_at'),
                        F.current_timestamp().alias('loaded_at')))

    return bronze


for name in BRONZE_TABLES:
    is_kafka = name == KAFKA_TABLE
    define_bronze_table(name, KAFKA_HINTS if is_kafka else BATCH_HINTS,
                        EXPECTATIONS[KAFKA_TABLE if is_kafka else 'batch'], BRONZE_DESCRIPTIONS[name])
