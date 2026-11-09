# Silver pipeline: bronze JSON text -> typed, checked, deduplicated tables, one per source.
#
# For each source in pulse/sources.py:
#   <table>_checked  (view)    parsed, typed, every rule evaluated            pulse/quality.py: check
#   rejected         (table)   rows failing a hard rule, with the reasons      shared by every source
#   <table>_valid    (view)    rows passing them, in the final shape; the key contract stops the update if
#                              a key is ever NULL here, since that means a rule is broken, not the data
#   <table>                    AUTO CDC, SCD type 1: one row per key, the highest sequence wins, so a late
#                              replay can't overwrite a newer revision
#   <table>_history            SCD type 2, for sources with history_columns: every distinct version
#   <table>_current            for sources with current_keys (REMIT): the latest revision per message
# Plus completeness (periods present per day) and rejected_summary (reasons per day).
#
# Pipeline configuration: pulse.src_path. Session time zone must be UTC (set in the pipeline's settings).
import sys

sys.path.append(spark.conf.get('pulse.src_path'))  # noqa: F821

from functools import reduce  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402

from pulse import pipeline_api as dp  # noqa: E402
from pulse.quality import check, daily_completeness, rejected_rows, valid_rows  # noqa: E402
from pulse.settings import BRONZE  # noqa: E402
from pulse.sources import SOURCES  # noqa: E402

dp.create_streaming_table(
    name='rejected',
    comment=('Every record any silver table rejected, with all its reasons and the record as received. Fix the '
             'rule or the source, then full-refresh silver: rows come back from bronze.'),
    cluster_by=['source_table', 'ingest_date'],
)


def define_silver_table(source):
    checked, valid = f'{source.table}_checked', f'{source.table}_valid'

    @dp.temporary_view(name=checked)
    def _checked():
        return check(source, spark.readStream.table(f'{BRONZE}.{source.bronze_table}'))  # noqa: F821

    @dp.append_flow(target='rejected', name=f'rejected_{source.table}')
    def _rejected():
        return rejected_rows(source, spark.readStream.table(checked))  # noqa: F821

    @dp.temporary_view(name=valid)
    @dp.expect_all_or_fail({'key_present': ' AND '.join(f'{key} IS NOT NULL' for key in source.keys)})
    def _valid():
        return valid_rows(source, spark.readStream.table(checked))  # noqa: F821

    dp.create_streaming_table(name=source.table, comment=source.comment, cluster_by=list(source.cluster_by))
    dp.create_auto_cdc_flow(target=source.table, source=valid, keys=list(source.keys),
                            sequence_by=F.struct(*source.sequence), stored_as_scd_type=1)

    if source.history_columns:
        dp.create_streaming_table(
            name=f'{source.table}_history', cluster_by=list(source.cluster_by),
            comment=(f'Every distinct version of {source.table}, valid from __START_AT to __END_AT (NULL: current). '
                     f'A new version only when {", ".join(source.history_columns)} changes.'))
        dp.create_auto_cdc_flow(target=f'{source.table}_history', source=valid, keys=list(source.keys),
                                sequence_by=F.struct(*source.sequence), stored_as_scd_type=2,
                                track_history_column_list=list(source.history_columns))

    if source.current_keys:
        dp.create_streaming_table(
            name=f'{source.table}_current', cluster_by=list(source.cluster_by),
            comment=f'The latest version of each {", ".join(source.current_keys)} in {source.table}.')
        dp.create_auto_cdc_flow(target=f'{source.table}_current', source=valid, keys=list(source.current_keys),
                                sequence_by=F.struct(*source.current_sequence), stored_as_scd_type=1)


for silver_source in SOURCES:
    define_silver_table(silver_source)

# Sources with one row per settlement period, where a missing period is a gap
PERIOD_TABLES = ('fuelhh', 'demand_outturn', 'system_prices', 'neso_demand')


@dp.materialized_view(name='completeness', cluster_by=['source_table', 'settlement_date'],
                      comment=('Per source and settlement date: periods present vs expected (46, 48 or 50). '
                               'A gap has no row, so no row rule can find it.'))
def completeness():
    tables = [daily_completeness(spark.read.table(t), t) for t in PERIOD_TABLES]  # noqa: F821
    return reduce(lambda a, b: a.unionByName(b), tables)


@dp.materialized_view(name='rejected_summary',
                      comment='Rejected rows per source, landing date and reason: the trend to alert on.')
def rejected_summary():
    return (spark.read.table('rejected')  # noqa: F821
            .select('source_table', 'ingest_date', F.explode('reject_reasons').alias('reason'))
            .groupBy('source_table', 'ingest_date', 'reason').count())
