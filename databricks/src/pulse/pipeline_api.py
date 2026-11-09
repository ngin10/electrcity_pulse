"""The Lakeflow Declarative Pipelines API under one set of names, whichever module the runtime provides.

Newer runtimes ship `pyspark.pipelines`; older ones only `dlt`, where a few functions have older names
(apply_changes for create_auto_cdc_flow, view for temporary_view, table for materialized_view). Only
importable inside a pipeline.
"""
try:
    from pyspark import pipelines as _dp
except ImportError:
    import dlt as _dp

table = _dp.table
materialized_view = getattr(_dp, 'materialized_view', None) or _dp.table
temporary_view = getattr(_dp, 'temporary_view', None) or _dp.view
create_streaming_table = _dp.create_streaming_table
create_auto_cdc_flow = getattr(_dp, 'create_auto_cdc_flow', None) or _dp.apply_changes
append_flow = _dp.append_flow
expect_all = _dp.expect_all
expect_all_or_fail = _dp.expect_all_or_fail
