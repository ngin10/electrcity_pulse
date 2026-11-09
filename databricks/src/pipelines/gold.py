# Gold pipeline: silver -> business-ready materialized views. The logic is in pulse/gold.py.
#
# Materialized views are recomputed from silver on each update (incrementally where Databricks can), which
# suits gold: daily totals and features depend on whole days, not on single new rows.
#
# Pipeline configuration: pulse.src_path, pulse.catalog. Session time zone must be UTC (set in the pipeline's settings).
import sys

sys.path.append(spark.conf.get('pulse.src_path'))  # noqa: F821

from pulse import gold  # noqa: E402
from pulse import pipeline_api as dp  # noqa: E402
from pulse.settings import CATALOG_DEFAULT, schema  # noqa: E402

SILVER = schema(spark.conf.get('pulse.catalog', CATALOG_DEFAULT), 'silver')  # noqa: F821


def silver(table):
    return spark.read.table(f'{SILVER}.{table}')  # noqa: F821


def this_pipeline(table):
    return spark.read.table(table)  # noqa: F821


@dp.materialized_view(name='dim_date', comment='One row per UK date: calendar, bank holidays, clock changes.')
def dim_date():
    return gold.dim_date(silver('fuelhh'), silver('bank_holidays'))


@dp.materialized_view(
    name='grid_half_hourly', cluster_by=['settlement_date'],
    comment=('One row per settlement period: generation by fuel and category (MW), renewable share, demand, '
             'prices, carbon intensity, embedded wind and solar.'))
def grid_half_hourly():
    return gold.grid_half_hourly(silver('fuelhh'), silver('demand_outturn'), silver('neso_demand'),
                                 silver('system_prices'), silver('market_index'), silver('carbon_intensity'),
                                 silver('pv_live'))


@dp.materialized_view(
    name='generation_daily', cluster_by=['settlement_date'],
    comment=('One row per settlement date: energy by category (MWh) and the energy-weighted renewable share. '
             'Use is_complete before trusting a day.'))
def generation_daily():
    return gold.generation_daily(this_pipeline('grid_half_hourly'))


@dp.materialized_view(name='weather_daily', cluster_by=['date'],
                      comment='One row per kind (day_ahead, historical), location and UK date.')
def weather_daily():
    return gold.weather_daily(silver('weather_day_ahead'), silver('weather_historical'))


@dp.materialized_view(
    name='outages_day_ahead',
    comment='Per target date: MW unavailable that day by fuel group, from REMIT as known at the forecast cutoff.')
def outages_day_ahead():
    return gold.outages_day_ahead(silver('remit_outages'), this_pipeline('dim_date'))


@dp.materialized_view(
    name='features_day_ahead', cluster_by=['target_date'],
    comment=('One row per target date: features known at 18:00 UTC the day before, and the target '
             '(target_renewable_share). Train on rows where is_trainable.'))
def features_day_ahead():
    return gold.features_day_ahead(
        this_pipeline('dim_date'), this_pipeline('generation_daily'), this_pipeline('grid_half_hourly'),
        silver('wind_solar_forecast'), silver('wind_forecast'), silver('demand_forecast'),
        this_pipeline('weather_daily'), this_pipeline('outages_day_ahead'))
