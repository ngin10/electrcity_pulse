"""Rebuild the 15 silver tables locally from data/landing, for exploration.

Mirrors databricks/src/pulse/sources.py: parse the JSON text, type each field, keep one row per key (the
highest sequence wins). It does not apply the hard rules: the EDA wants to see what they would catch.
Writes eda/out/silver/<table>.parquet and eda/out/build_stats.json (raw records vs rows kept).

    python eda/build_silver.py
"""
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
LANDING = (ROOT / 'data' / 'landing').as_posix()
OUT = ROOT / 'eda' / 'out' / 'silver'
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")


def ts(key):
    return f"CAST(TRY_CAST(rec->>'{key}' AS TIMESTAMPTZ) AS TIMESTAMP)"


def dbl(key):
    return f"TRY_CAST(rec->>'{key}' AS DOUBLE)"


def integer(key):
    return f"TRY_CAST(rec->>'{key}' AS INTEGER)"


def date(key):
    return f"TRY_CAST(rec->>'{key}' AS DATE)"


def text(key):
    return f"rec->>'{key}'"


SETTLEMENT = {'settlement_date': date('settlementDate'), 'settlement_period': integer('settlementPeriod'),
              'start_time': ts('startTime')}
BATCH_SEQ = ('fetched_at', 'source_file', 'record_index')

WIND_SOLAR = {
    'document_id': text('documentId'), 'document_revision_number': integer('documentRevisionNumber'),
    'publish_time': ts('publishTime'), 'business_type': text('businessType'), 'psr_type': text('psrType'),
    'quantity_mw': dbl('quantity'), **SETTLEMENT,
}
PRICES = {'system_sell_price_gbp_mwh': 'systemSellPrice', 'system_buy_price_gbp_mwh': 'systemBuyPrice',
          'net_imbalance_volume_mwh': 'netImbalanceVolume', 'reserve_scarcity_price_gbp_mwh': 'reserveScarcityPrice',
          'replacement_price_gbp_mwh': 'replacementPrice',
          'total_accepted_offer_volume_mwh': 'totalAcceptedOfferVolume',
          'total_accepted_bid_volume_mwh': 'totalAcceptedBidVolume'}
REMIT_TEXT = {'mrid': 'mrid', 'message_type': 'messageType', 'message_heading': 'messageHeading',
              'event_type': 'eventType', 'unavailability_type': 'unavailabilityType', 'event_status': 'eventStatus',
              'participant_id': 'participantId', 'asset_id': 'assetId', 'asset_type': 'assetType',
              'affected_unit': 'affectedUnit', 'fuel_type': 'fuelType', 'cause': 'cause',
              'related_information': 'relatedInformation'}
NESO = {'nd_mw': 'ND', 'tsd_mw': 'TSD', 'england_wales_demand_mw': 'ENGLAND_WALES_DEMAND',
        'embedded_wind_generation_mw': 'EMBEDDED_WIND_GENERATION',
        'embedded_wind_capacity_mw': 'EMBEDDED_WIND_CAPACITY',
        'embedded_solar_generation_mw': 'EMBEDDED_SOLAR_GENERATION',
        'embedded_solar_capacity_mw': 'EMBEDDED_SOLAR_CAPACITY', 'non_bm_stor_mw': 'NON_BM_STOR',
        'pump_storage_pumping_mw': 'PUMP_STORAGE_PUMPING', 'scottish_transfer_mw': 'SCOTTISH_TRANSFER',
        'ifa_flow_mw': 'IFA_FLOW', 'ifa2_flow_mw': 'IFA2_FLOW', 'britned_flow_mw': 'BRITNED_FLOW',
        'moyle_flow_mw': 'MOYLE_FLOW', 'east_west_flow_mw': 'EAST_WEST_FLOW', 'nemo_flow_mw': 'NEMO_FLOW',
        'nsl_flow_mw': 'NSL_FLOW', 'eleclink_flow_mw': 'ELECLINK_FLOW', 'viking_flow_mw': 'VIKING_FLOW',
        'greenlink_flow_mw': 'GREENLINK_FLOW'}
NESO_DATE = ("CAST(CASE length(rec->>'SETTLEMENT_DATE') "
             "WHEN 9 THEN try_strptime(rec->>'SETTLEMENT_DATE', '%d-%b-%y') "
             "WHEN 11 THEN try_strptime(rec->>'SETTLEMENT_DATE', '%d-%b-%Y') "
             "ELSE try_strptime(substr(rec->>'SETTLEMENT_DATE', 1, 10), '%Y-%m-%d') END AS DATE)")
CARBON_TS = "CAST(try_strptime(rec->>'{k}', '%Y-%m-%dT%H:%MZ') AS TIMESTAMP)"

# table -> (landing folder, JSON path of the records, fields, keys, sequence)
SOURCES = {
    'demand_outturn': ('elexon_demand_outturn', '$.data[*]', {
        **SETTLEMENT, 'publish_time': ts('publishTime'),
        'initial_demand_outturn_mw': dbl('initialDemandOutturn'),
        'initial_transmission_demand_outturn_mw': dbl('initialTransmissionSystemDemandOutturn')},
        ('settlement_date', 'settlement_period'), ('publish_time', *BATCH_SEQ)),
    'wind_forecast': ('elexon_wind_forecast', '$.data[*]', {
        'publish_time': ts('publishTime'), 'start_time': ts('startTime'), 'generation_mw': dbl('generation')},
        ('publish_time', 'start_time'), BATCH_SEQ),
    'demand_forecast': ('elexon_demand_forecast', '$.data[*]', {
        'publish_time': ts('publishTime'), **SETTLEMENT, 'boundary': text('boundary'), 'demand_mw': dbl('demand')},
        ('publish_time', 'settlement_date', 'settlement_period', 'boundary'), BATCH_SEQ),
    'wind_solar_forecast': ('elexon_wind_solar_forecast', '$.data[*]', {
        **WIND_SOLAR, 'process_type': text('processType')},
        ('document_id', 'process_type', 'psr_type', 'settlement_date', 'settlement_period'),
        ('document_revision_number', 'publish_time', *BATCH_SEQ)),
    'wind_solar_actual': ('elexon_wind_solar_actual', '$.data[*]', WIND_SOLAR,
                          ('psr_type', 'settlement_date', 'settlement_period'),
                          ('publish_time', 'document_revision_number', *BATCH_SEQ)),
    'system_prices': ('elexon_system_prices', '$.data[*]', {
        **SETTLEMENT, 'created_at': ts('createdDateTime'),
        **{name: dbl(key) for name, key in PRICES.items()},
        'price_derivation_code': text('priceDerivationCode'),
        'bsad_defaulted': "TRY_CAST(rec->>'bsadDefaulted' AS BOOLEAN)"},
        ('settlement_date', 'settlement_period'), ('created_at', *BATCH_SEQ)),
    'market_index': ('elexon_market_index', '$.data[*]', {
        'data_provider': text('dataProvider'), **SETTLEMENT, 'price_gbp_mwh': dbl('price'),
        'volume_mwh': dbl('volume')},
        ('data_provider', 'settlement_date', 'settlement_period'), BATCH_SEQ),
    'remit_outages': ('elexon_remit_outages', '$.data[*]', {
        **{name: text(key) for name, key in REMIT_TEXT.items()},
        'revision_number': integer('revisionNumber'), 'publish_time': ts('publishTime'),
        'created_time': ts('createdTime'), 'normal_capacity_mw': dbl('normalCapacity'),
        'available_capacity_mw': dbl('availableCapacity'), 'unavailable_capacity_mw': dbl('unavailableCapacity'),
        'event_start_time': ts('eventStartTime'), 'event_end_time': ts('eventEndTime'),
        'outage_profile_points': "json_array_length(rec->'outageProfile')"},
        ('mrid', 'revision_number'), ('publish_time', *BATCH_SEQ)),
    'carbon_intensity': ('carbon_intensity', '$.data[*]', {
        'period_start': CARBON_TS.format(k='from'), 'period_end': CARBON_TS.format(k='to'),
        'forecast_gco2_kwh': "TRY_CAST(rec->'intensity'->>'forecast' AS INTEGER)",
        'actual_gco2_kwh': "TRY_CAST(rec->'intensity'->>'actual' AS INTEGER)",
        'intensity_index': "lower(rec->'intensity'->>'index')"},
        ('period_start',), BATCH_SEQ),
    'neso_demand': ('neso_historic_demand', '$.result.records[*]', {
        'settlement_date': NESO_DATE, 'settlement_period': integer('SETTLEMENT_PERIOD'),
        **{name: dbl(key) for name, key in NESO.items()},
        'forecast_actual_indicator': text('FORECAST_ACTUAL_INDICATOR')},
        ('settlement_date', 'settlement_period'), BATCH_SEQ),
}


def records(folder, path):
    """One row per record of every response in a landing folder."""
    return f"""
        SELECT CAST(fetched_at AS TIMESTAMPTZ) AS fetched_at, filename AS source_file, http_status,
               unnest(recs) AS rec, unnest(range(len(recs))) AS record_index
        FROM (SELECT *, json_extract(raw_value, '{path}') AS recs
              FROM read_parquet('{LANDING}/{folder}/**/*.parquet', filename = true, hive_partitioning = false))"""


def build(table, select_sql, keys, sequence):
    """Type the records, count them, keep the winning row per key, write parquet."""
    con.execute(f'CREATE OR REPLACE TEMP TABLE typed AS {select_sql}')
    order = ', '.join(f'{c} DESC NULLS LAST' for c in sequence)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE silver AS SELECT * FROM typed
                    QUALIFY row_number() OVER (PARTITION BY {', '.join(keys)} ORDER BY {order}) = 1""")
    raw, kept = (con.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('typed', 'silver'))
    con.execute(f"COPY (SELECT * FROM silver ORDER BY {', '.join(keys)}) TO '{(OUT / table).as_posix()}.parquet'")
    print(f'{table:22s} raw records {raw:>10,}  rows {kept:>10,}')
    return {'raw_records': raw, 'rows': kept, 'keys': list(keys)}


stats = {}

# FUELHH: one Kafka message per record, wrapped in the producer's envelope
fuelhh_sql = f"""
    SELECT {date('settlementDate')} AS settlement_date, {integer('settlementPeriod')} AS settlement_period,
           {ts('startTime')} AS start_time, upper(trim(rec->>'fuelType')) AS fuel_type,
           {dbl('generation')} AS generation_mw, {ts('publishTime')} AS publish_time,
           ingested_at, kafka_partition, kafka_offset
    FROM (SELECT json_extract(raw_value, '$.payload') AS rec,
                 CAST(TRY_CAST(raw_value->>'ingested_at' AS TIMESTAMPTZ) AS TIMESTAMP) AS ingested_at,
                 kafka_partition, kafka_offset
          FROM read_parquet('{LANDING}/fuelhh/**/*.parquet', hive_partitioning = false))"""
stats['fuelhh'] = build('fuelhh', fuelhh_sql, ('settlement_date', 'settlement_period', 'fuel_type'),
                        ('publish_time', 'ingested_at', 'kafka_offset'))
# Every distinct value Elexon reported per key: how often figures are revised
con.execute(f"""COPY (SELECT settlement_date, fuel_type, count(*) AS messages,
                             count(DISTINCT generation_mw) AS distinct_values,
                             count(DISTINCT publish_time) AS distinct_publishes
                      FROM typed GROUP BY ALL HAVING count(*) > 0)
                TO '{(OUT / 'fuelhh_versions').as_posix()}.parquet'""")

for table, (folder, path, fields, keys, sequence) in SOURCES.items():
    columns = ', '.join(f'{sql} AS {name}' for name, sql in fields.items())
    sql = f'SELECT {columns}, CAST(fetched_at AS TIMESTAMP) AS fetched_at, source_file, record_index FROM ({records(folder, path)})'
    stats[table] = build(table, sql, keys, sequence)
    if table == 'remit_outages':
        con.execute(f"""COPY (SELECT * FROM silver QUALIFY row_number() OVER (
                                  PARTITION BY mrid ORDER BY revision_number DESC, publish_time DESC) = 1)
                        TO '{(OUT / 'remit_outages_current').as_posix()}.parquet'""")

# PV_Live: rows are arrays in the order `meta` names them
pv_sql = f"""
    SELECT TRY_CAST(row->>0 AS INTEGER) AS gsp_id, CAST(TRY_CAST(row->>1 AS TIMESTAMPTZ) AS TIMESTAMP) AS datetime_gmt,
           TRY_CAST(row->>2 AS DOUBLE) AS generation_mw, fetched_at, source_file, record_index
    FROM (SELECT CAST(CAST(fetched_at AS TIMESTAMPTZ) AS TIMESTAMP) AS fetched_at, filename AS source_file,
                 unnest(recs) AS row, unnest(range(len(recs))) AS record_index
          FROM (SELECT *, json_extract(raw_value, '$.data[*]') AS recs
                FROM read_parquet('{LANDING}/pv_live/**/*.parquet', filename = true, hive_partitioning = false)))"""
stats['pv_live'] = build('pv_live', pv_sql, ('gsp_id', 'datetime_gmt'), BATCH_SEQ)

# Bank holidays: {division: {events: [...]}}
bank_sql = f"""
    SELECT division, TRY_CAST(rec->>'date' AS DATE) AS holiday_date, rec->>'title' AS title, rec->>'notes' AS notes,
           TRY_CAST(rec->>'bunting' AS BOOLEAN) AS bunting, fetched_at, source_file, record_index
    FROM (SELECT division, fetched_at, source_file, unnest(recs) AS rec, unnest(range(len(recs))) AS record_index
          FROM (SELECT unnest(json_keys(raw_value)) AS division, raw_value,
                       CAST(CAST(fetched_at AS TIMESTAMPTZ) AS TIMESTAMP) AS fetched_at, filename AS source_file
                FROM read_parquet('{LANDING}/uk_bank_holidays/**/*.parquet', filename = true,
                                  hive_partitioning = false)),
               LATERAL (SELECT CAST(json_extract(raw_value, '$."' || division || '".events') AS JSON[]) AS recs))"""
stats['bank_holidays'] = build('bank_holidays', bank_sql, ('division', 'holiday_date'), BATCH_SEQ)

# Weather: 5 points per response, each with parallel hourly arrays; the point's position names it
POINTS = ('dogger_bank', 'east_anglia_offshore', 'irish_sea', 'scottish_highlands', 'southern_england')
VARIABLES = {'wind_speed_100m': 'wind_speed_100m_kmh', 'wind_speed_10m': 'wind_speed_10m_kmh',
             'shortwave_radiation': 'shortwave_radiation_wm2', 'temperature_2m': 'temperature_2m_c',
             'cloud_cover': 'cloud_cover_pct'}
for table, folder, suffix in (('weather_day_ahead', 'weather_forecast_day_ahead', '_previous_day1'),
                              ('weather_historical', 'weather_historical_forecast', '')):
    arrays = ', '.join(f"""unnest(CAST(json_extract(loc, '$.hourly.{v}{suffix}') AS DOUBLE[])) AS {c}"""
                       for v, c in VARIABLES.items())
    weather_sql = f"""
        SELECT list_extract({list(POINTS)}, location_index + 1) AS location,
               TRY_CAST(loc->>'latitude' AS DOUBLE) AS latitude, TRY_CAST(loc->>'longitude' AS DOUBLE) AS longitude,
               CAST(unnest(CAST(json_extract(loc, '$.hourly.time') AS VARCHAR[])) AS TIMESTAMP) AS time,
               {arrays}, fetched_at, source_file,
               unnest(range(CAST(json_array_length(loc, '$.hourly.time') AS BIGINT))) AS record_index
        FROM (SELECT CAST(CAST(fetched_at AS TIMESTAMPTZ) AS TIMESTAMP) AS fetched_at, filename AS source_file,
                     unnest(locs) AS loc, unnest(range(len(locs))) AS location_index
              FROM (SELECT *, json_extract(raw_value, '$[*]') AS locs
                    FROM read_parquet('{LANDING}/{folder}/**/*.parquet', filename = true,
                                      hive_partitioning = false)))"""
    stats[table] = build(table, weather_sql, ('location', 'time'), BATCH_SEQ)

# Landing facts per folder: responses, bytes, HTTP status, fetch window
for table, folder in {'fuelhh': 'fuelhh', 'pv_live': 'pv_live', 'bank_holidays': 'uk_bank_holidays',
                      'weather_day_ahead': 'weather_forecast_day_ahead',
                      'weather_historical': 'weather_historical_forecast',
                      **{t: s[0] for t, s in SOURCES.items()}}.items():
    files = list(Path(LANDING, folder).rglob('*.parquet'))
    src = f"read_parquet('{LANDING}/{folder}/**/*.parquet', hive_partitioning = false)"
    responses, raw_bytes = con.execute(f'SELECT count(*), sum(strlen(raw_value)) FROM {src}').fetchone()
    stats[table].update(landing_folder=folder, files=len(files), file_bytes=sum(f.stat().st_size for f in files),
                        responses=responses, raw_json_bytes=raw_bytes)
    if table != 'fuelhh':
        status = con.execute(f'SELECT http_status, count(*) FROM {src} GROUP BY 1').fetchall()
        stats[table]['http_status'] = {str(k): v for k, v in status}

(ROOT / 'eda' / 'out' / 'build_stats.json').write_text(json.dumps(stats, indent=1))
