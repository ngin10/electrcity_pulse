"""What each silver table is: how to read its bronze responses, its fields, rules, key and ordering.

Every schema here was written from a live response of the API (fetched November 2026). Names follow one
convention: snake_case, units in the name (_mw, _mwh, _gbp_mwh, _gco2_kwh), *_time / *_at for UTC instants,
*_date for UK settlement dates.

Keys and sequences, in short:
- Actual values (generation, demand, prices) have one row per period; the latest revision wins.
- Forecasts keep their publish time in the key: a day-ahead model may only use forecasts made before its
  cutoff, so silver must keep every forecast, not just the latest (which would leak the future).
- Each sequence ends with a unique tiebreaker (Kafka offset, or file plus record position), so the same
  input always gives the same output.
"""
from pulse.quality import KAFKA_LINEAGE, Field, SilverSource, outside, period_start_utc, periods_in_day, sql_list
from pulse.settings import KNOWN_FUEL_TYPES, WEATHER_POINTS

# --- Pieces several sources share -----------------------------------------------------------------------

ELEXON_RESPONSE = 'data ARRAY<STRING>'
ELEXON_RECORDS = 'response.data'
HTTP_OK = {'http_not_ok': 'http_status <> 200'}
BATCH_SEQUENCE = ('fetched_at', 'source_file', 'record_index')
FETCHED_AT = {'fetched_at': Field('fetched_at', 'TIMESTAMP')}


def settlement_fields(start_time=True):
    fields = {'settlement_date': Field('rec.settlementDate', 'DATE'),
              'settlement_period': Field('rec.settlementPeriod', 'INT')}
    if start_time:
        fields['start_time'] = Field('rec.startTime', 'TIMESTAMP')
    return fields


# Period 49 is real on the autumn clock-change day and impossible on any other day
PERIOD_RULE = {'period_out_of_range': f"settlement_period NOT BETWEEN 1 AND {periods_in_day('settlement_date')}"}
# The same instant written twice in a record must agree
START_RULE = {'start_time_mismatch': f"start_time <> {period_start_utc('settlement_date', 'settlement_period')}"}


def _source(**kwargs):
    """A batch source: responses must be HTTP 200, ordered by fetch time, clustered by key."""
    kwargs.setdefault('response_rules', HTTP_OK)
    kwargs.setdefault('sequence', BATCH_SEQUENCE)
    kwargs.setdefault('cluster_by', kwargs['keys'][:2])
    kwargs['fields'] = {**kwargs['fields'], **FETCHED_AT}
    return SilverSource(**kwargs)


# --- FUELHH: Kafka messages, one record each --------------------------------------------------------------

FUELHH = SilverSource(
    table='fuelhh',
    bronze_table='fuelhh',
    comment=('Elexon generation by fuel type (MW). Grain: one row per settlement date, period and fuel type; '
             'the latest revision. Interconnector (INT*) values are flows: positive import, negative export.'),
    # The envelope written by the producer (envelope.py); the Elexon record is in payload
    response_schema=('schema_version STRING, source STRING, dataset STRING, ingested_at STRING, '
                     'fetch_window STRING, payload STRING'),
    records='array(response.payload)',
    record_fields=('dataset', 'publishTime', 'startTime', 'settlementDate', 'settlementPeriod', 'fuelType',
                   'generation'),
    fields={
        **settlement_fields(),
        'fuel_type': Field('upper(trim(rec.fuelType))'),
        'generation_mw': Field('rec.generation', 'DOUBLE'),
        'publish_time': Field('rec.publishTime', 'TIMESTAMP'),
        'ingested_at': Field('response.ingested_at', 'TIMESTAMP'),
    },
    required=('settlement_date', 'settlement_period', 'fuel_type', 'generation_mw', 'publish_time'),
    response_rules={
        # Only version 1 of the envelope is understood; a new version waits in quarantine until it is
        'unsupported_schema_version': "is_json AND coalesce(response.schema_version, '') NOT IN ('1')",
    },
    hard_rules={
        **PERIOD_RULE,
        'wrong_dataset': "coalesce(rec.dataset, 'FUELHH') <> 'FUELHH'",
        'generation_implausible': outside('generation_mw', -10000, 40000),
    },
    soft_rules={
        **START_RULE,
        # New interconnectors appear every few years: keep the row (it is real supply) and flag it
        'unknown_fuel_type': f'fuel_type NOT IN ({sql_list(KNOWN_FUEL_TYPES)})',
        # Interconnectors go negative when GB exports; anything else negative is unusual
        'negative_generation': "generation_mw < 0 AND fuel_type NOT LIKE 'INT%'",
        'future_settlement_date': 'settlement_date > date_add(current_date(), 2)',
    },
    keys=('settlement_date', 'settlement_period', 'fuel_type'),
    # Elexon's publish time first, then when we fetched it; the offset breaks any remaining tie
    sequence=('publish_time', 'ingested_at', 'kafka_offset'),
    lineage=KAFKA_LINEAGE,
    cluster_by=('settlement_date',),
    # Every value Elexon ever reported for a period: for backtests "as known at the time"
    history_columns=('generation_mw',),
)

# --- Elexon batch datasets ----------------------------------------------------------------------------------

DEMAND_OUTTURN = _source(
    table='demand_outturn',
    bronze_table='elexon_demand_outturn',
    comment=('Elexon initial national demand outturn (MW). Grain: one row per settlement date and period; '
             'the latest publish.'),
    response_schema=ELEXON_RESPONSE,
    records=ELEXON_RECORDS,
    record_fields=('publishTime', 'startTime', 'settlementDate', 'settlementPeriod', 'initialDemandOutturn',
                   'initialTransmissionSystemDemandOutturn'),
    fields={
        **settlement_fields(),
        'publish_time': Field('rec.publishTime', 'TIMESTAMP'),
        'initial_demand_outturn_mw': Field('rec.initialDemandOutturn', 'DOUBLE'),
        'initial_transmission_demand_outturn_mw': Field('rec.initialTransmissionSystemDemandOutturn', 'DOUBLE'),
    },
    required=('settlement_date', 'settlement_period', 'initial_demand_outturn_mw'),
    hard_rules={**PERIOD_RULE, 'demand_implausible': outside('initial_demand_outturn_mw', 0, 80000)},
    soft_rules=START_RULE,
    keys=('settlement_date', 'settlement_period'),
    sequence=('publish_time', *BATCH_SEQUENCE),
)

WIND_FORECAST = _source(
    table='wind_forecast',
    bronze_table='elexon_wind_forecast',
    comment=('Elexon WINDFOR wind generation forecast (MW). Grain: one row per publish time and forecast '
             'start time: every forecast is kept, so features can use only those made before a cutoff.'),
    response_schema=ELEXON_RESPONSE,
    records=ELEXON_RECORDS,
    record_fields=('dataset', 'publishTime', 'startTime', 'generation'),
    fields={
        'publish_time': Field('rec.publishTime', 'TIMESTAMP'),
        'start_time': Field('rec.startTime', 'TIMESTAMP'),
        'generation_mw': Field('rec.generation', 'DOUBLE'),
    },
    required=('publish_time', 'start_time', 'generation_mw'),
    hard_rules={'generation_implausible': outside('generation_mw', 0, 50000)},
    keys=('publish_time', 'start_time'),
    cluster_by=('start_time',),
)

DEMAND_FORECAST = _source(
    table='demand_forecast',
    bronze_table='elexon_demand_forecast',
    comment=('Elexon NDF national demand forecast (MW), publishes 17:00-18:00 UTC. Grain: one row per publish '
             'time, settlement date, period and boundary.'),
    response_schema=ELEXON_RESPONSE,
    records=ELEXON_RECORDS,
    record_fields=('dataset', 'demand', 'publishTime', 'startTime', 'settlementDate', 'settlementPeriod',
                   'boundary'),
    fields={
        'publish_time': Field('rec.publishTime', 'TIMESTAMP'),
        **settlement_fields(),
        'boundary': Field('rec.boundary'),
        'demand_mw': Field('rec.demand', 'DOUBLE'),
    },
    required=('publish_time', 'settlement_date', 'settlement_period', 'demand_mw'),
    hard_rules={**PERIOD_RULE, 'demand_implausible': outside('demand_mw', 0, 80000)},
    soft_rules=START_RULE,
    keys=('publish_time', 'settlement_date', 'settlement_period', 'boundary'),
    cluster_by=('settlement_date',),
)

_DGWS_AGWS_FIELDS = ('dataset', 'documentId', 'documentRevisionNumber', 'publishTime', 'businessType',
                     'psrType', 'quantity', 'startTime', 'settlementDate', 'settlementPeriod')
PSR_TYPES = ('Solar', 'Wind Offshore', 'Wind Onshore')


def _wind_solar_fields():
    return {
        'document_id': Field('rec.documentId'),
        'document_revision_number': Field('rec.documentRevisionNumber', 'INT'),
        'publish_time': Field('rec.publishTime', 'TIMESTAMP'),
        'business_type': Field('rec.businessType'),
        'psr_type': Field('rec.psrType'),
        'quantity_mw': Field('rec.quantity', 'DOUBLE'),
        **settlement_fields(),
    }


WIND_SOLAR_FORECAST = _source(
    table='wind_solar_forecast',
    bronze_table='elexon_wind_solar_forecast',
    comment=('Elexon DGWS wind and solar generation forecasts (MW): day-ahead, intraday process and intraday '
             'total. Grain: one row per forecast document, process type, technology and settlement period.'),
    response_schema=ELEXON_RESPONSE,
    records=ELEXON_RECORDS,
    record_fields=_DGWS_AGWS_FIELDS + ('processType',),
    fields={**_wind_solar_fields(), 'process_type': Field('rec.processType')},
    required=('document_id', 'publish_time', 'process_type', 'psr_type', 'quantity_mw', 'settlement_date',
              'settlement_period'),
    hard_rules={**PERIOD_RULE, 'quantity_implausible': outside('quantity_mw', 0, 50000)},
    soft_rules={**START_RULE,
                'unknown_process_type': "process_type NOT IN ('Day ahead', 'Intraday process', 'Intraday total')",
                'unknown_psr_type': f'psr_type NOT IN ({sql_list(PSR_TYPES)})'},
    keys=('document_id', 'process_type', 'psr_type', 'settlement_date', 'settlement_period'),
    # A revised document replaces the earlier revision
    sequence=('document_revision_number', 'publish_time', *BATCH_SEQUENCE),
    cluster_by=('settlement_date',),
)

WIND_SOLAR_ACTUAL = _source(
    table='wind_solar_actual',
    bronze_table='elexon_wind_solar_actual',
    comment=('Elexon AGWS actual wind and solar generation (MW). Grain: one row per technology and settlement '
             'period; the latest publish.'),
    response_schema=ELEXON_RESPONSE,
    records=ELEXON_RECORDS,
    record_fields=_DGWS_AGWS_FIELDS,
    fields=_wind_solar_fields(),
    required=('psr_type', 'quantity_mw', 'settlement_date', 'settlement_period', 'publish_time'),
    hard_rules={**PERIOD_RULE, 'quantity_implausible': outside('quantity_mw', 0, 50000)},
    soft_rules={**START_RULE, 'unknown_psr_type': f'psr_type NOT IN ({sql_list(PSR_TYPES)})'},
    keys=('psr_type', 'settlement_date', 'settlement_period'),
    sequence=('publish_time', 'document_revision_number', *BATCH_SEQUENCE),
    cluster_by=('settlement_date',),
)

_PRICE_FIELDS = {
    'system_sell_price_gbp_mwh': 'systemSellPrice',
    'system_buy_price_gbp_mwh': 'systemBuyPrice',
    'net_imbalance_volume_mwh': 'netImbalanceVolume',
    'reserve_scarcity_price_gbp_mwh': 'reserveScarcityPrice',
    'replacement_price_gbp_mwh': 'replacementPrice',
    'total_accepted_offer_volume_mwh': 'totalAcceptedOfferVolume',
    'total_accepted_bid_volume_mwh': 'totalAcceptedBidVolume',
}

SYSTEM_PRICES = _source(
    table='system_prices',
    bronze_table='elexon_system_prices',
    comment=('Elexon system (imbalance) prices and volumes. Grain: one row per settlement date and period; '
             'the latest calculation. Since 2015 GB has a single price, so sell and buy prices are equal.'),
    response_schema=ELEXON_RESPONSE,
    records=ELEXON_RECORDS,
    record_fields=('settlementDate', 'settlementPeriod', 'startTime', 'createdDateTime', 'systemSellPrice',
                   'systemBuyPrice', 'bsadDefaulted', 'priceDerivationCode', 'reserveScarcityPrice',
                   'netImbalanceVolume', 'sellPriceAdjustment', 'buyPriceAdjustment', 'replacementPrice',
                   'replacementPriceReferenceVolume', 'totalAcceptedOfferVolume', 'totalAcceptedBidVolume',
                   'totalAdjustmentSellVolume', 'totalAdjustmentBuyVolume',
                   'totalSystemTaggedAcceptedOfferVolume', 'totalSystemTaggedAcceptedBidVolume',
                   'totalSystemTaggedAdjustmentSellVolume', 'totalSystemTaggedAdjustmentBuyVolume'),
    fields={
        **settlement_fields(),
        'created_at': Field('rec.createdDateTime', 'TIMESTAMP'),
        **{name: Field(f'rec.{key}', 'DOUBLE') for name, key in _PRICE_FIELDS.items()},
        'price_derivation_code': Field('rec.priceDerivationCode'),
        'bsad_defaulted': Field('rec.bsadDefaulted', 'BOOLEAN'),
    },
    required=('settlement_date', 'settlement_period', 'system_sell_price_gbp_mwh'),
    hard_rules={**PERIOD_RULE, 'price_implausible': outside('system_sell_price_gbp_mwh', -10000, 20000)},
    soft_rules=START_RULE,
    keys=('settlement_date', 'settlement_period'),
    sequence=('created_at', *BATCH_SEQUENCE),
)

MARKET_INDEX = _source(
    table='market_index',
    bronze_table='elexon_market_index',
    comment=('Elexon market index price (GBP/MWh) and volume (MWh). Grain: one row per data provider, '
             'settlement date and period. APXMIDP carries the prices; N2EXMIDP is mostly zero.'),
    response_schema=ELEXON_RESPONSE,
    records=ELEXON_RECORDS,
    record_fields=('dataset', 'startTime', 'dataProvider', 'settlementDate', 'settlementPeriod', 'price',
                   'volume'),
    fields={
        'data_provider': Field('rec.dataProvider'),
        **settlement_fields(),
        'price_gbp_mwh': Field('rec.price', 'DOUBLE'),
        'volume_mwh': Field('rec.volume', 'DOUBLE'),
    },
    required=('data_provider', 'settlement_date', 'settlement_period', 'price_gbp_mwh'),
    hard_rules={**PERIOD_RULE,
                'price_implausible': outside('price_gbp_mwh', -10000, 20000),
                'negative_volume': 'volume_mwh < 0'},
    soft_rules={**START_RULE, 'unknown_provider': "data_provider NOT IN ('APXMIDP', 'N2EXMIDP')"},
    keys=('data_provider', 'settlement_date', 'settlement_period'),
)

_REMIT_TEXT = {
    'mrid': 'mrid', 'message_type': 'messageType', 'message_heading': 'messageHeading',
    'event_type': 'eventType', 'unavailability_type': 'unavailabilityType', 'event_status': 'eventStatus',
    'participant_id': 'participantId', 'asset_id': 'assetId', 'asset_type': 'assetType',
    'affected_unit': 'affectedUnit', 'fuel_type': 'fuelType', 'cause': 'cause',
    'related_information': 'relatedInformation',
}

REMIT_OUTAGES = _source(
    table='remit_outages',
    bronze_table='elexon_remit_outages',
    comment=('Elexon REMIT unavailability notices. Grain: one row per message (mrid) and revision. '
             'remit_outages_current has the latest revision of each message.'),
    response_schema=ELEXON_RESPONSE,
    records=ELEXON_RECORDS,
    record_fields=('dataset', 'mrid', 'revisionNumber', 'publishTime', 'createdTime', 'messageType',
                   'messageHeading', 'eventType', 'unavailabilityType', 'participantId', 'registrationCode',
                   'assetId', 'assetType', 'affectedUnit', 'affectedUnitEIC', 'biddingZone', 'fuelType',
                   'normalCapacity', 'availableCapacity', 'unavailableCapacity', 'eventStatus',
                   'eventStartTime', 'eventEndTime', 'cause', 'relatedInformation', 'outageProfile',
                   'affectedArea', 'durationUncertainty'),
    fields={
        **{name: Field(f'rec.{key}') for name, key in _REMIT_TEXT.items()},
        'revision_number': Field('rec.revisionNumber', 'INT'),
        'publish_time': Field('rec.publishTime', 'TIMESTAMP'),
        'created_time': Field('rec.createdTime', 'TIMESTAMP'),
        'normal_capacity_mw': Field('rec.normalCapacity', 'DOUBLE'),
        'available_capacity_mw': Field('rec.availableCapacity', 'DOUBLE'),
        'unavailable_capacity_mw': Field('rec.unavailableCapacity', 'DOUBLE'),
        'event_start_time': Field('rec.eventStartTime', 'TIMESTAMP'),
        'event_end_time': Field('rec.eventEndTime', 'TIMESTAMP'),
        # Capacity over time within the event, kept as JSON text: parse it where it is needed
        'outage_profile_json': Field('rec.outageProfile'),
    },
    required=('mrid', 'revision_number', 'publish_time'),
    hard_rules={'negative_unavailable_capacity': 'unavailable_capacity_mw < 0'},
    soft_rules={
        'event_ends_before_start': 'event_end_time < event_start_time',
        'unknown_event_status': "event_status NOT IN ('Active', 'Inactive', 'Dismissed')",
        'unavailable_exceeds_normal': 'unavailable_capacity_mw > normal_capacity_mw',
    },
    keys=('mrid', 'revision_number'),
    sequence=('publish_time', *BATCH_SEQUENCE),
    cluster_by=('event_start_time',),
    current_keys=('mrid',),
    current_sequence=('revision_number', 'publish_time', *BATCH_SEQUENCE),
)

# --- Other batch sources ------------------------------------------------------------------------------------

BANK_HOLIDAYS = _source(
    table='bank_holidays',
    bronze_table='uk_bank_holidays',
    comment='UK bank holidays. Grain: one row per division (nation) and date.',
    # {division: {division, events: [...]}}: a map, so a new division needs no code change
    response_schema='MAP<STRING, STRING>',
    records=("flatten(transform(map_entries(response), d -> transform("
             "from_json(d.value, 'events ARRAY<STRING>').events, "
             "e -> to_json(named_struct('division', d.key, "
             "'title', get_json_object(e, '$.title'), 'date', get_json_object(e, '$.date'), "
             "'notes', get_json_object(e, '$.notes'), 'bunting', get_json_object(e, '$.bunting'))))))"),
    record_fields=('division', 'title', 'date', 'notes', 'bunting'),
    fields={
        'division': Field('rec.division'),
        'holiday_date': Field('rec.date', 'DATE'),
        'title': Field('rec.title'),
        'notes': Field('rec.notes'),
        'bunting': Field('rec.bunting', 'BOOLEAN'),
    },
    required=('division', 'holiday_date', 'title'),
    soft_rules={'unknown_division': "division NOT IN ('england-and-wales', 'scotland', 'northern-ireland')"},
    keys=('division', 'holiday_date'),
)

CARBON_INTENSITY = _source(
    table='carbon_intensity',
    bronze_table='carbon_intensity',
    comment=('GB carbon intensity (gCO2/kWh), forecast and actual. Grain: one row per half-hour (period_start); '
             'the latest fetch. actual is NULL until it has been measured.'),
    response_schema='data ARRAY<STRING>',
    records='response.data',
    record_fields=('from', 'to', 'intensity'),
    fields={
        # The API's own format, without seconds; anything else is a type error, never a guess
        'period_start': Field('rec.`from`', 'TIMESTAMP', fmt="yyyy-MM-dd'T'HH:mmX"),
        'period_end': Field('rec.`to`', 'TIMESTAMP', fmt="yyyy-MM-dd'T'HH:mmX"),
        'forecast_gco2_kwh': Field("get_json_object(rec.intensity, '$.forecast')", 'INT'),
        'actual_gco2_kwh': Field("get_json_object(rec.intensity, '$.actual')", 'INT'),
        'intensity_index': Field("lower(get_json_object(rec.intensity, '$.index'))"),
    },
    required=('period_start', 'period_end', 'forecast_gco2_kwh'),
    hard_rules={
        'period_not_30_min': 'period_end <> period_start + INTERVAL 30 MINUTES',
        'intensity_implausible': (f"{outside('forecast_gco2_kwh', 0, 1000)} OR "
                                  f"{outside('actual_gco2_kwh', 0, 1000)}"),
    },
    soft_rules={
        # A NULL with a meaning: the latest half-hours are not measured yet, and a later fetch fills them in
        'actual_not_yet_published': 'actual_gco2_kwh IS NULL',
        'unknown_index': "intensity_index NOT IN ('very low', 'low', 'moderate', 'high', 'very high')",
    },
    keys=('period_start',),
)

PV_LIVE = _source(
    table='pv_live',
    bronze_table='pv_live',
    comment=('Sheffield Solar PV_Live national solar generation estimate (MW). Grain: one row per GSP and '
             'datetime_gmt, which PV_Live uses for the end of each half-hour.'),
    # Rows are arrays in the order meta names them; the meta is checked so a reorder can't shift columns
    response_schema='meta ARRAY<STRING>, data ARRAY<STRING>',
    records=("transform(response.data, r -> to_json(named_struct("
             "'gsp_id', get_json_object(r, '$[0]'), 'datetime_gmt', get_json_object(r, '$[1]'), "
             "'generation_mw', get_json_object(r, '$[2]'))))"),
    record_fields=('gsp_id', 'datetime_gmt', 'generation_mw'),
    fields={
        'gsp_id': Field('rec.gsp_id', 'INT'),
        'datetime_gmt': Field('rec.datetime_gmt', 'TIMESTAMP'),
        'generation_mw': Field('rec.generation_mw', 'DOUBLE'),
    },
    required=('gsp_id', 'datetime_gmt', 'generation_mw'),
    response_rules={**HTTP_OK,
                    'unexpected_columns': "is_json AND NOT (response.meta <=> array('gsp_id', 'datetime_gmt', "
                                          "'generation_mw'))"},
    hard_rules={'generation_implausible': outside('generation_mw', 0, 40000)},
    keys=('gsp_id', 'datetime_gmt'),
    cluster_by=('datetime_gmt',),
)

_NESO_VALUES = {
    'nd_mw': 'ND', 'tsd_mw': 'TSD', 'england_wales_demand_mw': 'ENGLAND_WALES_DEMAND',
    'embedded_wind_generation_mw': 'EMBEDDED_WIND_GENERATION', 'embedded_wind_capacity_mw': 'EMBEDDED_WIND_CAPACITY',
    'embedded_solar_generation_mw': 'EMBEDDED_SOLAR_GENERATION',
    'embedded_solar_capacity_mw': 'EMBEDDED_SOLAR_CAPACITY',
    'non_bm_stor_mw': 'NON_BM_STOR', 'pump_storage_pumping_mw': 'PUMP_STORAGE_PUMPING',
    'scottish_transfer_mw': 'SCOTTISH_TRANSFER',
    'ifa_flow_mw': 'IFA_FLOW', 'ifa2_flow_mw': 'IFA2_FLOW', 'britned_flow_mw': 'BRITNED_FLOW',
    'moyle_flow_mw': 'MOYLE_FLOW', 'east_west_flow_mw': 'EAST_WEST_FLOW', 'nemo_flow_mw': 'NEMO_FLOW',
    'nsl_flow_mw': 'NSL_FLOW', 'eleclink_flow_mw': 'ELECLINK_FLOW', 'viking_flow_mw': 'VIKING_FLOW',
    'greenlink_flow_mw': 'GREENLINK_FLOW',
}
# NESO's yearly tables store the date three ways: '01-NOV-2021' (to 2022), '01-Nov-23' (2023) and
# '2024-11-01' (from 2024). Month names parse case-insensitively
_NESO_DATE = ("CAST(CASE length({v}) WHEN 9 THEN try_to_timestamp({v}, 'dd-MMM-yy') "
              "WHEN 11 THEN try_to_timestamp({v}, 'dd-MMM-yyyy') "
              "ELSE try_to_timestamp({v}, 'yyyy-MM-dd') END AS DATE)")

NESO_DEMAND = _source(
    table='neso_demand',
    bronze_table='neso_historic_demand',
    comment=('NESO historic demand (MW), with embedded wind and solar and interconnector flows. Grain: one row '
             'per settlement date and period; the latest fetch.'),
    response_schema='success STRING, result STRUCT<records: ARRAY<STRING>>',
    records='response.result.records',
    record_fields=('_id', '_full_text', 'SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'FORECAST_ACTUAL_INDICATOR',
                   *_NESO_VALUES.values()),
    fields={
        'settlement_date': Field('rec.SETTLEMENT_DATE', 'DATE', typed_sql=_NESO_DATE),
        'settlement_period': Field('rec.SETTLEMENT_PERIOD', 'INT'),
        **{name: Field(f'rec.{key}', 'DOUBLE') for name, key in _NESO_VALUES.items()},
        # 'F' marks forecast rows NESO adds ahead of the actuals in recent years; older years have none
        'forecast_actual_indicator': Field('rec.FORECAST_ACTUAL_INDICATOR'),
    },
    required=('settlement_date', 'settlement_period', 'nd_mw'),
    response_rules={**HTTP_OK, 'api_not_success': "is_json AND coalesce(response.success, '') <> 'true'"},
    hard_rules={**PERIOD_RULE, 'demand_implausible': outside('nd_mw', 0, 80000)},
    soft_rules={'forecast_not_actual': "forecast_actual_indicator = 'F'"},
    keys=('settlement_date', 'settlement_period'),
)

# --- Weather: one response holds 5 points, each with parallel hourly arrays ----------------------------

WEATHER_VARIABLES = (
    ('wind_speed_100m', 'wind_speed_100m_kmh', 0, 400),
    ('wind_speed_10m', 'wind_speed_10m_kmh', 0, 400),
    ('shortwave_radiation', 'shortwave_radiation_wm2', 0, 1500),
    ('temperature_2m', 'temperature_2m_c', -60, 60),
    ('cloud_cover', 'cloud_cover_pct', 0, 100),
)
_POINT_NAMES = sql_list(name for name, _, _ in WEATHER_POINTS)


def _expected(index):
    """SQL: the expected latitude (index 1) or longitude (index 2) for the row's location."""
    cases = ' '.join(f"WHEN '{p[0]}' THEN {p[index]}" for p in WEATHER_POINTS)
    return f'CASE location {cases} END'


def _weather(table, bronze_table, suffix, comment):
    hourly = ', '.join(f'`{v}{suffix}`: ARRAY<STRING>' for v, *_ in WEATHER_VARIABLES)
    values = ', '.join(f"'{v}', try_element_at(loc.hourly.`{v}{suffix}`, j + 1)" for v, *_ in WEATHER_VARIABLES)
    return _source(
        table=table,
        bronze_table=bronze_table,
        comment=comment,
        response_schema=(f'ARRAY<STRUCT<latitude: STRING, longitude: STRING, '
                         f'hourly: STRUCT<time: ARRAY<STRING>, {hourly}>>>'),
        # One record per point and hour: the hour's position j picks the same position in every array
        records=("flatten(transform(response, (loc, i) -> transform(loc.hourly.time, (t, j) -> to_json("
                 f"named_struct('location_index', i, 'latitude', loc.latitude, 'longitude', loc.longitude, "
                 f"'time', t, {values})))))"),
        record_fields=('location_index', 'latitude', 'longitude', 'time', *(v for v, *_ in WEATHER_VARIABLES)),
        fields={
            # The response lists points in request order, which is WEATHER_POINTS' order
            'location': Field(f'try_element_at(array({_POINT_NAMES}), CAST(rec.location_index AS INT) + 1)'),
            'latitude': Field('rec.latitude', 'DOUBLE'),
            'longitude': Field('rec.longitude', 'DOUBLE'),
            # Requested with timezone=UTC, so the local times in the response are UTC
            'time': Field('rec.time', 'TIMESTAMP', fmt="yyyy-MM-dd'T'HH:mm"),
            **{column: Field(f'rec.{v}', 'DOUBLE') for v, column, *_ in WEATHER_VARIABLES},
        },
        required=('location', 'time'),
        hard_rules={
            # Open-Meteo snaps to its grid (54.8 -> 54.797), so allow half a degree; more means the order changed
            'unexpected_location': (f'location IS NULL OR abs(latitude - {_expected(1)}) > 0.5 '
                                    f'OR abs(longitude - {_expected(2)}) > 0.5'),
            **{f'{column}_implausible': outside(column, low, high) for _, column, low, high in WEATHER_VARIABLES},
        },
        # Early dates can lack a value for some variables; the hour is still useful
        soft_rules={'missing_weather_value': ' OR '.join(f'{c} IS NULL' for _, c, *_ in WEATHER_VARIABLES)},
        keys=('location', 'time'),
        cluster_by=('time',),
    )


WEATHER_DAY_AHEAD = _weather(
    'weather_day_ahead', 'weather_forecast_day_ahead', '_previous_day1',
    ('Open-Meteo weather as forecast about a day earlier, hourly, at 5 UK wind and solar points. Grain: one row '
     'per location and hour (UTC). The leakage-safe weather input for a day-ahead model.'))

WEATHER_HISTORICAL = _weather(
    'weather_historical', 'weather_historical_forecast', '',
    ('Open-Meteo stitched short-range forecasts, hourly, at the same points. Grain: one row per location and '
     'hour (UTC). Close to the actual weather: for exploration, not as model input.'))

# --- All of them, in the order the pipeline defines them ----------------------------------------------------

SOURCES = (FUELHH, DEMAND_OUTTURN, WIND_FORECAST, DEMAND_FORECAST, WIND_SOLAR_FORECAST, WIND_SOLAR_ACTUAL,
           SYSTEM_PRICES, MARKET_INDEX, REMIT_OUTAGES, BANK_HOLIDAYS, CARBON_INTENSITY, PV_LIVE, NESO_DEMAND,
           WEATHER_DAY_AHEAD, WEATHER_HISTORICAL)
