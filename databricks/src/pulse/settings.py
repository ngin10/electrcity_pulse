"""Names and constants shared by every layer. Pipelines read overrides from their configuration."""

# The catalog holding the bronze, silver and gold schemas. A setting, not a constant: Free Edition may not let
# you create a catalog, and then the built-in `workspace` catalog is used. Pipelines read it from their
# configuration (pulse.catalog), the health notebook from a job parameter
CATALOG_DEFAULT = 'pulse'


def schema(catalog, layer):
    """The full name of a layer's schema, e.g. pulse.bronze."""
    return f'{catalog}.{layer}'


# Where bronze reads the landing files from (pipeline configuration pulse.landing):
# - the GCS landing zone itself, written by the ingestion code (consumer/ and batch_ingest/): Databricks
#   reads it through a Unity Catalog external location. Needs Databricks on Google Cloud.
# - a Unity Catalog volume holding a copy of it, made by tools/sync_landing.py: Databricks Free Edition,
#   which can't reach your own bucket.
LANDING_GCS = 'gs://databricks_1/electricity_gen_by_fuel/landing'


def landing_volume(catalog):
    return f'/Volumes/{catalog}/bronze/landing'

UK_TZ = 'Europe/London'

# --- Bronze: one table per landing folder ---------------------------------------------------------------

KAFKA_TABLE = 'fuelhh'
BATCH_TABLES = (
    'elexon_demand_outturn', 'elexon_wind_forecast', 'elexon_demand_forecast', 'elexon_wind_solar_forecast',
    'elexon_wind_solar_actual', 'elexon_system_prices', 'elexon_market_index', 'elexon_remit_outages',
    'uk_bank_holidays', 'carbon_intensity', 'pv_live', 'neso_historic_demand', 'weather_forecast_day_ahead',
    'weather_historical_forecast',
)
BRONZE_TABLES = (KAFKA_TABLE,) + BATCH_TABLES

# The landing files' column types, as written by the repo (consumer/store.py and batch_ingest/__main__.py).
# Auto Loader takes these as schema hints, so inference can never choose different types.
KAFKA_HINTS = ('raw_value STRING, kafka_topic STRING, kafka_partition BIGINT, kafka_offset BIGINT, '
               'kafka_key STRING, kafka_timestamp_ms BIGINT, consumed_at STRING, ingest_date DATE')
BATCH_HINTS = ('raw_value STRING, source STRING, request_url STRING, window_from STRING, window_to STRING, '
               'http_status BIGINT, fetched_at STRING, ingest_date DATE')

BRONZE_DESCRIPTIONS = {
    'fuelhh': 'Elexon FUELHH generation by fuel type, one row per Kafka message, every revision and replay',
    'elexon_demand_outturn': 'Elexon national demand outturn (INDO, ITSDO), one row per API response',
    'elexon_wind_forecast': 'Elexon WINDFOR wind generation forecasts, every publish, one row per API response',
    'elexon_demand_forecast': 'Elexon NDF national demand forecast, 17:00-18:00 UTC publishes, one row per API response',
    'elexon_wind_solar_forecast': 'Elexon DGWS wind and solar generation forecasts, one row per API response',
    'elexon_wind_solar_actual': 'Elexon AGWS actual wind and solar generation, one row per API response',
    'elexon_system_prices': 'Elexon system (imbalance) prices, one row per API response (one settlement date)',
    'elexon_market_index': 'Elexon MID market index price and volume, one row per API response',
    'elexon_remit_outages': 'Elexon REMIT outage notices with details, one row per API response (one day)',
    'uk_bank_holidays': 'GOV.UK bank holidays, one full snapshot per run',
    'carbon_intensity': 'Carbon Intensity API forecast and actual, one row per API response',
    'pv_live': 'Sheffield Solar PV_Live national solar estimate, one row per API response',
    'neso_historic_demand': 'NESO historic demand with embedded wind and solar, one row per API response',
    'weather_forecast_day_ahead': 'Open-Meteo forecasts as made the day before, 5 UK points, one row per API response',
    'weather_historical_forecast': 'Open-Meteo stitched short-range forecasts, 5 UK points, one row per API response',
}

# --- Silver ---------------------------------------------------------------------------------------------

# FUELHH fuel types seen in the data; a new one is kept and flagged (see sources.py), never rejected
KNOWN_FUEL_TYPES = (
    'BIOMASS', 'CCGT', 'COAL', 'NPSHYD', 'NUCLEAR', 'OCGT', 'OIL', 'OTHER', 'PS', 'WIND',
    'INTELEC', 'INTEW', 'INTFR', 'INTGRNL', 'INTIFA2', 'INTIRL', 'INTNED', 'INTNEM', 'INTNSL', 'INTVKL',
)

# The points requested by batch_ingest/sources.py (WEATHER_POINTS), in the same order: Open-Meteo answers
# with a list in request order, so a record's position says which point it is
WEATHER_POINTS = (
    ('dogger_bank', 54.8, 1.9),
    ('east_anglia_offshore', 52.6, 2.2),
    ('irish_sea', 53.8, -3.6),
    ('scottish_highlands', 57.0, -4.0),
    ('southern_england', 51.5, -1.0),
)

# --- Gold -----------------------------------------------------------------------------------------------

# How FUELHH fuel types add up. Interconnectors (INT*) are flows, not GB generation: positive is import
FUEL_CATEGORIES = {
    'renewable': ('WIND', 'NPSHYD', 'BIOMASS'),
    'fossil': ('CCGT', 'OCGT', 'COAL', 'OIL'),
    'nuclear': ('NUCLEAR',),
    'storage': ('PS',),
    'other': ('OTHER',),
}

# A day-ahead prediction for day D is made at this time on D-1 (UTC). Every feature must be known by then:
# Elexon's day-ahead wind and solar forecast is published around 16:45 and the 17:00-18:00 demand forecasts
# are what batch ingest fetches
FORECAST_CUTOFF_HOUR_UTC = 18
