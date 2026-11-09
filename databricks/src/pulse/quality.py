"""The silver engine: turn bronze rows (JSON text) into typed, checked records. Shared by every source.

A bronze row holds one Kafka message (FUELHH) or one whole API response (the batch sources). `check` turns
it into one row per record in four steps, each a plain DataFrame transformation:

1. Parse the response with an explicit schema in which each record stays as its own JSON text.
2. Explode to one row per record. posexplode_outer keeps a response with no records as one row, so an empty
   or broken response is reported instead of disappearing.
3. Parse each record with every value as STRING, then convert each field with try_cast or try_to_timestamp.
   The text is kept next to the typed value: text NULL means the field is missing, text present but typed
   NULL means it has the wrong type. Parsing straight into INT would turn both into the same NULL.
4. Evaluate the rules into two arrays: reject_reasons (hard rules: the row goes to quarantine) and warnings
   (soft rules: the row is kept, flagged).

Rules are SQL conditions that are TRUE when the row is bad. A condition on a NULL value is NULL, which never
fires, so every required field gets a missing_* rule, and value rules only judge values that are present.
"""
from dataclasses import dataclass, field

from pyspark.sql import DataFrame, functions as F

from pulse.settings import UK_TZ

# Valid JSON at the top level: an object or an array. Both functions return NULL for anything else,
# including invalid text, so together they test validity without a Databricks-only function
IS_JSON = '(json_object_keys(raw_value) IS NOT NULL OR json_array_length(raw_value) IS NOT NULL)'


# --- SQL helpers ------------------------------------------------------------------------------------------

def sql_str(value):
    """A SQL string literal, with backslashes and quotes escaped."""
    return "'" + str(value).replace('\\', '\\\\').replace("'", "\\'") + "'"


def sql_list(values):
    return ', '.join(sql_str(v) for v in values)


def periods_in_day(date_sql):
    """Settlement periods in a UK day: 48, or 46 and 50 on the clock-change days.

    A day is as long as the time from its local midnight to the next one. Needs session time zone UTC.
    """
    start = f"to_utc_timestamp({date_sql}, '{UK_TZ}')"
    end = f"to_utc_timestamp(date_add({date_sql}, 1), '{UK_TZ}')"
    return f'CAST((unix_timestamp({end}) - unix_timestamp({start})) / 1800 AS INT)'


def period_start_utc(date_sql, period_sql):
    """The UTC instant a settlement period starts: local midnight plus 30 minutes per period before it."""
    return f"timestampadd(MINUTE, ({period_sql} - 1) * 30, to_utc_timestamp({date_sql}, '{UK_TZ}'))"


def outside(column, low, high):
    """TRUE when a present value is outside [low, high]. NaN counts as outside (Spark sorts NaN above all)."""
    return f'{column} NOT BETWEEN {low} AND {high}'


def reasons(rules):
    """{name: condition TRUE when bad} -> an array of the names whose condition is TRUE, all of them."""
    if not rules:
        return F.array().cast('ARRAY<STRING>')
    return F.array_compact(F.array(*[F.when(F.expr(cond), F.lit(name)) for name, cond in rules.items()]))


# --- Source definitions -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Field:
    """One silver column: where its text comes from, and the type it must convert to."""
    # SQL giving the value as text, over `rec` (the record), `response` or the bronze columns
    source: str
    type: str = 'STRING'
    # Exact datetime pattern. Without one, ISO text is cast; with one, anything else is a type error
    fmt: str | None = None
    # A full conversion expression, for values with several formats; {v} stands for `source`
    typed_sql: str | None = None

    def typed(self):
        if self.typed_sql:
            return self.typed_sql.format(v=self.source)
        if self.fmt:
            ts = f'try_to_timestamp({self.source}, {sql_str(self.fmt)})'
            return f'CAST({ts} AS DATE)' if self.type == 'DATE' else ts
        if self.type == 'STRING':
            return self.source
        return f'try_cast({self.source} AS {self.type})'

    @property
    def converts(self):
        """Whether the text is converted, so it can have the wrong type."""
        return self.type != 'STRING' or bool(self.fmt or self.typed_sql)


BATCH_LINEAGE = ('request_url', 'source_file', 'ingest_date')
KAFKA_LINEAGE = ('kafka_partition', 'kafka_offset', 'kafka_key', 'source_file', 'ingest_date')


@dataclass(frozen=True)
class SilverSource:
    """Everything that differs between sources; check() and the pipeline do the rest the same way."""
    # Silver table, and the bronze table it reads
    table: str
    bronze_table: str
    # Table comment: say the grain (what one row is) and the key
    comment: str
    # Parses raw_value. Records are typed STRING so each stays as its own JSON text
    response_schema: str
    # SQL over `response` giving ARRAY<STRING>, one JSON object per record
    records: str
    # The keys a record is expected to have; each is parsed as STRING. Others raise unexpected_fields
    record_fields: tuple
    # Silver column -> Field
    fields: dict
    # Fields a usable record must have (missing_* rules)
    required: tuple
    # The grain: one silver row per key
    keys: tuple
    # Which version of a key wins: compared in order, the highest wins; ends with a unique tiebreaker
    sequence: tuple
    hard_rules: dict = field(default_factory=dict)
    soft_rules: dict = field(default_factory=dict)
    # Hard rules about the whole response, e.g. its HTTP status
    response_rules: dict = field(default_factory=dict)
    lineage: tuple = BATCH_LINEAGE
    cluster_by: tuple = ()
    # Also keep every version (SCD type 2) in <table>_history, tracking changes to these columns
    history_columns: tuple = ()
    # Also keep <table>_current: the latest version per these keys, ordered by current_sequence
    current_keys: tuple = ()
    current_sequence: tuple = ()

    @property
    def record_schema(self):
        return ', '.join(f'`{name}` STRING' for name in self.record_fields)

    @property
    def final_columns(self):
        """The silver table's columns: the fields, the warnings, then where each row came from."""
        extra = ['warnings', *self.lineage, 'record_index']
        return [*self.fields, *[c for c in extra if c not in self.fields]]

    def all_hard_rules(self):
        rules = {'invalid_json': 'NOT is_json', **self.response_rules,
                 # A valid response with no records at all: report it, it may be an outage at the source
                 'no_records': 'is_json AND record_json IS NULL',
                 # A record that is a string, number or list instead of an object
                 'record_not_object': 'record_json IS NOT NULL AND record_keys IS NULL'}
        # Key columns are always required: a NULL key would trip the pipeline's key contract and stop it
        for name in dict.fromkeys((*self.required, *(k for k in self.keys if k in self.fields))):
            rules[f'missing_{name}'] = f'record_keys IS NOT NULL AND {name}__text IS NULL'
        for name, f in self.fields.items():
            if f.converts:
                rules[f'bad_type_{name}'] = f'{name}__text IS NOT NULL AND {name} IS NULL'
            if f.type == 'DOUBLE':
                # NaN and infinity are valid doubles, so the type check passes them; they break every sum
                rules[f'not_finite_{name}'] = f"isnan({name}) OR abs({name}) = double('Infinity')"
        rules.update(self.hard_rules)
        return rules

    def all_soft_rules(self):
        # A key the contract doesn't know: from_json would silently drop it, so say so
        return {'unexpected_fields': f'size(array_except(record_keys, array({sql_list(self.record_fields)}))) > 0',
                **self.soft_rules}


# --- The four steps -----------------------------------------------------------------------------------------

def check(source, bronze):
    """Bronze rows -> one row per record: parsed, typed, with reject_reasons and warnings."""
    df = (bronze
          .withColumn('is_json', F.expr(IS_JSON))
          # 1. Parse with the explicit schema; fields not in it are ignored (and reported below)
          .withColumn('response', F.from_json('raw_value', source.response_schema))
          # 2. One row per record, numbered from 0; a response without records still gives one row
          .select('*', F.posexplode_outer(F.expr(source.records)).alias('record_index', 'record_json'))
          # json_object_keys is NULL unless the record is a JSON object, so it doubles as that check
          .withColumn('record_keys', F.expr('json_object_keys(record_json)'))
          .withColumn('rec', F.from_json('record_json', source.record_schema)))
    # 3. The text of every field, then its typed value. withColumns evaluates each batch against the columns
    # before it, so a field named like a bronze column (fetched_at) replaces that column with its typed value
    df = df.withColumns({f'{name}__text': F.expr(f.source) for name, f in source.fields.items()})
    df = df.withColumns({name: F.expr(f.typed()) for name, f in source.fields.items()})
    # 4. Every rule that fails, by name
    return df.select('*',
                     reasons(source.all_hard_rules()).alias('reject_reasons'),
                     reasons(source.all_soft_rules()).alias('warnings'),
                     F.current_timestamp().alias('checked_at'))


def valid_rows(source, checked):
    """Rows that passed every hard rule, in the silver table's shape."""
    return checked.where('size(reject_reasons) = 0').select(*source.final_columns)


def daily_completeness(silver, table):
    """Per settlement date: periods present vs the periods the day has. Row rules can't see a missing row."""
    return (silver.groupBy('settlement_date')
            .agg(F.countDistinct('settlement_period').alias('periods_present'), F.count('*').alias('rows_present'))
            .withColumn('periods_expected', F.expr(periods_in_day('settlement_date')))
            .withColumn('missing_periods', F.col('periods_expected') - F.col('periods_present'))
            .withColumn('is_complete', F.col('missing_periods') == 0)
            .withColumn('source_table', F.lit(table)))


def rejected_rows(source, checked):
    """Rows that failed a hard rule, in the shared shape of silver.rejected: nothing is dropped silently."""
    def column(name, sql_type):
        return F.col(name) if name in checked.columns else F.lit(None).cast(sql_type)

    return (checked.where('size(reject_reasons) > 0')
            .select(F.lit(source.table).alias('source_table'),
                    'reject_reasons', 'warnings',
                    # The record exactly as received; the whole response only when no record could be read
                    'record_json',
                    F.when(F.col('record_json').isNull(), F.col('raw_value')).alias('response_json'),
                    column('request_url', 'STRING').alias('request_url'),
                    column('kafka_partition', 'BIGINT').alias('kafka_partition'),
                    column('kafka_offset', 'BIGINT').alias('kafka_offset'),
                    'source_file', 'ingest_date',
                    F.current_timestamp().alias('rejected_at')))
