"""Local Spark for testing the pure DataFrame logic in databricks/src/pulse.

Skipped when PySpark or Java isn't installed, so the repo's main test run is unaffected. To run them:
pip install -r databricks/requirements-dev.txt (and have Java 17), then pytest databricks/tests.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip('pyspark')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@pytest.fixture(scope='session')
def spark():
    from pyspark.sql import SparkSession
    try:
        session = (SparkSession.builder.master('local[1]').appName('pulse-tests')
                   # The same settings the pipelines run with: UTC session time and ANSI SQL
                   .config('spark.sql.session.timeZone', 'UTC')
                   .config('spark.sql.ansi.enabled', 'true')
                   .config('spark.sql.shuffle.partitions', '1')
                   .config('spark.ui.enabled', 'false')
                   .getOrCreate())
    except Exception as error:  # no Java: PySpark can't start its JVM
        pytest.skip(f'Spark could not start: {error}')
    session.sparkContext.setLogLevel('ERROR')
    yield session
    session.stop()
