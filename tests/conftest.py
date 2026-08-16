import pytest
import pandas as pd


@pytest.fixture
def calibration_data():
    """Provide calibration fixture data."""
    return pd.read_parquet("calibration/fixtures/wdi.parquet")
