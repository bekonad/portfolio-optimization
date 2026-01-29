# tests/test_arima.py

import pytest
import pandas as pd
from pmdarima import auto_arima

PROCESSED_DIR = r"../data/processed"

# ------------------------------
# Test 1: ARIMA forecast basic functionality
# ------------------------------
def test_arima_forecast_shape():
    df = pd.read_csv(f'{PROCESSED_DIR}/prices_aligned.csv', index_col=0, parse_dates=True)
    tsla_series = df['TSLA']
    model = auto_arima(tsla_series, seasonal=False, suppress_warnings=True)
    forecast = model.predict(n_periods=5)
    assert len(forecast) == 5, "ARIMA forecast length mismatch"
    assert all([isinstance(x, float) for x in forecast]), "ARIMA forecast contains non-float values"
