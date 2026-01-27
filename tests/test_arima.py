import numpy as np
import pandas as pd

from scripts.task2_forecasting import fit_arima, forecast_arima


def test_arima_forecast_length():
    series = pd.Series(np.cumsum(np.random.randn(120)))

    model = fit_arima(series)
    preds = forecast_arima(model, steps=10)

    assert len(preds) == 10
