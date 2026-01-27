import numpy as np
import pandas as pd

from scripts.task2_forecasting import (
    prepare_lstm_data,
    build_lstm,
    forecast_lstm
)


def test_prepare_lstm_data_shapes():
    series = pd.Series(np.arange(100))
    X, y, scaler = prepare_lstm_data(series, window=10)

    assert X.shape[1] == 10
    assert X.shape[2] == 1
    assert len(X) == len(y)


def test_lstm_future_forecast_length():
    series = pd.Series(np.arange(100))
    X, y, scaler = prepare_lstm_data(series, window=10)

    model = build_lstm((10, 1))
    model.fit(X, y, epochs=1, batch_size=16, verbose=0)

    last_window = scaler.transform(series.values[-10:].reshape(-1, 1)).flatten()
    preds = forecast_lstm(model, last_window, scaler, steps=5)

    assert len(preds) == 5
