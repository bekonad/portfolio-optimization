import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from scripts.task2_forecasting import forecast_lstm


def test_forecast_lstm_with_mock_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.5]])

    last_window = np.array([0.1] * 10)
    scaler = MagicMock()
    scaler.inverse_transform.return_value = np.array([[100], [101], [102]])

    preds = forecast_lstm(
        model=mock_model,
        last_window=last_window,
        scaler=scaler,
        steps=3
    )

    assert len(preds) == 3
