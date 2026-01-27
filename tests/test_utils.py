import numpy as np
import pandas as pd

from scripts.task2_forecasting import adf_test, evaluate_forecast


def test_adf_test_returns_stat_and_pvalue():
    series = pd.Series(np.random.randn(100))
    stat, pval = adf_test(series)

    assert isinstance(stat, float)
    assert isinstance(pval, float)
    assert 0 <= pval <= 1


def test_evaluate_forecast_outputs_positive_errors():
    y_true = np.array([10, 12, 11, 13])
    y_pred = np.array([10, 11, 12, 14])

    mae, rmse, mape = evaluate_forecast(y_true, y_pred)

    assert mae >= 0
    assert rmse >= 0
    assert mape >= 0
