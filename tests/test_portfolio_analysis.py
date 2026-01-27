import pandas as pd
import numpy as np

from src.portfolio_analysis import (
    compute_expected_returns,
    compute_volatility,
    sharpe_ratio
)


def test_expected_returns_positive_trend():
    df = pd.DataFrame({"forecast": [100, 105, 110, 115]})
    ret = compute_expected_returns(df)
    assert ret > 0


def test_volatility_non_negative():
    df = pd.DataFrame({"forecast": [100, 102, 101, 103]})
    vol = compute_volatility(df)
    assert vol >= 0


def test_sharpe_ratio_valid():
    sr = sharpe_ratio(0.1, 0.2)
    assert sr > 0
